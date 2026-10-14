"""Recording and loading training data.

Recordings are written straight to .mmap files (N x 128 x 20 x 20), so the
separate pickle -> mmap conversion step is no longer needed. The frame count is
derived from the file size, so recordings don't have to be exactly 2048 frames.

Two layouts are read:
  * float32 without a header - what the original script and `convertmmap`
    produced, and what this tool wrote before float16 recordings existed,
  * float16 after a 64-byte header starting with F16_MAGIC - half the disk
    space (~200 MB instead of ~400 MB per 2048 frames). New recordings use it
    unless config "record_float16" is off. Frames are converted back to
    float32 when a training batch is built.
"""

import json
import logging
import os
import pickle
import threading
import time

import numpy as np

from .frames import SAMPLE_SHAPE, sample_array

log = logging.getLogger(__name__)

SAMPLE_FLOATS = int(np.prod(SAMPLE_SHAPE))
SAMPLE_BYTES = SAMPLE_FLOATS * 4  # a float32 sample (legacy layout)

F16_MAGIC = b"PBG-F16\n"
HEADER_SIZE = 64  # keeps the frames 64-byte aligned
F16_MAX = float(np.finfo(np.float16).max)


def recording_layout(path):
    """(dtype, data offset) of a recording file."""
    with open(path, "rb") as f:
        head = f.read(len(F16_MAGIC))
    if head == F16_MAGIC:
        return np.dtype(np.float16), HEADER_SIZE
    return np.dtype(np.float32), 0


def _frames_in(size, dtype, offset):
    return max(0, size - offset) // (SAMPLE_FLOATS * dtype.itemsize)


def frame_count(path):
    dtype, offset = recording_layout(path)
    return _frames_in(os.path.getsize(path), dtype, offset)


def open_recording(path, in_ram=False):
    """The frames of a recording as an (N, 128, 20, 20) array; float16 recordings
    stay float16 (also in RAM, so caching them takes half the memory)."""
    dtype, offset = recording_layout(path)
    n = _frames_in(os.path.getsize(path), dtype, offset)
    if n == 0:
        raise ValueError("%s is empty or not a recording" % path)
    mm = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(n,) + SAMPLE_SHAPE)
    return np.array(mm) if in_ram else mm


def create_recording(path, frames, float16=True):
    """A writable (frames, 128, 20, 20) memmap; float16 files get the header."""
    if not float16:
        return np.memmap(path, dtype=np.float32, mode="w+", shape=(frames,) + SAMPLE_SHAPE)
    with open(path, "wb") as f:
        f.write(F16_MAGIC.ljust(HEADER_SIZE, b"\0"))
    return np.memmap(path, dtype=np.float16, mode="r+", offset=HEADER_SIZE, shape=(frames,) + SAMPLE_SHAPE)


def store_sample(mm, index, sample):
    """mm[index] = sample, clipped to the float16 range so it can't become inf."""
    if mm.dtype == np.float16:
        sample = np.clip(sample, -F16_MAX, F16_MAX)
    mm[index] = sample


# ---------------------------------------------------------------- guided recordings
# A guided recording shows (and beeps) a target intensity over time while the user
# follows it. Every frame then has a known intensity, so training can learn *how
# strong* an expression is instead of only "on" - the way blendshape trackers such as
# Project Babble are trained on continuous labels. The neutral stretches between the
# ramps also give each recording neutral frames from the same session.
# (duration s, start intensity, end intensity, cue)
GUIDE_CYCLE = (
    (1.5, 0.0, 0.0, "neutral"),
    (2.0, 0.0, 1.0, "up"),
    (1.5, 1.0, 1.0, "hold"),
    (2.0, 1.0, 0.0, "down"),
    (1.0, 0.0, 0.0, "neutral"),
    (1.0, 0.0, 0.5, "up_half"),
    (1.5, 0.5, 0.5, "hold_half"),
    (1.0, 0.5, 0.0, "down"),
)
GUIDE_CYCLES = 3
GUIDE_SCRIPT = GUIDE_CYCLE * GUIDE_CYCLES + ((1.0, 0.0, 0.0, "neutral"),)
GUIDE_DURATION = sum(seg[0] for seg in GUIDE_SCRIPT)
# people follow a moving target about this much later; labels are shifted by it
GUIDE_LAG = 0.3


def guide_target(t):
    """(intensity 0..1, cue, seconds left in this segment) at `t` seconds into the script."""
    if t < 0:
        return 0.0, "neutral", -t
    for dur, a, b, cue in GUIDE_SCRIPT:
        if t < dur:
            return a + (b - a) * (t / dur), cue, dur - t
        t -= dur
    return 0.0, "done", 0.0


def labels_path(path):
    return path + ".labels.json"


def load_labels(path, frames=None):
    """Per-frame intensity (float32 array) of a guided recording, or None for a plain one."""
    lp = labels_path(path)
    if not os.path.exists(lp):
        return None
    with open(lp, "r", encoding="utf-8") as f:
        data = json.load(f)
    values = np.asarray(data.get("intensity", []), dtype=np.float32)
    if frames is not None and len(values) != frames:
        log.warning("Ignoring %s: %d labels for %d frames", lp, len(values), frames)
        return None
    return np.clip(values, 0.0, 1.0)


def remove_recording(path):
    os.remove(path)
    if os.path.exists(labels_path(path)):
        os.remove(labels_path(path))


class Recorder:
    """Captures every sample pushed to the FrameHub into a .mmap file."""

    def __init__(self, hub, path, frames, countdown=5.0, on_progress=None, on_done=None, guided=False,
                 on_guide=None, float16=True):
        self.hub = hub
        self.path = path
        self.guided = guided
        self.float16 = float16
        # guided: runs for the script's length; room for up to 200 samples/s, trimmed at the end
        self.frames = int(GUIDE_DURATION * 200) + 64 if guided else frames
        self.on_guide = on_guide or (lambda target, cue, left: None)
        self._times = []
        self._t0 = None
        self.countdown = countdown
        self.on_progress = on_progress or (lambda done, total, phase: None)
        self.on_done = on_done or (lambda ok, message: None)
        self._count = 0
        self._mm = None
        self._finished = threading.Event()
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="recorder").start()
        return self

    def cancel(self):
        self._cancel.set()
        self._finished.set()

    def _on_sample(self, eye, face):
        with self._lock:
            if self._mm is None or self._count >= self.frames:
                return
            store_sample(self._mm, self._count, sample_array(eye, face))
            self._count += 1
            if self.guided:
                self._times.append(time.monotonic() - self._t0)
            if self._count >= self.frames:
                self._finished.set()

    def _run(self):
        tmp = self.path + ".part"
        try:
            end = time.monotonic() + self.countdown
            while not self._cancel.is_set() and time.monotonic() < end:
                self.on_progress(0, self.frames, "countdown %.0f" % max(0.0, end - time.monotonic()))
                self._cancel.wait(0.25)
            if self._cancel.is_set():
                self.on_done(False, "Recording cancelled")
                return
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            self._mm = create_recording(tmp, self.frames, self.float16)
            self._t0 = time.monotonic()
            self.hub.add_listener(self._on_sample)
            last_seen, last_change = -1, time.monotonic()
            while not self._finished.wait(0.05 if self.guided else 0.1):
                if self.guided:
                    target, cue, left = guide_target(time.monotonic() - self._t0)
                    self.on_guide(target, cue, left)
                    if cue == "done":
                        break
                if self._count != last_seen:
                    last_seen, last_change = self._count, time.monotonic()
                elif time.monotonic() - last_change > 10:
                    log.warning("No frames for 10s while recording - is SRanipal running?")
                    last_change = time.monotonic()
                if self.guided:
                    elapsed = time.monotonic() - self._t0
                    self.on_progress(int(elapsed * 10), int(GUIDE_DURATION * 10), "guided")
                else:
                    self.on_progress(self._count, self.frames, "recording")
            self.hub.remove_listener(self._on_sample)
            with self._lock:
                count = self._count
                offset, sample_bytes = self._mm.offset, self._mm[0].nbytes
                self._mm.flush()
                self._mm = None
            if self.guided and not self._cancel.is_set() and count >= 64:
                os.truncate(tmp, offset + count * sample_bytes)  # drop the unused preallocated room
                intensity = [round(guide_target(t - GUIDE_LAG)[0], 4) for t in self._times[:count]]
                with open(labels_path(self.path) + ".tmp", "w", encoding="utf-8") as f:
                    json.dump({"version": 1, "kind": "guided", "lag": GUIDE_LAG, "intensity": intensity}, f)
                os.replace(labels_path(self.path) + ".tmp", labels_path(self.path))
            elif self._cancel.is_set() or count < self.frames:
                os.remove(tmp)
                self.on_done(False, "Recording cancelled")
                return
            os.replace(tmp, self.path)
            self.on_progress(count, count, "done")
            self.on_done(True, "Saved %d frames to %s%s" % (count, self.path, " (guided)" if self.guided else ""))
        except Exception as e:
            log.exception("recording failed")
            self.hub.remove_listener(self._on_sample)
            self.on_done(False, "Recording failed: %s" % e)


def convert_legacy_pickles(folder, on_progress=None, float16=True):
    """Convert old *-em.pkl recordings to .mmap (the old `convertmmap` command)."""
    converted = []
    files = [f for f in sorted(os.listdir(folder)) if f.endswith(".pkl")]
    for i, name in enumerate(files):
        src = os.path.join(folder, name)
        dst = src[:-3] + "mmap"
        if os.path.exists(dst):
            continue
        with open(src, "rb") as f:
            block = pickle.load(f)
        mm = create_recording(dst + ".part", len(block), float16)
        for j, (eye, face) in enumerate(block):
            store_sample(mm, j, np.concatenate((eye, face)))
        mm.flush()
        del mm
        os.replace(dst + ".part", dst)
        converted.append(dst)
        if on_progress:
            on_progress(i + 1, len(files))
    return converted

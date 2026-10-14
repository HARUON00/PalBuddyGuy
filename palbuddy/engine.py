"""Glue between frame input, the network and VRCFT output. Used by GUI and CLI.

PyTorch is imported lazily: tracking with ONNX Runtime never loads it (a few
hundred MB less RAM), and the app can even run tracking on a PC where only
onnxruntime is installed, using exported .onnx files.
"""

import importlib.util
import collections
import logging
import math
import os
import tempfile
import threading
import time

from . import onnx_runtime
from .config import Config, binary_bits, class_output_problems, merged_param_problems, osc_output_names
from .datasets import Recorder, convert_legacy_pickles
from .frames import FrameHub, ProxyClient, RateMeter, SRanipalReceiver
from .osc import OscSender
from .vrcft import VRCFTServer

log = logging.getLogger(__name__)


PLAN_REFRESH = 0.5  # s: how often the loop re-reads output settings (changes apply within this)
IDLE_INTERVAL = 0.1  # s: inference rate cap while nothing receives the output (VRCFT off, no OSC outputs)


def normalize(raw, max_power):
    """Map a raw network output to VRCFT's -1..1 range (same formula as before).
    Plain Python: np.clip on a scalar costs ~20 us, and this runs per class per frame."""
    v = float(raw) / max_power * 2.0 - 1.0
    return -1.0 if v < -1.0 else (1.0 if v > 1.0 else v)


SOFT_EDGE = 0.01  # softness keeps outputs of 1% / 99% at 0 / 1


def softener(temperature):
    """Output curve for sigmoid (BCE) models: divides the logit by `temperature`, so the
    output rises gradually with the expression instead of snapping between 0 and 1,
    then rescales so SOFT_EDGE still maps to 0 and 1 - SOFT_EDGE to 1 (neutral stays at 0).
    Returns None for temperature <= 1 (off)."""
    if temperature <= 1.0:
        return None
    inv = 1.0 / temperature
    edge = math.log(SOFT_EDGE / (1.0 - SOFT_EDGE)) * inv
    lo = 1.0 / (1.0 + math.exp(-edge))
    span = 1.0 - 2.0 * lo

    def soften(p):
        if p <= SOFT_EDGE:
            return 0.0
        if p >= 1.0 - SOFT_EDGE:
            return 1.0
        v = (1.0 / (1.0 + math.exp(-math.log(p / (1.0 - p)) * inv)) - lo) / span
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
    return soften


class OneEuro:
    """1-Euro filter (Casiez et al.), the smoothing Project Babble and VRCFaceTracking use:
    a low-pass whose cutoff rises with the speed of change, so small jitter is smoothed
    strongly while real, fast movements pass with little lag."""

    __slots__ = ("x", "dx", "last")

    D_CUTOFF = 1.0
    BETA = 0.9  # Babble's default speed coefficient

    def __init__(self):
        self.x = None
        self.dx = 0.0
        self.last = None

    @staticmethod
    def _alpha(cutoff, dt):
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def update(self, v, now, min_cutoff):
        if self.x is None:
            self.x, self.last = v, now
            return v
        dt = max(1e-4, now - self.last)
        self.last = now
        dx = (v - self.x) / dt
        self.dx += (dx - self.dx) * self._alpha(self.D_CUTOFF, dt)
        cutoff = min_cutoff + self.BETA * abs(self.dx)
        self.x += (v - self.x) * self._alpha(cutoff, dt)
        return self.x


def smoothing_cutoff(smoothing):
    """Smoothing slider (0..0.9) -> 1-Euro min cutoff in Hz: 0.5 ~ 2.8 Hz (Babble uses 3),
    0.9 ~ 0.4 Hz. None = off."""
    if smoothing <= 0.0:
        return None
    return 0.3 + 10.0 * (1.0 - smoothing) ** 2


STABILITY_WINDOW = 0.4  # s: median window at stability 1 (a syllable is ~0.1-0.25 s)
STABILITY_TAU = 0.3  # s: low-pass time constant at stability 1


class Stabilizer:
    """Per-class filter for expressions that should hold instead of flickering with
    speech: a running median over the last stability * STABILITY_WINDOW seconds
    removes short spikes, then a time-based low-pass evens out what's left. Time-based,
    so it behaves the same at 30, 60 or 120 fps."""

    __slots__ = ("samples", "value", "last")

    def __init__(self):
        self.samples = collections.deque()
        self.value = None
        self.last = None

    def update(self, w, now, stability):
        window = stability * STABILITY_WINDOW
        samples = self.samples
        samples.append((now, w))
        while samples[0][0] < now - window:
            samples.popleft()
        ordered = sorted(v for _, v in samples)
        n = len(ordered)
        median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) * 0.5
        if self.value is None:
            self.value = median
        else:
            dt = now - self.last
            a = dt / (dt + stability * STABILITY_TAU)
            self.value += (median - self.value) * a
        self.last = now
        return self.value


class Engine:
    def __init__(self, cfg: Config, config_path=None):
        self.cfg = cfg
        self.config_path = config_path
        self._device = None  # training device, resolved on first use (imports torch)
        self.hub = FrameHub()
        self.source = None
        self.vrcft = None
        self.model = None
        self.model_dirty = False  # trained but not saved

        self.busy = None  # "recording" / "training" / None
        self._recorder = None
        self._train_stop = threading.Event()

        self._infer_thread = None
        self._infer_stop = threading.Event()
        self._fastcal_request = threading.Event()
        self.fastcal_state = None  # text shown while fastcal runs
        self.infer_rate = RateMeter()
        self.infer_latency_ms = 0.0
        self.infer_backend = None  # e.g. "CPU int8 ×1"
        self._cpu_sample = (time.monotonic(), time.process_time())
        self.process_cpu_pct = 0.0
        self.last_raw = None  # np.ndarray of raw outputs
        self.last_out = {}  # class index -> normalized value sent to VRCFT
        self.last_merged = {}  # merged parameter name -> value sent over OSC
        self.last_class_raw = {}  # class index -> weight before the sensitivity range (0..1)
        self.last_weights = {}  # class index -> final weight (0..1)
        self.osc = None
        self._osc_plan_key = None
        self._osc_plan = []
        self._bool_state = {}  # OSC bool name -> last value sent
        self._bool_refresh = 0.0
        self._flag_names = {}  # (name, bits, signed) -> binary bool names
        self.idle = False  # inference throttled because nothing receives the output
        self._full_rate_until = 0.0  # e.g. while the GUI measures sensitivity
        self._smoothed = {}
        self._stabilizers = {}
        self.compare_results = None  # last model comparison (see compare_async)

    # ------------------------------------------------------------ lifecycle
    def start(self):
        cfg = self.cfg
        if cfg.source == "proxy":
            self.source = ProxyClient(self.hub, cfg.bind_host, cfg.proxy_port).start()
        else:
            self.source = SRanipalReceiver(self.hub, cfg.bind_host, cfg.face_port, cfg.eye_port,
                                           swapped=cfg.swapped, mode=cfg.input_mode,
                                           stall_timeout=cfg.stall_timeout).start()
        self.vrcft = VRCFTServer(cfg.bind_host, cfg.vrcft_port)
        self.vrcft.max_mode = cfg.vrcft_override_mode == "max"
        self.vrcft.start()
        self.osc = OscSender(cfg.osc_host, cfg.osc_port)
        return self

    def set_osc_target(self, host, port):
        self.cfg.osc_host, self.cfg.osc_port = host, int(port)
        old, self.osc = self.osc, OscSender(host, int(port))
        if old is not None:
            old.close()

    def active_merged_params(self):
        """Enabled merged parameters whose settings are valid."""
        return [obj for kind, obj in self.osc_plan() if kind == "merged"]

    def osc_plan(self):
        """[(kind, obj)] of the OSC outputs to send: classes with an OSC name ("class") and
        enabled merged parameters ("merged") whose settings are valid (no clash with
        VRCFaceTracking's names, no parameter written twice). Cached until settings change."""
        cfg = self.cfg
        key = (tuple((c.name, c.osc_name, c.osc_format, c.osc_bits) for c in cfg.classes),
               tuple((m.name, tuple((t.cls, t.weight) for t in m.terms), m.out_min, m.out_max, m.out_neutral,
                      m.enabled, m.osc_format, m.osc_bits) for m in cfg.merged_params))
        if key != self._osc_plan_key:
            plan, used = [], set()
            names = {c.name for c in cfg.classes}
            candidates = [("class", c) for c in cfg.classes if c.osc_name and not class_output_problems(c)]
            candidates += [("merged", m) for m in cfg.merged_params
                           if m.enabled and not merged_param_problems(m, names)]
            for kind, obj in candidates:
                out_name, signed = (obj.osc_name, False) if kind == "class" else (obj.name, obj.signed)
                written = {n.lower() for n in osc_output_names(out_name, obj.osc_format, obj.osc_bits, signed)}
                if written & used:
                    log.warning("OSC output '%s' skipped: another output writes the same parameter", out_name)
                    continue
                used |= written
                plan.append((kind, obj))
            self._osc_plan_key, self._osc_plan = key, plan
            self._bool_state = {}
        return self._osc_plan

    def _send_output(self, name, fmt, bits, value, binary_value, negative, signed, force=False):
        """Send one OSC output as float and/or binary bools (bools only when they change)."""
        if fmt in ("float", "both"):
            self.osc.send_parameter(name, float(value))
        if fmt in ("binary", "both"):
            names = self._flag_names.get((name, bits, signed))
            if names is None:
                names = self._flag_names[(name, bits, signed)] = osc_output_names(name, "binary", bits, signed)
            values = binary_bits(binary_value, bits)
            if signed:
                values.append(bool(negative))
            state = self._bool_state
            for flag, b in zip(names, values):
                if force or state.get(flag) != b:
                    self.osc.send_parameter(flag, b)
                    state[flag] = b

    def _send_osc(self, weights, merged, force=False):
        for kind, obj in self._osc_plan:
            if kind == "class":
                w = weights.get(obj.name, 0.0)
                self._send_output(obj.osc_name, obj.osc_format, obj.osc_bits, w, w, False, False, force)
            elif obj.name in merged:
                value = merged[obj.name]
                mag, neg = obj.binary_input_of(value) if obj.osc_format != "float" else (0.0, False)
                self._send_output(obj.name, obj.osc_format, obj.osc_bits, value, mag, neg, obj.signed, force)

    def request_full_rate(self, seconds):
        """Suspend the idle rate for a while (measurements want every frame)."""
        self._full_rate_until = time.monotonic() + seconds

    def _send_merged_neutral(self):
        """Tracking stopped: leave every OSC output at its neutral value, not frozen."""
        if self.osc is not None and self.cfg.osc_enabled:
            self.osc_plan()
            neutral = {m.name: m.neutral for m in self.active_merged_params()}
            self._send_osc({}, neutral, force=True)
        self.last_merged = {}
        self._bool_state = {}

    @property
    def device(self):
        if self._device is None:
            from .model import pick_device
            self._device = pick_device()
            log.info("Training device: %s", self._device)
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    def stop(self):
        self.stop_inference()
        self._train_stop.set()
        if self._recorder:
            self._recorder.cancel()
        for part in (self.source, self.vrcft):
            if part is not None:
                part.stop()
        if self.osc is not None:
            self.osc.close()

    def save_config(self):
        if self.config_path:
            self.cfg.save(self.config_path)

    # ------------------------------------------------------------ status
    def mode_hint(self, streams):
        """Suggest an input mode when the connected trackers don't match the selected one.

        Returns "both", "single" (only one tracker streams; which one can't be told
        from the port, so the user picks face/eye using the camera preview) or None."""
        mode = self.cfg.input_mode
        ok = {role: streams.get(role, {}).get("state") == "ok" for role in ("eye", "face")}
        if not any(ok.values()) or self.cfg.source == "proxy":
            return None
        if mode == "both" and not all(ok.values()):
            return "single"
        if mode != "both" and all(ok.values()):
            return "both"
        return None

    def _update_cpu(self):
        now, cpu = time.monotonic(), time.process_time()
        t0, c0 = self._cpu_sample
        if now - t0 >= 1.0:
            # % of one core used by the whole process (all threads)
            self.process_cpu_pct = (cpu - c0) / (now - t0) * 100
            self._cpu_sample = (now, cpu)
        return self.process_cpu_pct

    def status(self):
        streams = self.source.snapshot(self.cfg.stall_timeout) if self.source else {}
        return {
            "infer_backend": self.infer_backend,
            "process_cpu": self._update_cpu(),
            "streams": streams,
            "input_mode": self.cfg.input_mode,
            "mode_hint": self.mode_hint(streams),
            "sample_fps": self.hub.sample_rate.rate(),
            "vrcft": self.vrcft.snapshot() if self.vrcft else {"connected": False},
            "inferring": self.inferring,
            "infer_fps": self.infer_rate.rate(),
            "latency_ms": self.infer_latency_ms,
            "busy": self.busy,
            "fastcal": self.fastcal_state,
            "device": str(self._device) if self._device is not None else None,
            "idle": self.idle and self.inferring and time.monotonic() >= self._full_rate_until,
            "model_loaded": self.model is not None or self.inferring,
            "model_dirty": self.model_dirty,
            "swapped": self.cfg.swapped,
        }

    def set_input_mode(self, mode):
        if mode == self.cfg.input_mode:
            return
        if self.busy:
            raise RuntimeError("Wait until %s has finished" % self.busy)
        self.stop_inference()
        self.cfg.input_mode = mode
        if self.source:
            self.source.set_mode(mode)
        if self.model is not None and self.model.input_mode != mode:
            log.info("Unloaded the model: it was trained for input '%s'. Load or train a '%s' model.",
                     self.model.input_mode, mode)
            self.model = None
            self.model_dirty = False
        self.save_config()
        log.info("Input mode: %s", mode)

    def set_swapped(self, value):
        self.cfg.swapped = bool(value)
        if self.source:
            self.source.set_swapped(value)
        self.save_config()

    # ------------------------------------------------------------ model
    def model_path(self):
        return self.cfg.model_path

    def model_base(self, path=None):
        return os.path.splitext(path or self.cfg.model_path)[0]

    def load_model(self, path=None):
        from .model import BuddyNet
        path = path or self.cfg.model_path
        if path.endswith(".onnx"):
            raise ValueError("Select the .pt file; the matching .onnx file is used automatically for tracking")
        # loaded on the CPU; the inference runtime / trainer move it where needed, so
        # CPU-only tracking never initialises CUDA (saves its VRAM context)
        self.model = BuddyNet.load(path, None, expected_outputs=self.cfg.num_classes,
                                   expected_mode=self.cfg.input_mode).eval()
        self.model_dirty = False
        if path != self.cfg.model_path:
            self.cfg.model_path = path
            self.save_config()
        log.info("Loaded model %s", path)
        return self.model

    def save_model(self, path=None):
        if self.model is None:
            raise RuntimeError("No model to save - train or load one first")
        path = path or self.cfg.model_path
        self.model.save(path, [c.name for c in self.cfg.classes])
        self.model_dirty = False
        log.info("Saved model to %s", path)
        if onnx_runtime.available():
            try:
                self._export_onnx(self.model_base(path))
            except Exception as e:
                log.warning("ONNX export failed (%s); tracking will use PyTorch", e)

    def _export_onnx(self, base):
        from .onnx_export import export_onnx
        return export_onnx(self.model, base, [c.name for c in self.cfg.classes])

    def _onnx_up_to_date(self, base):
        paths = onnx_runtime.onnx_paths(base)
        if not (os.path.exists(paths["fp32"]) and os.path.exists(paths["meta"])):
            return False
        pt = base + ".pt"
        if os.path.exists(self.cfg.model_path) and self.model_base() == base:
            pt = self.cfg.model_path
        return not os.path.exists(pt) or os.path.getmtime(paths["meta"]) >= os.path.getmtime(pt)

    def onnx_model_base(self):
        """Base path of an ONNX export of the current model, exporting when needed."""
        if self.model is not None and self.model_dirty:
            # trained but not saved: export to a temp file, not next to the .pt
            base = os.path.join(tempfile.gettempdir(), "palbuddy-unsaved-%d" % os.getpid())
            self._export_onnx(base)
            return base
        base = self.model_base()
        if self._onnx_up_to_date(base):
            return base
        if self.model is None:
            if not os.path.exists(self.cfg.model_path):
                raise FileNotFoundError("No trained model found (%s). Train and save one first." % self.cfg.model_path)
            if importlib.util.find_spec("torch") is None:
                raise RuntimeError("The .onnx export of %s is missing or older than the model, and PyTorch "
                                   "isn't installed to create it." % self.cfg.model_path)
            self.load_model()
        self._export_onnx(base)
        return base

    def use_onnx(self):
        choice = self.cfg.infer_engine
        if choice == "pytorch":
            return False
        if choice == "onnx" and not onnx_runtime.available():
            raise RuntimeError("ONNX Runtime is not installed (pip install onnxruntime, or onnxruntime-directml "
                               "on Windows)")
        return onnx_runtime.available()

    def ensure_model(self):
        if self.model is None:
            self.load_model()
        return self.model

    # ------------------------------------------------------------ recording
    def record(self, name, on_progress=None, on_done=None, frames=None, guided=False, on_guide=None):
        """guided: follow the on-screen / beeped intensity script (datasets.GUIDE_SCRIPT);
        the recording gets per-frame intensity labels."""
        if self.busy:
            raise RuntimeError("Already %s" % self.busy)
        name = name.strip()
        if not name:
            raise ValueError("Enter a dataset name")
        filename = unique_recording_name(self.cfg, name)
        if filename != (name if name.endswith(".mmap") else name + "-em.mmap"):
            log.info("A recording named '%s' already exists; saving as %s", name, filename)
        path = self.cfg.dataset_path(filename)
        self.busy = "recording"

        def done(ok, message):
            self.busy = None
            self._recorder = None
            log.info(message)
            if on_done:
                on_done(ok, message, filename)

        self._recorder = Recorder(self.hub, path, frames or self.cfg.record_frames, self.cfg.record_countdown,
                                  on_progress, done, guided=guided, on_guide=on_guide,
                                  float16=self.cfg.record_float16).start()
        return filename

    def cancel_record(self):
        if self._recorder:
            self._recorder.cancel()

    def convert_pickles(self):
        return convert_legacy_pickles(self.cfg.dataset_folder, float16=self.cfg.record_float16)

    # ------------------------------------------------------------ training
    def train_async(self, on_progress=None, on_done=None, resume=False):
        from . import trainer  # imports torch
        if self.busy:
            raise RuntimeError("Already %s" % self.busy)
        problems = self.cfg.validate()
        if problems:
            raise ValueError("\n".join(problems))
        self.stop_inference()
        self.busy = "training"
        self._train_stop.clear()

        def run():
            ok, message, history = False, "", []
            try:
                init = None
                if resume:
                    init = self.model if self.model is not None else self.load_model()
                    if init.input_mode != self.cfg.input_mode:
                        raise ValueError("The current model was trained for input '%s'" % init.input_mode)
                    if init.arch != self.cfg.model_arch:
                        raise ValueError("The current model is a '%s' model; untick 'continue' to train a '%s' one"
                                         % (init.arch, self.cfg.model_arch))
                    if init.output != trainer.OUTPUT_FOR_LOSS[self.cfg.loss]:
                        raise ValueError("The current model was trained with the other loss; untick 'continue' "
                                         "to train a new one")
                model, history = trainer.train(self.cfg, on_progress=on_progress, stop_event=self._train_stop,
                                               log_fn=log.info, init_model=init, device=self.device)
                self.model = model
                self.model_dirty = True
                ok, message = True, "Training finished (final loss %.6f). Don't forget to save." % history[-1]
                metrics = getattr(model, "val_metrics", None)
                if metrics:
                    message = "Training finished (loss %.6f, validation accuracy %.1f%%). Don't forget to save." % (
                        history[-1], metrics["val_acc"] * 100)
            except trainer.TrainingCancelled:
                message = "Training cancelled"
            except Exception as e:
                log.exception("training failed")
                message = "Training failed: %s" % e
            finally:
                self.busy = None
                if self.device.type == "cuda":
                    import torch
                    torch.cuda.empty_cache()
            log.info(message)
            if on_done:
                on_done(ok, message, history)

        threading.Thread(target=run, daemon=True, name="trainer").start()

    def cancel_training(self):
        self._train_stop.set()

    def compare_async(self, candidates=None, on_progress=None, on_done=None):
        """Train several (arch, loss) candidates on the same frames, measure how well
        each does on the held-out frames and how fast it runs here, and recommend one.
        on_done(ok, message, results, best_index). Nothing is applied until
        apply_candidate()."""
        from . import trainer  # imports torch
        if self.busy:
            raise RuntimeError("Already %s" % self.busy)
        problems = self.cfg.validate()
        if problems:
            raise ValueError("\n".join(problems))
        candidates = tuple(candidates or trainer.DEFAULT_CANDIDATES)
        if not candidates:
            raise ValueError("Pick at least one model to compare")
        self.stop_inference()
        self.busy = "comparing"
        self._train_stop.clear()

        def run():
            ok, message, results, best = False, "", [], None
            try:
                results = trainer.compare(self.cfg, candidates, on_progress=on_progress,
                                          stop_event=self._train_stop, log_fn=log.info, device=self.device)
                for r in results:
                    r.update(self._measure(r["model"]))
                best = trainer.recommend(results)
                self.compare_results = results
                ok = True
                r = results[best]
                message = "Compared %d models. Recommended: %s + %s (score %.1f, %.2f ms)" % (
                    len(results), r["arch"], r["loss"].upper(), r["metrics"]["val_score"], r["ms"])
            except trainer.TrainingCancelled:
                message = "Comparison cancelled"
            except Exception as e:
                log.exception("comparison failed")
                message = "Comparison failed: %s" % e
            finally:
                self.busy = None
                if self.device.type == "cuda":
                    import torch
                    torch.cuda.empty_cache()
            log.info(message)
            if on_done:
                on_done(ok, message, results, best)

        threading.Thread(target=run, daemon=True, name="compare").start()

    def _measure(self, model, frames=150):
        """Tracking cost of a model on this PC with the configured CPU settings:
        {"ms": per frame, "backend", "params", "size_mb": file size used for tracking}."""
        import numpy as np
        params = sum(p.numel() for p in model.parameters())
        channels = 128 if model.input_mode == "both" else 64
        x = np.random.default_rng(0).random((1, channels, 20, 20), dtype=np.float32)
        if onnx_runtime.available():
            from .onnx_export import export_onnx
            base = os.path.join(tempfile.gettempdir(), "palbuddy-compare-%d" % os.getpid())
            paths = export_onnx(model, base)
            rt = onnx_runtime.OnnxRuntime(base, "cpu", self.cfg.infer_threads, self.cfg.infer_int8)
            size = os.path.getsize(paths["int8"] if rt.quantized else paths["fp32"])
        else:
            from .inference import Runtime
            rt = Runtime(model, "cpu", self.cfg.infer_threads, self.cfg.infer_int8)
            size = params * (1 if rt.quantized else 4)
        rt.activate()
        try:
            for _ in range(15):
                rt.predict_array(x)
            t0 = time.perf_counter()
            for _ in range(frames):
                rt.predict_array(x)
            ms = (time.perf_counter() - t0) / frames * 1000
        finally:
            rt.release()
        return {"ms": ms, "backend": rt.describe(), "params": params, "size_mb": size / 1e6}

    def apply_candidate(self, result):
        """Use a compared model right away: it becomes the current model, is saved to
        the model path (the previous file is kept as <name>.prev.pt) and tracking
        restarts with it if it was running."""
        if self.busy:
            raise RuntimeError("Wait until %s has finished" % self.busy)
        was_inferring = self.inferring
        self.stop_inference()
        path = self.cfg.model_path
        if os.path.exists(path):
            import shutil
            shutil.copy2(path, self.model_base(path) + ".prev.pt")
        self.model = result["model"].eval()
        self.model.val_metrics = result["metrics"]
        self.cfg.model_arch, self.cfg.loss = result["arch"], result["loss"]
        self.save_model(path)
        self.save_config()
        log.info("Now using the %s + %s model", result["arch"], result["loss"].upper())
        if was_inferring:
            self.start_inference()

    # ------------------------------------------------------------ inference
    @property
    def inferring(self):
        return self._infer_thread is not None and self._infer_thread.is_alive()

    def start_inference(self):
        if self.inferring:
            return
        if self.busy in ("training", "comparing"):
            raise RuntimeError("Wait for training to finish")
        runtime = self._make_runtime()  # errors surface here, in the caller
        from .params import V1_UNSUPPORTED
        for _, target, _ in self.cfg.targets():
            if target in V1_UNSUPPORTED:
                log.warning("%s: VRCFaceTracking v6 always sends 0 for this v1 parameter; the tongue "
                            "directions are driven instead", target)
        self._infer_stop.clear()
        self._smoothed = {}
        self._stabilizers = {}
        self._infer_thread = threading.Thread(target=self._infer_loop, args=(runtime,), daemon=True,
                                              name="inference")
        self._infer_thread.start()

    def _make_runtime(self):
        cfg = self.cfg
        if self.use_onnx():
            try:
                return onnx_runtime.OnnxRuntime(self.onnx_model_base(), cfg.infer_device, cfg.infer_threads,
                                                cfg.infer_int8, expected_mode=cfg.input_mode,
                                                expected_outputs=cfg.num_classes)
            except Exception as e:
                if cfg.infer_engine == "onnx" or importlib.util.find_spec("torch") is None:
                    raise
                log.warning("ONNX Runtime unavailable for this model (%s); using PyTorch", e)
        from .inference import Runtime
        return Runtime(self.ensure_model(), cfg.infer_device, cfg.infer_threads, cfg.infer_int8)

    def stop_inference(self):
        self._infer_stop.set()
        self._fastcal_request.clear()
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=3)
        self._infer_thread = None

    def request_fastcal(self):
        if not self.inferring:
            self.start_inference()
        self._fastcal_request.set()

    def restart_inference(self):
        """Apply changed performance settings to a running tracker."""
        if self.inferring:
            self.stop_inference()
            self.start_inference()

    def run_benchmark(self, on_done=None):
        """Compare CPU fp32 / CPU int8 / GPU on this PC in the background. Tracking
        is paused meanwhile so it doesn't skew the numbers."""
        if self.busy:
            raise RuntimeError("Already %s" % self.busy)
        was_inferring = self.inferring
        self.stop_inference()
        self.busy = "benchmarking"
        cfg = self.cfg

        def run():
            results, error = [], None
            try:
                has_torch = importlib.util.find_spec("torch") is not None
                model = self.model
                if has_torch:
                    from .inference import benchmark
                    from .model import BuddyNet
                    if model is None:  # speed doesn't depend on the weights
                        model = BuddyNet(max(1, cfg.num_classes), cfg.input_mode, cfg.model_arch)
                    results += benchmark(model, threads=cfg.infer_threads)
                if onnx_runtime.available():
                    if has_torch:
                        from .onnx_export import export_onnx
                        base = os.path.join(tempfile.gettempdir(), "palbuddy-bench-%d" % os.getpid())
                        export_onnx(model, base)
                    else:
                        base = self.onnx_model_base()
                    results += onnx_runtime.benchmark_onnx(base, threads=cfg.infer_threads)
            except Exception as e:
                log.exception("benchmark failed")
                error = str(e)
            finally:
                self.busy = None
            if was_inferring:
                try:
                    self.start_inference()
                except Exception:
                    log.exception("could not resume tracking")
            if on_done:
                on_done(results, error)

        threading.Thread(target=run, daemon=True, name="benchmark").start()

    def _infer_loop(self, runtime):
        cfg = self.cfg
        runtime.activate()
        runtime.warmup()
        predict = runtime.predict
        self.infer_latency_ms = 0.0
        self.infer_backend = runtime.describe()
        seq = 0
        min_interval = 1.0 / cfg.max_send_rate if cfg.max_send_rate > 0 else 0.0
        infer_interval = 1.0 / cfg.max_infer_rate if cfg.max_infer_rate > 0 else 0.0
        next_due = 0.0
        last_send = 0.0
        plan_at = -PLAN_REFRESH
        classes, targets, merged_params, interval = [], [], [], infer_interval
        sigmoid_outputs = getattr(runtime, "output", "relu") == "sigmoid"
        soften, soften_key = None, None
        waiting_logged = False
        log.info("Inference started (%s%s)", self.infer_backend,
                 ", max %.0f Hz" % cfg.max_infer_rate if infer_interval else "")
        while not self._infer_stop.is_set():
            if self._fastcal_request.is_set():
                self._fastcal_request.clear()
                try:
                    self._run_fastcal(predict)
                except Exception:
                    log.exception("fastcal failed")
                self.fastcal_state = None
                continue
            now = time.monotonic()
            if now - plan_at >= PLAN_REFRESH:
                # output settings, re-read periodically instead of per frame
                plan_at = now
                classes = list(cfg.classes)
                targets = [(idx, classes[idx].name, target) for idx, target, _ in cfg.targets() if idx < len(classes)]
                self.osc_plan()
                merged_params = self.active_merged_params()
                if sigmoid_outputs and cfg.softness != soften_key:
                    soften_key, soften = cfg.softness, softener(cfg.softness)
                consumer = (targets and self.vrcft.connected) or (cfg.osc_enabled and self._osc_plan)
                if self.idle != (not consumer):
                    self.idle = not consumer
                    log.info("Nothing receives the output: tracking at 10 Hz until VRCFaceTracking connects or "
                             "an OSC output is set up" if self.idle else "Output connected: tracking at full rate")
            interval = max(infer_interval, IDLE_INTERVAL) if self.idle and now >= self._full_rate_until \
                else infer_interval
            if interval:
                # rate cap: sleep until the next slot, then take the newest frame
                # (frames in between are skipped, not queued)
                delay = next_due - now
                if delay > 0 and self._infer_stop.wait(min(delay, PLAN_REFRESH)):
                    break
                if delay > PLAN_REFRESH:
                    continue
            # Only run the network on *new* frames (the old loop re-ran on the
            # same frame every 10ms and slept even when a new frame was ready).
            seq, sample = self.hub.wait_next(seq, timeout=0.5)
            if sample is None:
                if not waiting_logged:
                    log.info("Waiting for tracker data...")
                    waiting_logged = True
                continue
            waiting_logged = False
            t0 = time.perf_counter()
            raw = predict(sample)
            if raw is None:
                continue
            ms = (time.perf_counter() - t0) * 1000
            # smoothed so the display is readable; first value taken as is
            self.infer_latency_ms = ms if not self.infer_latency_ms else self.infer_latency_ms * 0.9 + ms * 0.1
            self.infer_rate.tick()
            if interval:
                next_due = max(next_due + interval, time.monotonic())
            self.last_raw = raw

            # per-class weight 0..1 (raw / max_power), smoothed; VRCFT gets 2w-1 in -1..1
            cutoff = smoothing_cutoff(cfg.smoothing)
            frame_time = time.monotonic()
            raw_values = raw.tolist()  # Python floats: much faster than indexing numpy per element
            weights, class_raw, final = {}, {}, {}
            for idx, c in enumerate(classes):
                if idx >= len(raw_values):
                    break
                r = (soften(raw_values[idx]) if soften else raw_values[idx]) / c.max_power
                w_raw = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)
                class_raw[idx] = w_raw
                w = c.remap(w_raw)  # sensitivity range, e.g. 0.2..0.8 -> 0..1
                if c.stability > 0.0:
                    st = self._stabilizers.get(idx)
                    if st is None:
                        st = self._stabilizers[idx] = Stabilizer()
                    w = st.update(w, frame_time, c.stability)
                if cutoff is not None:
                    f = self._smoothed.get(idx)
                    if f is None:
                        f = self._smoothed[idx] = OneEuro()
                    w = f.update(w, frame_time, cutoff)
                    w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)
                weights[c.name] = w
                final[idx] = w
            self.last_class_raw, self.last_weights = class_raw, final
            out, pairs = {}, []
            for idx, name, target in targets:
                if name in weights:
                    v = weights[name] * 2.0 - 1.0
                    out[idx] = v
                    pairs.append((target, v))
            self.last_out = out
            merged = {m.name: m.combine(weights) for m in merged_params}
            self.last_merged = merged
            now = time.monotonic()
            if now - last_send >= min_interval:
                if pairs:
                    self.vrcft.send_targets(pairs)
                if cfg.osc_enabled and self._osc_plan:
                    refresh = now - self._bool_refresh > 1.0  # resend all bools now and then
                    if refresh:
                        self._bool_refresh = now
                    self._send_osc(weights, merged, force=refresh)
                last_send = now
        runtime.release()
        self._send_merged_neutral()  # leave merged parameters at neutral, not frozen
        if self.vrcft is not None:
            self.vrcft.clear()  # v6 module: fall back to plain SRanipal right away
        self.infer_backend = None
        log.info("Inference stopped")

    def _run_fastcal(self, predict):
        """Puppet each target shape on the avatar and record the network output
        while the user copies it; the average becomes that class's max_power."""
        targets = self.cfg.targets()
        if not targets:
            log.warning("No classes have a target shape; nothing to calibrate")
            return
        if not self.vrcft.connected:
            log.warning("VRCFT is not connected - the avatar can't be puppeted, calibrating blind")

        def stopped():
            return self._infer_stop.is_set()

        for t in range(10, 0, -1):
            self.fastcal_state = "Starting in %d s - follow the avatar" % t
            if self._infer_stop.wait(1):
                return
        self.vrcft.send_targets([(target, -1.0) for _, target, _ in targets])

        results = {}
        for n, (idx, target, _) in enumerate(targets):
            name = self.cfg.classes[idx].name
            self.fastcal_state = "Calibrating %s (%d/%d)" % (name, n + 1, len(targets))
            log.info(self.fastcal_state)
            for i in range(101):  # ease in over 1s
                self.vrcft.send_targets([(target, i / 50 - 1)])
                if self._infer_stop.wait(0.01):
                    return
            if self._infer_stop.wait(2):
                return
            total, count, seq = 0.0, 0, self.hub.latest()[0]
            end = time.monotonic() + 1.0
            while time.monotonic() < end and not stopped():
                seq, sample = self.hub.wait_next(seq, timeout=0.2)
                raw = predict(sample) if sample is not None else None
                if raw is not None:
                    total += float(raw[idx])
                    count += 1
            results[idx] = (total / count if count else 0.0) + 1e-9
            for i in range(101):  # ease out
                self.vrcft.send_targets([(target, 1 - i / 50)])
                if self._infer_stop.wait(0.01):
                    return

        for idx, value in results.items():
            if value < 0.01:
                log.warning("Class '%s' barely activated (%.4f) - keeping previous max_power",
                            self.cfg.classes[idx].name, value)
                continue
            self.cfg.classes[idx].max_power = round(value, 4)
        self.save_config()
        log.info("FastCal finished: %s", {self.cfg.classes[i].name: round(v, 4) for i, v in results.items()})


def unique_recording_name(cfg, name):
    """File name for a new recording that never overwrites an existing one:
    "smile" -> smile-em.mmap, then smile-2-em.mmap, smile-3-em.mmap, ...
    ("smile.mmap" -> smile.mmap, smile-2.mmap, ...)."""
    stem, suffix = (name[:-5], ".mmap") if name.endswith(".mmap") else (name, "-em.mmap")

    def taken(filename):
        path = cfg.dataset_path(filename)
        return any(os.path.exists(p) for p in (path, path + ".part", path + ".labels.json"))

    filename, n = stem + suffix, 2
    while taken(filename):
        filename = "%s-%d%s" % (stem, n, suffix)
        n += 1
    return filename


def dataset_files(cfg):
    folder = cfg.dataset_folder
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.endswith(".mmap"))

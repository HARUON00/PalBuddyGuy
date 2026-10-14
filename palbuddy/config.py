"""Persistent settings (config.json) shared by the GUI and the CLI.

Replaces the constants that used to be hand-edited at the top of script.py
(DATASET_FOLDER, order, to_replace, max_power_array).
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .params import is_valid_target
from .vrcft_names import vrcft_conflict

# config.json lives next to script.py / PalBuddyGuy.bat, independent of the working directory
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(APP_DIR, "config.json")

# Which trackers feed the network. Recordings always store both halves
# (eye channels 0-63, face channels 64-127; a missing tracker is zero-filled),
# so a recording made with both trackers can be reused for any mode.
INPUT_MODES = ("both", "face", "eye")
MODEL_ARCHS = ("standard", "lite", "compact")  # same names as model.ARCHS (kept here: no torch import)
LOSSES = ("mse", "bce")
INPUT_CHANNELS = {"both": slice(0, 128), "eye": slice(0, 64), "face": slice(64, 128)}


def channels_for(mode):
    return INPUT_CHANNELS[mode]


@dataclass
class ExpressionClass:
    """One output of the network.

    files: dataset recordings (.mmap, or legacy .pkl) that show this expression.
    target: SRanipal lip shape to drive with this output (name or index), or None
            to only train it (e.g. the mandatory "neutral" class).
    max_power: raw network output that maps to a fully expressed shape (+1).
    in_min, in_max: sensitivity - the part of the 0..1 range the expression really uses;
            stretched back to 0..1 (e.g. 0.2..0.8 -> 0..1).
    stability: filters out brief changes (median + low-pass over up to STABILITY_WINDOW s), for
            expressions that should hold rather than flicker along with speech.
    osc_name: optional custom avatar parameter the class weight (0..1) is sent to over OSC,
            as a float and/or binary bools (osc_format, osc_bits).
    """

    name: str
    files: List[str] = field(default_factory=list)
    target: Optional[str] = None
    max_power: float = 0.9
    in_min: float = 0.0
    in_max: float = 1.0
    stability: float = 0.0  # 0 = off .. 1 = strongest: ignore brief flickers (e.g. while talking)
    osc_name: Optional[str] = None
    osc_format: str = "float"
    osc_bits: int = 4

    def remap(self, w):
        """Apply the sensitivity range to a weight in 0..1."""
        span = self.in_max - self.in_min
        if span <= 1e-6:
            return 1.0 if w >= self.in_max else 0.0
        return min(1.0, max(0.0, (w - self.in_min) / span))


OSC_FORMATS = ("float", "binary", "both")
MAX_BINARY_BITS = 8


def osc_output_names(name, fmt, bits, signed):
    """Every avatar parameter name an OSC output writes, VRCFaceTracking style:
    float "<name>", binary bools "<name>1", "<name>2", "<name>4"... and "<name>Negative"."""
    names = []
    if fmt in ("float", "both"):
        names.append(name)
    if fmt in ("binary", "both"):
        names += ["%s%d" % (name, 1 << i) for i in range(bits)]
        if signed:
            names.append(name + "Negative")
    return names


def binary_bits(value, bits):
    """Same encoding as VRCFaceTracking's BinaryBaseParameter: bit i of int(value * 2^bits),
    all bits set for value ~1. value in 0..1."""
    value = min(1.0, max(0.0, float(value)))
    if value > 0.99999:
        return [True] * bits
    big = int(value * (1 << bits))
    return [bool((big >> i) & 1) for i in range(bits)]


PARAM_NAME_RE = re.compile(r"^[A-Za-z0-9_./-]+$")

# (min, max) presets for merged parameters: combined -1 -> min, 0 -> neutral (middle), +1 -> max
RANGE_PRESETS = {"-1..1": (-1.0, 1.0), "0..1": (0.0, 1.0), "0..2": (0.0, 2.0)}
MAX_TERM_WEIGHT = 10.0


@dataclass
class MergedTerm:
    """One class in a merged parameter and how much it contributes (negative = pulls down)."""

    cls: str
    weight: float = 1.0


@dataclass
class MergedParam:
    """A custom avatar parameter combining two or more trained classes, sent over OSC.

    combined = sum(term.weight * class weight), clamped to -1..1 (0 when all classes are 0),
    then mapped piecewise-linearly: -1 -> out_min, 0 -> neutral, +1 -> out_max.
    E.g. smile(+1) + sad(-1) -> -1..1, or in the spirit of VRCFT's EyeLidExpandedSqueeze:
    0.2*wide + 0.8*open - 1*squeeze. neutral defaults to the middle of the range.
    """

    name: str
    terms: List[MergedTerm] = field(default_factory=list)
    out_min: float = -1.0
    out_max: float = 1.0
    out_neutral: Optional[float] = None  # None = middle of out_min..out_max
    enabled: bool = True
    osc_format: str = "float"
    osc_bits: int = 4

    @classmethod
    def pair(cls, name, positive=None, negative=None, out_min=-1.0, out_max=1.0, **kw):
        """The classic "positive - negative" merged parameter."""
        terms = [MergedTerm(c, w) for c, w in ((positive, 1.0), (negative, -1.0)) if c]
        return cls(name, terms, out_min, out_max, **kw)

    @property
    def classes(self):
        return [t.cls for t in self.terms]

    @property
    def neutral(self):
        return (self.out_min + self.out_max) / 2.0 if self.out_neutral is None else self.out_neutral

    def combined(self, weights):
        """Weighted sum of the classes, -1..1."""
        total = 0.0
        for t in self.terms:
            total += t.weight * weights.get(t.cls, 0.0)
        return min(1.0, max(-1.0, total))

    def value_of(self, c):
        """Map a combined value (-1..1) to the output range through the neutral point."""
        n = self.neutral
        return n + c * (self.out_max - n) if c >= 0 else n + c * (n - self.out_min)

    def combine(self, weights):
        """weights: {class name: weight 0..1} -> output value."""
        return self.value_of(self.combined(weights))

    @property
    def signed(self):
        """A range crossing 0: binary output uses sign + magnitude (a Negative bool), like VRCFT."""
        return min(self.out_min, self.out_max) < 0 < max(self.out_min, self.out_max)

    def binary_input_of(self, value):
        """(value 0..1 to encode, negative) for binary output of an output value. Signed ranges:
        magnitude relative to the end on that side of 0; otherwise the position within the range."""
        if self.signed:
            end = max(self.out_min, self.out_max) if value >= 0 else min(self.out_min, self.out_max)
            return (value / end if end else 0.0), value < 0
        return (value - self.out_min) / (self.out_max - self.out_min), False

    def binary_input(self, weights):
        return self.binary_input_of(self.combine(weights))

    def formula(self):
        """Human-readable sum, e.g. "smile - sad" or "0.2*wide + 0.8*open - squeeze"."""
        parts = []
        for i, t in enumerate(self.terms):
            body = t.cls if abs(t.weight) == 1 else "%g*%s" % (abs(t.weight), t.cls)
            sign = "-" if t.weight < 0 else "+"
            parts.append(("-" if t.weight < 0 else "") + body if i == 0 else " %s %s" % (sign, body))
        return "".join(parts).strip() or "-"

    @property
    def beyond_sync_range(self):
        """VRChat syncs float parameters as -1..1; larger values only work locally."""
        return min(self.out_min, self.out_max) < -1.0 or max(self.out_min, self.out_max) > 1.0


def parse_terms(text):
    """"smile,-sad,0.2*wide,-0.5*squeeze" -> [MergedTerm]."""
    terms = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        sign = -1.0 if part.startswith("-") else 1.0
        part = part.lstrip("+-").strip()
        if "*" in part:
            coef, name = part.split("*", 1)
            weight = sign * float(coef)
        else:
            name, weight = part, sign
        terms.append(MergedTerm(name.strip(), weight))
    return terms


@dataclass
class Config:
    dataset_folder: str = "datasets"
    model_path: str = "buddyguy.pt"

    # Networking. Listening on 127.0.0.1 keeps the ports off the LAN.
    bind_host: str = "127.0.0.1"
    face_port: int = 18452  # SRanipal stream that is the face tracker when not swapped
    eye_port: int = 18453
    proxy_port: int = 18454  # tvm_proxy.py output (legacy two-process mode)
    vrcft_port: int = 26421
    # "direct": receive from SRanipal in this process (fastest, default)
    # "proxy":  connect to a separately running tvm_proxy.py
    source: str = "direct"
    swapped: bool = False
    stall_timeout: float = 2.0  # seconds without frames before a stream counts as stalled
    # "both" (eye + face tracker), "face" (facial tracker only) or "eye" (Pro Eye only)
    input_mode: str = "both"

    # Recording
    record_frames: int = 2048
    record_countdown: float = 5.0
    # store new recordings as float16 (half the disk space); older float32 ones still load
    record_float16: bool = True

    # Training
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 5e-5
    mixed_precision: bool = True
    cache_datasets_in_ram: bool = False
    # "standard" (original network), "lite" (~8x smaller) or "compact" (~30x smaller,
    # standardised input; see model.py). The training tab's "Compare" picks one on your data.
    model_arch: str = "standard"
    # "mse": original (ReLU outputs, MSE against one-hot targets); "bce": sigmoid outputs
    # trained with binary cross-entropy, each class independent of the others
    loss: str = "mse"
    # blend half of the expression samples with neutral frames (and their targets) so the
    # model learns in-between intensities; see trainer.BatchSampler
    mixup: bool = True
    # shift / scale / add noise to the training features (see trainer.augment_batch) so the
    # model doesn't depend on one session's exact headset position and lighting
    augment: bool = False
    # AdamW weight decay; 0 = plain Adam (original). It is multiplied by the learning rate,
    # so it has a noticeable effect mainly with the compact model's larger step
    weight_decay: float = 0.0
    # lite / compact: train a standard teacher first and let the smaller model learn from its
    # outputs as well as the labels (about twice the training time); see trainer.distill
    distill: bool = False
    validation_split: float = 0.1  # end of each recording held out to measure accuracy; 0 = off
    # validate on a whole recording of each class that has several (its last file) instead of
    # recording ends: measures another session / headset fit, i.e. overfitting. Fewer training frames.
    validation_sessions: bool = False

    # Inference
    smoothing: float = 0.0  # 0 = off .. 0.9 = strongest (1-Euro filter, see engine.smoothing_cutoff)
    # BCE (sigmoid) models only: 1 = off; higher = outputs follow the expression more
    # gradually instead of jumping between 0 and 1 (see engine.softener)
    softness: float = 1.0
    max_send_rate: float = 120.0  # Hz cap for VRCFT updates
    # VRCFT v6 module: "replace" SRanipal's value of a driven shape, or keep the "max" of both
    vrcft_override_mode: str = "replace"
    vrcft_wrap_sranipal: bool = True  # module installer: run SRanipal inside the PalBuddyGuy module
    vrcft_custom_libs: str = ""  # VRCFT modules folder; empty = %APPDATA%\VRCFaceTracking\CustomLibs
    # "auto": ONNX Runtime if installed, else PyTorch. ONNX Runtime is faster on the
    # CPU (int8) and tracking with it never loads PyTorch (~500 MB less RAM).
    infer_engine: str = "auto"  # "auto", "onnx" or "pytorch"
    # CPU by default: int8 inference costs ~1/3 of one core and leaves the GPU to VR
    infer_device: str = "cpu"  # "cpu", "auto" (GPU if available) or "gpu"
    infer_threads: int = 1  # CPU threads for inference; more = lower latency but more total CPU
    infer_int8: bool = True  # int8-quantize the linear layers when inferring on the CPU
    max_infer_rate: float = 0.0  # Hz cap for running the network, 0 = every new frame

    # Process placement (see palbuddy/system.py). Tracking needs little CPU, so by
    # default it yields to VR / the game when the CPU is busy.
    process_priority: str = "below_normal"  # "above_normal", "normal", "below_normal", "idle"
    cpu_affinity: str = "all"  # "all" or "ecores" (hybrid Intel CPUs: efficiency cores only)
    efficiency_mode: bool = False  # Windows 11 EcoQoS

    # GUI
    language: str = "auto"  # "auto", "en" or "ko"
    show_preview: bool = True
    # False: only the settings a first-time user needs; the rest keep their values but are hidden
    ui_advanced: bool = False

    classes: List[ExpressionClass] = field(default_factory=list)

    # Merged parameters (sent straight to VRChat over OSC, see palbuddy/osc.py)
    merged_params: List[MergedParam] = field(default_factory=list)
    osc_enabled: bool = True
    osc_host: str = "127.0.0.1"
    osc_port: int = 9000

    # ---------------------------------------------------------------- helpers
    @property
    def num_classes(self):
        return len(self.classes)

    def targets(self):
        """[(class_index, target_name, max_power)] for every class that drives a shape.
        Target names are SRanipal lip shapes or Unified Expressions (see params.py)."""
        out = []
        for i, c in enumerate(self.classes):
            if c.target not in (None, "") and is_valid_target(c.target):
                out.append((i, c.target, c.max_power))
        return out

    def dataset_path(self, filename):
        return filename if os.path.isabs(filename) else os.path.join(self.dataset_folder, filename)

    def validate(self):
        problems = []
        if self.model_arch not in MODEL_ARCHS:
            problems.append("model_arch must be one of %s." % ", ".join(MODEL_ARCHS))
        if self.loss not in LOSSES:
            problems.append("loss must be one of %s." % ", ".join(LOSSES))
        if not 0 <= self.weight_decay <= 10:
            problems.append("weight_decay must be between 0 and 10.")
        if not 0 <= self.validation_split < 0.5:
            problems.append("validation_split must be between 0 and 0.5.")
        if self.input_mode not in INPUT_MODES:
            problems.append("input_mode must be one of %s." % ", ".join(INPUT_MODES))
        if not self.classes:
            problems.append("No expression classes are configured.")
        for c in self.classes:
            if c.target not in (None, ""):
                if not is_valid_target(c.target):
                    problems.append("Class '%s' has an unknown target shape '%s'." % (c.name, c.target))
            if c.max_power <= 0:
                problems.append("Class '%s' needs max_power > 0." % c.name)
        return problems

    def output_problems(self):
        """Problems with OSC outputs (merged parameters, class OSC names). These don't block
        training; invalid outputs are just not sent."""
        problems = []
        names = {c.name for c in self.classes}
        seen = {}
        for c in self.classes:
            if c.osc_name:
                problems.extend(class_output_problems(c))
                for n in osc_output_names(c.osc_name, c.osc_format, c.osc_bits, False):
                    seen.setdefault(n.lower(), []).append(c.osc_name)
        for m in self.merged_params:
            problems.extend(merged_param_problems(m, names))
            for n in osc_output_names(m.name, m.osc_format, m.osc_bits, m.signed):
                seen.setdefault(n.lower(), []).append(m.name)
        for n, owners in seen.items():
            if len(owners) > 1:
                problems.append("OSC parameter '%s' is written by more than one output (%s)." % (n, ", ".join(owners)))
        return problems

    # ---------------------------------------------------------------- io
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Unknown keys (from newer/older versions or hand edits) are ignored."""
        data = dict(data)
        class_fields = set(ExpressionClass.__dataclass_fields__)
        classes = [ExpressionClass(**{k: v for k, v in c.items() if k in class_fields})
                   for c in data.pop("classes", []) if isinstance(c, dict) and "name" in c]
        merged = [merged_from_dict(m) for m in data.pop("merged_params", []) if isinstance(m, dict) and "name" in m]
        known = set(cls.__dataclass_fields__)
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.classes = classes
        cfg.merged_params = merged
        return cfg

    def save(self, path=DEFAULT_CONFIG_PATH):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path=DEFAULT_CONFIG_PATH):
        if not os.path.exists(path):
            cfg = default_config()
            cfg.save(path)
            return cfg
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _osc_name_problems(label, name, fmt, bits, signed):
    problems = []
    if not PARAM_NAME_RE.match(name or ""):
        return ["%s '%s': use letters, digits, _ . / - only." % (label, name)]
    if fmt not in OSC_FORMATS:
        problems.append("%s '%s': format must be one of %s." % (label, name, ", ".join(OSC_FORMATS)))
    if fmt != "float" and not 1 <= int(bits) <= MAX_BINARY_BITS:
        problems.append("%s '%s': binary needs 1-%d bits." % (label, name, MAX_BINARY_BITS))
    for n in osc_output_names(name, fmt, int(bits), signed):
        clash = vrcft_conflict(n)
        if clash:
            problems.append("%s '%s': '%s' would clash with VRCFaceTracking's parameter '%s' and disturb "
                            "normal face tracking. Pick another name." % (label, name, n, clash))
            break
    return problems


def class_output_problems(c):
    """Reasons a class's direct OSC output can't be used (empty = fine or not configured)."""
    if not c.osc_name:
        return []
    return _osc_name_problems("OSC parameter", c.osc_name, c.osc_format, c.osc_bits, False)


def merged_from_dict(d):
    """Also reads the older {"positive": ..., "negative": ...} form."""
    fields_ = set(MergedParam.__dataclass_fields__)
    kw = {k: v for k, v in d.items() if k in fields_ and k != "terms"}
    if "terms" in d:
        kw["terms"] = [MergedTerm(str(t.get("cls", "")), float(t.get("weight", 1.0)))
                       for t in d["terms"] if isinstance(t, dict)]
    else:
        kw["terms"] = [MergedTerm(c, w) for c, w in ((d.get("positive"), 1.0), (d.get("negative"), -1.0)) if c]
    return MergedParam(**kw)


def merged_param_problems(m, class_names):
    """Reasons a merged parameter can't be used (empty list = fine)."""
    problems = _osc_name_problems("Merged parameter", m.name, m.osc_format, m.osc_bits, m.signed)
    if not any(t.cls and t.weight for t in m.terms):
        problems.append("Merged parameter '%s' needs at least one class with a non-zero weight." % m.name)
    for t in m.terms:
        if t.cls not in class_names:
            problems.append("Merged parameter '%s' uses unknown class '%s'." % (m.name, t.cls))
        if abs(t.weight) > MAX_TERM_WEIGHT:
            problems.append("Merged parameter '%s': weight of '%s' must be within +-%g." % (m.name, t.cls,
                                                                                           MAX_TERM_WEIGHT))
    if m.out_min == m.out_max:
        problems.append("Merged parameter '%s' needs different min and max values." % m.name)
    elif not min(m.out_min, m.out_max) <= m.neutral <= max(m.out_min, m.out_max):
        problems.append("Merged parameter '%s': the neutral value must lie between min and max." % m.name)
    return problems


def default_config():
    """Same expressions and shape mapping as the original example script."""
    cfg = Config()
    cfg.classes = [
        ExpressionClass("neutral", ["neutral2-em.mmap", "neutral1-em.mmap", "nut3-em.mmap"]),
        ExpressionClass("happy", ["happy-em.mmap", "happy3-em.mmap"], "JawLeft"),
        ExpressionClass("mad", ["mad1.mmap", "mad2.mmap", "mad4-em.mmap"], "JawRight"),
        ExpressionClass("sad", ["sad-em.mmap"], "MouthPout"),
        ExpressionClass("purseleft", ["purseleft-em.mmap"], "MouthUpperUpLeft"),
        ExpressionClass("purseright", ["purseright-em.mmap"], "MouthUpperUpRight"),
        ExpressionClass("open", ["open-em.mmap"]),
        ExpressionClass("pog", ["pog-em.mmap"], "JawForward"),
        ExpressionClass("nwigleft", ["nwigleft-em.mmap"]),
        ExpressionClass("nwigright", ["nwigright-em.mmap"]),
        ExpressionClass("showteth", ["showteth-em.mmap"]),
    ]
    return cfg

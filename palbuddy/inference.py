"""Inference backends: where and how the network runs while tracking.

Measured on a 2.8 GHz Xeon core (batch 1, one thread): the original fp32 model
takes ~12.6 ms per frame, 97% of its 103 MB of weights sit in `linear1`, and
reading them every frame dominates. int8 dynamic quantization of the linear
layers brings that to ~6.9 ms and 28 MB with a max output difference of ~0.003,
without retraining. More CPU threads lower latency but raise total CPU use.

On a GPU the per-frame cost is tiny, but CUDA needs a few hundred MB of VRAM
for its context and the work competes with VR rendering. Which is better
depends on the PC, so the GUI can benchmark both.
"""

import contextlib
import copy
import logging
import time
import warnings

import numpy as np
import torch

log = logging.getLogger(__name__)

DEVICES = ("auto", "cpu", "gpu")


def gpu_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def resolve_device(choice):
    if choice in ("auto", "gpu"):
        dev = gpu_device()
        if dev is not None:
            return dev
        if choice == "gpu":
            log.warning("No usable GPU found, running inference on the CPU")
    return torch.device("cpu")


def quantize_int8(model):
    """int8 dynamic quantization of the Linear layers (CPU only). Returns None if
    this PyTorch build can't do it."""
    engines = getattr(torch.backends.quantized, "supported_engines", [])
    for engine in ("fbgemm", "x86", "qnnpack"):
        if engine in engines:
            torch.backends.quantized.engine = engine
            break
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # torch.ao deprecation notices
            return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception as e:  # pragma: no cover - depends on the torch build
        log.warning("int8 quantization unavailable (%s); using fp32", e)
        return None


@contextlib.contextmanager
def torch_threads(n):
    """Temporarily set torch's intra-op thread count (it is process global)."""
    old = torch.get_num_threads()
    if n and n > 0:
        torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(old)


class Runtime:
    """A ready-to-run copy of the model plus a preallocated input buffer."""

    def __init__(self, model, device_choice="auto", threads=1, int8=True):
        self.device = resolve_device(device_choice)
        self.threads = threads
        self.input_mode = model.input_mode
        self.num_outputs = model.num_outputs
        self.output = getattr(model, "output", "relu")
        base = copy.deepcopy(model).float().eval().to(self.device)
        self.quantized = False
        if self.device.type == "cpu" and int8:
            q = quantize_int8(base)
            if q is not None:
                base, self.quantized = q, True
        self.model = base
        mode = self.input_mode
        pin = self.device.type == "cuda"
        self._pin = pin
        self.host = torch.empty((1, 128 if mode == "both" else 64, 20, 20), dtype=torch.float32, pin_memory=pin)
        self.host_np = self.host.numpy()
        # where each tracker's features go in the input tensor (None = not used)
        self.eye_at = 0 if mode in ("both", "eye") else None
        self.face_at = 64 if mode == "both" else (0 if mode == "face" else None)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def describe(self):
        if self.device.type == "cpu":
            return "CPU %s ×%d" % ("int8" if self.quantized else "fp32", self.threads)
        return "%s fp32" % self.device.type.upper()

    def activate(self):
        """Apply the thread count (torch's is process global); call before running."""
        self._old_threads = torch.get_num_threads()
        if self.device.type == "cpu" and self.threads > 0:
            torch.set_num_threads(self.threads)

    def release(self):
        """Restore the thread count so later CPU training isn't limited to 1 thread."""
        torch.set_num_threads(getattr(self, "_old_threads", torch.get_num_threads()))

    def warmup(self, runs=3):
        """First calls are slow (allocations, cudnn autotune, quantized kernels);
        do them before real frames so the first tracked frames aren't delayed."""
        x = np.zeros(tuple(self.host.shape), dtype=np.float32)
        for _ in range(runs):
            self.predict_array(x)

    @torch.inference_mode()
    def predict(self, sample):
        """sample = (eye_bytes|None, face_bytes|None) -> np.ndarray of raw outputs,
        or None if a tracker this model needs is missing from the sample."""
        eye, face = sample
        for data, at in ((eye, self.eye_at), (face, self.face_at)):
            if at is None:
                continue
            if data is None:
                return None
            self.host_np[0, at:at + 64] = np.frombuffer(data, dtype=np.float32).reshape(64, 20, 20)
        out = self.model(self.host.to(self.device, non_blocking=self._pin))
        return out[0].float().cpu().numpy()

    @torch.inference_mode()
    def predict_array(self, x):
        return self.model(torch.from_numpy(x).to(self.device))[0].float().cpu().numpy()


def benchmark(model, threads=1, frames=200, include_gpu=True, log_fn=log.info):
    """Time the available backends on `model` (weights don't matter for speed).

    Returns a list of dicts: backend, ms (wall per frame), cpu_ms (process CPU
    time per frame, all threads), cpu_pct (% of one core at 60 fps).
    Run it while tracking is paused, other work in the process skews cpu_ms."""
    options = [("cpu", False), ("cpu", True)]
    if include_gpu and gpu_device() is not None:
        options.append(("gpu", False))
    rng = np.random.default_rng(0)
    channels = 128 if model.input_mode == "both" else 64
    x = rng.random((1, channels, 20, 20), dtype=np.float32)
    results = []
    for device, int8 in options:
        rt = Runtime(model, device, threads, int8)
        if int8 and not rt.quantized:
            continue
        with torch_threads(threads if rt.device.type == "cpu" else 0):
            for _ in range(20):
                rt.predict_array(x)
            t0, c0 = time.perf_counter(), time.process_time()
            for _ in range(frames):
                rt.predict_array(x)
            ms = (time.perf_counter() - t0) / frames * 1000
            cpu_ms = (time.process_time() - c0) / frames * 1000
        r = {"backend": rt.describe(), "ms": ms, "cpu_ms": cpu_ms, "cpu_pct": cpu_ms * 60 / 10}
        results.append(r)
        log_fn("Benchmark %-14s %6.2f ms/frame, CPU %5.2f ms (%3.0f%% of a core at 60 fps)" % (
            r["backend"], ms, cpu_ms, r["cpu_pct"]))
        del rt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results

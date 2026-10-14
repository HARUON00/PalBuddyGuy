"""Tracking with ONNX Runtime - no PyTorch import at all.

Measured on one 2.8 GHz Xeon core (batch 1): the standard model takes ~4.2 ms
per frame with the int8 ONNX file vs ~6.9 ms for PyTorch int8 and ~12.6 ms for
PyTorch fp32, and a process that only uses ONNX Runtime needs ~100 MB of RAM
instead of ~600 MB.

GPU use goes through DirectML (onnxruntime-directml on Windows; works on
NVIDIA, AMD and Intel GPUs) or CUDA (onnxruntime-gpu) when installed.
"""

import importlib.util
import json
import logging
import os
import time

import numpy as np

log = logging.getLogger(__name__)

GPU_PROVIDERS = ("DmlExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider")


def available():
    return importlib.util.find_spec("onnxruntime") is not None


def onnx_paths(base):
    # same layout as onnx_export.onnx_paths (duplicated so this module never imports torch)
    return {"fp32": base + ".onnx", "int8": base + ".int8.onnx", "meta": base + ".json"}


def read_meta(base):
    with open(onnx_paths(base)["meta"], "r", encoding="utf-8") as f:
        return json.load(f)


class OnnxRuntime:
    """Same interface as inference.Runtime."""

    def __init__(self, base, device_choice="auto", threads=1, int8=True, expected_mode=None, expected_outputs=None):
        import onnxruntime as ort

        paths = onnx_paths(base)
        meta = read_meta(base)
        self.input_mode = meta["input_mode"]
        self.num_outputs = meta["num_outputs"]
        self.arch = meta.get("arch", "standard")
        self.output = meta.get("output", "relu")
        if expected_mode is not None and self.input_mode != expected_mode:
            raise ValueError("ONNX model %s was trained for input '%s' but '%s' is selected"
                             % (paths["fp32"], self.input_mode, expected_mode))
        if expected_outputs is not None and self.num_outputs != expected_outputs:
            raise ValueError("ONNX model %s has %d outputs but %d classes are configured"
                             % (paths["fp32"], self.num_outputs, expected_outputs))
        self.threads = threads

        providers = []
        if device_choice in ("auto", "gpu"):
            providers = [p for p in GPU_PROVIDERS if p in ort.get_available_providers()]
            if not providers and device_choice == "gpu":
                log.warning("ONNX Runtime has no GPU provider here (install onnxruntime-directml); using the CPU")
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # don't let idle worker threads spin between frames (CPU the game could use)
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if "DmlExecutionProvider" in providers:
            so.enable_mem_pattern = False  # required by DirectML

        self.session = None
        if providers:  # GPUs get the fp32 model; int8 kernels are a CPU optimisation
            try:
                self.session = ort.InferenceSession(paths["fp32"], so, providers=providers + ["CPUExecutionProvider"])
                self.quantized = False
            except Exception as e:  # e.g. DirectML installed but the GPU/driver refuses
                log.warning("ONNX Runtime GPU providers %s failed (%s); using the CPU", providers, e)
        if self.session is None:
            self.quantized = bool(int8 and os.path.exists(paths["int8"]))
            self.session = ort.InferenceSession(paths["int8"] if self.quantized else paths["fp32"], so,
                                                providers=["CPUExecutionProvider"])
        self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name

        mode = self.input_mode
        self.x = np.zeros((1, 128 if mode == "both" else 64, 20, 20), dtype=np.float32)
        self.eye_at = 0 if mode in ("both", "eye") else None
        self.face_at = 64 if mode == "both" else (0 if mode == "face" else None)

    def describe(self):
        if self.provider == "CPUExecutionProvider":
            return "ONNX CPU %s ×%d" % ("int8" if self.quantized else "fp32", self.threads)
        return "ONNX %s" % self.provider.replace("ExecutionProvider", "")

    def activate(self):
        pass  # threads are part of the session options

    def release(self):
        pass

    def warmup(self, runs=3):
        for _ in range(runs):
            self.predict_array(self.x)

    def predict(self, sample):
        eye, face = sample
        for data, at in ((eye, self.eye_at), (face, self.face_at)):
            if at is None:
                continue
            if data is None:
                return None
            self.x[0, at:at + 64] = np.frombuffer(data, dtype=np.float32).reshape(64, 20, 20)
        return self.session.run(None, {self.input_name: self.x})[0][0]

    def predict_array(self, x):
        return self.session.run(None, {self.input_name: x})[0][0]


def benchmark_onnx(base, threads=1, frames=200, include_gpu=True, log_fn=log.info):
    results = []
    options = [("cpu", False), ("cpu", True)]
    if include_gpu:
        import onnxruntime as ort
        if any(p in ort.get_available_providers() for p in GPU_PROVIDERS):
            options.append(("gpu", False))
    x = None
    for device, int8 in options:
        rt = OnnxRuntime(base, device, threads, int8)
        if int8 and not rt.quantized:
            continue
        if x is None:
            x = np.random.default_rng(0).random(rt.x.shape, dtype=np.float32)
        for _ in range(20):
            rt.predict_array(x)
        t0, c0 = time.perf_counter(), time.process_time()
        for _ in range(frames):
            rt.predict_array(x)
        ms = (time.perf_counter() - t0) / frames * 1000
        cpu_ms = (time.process_time() - c0) / frames * 1000
        r = {"backend": rt.describe(), "ms": ms, "cpu_ms": cpu_ms, "cpu_pct": cpu_ms * 60 / 10}
        results.append(r)
        log_fn("Benchmark %-18s %6.2f ms/frame, CPU %5.2f ms (%3.0f%% of a core at 60 fps)" % (
            r["backend"], ms, cpu_ms, r["cpu_pct"]))
    return results

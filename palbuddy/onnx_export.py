"""Export a trained BuddyNet to ONNX (needs PyTorch; runs in the training setup).

Writes, next to each other:
  <base>.onnx        fp32 model
  <base>.int8.onnx   linear layers int8-quantized (needs the `onnx` package),
                     ~3x faster than fp32 on the CPU with ONNX Runtime
  <base>.json        metadata: input mode, outputs, class names, arch
"""

import copy
import json
import logging
import os
import warnings

import torch

log = logging.getLogger(__name__)


class _DropPreprocessHint(logging.Filter):
    """onnxruntime.quantization logs this advice on the root logger for every model;
    it doesn't apply to our tiny network."""

    def filter(self, record):
        return "pre-processing before quantization" not in record.getMessage()


def onnx_paths(base):
    """base is the model path without extension (e.g. .../buddyguy)."""
    return {"fp32": base + ".onnx", "int8": base + ".int8.onnx", "meta": base + ".json"}


def export_onnx(model, base, class_names=None, quantize=True):
    paths = onnx_paths(base)
    folder = os.path.dirname(os.path.abspath(base))
    os.makedirs(folder, exist_ok=True)
    model = copy.deepcopy(model).float().eval().to("cpu")  # leave the caller's model untouched
    channels = 128 if model.input_mode == "both" else 64
    dummy = torch.zeros(1, channels, 20, 20)
    tmp = paths["fp32"] + ".tmp"
    kwargs = dict(input_names=["x"], output_names=["y"], opset_version=13)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(model, dummy, tmp, dynamo=False, **kwargs)
        except TypeError:  # older torch without the dynamo argument
            torch.onnx.export(model, dummy, tmp, **kwargs)
    os.replace(tmp, paths["fp32"])

    int8 = False
    if quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            tmp = paths["int8"] + ".tmp"
            hint_filter = _DropPreprocessHint()
            logging.getLogger().addFilter(hint_filter)
            try:
                # Linear layers export as Gemm, which isn't in the default op list
                quantize_dynamic(paths["fp32"], tmp, weight_type=QuantType.QInt8,
                                 op_types_to_quantize=["MatMul", "Gemm"])
            finally:
                logging.getLogger().removeFilter(hint_filter)
            os.replace(tmp, paths["int8"])
            int8 = True
        except Exception as e:  # onnx / onnxruntime missing or too old
            log.info("int8 ONNX model not created (%s); fp32 only", e)
            if os.path.exists(paths["int8"]):
                os.remove(paths["int8"])  # never leave a stale int8 file for a new model

    meta = {"input_mode": model.input_mode, "num_outputs": model.num_outputs, "arch": getattr(model, "arch", "standard"),
            "output": getattr(model, "output", "relu"),
            "class_names": list(class_names) if class_names else None, "int8": int8}
    with open(paths["meta"] + ".tmp", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(paths["meta"] + ".tmp", paths["meta"])
    log.info("Exported ONNX model to %s%s", paths["fp32"], " (+ int8)" if int8 else "")
    return paths

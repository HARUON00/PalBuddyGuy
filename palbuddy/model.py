"""The expression network.

Architecture and checkpoint format are identical to the original script, so
existing buddyguy.pt files load unchanged. Differences:
  * a real nn.Module: dropout is controlled by train()/eval() instead of
    monkey-patching the dropout functions to identity for inference,
  * flatten(1) instead of reshaping with the global batch_size (which broke
    whenever the last batch or an inference batch had a different size),
  * runs on CPU when CUDA isn't available,
  * can take only the eye (64 ch) or only the face (64 ch) features, for
    setups with a single tracker. The checkpoint records which,
  * an optional "lite" architecture (same layout, fewer channels, ~8x fewer
    weights, ~5x less compute). Whether it tracks as well as "standard"
    depends on the data; compare the validation accuracy after training,
  * an optional "compact" architecture built for the small datasets this
    tool sees: the input features are standardised per channel (statistics
    from the training recordings, stored in the checkpoint), a 1x1 conv
    reduces 128 -> 64 channels, and the head is 6400 -> 128. ~1/25 of the
    standard weights and ~1/12 of its compute, so it overfits less,
  * optional sigmoid outputs (trained with BCE, see trainer.py) instead of the
    original ReLU outputs trained with MSE. Each class is then an independent
    "is this expression showing" probability instead of competing with the
    others for a one-hot target.
"""

import logging
import os

import torch
from torch import nn

log = logging.getLogger(__name__)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# arch -> (conv channels, hidden units). "standard" is the original network.
ARCHS = {"standard": (256, 1024), "lite": (128, 256), "compact": (64, 128)}
OUTPUTS = ("relu", "sigmoid")
LAYERS = {"standard": ("conv1", "conv2", "linear1", "linear2"),
          "lite": ("conv1", "conv2", "linear1", "linear2"),
          "compact": ("reduce", "conv1", "conv2", "linear1", "linear2")}


class BuddyNet(nn.Module):
    def __init__(self, num_outputs, input_mode="both", arch="standard", output="relu"):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError("unknown model arch '%s'" % arch)
        if output not in OUTPUTS:
            raise ValueError("unknown output '%s'" % output)
        self.num_outputs = num_outputs
        self.input_mode = input_mode
        self.arch = arch
        self.output = output
        channels, hidden = ARCHS[arch]
        in_channels = 128 if input_mode == "both" else 64
        self.normalized = arch == "compact"
        # per-channel input standardisation (identity until set_normalization)
        self.register_buffer("in_mean", torch.zeros(1, in_channels, 1, 1))
        self.register_buffer("in_scale", torch.ones(1, in_channels, 1, 1))
        if arch == "compact":
            self.reduce = nn.Conv2d(in_channels, channels, 1)
            in_channels = channels
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.linear1 = nn.Linear(channels * 100, hidden)
        self.linear2 = nn.Linear(hidden, num_outputs)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        # the original drops 70% of the raw features; the compact net has far fewer
        # weights to regularise and standardised inputs, so it drops less
        self.dropout_in = nn.Dropout(p=0.3 if arch == "compact" else 0.7)

    def set_normalization(self, mean, std):
        """mean, std: per input channel (numpy or tensor, length = input channels)."""
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(self.in_mean.shape)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(self.in_scale.shape)
        self.in_mean.copy_(mean)
        self.in_scale.copy_(1.0 / std)

    def forward(self, x, logits=False):
        """Outputs per class; logits=True skips the sigmoid (for BCE training)."""
        if self.normalized:
            x = (x - self.in_mean) * self.in_scale
        x = self.dropout_in(x)
        if self.arch == "compact":
            x = self.act(self.reduce(x))
        x = self.act(self.conv1(x))
        x = x + self.act(self.conv2(self.dropout(x)))
        x = self.act(self.linear1(self.dropout(x.flatten(1))))
        x = self.linear2(self.dropout(x))
        if self.output == "sigmoid":
            return x if logits else torch.sigmoid(x)
        return self.act(x)

    # ------------------------------------------------------------ checkpoints
    def save(self, path, class_names=None):
        tmp = path + ".tmp"
        data = {name: getattr(self, name).state_dict() for name in LAYERS[self.arch]}
        if class_names is not None:
            data["class_names"] = list(class_names)
        data["input_mode"] = self.input_mode
        data["arch"] = self.arch
        data["output"] = self.output
        if self.normalized:
            data["norm"] = {"mean": self.in_mean.clone(), "scale": self.in_scale.clone()}
        torch.save(data, tmp)
        os.replace(tmp, path)  # never leave a half-written checkpoint behind

    @classmethod
    def load(cls, path, device=None, expected_outputs=None, expected_mode=None):
        data = torch.load(path, map_location="cpu")
        n = data["linear2"]["weight"].shape[0]
        if expected_outputs is not None and n != expected_outputs:
            raise ValueError(
                "Model in %s has %d outputs but %d classes are configured. Retrain or fix the class list."
                % (path, n, expected_outputs))
        first = data["reduce"] if "reduce" in data else data["conv1"]
        in_channels = first["weight"].shape[1]
        # checkpoints from the original script have no "input_mode" and are always "both"
        mode = data.get("input_mode") or ("both" if in_channels == 128 else expected_mode or "face")
        if expected_mode is not None and mode != expected_mode:
            raise ValueError(
                "Model in %s was trained for input '%s' but input mode '%s' is selected. "
                "Switch the input mode back or retrain." % (path, mode, expected_mode))
        channels = data["conv1"]["weight"].shape[0]
        arch = data.get("arch") or next((a for a, (c, _) in ARCHS.items() if c == channels), "standard")
        model = cls(n, mode, arch, data.get("output", "relu"))
        for name in LAYERS[arch]:
            getattr(model, name).load_state_dict(data[name])
        if "norm" in data:
            model.in_mean.copy_(data["norm"]["mean"])
            model.in_scale.copy_(data["norm"]["scale"])
        model.class_names = data.get("class_names")
        if device is not None:
            model.to(device)
        return model

    def describe(self):
        return "%s/%s" % (self.arch, self.output)


def model_size(arch, input_mode="both", num_outputs=8):
    """(parameters, multiply-adds per frame) of an architecture."""
    channels, hidden = ARCHS[arch]
    in_ch = 128 if input_mode == "both" else 64
    params = macs = 0
    if arch == "compact":
        params += in_ch * channels + channels
        macs += in_ch * channels * 400
        in_ch = channels
    for cin in (in_ch, channels):
        params += cin * channels * 9 + channels
        macs += cin * channels * 9 * 100
    params += channels * 100 * hidden + hidden + hidden * num_outputs + num_outputs
    macs += channels * 100 * hidden + hidden * num_outputs
    return params, macs

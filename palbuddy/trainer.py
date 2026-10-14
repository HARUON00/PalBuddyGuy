"""Training loop.

Speedups over the original per-sample Python loop:
  * batches are gathered with one sorted fancy-index read per recording file
    instead of 128 random single-frame reads,
  * a background thread prepares the next batches (pinned memory, async copy)
    while the GPU trains on the current one,
  * optional automatic mixed precision on CUDA,
  * cudnn autotuning.
The sampling distribution is the same: uniform class, uniform file within the
class, uniform frame within the file.
"""

import contextlib
import logging
import queue
import threading
import time

import numpy as np
import torch

from .config import LOSSES, channels_for
from .datasets import load_labels, open_recording
from .model import BuddyNet, pick_device

log = logging.getLogger(__name__)


class TrainingCancelled(Exception):
    pass


NEUTRAL = 0  # the first class is the neutral face (the GUI and README require it)


def targets_for(c, values, num_classes):
    """Target rows for frames of class `c`. values: per-frame intensity from a guided
    recording (None = the whole recording shows the expression fully). A guided
    frame at intensity v is v * class + (1 - v) * neutral."""
    n = len(values) if values is not None else 1
    t = np.zeros((n, num_classes), dtype=np.float32)
    if values is None or c == NEUTRAL:
        t[:, c] = 1.0
    else:
        t[:, c] = values
        t[:, NEUTRAL] += 1.0 - values
    return t


class BatchSampler:
    """Uniform class, uniform file within the class, uniform frame within the file.

    mixup: fraction of the non-neutral samples blended with a random neutral frame,
    x = l * x_expr + (1 - l) * x_neutral with the targets blended the same way
    (l uniform in 0..1). It teaches in-between intensities from recordings that only
    show the full expression, and keeps the model from jumping between 0 and 1."""

    def __init__(self, recordings_per_class, batch_size, seed=None, channels=slice(0, 128), labels=None,
                 mixup=0.0):
        self.classes = recordings_per_class  # list[list[array (N,128,20,20)]]
        self.labels = labels or [[None] * len(recs) for recs in recordings_per_class]
        self.batch_size = batch_size
        self.channels = channels  # only the tracker(s) in use are read from disk
        self.rng = np.random.default_rng(seed)
        self.num_classes = len(recordings_per_class)
        self.eye = np.eye(self.num_classes, dtype=np.float32)
        self.mixup = mixup if self.num_classes > 1 else 0.0

    def _gather(self, cls, x, t):
        ch = self.channels
        for c in np.unique(cls):
            slots = np.nonzero(cls == c)[0]
            recs = self.classes[c]
            files = self.rng.integers(0, len(recs), size=len(slots))
            for f in np.unique(files):
                s = slots[files == f]
                rec = recs[f]
                idx = self.rng.integers(0, len(rec), size=len(s))
                order = np.argsort(idx)  # sequential-ish disk access for memmaps
                x[s[order]] = rec[idx[order], ch]
                lab = self.labels[c][f]
                t[s[order]] = targets_for(c, None if lab is None else lab[idx[order]], self.num_classes)

    def sample(self):
        b = self.batch_size
        cls = self.rng.integers(0, self.num_classes, size=b)
        ch = self.channels
        x = np.empty((b, ch.stop - ch.start, 20, 20), dtype=np.float32)
        t = np.empty((b, self.num_classes), dtype=np.float32)
        self._gather(cls, x, t)
        if self.mixup > 0:
            mix = np.nonzero((cls != NEUTRAL) & (self.rng.random(b) < self.mixup))[0]
            if len(mix):
                nx = np.empty((len(mix),) + x.shape[1:], dtype=np.float32)
                nt = np.empty((len(mix), self.num_classes), dtype=np.float32)
                self._gather(np.full(len(mix), NEUTRAL), nx, nt)
                lam = self.rng.random(len(mix)).astype(np.float32)
                x[mix] = lam[:, None, None, None] * x[mix] + (1 - lam[:, None, None, None]) * nx
                t[mix] = lam[:, None] * t[mix] + (1 - lam[:, None]) * nt
        return x, t


def check_recording(rec, channels, path):
    """Refuse recordings that have no data for a tracker the model needs, e.g.
    training "both" on a recording made with only the facial tracker."""
    probe = rec[np.linspace(0, len(rec) - 1, num=min(8, len(rec))).astype(int)]
    for name, sl in (("eye", slice(0, 64)), ("face", slice(64, 128))):
        inside = sl.start >= channels.start and sl.stop <= channels.stop
        if inside and not np.any(probe[:, sl]):
            raise ValueError("Recording '%s' contains no %s tracker data (it was recorded without that "
                             "tracker), so it can't be used in this input mode." % (path, name))


VAL_FRAMES_PER_FILE = 32
VAL_FRAMES_PER_SESSION = 128  # frames taken from a recording held out as a whole


def session_holdout(recordings):
    """{(class, recording index)} to hold out whole: the last recording of every
    class that has more than one."""
    return {(c, len(recs) - 1) for c, recs in enumerate(recordings) if len(recs) > 1}


def _val_frames(c, rec, lab, idx, channels, num_classes):
    x = np.ascontiguousarray(rec[idx, channels], dtype=np.float32)
    t = (targets_for(c, lab[idx], num_classes) if lab is not None
         else np.repeat(targets_for(c, None, num_classes), len(idx), axis=0))
    return x, t


def split_recordings(recordings, fraction, channels, labels=None, holdout=()):
    """Hold out frames for validation.

    Recordings listed in `holdout` ((class, index) pairs, see session_holdout)
    go to validation whole: they were made at another time / headset fit than
    the class's training recordings, so they show how well the model handles a
    new session - the situation where an overfitted model breaks down. The
    other recordings of such a class are not split. Every other recording has
    its *end* held out: neighbouring frames are nearly identical, so a random
    split would leak, but the tail is still the same session and therefore
    optimistic. Returns (train recordings, train labels, val_x, val_targets);
    the val arrays are None when nothing could be held out."""
    num_classes = len(recordings)
    labels = labels or [[None] * len(recs) for recs in recordings]
    if fraction <= 0:
        return recordings, labels, None, None
    holdout = set(holdout)
    whole_classes = {c for c, _ in holdout}
    train, train_labels, xs, ts = [], [], [], []
    for c, recs in enumerate(recordings):
        kept, kept_labels = [], []
        for i, (rec, lab) in enumerate(zip(recs, labels[c])):
            if (c, i) in holdout:
                idx = np.linspace(0, len(rec) - 1, num=min(VAL_FRAMES_PER_SESSION, len(rec))).astype(int)
                x, t = _val_frames(c, rec, lab, idx, channels, num_classes)
                xs.append(x)
                ts.append(t)
                continue
            if c in whole_classes:
                kept.append(rec)
                kept_labels.append(lab)
                continue
            n_train = int(len(rec) * (1 - fraction))
            if n_train < 16 or len(rec) - n_train < 4:
                kept.append(rec)  # too short to split
                kept_labels.append(lab)
                continue
            kept.append(rec[:n_train])
            kept_labels.append(None if lab is None else lab[:n_train])
            idx = np.linspace(n_train, len(rec) - 1, num=min(VAL_FRAMES_PER_FILE, len(rec) - n_train)).astype(int)
            x, t = _val_frames(c, rec, lab, idx, channels, num_classes)
            xs.append(x)
            ts.append(t)
        train.append(kept)
        train_labels.append(kept_labels)
    if not xs:
        return train, train_labels, None, None
    return train, train_labels, np.concatenate(xs), np.concatenate(ts)


HIT_THRESHOLD = 0.5  # the shown expression's output counts as recognised above this
FALSE_THRESHOLD = 0.3  # another class's output counts as a false activation above this


@torch.no_grad()
def evaluate(model, val_x, val_t, num_classes, device, batch=256):
    """Metrics on the held-out frames. val_t: target rows (N x classes, soft for guided
    recordings) or class indices (N,) meaning one-hot targets.
      loss        MSE against the targets (same scale for every model, so runs are comparable)
      acc         the expression with the largest target has the highest output
      hit         on frames where the expression is clearly shown (target >= 0.75),
                  its output is above HIT_THRESHOLD
      false       some class whose target is ~0 has an output above FALSE_THRESHOLD
                  (a wrong shape moves)
      err         mean |output - target| of the shown expression (how well intensity is followed)
      per_class   acc per class
      score       one number to rank models, see score()
    """
    val_t = np.asarray(val_t)
    if val_t.ndim == 1:
        val_t = np.eye(num_classes, dtype=np.float32)[val_t]
    was_training = model.training
    model.eval()
    cls_all = val_t.argmax(axis=1)
    loss_sum = err_sum = 0.0
    correct = np.zeros(num_classes)
    hits, clear, falses = 0.0, 0.0, 0.0
    counts = np.bincount(cls_all, minlength=num_classes).astype(float)
    for i in range(0, len(val_x), batch):
        x = torch.from_numpy(val_x[i:i + batch]).to(device)
        target = torch.from_numpy(np.ascontiguousarray(val_t[i:i + batch], dtype=np.float32)).to(device)
        pred = model(x).float()
        cls = target.argmax(dim=1)
        loss_sum += float(((pred - target) ** 2).mean(dim=1).sum())
        np.add.at(correct, cls_all[i:i + batch], (pred.argmax(dim=1) == cls).cpu().numpy())
        own = pred.gather(1, cls[:, None])[:, 0]
        own_t = target.gather(1, cls[:, None])[:, 0]
        err_sum += float((own - own_t).abs().sum())
        shown = own_t >= 0.75
        clear += float(shown.sum())
        hits += float(((own > HIT_THRESHOLD) & shown).sum())
        others = pred.masked_fill(target > 0.05, 0.0).max(dim=1).values
        falses += float((others > FALSE_THRESHOLD).sum())
    if was_training:
        model.train()
    total = counts.sum()
    m = {"loss": loss_sum / len(val_x), "acc": float(correct.sum() / total),
         "hit": hits / clear if clear else 0.0, "false": falses / total, "err": err_sum / total,
         "per_class": [float(c / n) if n else float("nan") for c, n in zip(correct, counts)]}
    m["score"] = score(m)
    return m


def score(m):
    """0..100. False activations weigh double: a wrong shape moving is more visible
    on the avatar than a correct one moving a bit less. The intensity error counts
    once, so a model that follows guided intensities closely ranks higher."""
    return 100.0 * (m["acc"] + m["hit"] + 2.0 * (1.0 - m["false"]) + (1.0 - min(1.0, m["err"]))) / 5.0


NORM_FRAMES = 4096


def input_statistics(recordings, channels, seed=0):
    """Per-channel mean and std of the training frames (for the compact model)."""
    rng = np.random.default_rng(seed)
    recs = [rec for recs in recordings for rec in recs]
    per = max(1, NORM_FRAMES // len(recs))
    parts = []
    for rec in recs:
        idx = np.sort(rng.choice(len(rec), size=min(per, len(rec)), replace=False))
        parts.append(np.asarray(rec[idx, channels], dtype=np.float32))
    x = np.concatenate(parts)
    mean = x.mean(axis=(0, 2, 3), dtype=np.float64)
    std = x.std(axis=(0, 2, 3), dtype=np.float64)
    # channels that barely move in the training data mustn't turn tiny runtime
    # differences into huge inputs
    floor = max(1e-6, 0.05 * float(np.median(std)))
    return mean.astype(np.float32), np.maximum(std, floor).astype(np.float32)


def load_recordings(cfg, log_fn=print):
    channels = channels_for(cfg.input_mode)
    recordings, labels = [], []
    total = guided = 0
    for c in cfg.classes:
        recs, labs = [], []
        for name in c.files:
            path = cfg.dataset_path(name)
            try:
                rec = open_recording(path, in_ram=cfg.cache_datasets_in_ram)
            except FileNotFoundError:
                raise FileNotFoundError("Recording '%s' of class '%s' not found" % (path, c.name))
            check_recording(rec, channels, path)
            lab = load_labels(path, len(rec))
            guided += lab is not None
            recs.append(rec)
            labs.append(lab)
            total += len(rec)
        if not recs:
            raise ValueError("Class '%s' has no recordings" % c.name)
        recordings.append(recs)
        labels.append(labs)
    log_fn("Loaded %d recordings (%d frames, %d guided) for %d classes" % (
        sum(map(len, recordings)), total, guided, len(recordings)))
    return recordings, labels, total


MIXUP_FRACTION = 0.5  # share of expression samples blended with neutral when cfg.mixup is on
OUTPUT_FOR_LOSS = {"mse": "relu", "bce": "sigmoid"}
# the compact net (fewer weights, standardised input) needs a larger step to train in the same epochs
ARCH_LR_SCALE = {"compact": 10.0}


class Data:
    """Recordings split into training frames and held-out validation frames."""

    def __init__(self, cfg, log_fn=print):
        recordings, labels, self.total_frames = load_recordings(cfg, log_fn)
        self.channels = channels_for(cfg.input_mode)
        holdout = session_holdout(recordings) if cfg.validation_sessions and cfg.validation_split > 0 else set()
        self.recordings, self.labels, self.val_x, self.val_t = split_recordings(
            recordings, cfg.validation_split, self.channels, labels, holdout)
        if holdout:
            held = sorted(holdout)
            log_fn("Validating on whole recordings from another session: %s" % ", ".join(
                "%s (%s)" % (cfg.classes[c].files[i], cfg.classes[c].name) for c, i in held))
            single = [cfg.classes[c].name for c in range(len(recordings)) if c not in {c for c, _ in held}]
            if single:
                log_fn("Only one recording, validated on its end instead: %s - record these again on another "
                       "day (or after putting the headset back on) to check them across sessions"
                       % ", ".join(single))
        elif cfg.validation_sessions and self.val_x is not None:
            log_fn("No class has two recordings, so validation uses the end of each recording")
        if self.val_x is not None:
            log_fn("Holding out %d frames for validation" % len(self.val_x))
        self._stats = None

    def statistics(self):
        if self._stats is None:
            self._stats = input_statistics(self.recordings, self.channels)
        return self._stats


def train(cfg, on_progress=None, stop_event=None, log_fn=print, init_model=None, device=None,
          data=None, arch=None, loss=None, seed=None):
    """Train a model on the configured classes and return it (on `device`).
    arch / loss default to cfg.model_arch / cfg.loss; data (a Data) can be shared
    between runs so they see the same frames."""
    device = device or pick_device()
    stop_event = stop_event or threading.Event()
    on_progress = on_progress or (lambda **kw: None)
    arch = arch or cfg.model_arch
    loss_name = loss or cfg.loss
    if loss_name not in LOSSES:
        raise ValueError("loss must be one of %s" % ", ".join(LOSSES))

    data = data or Data(cfg, log_fn)
    recordings, val_x, val_t, channels = data.recordings, data.val_x, data.val_t, data.channels
    if init_model is not None:
        model = init_model
        if model.output != OUTPUT_FOR_LOSS[loss_name]:
            raise ValueError("The current model has %s outputs; it can't continue training with %s loss"
                             % (model.output, loss_name.upper()))
    else:
        model = BuddyNet(cfg.num_classes, cfg.input_mode, arch, OUTPUT_FOR_LOSS[loss_name])
        if model.normalized:
            model.set_normalization(*data.statistics())
    if model.num_outputs != cfg.num_classes:
        raise ValueError("Model output count does not match the class list")
    if model.input_mode != cfg.input_mode:
        raise ValueError("Model input mode '%s' does not match '%s'" % (model.input_mode, cfg.input_mode))
    model.to(device).train()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    log_fn("Training a %s model with %s loss" % (model.arch, loss_name.upper()))

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate * ARCH_LR_SCALE.get(model.arch, 1.0))
    use_amp = cfg.mixed_precision and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    mse = torch.nn.MSELoss()
    bce = torch.nn.BCEWithLogitsLoss()

    # one "epoch" = as many samples as the original: 2048 frames per class
    steps = max(1, (2048 * cfg.num_classes) // cfg.batch_size)
    # one sampler (and RNG) per loader thread: numpy Generators aren't thread safe
    seeds = np.random.SeedSequence(seed).spawn(2)
    if seed is not None:
        torch.manual_seed(seed)
    q = queue.Queue(maxsize=6)
    producer_stop = threading.Event()

    def put(item):
        while not producer_stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return
            except queue.Full:
                pass

    def producer(seed):
        pin = device.type == "cuda"
        sampler = BatchSampler(recordings, cfg.batch_size, seed=seed, channels=channels, labels=data.labels,
                               mixup=MIXUP_FRACTION if cfg.mixup else 0.0)
        try:
            while not producer_stop.is_set():
                x, y = sampler.sample()
                x, y = torch.from_numpy(x), torch.from_numpy(y)
                if pin:
                    x, y = x.pin_memory(), y.pin_memory()
                put((x, y))
        except Exception as e:  # surface loader errors in the training thread
            put(e)

    workers = [threading.Thread(target=producer, args=(seeds[i],), daemon=True, name="batch-loader-%d" % i)
               for i in range(2)]
    for w in workers:
        w.start()

    history = []
    best = None  # (score, epoch, weights): the epoch that did best on the held-out frames is kept
    start = time.monotonic()
    try:
        for epoch in range(cfg.epochs):
            running = 0.0
            for step in range(steps):
                if stop_event.is_set():
                    raise TrainingCancelled()
                item = q.get()
                if isinstance(item, Exception):
                    raise item
                x = item[0].to(device, non_blocking=True)
                y = item[1].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with (torch.autocast(device_type="cuda") if use_amp else contextlib.nullcontext()):
                    pred = model(x, logits=True)
                if loss_name == "bce":
                    loss = bce(pred.float(), y) * 100
                else:
                    loss = mse(pred.float(), y) * 100  # same scale as the original
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                value = float(loss.detach()) / 100
                running += value
                if step % 4 == 0 or step == steps - 1:
                    on_progress(epoch=epoch, epochs=cfg.epochs, step=step + 1, steps=steps, loss=value,
                                avg=running / (step + 1), elapsed=time.monotonic() - start)
            avg = running / steps
            history.append(avg)
            val = {}
            if val_x is not None:
                m = evaluate(model, val_x, val_t, cfg.num_classes, device)
                val = {"val_loss": m["loss"], "val_acc": m["acc"], "val_hit": m["hit"], "val_false": m["false"],
                       "val_err": m["err"], "val_score": m["score"]}
                if best is None or m["score"] > best[0]:
                    names = [c.name for c in cfg.classes]
                    metrics = dict(val, per_class=dict(zip(names, m["per_class"])), epoch=epoch + 1)
                    best = (m["score"], metrics, {k: v.detach().clone() for k, v in model.state_dict().items()})
                log_fn("Epoch %d/%d  avg loss %.6f  val: accuracy %.1f%%  recognised %.1f%%  false %.1f%%  "
                       "intensity error %.2f  score %.1f" % (epoch + 1, cfg.epochs, avg, m["acc"] * 100,
                                                             m["hit"] * 100, m["false"] * 100, m["err"], m["score"]))
            else:
                log_fn("Epoch %d/%d  avg loss %.6f" % (epoch + 1, cfg.epochs, avg))
            on_progress(epoch=epoch + 1, epochs=cfg.epochs, step=steps, steps=steps, loss=avg, avg=avg,
                        elapsed=time.monotonic() - start, epoch_done=True, **val)
    finally:
        producer_stop.set()
        for w in workers:
            w.join(timeout=5)
    model.eval()
    if best is not None:
        _, metrics, weights = best
        if metrics["epoch"] != len(history):
            model.load_state_dict(weights)
            log_fn("Keeping the weights of epoch %d, which did best on the held-out frames" % metrics["epoch"])
        model.val_metrics = metrics
        worst = sorted(metrics["per_class"].items(), key=lambda kv: kv[1])[:3]
        log_fn("Validation accuracy per class (worst first): %s" % ", ".join(
            "%s %.0f%%" % (name, acc * 100) for name, acc in worst))
    if loss_name == "mse" and history and history[-1] > 0.001:
        log_fn("Warning: final loss %.6f is above 0.001 - check your recordings / class list" % history[-1])
    return model, history


# candidates the comparison trains by default: (arch, loss)
DEFAULT_CANDIDATES = (("standard", "mse"), ("lite", "mse"), ("standard", "bce"), ("compact", "bce"),
                      ("compact", "mse"))


def compare(cfg, candidates=DEFAULT_CANDIDATES, on_progress=None, stop_event=None, log_fn=print, device=None,
            seed=1234):
    """Train every (arch, loss) candidate on the same frames with the same sampling
    seed and return [{"arch", "loss", "model", "metrics", "history"}] in candidate order."""
    if cfg.validation_split <= 0:
        raise ValueError("Comparing models needs held-out frames: set the validation split above 0")
    on_progress = on_progress or (lambda **kw: None)
    data = Data(cfg, log_fn)
    if data.val_x is None:
        raise ValueError("The recordings are too short to hold out frames for comparing models")
    results = []
    for i, (arch, loss_name) in enumerate(candidates):
        log_fn("Comparing %d/%d: %s + %s" % (i + 1, len(candidates), arch, loss_name.upper()))
        progress = (lambda k: lambda **info: on_progress(candidate=k, candidates=len(candidates), **info))(i)
        model, history = train(cfg, on_progress=progress, stop_event=stop_event, log_fn=log_fn, device=device,
                               data=data, arch=arch, loss=loss_name, seed=seed)
        results.append({"arch": arch, "loss": loss_name, "model": model.cpu(), "history": history,
                        "metrics": model.val_metrics})
    return results


def recommend(results, speed_key="ms", tolerance=0.5):
    """Index of the best result: highest score; within `tolerance` points of it, the
    fastest (then the smallest) wins, since those models track just as well for less."""
    if not results:
        return None
    top = max(r["metrics"]["val_score"] for r in results)
    close = [i for i, r in enumerate(results) if r["metrics"]["val_score"] >= top - tolerance]
    return min(close, key=lambda i: (results[i].get(speed_key) or float("inf"), results[i].get("params") or 0))

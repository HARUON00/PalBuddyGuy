"""Tkinter GUI for recording, training, calibrating and running the tracker.

Tk is not thread safe: worker threads only put messages on `self.events`,
which the Tk main loop drains every 50 ms.
"""

import dataclasses
import locale
import logging
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .config import (LOSSES, MAX_BINARY_BITS, MODEL_ARCHS, OSC_FORMATS, RANGE_PRESETS, ExpressionClass,
                     MergedParam, MergedTerm, class_output_problems, merged_param_problems)
from .datasets import frame_count, labels_path, remove_recording
from .engine import STABILITY_WINDOW, dataset_files
from .frames import decode_camera
from .i18n import Translator
from .params import all_target_names

log = logging.getLogger(__name__)

STATE_COLORS = {"ok": "#2e9d4f", "stalled": "#d69a00", "disconnected": "#c0392b", "not_used": "#9a9a9a"}
MODES = ("both", "face", "eye")
PREVIEW_SCALE = 2


def detect_language(setting):
    if setting in ("en", "ko"):
        return setting
    try:
        loc = (locale.getlocale()[0] or "") + (os.environ.get("LANG") or "")
    except ValueError:
        loc = ""
    return "ko" if "ko" in loc.lower() or "korean" in loc.lower() else "en"


class QueueLogHandler(logging.Handler):
    def __init__(self, q):
        super().__init__()
        self.q = q
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            self.q.put(("log", self.format(record)))
        except Exception:
            pass


def gray_to_photo(img):
    """numpy uint8 (H, W) -> tk.PhotoImage via an in-memory PGM (no PIL needed)."""
    h, w = img.shape
    data = b"P5 %d %d 255\n" % (w, h) + np.ascontiguousarray(img).tobytes()
    return tk.PhotoImage(data=data, format="PPM")


def zoom_into(target, img, factor):
    """Draw a small grayscale image into an existing PhotoImage, scaled by Tk. About 5x cheaper
    than scaling in numpy and creating a new PhotoImage per frame (measured 1.6 vs 9 ms)."""
    small = gray_to_photo(img)
    target.tk.call(target, "copy", small, "-zoom", factor, factor)


def format_row(parent, row, app, fmt_var, bits_var):
    """OSC format (float / binary / both) + binary resolution widgets on one grid row."""
    t = app.t
    labels = {f: t("fmt_" + f) for f in OSC_FORMATS}
    shown = tk.StringVar(value=labels.get(fmt_var.get(), labels["float"]))
    ttk.Label(parent, text=t("osc_format")).grid(row=row, column=0, sticky="w", pady=2)
    f = ttk.Frame(parent)
    f.grid(row=row, column=1, columnspan=3, sticky="w", pady=2)
    box = ttk.Combobox(f, textvariable=shown, values=list(labels.values()), state="readonly", width=16)
    box.pack(side="left")
    ttk.Label(f, text=t("osc_bits")).pack(side="left", padx=(10, 4))
    spin = ttk.Spinbox(f, from_=1, to=MAX_BINARY_BITS, textvariable=bits_var, width=4)
    spin.pack(side="left")

    def sync(*_):
        fmt_var.set(next(k for k, v in labels.items() if v == shown.get()))
        spin.state(["disabled"] if fmt_var.get() == "float" else ["!disabled"])
    box.bind("<<ComboboxSelected>>", sync)
    sync()


class ClassDialog(tk.Toplevel):
    def __init__(self, app, cls=None):
        super().__init__(app.root)
        self.app = app
        t = app.t
        self.title(t("class_dialog"))
        self.transient(app.root)
        self.result = None
        cls = cls or ExpressionClass("", [], None, 0.9)

        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text=t("col_name")).grid(row=0, column=0, sticky="w")
        self.name = tk.StringVar(value=cls.name)
        ttk.Entry(frm, textvariable=self.name, width=30).grid(row=0, column=1, sticky="ew")

        ttk.Label(frm, text=t("col_target")).grid(row=1, column=0, sticky="w")
        self.none_label = t("none")
        shapes = [self.none_label] + all_target_names()  # SRanipal names, then Unified-only (v6 module)
        self.target = tk.StringVar(value=cls.target or self.none_label)
        ttk.Combobox(frm, textvariable=self.target, values=shapes, state="readonly", width=28).grid(
            row=1, column=1, sticky="ew")

        ttk.Label(frm, text=t("col_power")).grid(row=2, column=0, sticky="w")
        self.power = tk.StringVar(value=str(cls.max_power))
        ttk.Entry(frm, textvariable=self.power, width=10).grid(row=2, column=1, sticky="w")

        self.original = cls
        ttk.Label(frm, text=t("osc_name")).grid(row=3, column=0, sticky="w", pady=(6, 2))
        self.osc_name = tk.StringVar(value=cls.osc_name or "")
        ttk.Entry(frm, textvariable=self.osc_name, width=30).grid(row=3, column=1, sticky="ew", pady=(6, 2))
        self.osc_format = tk.StringVar(value=cls.osc_format)
        self.osc_bits = tk.IntVar(value=cls.osc_bits)
        fmt_frame = ttk.Frame(frm)
        fmt_frame.grid(row=4, column=0, columnspan=2, sticky="w")
        format_row(fmt_frame, 0, app, self.osc_format, self.osc_bits)
        ttk.Label(frm, text=t("osc_name_hint"), foreground="#666", wraplength=420, justify="left").grid(
            row=5, column=0, columnspan=2, sticky="w")

        ttk.Label(frm, text=t("files_hint")).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.files = tk.Listbox(frm, selectmode="multiple", height=10, exportselection=False)
        self.files.grid(row=7, column=0, columnspan=2, sticky="nsew")
        available = dataset_files(app.engine.cfg)
        for f in cls.files:  # keep entries whose file is missing (e.g. on another drive)
            if f not in available:
                available.append(f)
        for i, f in enumerate(available):
            self.files.insert("end", f)
            if f in cls.files:
                self.files.selection_set(i)

        btns = ttk.Frame(frm)
        btns.grid(row=8, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(btns, text="OK", command=self.ok).pack(side="left", padx=4)
        ttk.Button(btns, text=t("cancel"), command=self.destroy).pack(side="left")
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(7, weight=1)
        self.grab_set()
        self.wait_window()

    def ok(self):
        try:
            power = float(self.power.get())
            if power <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(self.app.t("error"), "max power > 0", parent=self)
            return
        name = self.name.get().strip()
        if not name:
            return
        target = self.target.get()
        files = [self.files.get(i) for i in self.files.curselection()]
        try:
            bits = int(self.osc_bits.get())
        except (ValueError, tk.TclError):
            bits = 0
        # keep fields this dialog doesn't edit (sensitivity range)
        result = dataclasses.replace(self.original, name=name, files=files,
                                     target=None if target == self.none_label else target, max_power=power,
                                     osc_name=self.osc_name.get().strip() or None,
                                     osc_format=self.osc_format.get(), osc_bits=bits)
        problems = class_output_problems(result)
        if problems:
            messagebox.showerror(self.app.t("error"), "\n".join(problems), parent=self)
            return
        self.result = result
        self.destroy()


class MergedDialog(tk.Toplevel):
    """Edit one merged parameter: name, weighted classes, output range with neutral point."""

    MAX_TERMS = 8

    def __init__(self, app, param=None):
        super().__init__(app.root)
        self.app = app
        t = app.t
        self.title(t("merged_dialog"))
        self.transient(app.root)
        self.result = None
        param = param or MergedParam("", [MergedTerm(app.cfg.classes[1].name if len(app.cfg.classes) > 1 else "", 1.0)])
        self.class_names = [c.name for c in app.cfg.classes]

        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)
        self.name = tk.StringVar(value=param.name)
        ttk.Label(frm, text=t("col_name")).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(frm, textvariable=self.name, width=28).grid(row=0, column=1, columnspan=5, sticky="w", pady=2)

        ttk.Label(frm, text=t("merged_terms")).grid(row=1, column=0, sticky="nw", pady=(6, 2))
        self.terms_frame = ttk.Frame(frm)
        self.terms_frame.grid(row=1, column=1, columnspan=5, sticky="w", pady=(6, 2))
        self.term_rows = []  # (class var, weight var, row frame)
        for term in param.terms:
            self._add_term_row(term.cls, term.weight)
        self.add_term_btn = ttk.Button(frm, text=t("merged_add_term"), command=lambda: self._add_term_row("", 1.0))
        self.add_term_btn.grid(row=2, column=1, sticky="w")
        ttk.Label(frm, text=t("merged_terms_hint"), foreground="#666", wraplength=420, justify="left").grid(
            row=3, column=0, columnspan=6, sticky="w", pady=(2, 6))

        self.min = tk.StringVar(value="%g" % param.out_min)
        self.mid = tk.StringVar(value="%g" % param.neutral)
        self.max = tk.StringVar(value="%g" % param.out_max)
        self.enabled = tk.BooleanVar(value=param.enabled)
        self.preset_labels = {k: t("range_" + k.replace("..", "_").replace("-", "m")) for k in RANGE_PRESETS}
        self.preset_labels["custom"] = t("range_custom")
        current = next((k for k, v in RANGE_PRESETS.items() if v == (param.out_min, param.out_max)), "custom")
        self.preset = tk.StringVar(value=self.preset_labels[current])
        ttk.Label(frm, text=t("merged_range")).grid(row=4, column=0, sticky="w", pady=2)
        preset_box = ttk.Combobox(frm, textvariable=self.preset, values=list(self.preset_labels.values()),
                                  state="readonly", width=26)
        preset_box.grid(row=4, column=1, columnspan=5, sticky="w")
        preset_box.bind("<<ComboboxSelected>>", lambda e: self._apply_preset())
        entries = {}
        for col, (label, var) in enumerate((("min", self.min), (t("merged_neutral"), self.mid), ("max", self.max))):
            ttk.Label(frm, text=label).grid(row=5, column=col * 2, sticky="e", padx=(0, 4))
            entries[label] = ttk.Entry(frm, textvariable=var, width=8)
            entries[label].grid(row=5, column=col * 2 + 1, sticky="w")
        self.min_entry, self.max_entry = entries["min"], entries["max"]
        self.osc_format = tk.StringVar(value=param.osc_format)
        self.osc_bits = tk.IntVar(value=param.osc_bits)
        fmt = ttk.Frame(frm)
        fmt.grid(row=6, column=0, columnspan=6, sticky="w")
        format_row(fmt, 0, app, self.osc_format, self.osc_bits)
        ttk.Checkbutton(frm, text=t("merged_enabled"), variable=self.enabled).grid(
            row=7, column=0, columnspan=6, sticky="w", pady=(4, 0))
        self.preview = ttk.Label(frm, text="", foreground="#666", wraplength=440, justify="left")
        self.preview.grid(row=8, column=0, columnspan=6, sticky="w", pady=(6, 0))
        for var in (self.min, self.mid, self.max, self.osc_format):
            var.trace_add("write", lambda *a: self._update_preview())
        btns = ttk.Frame(frm)
        btns.grid(row=9, column=0, columnspan=6, sticky="e", pady=(8, 0))
        ttk.Button(btns, text="OK", command=self.ok).pack(side="left", padx=4)
        ttk.Button(btns, text=t("cancel"), command=self.destroy).pack(side="left")
        self._apply_preset(initial=True)
        self.grab_set()
        self.wait_window()

    # -------------------------------------------------------------- terms
    def _add_term_row(self, cls, weight):
        if len(self.term_rows) >= self.MAX_TERMS:
            return
        row = ttk.Frame(self.terms_frame)
        row.pack(anchor="w", pady=1)
        cls_var, w_var = tk.StringVar(value=cls), tk.StringVar(value="%g" % weight)
        ttk.Combobox(row, textvariable=cls_var, values=self.class_names, state="readonly", width=16).pack(side="left")
        ttk.Label(row, text=" × ").pack(side="left")
        ttk.Entry(row, textvariable=w_var, width=6).pack(side="left")
        entry = (cls_var, w_var, row)
        ttk.Button(row, text="−", width=3, command=lambda: self._remove_term_row(entry)).pack(side="left", padx=4)
        self.term_rows.append(entry)
        for var in (cls_var, w_var):
            var.trace_add("write", lambda *a: self._update_preview())
        self._update_preview()

    def _remove_term_row(self, entry):
        if entry in self.term_rows:
            self.term_rows.remove(entry)
            entry[2].destroy()
            self._update_preview()

    # -------------------------------------------------------------- range
    def _preset_key(self):
        return next(k for k, v in self.preset_labels.items() if v == self.preset.get())

    def _apply_preset(self, initial=False):
        key = self._preset_key()
        custom = key == "custom"
        for e in (self.min_entry, self.max_entry):
            e.state(["!disabled"] if custom else ["disabled"])
        if not custom and not initial:
            lo, hi = RANGE_PRESETS[key]
            self.min.set("%g" % lo)
            self.max.set("%g" % hi)
            self.mid.set("%g" % ((lo + hi) / 2))
        self._update_preview()

    def _read(self):
        terms = [MergedTerm(c.get(), float(w.get())) for c, w, _ in self.term_rows if c.get()]
        try:
            bits = int(self.osc_bits.get())
        except (ValueError, tk.TclError):
            bits = 0
        lo, mid, hi = float(self.min.get()), float(self.mid.get()), float(self.max.get())
        return MergedParam(self.name.get().strip(), terms, lo, hi, None if mid == (lo + hi) / 2 else mid,
                           bool(self.enabled.get()), self.osc_format.get(), bits)

    def _update_preview(self):
        if not hasattr(self, "preview"):
            return
        t = self.app.t
        try:
            m = self._read()
        except ValueError:
            self.preview.configure(text="")
            return
        text = t("merged_preview2") % (m.formula(), m.out_min, m.neutral, m.out_max)
        if m.osc_format != "float":
            text += "\n" + t("binary_signed" if m.signed else "binary_unsigned")
        if m.beyond_sync_range:
            text += "\n" + t("merged_sync_warning")
        self.preview.configure(text=text, foreground="#c05000" if m.beyond_sync_range else "#666")

    def ok(self):
        try:
            m = self._read()
        except ValueError:
            messagebox.showerror(self.app.t("error"), self.app.t("merged_bad_number"), parent=self)
            return
        problems = merged_param_problems(m, set(self.class_names))
        others = [p for p in self.app.cfg.merged_params if p is not self.app._editing_merged]
        if any(p.name == m.name for p in others):
            problems.append(self.app.t("merged_duplicate") % m.name)
        if problems:
            messagebox.showerror(self.app.t("error"), "\n".join(problems), parent=self)
            return
        self.result = m
        self.destroy()


class GuideSound:
    """Beeps while a guided recording runs, so it can be followed with the headset on:
    silence = neutral face, the higher the pitch the stronger the expression."""

    def __init__(self):
        self.target = 0.0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        try:
            import winsound  # noqa: F401  (Windows only)
        except ImportError:
            return
        self.stop()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="guide-sound")
        self._thread.start()

    def stop(self):
        self._stop.set()
        self.target = 0.0

    def _run(self):
        import winsound
        while not self._stop.is_set():
            v = self.target
            if v > 0.02:
                winsound.Beep(int(300 + 700 * v), 70)  # blocks for the beep's length
                self._stop.wait(0.05)
            else:
                self._stop.wait(0.1)


class CompareDialog(tk.Toplevel):
    """Train several model / loss combinations on the same recordings, show how each
    does on the held-out frames and how fast it runs here, and apply the one picked."""

    COLUMNS = ("loss", "score", "acc", "hit", "false", "err", "ms", "size")

    def __init__(self, app):
        super().__init__(app.root)
        from .trainer import DEFAULT_CANDIDATES
        self.app = app
        t = app.t
        self.title(t("compare_title"))
        self.transient(app.root)
        self.results, self.best = [], None
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text=t("compare_help"), wraplength=640, justify="left").pack(anchor="w")

        pick = ttk.LabelFrame(frm, text=t("compare_pick"), padding=6)
        pick.pack(fill="x", pady=(8, 4))
        self.picks = []
        for i, (arch, loss) in enumerate(DEFAULT_CANDIDATES):
            var = tk.BooleanVar(value=True)
            self.picks.append((arch, loss, var))
            ttk.Checkbutton(pick, text="%s + %s" % (app.arch_labels[arch], app.loss_labels[loss]),
                            variable=var).grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 16))
        ttk.Label(frm, text=t("compare_epochs") % app.cfg.epochs, foreground="#666").pack(anchor="w")

        ctl = ttk.Frame(frm)
        ctl.pack(fill="x", pady=4)
        self.start_btn = ttk.Button(ctl, text=t("compare_start"), command=self.start)
        self.start_btn.pack(side="left")
        ttk.Button(ctl, text=t("stop"), command=app.engine.cancel_training).pack(side="left", padx=4)
        self.bar = ttk.Progressbar(frm, maximum=1.0)
        self.bar.pack(fill="x", pady=(4, 2))
        self.status = ttk.Label(frm, text="")
        self.status.pack(anchor="w")

        box = ttk.LabelFrame(frm, text=t("compare_results"), padding=6)
        box.pack(fill="both", expand=True, pady=(6, 0))
        self.tree = ttk.Treeview(box, columns=self.COLUMNS, show="tree headings", height=6, selectmode="browse")
        self.tree.heading("#0", text=t("arch"))
        self.tree.column("#0", width=190)
        for col in self.COLUMNS:
            self.tree.heading(col, text=t("cmp_" + col))
            self.tree.column(col, width=96 if col == "loss" else 72, anchor="w" if col == "loss" else "e")
        self.tree.tag_configure("best", background="#dff3e3")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda e: self.apply())
        ttk.Label(box, text=t("compare_legend"), foreground="#666", wraplength=640, justify="left").pack(
            anchor="w", pady=(4, 0))
        self.verdict = ttk.Label(frm, text="", wraplength=640, justify="left")
        self.verdict.pack(anchor="w", pady=(6, 0))

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(8, 0))
        self.apply_btn = ttk.Button(btns, text=t("compare_apply"), command=self.apply)
        self.apply_btn.pack(side="left")
        self.apply_btn.state(["disabled"])
        ttk.Button(btns, text=t("close"), command=self.close).pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self.close)

    def close(self):
        if self.app.engine.busy != "comparing":
            self.app.engine.compare_results = None  # the trained candidates hold a few hundred MB
        self.results = []
        self.destroy()

    def start(self):
        candidates = [(a, l) for a, l, var in self.picks if var.get()]
        app = self.app
        try:
            app.engine.compare_async(
                candidates,
                on_progress=lambda **info: app.events.put(("compare_progress", info)),
                on_done=lambda ok, m, r, b: app.events.put(("compare_done", ok, m, r, b)))
        except Exception as e:
            app._error(e)
            return
        self.start_btn.state(["disabled"])
        self.apply_btn.state(["disabled"])
        self.tree.delete(*self.tree.get_children())
        self.verdict.configure(text="")
        self.bar["value"] = 0
        self.status.configure(text=app.t("compare_running"))

    def progress(self, info):
        k, n = info["candidate"], info["candidates"]
        epoch = min(info["epoch"], info["epochs"] - 1)
        done = (epoch * info["steps"] + info["step"]) / max(1, info["epochs"] * info["steps"])
        if info.get("epoch_done"):
            done = info["epoch"] / info["epochs"]
        self.bar["value"] = (k + done) / n
        self.status.configure(text=self.app.t("compare_progress") % (k + 1, n, min(info["epoch"] + 1, info["epochs"]),
                                                                      info["epochs"]))

    def done(self, ok, message, results, best):
        self.start_btn.state(["!disabled"])
        if message:
            self.status.configure(text=message)
        if not ok:
            return
        self.bar["value"] = 1.0
        self.results, self.best = results, best
        t = self.app.t
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(results):
            m = r["metrics"]
            current = r["arch"] == self.app.cfg.model_arch and r["loss"] == self.app.cfg.loss
            name = ("★ " if i == best else "") + self.app.arch_labels[r["arch"]]
            self.tree.insert("", "end", iid=str(i), text=name, tags=("best",) if i == best else (), values=(
                r["loss"].upper(), "%.1f" % m["val_score"], "%.1f%%" % (m["val_acc"] * 100),
                "%.1f%%" % (m["val_hit"] * 100), "%.1f%%" % (m["val_false"] * 100), "%.2f" % m.get("val_err", 0),
                "%.2f" % r["ms"],
                "%.1f" % r["size_mb"]))
            if current and i != best:
                self.tree.item(str(i), text=name + " " + t("compare_current"))
        if best is not None:
            self.tree.selection_set(str(best))
            self.tree.see(str(best))
            r = results[best]
            self.verdict.configure(text=t("compare_verdict") % (
                self.app.arch_labels[r["arch"]], r["loss"].upper(), r["metrics"]["val_score"], r["ms"]))
            self.apply_btn.state(["!disabled"])

    def apply(self):
        sel = self.tree.selection()
        if not sel or not self.results:
            return
        r = self.results[int(sel[0])]
        app = self.app
        try:
            app.engine.apply_candidate(r)
        except Exception as e:
            app._error(e)
            return
        app.show_model_choice()
        app._rebuild_bars()
        app.train_status.configure(text=app.t("compare_applied") % (app.arch_labels[r["arch"]], r["loss"].upper()))
        self.verdict.configure(text=app.t("compare_applied") % (app.arch_labels[r["arch"]], r["loss"].upper()))
        self.done(True, "", self.results, self.best)


class App:
    def __init__(self, engine):
        self.engine = engine
        self.cfg = engine.cfg
        self.t = Translator(detect_language(self.cfg.language))
        self.events = queue.Queue()
        logging.getLogger().addHandler(QueueLogHandler(self.events))

        self.root = tk.Tk()
        self.root.title("%s" % self.t("title"))
        self.root.geometry("1000x720")
        self.root.minsize(860, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        style = ttk.Style(self.root)
        if "clam" in style.theme_names() and os.name != "nt":
            style.theme_use("clam")

        self._photo = None
        self.loss_history = []
        self.last_val = None  # (validation loss, validation accuracy) of the last epoch
        self.compare_dialog = None
        self.bars = {}

        self._build_status_bar()
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=6, pady=6)
        self._build_live_tab()
        self._build_record_tab()
        self._build_train_tab()
        self._build_merged_tab()
        self._build_settings_tab()
        self._build_log_tab()

        self.refresh_classes()
        self.refresh_recordings()
        self.root.after(50, self._pump_events)
        self.root.after(200, self._refresh_status)
        self.root.after(66, self._refresh_preview)

    # ================================================================ layout
    def _build_status_bar(self):
        t = self.t
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(fill="x")
        self.status_labels = {}
        for key in ("eye", "face", "vrcft"):
            f = ttk.Frame(bar)
            f.pack(side="left", padx=(0, 18))
            dot = tk.Canvas(f, width=12, height=12, highlightthickness=0)
            dot.create_oval(1, 1, 11, 11, fill=STATE_COLORS["disconnected"], outline="", tags="dot")
            dot.pack(side="left", padx=(0, 4))
            lbl = ttk.Label(f, text=t(key))
            lbl.pack(side="left")
            self.status_labels[key] = (dot, lbl)
        self.device_label = ttk.Label(bar, text="")
        self.device_label.pack(side="right")

    def _build_live_tab(self):
        t = self.t
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text=t("tab_live"))

        left = ttk.LabelFrame(tab, text=t("camera"), padding=6)
        left.pack(side="left", fill="y")
        size = 200 * PREVIEW_SCALE
        self.preview = tk.Canvas(left, width=size, height=size, bg="black", highlightthickness=0)
        self.preview.pack()
        self.preview_image = self.preview.create_image(0, 0, anchor="nw")
        row = ttk.Frame(left)
        row.pack(fill="x", pady=6)
        ttk.Button(row, text=t("swap"), command=self.on_swap).pack(side="left")
        self.show_preview = tk.BooleanVar(value=self.cfg.show_preview)
        ttk.Checkbutton(row, text=t("preview"), variable=self.show_preview,
                        command=self.on_preview_toggle).pack(side="left", padx=8)

        self.mode_labels = {m: t("mode_" + m) for m in MODES}
        self.mode_var = tk.StringVar(value=self.mode_labels.get(self.cfg.input_mode, self.mode_labels["both"]))
        mode_box = self.mode_box = ttk.LabelFrame(self.nb, text=t("input_mode"), padding=6)
        mode_combo = ttk.Combobox(mode_box, textvariable=self.mode_var, state="readonly", width=26,
                                  values=[self.mode_labels[m] for m in MODES])
        mode_combo.pack(anchor="w")
        mode_combo.bind("<<ComboboxSelected>>", lambda e: self.on_mode(self._mode_from_label()))
        ttk.Label(mode_box, text=t("mode_retrain"), foreground="#666", wraplength=520).pack(anchor="w", pady=(4, 0))
        self.hint_frame = ttk.Frame(mode_box)
        self.hint_label = ttk.Label(self.hint_frame, text="", foreground="#c05000", wraplength=520)
        self.hint_label.pack(anchor="w")
        hint_btns = ttk.Frame(self.hint_frame)
        hint_btns.pack(anchor="w", pady=(2, 0))
        self.hint_buttons = {m: ttk.Button(hint_btns, text=t("use_" + m), command=lambda m=m: self.on_mode(m))
                             for m in MODES}
        self.shown_hint = None

        right = ttk.Frame(tab, padding=(10, 0))
        right.pack(side="left", fill="both", expand=True)
        ctl = ttk.Frame(right)
        ctl.pack(fill="x")
        self.infer_btn = ttk.Button(ctl, text=t("start_infer"), command=self.on_toggle_infer)
        self.infer_btn.pack(side="left")
        ttk.Button(ctl, text=t("fastcal"), command=self.on_fastcal).pack(side="left", padx=6)
        ttk.Label(ctl, text=t("smoothing")).pack(side="left", padx=(16, 4))
        self.smoothing = tk.DoubleVar(value=self.cfg.smoothing)
        ttk.Scale(ctl, from_=0.0, to=0.9, variable=self.smoothing, length=140,
                  command=self.on_smoothing).pack(side="left")
        soft_row = ttk.Frame(right)
        soft_row.pack(fill="x", pady=(4, 0))
        ttk.Label(soft_row, text=t("softness")).pack(side="left")
        self.softness = tk.DoubleVar(value=self.cfg.softness)
        ttk.Scale(soft_row, from_=1.0, to=4.0, variable=self.softness, length=140,
                  command=self.on_softness).pack(side="left", padx=4)
        self.softness_label = ttk.Label(soft_row, text="", width=5)
        self.softness_label.pack(side="left")
        ttk.Label(soft_row, text=t("softness_help"), foreground="#666").pack(side="left", padx=(6, 0))
        self._show_softness()
        self.infer_stats = ttk.Label(right, text=t("not_tracking"))
        self.infer_stats.pack(anchor="w", pady=(6, 0))
        self.fastcal_label = ttk.Label(right, text=t("fastcal_help"), foreground="#666")
        self.fastcal_label.pack(anchor="w", pady=(2, 6))
        mode_box.pack(in_=right, fill="x", pady=(0, 6))
        mode_box.lift(right)  # created before `right`, so raise it above its container

        self._build_sensitivity_panel(right)
        self.bars_frame = ttk.LabelFrame(right, text=t("outputs"), padding=6)
        self.bars_frame.pack(fill="both", expand=True)

    def _build_sensitivity_panel(self, parent):
        """Per-class sensitivity: the part of 0..1 the expression really reaches is stretched
        back to 0..1. Click a class in the list above to edit it."""
        t = self.t
        self.sens_class = None
        self._sens_save_job = None
        self._sens_auto = None
        box = self.sens_box = ttk.LabelFrame(parent, text=t("sens_title_none"), padding=6)
        box.pack(side="bottom", fill="x", pady=(6, 0))
        self.sens_lo = tk.DoubleVar(value=0.0)
        self.sens_hi = tk.DoubleVar(value=1.0)
        for row, (label, var) in enumerate(((t("sens_low"), self.sens_lo), (t("sens_high"), self.sens_hi))):
            ttk.Label(box, text=label, width=10).grid(row=row, column=0, sticky="w")
            ttk.Scale(box, from_=0.0, to=1.0, variable=var, length=300,
                      command=lambda _v, which=row: self.on_sensitivity(which)).grid(row=row, column=1, sticky="ew")
        self.sens_value = ttk.Label(box, text="", width=16)
        self.sens_value.grid(row=0, column=2, rowspan=2, padx=6)
        btns = ttk.Frame(box)
        btns.grid(row=0, column=3, rowspan=2, sticky="e")
        ttk.Button(btns, text=t("sens_auto"), command=self.on_sensitivity_auto).pack(fill="x")
        ttk.Button(btns, text=t("sens_reset"), command=self.on_sensitivity_reset).pack(fill="x", pady=(2, 0))
        self.sens_stab = tk.DoubleVar(value=0.0)
        ttk.Label(box, text=t("sens_stability"), width=10).grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Scale(box, from_=0.0, to=1.0, variable=self.sens_stab, length=300,
                  command=lambda _v: self.on_stability()).grid(row=2, column=1, sticky="ew", pady=(4, 0))
        self.sens_stab_value = ttk.Label(box, text="", width=16)
        self.sens_stab_value.grid(row=2, column=2, padx=6, pady=(4, 0))
        self.sens_hint = ttk.Label(box, text=t("sens_hint") + "\n" + t("stability_hint"), foreground="#666",
                                   wraplength=520, justify="left")
        self.sens_hint.grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 0))
        box.columnconfigure(1, weight=1)
        self._select_sensitivity(None)

    def _select_sensitivity(self, index):
        t = self.t
        self.sens_class = index
        for i, row in self.bars.items():
            row[4].configure(background="#cfe0ff" if i == index else self._label_bg)
        if index is None or index >= len(self.cfg.classes):
            self.sens_class = None
            self.sens_box.configure(text=t("sens_title_none"))
            return
        c = self.cfg.classes[index]
        self.sens_box.configure(text=t("sens_title") % c.name)
        self.sens_lo.set(c.in_min)
        self.sens_hi.set(c.in_max)
        self.sens_stab.set(c.stability)
        self._show_sensitivity_values()

    def _show_sensitivity_values(self):
        if self.sens_class is None:
            return
        c = self.cfg.classes[self.sens_class]
        self.sens_value.configure(text="%.2f .. %.2f" % (c.in_min, c.in_max))
        self.sens_stab_value.configure(text=self.t("off") if c.stability <= 0 else
                                       self.t("stability_value") % (c.stability * STABILITY_WINDOW * 1000))

    def on_stability(self):
        if self.sens_class is None:
            return
        c = self.cfg.classes[self.sens_class]
        c.stability = round(self.sens_stab.get(), 2)
        self._show_sensitivity_values()
        if self._sens_save_job:
            self.root.after_cancel(self._sens_save_job)
        self._sens_save_job = self.root.after(600, self.engine.save_config)

    def on_sensitivity(self, which):
        if self.sens_class is None:
            return
        c = self.cfg.classes[self.sens_class]
        lo, hi = round(self.sens_lo.get(), 2), round(self.sens_hi.get(), 2)
        if hi - lo < 0.05:  # keep a usable range; move the other handle
            if which == 0:
                hi = min(1.0, lo + 0.05)
                lo = hi - 0.05
            else:
                lo = max(0.0, hi - 0.05)
                hi = lo + 0.05
            lo, hi = round(lo, 2), round(hi, 2)
            self.sens_lo.set(lo)
            self.sens_hi.set(hi)
        c.in_min, c.in_max = lo, hi
        self._show_sensitivity_values()
        if self._sens_save_job:
            self.root.after_cancel(self._sens_save_job)
        self._sens_save_job = self.root.after(600, self.engine.save_config)

    def on_sensitivity_reset(self):
        if self.sens_class is not None:
            self.sens_lo.set(0.0)
            self.sens_hi.set(1.0)
            self.on_sensitivity(1)

    def on_sensitivity_auto(self):
        """Watch the selected class for a few seconds (neutral face, then the full expression)
        and use the observed spread (5th..95th percentile) as its range."""
        t = self.t
        if self.sens_class is None or not self.engine.inferring:
            self._error(t("sens_need_tracking"))
            return
        self._sens_auto = {"index": self.sens_class, "samples": [], "end": time.monotonic() + 5.0}
        self.engine.request_full_rate(5.5)  # every frame, even before VRCFaceTracking is connected
        self._sens_auto_tick()

    def _sens_auto_tick(self):
        auto = self._sens_auto
        if auto is None:
            return
        value = self.engine.last_class_raw.get(auto["index"])
        if value is not None:
            auto["samples"].append(value)
        left = auto["end"] - time.monotonic()
        if left > 0:
            self.sens_hint.configure(text=self.t("sens_auto_running") % left, foreground="#c05000")
            self.root.after(50, self._sens_auto_tick)
            return
        self._sens_auto = None
        self.sens_hint.configure(text=self.t("sens_hint"), foreground="#666")
        samples = np.asarray(auto["samples"])
        if len(samples) < 10:
            return
        lo, hi = float(np.percentile(samples, 5)), float(np.percentile(samples, 95))
        if hi - lo < 0.05:
            self._error(self.t("sens_auto_flat"))
            return
        if self.sens_class == auto["index"]:
            self.sens_lo.set(round(lo, 2))
            self.sens_hi.set(round(hi, 2))
            self.on_sensitivity(1)

    def _rebuild_bars(self):
        for w in self.bars_frame.winfo_children():
            w.destroy()
        self.bars = {}
        for i, c in enumerate(self.cfg.classes):
            name = tk.Label(self.bars_frame, text=c.name, width=14, anchor="w", cursor="hand2")
            name.grid(row=i, column=0, sticky="w")
            self._label_bg = name.cget("background")
            cv = tk.Canvas(self.bars_frame, height=16, width=240, bg="#e6e6e6", highlightthickness=0, cursor="hand2")
            cv.grid(row=i, column=1, sticky="ew", pady=2)
            raw = cv.create_rectangle(0, 0, 0, 10, fill="#8aa4c8", outline="")
            sent = cv.create_rectangle(0, 10, 0, 16, fill="#2e6fd1", outline="")
            lo = cv.create_line(0, 0, 0, 16, fill="#e07000", width=2)
            hi = cv.create_line(0, 0, 0, 16, fill="#e07000", width=2)
            txt = ttk.Label(self.bars_frame, text="", width=13)
            txt.grid(row=i, column=2, sticky="w", padx=6)
            outputs = [c.target] if c.target else []
            if c.osc_name:
                outputs.append("OSC " + c.osc_name)
            ttk.Label(self.bars_frame, text=", ".join(outputs), foreground="#666").grid(row=i, column=3, sticky="w")
            for widget in (name, cv):
                widget.bind("<Button-1>", lambda e, idx=i: self._select_sensitivity(idx))
            self.bars[i] = (cv, raw, sent, txt, name, lo, hi)
        self.bars_frame.columnconfigure(1, weight=1)
        if hasattr(self, "sens_box"):
            self._select_sensitivity(self.sens_class if self.sens_class is not None
                                     and self.sens_class < len(self.cfg.classes) else None)

    def _build_record_tab(self):
        t = self.t
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text=t("tab_record"))
        form = ttk.Frame(tab)
        form.pack(fill="x")
        ttk.Label(form, text=t("rec_name")).grid(row=0, column=0, sticky="w")
        self.rec_name = tk.StringVar()
        ttk.Entry(form, textvariable=self.rec_name, width=24).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(form, text=t("rec_frames")).grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.rec_frames = tk.IntVar(value=self.cfg.record_frames)
        ttk.Spinbox(form, from_=256, to=16384, increment=256, textvariable=self.rec_frames, width=8).grid(
            row=0, column=3, padx=6)
        ttk.Label(form, text=t("rec_countdown")).grid(row=0, column=4, sticky="w", padx=(12, 0))
        self.rec_countdown = tk.DoubleVar(value=self.cfg.record_countdown)
        ttk.Spinbox(form, from_=0, to=30, increment=1, textvariable=self.rec_countdown, width=5).grid(
            row=0, column=5, padx=6)
        self.rec_btn = ttk.Button(form, text=t("record"), command=self.on_record)
        self.rec_btn.grid(row=0, column=6, padx=(12, 4))
        self.guided_btn = ttk.Button(form, text=t("record_guided"), command=lambda: self.on_record(guided=True))
        self.guided_btn.grid(row=0, column=7, padx=(0, 4))
        ttk.Button(form, text=t("cancel"), command=self.engine.cancel_record).grid(row=0, column=8)
        ttk.Label(tab, text=t("rec_help"), foreground="#666", wraplength=900).pack(anchor="w", pady=6)
        ttk.Label(tab, text=t("guided_help"), foreground="#666", wraplength=900, justify="left").pack(
            anchor="w", pady=(0, 6))
        guide = ttk.Frame(tab)
        guide.pack(fill="x", pady=(0, 6))
        self.guide_canvas = tk.Canvas(guide, height=46, bg="white", highlightthickness=1,
                                      highlightbackground="#ccc")
        self.guide_canvas.pack(side="left", fill="x", expand=True)
        self.guide_cue = ttk.Label(guide, text="", width=26, font=("TkDefaultFont", 14, "bold"))
        self.guide_cue.pack(side="left", padx=8)
        self.guide_sound = GuideSound()
        self.rec_progress = ttk.Progressbar(tab, maximum=1.0)
        self.rec_progress.pack(fill="x")
        self.rec_status = ttk.Label(tab, text="")
        self.rec_status.pack(anchor="w", pady=(2, 8))

        box = ttk.LabelFrame(tab, text=t("recordings"), padding=6)
        box.pack(fill="both", expand=True)
        cols = ("frames", "size", "guided")
        self.rec_tree = ttk.Treeview(box, columns=cols, show="tree headings", selectmode="extended")
        self.rec_tree.heading("#0", text=t("col_name"))
        self.rec_tree.heading("frames", text=t("rec_frames"))
        self.rec_tree.heading("size", text="MB")
        self.rec_tree.heading("guided", text=t("col_guided"))
        self.rec_tree.column("guided", width=70, anchor="center")
        self.rec_tree.column("frames", width=90, anchor="e")
        self.rec_tree.column("size", width=90, anchor="e")
        self.rec_tree.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(box)
        side.pack(side="left", fill="y", padx=6)
        for key, cmd in (("refresh", self.refresh_recordings), ("add_to_class", self.on_add_to_class),
                         ("delete", self.on_delete_recording), ("convert_pkl", self.on_convert)):
            ttk.Button(side, text=t(key), command=cmd).pack(fill="x", pady=2)

    def _build_train_tab(self):
        t = self.t
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text=t("tab_train"))
        box = ttk.LabelFrame(tab, text=t("classes"), padding=6)
        box.pack(fill="both", expand=True)
        cols = ("target", "osc", "power", "files")
        self.class_tree = ttk.Treeview(box, columns=cols, show="tree headings", height=8, selectmode="browse")
        self.class_tree.heading("#0", text=t("col_name"))
        self.class_tree.heading("target", text=t("col_target"))
        self.class_tree.heading("osc", text=t("col_osc"))
        self.class_tree.heading("power", text=t("col_power"))
        self.class_tree.heading("files", text=t("col_files"))
        self.class_tree.column("#0", width=120)
        self.class_tree.column("target", width=140)
        self.class_tree.column("osc", width=140)
        self.class_tree.column("power", width=80, anchor="e")
        self.class_tree.column("files", width=300)
        self.class_tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(box, command=self.class_tree.yview)
        self.class_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="left", fill="y")
        self.class_tree.bind("<Double-1>", lambda e: self.on_edit_class())
        side = ttk.Frame(box)
        side.pack(side="left", fill="y", padx=6)
        for key, cmd in (("add", self.on_add_class), ("edit", self.on_edit_class),
                         ("remove", self.on_remove_class), ("up", lambda: self.on_move_class(-1)),
                         ("down", lambda: self.on_move_class(1))):
            ttk.Button(side, text=t(key), command=cmd).pack(fill="x", pady=2)

        opts = ttk.Frame(tab)
        opts.pack(fill="x", pady=6)
        self.epochs = tk.IntVar(value=self.cfg.epochs)
        self.batch = tk.IntVar(value=self.cfg.batch_size)
        self.lr = tk.StringVar(value=str(self.cfg.learning_rate))
        self.amp = tk.BooleanVar(value=self.cfg.mixed_precision)
        self.cache = tk.BooleanVar(value=self.cfg.cache_datasets_in_ram)
        self.resume = tk.BooleanVar(value=False)
        ttk.Label(opts, text=t("epochs")).pack(side="left")
        ttk.Spinbox(opts, from_=1, to=500, textvariable=self.epochs, width=5).pack(side="left", padx=(4, 12))
        ttk.Label(opts, text=t("batch")).pack(side="left")
        ttk.Spinbox(opts, from_=8, to=1024, increment=8, textvariable=self.batch, width=6).pack(
            side="left", padx=(4, 12))
        ttk.Label(opts, text=t("lr")).pack(side="left")
        ttk.Entry(opts, textvariable=self.lr, width=9).pack(side="left", padx=(4, 12))
        self.val_sessions = tk.BooleanVar(value=self.cfg.validation_sessions)
        ttk.Checkbutton(opts, text=t("val_sessions"), variable=self.val_sessions).pack(side="left", padx=(12, 0))
        opts2 = ttk.Frame(tab)
        opts2.pack(fill="x", pady=(0, 6))
        ttk.Checkbutton(opts2, text=t("amp"), variable=self.amp).pack(side="left")
        ttk.Checkbutton(opts2, text=t("cache"), variable=self.cache).pack(side="left", padx=12)
        ttk.Checkbutton(opts2, text=t("resume"), variable=self.resume).pack(side="left")
        self.mixup = tk.BooleanVar(value=self.cfg.mixup)
        ttk.Checkbutton(opts2, text=t("mixup"), variable=self.mixup).pack(side="left", padx=12)
        opts3 = ttk.Frame(tab)
        opts3.pack(fill="x", pady=(0, 6))
        self.arch_labels = {a: t("arch_" + a) for a in MODEL_ARCHS}
        self.arch_var = tk.StringVar(value=self.arch_labels.get(self.cfg.model_arch, self.arch_labels["standard"]))
        ttk.Label(opts3, text=t("arch")).pack(side="left", padx=(0, 4))
        ttk.Combobox(opts3, textvariable=self.arch_var, state="readonly", width=26,
                     values=list(self.arch_labels.values())).pack(side="left")
        self.loss_labels = {k: t("loss_" + k) for k in LOSSES}
        self.loss_var = tk.StringVar(value=self.loss_labels.get(self.cfg.loss, self.loss_labels["mse"]))
        ttk.Label(opts3, text=t("loss_fn")).pack(side="left", padx=(16, 4))
        ttk.Combobox(opts3, textvariable=self.loss_var, state="readonly", width=30,
                     values=list(self.loss_labels.values())).pack(side="left")

        ctl = ttk.Frame(tab)
        ctl.pack(fill="x")
        self.train_btn = ttk.Button(ctl, text=t("train"), command=self.on_train)
        self.train_btn.pack(side="left")
        ttk.Button(ctl, text=t("stop"), command=self.engine.cancel_training).pack(side="left", padx=4)
        ttk.Button(ctl, text=t("save_model"), command=self.on_save_model).pack(side="left", padx=(16, 4))
        ttk.Button(ctl, text=t("load_model"), command=self.on_load_model).pack(side="left")
        self.compare_btn = ttk.Button(ctl, text=t("compare"), command=self.on_compare)
        self.compare_btn.pack(side="left", padx=(16, 0))
        self.train_progress = ttk.Progressbar(tab, maximum=1.0)
        self.train_progress.pack(fill="x", pady=(8, 2))
        self.train_status = ttk.Label(tab, text="")
        self.train_status.pack(anchor="w")
        chart = ttk.LabelFrame(tab, text=t("loss"), padding=4)
        chart.pack(fill="both", expand=True, pady=(6, 0))
        self.loss_canvas = tk.Canvas(chart, height=160, bg="white", highlightthickness=0)
        self.loss_canvas.pack(fill="both", expand=True)
        self.loss_canvas.bind("<Configure>", lambda e: self._draw_loss())

    def _build_merged_tab(self):
        t = self.t
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text=t("tab_merged"))
        ttk.Label(tab, text=t("merged_help"), foreground="#666", wraplength=940, justify="left").pack(anchor="w")
        box = ttk.Frame(tab)
        box.pack(fill="both", expand=True, pady=6)
        cols = ("terms", "range", "format", "value")
        self.merged_tree = ttk.Treeview(box, columns=cols, show="tree headings", selectmode="browse")
        self.merged_tree.heading("#0", text=t("col_name"))
        for col, key, width in (("terms", "merged_terms", 270), ("range", "merged_range", 140),
                                ("format", "merged_format", 140), ("value", "merged_value", 100)):
            self.merged_tree.heading(col, text=t(key))
            self.merged_tree.column(col, width=width, anchor="w" if col != "value" else "e")
        self.merged_tree.pack(side="left", fill="both", expand=True)
        self.merged_tree.bind("<Double-1>", lambda e: self.on_edit_merged())
        side = ttk.Frame(box)
        side.pack(side="left", fill="y", padx=6)
        for key, cmd in (("add", self.on_add_merged), ("edit", self.on_edit_merged), ("remove", self.on_remove_merged)):
            ttk.Button(side, text=t(key), command=cmd).pack(fill="x", pady=2)

        osc = ttk.LabelFrame(tab, text="OSC", padding=6)
        osc.pack(fill="x")
        self.osc_enabled = tk.BooleanVar(value=self.cfg.osc_enabled)
        self.osc_host = tk.StringVar(value=self.cfg.osc_host)
        self.osc_port = tk.StringVar(value=str(self.cfg.osc_port))
        ttk.Checkbutton(osc, text=t("osc_enabled"), variable=self.osc_enabled).pack(side="left")
        ttk.Label(osc, text=t("osc_target")).pack(side="left", padx=(16, 4))
        ttk.Entry(osc, textvariable=self.osc_host, width=14).pack(side="left")
        ttk.Label(osc, text=":").pack(side="left")
        ttk.Entry(osc, textvariable=self.osc_port, width=6).pack(side="left")
        ttk.Button(osc, text=t("perf_apply"), command=self.on_apply_osc).pack(side="left", padx=8)
        self._editing_merged = None
        self.refresh_merged()

    def refresh_merged(self):
        t = self.t
        self.merged_tree.delete(*self.merged_tree.get_children())
        names = {c.name for c in self.cfg.classes}
        for i, m in enumerate(self.cfg.merged_params):
            problems = merged_param_problems(m, names)
            value = t("merged_off") if not m.enabled else ("! " + problems[0] if problems else "")
            fmt = t("fmt_" + m.osc_format) if m.osc_format in ("float", "binary", "both") else m.osc_format
            if m.osc_format != "float":
                fmt += " %d bit" % m.osc_bits
            self.merged_tree.insert("", "end", iid=str(i), text=m.name, values=(
                m.formula(), "%g .. %g .. %g" % (m.out_min, m.neutral, m.out_max), fmt, value))

    def _update_merged_values(self):
        values = self.engine.last_merged
        for i, m in enumerate(self.cfg.merged_params):
            if m.name in values and self.merged_tree.exists(str(i)):
                self.merged_tree.set(str(i), "value", "%.3f" % values[m.name])

    def _selected_merged(self):
        sel = self.merged_tree.selection()
        return int(sel[0]) if sel else None

    def on_add_merged(self):
        self._editing_merged = None
        d = MergedDialog(self)
        if d.result:
            self.cfg.merged_params.append(d.result)
            self.engine.save_config()
            self.refresh_merged()

    def on_edit_merged(self):
        i = self._selected_merged()
        if i is None:
            return
        self._editing_merged = self.cfg.merged_params[i]
        d = MergedDialog(self, self.cfg.merged_params[i])
        self._editing_merged = None
        if d.result:
            self.cfg.merged_params[i] = d.result
            self.engine.save_config()
            self.refresh_merged()

    def on_remove_merged(self):
        i = self._selected_merged()
        if i is not None:
            del self.cfg.merged_params[i]
            self.engine.save_config()
            self.refresh_merged()

    def on_apply_osc(self):
        try:
            port = int(self.osc_port.get())
            if not 0 < port < 65536:
                raise ValueError
        except ValueError:
            self._error("OSC port")
            return
        self.cfg.osc_enabled = bool(self.osc_enabled.get())
        self.engine.set_osc_target(self.osc_host.get().strip() or "127.0.0.1", port)
        self.engine.save_config()

    def _build_settings_tab(self):
        t = self.t
        tab = ttk.Frame(self.nb, padding=12)
        self.nb.add(tab, text=t("tab_settings"))
        c = self.cfg
        self.settings = {
            "dataset_folder": tk.StringVar(value=c.dataset_folder),
            "model_path": tk.StringVar(value=c.model_path),
            "bind_host": tk.StringVar(value=c.bind_host),
            "face_port": tk.IntVar(value=c.face_port),
            "eye_port": tk.IntVar(value=c.eye_port),
            "proxy_port": tk.IntVar(value=c.proxy_port),
            "vrcft_port": tk.IntVar(value=c.vrcft_port),
            "max_send_rate": tk.DoubleVar(value=c.max_send_rate),
        }
        self.source_var = tk.StringVar(value=c.source)
        self.lang_var = tk.StringVar(value=c.language)
        row = 0

        def line(label, widget_factory):
            nonlocal row
            ttk.Label(tab, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 10))
            widget_factory().grid(row=row, column=1, sticky="w", pady=3)
            row += 1

        def path_row(label, key, is_dir):
            def make():
                f = ttk.Frame(tab)
                ttk.Entry(f, textvariable=self.settings[key], width=34).pack(side="left")

                def browse():
                    if is_dir:
                        p = filedialog.askdirectory(initialdir=self.settings[key].get() or ".")
                    else:
                        p = filedialog.asksaveasfilename(defaultextension=".pt",
                                                         filetypes=[("PyTorch", "*.pt"), ("*", "*")])
                    if p:
                        self.settings[key].set(p)
                ttk.Button(f, text=t("browse"), command=browse).pack(side="left", padx=4)
                return f
            line(label, make)

        path_row(t("dataset_folder"), "dataset_folder", True)
        path_row(t("model_path"), "model_path", False)

        def source_widget():
            f = ttk.Frame(tab)
            ttk.Radiobutton(f, text=t("source_direct"), value="direct", variable=self.source_var).pack(anchor="w")
            ttk.Radiobutton(f, text=t("source_proxy"), value="proxy", variable=self.source_var).pack(anchor="w")
            return f
        line(t("source"), source_widget)
        line(t("bind_host"), lambda: ttk.Entry(tab, textvariable=self.settings["bind_host"], width=16))
        for key in ("face_port", "eye_port", "proxy_port", "vrcft_port"):
            line(t(key), lambda key=key: ttk.Entry(tab, textvariable=self.settings[key], width=8))
        line(t("max_rate"), lambda: ttk.Entry(tab, textvariable=self.settings["max_send_rate"], width=8))
        line(t("language"), lambda: ttk.Combobox(tab, textvariable=self.lang_var, values=("auto", "en", "ko"),
                                                 state="readonly", width=8))
        ttk.Button(tab, text=t("save"), command=self.on_save_settings).grid(row=row, column=1, sticky="w", pady=12)
        self._build_vrcft_box(tab).grid(row=row + 1, column=0, columnspan=2, sticky="we", pady=(4, 0))
        self._build_perf_box(tab).grid(row=0, column=2, rowspan=row + 1, sticky="nw", padx=(24, 0))

    def _build_vrcft_box(self, parent):
        t = self.t
        box = ttk.LabelFrame(parent, text=t("vrcft_module"), padding=8)
        self.wrap_var = tk.BooleanVar(value=self.cfg.vrcft_wrap_sranipal)
        self.max_mode_var = tk.BooleanVar(value=self.cfg.vrcft_override_mode == "max")
        self.vrcft_status = ttk.Label(box, text="", wraplength=520, justify="left")
        self.vrcft_status.pack(anchor="w")
        ttk.Checkbutton(box, text=t("wrap_sranipal"), variable=self.wrap_var).pack(anchor="w", pady=(4, 0))
        ttk.Checkbutton(box, text=t("max_mode"), variable=self.max_mode_var,
                        command=self.on_max_mode).pack(anchor="w")
        btns = ttk.Frame(box)
        btns.pack(anchor="w", pady=(6, 0))
        ttk.Button(btns, text=t("install"), command=self.on_vrcft_install).pack(side="left")
        ttk.Button(btns, text=t("uninstall"), command=self.on_vrcft_uninstall).pack(side="left", padx=6)
        self._refresh_vrcft_status()
        return box

    def _refresh_vrcft_status(self):
        from . import vrcft_install
        t = self.t
        try:
            st = vrcft_install.status(self.cfg)
        except Exception as e:
            self.vrcft_status.configure(text=str(e))
            return
        if st["installed"]:
            text = t("mod_installed") % (st["installed_version"] or "?")
            if st["available_version"] and st["installed_version"] != st["available_version"]:
                text += "  " + t("mod_update") % st["available_version"]
        else:
            text = t("mod_not_installed")
        if st["sranipal"]:
            s = st["sranipal"][0]
            text += "\n" + (t("sran_wrapped") if s["disabled"] else t("sran_found")) % s["name"]
        else:
            text += "\n" + t("sran_missing")
        if not st["vrcft_found"]:
            text += "\n" + t("vrcft_missing") % st["custom_libs"]
        self.vrcft_status.configure(text=text)

    def on_max_mode(self):
        self.cfg.vrcft_override_mode = "max" if self.max_mode_var.get() else "replace"
        self.engine.vrcft.max_mode = self.cfg.vrcft_override_mode == "max"
        self.engine.vrcft._table_sent = False  # resend the table with the new mode
        self.engine.save_config()

    def on_vrcft_install(self):
        from . import vrcft_install
        try:
            self.cfg.vrcft_wrap_sranipal = bool(self.wrap_var.get())
            self.engine.save_config()
            steps = vrcft_install.install(self.cfg)
            messagebox.showinfo(self.t("vrcft_module"), "\n".join(steps), parent=self.root)
        except Exception as e:
            self._error(e)
        self._refresh_vrcft_status()

    def on_vrcft_uninstall(self):
        from . import vrcft_install
        try:
            steps = vrcft_install.uninstall(self.cfg)
            messagebox.showinfo(self.t("vrcft_module"), "\n".join(steps), parent=self.root)
        except Exception as e:
            self._error(e)
        self._refresh_vrcft_status()

    def _build_perf_box(self, parent):
        t = self.t
        c = self.cfg
        box = ttk.LabelFrame(parent, text=t("perf"), padding=8)
        self.engine_labels = {e: t("eng_" + e) for e in ("auto", "onnx", "pytorch")}
        self.engine_var = tk.StringVar(value=self.engine_labels.get(c.infer_engine, self.engine_labels["auto"]))
        self.dev_labels = {d: t("dev_" + d) for d in ("auto", "cpu", "gpu")}
        self.dev_var = tk.StringVar(value=self.dev_labels.get(c.infer_device, self.dev_labels["auto"]))
        self.threads_var = tk.IntVar(value=c.infer_threads)
        self.int8_var = tk.BooleanVar(value=c.infer_int8)
        self.rate_var = tk.DoubleVar(value=c.max_infer_rate)
        row = 0
        for label, var, labels in (("infer_engine", self.engine_var, self.engine_labels),
                                   ("infer_device", self.dev_var, self.dev_labels)):
            ttk.Label(box, text=t(label)).grid(row=row, column=0, columnspan=2, sticky="w")
            ttk.Combobox(box, textvariable=var, state="readonly", width=30, values=list(labels.values())).grid(
                row=row + 1, column=0, columnspan=2, sticky="w", pady=(0, 4))
            row += 2
        from . import system
        self.prio_labels = {p: t("prio_" + p) for p in ("above_normal", "normal", "below_normal", "idle")}
        self.prio_var = tk.StringVar(value=self.prio_labels.get(c.process_priority, self.prio_labels["normal"]))
        ttk.Label(box, text=t("priority")).grid(row=row, column=0, columnspan=2, sticky="w")
        ttk.Combobox(box, textvariable=self.prio_var, state="readonly", width=30,
                     values=list(self.prio_labels.values())).grid(row=row + 1, column=0, columnspan=2, sticky="w",
                                                                  pady=(0, 2))
        self.ecores_var = tk.BooleanVar(value=c.cpu_affinity == "ecores")
        self.eco_var = tk.BooleanVar(value=c.efficiency_mode)
        cb = ttk.Checkbutton(box, text=t("ecores"), variable=self.ecores_var)
        cb.grid(row=row + 2, column=0, columnspan=2, sticky="w")
        if not system.is_hybrid_cpu():
            cb.state(["disabled"])
        cb = ttk.Checkbutton(box, text=t("eco_mode"), variable=self.eco_var)
        cb.grid(row=row + 3, column=0, columnspan=2, sticky="w", pady=(0, 4))
        if not system.IS_WINDOWS:
            cb.state(["disabled"])
        row += 4
        ttk.Label(box, text=t("infer_threads")).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(box, from_=1, to=max(1, os.cpu_count() or 1), textvariable=self.threads_var,
                    width=5).grid(row=row, column=1, sticky="w", padx=(6, 0))
        ttk.Checkbutton(box, text=t("infer_int8"), variable=self.int8_var).grid(
            row=row + 1, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(box, text=t("max_infer_rate")).grid(row=row + 2, column=0, sticky="w", pady=2)
        ttk.Spinbox(box, from_=0, to=240, increment=10, textvariable=self.rate_var, width=6).grid(
            row=row + 2, column=1, sticky="w", padx=(6, 0))
        btns = ttk.Frame(box)
        btns.grid(row=row + 3, column=0, columnspan=2, sticky="w", pady=(8, 4))
        ttk.Button(btns, text=t("perf_apply"), command=self.on_apply_perf).pack(side="left")
        self.bench_btn = ttk.Button(btns, text=t("benchmark"), command=self.on_benchmark)
        self.bench_btn.pack(side="left", padx=6)
        ttk.Label(box, text=t("bench_note"), foreground="#666", wraplength=330).grid(
            row=row + 4, column=0, columnspan=2, sticky="w")
        self.bench_result = ttk.Label(box, text="", font=("TkFixedFont", 9), justify="left")
        self.bench_result.grid(row=row + 5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        return box

    def _build_log_tab(self):
        tab = ttk.Frame(self.nb, padding=4)
        self.nb.add(tab, text=self.t("tab_log"))
        self.log_text = tk.Text(tab, height=20, state="disabled", wrap="word")
        sb = ttk.Scrollbar(tab, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")

    # ================================================================ refresh
    def _pump_events(self):
        try:
            while True:
                kind, *payload = self.events.get_nowait()
                handler = getattr(self, "_ev_" + kind, None)
                if handler:
                    handler(*payload)
        except queue.Empty:
            pass
        self.root.after(50, self._pump_events)

    def _ev_log(self, line):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        if int(self.log_text.index("end-1c").split(".")[0]) > 2000:
            self.log_text.delete("1.0", "500.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _ev_rec_progress(self, done, total, phase):
        self.rec_progress["value"] = done / max(1, total)
        self.rec_status.configure(text="%s  %d / %d" % (phase, done, total))

    def _ev_guide(self, value, cue, left):
        self.guide_sound.target = value if cue != "done" else 0.0
        cv = self.guide_canvas
        cv.delete("all")
        w, h = cv.winfo_width(), cv.winfo_height()
        cv.create_rectangle(0, 0, w * value, h, fill="#2e9d4f", width=0)
        for mark in (0.5, 1.0):
            cv.create_line(w * mark - 1, 0, w * mark - 1, h, fill="#999", dash=(3, 3))
        self.guide_cue.configure(text=self.t("cue_" + cue) if cue != "done" else "")

    def _ev_rec_done(self, ok, message, filename):
        self.guide_sound.stop()
        self.guide_canvas.delete("all")
        self.guide_cue.configure(text="")
        self.guided_btn.state(["!disabled"])
        self.rec_status.configure(text=message)
        self.rec_btn.state(["!disabled"])
        self.refresh_recordings()

    def _ev_train_progress(self, info):
        total = info["epochs"] * info["steps"]
        done = min(info["epoch"], info["epochs"] - 1) * info["steps"] + info["step"]
        if info.get("epoch_done"):
            done = info["epoch"] * info["steps"]
            if "val_acc" in info:
                self.last_val = (info["val_loss"], info["val_acc"])
        else:
            self.loss_history.append(info["loss"])
        self.train_progress["value"] = done / max(1, total)
        elapsed = info["elapsed"]
        eta = elapsed / max(done, 1) * (total - done)
        self.train_status.configure(text="epoch %d/%d   step %d/%d   loss %.6f   avg %.6f   %.0fs (ETA %.0fs)" % (
            min(info["epoch"] + 1, info["epochs"]), info["epochs"], info["step"], info["steps"],
            info["loss"], info["avg"], elapsed, eta) + (
            "   " + self.t("val_stats") % (self.last_val[1] * 100, self.last_val[0]) if self.last_val else ""))
        self._draw_loss()

    def _ev_train_done(self, ok, message, history):
        self.train_btn.state(["!disabled"])
        self.train_status.configure(text=message)
        if ok:
            self._rebuild_bars()

    def _refresh_status(self):
        if self._minimized():  # nothing to draw; check back once a second
            self.root.after(1000, self._refresh_status)
            return
        s = self.engine.status()
        t = self.t
        mode = s["input_mode"]
        for key in ("eye", "face"):
            snap = s["streams"].get(key, {"state": "disconnected", "fps": 0})
            state = snap["state"]
            if mode not in ("both", key) and state != "ok":
                state = "not_used"
            dot, lbl = self.status_labels[key]
            dot.itemconfigure("dot", fill=STATE_COLORS[state])
            lbl.configure(text="%s: %s%s" % (t(key), t(state),
                                             " (%.0f fps)" % snap["fps"] if state == "ok" else ""))
        hint = s.get("mode_hint")
        if hint != self.shown_hint:
            self.shown_hint = hint
            for b in self.hint_buttons.values():
                b.pack_forget()
            if hint:
                self.hint_label.configure(text=t("hint_" + hint))
                for m in (("both",) if hint == "both" else ("face", "eye")):
                    self.hint_buttons[m].pack(side="left", padx=(0, 6))
                self.hint_frame.pack(anchor="w", pady=(6, 0))
            else:
                self.hint_frame.pack_forget()
        v = s["vrcft"]
        dot, lbl = self.status_labels["vrcft"]
        state = "ok" if v.get("connected") else "disconnected"
        dot.itemconfigure("dot", fill=STATE_COLORS[state])
        proto = {1: t("proto_old"), 2: t("proto_v6")}.get(v.get("protocol"), "")
        lbl.configure(text="%s: %s%s" % (t("vrcft"), t(state), " (%s)" % proto if state == "ok" and proto else ""))
        model = "model ✓" if s["model_loaded"] else "model ✗"
        if s["model_dirty"]:
            model += "*"
        self.device_label.configure(text="%s: %s   %s" % (t("device"), s["device"] or "-", model))

        if s["inferring"] and not s["infer_backend"]:
            self.infer_btn.configure(text=t("stop_infer"))
            self.infer_stats.configure(text=t("preparing"))
        elif s["inferring"]:
            self.infer_btn.configure(text=t("stop_infer"))
            self.infer_stats.configure(text=t("infer_stats") % (s["infer_fps"], s["latency_ms"],
                                                                s["infer_backend"] or "-", s["process_cpu"])
                                       + ("  ·  " + t("idle_note") if s.get("idle") else ""))
        else:
            self.infer_btn.configure(text=t("start_infer"))
            self.infer_stats.configure(text=t("not_tracking"))
        self.fastcal_label.configure(text=s["fastcal"] or t("fastcal_help"),
                                     foreground="#c05000" if s["fastcal"] else "#666")
        if s["inferring"]:
            self._update_bars()
            self._update_merged_values()
        self.root.after(100 if s["inferring"] else 250, self._refresh_status)

    def _update_bars(self):
        raw = self.engine.last_raw
        if raw is None:
            return
        class_raw, final = self.engine.last_class_raw, self.engine.last_weights
        for i, (cv, r_rect, s_rect, txt, _name, lo, hi) in self.bars.items():
            if i >= len(raw) or i >= len(self.cfg.classes):
                continue
            w = max(cv.winfo_width(), 1)
            c = self.cfg.classes[i]
            before, after = class_raw.get(i, 0.0), final.get(i, 0.0)
            cv.coords(r_rect, 0, 0, w * before, 10)  # before the sensitivity range
            cv.coords(s_rect, 0, 10, w * after, 16)  # what is sent
            cv.coords(lo, w * c.in_min, 0, w * c.in_min, 16)
            cv.coords(hi, w * c.in_max - 1, 0, w * c.in_max - 1, 16)
            txt.configure(text="%.2f → %.2f" % (before, after))

    def _minimized(self):
        try:
            return self.root.state() == "iconic"
        except tk.TclError:
            return True

    def _refresh_preview(self):
        if self.show_preview.get() and self.nb.index("current") == 0 and not self._minimized():
            self.engine.hub.want_cameras()  # receivers only keep camera frames while shown
            cams = self.engine.hub.cameras
            mode = self.cfg.input_mode
            img = np.zeros((200, 200), dtype=np.uint8)
            if cams.get("eye") is not None and mode in ("both", "eye"):
                img[:100] = decode_camera(cams["eye"])
            if cams.get("face") is not None and mode in ("both", "face"):
                img[100:] = decode_camera(cams["face"], flipped=True)
            if self._photo is None:
                self._photo = tk.PhotoImage(width=200 * PREVIEW_SCALE, height=200 * PREVIEW_SCALE)
                self.preview.itemconfigure(self.preview_image, image=self._photo)
            zoom_into(self._photo, img, PREVIEW_SCALE)
        self.root.after(66, self._refresh_preview)

    def _draw_loss(self):
        cv = self.loss_canvas
        cv.delete("all")
        hist = self.loss_history
        w, h = cv.winfo_width(), cv.winfo_height()
        if len(hist) < 2 or w < 10:
            return
        vals = np.log10(np.maximum(np.asarray(hist[-2000:]), 1e-7))
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-6:
            hi = lo + 1
        pad = 30
        # 0.001 target line
        target = np.log10(1e-3)
        if lo <= target <= hi:
            y = h - 8 - (target - lo) / (hi - lo) * (h - 16)
            cv.create_line(pad, y, w, y, fill="#2e9d4f", dash=(4, 3))
            cv.create_text(2, y, text="1e-3", anchor="w", fill="#2e9d4f", font=("TkDefaultFont", 7))
        xs = np.linspace(pad, w - 4, len(vals))
        ys = h - 8 - (vals - lo) / (hi - lo) * (h - 16)
        cv.create_line(*np.column_stack((xs, ys)).ravel().tolist(), fill="#2e6fd1")
        cv.create_text(2, 8, text="%.0e" % 10 ** hi, anchor="w", font=("TkDefaultFont", 7))
        cv.create_text(2, h - 8, text="%.0e" % 10 ** lo, anchor="w", font=("TkDefaultFont", 7))

    def refresh_classes(self):
        self.class_tree.delete(*self.class_tree.get_children())
        for i, c in enumerate(self.cfg.classes):
            osc = ""
            if c.osc_name:
                osc = c.osc_name + ("" if c.osc_format == "float" else " (%s %d bit)" % (c.osc_format, c.osc_bits))
                if class_output_problems(c):
                    osc = "! " + osc
            self.class_tree.insert("", "end", iid=str(i), text=c.name,
                                   values=(c.target or "", osc, "%.4g" % c.max_power, ", ".join(c.files)))
        self._rebuild_bars()

    def refresh_recordings(self):
        self.rec_tree.delete(*self.rec_tree.get_children())
        for f in dataset_files(self.cfg):
            path = self.cfg.dataset_path(f)
            try:
                size = os.path.getsize(path)
                self.rec_tree.insert("", "end", iid=f, text=f, values=(
                    frame_count(path), "%.0f" % (size / 2**20), "✓" if os.path.exists(labels_path(path)) else ""))
            except OSError:
                pass

    # ================================================================ actions
    def _error(self, e):
        messagebox.showerror(self.t("error"), str(e), parent=self.root)

    def on_swap(self):
        self.engine.set_swapped(not self.cfg.swapped)

    def _mode_from_label(self):
        label = self.mode_var.get()
        return next(m for m, text in self.mode_labels.items() if text == label)

    def on_mode(self, mode):
        try:
            self.engine.set_input_mode(mode)
        except Exception as e:
            self._error(e)
        self.mode_var.set(self.mode_labels[self.cfg.input_mode])

    def on_preview_toggle(self):
        self.cfg.show_preview = self.show_preview.get()
        self.engine.save_config()

    def on_smoothing(self, _value=None):
        self.cfg.smoothing = round(float(self.smoothing.get()), 2)

    def on_softness(self, _value=None):
        self.cfg.softness = round(float(self.softness.get()), 1)
        self._show_softness()

    def _show_softness(self):
        v = self.cfg.softness
        self.softness_label.configure(text=self.t("off") if v <= 1.0 else "×%.1f" % v)

    def on_toggle_infer(self):
        try:
            if self.engine.inferring:
                self.engine.stop_inference()
            else:
                self.engine.start_inference()
        except Exception as e:
            self._error(e)

    def on_fastcal(self):
        try:
            self.engine.request_fastcal()
        except Exception as e:
            self._error(e)

    def on_record(self, guided=False):
        try:
            self.cfg.record_frames = int(self.rec_frames.get())
            self.cfg.record_countdown = float(self.rec_countdown.get())
            self.engine.record(
                self.rec_name.get(),
                on_progress=lambda d, n, p: self.events.put(("rec_progress", d, n, p)),
                on_done=lambda ok, m, f: self.events.put(("rec_done", ok, m, f)),
                guided=guided, on_guide=lambda v, cue, left: self.events.put(("guide", v, cue, left)))
            self.rec_btn.state(["disabled"])
            self.guided_btn.state(["disabled"])
            if guided:
                self.guide_sound.start()
            self.engine.save_config()
        except Exception as e:
            self._error(e)

    def _selected_class(self):
        sel = self.class_tree.selection()
        return int(sel[0]) if sel else None

    def on_add_to_class(self):
        files = list(self.rec_tree.selection())
        if not files or not self.cfg.classes:
            return
        names = [c.name for c in self.cfg.classes]
        win = tk.Toplevel(self.root)
        win.title(self.t("pick_class"))
        win.transient(self.root)
        var = tk.StringVar(value=names[0])
        ttk.Label(win, text=self.t("pick_class"), padding=8).pack()
        ttk.Combobox(win, values=names, textvariable=var, state="readonly").pack(padx=8)

        def ok():
            c = self.cfg.classes[names.index(var.get())]
            for f in files:
                if f not in c.files:
                    c.files.append(f)
            self.engine.save_config()
            self.refresh_classes()
            win.destroy()
        ttk.Button(win, text="OK", command=ok).pack(pady=8)
        win.grab_set()

    def on_delete_recording(self):
        for f in self.rec_tree.selection():
            if messagebox.askyesno(self.t("delete"), self.t("confirm_delete") % f, parent=self.root):
                try:
                    remove_recording(self.cfg.dataset_path(f))
                except OSError as e:
                    self._error(e)
        self.refresh_recordings()

    def on_convert(self):
        try:
            done = self.engine.convert_pickles()
            log.info("Converted %d legacy recordings", len(done))
        except Exception as e:
            self._error(e)
        self.refresh_recordings()

    def on_add_class(self):
        d = ClassDialog(self)
        if d.result:
            self.cfg.classes.append(d.result)
            self._classes_changed()

    def on_edit_class(self):
        i = self._selected_class()
        if i is None:
            return
        old_name = self.cfg.classes[i].name
        d = ClassDialog(self, self.cfg.classes[i])
        if d.result:
            self.cfg.classes[i] = d.result
            if d.result.name != old_name:  # keep merged parameters pointing at the renamed class
                for m in self.cfg.merged_params:
                    for term in m.terms:
                        if term.cls == old_name:
                            term.cls = d.result.name
            self._classes_changed()

    def on_remove_class(self):
        i = self._selected_class()
        if i is not None:
            del self.cfg.classes[i]
            self._classes_changed()

    def on_move_class(self, delta):
        i = self._selected_class()
        if i is None:
            return
        j = i + delta
        if 0 <= j < len(self.cfg.classes):
            cl = self.cfg.classes
            cl[i], cl[j] = cl[j], cl[i]
            self._classes_changed()
            self.class_tree.selection_set(str(j))

    def _classes_changed(self):
        # a model trained for a different class list can't be used any more
        m = self.engine.model
        if m is not None and m.num_outputs != self.cfg.num_classes:
            self.engine.stop_inference()
        self.engine.save_config()
        self.refresh_classes()
        self.refresh_merged()

    def _read_train_options(self):
        self.cfg.epochs = int(self.epochs.get())
        self.cfg.batch_size = int(self.batch.get())
        self.cfg.learning_rate = float(self.lr.get())
        self.cfg.mixed_precision = bool(self.amp.get())
        self.cfg.cache_datasets_in_ram = bool(self.cache.get())
        self.cfg.mixup = bool(self.mixup.get())
        self.cfg.validation_sessions = bool(self.val_sessions.get())
        self.cfg.model_arch = next(a for a, text in self.arch_labels.items() if text == self.arch_var.get())
        self.cfg.loss = next(k for k, text in self.loss_labels.items() if text == self.loss_var.get())

    def show_model_choice(self):
        """Put the current config's model/loss back into the training tab's boxes."""
        self.arch_var.set(self.arch_labels[self.cfg.model_arch])
        self.loss_var.set(self.loss_labels[self.cfg.loss])

    def on_compare(self):
        try:
            self._read_train_options()
            self.engine.save_config()
        except Exception as e:
            self._error(e)
            return
        if self.compare_dialog is not None and self.compare_dialog.winfo_exists():
            self.compare_dialog.lift()
            return
        self.compare_dialog = CompareDialog(self)

    def _ev_compare_progress(self, info):
        if self.compare_dialog is not None and self.compare_dialog.winfo_exists():
            self.compare_dialog.progress(info)

    def _ev_compare_done(self, ok, message, results, best):
        self.train_status.configure(text=message)
        if ok and (self.compare_dialog is None or not self.compare_dialog.winfo_exists()):
            self.compare_dialog = CompareDialog(self)  # closed meanwhile: show the results anyway
        if self.compare_dialog is not None and self.compare_dialog.winfo_exists():
            self.compare_dialog.done(ok, message, results, best)

    def on_train(self):
        try:
            self._read_train_options()
            self.last_val = None
            self.engine.save_config()
            self.loss_history = []
            self.engine.train_async(
                on_progress=lambda **info: self.events.put(("train_progress", info)),
                on_done=lambda ok, m, h: self.events.put(("train_done", ok, m, h)),
                resume=bool(self.resume.get()))
            self.train_btn.state(["disabled"])
        except Exception as e:
            self._error(e)

    def on_save_model(self):
        try:
            self.engine.save_model()
        except Exception as e:
            self._error(e)

    def on_load_model(self):
        path = filedialog.askopenfilename(initialfile=self.cfg.model_path,
                                          filetypes=[("PyTorch", "*.pt"), ("*", "*")])
        if not path:
            return
        try:
            self.engine.load_model(path)
            self._rebuild_bars()
        except Exception as e:
            self._error(e)

    def _read_perf(self):
        self.cfg.infer_engine = next(e for e, text in self.engine_labels.items() if text == self.engine_var.get())
        dev = next(d for d, text in self.dev_labels.items() if text == self.dev_var.get())
        threads = max(1, int(self.threads_var.get()))
        rate = max(0.0, float(self.rate_var.get()))
        return dev, threads, bool(self.int8_var.get()), rate

    def on_apply_perf(self):
        try:
            from . import system
            c = self.cfg
            c.infer_device, c.infer_threads, c.infer_int8, c.max_infer_rate = self._read_perf()
            c.process_priority = next(p for p, text in self.prio_labels.items() if text == self.prio_var.get())
            c.cpu_affinity = "ecores" if self.ecores_var.get() else "all"
            c.efficiency_mode = bool(self.eco_var.get())
            system.apply(c)
            self.engine.save_config()
            self.engine.restart_inference()
        except Exception as e:
            self._error(e)

    def on_benchmark(self):
        try:
            self.cfg.infer_threads = self._read_perf()[1]
            self.engine.run_benchmark(on_done=lambda r, err: self.events.put(("bench_done", r, err)))
            self.bench_btn.state(["disabled"])
            self.bench_result.configure(text=self.t("bench_running"))
        except Exception as e:
            self._error(e)

    def _ev_bench_done(self, results, error):
        self.bench_btn.state(["!disabled"])
        if error:
            self.bench_result.configure(text=error)
            return
        self.bench_result.configure(text="\n".join(
            self.t("bench_line") % (r["backend"], r["ms"], r["cpu_pct"]) for r in results))

    def on_save_settings(self):
        try:
            for key, var in self.settings.items():
                setattr(self.cfg, key, var.get())
            self.cfg.source = self.source_var.get()
            self.cfg.language = self.lang_var.get()
            self.engine.save_config()
            self.refresh_recordings()
            messagebox.showinfo(self.t("save"), self.t("saved_restart"), parent=self.root)
        except Exception as e:
            self._error(e)

    def on_close(self):
        if self.engine.model_dirty and not messagebox.askyesno(self.t("title"), self.t("unsaved_quit"),
                                                               parent=self.root):
            return
        self.engine.save_config()
        self.engine.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def run_gui(engine):
    App(engine).run()

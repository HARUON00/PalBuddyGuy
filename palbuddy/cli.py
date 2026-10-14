"""Entry point: GUI by default, `--cli` for the original text interface."""

import argparse
import logging
import os
import sys
import threading
import time

from . import __version__
from .config import DEFAULT_CONFIG_PATH, LOSSES, MODEL_ARCHS, Config

log = logging.getLogger("palbuddy")

HELP = """commands:
  status             connection / fps overview
  swap               swap eye and face streams (saved in config.json)
  mode [both|face|eye]  which trackers feed the network (show / change)
  record [name]      record a dataset
  train              train on the classes in config.json
  save / load        save / load the model
  infer              start sending tracking to VRCFT (enter 'stop' to end)
  fastcal            calibrate max power by puppeting the avatar
  restcal            hold a neutral face for 5 s: each class's low end goes just above it
  stop               stop inference / training / recording
  arch [standard|lite|compact]  model used for the next training
  loss [mse|bce]        training method for the next training
  compare               train every model candidate, show scores, apply the recommended one
  vrcft [status|install|uninstall]  VRCFaceTracking v6 module
  merged                             list merged parameters (sent to VRChat over OSC)
  merged add NAME TERMS [MIN NEUTRAL MAX [FORMAT BITS]]
                                     TERMS: classes with weights, e.g. "smile,-sad" or "0.2*wide,0.8*open,-squeeze"
                                     e.g. "merged add PBG_SmileSad smile,-sad -1 0 1 binary 4"
  merged remove NAME
  oscout CLASS OSCNAME [float|binary|both] [BITS]   send a class straight to VRChat (OSCNAME - = off)
  sens CLASS LOW HIGH                sensitivity: stretch LOW..HIGH of the class to 0..1
  perf [engine auto|onnx|pytorch] [device auto|cpu|gpu] [threads N] [int8 on|off] [rate HZ]
       [priority above_normal|normal|below_normal|idle] [affinity all|ecores] [eco on|off]
                     show / change inference performance settings
  bench              compare CPU fp32 / CPU int8 / GPU on this PC
  stats              frame rate over 5 seconds
  convertmmap        convert old .pkl recordings to .mmap
  quit"""


def run_cli(engine):
    print("Pal Buddy Guy %s - type 'help' for commands" % __version__)

    def progress_printer(prefix):
        def fn(done, total, phase):
            print("\r%s %s %d / %d      " % (prefix, phase, done, total), end="", flush=True)
        return fn

    def train_progress(**i):
        print("\repoch %d/%d step %d/%d  loss %.6f  avg %.6f     " % (
            min(i["epoch"] + 1, i["epochs"]), i["epochs"], i["step"], i["steps"], i["loss"], i["avg"]),
            end="", flush=True)

    def compare_progress(**i):
        print("\rmodel %d/%d  epoch %d/%d step %d/%d     " % (
            i["candidate"] + 1, i["candidates"], min(i["epoch"] + 1, i["epochs"]), i["epochs"], i["step"],
            i["steps"]), end="", flush=True)

    done_event = threading.Event()

    def finished(ok, message, *_):
        print("\n" + message)
        done_event.set()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            line = "quit"
        if not line:
            continue
        cmd, _, arg = line.partition(" ")
        try:
            if cmd == "help":
                print(HELP)
            elif cmd == "status":
                s = engine.status()
                for role, st in s["streams"].items():
                    print("  %-5s %-12s %5.1f fps  %s" % (role, st["state"], st["fps"], st.get("peer") or ""))
                print("  input mode %s%s" % (s["input_mode"], {
                    "single": "  (only one tracker is streaming: 'mode face' or 'mode eye')",
                    "both": "  (both trackers are streaming: 'mode both')"}.get(s["mode_hint"], "")))
                if s["inferring"]:
                    print("  tracking %.0f fps, %.1f ms/frame, %s, process CPU %.0f%%" % (
                        s["infer_fps"], s["latency_ms"], s["infer_backend"], s["process_cpu"]))
                print("  vrcft %s  device %s  inferring %s  model %s" % (
                    "connected" if s["vrcft"]["connected"] else "disconnected", s["device"] or "-", s["inferring"],
                    "loaded" if s["model_loaded"] else "-"))
            elif cmd == "swap":
                engine.set_swapped(not engine.cfg.swapped)
                print("swapped = %s" % engine.cfg.swapped)
            elif cmd == "mode":
                if arg:
                    if arg not in ("both", "face", "eye"):
                        raise ValueError("mode must be both, face or eye")
                    engine.set_input_mode(arg)
                print("input mode = %s" % engine.cfg.input_mode)
                hint = engine.status()["mode_hint"]
                if hint == "single":
                    print("only one tracker is streaming: use 'mode face' or 'mode eye'")
                elif hint:
                    print("both trackers are streaming: consider 'mode both'")
            elif cmd == "perf":
                c = engine.cfg
                words = arg.split()
                for key, value in zip(words[::2], words[1::2]):
                    if key == "engine" and value in ("auto", "onnx", "pytorch"):
                        c.infer_engine = value
                    elif key == "device" and value in ("auto", "cpu", "gpu"):
                        c.infer_device = value
                    elif key == "threads":
                        c.infer_threads = max(1, int(value))
                    elif key == "int8":
                        c.infer_int8 = value.lower() in ("on", "1", "true", "yes")
                    elif key == "rate":
                        c.max_infer_rate = max(0.0, float(value))
                    elif key == "priority" and value in ("above_normal", "normal", "below_normal", "idle"):
                        c.process_priority = value
                    elif key == "affinity" and value in ("all", "ecores"):
                        c.cpu_affinity = value
                    elif key == "eco":
                        c.efficiency_mode = value.lower() in ("on", "1", "true", "yes")
                    else:
                        raise ValueError("unknown perf setting %s %s" % (key, value))
                if words:
                    from . import system
                    system.apply(c)
                    engine.save_config()
                    engine.restart_inference()
                print("priority %s  affinity %s  eco %s" % (
                    c.process_priority, c.cpu_affinity, "on" if c.efficiency_mode else "off"))
                print("engine %s  device %s  threads %d  int8 %s  rate %s  (running: %s)" % (
                    c.infer_engine, c.infer_device, c.infer_threads, "on" if c.infer_int8 else "off",
                    "%.0f Hz" % c.max_infer_rate if c.max_infer_rate else "every frame",
                    engine.infer_backend or "not tracking"))
            elif cmd == "vrcft":
                from . import vrcft_install
                action = arg or "status"
                if action == "install":
                    print("\n".join(vrcft_install.install(engine.cfg)))
                elif action == "uninstall":
                    print("\n".join(vrcft_install.uninstall(engine.cfg)))
                else:
                    st = vrcft_install.status(engine.cfg)
                    print("CustomLibs: %s" % st["custom_libs"])
                    print("installed: %s (available %s)" % (st["installed_version"] or "no", st["available_version"]))
                    for m in st["sranipal"]:
                        print("SRanipal module: %s (%s)" % (m["name"], "wrapped" if m["disabled"] else "standalone"))
                    print("connected module protocol: %s" % engine.vrcft.protocol)
            elif cmd == "merged":
                from .config import MergedParam, merged_param_problems, parse_terms
                c = engine.cfg
                words = arg.split()
                if words[:1] == ["add"] and len(words) in (3, 6, 8):
                    lo, mid, hi = (float(words[3]), float(words[4]), float(words[5])) if len(words) >= 6 \
                        else (-1.0, None, 1.0)
                    fmt, bits = (words[6], int(words[7])) if len(words) == 8 else ("float", 4)
                    m = MergedParam(words[1], parse_terms(words[2]), lo, hi, mid, True, fmt, bits)
                    problems = merged_param_problems(m, {x.name for x in c.classes})
                    if any(p.name == m.name for p in c.merged_params):
                        problems.append("'%s' already exists" % m.name)
                    if problems:
                        raise ValueError("; ".join(problems))
                    c.merged_params.append(m)
                    engine.save_config()
                elif words[:1] == ["remove"] and len(words) == 2:
                    c.merged_params = [m for m in c.merged_params if m.name != words[1]]
                    engine.save_config()
                elif words:
                    raise ValueError("usage: merged | merged add NAME TERMS [MIN NEUTRAL MAX [FORMAT BITS]] | "
                                     "merged remove NAME")
                for m in c.merged_params:
                    print("  %-16s = %-32s  %g .. %g .. %g%s  now %s" % (
                        m.name, m.formula(), m.out_min, m.neutral, m.out_max, "" if m.enabled else " (off)",
                        "%.3f" % engine.last_merged[m.name] if m.name in engine.last_merged else "-"))
                print("  OSC %s -> %s:%d" % ("on" if c.osc_enabled else "off", c.osc_host, c.osc_port))
            elif cmd in ("oscout", "sens"):
                from .config import class_output_problems
                words = arg.split()
                cls = next((c for c in engine.cfg.classes if words and c.name == words[0]), None)
                if cls is None:
                    raise ValueError("unknown class; classes: %s" % ", ".join(c.name for c in engine.cfg.classes))
                if cmd == "sens":
                    lo, hi = float(words[1]), float(words[2])
                    if not 0 <= lo < hi <= 1:
                        raise ValueError("need 0 <= LOW < HIGH <= 1")
                    cls.in_min, cls.in_max = lo, hi
                else:
                    old = (cls.osc_name, cls.osc_format, cls.osc_bits)
                    cls.osc_name = None if words[1] == "-" else words[1]
                    cls.osc_format = words[2] if len(words) > 2 else "float"
                    cls.osc_bits = int(words[3]) if len(words) > 3 else 4
                    problems = class_output_problems(cls)
                    if problems:
                        cls.osc_name, cls.osc_format, cls.osc_bits = old
                        raise ValueError("; ".join(problems))
                engine.save_config()
                print("%s: sensitivity %.2f..%.2f, OSC %s" % (cls.name, cls.in_min, cls.in_max, cls.osc_name and
                      "%s (%s%s)" % (cls.osc_name, cls.osc_format, "" if cls.osc_format == "float"
                                     else " %d bit" % cls.osc_bits) or "-"))
            elif cmd == "arch":
                if arg:
                    if arg not in MODEL_ARCHS:
                        raise ValueError("arch must be one of %s" % ", ".join(MODEL_ARCHS))
                    engine.cfg.model_arch = arg
                    engine.save_config()
                print("model arch for training = %s" % engine.cfg.model_arch)
            elif cmd == "loss":
                if arg:
                    if arg not in LOSSES:
                        raise ValueError("loss must be one of %s" % ", ".join(LOSSES))
                    engine.cfg.loss = arg
                    engine.save_config()
                print("training loss = %s" % engine.cfg.loss)
            elif cmd == "compare":
                done_event.clear()
                outcome = {}
                engine.compare_async(on_progress=compare_progress,
                                     on_done=lambda ok, m, r, b: (outcome.update(ok=ok, m=m, r=r, b=b),
                                                                  done_event.set()))
                done_event.wait()
                print("\n" + outcome["m"])
                if outcome["ok"]:
                    for i, r in enumerate(outcome["r"]):
                        m = r["metrics"]
                        print("%s %d) %-8s %-3s %-6s score %5.1f  accuracy %5.1f%%  recognised %5.1f%%  false %5.1f%%  "
                              "%.2f ms  %.1f MB" % ("*" if i == outcome["b"] else " ", i + 1, r["arch"],
                                                    r["loss"].upper(),
                                                    ("aug" if r.get("augment") else "") + ("+kd" if r.get("distill") else ""),
                                                    m["val_score"], m["val_acc"] * 100,
                                                    m["val_hit"] * 100, m["val_false"] * 100, r["ms"], r["size_mb"]))
                    pick = input("apply which? [%d, Enter = recommended, n = none]: " % (outcome["b"] + 1)).strip()
                    if pick.lower() != "n":
                        r = outcome["r"][int(pick) - 1 if pick else outcome["b"]]
                        engine.apply_candidate(r)
                        print("now using %s + %s (saved to %s)" % (r["arch"], r["loss"], engine.cfg.model_path))
                    engine.compare_results = None
            elif cmd == "bench":
                done_event.clear()
                engine.run_benchmark(on_done=lambda r, err: (print(err) if err else None, done_event.set()))
                done_event.wait()
            elif cmd == "record":
                name = arg or input("dataset name: ")
                done_event.clear()
                engine.record(name, on_progress=progress_printer("recording"), on_done=finished)
                done_event.wait()
            elif cmd == "train":
                done_event.clear()
                engine.train_async(on_progress=train_progress, on_done=finished)
                done_event.wait()
            elif cmd == "save":
                engine.save_model()
            elif cmd == "load":
                engine.load_model(arg or None)
            elif cmd == "infer":
                engine.start_inference()
                print("tracking started, 'stop' to end")
            elif cmd == "fastcal":
                engine.request_fastcal()
            elif cmd == "restcal":
                done_event.clear()
                outcome = {}
                print("hold a neutral face for 5 s...")
                engine.calibrate_rest(on_done=lambda r, e: (outcome.update(r=r, e=e), done_event.set()))
                done_event.wait()
                if outcome["e"]:
                    print(outcome["e"])
                for r in outcome["r"]:
                    print("%-14s neutral %.2f -> low end %.2f%s" % (
                        r["name"], r["rest"], r["in_min"],
                        "   <- model mistakes neutral for this: record neutral again and retrain" if r["high"] else ""))
            elif cmd == "stop":
                engine.stop_inference()
                engine.cancel_training()
                engine.cancel_record()
            elif cmd == "stats":
                time.sleep(5)
                print("%.1f samples/s" % engine.hub.sample_rate.rate())
            elif cmd == "convertmmap":
                print("converted: %s" % engine.convert_pickles())
            elif cmd in ("quit", "exit"):
                if engine.model_dirty and input("model not saved, quit anyway? [y/N] ").lower() != "y":
                    continue
                break
            else:
                print("unknown command, type 'help'")
        except Exception as e:
            print("error: %s" % e)
    engine.stop()


def show_fatal(message):
    """Without a console (double-clicked launcher, pythonw) errors would be invisible."""
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("Pal Buddy Guy", message)
        root.destroy()
    except Exception:
        pass


def show_warning(message):
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showwarning("Pal Buddy Guy", message)
        root.destroy()
    except Exception:
        pass


def setup_logging(verbose, config_path):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s %(levelname).1s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    if sys.stderr is not None:  # None under pythonw
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        root.addHandler(h)
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(config_path)), "palbuddy.log")
        h = logging.FileHandler(path, mode="w", encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname).1s %(name)s: %(message)s"))
        root.addHandler(h)
    except OSError:
        pass


def main(argv=None):
    p = argparse.ArgumentParser(prog="palbuddy", description="Vive Pro Eye / facial tracker expression trainer")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="path to config.json")
    p.add_argument("--cli", action="store_true", help="text interface instead of the GUI")
    p.add_argument("--infer", action="store_true", help="start tracking immediately (headless use)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    args.config = os.path.abspath(args.config)
    # relative paths in the config (datasets/, buddyguy.pt) are relative to the config file,
    # no matter where the app was started from (double-click, shortcut, terminal)
    os.chdir(os.path.dirname(args.config))

    setup_logging(args.verbose, args.config)
    gui = not args.cli
    try:
        cfg = Config.load(args.config)
    except Exception as e:
        # keep the broken file for the user, start with defaults instead of refusing to start
        backup = "%s.broken-%s" % (args.config, time.strftime("%Y%m%d-%H%M%S"))
        os.replace(args.config, backup)
        log.error("Could not read %s (%s); moved it to %s and started with default settings", args.config, e, backup)
        cfg = Config.load(args.config)
        if gui:
            show_warning("config.json could not be read and was moved to\n%s\nDefault settings are used." % backup)
    for problem in cfg.validate():
        log.warning(problem)
    from . import system
    system.apply(cfg)

    # Idle OpenMP worker threads otherwise busy-spin for a while after every op,
    # burning CPU that VRChat could use. Must be set before torch is imported.
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    try:
        from .engine import Engine  # imports torch; keep --help fast
    except ImportError as e:
        msg = "PyTorch is not installed (%s).\nRun PalBuddyGuy.bat again, or delete the .venv folder to reinstall." % e
        log.error(msg)
        if gui:
            show_fatal(msg)
        return 1
    engine = Engine(cfg, args.config)
    try:
        engine.start()
    except OSError as e:
        msg = ("Could not open a network port (%s).\n"
               "Is Pal Buddy Guy or tvm_proxy.py already running?" % e)
        log.error(msg)
        if gui:
            show_fatal(msg)
        return 1
    if args.infer:
        try:
            engine.start_inference()
        except Exception as e:
            log.error("Can't start tracking: %s", e)

    if not gui:
        run_cli(engine)
        return 0
    try:
        from .gui import run_gui
    except ImportError as e:
        log.error("GUI unavailable (%s); falling back to --cli", e)
        if sys.stdin is None:
            return 1
        run_cli(engine)
        return 0
    try:
        run_gui(engine)
    except Exception as e:
        log.exception("GUI crashed")
        engine.stop()
        show_fatal("Unexpected error: %s\nSee palbuddy.log for details." % e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

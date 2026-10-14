"""Install / remove the Pal Buddy Guy module for VRCFaceTracking v6.

VRCFaceTracking v6 hands eye tracking to one module and expressions to one module,
whichever initialises first, so the SRanipal module and a separate Pal Buddy Guy
module would race. The Pal Buddy Guy module therefore runs the SRanipal module
inside itself and lays the trained shapes on top. Installing:

  1. copies vrcft-module/prebuilt/* to <CustomLibs>/<module id>/,
  2. finds the SRanipal module and disables it by renaming its module.json to
     module.json.palbuddyguy-disabled (VRCFaceTracking skips folders without a
     module.json; nothing is deleted),
  3. writes palbuddyguy.json (app host/port, path of the SRanipal module DLL).

Uninstalling removes the module folder and renames the SRanipal module.json back.
VRCFaceTracking must be restarted afterwards.
"""

import json
import logging
import os
import shutil
import sys

from .config import APP_DIR

log = logging.getLogger(__name__)

MODULE_ID = "01db7fc9-61b0-4799-887e-b49b3beb043d"
DISABLED_SUFFIX = ".palbuddyguy-disabled"
PREBUILT_DIR = os.path.join(APP_DIR, "vrcft-module", "prebuilt")
MODULE_FILES = ("PalBuddyGuy.VRCFT.dll", "module.json")


def default_custom_libs():
    """Where VRCFaceTracking keeps modules: Environment.SpecialFolder.ApplicationData/VRCFaceTracking/CustomLibs."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser(r"~\AppData\Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "VRCFaceTracking", "CustomLibs")


def custom_libs_dir(cfg):
    return cfg.vrcft_custom_libs or default_custom_libs()


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def find_sranipal_modules(custom_libs):
    """[{dir, dll, name, disabled}] for SRanipal modules (not ours) under CustomLibs."""
    found = []
    if not os.path.isdir(custom_libs):
        return found
    for entry in sorted(os.listdir(custom_libs)):
        folder = os.path.join(custom_libs, entry)
        if not os.path.isdir(folder) or entry.lower() == MODULE_ID:
            continue
        for filename, disabled in (("module.json", False), ("module.json" + DISABLED_SUFFIX, True)):
            meta = _read_json(os.path.join(folder, filename))
            if not meta:
                continue
            name, dll = str(meta.get("ModuleName", "")), str(meta.get("DllFileName", ""))
            if "sranipal" in name.lower() or "sranipal" in dll.lower():
                found.append({"dir": folder, "dll": os.path.join(folder, dll), "name": name, "disabled": disabled})
    return found


def status(cfg):
    libs = custom_libs_dir(cfg)
    target = os.path.join(libs, MODULE_ID)
    meta = _read_json(os.path.join(target, "module.json"))
    prebuilt = _read_json(os.path.join(PREBUILT_DIR, "module.json"))
    return {
        "custom_libs": libs,
        "vrcft_found": os.path.isdir(os.path.dirname(libs)),
        "installed": meta is not None,
        "installed_version": meta.get("Version") if meta else None,
        "available_version": prebuilt.get("Version") if prebuilt else None,
        "sranipal": find_sranipal_modules(libs),
    }


class VRCFTRunningError(RuntimeError):
    pass


def _locked(e):
    return VRCFTRunningError("A file is in use (%s). Close VRCFaceTracking, then try again." % e)


def install(cfg):
    """Returns a list of human-readable steps that were done."""
    try:
        return _install(cfg)
    except PermissionError as e:
        raise _locked(e) from e


def _install(cfg):
    missing = [f for f in MODULE_FILES if not os.path.exists(os.path.join(PREBUILT_DIR, f))]
    if missing:
        raise FileNotFoundError("Prebuilt module files missing in %s: %s" % (PREBUILT_DIR, ", ".join(missing)))
    libs = custom_libs_dir(cfg)
    target = os.path.join(libs, MODULE_ID)
    os.makedirs(target, exist_ok=True)
    steps = []
    for f in MODULE_FILES:
        shutil.copy2(os.path.join(PREBUILT_DIR, f), os.path.join(target, f))
    steps.append("Installed the Pal Buddy Guy module to %s" % target)

    inner = None
    if cfg.vrcft_wrap_sranipal:
        for m in find_sranipal_modules(libs):
            if not m["disabled"]:
                os.replace(os.path.join(m["dir"], "module.json"),
                           os.path.join(m["dir"], "module.json" + DISABLED_SUFFIX))
                steps.append("Disabled the standalone module '%s' (it now runs inside Pal Buddy Guy)" % m["name"])
            if inner is None and os.path.exists(m["dll"]):
                inner = m["dll"]
        if inner is None:
            steps.append("SRanipal module not found: install it in VRCFaceTracking first for eye/lip tracking, "
                         "then install this module again")
    host = cfg.bind_host if cfg.bind_host not in ("", "0.0.0.0") else "127.0.0.1"
    settings = {"host": host, "port": cfg.vrcft_port, "innerModule": inner}
    with open(os.path.join(target, "palbuddyguy.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    steps.append("Restart VRCFaceTracking to load it")
    for s in steps:
        log.info(s)
    return steps


def uninstall(cfg):
    try:
        return _uninstall(cfg)
    except PermissionError as e:
        raise _locked(e) from e


def _uninstall(cfg):
    libs = custom_libs_dir(cfg)
    target = os.path.join(libs, MODULE_ID)
    steps = []
    if os.path.isdir(target):
        shutil.rmtree(target)
        steps.append("Removed %s" % target)
    for m in find_sranipal_modules(libs):
        if m["disabled"]:
            enabled = os.path.join(m["dir"], "module.json")
            if not os.path.exists(enabled):
                os.replace(os.path.join(m["dir"], "module.json" + DISABLED_SUFFIX), enabled)
                steps.append("Re-enabled the module '%s'" % m["name"])
    steps.append("Restart VRCFaceTracking to apply")
    for s in steps:
        log.info(s)
    return steps

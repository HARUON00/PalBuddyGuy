"""Process priority / CPU placement, so tracking yields to VR and the game.

Windows: SetPriorityClass, SetProcessAffinityMask (E-cores found via
GetLogicalProcessorInformationEx on hybrid CPUs like the 12th-gen Intel
Core), and EcoQoS "efficiency mode" via SetProcessInformation.
Elsewhere: nice values; affinity/efficiency mode are Windows only.
"""

import ctypes
import logging
import os
import struct
import sys

log = logging.getLogger(__name__)

PRIORITIES = ("normal", "below_normal", "idle", "above_normal")

_WIN_CLASSES = {"idle": 0x40, "below_normal": 0x4000, "normal": 0x20, "above_normal": 0x8000}
_NICE = {"idle": 19, "below_normal": 10, "normal": 0, "above_normal": -5}
IS_WINDOWS = sys.platform == "win32"


def _kernel32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.GetPriorityClass.argtypes = [ctypes.c_void_p]
    k.GetPriorityClass.restype = ctypes.c_uint32
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                         ctypes.POINTER(ctypes.c_size_t)]
    k.GetLogicalProcessorInformationEx.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    k.GetLogicalProcessorInformationEx.restype = ctypes.c_bool
    return k


def set_priority(level):
    """Returns True if applied."""
    if level not in PRIORITIES:
        raise ValueError("priority must be one of %s" % ", ".join(PRIORITIES))
    try:
        if IS_WINDOWS:
            k = _kernel32()
            if not k.SetPriorityClass(k.GetCurrentProcess(), _WIN_CLASSES[level]):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            current = os.getpriority(os.PRIO_PROCESS, 0)
            target = _NICE[level]
            if target < current and os.geteuid() != 0:
                log.info("Raising priority needs root on this OS; keeping nice %d", current)
                return False
            os.setpriority(os.PRIO_PROCESS, 0, target)
        log.info("Process priority: %s", level)
        return True
    except (OSError, AttributeError) as e:
        log.warning("Could not set process priority to %s: %s", level, e)
        return False


def core_efficiency_classes():
    """[(efficiency_class, affinity_mask)] per physical core (processor group 0), or []
    if unavailable. Higher class = faster core; hybrid CPUs report P-cores > E-cores."""
    if not IS_WINDOWS:
        return []
    k = _kernel32()
    relation_processor_core = 0
    length = ctypes.c_uint32(0)
    k.GetLogicalProcessorInformationEx(relation_processor_core, None, ctypes.byref(length))
    buf = ctypes.create_string_buffer(length.value)
    if not k.GetLogicalProcessorInformationEx(relation_processor_core, buf, ctypes.byref(length)):
        return []
    return parse_core_info(buf.raw, length.value)


def parse_core_info(raw, length):
    """Parse a GetLogicalProcessorInformationEx(RelationProcessorCore) buffer into
    [(efficiency_class, affinity_mask)] for processor group 0."""
    relation_processor_core = 0
    cores, offset = [], 0
    while offset < length:
        relationship, size = struct.unpack_from("<II", raw, offset)
        if size == 0:
            break
        if relationship == relation_processor_core:
            # PROCESSOR_RELATIONSHIP (at +8): Flags u8, EfficiencyClass u8, Reserved[20], GroupCount u16,
            # GROUP_AFFINITY GroupMask[] {KAFFINITY Mask; WORD Group; WORD Reserved[3]} at +24
            efficiency = raw[offset + 8 + 1]
            group_count = struct.unpack_from("<H", raw, offset + 8 + 22)[0]
            base = offset + 8 + 24
            for g in range(group_count):
                mask, group = struct.unpack_from("<QH", raw, base + g * 16)
                if group == 0:
                    cores.append((efficiency, mask))
        offset += size
    return cores


def efficiency_core_mask():
    """Affinity mask of the E-cores, or 0 if the CPU isn't hybrid / unknown."""
    cores = core_efficiency_classes()
    classes = {c for c, _ in cores}
    if len(classes) < 2:
        return 0
    lowest = min(classes)
    mask = 0
    for c, m in cores:
        if c == lowest:
            mask |= m
    return mask


def is_hybrid_cpu():
    return efficiency_core_mask() != 0


def set_affinity(mode):
    """mode: "all" or "ecores". Returns True if applied."""
    try:
        if mode == "all":
            if IS_WINDOWS:
                k = _kernel32()
                proc_mask, sys_mask = ctypes.c_size_t(), ctypes.c_size_t()
                k.GetProcessAffinityMask(k.GetCurrentProcess(), ctypes.byref(proc_mask), ctypes.byref(sys_mask))
                k.SetProcessAffinityMask(k.GetCurrentProcess(), sys_mask.value)
            elif hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, range(os.cpu_count() or 1))
            return True
        if mode != "ecores":
            raise ValueError("affinity must be 'all' or 'ecores'")
        mask = efficiency_core_mask()
        if not mask:
            log.info("No efficiency cores detected; using all cores")
            return False
        k = _kernel32()
        if not k.SetProcessAffinityMask(k.GetCurrentProcess(), mask):
            raise ctypes.WinError(ctypes.get_last_error())
        log.info("Running on efficiency cores only (mask 0x%x)", mask)
        return True
    except (OSError, AttributeError, ValueError) as e:
        log.warning("Could not set CPU affinity (%s): %s", mode, e)
        return False


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_uint32), ("ControlMask", ctypes.c_uint32), ("StateMask", ctypes.c_uint32)]


def set_efficiency_mode(enabled):
    """Windows 11 EcoQoS: the scheduler prefers E-cores and low clocks for this
    process. Returns True if applied."""
    if not IS_WINDOWS:
        return False
    try:
        k = _kernel32()
        process_power_throttling = 4
        execution_speed = 0x1
        # off = hand control back to Windows (ControlMask 0), not "force high QoS"
        state = _PowerThrottlingState(1, execution_speed, execution_speed) if enabled else \
            _PowerThrottlingState(1, 0, 0)
        k.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        if not k.SetProcessInformation(k.GetCurrentProcess(), process_power_throttling, ctypes.byref(state),
                                       ctypes.sizeof(state)):
            raise ctypes.WinError(ctypes.get_last_error())
        log.info("Efficiency mode (EcoQoS): %s", "on" if enabled else "off")
        return True
    except (OSError, AttributeError) as e:
        log.warning("Could not change efficiency mode: %s", e)
        return False


def apply(cfg):
    """Apply the config's priority / affinity / efficiency settings."""
    set_priority(cfg.process_priority)
    set_affinity(cfg.cpu_affinity)
    if IS_WINDOWS:
        set_efficiency_mode(cfg.efficiency_mode)

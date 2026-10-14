"""Connection to the VRCFaceTracking PalBuddyGuy module (TCP, port 26421).

Two module generations are supported; which one connected is detected from
what it sends first.

Protocol 1 - the original module (VRCFaceTracking up to v4):
  us -> VRCFT:  0x02 | count | count * (sranipal_index:u8, value:u16 BE)
                value = (v + 1) * 32767, v in [-1, 1]
  VRCFT -> us:  0x01 + 37 * u16 (eye data), 0x02 + 60 * u16 (lip data)

Protocol 2 - vrcft-module/ (VRCFaceTracking v6), module sends "PBG2" + version first:
  us -> VRCFT:  0x05 | mode | count | count * (len:u8, unified_expression_name)   target table
                0x03 | count | count * (slot:u8, weight:u16 BE)  weight 0..65535 = 0..1
                0x03 | 0     tracking stopped: module falls back to plain SRanipal
"""

import logging
import socket
import threading
import time

from .net import Disconnected, close_quietly, make_listener, recv_exact, tune_stream

log = logging.getLogger(__name__)

from .params import TRANSFORMS, legacy_index, unified_targets

MSG_PARAMS = 2
MSG_TABLE = 5
MSG_WEIGHTS = 3
INCOMING_SIZES = {1: 37, 2: 60}
HELLO = b"PBG2"
PROTOCOL_DETECT_TIMEOUT = 1.0  # old modules may never speak first


def encode_value(v):
    v = int((float(v) + 1.0) * 32767.0)
    return min(65535, max(0, v))


def encode_table(names, max_mode=False):
    out = bytearray((MSG_TABLE, 1 if max_mode else 0, len(names)))
    for name in names:
        raw = name.encode("ascii")
        out += bytes((len(raw),)) + raw
    return bytes(out)


def encode_weights(slot_weights):
    """slot_weights: iterable of (slot, weight 0..1)."""
    slot_weights = list(slot_weights)
    out = bytearray((MSG_WEIGHTS, len(slot_weights)))
    for slot, w in slot_weights:
        v = min(65535, max(0, int(round(float(w) * 65535))))
        out += bytes((slot, v >> 8, v & 0xFF))
    return bytes(out)


def encode_params(pairs):
    """pairs: iterable of (shape_index, value in [-1, 1]) -> one packet."""
    pairs = list(pairs)
    out = bytearray((MSG_PARAMS, len(pairs)))
    for key, value in pairs:
        v = encode_value(value)
        out += bytes((key, v >> 8, v & 0xFF))
    return bytes(out)


class VRCFTServer:
    """Accepts the VRCFT module and sends parameter packets to it.

    Compared to the original:
      * each update is a single sendall() of one packet (was 1 + 3*N tiny sends,
        which with Nagle enabled added latency and could interleave between
        threads), TCP_NODELAY is on,
      * a send timeout means a frozen VRCFT can't stall the inference loop,
      * a reconnecting VRCFT (e.g. after restarting VRChat) replaces the old socket,
      * the reader detects EOF instead of spinning forever on recv() == b"".
    """

    def __init__(self, host="127.0.0.1", port=26421, send_timeout=0.25):
        self.host = host
        self.port = port
        self.send_timeout = send_timeout
        self._conn = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._listener = None
        self.peer = None
        self.connections = 0
        self.packets_sent = 0
        self.last_received = {}
        self.protocol = None  # 1 = original module, 2 = VRCFT v6 module, None = not known yet
        self.module_version = None
        self.max_mode = False  # protocol 2: max(SRanipal, ours) instead of replacing
        self._table = None  # protocol 2: target names in table order
        self._slots = {}  # target name -> table slots
        self._table_sent = False
        self._warned_unmapped = set()

    @property
    def connected(self):
        return self._conn is not None

    def start(self):
        self._listener = make_listener(self.host, self.port)
        self.port = self._listener.getsockname()[1]
        threading.Thread(target=self._accept_loop, daemon=True, name="vrcft-accept").start()
        log.info("Waiting for VRCFaceTracking on %s:%d", self.host, self.port)
        return self

    def stop(self):
        self._stop.set()
        close_quietly(self._listener)
        self._drop()

    def _drop(self, conn=None):
        with self._lock:
            if conn is None or self._conn is conn:
                close_quietly(self._conn)
                self._conn = None
                self.peer = None

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                conn, addr = self._listener.accept()
            except OSError:
                if self._stop.is_set():
                    return
                time.sleep(0.2)
                continue
            tune_stream(conn)
            conn.settimeout(self.send_timeout)
            with self._lock:
                old = self._conn
                self._conn = conn
                self.peer = "%s:%d" % addr
                self.connections += 1
                self.protocol = None
                self.module_version = None
                self._table_sent = False
            if old is not None:
                close_quietly(old)
            log.info("VRCFaceTracking connected from %s", self.peer)
            threading.Thread(target=self._reader, args=(conn,), daemon=True, name="vrcft-reader").start()

    def _reader(self, conn):
        connected_at = time.monotonic()
        try:
            while not self._stop.is_set():
                try:
                    rid = recv_exact(conn, 1)[0]
                except socket.timeout:
                    if self.protocol is None and time.monotonic() - connected_at > PROTOCOL_DETECT_TIMEOUT:
                        self._set_protocol(conn, 1)
                    continue  # just idle; the timeout exists for sends
                if rid == HELLO[0] and self.protocol is None:
                    rest = self._read_blocking(conn, len(HELLO))  # "BG2" + version
                    if bytes(rest[:3]) != HELLO[1:]:
                        log.warning("Unexpected greeting from VRCFT: %r", bytes(rest))
                        break
                    self.module_version = rest[3]
                    self._set_protocol(conn, 2)
                    continue
                if self.protocol is None:
                    self._set_protocol(conn, 1)
                count = INCOMING_SIZES.get(rid)
                if count is None:
                    log.warning("Invalid message id %d from VRCFT", rid)
                    break
                raw = self._read_blocking(conn, count * 2)
                self.last_received[rid] = [
                    ((raw[i] << 8) | raw[i + 1]) / 32767.0 - 1.0 for i in range(0, len(raw), 2)
                ]
        except (Disconnected, OSError) as e:
            if not self._stop.is_set():
                log.info("VRCFaceTracking disconnected: %s", e)
        finally:
            self._drop(conn)

    def _set_protocol(self, conn, protocol):
        with self._lock:
            if self._conn is not conn:
                return
            self.protocol = protocol
            self._table_sent = False
        log.info("VRCFaceTracking module speaks protocol %d%s", protocol,
                 " (VRCFT v6 module %d)" % self.module_version if protocol == 2 else " (original module)")

    @staticmethod
    def _read_blocking(conn, length):
        buf = bytearray()
        while len(buf) < length:
            try:
                chunk = conn.recv(length - len(buf))
            except socket.timeout:
                continue
            if not chunk:
                raise Disconnected("peer closed the connection")
            buf += chunk
        return buf

    def send_packet(self, packet):
        with self._lock:
            conn = self._conn
            if conn is None:
                return False
            try:
                conn.sendall(packet)
                self.packets_sent += 1
                return True
            except OSError as e:
                log.info("Send to VRCFT failed (%s); waiting for reconnect", e)
                close_quietly(conn)
                self._conn = None
                self.peer = None
                return False

    def send_params(self, pairs):
        """Protocol 1 only: pairs of (sranipal_index, value -1..1)."""
        return self.send_packet(encode_params(pairs))

    def _build_table(self, names):
        table = []
        for name in names:
            for unified in unified_targets(name):
                if unified not in table:
                    table.append(unified)
        if len(table) > 255:
            raise ValueError("too many targets")
        self._table = tuple(table)
        self._slots = {u: i for i, u in enumerate(table)}  # unified name -> slot
        self._known = set(names)
        # per target: (slots, scale, offset), so a frame is only a little arithmetic
        self._compiled = {n: (tuple(self._slots[u] for u in unified_targets(n)),) + TRANSFORMS.get(n, (1.0, 0.0))
                          for n in names}
        self._table_sent = False

    def send_targets(self, pairs):
        """pairs: (target name, value -1..1). Target names are SRanipal lip shapes or
        Unified Expressions; they're translated for whichever module is connected."""
        pairs = list(pairs)
        protocol = self.protocol
        if protocol is None or not self.connected:
            return False
        if protocol == 1:
            legacy = []
            for name, v in pairs:
                idx = legacy_index(name)
                if idx is None:
                    if name not in self._warned_unmapped:
                        self._warned_unmapped.add(name)
                        log.warning("Target %s needs the VRCFaceTracking v6 module; the old module can't show it", name)
                    continue
                legacy.append((idx, v))
            return self.send_packet(encode_params(legacy)) if legacy else False
        names = tuple(name for name, _ in pairs)
        if self._table is None or any(n not in self._known for n in names):
            self._build_table(names)
        packet = b""
        if not self._table_sent:
            packet += encode_table(self._table, self.max_mode)
        slot_weights = []
        compiled = self._compiled
        for name, v in pairs:
            slots, scale, offset = compiled[name]
            w = (v + 1.0) * 0.5  # -1..1 -> 0..1, then the target's transform (same as params.unified_weights)
            w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)
            uw = min(1.0, offset + scale * w) if w > 0.0 else 0.0
            for slot in slots:
                slot_weights.append((slot, uw))
        packet += encode_weights(slot_weights)
        ok = self.send_packet(packet)
        if ok:
            self._table_sent = True
        return ok

    def clear(self):
        """Tell a v6 module that tracking stopped, so it shows plain SRanipal again
        (it would also time out on its own after 0.5 s)."""
        if self.protocol == 2 and self.connected:
            self.send_packet(encode_weights([]))

    def snapshot(self):
        return {"connected": self.connected, "peer": self.peer, "connections": self.connections,
                "packets": self.packets_sent, "protocol": self.protocol}

"""Minimal OSC sender for VRChat avatar parameters (UDP, default 127.0.0.1:9000).

Used for merged parameters: values VRCFaceTracking can't carry, e.g. a custom
"SmileSad" parameter in -1..1. VRCFaceTracking sends its own parameters to the
same port; VRChat accepts several senders as long as the names differ.
"""

import logging
import socket
import struct

log = logging.getLogger(__name__)

AVATAR_PREFIX = "/avatar/parameters/"


def _pad(raw):
    """OSC strings are null-terminated and padded to a multiple of 4 bytes."""
    raw += b"\0"
    return raw + b"\0" * (-len(raw) % 4)


def encode_float(address, value):
    return _pad(address.encode("utf-8")) + _pad(b",f") + struct.pack(">f", float(value))


def encode_bool(address, value):
    """OSC booleans carry no argument data: type tag T or F."""
    return _pad(address.encode("utf-8")) + _pad(b",T" if value else b",F")


class OscSender:
    def __init__(self, host="127.0.0.1", port=9000):
        self.address = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._warned = False
        self.messages_sent = 0

    def send_float(self, address, value):
        return self._send(encode_float(address, value))

    def send_bool(self, address, value):
        return self._send(encode_bool(address, value))

    def _send(self, packet):
        try:
            self._sock.sendto(packet, self.address)
            self.messages_sent += 1
            return True
        except OSError as e:  # UDP: only fails for local reasons (bad host, no route)
            if not self._warned:
                log.warning("OSC send to %s:%d failed: %s", self.address[0], self.address[1], e)
                self._warned = True
            return False

    def send_parameter(self, name, value):
        if isinstance(value, bool):
            return self.send_bool(AVATAR_PREFIX + name, value)
        return self.send_float(AVATAR_PREFIX + name, value)

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass

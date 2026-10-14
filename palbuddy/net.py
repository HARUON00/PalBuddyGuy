"""Small socket helpers shared by every connection in the project."""

import os
import socket


class Disconnected(ConnectionError):
    """The peer closed the connection (recv returned 0 bytes)."""


def recv_exact_into(sock, view):
    """Fill `view` (a writable memoryview) completely from `sock`.

    The original scripts looped on `recv` and concatenated bytes objects; when
    the peer went away recv() returned b"" forever and the thread spun at 100%
    CPU without ever noticing the disconnect. This raises Disconnected instead
    and reuses one preallocated buffer, so no garbage is created per packet.
    """
    got = 0
    total = len(view)
    while got < total:
        n = sock.recv_into(view[got:], total - got)
        if n == 0:
            raise Disconnected("peer closed the connection")
        got += n


def recv_exact(sock, length):
    buf = bytearray(length)
    recv_exact_into(sock, memoryview(buf))
    return buf


def make_listener(host, port, backlog=2):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if os.name != "nt":
        # Lets the ports be reopened immediately after a restart. On Windows
        # SO_REUSEADDR would allow two processes to share the port, so skip it.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(backlog)
    return s


def tune_stream(sock, low_latency=True, keepalive=True):
    """Disable Nagle (we send tiny, latency-sensitive packets) and turn on keepalive
    so half-dead peers are detected."""
    try:
        if low_latency:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if keepalive:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        pass


def close_quietly(sock):
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass

"""A bounded, latest-pending DEALER link with one socket-owning worker."""

from __future__ import annotations

import ipaddress
import math
import re
import threading
import time
from urllib.parse import urlsplit

from alohamini.errors import ProtocolError, ResponseTimeoutError
from alohamini.remote.protocol import (
    MAX_FRAME_BYTES,
    MAX_MESSAGE_BYTES,
    MAX_MESSAGE_FRAMES,
    decode_message,
    encode_message,
)


def _validate_endpoint(endpoint: str) -> bool:
    """Accept a TCP connect endpoint; return whether IPv6 is required."""
    if not isinstance(endpoint, str) or any(c.isspace() for c in endpoint):
        raise ValueError("endpoint must be a TCP URL with a host and port")
    try:
        parsed = urlsplit(endpoint)
        host, port = parsed.hostname, parsed.port
        if (
            parsed.scheme != "tcp"
            or endpoint != f"tcp://{parsed.netloc}"
            or not host
            or port is None
            or not 1 <= port <= 65535
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if len(host) > 253 or any(
                not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
                for label in host.split(".")
            ):
                raise ValueError from None
            return False
        return address.version == 6
    except ValueError as exc:
        raise ValueError("endpoint must be a TCP URL with a valid host and port") from exc


class LatestLink:
    """Keep at most one active RPC, one latest pending request and one reply.

    submit() and poll() perform no network I/O. A single deadline covers connection,
    transmission and receipt for each active request. Failure ends this link; old
    observations or actions are never replayed on reconnection. A new session needs
    a new instance. The caller validates response IDs, context and action age.
    """

    def __init__(self, endpoint: str, *, timeout_s: float = 2.0) -> None:
        self._ipv6 = _validate_endpoint(endpoint)
        if (
            type(timeout_s) not in (int, float)
            or not math.isfinite(timeout_s)
            or not 0 < timeout_s <= 60
        ):
            raise ValueError("timeout_s must be finite and in (0, 60]")
        self._endpoint = endpoint
        self._timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: list[bytes] | None = None
        self._response: tuple | None = None
        self._error: Exception | None = None
        self._closed = False

    @property
    def error(self) -> Exception | None:
        with self._lock:
            return self._error

    def start(self) -> LatestLink:
        with self._lock:
            if self._closed:
                raise RuntimeError("Remote link is closed")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="remote-policy-link", daemon=True
                )
                self._thread.start()
        return self

    def submit(self, kind: str, body: dict, images: dict[str, bytes] | None = None) -> None:
        parts = encode_message(kind, body, images)
        with self._lock:
            if self._closed or self._thread is None:
                raise RuntimeError("Remote link is not running")
            if self._error is not None:
                raise RuntimeError("Remote link has failed") from self._error
            self._pending = parts
            self._wake.set()

    def poll(self) -> tuple | None:
        with self._lock:
            response, self._response = self._response, None
            return response

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._pending = None
            self._response = None
            self._stop.set()
            self._wake.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ResponseTimeoutError(
                f"Remote request timed out after {self._timeout_s:g}s: {self._endpoint}"
            )
        return remaining

    def _wait(self, socket, event: int, deadline: float) -> bool:
        while not self._stop.is_set():
            remaining = self._remaining(deadline)
            if socket.poll(timeout=max(1, min(20, math.ceil(remaining * 1000))), flags=event):
                return True
        return False

    @staticmethod
    def _receive(socket, zmq) -> tuple:
        parts = []
        size = 0
        while True:
            part = socket.recv(flags=zmq.NOBLOCK)
            parts.append(part)
            size += len(part)
            if (
                len(parts) > MAX_MESSAGE_FRAMES
                or len(part) > MAX_FRAME_BYTES
                or size > MAX_MESSAGE_BYTES
            ):
                raise ProtocolError("Remote response exceeds the multipart size limit")
            if not socket.getsockopt(zmq.RCVMORE):
                return decode_message(parts)

    def _run(self) -> None:
        context = socket = None
        try:
            import zmq

            context = zmq.Context()
            socket = context.socket(zmq.DEALER)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.setsockopt(zmq.SNDHWM, 1)
            socket.setsockopt(zmq.RCVHWM, 1)
            socket.setsockopt(zmq.MAXMSGSIZE, MAX_FRAME_BYTES)
            if self._ipv6:
                socket.setsockopt(zmq.IPV6, 1)
            socket.connect(self._endpoint)
            while not self._stop.is_set():
                with self._lock:
                    parts, self._pending = self._pending, None
                    self._wake.clear()
                if parts is None:
                    self._wake.wait(timeout=0.02)
                    continue
                deadline = time.monotonic() + self._timeout_s
                while self._wait(socket, zmq.POLLOUT, deadline):
                    try:
                        socket.send_multipart(parts, flags=zmq.NOBLOCK)
                        break
                    except zmq.Again:
                        continue
                if not self._wait(socket, zmq.POLLIN, deadline):
                    break
                response = self._receive(socket, zmq)
                self._remaining(deadline)
                with self._lock:
                    if not self._stop.is_set():
                        self._response = response
        except Exception as exc:
            with self._lock:
                if not self._closed:
                    self._error = exc
                self._pending = None
                self._response = None
        finally:
            if socket is not None:
                socket.close(linger=0)
            if context is not None:
                context.term()

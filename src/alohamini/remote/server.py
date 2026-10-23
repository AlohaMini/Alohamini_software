"""Cloud flow-policy inference over bounded JSON and unchanged Host JPEGs."""

from __future__ import annotations

import logging
import math
import time
from copy import deepcopy
from dataclasses import dataclass

from alohamini.errors import ProtocolError
from alohamini.remote.link import _validate_endpoint
from alohamini.remote.protocol import (
    MAX_FRAME_BYTES,
    MAX_MESSAGE_BYTES,
    MAX_MESSAGE_FRAMES,
    context_for,
    decode_message,
    encode_message,
    snapshot_from,
)


class NativeChunkPolicy:
    """Load only the cloud model; never connect to a Host or open hardware."""

    def __init__(
        self, checkpoint, *, device="cuda", chunk_size=None, overrides=None, fixed_current=None,
        rtc=None,
    ):
        from alohamini.learning.checkpoint import read_checkpoint

        _, manifest = read_checkpoint(checkpoint)
        if manifest["kind"] not in ("pi05", "smolvla"):
            raise ValueError(
                "Remote inference currently supports only pi05 and smolvla checkpoints"
            )
        if chunk_size is not None and (type(chunk_size) is not int or not 1 <= chunk_size <= 512):
            raise ValueError("chunk_size must be an integer in [1, 512]")
        options = dict(overrides or {})
        if {"device", "checkpoint", "fixed_current", "offline"} & options.keys():
            raise ValueError("Remote policy overrides contain a reserved option")
        from alohamini.learning.policy import NativePolicy

        self.policy = NativePolicy(
            checkpoint, device=device, fixed_current=fixed_current, **options
        )
        from alohamini.rtc import RTCConfig

        self.rtc_config = rtc or RTCConfig()
        self.rtc = self.rtc_config.to_dict() if self.rtc_config.enabled else None
        horizon = (
            self.policy.config.network["action_horizon"] if manifest["kind"] == "pi05"
            else self.policy.config.chunk_size
        )
        available = horizon if self.rtc else min(horizon, self.policy.config.n_action_steps)
        self.chunk_size = available if chunk_size is None else chunk_size
        if not 1 <= self.chunk_size <= 512:
            raise ValueError("Remote chunk_size must be an integer in [1, 512]")
        if self.chunk_size > available:
            raise ValueError("Remote chunk_size exceeds the policy execution horizon")
        if self.rtc:
            self.policy.validate_rtc(self.rtc_config)
            self.rtc_config.validate_horizon(self.chunk_size)
        self.names = tuple(self.policy.names)
        self.cameras = tuple(self.policy.manifest["cameras"])
        self.fps = self.policy.fps
        self.fixed_current = tuple(self.policy.fixed_current)
        self._base_fixed = deepcopy(self.policy.fixed_dimensions)
        self._base_deployment = deepcopy(self.policy.fixed_deployment)
        self._initialized = False

    def initialize(self, snapshot, client_id, fps, *, fixed=None):
        from alohamini.apps.teleoperation import ready_units
        from alohamini.datasets.record import state_names
        from alohamini.fixed import VELOCITIES, validate

        if fps != self.fps:
            raise ValueError("Remote control fps must exactly match the checkpoint")
        if list(self.names) != state_names(snapshot.robot_model):
            raise ValueError("Remote action names differ from the complete Host action contract")
        if ready_units(snapshot, snapshot.robot_model, client_id) is None:
            raise ValueError("Remote initialization requires a controllable Host snapshot")
        if not set(self.cameras) <= set(snapshot.payload["_robot_metadata"].get("cameras", ())):
            raise ValueError("Host does not provide every checkpoint camera")
        self.reset()
        self.policy.bind_robot(snapshot)
        captured = validate(fixed, snapshot.robot_model)
        if set(captured) != set(self.fixed_current):
            raise ValueError("Pi fixed-current selection differs from the cloud configuration")
        # Age was checked by capture() on the Pi. Check the transmitted values
        # against that exact snapshot without comparing unrelated machine clocks.
        for name, value in captured.items():
            expected = 0.0 if name in VELOCITIES else snapshot.payload[name]
            if value != expected:
                raise ValueError(
                    f"Pi fixed target differs from its initialization snapshot: {name}"
                )
        if captured:
            targets = validate(self.policy.fixed_dimensions, snapshot.robot_model)
            targets.update(captured)
            config = dict(version=1, targets=targets, source="checkpoint_means_and_pi_feedback")
            validate(config, snapshot.robot_model)
            self.policy.fixed_dimensions = config
            self.policy.fixed_deployment.update(
                fixed_dimensions=config,
                current_capture=dict(
                    targets=captured,
                    request_started_s=snapshot.request_started_s,
                    received_s=snapshot.received_s,
                    **context_for(snapshot),
                    robot_metadata=deepcopy(snapshot.payload["_robot_metadata"]),
                ),
            )
            self.policy._fixed_current_bound = True
        self._initialized = True

    def predict(self, snapshot):
        if not self._initialized:
            raise RuntimeError("Remote policy has not been initialized")
        chunk = self.policy.predict_action_chunk(snapshot)
        if chunk.shape[0] < self.chunk_size:
            raise ValueError("Policy returned fewer actions than the configured remote chunk")
        return chunk[: self.chunk_size]

    def predict_rtc(self, snapshot, previous_actions, inference_delay):
        if not self._initialized or not self.rtc:
            raise RuntimeError("Remote RTC policy has not been initialized")
        chunk = self.policy.predict_action_chunk(
            snapshot, rtc=self.rtc_config, previous_actions=previous_actions,
            inference_delay=inference_delay,
        )
        return chunk[: self.chunk_size]

    def reset(self):
        self._initialized = False
        self.policy.reset()
        self.policy.fixed_dimensions = deepcopy(self._base_fixed)
        self.policy.fixed_deployment = deepcopy(self._base_deployment)
        self.policy._fixed_current_bound = False


def _token(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 64:
        raise ProtocolError(f"Invalid remote {label}")
    return value


def _number(value, label, *, positive=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or (positive and value <= 0):
        raise ProtocolError(f"Invalid remote {label}")
    return value


def _identity(body):
    session = _token(body.get("session"), "session")
    request_id = body.get("id")
    if type(request_id) is not int or not 1 <= request_id <= 2**63 - 1:
        raise ProtocolError("Remote request id must be a positive integer")
    return session, request_id


def _observation(body, images):
    fps, tick = _number(body.get("fps"), "fps", positive=True), body.get("tick")
    if fps > 50 or type(tick) is not int or not 0 <= tick <= 2**63 - 1:
        raise ProtocolError("Invalid remote control tick or fps")
    client_id = _token(body.get("client_id"), "client_id")
    snapshot = snapshot_from(body.get("snapshot"), images)
    context = context_for(snapshot)
    if body.get("context") != context:
        raise ProtocolError("Remote context does not match its Host snapshot")
    safety = snapshot.payload["_safety"]
    if (
        safety.get("fault")
        or safety.get("phase") not in ("ready", "active")
        or safety.get("feedback_valid") is not True
        or safety.get("lift_reference_valid") is not True
        or safety.get("control_owner") not in (None, client_id)
    ):
        raise ProtocolError("Remote snapshot is not controllable by this client")
    return snapshot, context, client_id, fps, tick


@dataclass
class _Session:
    route: bytes
    token: str
    context: dict
    client_id: str
    fps: float
    request_id: int
    tick: int
    received_s: float
    last_seen_s: float


class _Service:
    def __init__(self, backend, timeout_s):
        self.backend, self.timeout_s = backend, timeout_s
        self.session = None
        self.closed = None
        self.plans = {}

    def end(self):
        self.plans.clear()
        if self.session is not None:
            current, self.session = self.session, None
            self.closed = (current.route, current.token, current.request_id)
            self.backend.reset()

    def expire(self):
        if self.session and time.monotonic() - self.session.last_seen_s > self.timeout_s:
            self.end()

    def handle(self, route, kind, body, images):
        token, request_id = _identity(body)
        current = self.session
        if kind == "stop" and self.closed and (route, token, request_id) == self.closed:
            return "ack", {"session": token, "id": request_id}
        if kind == "hello":
            if current is not None:
                raise ProtocolError("Remote policy already has an active session")
            if self.closed and (route, token) == self.closed[:2]:
                raise ProtocolError("Closed remote sessions cannot be replayed")
            snapshot, context, client_id, fps, tick = _observation(body, images)
            try:
                self.backend.initialize(snapshot, client_id, fps, fixed=body.get("fixed"))
                if fps != self.backend.fps:
                    raise ValueError("Remote backend fps differs from the requested fps")
            except KeyError as exc:
                self.backend.reset()
                raise ProtocolError(
                    f"Host initialization snapshot is missing a field: {exc}"
                ) from exc
            except (ValueError, TypeError, RuntimeError):
                self.backend.reset()
                raise
            self.session = _Session(
                route, token, context, client_id, fps, request_id, tick,
                snapshot.received_s, time.monotonic(),
            )
            return "ready", {
                "session": token, "id": request_id, "context": context, "fps": fps,
                "names": list(self.backend.names), "cameras": list(self.backend.cameras),
                "chunk_size": self.backend.chunk_size,
                "fixed_current": list(getattr(self.backend, "fixed_current", ())),
                **({"rtc": self.backend.rtc} if getattr(self.backend, "rtc", None) else {}),
            }
        if current is None or (route, token) != (current.route, current.token):
            raise ProtocolError("No matching active remote session")
        if kind == "stop":
            if request_id < current.request_id:
                raise ProtocolError("Stale remote stop request")
            current.request_id = request_id
            self.end()
            return "ack", {"session": token, "id": request_id}
        if request_id <= current.request_id:
            raise ProtocolError("Remote requests must have increasing ids")
        if kind == "ping":
            current.request_id, current.last_seen_s = request_id, time.monotonic()
            return "pong", {"session": token, "id": request_id}
        if kind != "observation":
            raise ProtocolError("Expected hello, observation, ping or stop")
        try:
            snapshot, context, client_id, fps, tick = _observation(body, images)
        except (ProtocolError, ValueError, TypeError):
            self.end()
            raise
        if (
            context != current.context or client_id != current.client_id or fps != current.fps
            or tick < current.tick or snapshot.received_s < current.received_s
        ):
            self.end()
            raise ProtocolError("Remote Host context, control rate or observation order changed")
        current.request_id, current.tick = request_id, tick
        current.received_s, current.last_seen_s = snapshot.received_s, time.monotonic()
        started = time.monotonic()
        try:
            if getattr(self.backend, "rtc", None):
                rtc = body.get("rtc")
                if not isinstance(rtc, dict) or set(rtc) != {"plan_id", "inference_delay"}:
                    raise ProtocolError("RTC observations require plan_id and inference_delay")
                plan_id, delay = rtc["plan_id"], rtc["inference_delay"]
                if type(delay) is not int or not 0 <= delay < self.backend.chunk_size:
                    raise ProtocolError("Invalid RTC inference delay")
                previous = None
                if plan_id is not None:
                    if type(plan_id) is not int or plan_id not in self.plans:
                        raise ProtocolError("RTC prefix refers to an unknown action plan")
                    previous_tick, plan = self.plans[plan_id]
                    offset = tick - previous_tick
                    if not 0 < offset < len(plan) or delay >= len(plan) - offset:
                        raise ProtocolError("RTC delay exceeds the remaining action overlap")
                    previous = plan[offset:]
                elif self.plans:
                    raise ProtocolError("RTC must identify the accepted previous action plan")
                actions = self.backend.predict_rtc(snapshot, previous, delay)
            else:
                if "rtc" in body:
                    raise ProtocolError("RTC was not enabled by the server")
                actions = self.backend.predict(snapshot)
            inference_s = time.monotonic() - started
            if inference_s > self.timeout_s:
                raise TimeoutError("Remote inference exceeded the request time budget")
            actions = actions.tolist() if hasattr(actions, "tolist") else actions
            if not isinstance(actions, list) or len(actions) != self.backend.chunk_size:
                raise ValueError("Remote policy returned an invalid chunk length")
            for row in actions:
                if not isinstance(row, list) or len(row) != len(self.backend.names):
                    raise ValueError("Remote policy returned invalid action dimensions")
                for value in row:
                    _number(value, "absolute action")
        except KeyError as exc:
            self.end()
            raise ProtocolError(f"Host observation is missing a policy input: {exc}") from exc
        except (ProtocolError, ValueError, TypeError, RuntimeError, TimeoutError):
            self.end()
            raise
        if getattr(self.backend, "rtc", None):
            self.plans[request_id] = (tick, actions)
            while len(self.plans) > 16:
                del self.plans[next(iter(self.plans))]
        return "actions", {
            "session": token, "id": request_id, "tick": tick, "fps": fps,
            "context": context, "names": list(self.backend.names),
            "actions": actions, "inference_s": inference_s,
        }


class _FatalMultipartError(Exception):
    """Fail closed instead of draining an unbounded malformed multipart message."""


def _receive_frames(socket, zmq):
    parts, total, count, exceeded = [], 0, 0, False
    while socket.getsockopt(zmq.RCVMORE):
        part = socket.recv(flags=zmq.NOBLOCK)
        count += 1
        if count > MAX_MESSAGE_FRAMES + 32:
            raise _FatalMultipartError("Remote multipart has excessive trailing frames")
        total += len(part)
        exceeded = exceeded or (
            count > MAX_MESSAGE_FRAMES or len(part) > MAX_FRAME_BYTES or total > MAX_MESSAGE_BYTES
        )
        if not exceeded:
            parts.append(part)
    if exceeded:
        raise ProtocolError("Remote request exceeds the multipart size limit")
    return parts


def serve(endpoint, backend, stop_event, *, request_timeout_s=10.0, poll_timeout_ms=50):
    """Serve one robot session with bounded queues; all times retained from the Pi.

    The timeout bounds idle sessions and rejects over-budget results. It cannot
    interrupt a running GPU kernel. The Pi independently expires pending actions.
    """
    ipv6 = _validate_endpoint(endpoint)
    _number(request_timeout_s, "request timeout", positive=True)
    if type(poll_timeout_ms) is not int or not 1 <= poll_timeout_ms <= 1000:
        raise ValueError("poll_timeout_ms must be an integer in [1, 1000]")
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    service = _Service(backend, request_timeout_s)
    try:
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 2)
        socket.setsockopt(zmq.RCVHWM, 2)
        socket.setsockopt(zmq.MAXMSGSIZE, MAX_FRAME_BYTES)
        socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        if ipv6:
            socket.setsockopt(zmq.IPV6, 1)
        socket.bind(endpoint)
        print(f"Remote policy server listening on {endpoint}", flush=True)
        while not stop_event.is_set():
            service.expire()
            if not socket.poll(timeout=poll_timeout_ms, flags=zmq.POLLIN):
                continue
            route = socket.recv(flags=zmq.NOBLOCK)
            body = {}
            try:
                parts = _receive_frames(socket, zmq)
                kind, body, images = decode_message(parts)
                kind, response = service.handle(route, kind, body, images)
            except (ProtocolError, ValueError, TypeError, RuntimeError, TimeoutError) as exc:
                kind, response = "error", {
                    "session": body.get("session"), "id": body.get("id"),
                    "message": str(exc)[:4096],
                }
            if stop_event.is_set():
                break
            try:
                socket.send_multipart([route, *encode_message(kind, response)], flags=zmq.NOBLOCK)
            except zmq.ZMQError as exc:
                if exc.errno not in (zmq.EAGAIN, zmq.EHOSTUNREACH):
                    raise
                logging.warning("Remote policy reply could not be delivered: %s", exc)
                if service.session and service.session.route == route:
                    service.end()
    finally:
        try:
            service.end()
        finally:
            socket.close(linger=0)
            context.term()

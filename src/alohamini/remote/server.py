"""Cloud trajectory inference over bounded JSON and unchanged Host JPEGs."""

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


class NativePlanPolicy:
    """Load only the cloud model; never connect to a Host or open hardware."""

    def __init__(
        self,
        checkpoint,
        *,
        device="cuda",
        chunk_size=None,
        overrides=None,
        fixed_current=None,
        execution=None,
    ):
        from alohamini.learning.checkpoint import read_checkpoint
        from alohamini.policies.registry import algorithm

        _, manifest = read_checkpoint(checkpoint)
        component = algorithm(manifest["kind"])
        if not getattr(component, "supports_remote_chunks", False):
            raise ValueError(f"{manifest['kind']} does not declare remote action-chunk support")
        if chunk_size is not None and (type(chunk_size) is not int or not 1 <= chunk_size <= 512):
            raise ValueError("chunk_size must be an integer in [1, 512]")
        options = dict(overrides or {})
        if {"device", "checkpoint", "fixed_current", "offline"} & options.keys():
            raise ValueError("Remote policy overrides contain a reserved option")
        from alohamini.learning.policy import NativePolicy

        self.policy = NativePolicy(
            checkpoint, device=device, fixed_current=fixed_current, **options
        )
        from dataclasses import replace

        from alohamini.trajectory_execution import ExecutionConfig

        self.plan_capabilities = self.policy.plan_capabilities
        requested = replace(execution or ExecutionConfig(continuity="auto"), enabled=True)
        self.execution_config = self.plan_capabilities.resolve(requested)
        self.execution = self.execution_config.to_dict()
        self.chunk_size = chunk_size
        if chunk_size is not None:
            if self.plan_capabilities.kinds not in (("dense",), ("tcp",)):
                raise ValueError("--chunk-size only applies to sampled action plans")
            if chunk_size > component.action_horizon(self.policy.config):
                raise ValueError("Remote chunk_size exceeds the policy execution horizon")
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

        self.fps = fps  # Control cadence is separate from the checkpoint's data FPS.
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

    def predict_plan(self, snapshot, prefix=None):
        if not self._initialized:
            raise RuntimeError("Remote policy has not been initialized")
        plan = self.policy.predict_plan(
            snapshot,
            prefix=prefix,
            continuity=self.execution_config.continuity,
        )
        if self.chunk_size is not None:
            from alohamini.trajectory import DenseActionPlan

            if getattr(plan, "kind", None) == "tcp":
                from dataclasses import replace

                plan = replace(plan, actions=plan.actions[: self.chunk_size])
            elif isinstance(plan, DenseActionPlan):
                plan = DenseActionPlan(
                    plan.names, plan.actions[: self.chunk_size], plan.period_s, plan.start_s
                )
            else:
                raise ValueError("--chunk-size only applies to sampled action plans")
        return plan

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
                route,
                token,
                context,
                client_id,
                fps,
                request_id,
                tick,
                snapshot.received_s,
                time.monotonic(),
            )
            return "ready", {
                "session": token,
                "id": request_id,
                "context": context,
                "fps": fps,
                "names": list(self.backend.names),
                "cameras": list(self.backend.cameras),
                "fixed_current": list(getattr(self.backend, "fixed_current", ())),
                "plan_kinds": list(self.backend.plan_capabilities.kinds),
                "execution": self.backend.execution,
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
            context != current.context
            or client_id != current.client_id
            or fps != current.fps
            or tick < current.tick
            or snapshot.received_s < current.received_s
        ):
            self.end()
            raise ProtocolError("Remote Host context, control rate or observation order changed")
        current.request_id, current.tick = request_id, tick
        current.received_s, current.last_seen_s = snapshot.received_s, time.monotonic()
        started = time.monotonic()
        try:
            from alohamini.trajectory import committed_prefix, plan_from_dict

            plan_id = body.get("plan_id")
            delay = _number(body.get("inference_delay_s", 0.0), "inference delay")
            if delay < 0:
                raise ProtocolError("Inference delay must be nonnegative")
            prefix = None
            if plan_id is not None:
                if type(plan_id) is not int or plan_id not in self.plans:
                    raise ProtocolError("Prefix refers to an unknown accepted trajectory plan")
                previous_time, previous = self.plans[plan_id]
                offset = snapshot.request_started_s - previous_time
                if offset <= 0 or offset + delay >= previous.end_s:
                    raise ProtocolError("Inference delay exceeds the remaining trajectory overlap")
                prefix = committed_prefix(previous, offset, fps, delay)
            elif self.plans and self.backend.execution.get("continuity", "none") != "none":
                raise ProtocolError("Continuity requires the accepted previous trajectory plan")
            plan = self.backend.predict_plan(snapshot, prefix)
            plan = plan_from_dict(plan.to_dict(), names=self.backend.names)
            if plan.to_dict()["kind"] not in self.backend.plan_capabilities.kinds:
                raise ValueError("Policy returned an undeclared trajectory kind")
            inference_s = time.monotonic() - started
            if inference_s > self.timeout_s:
                raise TimeoutError("Remote inference exceeded the request time budget")
        except KeyError as exc:
            self.end()
            raise ProtocolError(f"Host observation is missing a policy input: {exc}") from exc
        except (ProtocolError, ValueError, TypeError, RuntimeError, TimeoutError):
            self.end()
            raise
        self.plans[request_id] = (snapshot.request_started_s, plan)
        while len(self.plans) > 16:
            del self.plans[next(iter(self.plans))]
        return "plan", {
            "session": token,
            "id": request_id,
            "tick": tick,
            "fps": fps,
            "context": context,
            "names": list(self.backend.names),
            "plan": plan.to_dict(),
            "inference_s": inference_s,
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
                kind, response = (
                    "error",
                    {
                        "session": body.get("session"),
                        "id": body.get("id"),
                        "message": str(exc)[:4096],
                    },
                )
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

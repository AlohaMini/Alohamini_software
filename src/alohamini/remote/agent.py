"""Pi-side action scheduling; models and image decoding live on the server."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from uuid import uuid4

from alohamini._validation import finite_number
from alohamini.apps.teleoperation import ready_units, stop_owned_robot
from alohamini.client import HostClient, control_feedback_valid
from alohamini.errors import ProtocolError
from alohamini.fixed import FixedGuard, capture, dimensions
from alohamini.protocol import command_target_keys
from alohamini.remote.link import LatestLink
from alohamini.remote.protocol import context_for, snapshot_body


@dataclass(frozen=True)
class ObservationRequest:
    tick: int
    sampled_s: float


class ActionBuffer:
    """Anchor predictions to observation time, skipping elapsed action prefixes.

    Tick numbers follow the Pi clock even when inference or feedback stalls.
    This is an absolute-target buffer, not a relative-action integrator.
    """

    def __init__(self, session, context, names, fps, max_age_s, chunk_size):
        self.session, self.context = session, context
        self.names, self.fps = tuple(names), fps
        self.max_age_s, self.chunk_size = max_age_s, chunk_size
        self.requests = OrderedDict()
        self.targets = {}
        self.last_response_id = 0
        self.skipped = 0
        self.expired = 0
        self.last_round_trip_s = None
        self.accepted_plan_id = None

    def remember(self, request_id, tick, sampled_s):
        self.requests[request_id] = ObservationRequest(tick, sampled_s)
        while len(self.requests) > 256:
            self.requests.popitem(last=False)

    def accept(self, body, *, now, tick):
        request_id = body.get("id")
        if type(request_id) is not int or request_id <= self.last_response_id:
            raise ProtocolError("Remote action response has a repeated or invalid request ID")
        record = self.requests.get(request_id)
        if record is None:
            raise ProtocolError("Remote action response does not match a retained observation")
        if (
            body.get("session") != self.session
            or body.get("context") != self.context
            or type(body.get("tick")) is not int
            or body["tick"] != record.tick
            or type(body.get("fps")) not in (int, float)
            or body["fps"] != self.fps
            or body.get("names") != list(self.names)
        ):
            raise ProtocolError("Remote action response changed the negotiated control contract")
        actions = body.get("actions")
        if not isinstance(actions, list) or not 1 <= len(actions) <= self.chunk_size:
            raise ProtocolError("Remote action chunk has an invalid length")
        for row in actions:
            if not isinstance(row, list) or len(row) != len(self.names):
                raise ProtocolError("Remote action chunk has an invalid target shape")
            for value in row:
                finite_number(value, "remote action target")
        inference_s = body.get("inference_s")
        finite_number(inference_s, "remote inference time")
        if inference_s < 0:
            raise ProtocolError("Remote inference time must be nonnegative")
        self.last_response_id = request_id
        self.last_round_trip_s = now - record.sampled_s
        for old_id in list(self.requests):
            if old_id <= request_id:
                del self.requests[old_id]
        # Never retain an older plan's tail after accepting a newer plan.
        self.targets.clear()
        self.accepted_plan_id = None
        age = now - record.sampled_s
        if not 0 <= age < self.max_age_s:
            self.expired += 1
            return 0
        prefix = max(0, tick - record.tick)
        self.skipped += min(prefix, len(actions))
        for index in range(prefix, len(actions)):
            self.targets[record.tick + index] = (
                dict(zip(self.names, map(float, actions[index]), strict=True)),
                record.sampled_s + self.max_age_s,
            )
        if prefix >= len(actions):
            self.expired += 1
        else:
            self.accepted_plan_id = request_id
        return max(0, len(actions) - prefix)

    def pop(self, tick, now):
        for old_tick in list(self.targets):
            if old_tick < tick or now >= self.targets[old_tick][1]:
                del self.targets[old_tick]
        entry = self.targets.pop(tick, None)
        return None if entry is None else entry[0]


def _healthy(client, snapshot, robot_model, expected_context=None):
    if ready_units(snapshot, robot_model, client.client_id) is None or not control_feedback_valid(
        snapshot
    ):
        raise RuntimeError("Fresh, controllable Host feedback is required for remote inference")
    context = context_for(snapshot)
    if expected_context is not None and context != expected_context:
        raise RuntimeError("Host session, control epoch, calibration or lift reference changed")
    return context


def _check_ready(body, session, request_id, context, fps, robot_model, images, fixed_names):
    names, cameras, chunk_size = body.get("names"), body.get("cameras"), body.get("chunk_size")
    expected_names = command_target_keys(robot_model) - {"lift_axis.stop"}
    if (
        body.get("session") != session
        or type(body.get("id")) is not int
        or body["id"] != request_id
        or body.get("context") != context
        or type(body.get("fps")) not in (int, float)
        or body["fps"] != fps
        or not isinstance(names, list)
        or any(not isinstance(name, str) for name in names)
        or len(names) != len(expected_names)
        or set(names) != expected_names
        or not isinstance(cameras, list)
        or any(not isinstance(name, str) for name in cameras)
        or len(cameras) != len(set(cameras))
        or not set(cameras) <= set(images)
        or type(chunk_size) is not int
        or not 1 <= chunk_size <= 512
        or body.get("fixed_current", []) != fixed_names
    ):
        raise ProtocolError("Remote server returned an incompatible policy contract")
    return names, cameras, chunk_size


def _response(link):
    if link.error is not None:
        raise RuntimeError(f"Remote inference connection failed: {link.error}") from link.error
    result = link.poll()
    if result is not None:
        kind, body, images = result
        if images:
            raise ProtocolError("Remote server must not return image frames")
        if kind == "error":
            raise RuntimeError(f"Remote inference failed: {body.get('message', 'unknown error')}")
    return result


def run_session(
    client,
    link,
    robot_model,
    *,
    fps=30,
    duration_s=90.0,
    max_prediction_age_s=1.0,
    rpc_timeout_s=10.0,
    ack_timeout_s=0.5,
    fixed_current=None,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Run a bounded episode, stopping the owned Host lease on every exit.

    A successful socket send is not a Host acknowledgement. Fresh Host command
    identities must advance while targets are being sent. Reconnection requires
    an explicit new session; no queued prediction survives a changed context.
    """
    if type(fps) is not int or not 1 <= fps <= 30:
        raise ValueError("Remote control fps must be an integer in [1, 30]")
    for name, value in (
        ("duration", duration_s),
        ("prediction age", max_prediction_age_s),
        ("RPC timeout", rpc_timeout_s),
        ("Host acknowledgement timeout", ack_timeout_s),
    ):
        finite_number(value, name)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    session, request_id = uuid4().hex, 1
    identity = None
    stats = dict(sent=0, observations=0, acknowledged_sequence=0)
    try:
        client.connect_control()
        initial = client.refresh(include_images=True)
        context = _healthy(client, initial, robot_model)
        fixed_names = dimensions(fixed_current, robot_model)
        fixed = capture(initial, fixed_names) if fixed_names else None
        fixed_guard = FixedGuard(fixed, initial.payload["_robot_metadata"]) if fixed else None
        link.start()
        link.submit(
            "hello",
            dict(
                session=session,
                id=request_id,
                tick=0,
                fps=fps,
                client_id=client.client_id,
                context=context,
                snapshot=snapshot_body(initial),
                fixed=fixed,
            ),
            initial.images,
        )
        deadline = clock() + rpc_timeout_s
        ready = None
        while clock() < deadline:
            result = _response(link)
            if result is not None:
                if result[0] != "ready":
                    raise ProtocolError("Expected remote policy readiness before control")
                ready = result[1]
                break
            sleep(0.005)
        if ready is None:
            raise RuntimeError("Remote policy startup timed out; no motion was commanded")
        names, cameras, chunk_size = _check_ready(
            ready, session, request_id, context, fps, robot_model, initial.images, fixed_names
        )
        from alohamini.rtc import RTCConfig

        rtc = None
        if "rtc" in ready:
            try:
                rtc = RTCConfig(**ready["rtc"])
                rtc.validate_horizon(chunk_size)
                if not rtc.enabled:
                    raise ValueError("RTC server configuration must be enabled")
            except (TypeError, ValueError) as exc:
                raise ProtocolError(f"Invalid RTC server settings: {exc}") from exc
            max_prediction_age_s = min(max_prediction_age_s, rtc.max_prediction_age_s)
        buffer = ActionBuffer(session, context, names, fps, max_prediction_age_s, chunk_size)
        pending_rtc = False
        last_request_tick = -1
        origin = clock()
        first_action_deadline = origin + rpc_timeout_s
        last_tick = -1
        next_tick_time = origin
        last_ack_s = None
        acknowledged = 0
        report_at, report_sent = origin + 1.0, 0
        while clock() - origin < duration_s:
            delay = next_tick_time - clock()
            if delay > 0:
                sleep(delay)
            snapshot = client.refresh(include_images=True)
            _healthy(client, snapshot, robot_model, context)
            if fixed_guard is not None:
                fixed_guard.check(snapshot)
            now = clock()
            if now - origin >= duration_s:
                break
            tick = max(0, math.floor((now - origin) * fps))
            next_tick_time = origin + (tick + 1) / fps
            command = snapshot.payload["_safety"].get("command", {})
            if (
                command.get("client_id") == client.client_id
                and type(command.get("sequence")) is int
                and command["sequence"] > acknowledged
            ):
                acknowledged = command["sequence"]
                stats["acknowledged_sequence"] = acknowledged
                last_ack_s = now
            if last_ack_s is not None and now - last_ack_s > ack_timeout_s:
                raise RuntimeError("Host stopped acknowledging remote control commands")
            result = _response(link)
            if result is not None:
                if result[0] != "actions":
                    raise ProtocolError("Expected a remote action chunk")
                buffer.accept(result[1], now=now, tick=tick)
                pending_rtc = False
            if not set(cameras) <= set(snapshot.images):
                raise RuntimeError("Host is missing a policy camera")
            source_tick = max(0, math.floor((snapshot.request_started_s - origin) * fps))
            if rtc is None or (
                not pending_rtc and (
                    last_request_tick < 0 or source_tick >= last_request_tick + rtc.replan_steps
                )
            ):
                request_id += 1
                body = dict(
                    session=session,
                    id=request_id,
                    tick=source_tick,
                    fps=fps,
                    client_id=client.client_id,
                    context=context,
                    snapshot=snapshot_body(snapshot),
                )
                if rtc is not None:
                    measured_delay = math.ceil((buffer.last_round_trip_s or 0) * fps) + 1
                    body["rtc"] = dict(
                        plan_id=buffer.accepted_plan_id,
                        inference_delay=max(
                            rtc.inference_delay_steps, measured_delay, tick - source_tick + 1
                        ),
                    )
                    pending_rtc = True
                    last_request_tick = source_tick
                buffer.remember(request_id, source_tick, snapshot.request_started_s)
                link.submit(
                    "observation", body,
                    {camera: snapshot.images[camera] for camera in cameras},
                )
                stats["observations"] += 1
            if tick <= last_tick:
                continue
            targets = buffer.pop(tick, now)
            if targets is None:
                if stats["sent"] or now >= first_action_deadline:
                    raise RuntimeError(
                        "No unexpired remote action covers the current tick; control stopped. "
                        "Check network/inference latency and the configured action horizon."
                    )
                continue
            if fixed is not None:
                targets.update(fixed["targets"])
            submitted = client.send_command(targets, based_on=snapshot)
            if submitted is None:
                raise RuntimeError("Host command channel is unavailable")
            identity = submitted
            if last_ack_s is None:
                last_ack_s = now
            stats["sent"] += 1
            last_tick = tick
            if now >= report_at:
                elapsed = now - (report_at - 1.0)
                print(
                    f"[REMOTE] sent_hz={(stats['sent'] - report_sent) / elapsed:.1f} "
                    f"buffer={len(buffer.targets)} "
                    f"observation_to_result_s={buffer.last_round_trip_s:.3f} "
                    f"skipped={buffer.skipped} expired_chunks={buffer.expired}",
                    flush=True,
                )
                report_at, report_sent = now + 1.0, stats["sent"]
        return stats
    finally:
        stop_owned_robot(client, robot_model, identity)
        # Best effort only: hardware stop never waits for the cloud.
        try:
            link.submit("stop", dict(session=session, id=request_id + 1))
            deadline = clock() + min(rpc_timeout_s, 0.2)
            while clock() < deadline and link.error is None:
                result = link.poll()
                if result is not None and result[0] == "ack":
                    break
                sleep(0.005)
        except Exception:
            pass
        link.close()


def run(server, *, host="127.0.0.1", robot_model, rpc_timeout_s=10.0, **options):
    """Connect directly from the Pi to its local Host and the cloud endpoint."""
    with HostClient(host, expected_model=robot_model, timeout_s=0.2) as client:
        link = LatestLink(server, timeout_s=rpc_timeout_s)
        return run_session(client, link, robot_model, rpc_timeout_s=rpc_timeout_s, **options)

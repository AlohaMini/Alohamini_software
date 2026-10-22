"""Real loopback transport with an in-memory Host and a model-free cloud backend."""

import copy
import socket
import threading
import time
from dataclasses import asdict
from unittest.mock import patch

import pytest

from alohamini.model import get_robot_model
from alohamini.protocol import HostSnapshot, command_target_keys
from alohamini.remote.agent import run_session
from alohamini.remote.link import LatestLink
from alohamini.remote.server import serve
from alohamini.schema import CommandIdentity

pytest.importorskip("zmq")

MODEL = "alohamini2pro"
# Deliberately opaque JPEG-framed bytes: any image decoding/reencoding would fail
# or change this payload. Neither the Pi executor nor the fake backend needs pixels.
JPEG = b"\xff\xd8\x00original Pi camera payload\xff\xd9"
NAMES = sorted(command_target_keys(MODEL) - {"lift_axis.stop"})


class MemoryHost:
    client_id = "edge-integration"

    def __init__(self):
        self.sent = []
        self.connected = False
        self.payload = {name: 0.0 for name in NAMES}
        motors = {}
        for motor in get_robot_model(MODEL).actuators:
            if motor.name.startswith("arm_"):
                gripper = motor.name.endswith("gripper")
                self.payload[f"{motor.name}.pos"] = 50.0 if gripper else 0.0
                motors[motor.name] = {
                    "normalization": "range_0_100" if gripper else "range_m100_100"
                }
        self.payload.update(
            {
                "lift_axis.height_mm": 100.0,
                "lift_axis.reference_sequence": 1,
                "_robot_metadata": {
                    "schema_version": 1,
                    "robot_model": MODEL,
                    "motors": motors,
                    "cameras": ["front"],
                },
                "_safety": {
                    "version": 1,
                    "host_session_id": "loopback-host",
                    "control_epoch": 0,
                    "control_owner": None,
                    "phase": "ready",
                    "feedback_valid": True,
                    "lift_reference_valid": True,
                    "command_watchdog_timeout_s": 1.0,
                },
            }
        )

    def connect_control(self):
        self.connected = True

    def refresh(self, *, include_images=False):
        now = time.monotonic()
        payload = copy.deepcopy(self.payload)
        if self.sent:
            payload["_safety"].update(
                control_owner=self.client_id, command=asdict(self.sent[-1][2])
            )
        return HostSnapshot(payload, {"front": JPEG} if include_images else {}, now, now)

    def send_command(self, targets, *, based_on):
        assert self.connected
        status = based_on.payload["_safety"]
        identity = CommandIdentity(
            self.client_id, len(self.sent) + 1, status["host_session_id"], status["control_epoch"]
        )
        self.sent.append((time.monotonic(), dict(targets), identity))
        return identity


class FakePolicy:
    names = NAMES
    cameras = ["front"]
    fps = 30
    chunk_size = 30
    fixed_current = ()

    def __init__(self, *, slow_s=0.0, disconnect=None):
        self.slow_s = slow_s
        self.disconnect = disconnect
        self.images = []
        self.calls = 0
        self.slow_span = None
        self.slow_finished = threading.Event()

    def initialize(self, snapshot, client_id, fps, *, fixed=None):
        assert fps == self.fps
        assert client_id == MemoryHost.client_id
        assert fixed is None
        self.images.append(dict(snapshot.images))

    def predict(self, snapshot):
        self.calls += 1
        self.images.append(dict(snapshot.images))
        if self.calls == 2:
            started = time.monotonic()
            if self.disconnect is not None:
                self.disconnect.set()
            time.sleep(self.slow_s)
            self.slow_span = (started, time.monotonic())
            self.slow_finished.set()
        baseline = [snapshot.payload[name] for name in self.names]
        index = self.names.index("arm_left_shoulder_pan.pos")
        actions = []
        for step in range(self.chunk_size):
            row = list(baseline)
            row[index] = self.calls + step / 100
            actions.append(row)
        return actions

    def reset(self):
        pass


class LoopbackServer:
    def __init__(self, backend):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        self.endpoint = f"tcp://127.0.0.1:{port}"
        self.stop = threading.Event()
        self.backend = backend
        self.error = None
        self.worker = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            serve(
                self.endpoint,
                self.backend,
                self.stop,
                request_timeout_s=1.5,
                poll_timeout_ms=5,
            )
        except Exception as exc:
            self.error = exc

    def __enter__(self):
        self.worker.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.worker.join(timeout=1.0)
        assert not self.worker.is_alive(), "loopback cloud worker did not stop"
        assert self.error is None, self.error


def test_jpeg_passes_unchanged_and_local_actions_continue_while_cloud_is_busy():
    host = MemoryHost()
    backend = FakePolicy(slow_s=0.25)
    with LoopbackServer(backend) as server:
        link = LatestLink(server.endpoint, timeout_s=0.8)
        with patch("alohamini.remote.agent.stop_owned_robot") as stop:
            stats = run_session(
                host,
                link,
                MODEL,
                duration_s=0.45,
                max_prediction_age_s=0.9,
                rpc_timeout_s=0.8,
            )
        assert backend.slow_finished.is_set()
        started, finished = backend.slow_span
        during_inference = [entry for entry in host.sent if started < entry[0] < finished]
        assert len(during_inference) >= 3
        assert stats["sent"] == len(host.sent)
        assert stats["acknowledged_sequence"] >= stats["sent"] - 1
        assert backend.images and all(images == {"front": JPEG} for images in backend.images)
        stop.assert_called_once_with(host, MODEL, host.sent[-1][2])
        assert not link._thread.is_alive()


def test_expired_plan_stops_owned_host_even_when_a_long_chunk_remains():
    host = MemoryHost()
    backend = FakePolicy(slow_s=0.3)
    with LoopbackServer(backend) as server:
        link = LatestLink(server.endpoint, timeout_s=0.8)
        with patch("alohamini.remote.agent.stop_owned_robot") as stop:
            with pytest.raises(RuntimeError, match="No unexpired remote action"):
                run_session(
                    host,
                    link,
                    MODEL,
                    duration_s=1,
                    max_prediction_age_s=0.13,
                    rpc_timeout_s=0.8,
                )
        assert 1 <= len(host.sent) < backend.chunk_size
        stop.assert_called_once_with(host, MODEL, host.sent[-1][2])
        sent_when_stopped = list(host.sent)
        assert backend.slow_finished.wait(0.6)
        assert host.sent == sent_when_stopped, "a late chunk must not revive stopped control"
        assert not link._thread.is_alive()


def test_cloud_disconnect_times_out_and_never_replays_remaining_actions():
    host = MemoryHost()
    backend = FakePolicy()
    with LoopbackServer(backend) as server:
        backend.disconnect = server.stop
        link = LatestLink(server.endpoint, timeout_s=0.12)
        with patch("alohamini.remote.agent.stop_owned_robot") as stop:
            with pytest.raises(RuntimeError, match="Remote inference connection failed"):
                run_session(
                    host,
                    link,
                    MODEL,
                    duration_s=1,
                    max_prediction_age_s=0.9,
                    rpc_timeout_s=0.8,
                )
        assert 1 <= len(host.sent) < backend.chunk_size
        assert link.error is not None
        stop.assert_called_once_with(host, MODEL, host.sent[-1][2])
        assert not link._thread.is_alive()

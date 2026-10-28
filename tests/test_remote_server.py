"""Cloud protocol and absolute chunks without hardware or a deployed model."""

import copy
import socket
import threading
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import zmq
from test_dataset import jpeg
from test_replay import replay_snapshot

from alohamini.datasets.record import state_names
from alohamini.errors import ProtocolError
from alohamini.protocol import HostSnapshot
from alohamini.remote.protocol import context_for, decode_message, encode_message, snapshot_body
from alohamini.remote.server import (
    NativeChunkPolicy,
    _FatalMultipartError,
    _receive_frames,
    _Service,
    serve,
)


def snapshot():
    source = replay_snapshot()
    source.payload["lift_axis.reference_sequence"] = 2
    return HostSnapshot(source.payload, {"forward": jpeg()}, 100.0, 100.02)


def request(source=None, *, session="robot-run", request_id=1, tick=0):
    source = snapshot() if source is None else source
    return {
        "session": session, "id": request_id, "tick": tick, "fps": 30,
        "client_id": "pi-client", "context": context_for(source),
        "snapshot": snapshot_body(source), "fixed": None,
    }, source.images


class Backend:
    fps, chunk_size = 30, 2
    names, cameras, fixed_current = ("joint.pos",), ("forward",), ()

    def __init__(self):
        self.initialize = Mock()
        self.predict = Mock(return_value=torch.tensor([[1.0], [2.0]]))
        self.reset = Mock()


@contextmanager
def running_server(backend, *, timeout=2.0):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        endpoint = f"tcp://127.0.0.1:{reservation.getsockname()[1]}"
    stop, errors = threading.Event(), []

    def run():
        try:
            serve(endpoint, backend, stop, request_timeout_s=timeout, poll_timeout_ms=10)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    context = zmq.Context()
    client = context.socket(zmq.DEALER)
    client.setsockopt(zmq.LINGER, 0)
    client.connect(endpoint)
    try:
        yield client
    finally:
        stop.set()
        client.close(linger=0)
        context.term()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert not errors, errors


def rpc(client, kind, body, images=None):
    client.send_multipart(encode_message(kind, body, images))
    assert client.poll(timeout=2000)
    kind, response, images = decode_message(client.recv_multipart())
    assert images == {}
    return kind, response


def test_server_roundtrip_preserves_pi_clock_jpeg_and_returns_absolute_chunk():
    backend = Backend()
    initial = snapshot()
    hello, images = request(initial)
    with running_server(backend) as client, patch(
        "alohamini.datasets.images.decode_host_image", side_effect=AssertionError("No proxy decode")
    ):
        kind, ready = rpc(client, "hello", hello, images)
        assert kind == "ready"
        assert ready == {
            "session": "robot-run", "id": 1, "context": context_for(initial), "fps": 30,
            "names": ["joint.pos"], "cameras": ["forward"], "chunk_size": 2,
            "fixed_current": [],
        }
        observation, images = request(initial, request_id=3, tick=12)
        kind, actions = rpc(client, "observation", observation, images)
        assert kind == "actions"
        assert actions["actions"] == [[1.0], [2.0]]
        for field in ("session", "id", "tick", "fps", "context"):
            assert actions[field] == observation[field]
        assert actions["inference_s"] >= 0
        received = backend.predict.call_args.args[0]
        assert (received.request_started_s, received.received_s) == (100.0, 100.02)
        assert received.images == initial.images
        assert received.payload == initial.payload
        assert rpc(client, "stop", {"session": "robot-run", "id": 3})[0] == "ack"
        assert rpc(client, "stop", {"session": "robot-run", "id": 3})[0] == "ack"
    backend.reset.assert_called_once()


def test_server_rejects_replayed_ids_and_other_sessions_without_extra_prediction():
    backend = Backend()
    hello, images = request()
    with running_server(backend) as client:
        assert rpc(client, "hello", hello, images)[0] == "ready"
        observation, images = request(request_id=2, tick=1)
        assert rpc(client, "observation", observation, images)[0] == "actions"
        assert rpc(client, "observation", observation, images)[0] == "error"
        other, images = request(session="another-run", request_id=3)
        assert rpc(client, "hello", other, images)[0] == "error"
        assert rpc(client, "observation", other, images)[0] == "error"
        assert rpc(client, "ping", {"session": "robot-run", "id": 4})[0] == "pong"
    backend.predict.assert_called_once()


@pytest.mark.parametrize("change", ["host", "epoch", "lift", "metadata", "fps", "time", "tick"])
def test_changed_context_or_order_does_not_run_inference(change):
    backend = Backend()
    service = _Service(backend, 2.0)
    hello, images = request(tick=5)
    service.handle(b"pi", "hello", hello, images)
    altered = snapshot()
    if change == "host":
        altered.payload["_safety"]["host_session_id"] = "restarted"
    elif change == "epoch":
        altered.payload["_safety"]["control_epoch"] += 1
    elif change == "lift":
        altered.payload["lift_axis.reference_sequence"] += 1
    elif change == "metadata":
        altered.payload["_robot_metadata"]["motors"]["arm_left_gripper"]["range_min"] += 1
    elif change == "time":
        altered = HostSnapshot(altered.payload, altered.images, 90.0, 90.02)
    observation, images = request(altered, request_id=2, tick=4 if change == "tick" else 6)
    if change == "fps":
        observation["fps"] = 20
    with pytest.raises(ProtocolError):
        service.handle(b"pi", "observation", observation, images)
    backend.predict.assert_not_called()
    backend.reset.assert_called_once()


@pytest.mark.parametrize("result", [[], [[1.0]], [[1.0, 2.0], [3.0, 4.0]], [[float("nan")], [1.0]]])
def test_invalid_chunks_are_errors_and_end_session(result):
    backend = Backend()
    backend.predict.return_value = result
    with running_server(backend) as client:
        hello, images = request()
        assert rpc(client, "hello", hello, images)[0] == "ready"
        observation, images = request(request_id=2, tick=1)
        assert rpc(client, "observation", observation, images)[0] == "error"
        assert rpc(client, "observation", {**observation, "id": 3}, images)[0] == "error"
    backend.predict.assert_called_once()
    backend.reset.assert_called_once()


def test_idle_and_inference_timeouts_reset_without_replaying_actions(monkeypatch):
    backend = Backend()
    service = _Service(backend, 0.1)
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr("alohamini.remote.server.time.monotonic", lambda: clock.now)
    hello, images = request()
    service.handle(b"pi", "hello", hello, images)
    clock.now += 0.2
    service.expire()
    assert service.session is None
    backend.reset.assert_called_once()
    next_hello, images = request(session="new-run")
    service.handle(b"pi", "hello", next_hello, images)

    def slow(_snapshot):
        clock.now += 0.2
        return [[1.0], [2.0]]

    backend.predict.side_effect = slow
    observation, images = request(session="new-run", request_id=2, tick=1)
    with pytest.raises(TimeoutError):
        service.handle(b"pi", "observation", observation, images)
    assert service.session is None and backend.reset.call_count == 2


def test_malformed_request_is_reported_and_server_remains_available():
    backend = Backend()
    with running_server(backend) as client:
        client.send_multipart([b"not-json"])
        assert client.poll(timeout=2000)
        kind, error, _ = decode_message(client.recv_multipart())
        assert kind == "error" and error["message"]
        hello, images = request()
        assert rpc(client, "hello", hello, images)[0] == "ready"


def test_oversized_multipart_is_drained_before_the_next_request():
    from alohamini.remote.protocol import MAX_MESSAGE_FRAMES

    with running_server(Backend()) as client:
        client.send_multipart([b"{}"] * (MAX_MESSAGE_FRAMES + 1))
        assert client.poll(timeout=2000)
        assert decode_message(client.recv_multipart())[0] == "error"
        hello, images = request()
        assert rpc(client, "hello", hello, images)[0] == "ready"


def test_multipart_byte_limit_and_excessive_drain_are_bounded(monkeypatch):
    monkeypatch.setattr("alohamini.remote.server.MAX_MESSAGE_BYTES", 5)
    remaining = [b"1234", b"5678"]
    transport = SimpleNamespace(
        getsockopt=lambda _: bool(remaining), recv=Mock(side_effect=lambda **_: remaining.pop(0))
    )
    with pytest.raises(ProtocolError, match="size limit"):
        _receive_frames(transport, zmq)
    assert not remaining and transport.recv.call_count == 2
    monkeypatch.setattr("alohamini.remote.server.MAX_MESSAGE_FRAMES", 2)
    remaining.extend([b"x"] * 100)
    transport.recv.reset_mock()
    with pytest.raises(_FatalMultipartError):
        _receive_frames(transport, zmq)
    assert transport.recv.call_count == 35 and len(remaining) == 65


class RelativeProcessor:
    def action(self, action, *, context):
        return action + context[:, None, :]


def native_policy_stub():
    from alohamini.learning.policy import NativePolicy

    policy = NativePolicy.__new__(NativePolicy)
    policy.names = ["joint.pos", "fixed.pos"]
    policy.fixed_current = []
    policy.fixed_dimensions = {"targets": {"fixed.pos": 7.0}}
    policy.config = SimpleNamespace(inference_action_scale_dims=(0,), inference_action_scale=2.0)
    policy.processor = RelativeProcessor()
    batch = {"_action_context": torch.tensor([[10.0, 20.0]])}
    policy._observation_batch = Mock(return_value=batch)
    policy.inference_context = nullcontext
    policy.model = SimpleNamespace(predict_action_chunk=Mock(return_value=torch.ones(1, 2, 2)))
    return policy, batch


def test_native_chunk_uses_current_context_absolute_conversion_scaling_and_fixed_targets():
    policy, batch = native_policy_stub()
    result = policy.predict_action_chunk(snapshot())
    assert result.device.type == "cpu" and result.shape == (2, 2)
    torch.testing.assert_close(result, torch.tensor([[22.0, 7.0], [22.0, 7.0]]))
    policy.model.predict_action_chunk.assert_called_once_with(batch)


def test_execution_window_is_trimmed_before_absolute_decoding_and_validity_checks():
    policy, _ = native_policy_stub()
    policy.model.predict_action_chunk.return_value[0, 1] = float("inf")
    result = policy.predict_action_chunk(snapshot(), max_steps=1)
    torch.testing.assert_close(result, torch.tensor([[22.0, 7.0]]))
    for invalid in (0, -1, True, 3):
        with pytest.raises(ValueError, match="Execution window"):
            policy.predict_action_chunk(snapshot(), max_steps=invalid)


def test_live_batch_keeps_rgb_task_and_shared_preprocessing_without_mutating_snapshot():
    from alohamini.learning.policy import NativePolicy

    policy = NativePolicy.__new__(NativePolicy)
    policy._check_calibration = Mock()
    policy.selection, policy.fixed_current, policy.fixed_state_references = None, [], {}
    policy.manifest = {"cameras": ["forward"], "image_size": [8, 8]}
    policy.algorithm, policy.task = SimpleNamespace(include_task=True), "move the block"
    policy.processor = lambda batch: batch
    current = snapshot()
    original = copy.deepcopy(current)
    batch = policy._observation_batch(current)
    assert batch["task"] == ["move the block"]
    image = batch["observation.images.forward"]
    assert image.shape == (1, 3, 8, 8) and image.dtype == torch.float32
    assert image[0, 0].mean() > image[0, 2].mean()
    assert current.payload == original.payload and current.images == original.images


@pytest.mark.parametrize("shape", [(2, 2), (2, 1, 2), (1, 0, 2), (1, 2, 0)])
def test_native_chunk_rejects_invalid_batch_time_or_coordinate_dimensions(shape):
    policy, _ = native_policy_stub()
    policy.model.predict_action_chunk.return_value = torch.zeros(shape)
    with pytest.raises(ValueError, match="invalid action chunk"):
        policy.predict_action_chunk(snapshot())


def test_native_chunk_rejects_nonfinite_values():
    policy, _ = native_policy_stub()
    policy.model.predict_action_chunk.return_value[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="invalid absolute action chunk"):
        policy.predict_action_chunk(snapshot())


def test_pi05_padded_model_chunk_restores_18_absolute_coordinates_from_original_state():
    from test_pi05 import processor

    from alohamini.policies.pi05.adapter import PI05BatchProcessor

    policy, _ = native_policy_stub()
    policy.names = state_names("alohamini2pro")
    policy.fixed_dimensions = {"targets": {"lift_axis.height_mm": 117.0}}
    actual_processor = PI05BatchProcessor.__new__(PI05BatchProcessor)
    actual_processor.transform = processor()
    policy.processor = actual_processor
    state = torch.arange(18.0)[None]
    batch = {"_action_context": state}
    policy._observation_batch.return_value = batch
    raw = torch.zeros(1, 2, 32)
    raw[..., 18:] = 123.0  # Model padding is not part of the physical Host action vector.
    policy.model.predict_action_chunk.return_value = raw
    expected = actual_processor.transform.restore_actions(raw, state)[0]
    expected[:, 0] *= 2.0
    expected[:, -1] = 117.0
    result = policy.predict_action_chunk(snapshot())
    assert result.shape == (2, 18) and result.device.type == "cpu"
    torch.testing.assert_close(result, expected)
    raw[0, 0, 31] = float("nan")
    with pytest.raises(ValueError, match="invalid absolute action chunk"):
        policy.predict_action_chunk(snapshot())


def test_remote_backend_rejects_non_pi05_before_loading_model(monkeypatch):
    monkeypatch.setattr(
        "alohamini.learning.checkpoint.read_checkpoint", lambda _: (None, {"kind": "act"})
    )
    constructor = Mock(side_effect=AssertionError("Model must not load"))
    monkeypatch.setattr("alohamini.learning.policy.NativePolicy", constructor)
    with pytest.raises(ValueError, match="pi05, smolvla and spirit"):
        NativeChunkPolicy("checkpoint")
    constructor.assert_not_called()


@pytest.mark.parametrize("kind", ["pi05", "smolvla", "spirit"])
def test_backend_pi_capture_does_not_compare_pi_timestamps_to_cloud_clock(monkeypatch, kind):
    live = snapshot()
    policy = SimpleNamespace(
        config=SimpleNamespace(network={"action_horizon": 5}, chunk_size=5, n_action_steps=4),
        names=state_names(live.robot_model), manifest={"cameras": ["forward"]}, fps=30,
        fixed_current=["lift_axis.height_mm"], fixed_dimensions=None,
        fixed_deployment={}, reset=Mock(), bind_robot=Mock(),
        predict_action_chunk=Mock(return_value=torch.zeros(5, 18)),
    )
    monkeypatch.setattr(
        "alohamini.learning.checkpoint.read_checkpoint", lambda _: (None, {"kind": kind})
    )
    monkeypatch.setattr("alohamini.learning.policy.NativePolicy", Mock(return_value=policy))
    monkeypatch.setattr(
        "alohamini.client.control_feedback_valid", Mock(side_effect=AssertionError("Pi clock only"))
    )
    backend = NativeChunkPolicy("checkpoint", chunk_size=3, fixed_current="lift_axis")
    fixed = {"version": 1, "targets": {"lift_axis.height_mm": live.payload["lift_axis.height_mm"]}}
    backend.initialize(live, "pi-client", 30, fixed=fixed)
    assert policy._fixed_current_bound
    capture = policy.fixed_deployment["current_capture"]
    assert (capture["request_started_s"], capture["received_s"]) == (100.0, 100.02)
    assert backend.predict(live).shape == (3, 18)
    policy.predict_action_chunk.assert_called_once_with(live, max_steps=3)
    with pytest.raises(ValueError, match="exactly match"):
        backend.initialize(live, "pi-client", 20, fixed=fixed)
    with pytest.raises(ValueError, match="selection differs"):
        backend.initialize(live, "pi-client", 30, fixed=None)
    wrong = copy.deepcopy(fixed)
    wrong["targets"]["lift_axis.height_mm"] += 1
    with pytest.raises(ValueError, match="differs from its initialization snapshot"):
        backend.initialize(live, "pi-client", 30, fixed=wrong)
    with pytest.raises(ValueError, match="execution horizon"):
        NativeChunkPolicy("checkpoint", chunk_size=5)
    with pytest.raises(ValueError, match=r"\[1, 512\]"):
        NativeChunkPolicy("checkpoint", chunk_size=513)

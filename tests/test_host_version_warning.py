"""Optional source identity diagnostics must never authorize or reject control."""

import logging
from collections import deque
from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import pytest
from support import state_frames, state_payload

from alohamini.client import HostClient
from alohamini.errors import ProtocolError


class ReplySocket:
    """One deterministic reply per request, with no TCP or physical Host."""

    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.parts = deque()
        self.sent = []

    def send(self, token, flags=0):
        assert not self.parts
        self.sent.append(token)
        self.parts.extend(state_frames(token, next(self.payloads)))

    def recv(self, flags=0):
        return self.parts.popleft()

    def poll(self, timeout, event):
        import zmq

        return True if event == zmq.POLLOUT else bool(self.parts)

    def getsockopt(self, option):
        import zmq

        assert option == zmq.RCVMORE
        return bool(self.parts)

    def close(self, **kwargs):
        pass


def payload(software=None, *, session="host-one"):
    result = state_payload()
    result["_safety"].update(host_session_id=session, control_epoch=0)
    if software is not None:
        result["_software"] = deepcopy(software)
    return result


@pytest.fixture
def client_factory(monkeypatch):
    clients = []

    def create(payloads):
        client = HostClient("in-memory", expected_model="alohamini2pro", request_window=1)
        client._socket = ReplySocket(payloads)
        monkeypatch.setattr(client, "_connect", lambda zmq: None)
        monkeypatch.setattr(
            client, "_connect_commands", Mock(side_effect=AssertionError("no command channel"))
        )
        clients.append(client)
        return client

    yield create
    for client in clients:
        client.close()


@pytest.fixture
def local_identity(monkeypatch):
    identity = {
        "schema_version": 1,
        "package_version": "0.9.0",
        "source_sha256": "a" * 64,
        "git_revision": "b" * 40,
    }
    monkeypatch.setattr("alohamini.client.software_identity", lambda: deepcopy(identity))
    return identity


def diagnostic_messages(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "alohamini.client" and record.levelno >= logging.WARNING
    ]


@pytest.mark.parametrize("different_history", [False, True])
def test_equal_sources_do_not_warn_even_when_version_or_git_history_differ(
    local_identity, client_factory, caplog, different_history
):
    remote = deepcopy(local_identity)
    if different_history:
        remote.update(package_version="different-build", git_revision="c" * 40)
    client = client_factory([payload(remote), payload(remote)])
    for _ in range(2):
        snapshot = client.read()
        assert snapshot.payload["_software"] == remote
        assert snapshot.robot_model == "alohamini2pro"
        assert snapshot.payload["arm_left_shoulder_pan.pos"] == -12.5
    assert not diagnostic_messages(caplog)
    assert client._command_socket is None


def test_different_source_warns_once_and_preserves_snapshot_and_command_context(
    local_identity, client_factory, caplog
):
    remote = {**local_identity, "source_sha256": "d" * 64}
    client = client_factory([payload(remote)] * 3)
    snapshots = [client.read() for _ in range(3)]
    messages = diagnostic_messages(caplog)
    assert len(messages) == 1
    assert "pc/host code mismatch" in messages[0].lower()
    assert all(snapshot.payload["_software"] == remote for snapshot in snapshots)
    assert all(snapshot.payload["_safety"]["control_epoch"] == 0 for snapshot in snapshots)
    assert client._command_context.host_session_id == "host-one"
    assert client._command_socket is None


def test_old_host_without_identity_warns_once_and_remains_readable(
    local_identity, client_factory, caplog
):
    client = client_factory([payload(), payload()])
    assert client.read().robot_model == "alohamini2pro"
    assert client.read().payload["lift_axis.height_mm"] == 120.0
    messages = diagnostic_messages(caplog)
    assert len(messages) == 1
    assert "cannot verify" in messages[0].lower()


@pytest.mark.parametrize(
    "remote",
    [
        {"schema_version": 1, "source_sha256": None},
        {"schema_version": 1, "source_sha256": "not-a-source-digest"},
        {"schema_version": 2, "source_sha256": "a" * 64},
        {"schema_version": True, "source_sha256": "a" * 64},
        {"schema_version": 1.0, "source_sha256": "a" * 64},
        "old-unstructured-version",
    ],
)
def test_unverifiable_optional_identity_does_not_become_a_protocol_requirement(
    local_identity, client_factory, caplog, remote
):
    client = client_factory([payload(remote), payload(remote)])
    for _ in range(2):
        assert client.read().payload["_software"] == remote
    messages = diagnostic_messages(caplog)
    assert len(messages) == 1
    assert "cannot verify" in messages[0].lower()


@pytest.mark.parametrize("remote_known", [True, False])
def test_unavailable_local_source_is_not_treated_as_a_matching_hash(
    local_identity, client_factory, caplog, remote_known
):
    remote = deepcopy(local_identity)
    local_identity["source_sha256"] = None
    if not remote_known:
        remote["source_sha256"] = None
    client = client_factory([payload(remote), payload(remote)])
    assert client.read().payload["_software"] == remote
    assert client.read().payload["_software"] == remote
    messages = diagnostic_messages(caplog)
    assert len(messages) == 1
    assert "cannot verify" in messages[0].lower()


def test_new_host_session_rechecks_sources_without_repeating_for_each_snapshot(
    local_identity, client_factory, caplog
):
    mismatch = {**local_identity, "source_sha256": "d" * 64}
    sequence = [
        payload(local_identity, session="same-source"),
        payload(mismatch, session="mismatch-one"),
        payload(mismatch, session="mismatch-one"),
        payload(local_identity, session="matching-again"),
        payload(mismatch, session="mismatch-two"),
    ]
    client = client_factory(sequence)
    for expected in sequence:
        assert client.read().payload == expected
    assert len(diagnostic_messages(caplog)) == 2
    assert client._command_context.host_session_id == "mismatch-two"


def test_invalid_reply_is_rejected_before_version_diagnostics_and_next_valid_reply_is_checked(
    local_identity, client_factory, caplog
):
    mismatch = {**local_identity, "source_sha256": "d" * 64}
    invalid = payload(mismatch)
    invalid["_robot_metadata"]["schema_version"] = 99
    client = client_factory([invalid])
    with pytest.raises(ProtocolError, match="schema_version"):
        client.read()
    assert not diagnostic_messages(caplog)
    # Protocol failure intentionally discards the transport; simulate a new one.
    client._socket = ReplySocket([payload(mismatch)])
    assert client.read().payload["_software"] == mismatch
    assert len(diagnostic_messages(caplog)) == 1


def test_host_publishes_one_startup_identity_without_changing_robot_metadata(monkeypatch):
    from test_host import host_fixture

    from alohamini.runtime.lifecycle import CycleResult, HostPhase

    identity = {
        "schema_version": 1,
        "package_version": "0.9.0",
        "source_sha256": "a" * 64,
        "git_revision": "b" * 40,
    }
    identify = Mock(return_value=deepcopy(identity))
    monkeypatch.setattr("alohamini.runtime.host.software_identity", identify)
    host, _ = host_fixture()
    metadata = deepcopy(host._metadata)
    status = replace(host.supervisor.status, phase=HostPhase.READY)
    first = host._payload(CycleResult((), False, status))
    second = host._payload(CycleResult((), False, status))
    assert first["_software"] == second["_software"] == identity
    assert first["_robot_metadata"] == metadata
    assert "_software" not in first["_robot_metadata"]
    assert identify.call_count == 1

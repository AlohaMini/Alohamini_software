"""RTC negotiation and accepted-plan alignment, without a robot or GPU service."""

from unittest.mock import Mock, patch

import pytest
from test_remote_agent import MODEL, FakeClock, FakeHost, FakeLink
from test_remote_server import Backend, request

from alohamini.errors import ProtocolError
from alohamini.remote.agent import run_session
from alohamini.remote.server import _Service
from alohamini.rtc import RTCConfig


def service():
    backend = Backend()
    backend.chunk_size = 8
    backend.rtc = RTCConfig(enabled=True, replan_steps=2, inference_delay_steps=1).to_dict()
    backend.predict_rtc = Mock(return_value=[[float(i)] for i in range(8)])
    server = _Service(backend, 5.0)
    body, images = request()
    kind, ready = server.handle(b"pi", "hello", body, images)
    assert kind == "ready" and ready["rtc"] == backend.rtc
    return server, backend


def observe(server, request_id, tick, plan_id=None, delay=1):
    body, images = request(request_id=request_id, tick=tick)
    body["rtc"] = dict(plan_id=plan_id, inference_delay=delay)
    return server.handle(b"pi", "observation", body, images)


def test_cloud_guides_from_accepted_plan_even_if_a_newer_response_was_not_accepted():
    server, backend = service()
    observe(server, 2, 0)
    assert backend.predict_rtc.call_args.args[1:] == (None, 1)
    observe(server, 3, 2, plan_id=2)
    assert backend.predict_rtc.call_args.args[1:] == ([[float(i)] for i in range(2, 8)], 1)
    backend.predict_rtc.return_value = [[100.0 + i] for i in range(8)]
    observe(server, 4, 4, plan_id=2)  # Plan 3 was never accepted by the Pi.
    assert backend.predict_rtc.call_args.args[1:] == ([[float(i)] for i in range(4, 8)], 1)
    assert backend.predict.call_count == 0
    server.end()
    assert not server.plans


@pytest.mark.parametrize("plan_id,tick,delay", [(999, 2, 1), (2, 0, 1), (2, 7, 1), (None, 2, 1)])
def test_invalid_rtc_prefix_closes_session_before_sampling(plan_id, tick, delay):
    server, backend = service()
    observe(server, 2, 0)
    with pytest.raises(ProtocolError):
        observe(server, 3, tick, plan_id=plan_id, delay=delay)
    assert backend.predict_rtc.call_count == 1
    assert server.session is None and not server.plans


class RTCLink(FakeLink):
    def submit(self, kind, body, images=None):
        super().submit(kind, body, images)
        if kind == "hello":
            self.pending[-1][1]["rtc"] = RTCConfig(
                enabled=True, replan_steps=3, inference_delay_steps=1,
            ).to_dict()


def test_pi_replans_at_rtc_interval_with_last_accepted_plan_and_measured_delay():
    clock, events = FakeClock(), []
    host, link = FakeHost(clock, events), RTCLink(clock, events)
    with (
        patch("alohamini.remote.agent.stop_owned_robot") as stop,
        patch("alohamini.client.time.monotonic", clock),
    ):
        stats = run_session(
            host, link, MODEL, fps=10, duration_s=0.95,
            clock=clock, sleep=clock.sleep,
        )
    assert stats["sent"] >= 8
    observations = link.observations
    assert 3 <= len(observations) <= 4
    assert observations[0]["rtc"]["plan_id"] is None
    for old, new in zip(observations[:-1], observations[1:], strict=True):
        assert new["tick"] - old["tick"] >= 3
        assert new["rtc"]["plan_id"] == old["id"]
        assert new["rtc"]["inference_delay"] >= 2
    stop.assert_called_once()
    assert link.closed


def test_cloud_cli_enables_runtime_rtc_and_leaves_training_overrides_separate():
    from alohamini.remote.cli import main

    with (
        patch("alohamini.remote.server.NativeChunkPolicy") as policy,
        patch("alohamini.remote.server.serve"),
    ):
        assert main([
            "serve", "--policy.path", "/model", "--rtc", "--rtc.replan_steps", "6",
        ]) == 0
    assert policy.call_args.kwargs["rtc"].replan_steps == 6
    assert policy.call_args.kwargs["rtc"].enabled
    assert "rtc" not in policy.call_args.kwargs["overrides"]

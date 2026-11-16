"""Accepted-plan prefixes and B-spline transport across independent machine clocks."""

import math
from unittest.mock import Mock, patch

import pytest
from test_remote_server import Backend, request, snapshot

from alohamini.errors import ProtocolError
from alohamini.protocol import HostSnapshot
from alohamini.remote.agent import PlanBuffer
from alohamini.remote.server import _Service
from alohamini.trajectory import BSplineActionPlan, DenseActionPlan
from alohamini.trajectory_execution import ExecutionConfig, PlanCapabilities, prediction_timing


def service():
    backend = Backend()
    backend.fps = 10
    backend.execution = ExecutionConfig(enabled=True, continuity="rtc").to_dict()
    backend.predict_plan = Mock(
        return_value=DenseActionPlan(("joint.pos",), [[i] for i in range(8)], 0.1)
    )
    server = _Service(backend, 5.0)
    body, images = request()
    body["fps"] = 10
    server.handle(b"pi", "hello", body, images)
    return server, backend


def observe(server, request_id, tick, plan_id=None, delay=0.1):
    source = snapshot()
    source = HostSnapshot(source.payload, source.images, 100 + tick / 10, 100 + tick / 10 + 0.02)
    body, images = request(source, request_id=request_id, tick=tick)
    body.update(fps=10, plan_id=plan_id, inference_delay_s=delay)
    return server.handle(b"pi", "observation", body, images)


def test_cloud_conditions_on_the_accepted_plan_not_the_newest_generated_plan():
    server, backend = service()
    observe(server, 2, 0)
    assert backend.predict_plan.call_args.args[1] is None
    observe(server, 3, 2, plan_id=2)
    assert backend.predict_plan.call_args.args[1].actions[0] == (2.0,)
    backend.predict_plan.return_value = DenseActionPlan(
        ("joint.pos",), [[100 + i] for i in range(8)], 0.1
    )
    observe(server, 4, 4, plan_id=2)
    assert backend.predict_plan.call_args.args[1].actions[0] == (4.0,)
    server.end()
    assert not server.plans


@pytest.mark.parametrize(
    "plan_id,tick,delay", [(999, 2, 0.1), (2, 0, 0.1), (2, 7, 0.2), (None, 2, 0.1)]
)
def test_invalid_prefix_is_rejected_before_sampling(plan_id, tick, delay):
    server, backend = service()
    observe(server, 2, 0)
    with pytest.raises(ProtocolError):
        observe(server, 3, tick, plan_id, delay)
    assert backend.predict_plan.call_count == 1 and server.session is None


def test_remote_transmits_spline_parameters_and_pi_queries_its_own_clock():
    server, backend = service()
    backend.plan_capabilities = PlanCapabilities(kinds=("bspline",))
    backend.execution = ExecutionConfig(enabled=True).to_dict()
    backend.predict_plan.return_value = BSplineActionPlan(
        ("joint.pos",), [0, 0, 0, 0, 1, 1, 1, 1], [[0], [1], [2], [3]], 3
    )
    kind, body = observe(server, 2, 0)
    assert kind == "plan" and body["plan"]["kind"] == "bspline"
    assert "actions" not in body
    buffer = PlanBuffer(body["session"], body["context"], body["names"], 10, 2, ["bspline"])
    buffer.remember(2, 0, 7000.0)  # Pi clock origin differs from the server clock.
    assert buffer.accept(body, now=7000.2, tick=2) > 0
    assert buffer.pop(2, 7000.2)["joint.pos"] == pytest.approx(0.6)


def test_cloud_cli_uses_generic_execution_options():
    from alohamini.remote.cli import main

    with (
        patch("alohamini.remote.server.NativePlanPolicy") as policy,
        patch("alohamini.remote.server.serve"),
    ):
        assert (
            main(
                [
                    "serve",
                    "--policy.path",
                    "/model",
                    "--inference.type=rtc",
                    "--inference.replan_interval_s",
                    "0.2",
                    "--policy.time_scale",
                    "1.2",
                    "--policy.temporal_ensemble_coeff",
                    "none",
                ]
            )
            == 0
        )
    config = policy.call_args.kwargs["execution"]
    assert config.replan_interval_s == 0.2 and config.continuity == "rtc"
    assert "execution" not in policy.call_args.kwargs["overrides"]
    assert policy.call_args.kwargs["overrides"]["time_scale"] == 1.2
    assert policy.call_args.kwargs["overrides"]["temporal_ensemble_coeff"] is None


def test_pi_retains_old_plan_until_a_future_spline_domain_starts():
    buffer = PlanBuffer("run", {}, ["q"], 10, 2, ["dense", "bspline"])

    def response(number, tick, plan):
        return dict(id=number, session="run", context={}, tick=tick, fps=10,
                    names=["q"], plan=plan.to_dict(), inference_s=0.01)

    old = DenseActionPlan(("q",), [[i] for i in range(10)], 0.1)
    buffer.remember(1, 0, 100)
    buffer.accept(response(1, 0, old), now=100.1, tick=1)
    new = BSplineActionPlan(("q",), [.3]*4+[1.3]*4, [[20], [21], [22], [23]])
    buffer.remember(2, 2, 100.2)
    buffer.accept(response(2, 2, new), now=100.3, tick=3)
    assert buffer.pop(3, 100.3) == {"q": 3.0}
    assert buffer.accepted_plan_id == 1
    assert buffer.pop(5, 100.5)["q"] == pytest.approx(20)
    assert buffer.accepted_plan_id == 2


@pytest.mark.parametrize("strategy", ["rtc", "phase"])
@pytest.mark.parametrize("control_fps", [10, 30])
def test_default_remote_continuity_handover_keeps_a_usable_prefix(strategy, control_fps):
    from test_trajectory import continuity_policy

    policy = continuity_policy(strategy, names=("joint.pos",), fps=10)
    settings = ExecutionConfig(enabled=True, continuity=strategy)
    backend = Backend()
    backend.fps = control_fps
    backend.execution = settings.to_dict()
    backend.plan_capabilities = policy.plan_capabilities
    backend.predict_plan = lambda source, prefix=None: policy.predict_plan(
        source, prefix=prefix, continuity=strategy
    )
    server = _Service(backend, 5.0)
    hello, images = request()
    hello["fps"] = control_fps
    _, ready = server.handle(b"pi", "hello", hello, images)
    buffer = PlanBuffer(
        ready["session"], ready["context"], ready["names"], control_fps, 5,
        ready["plan_kinds"],
    )

    def predict(request_id, sampled_s, *, plan_id=None, delay=0.0):
        initial = snapshot()
        source = HostSnapshot(initial.payload, initial.images, sampled_s, sampled_s + 0.02)
        tick = max(0, math.floor((sampled_s - 100) * control_fps))
        body, frames = request(source, request_id=request_id, tick=tick)
        body.update(fps=control_fps, plan_id=plan_id, inference_delay_s=delay)
        buffer.remember(request_id, tick, sampled_s)
        kind, response = server.handle(b"pi", "observation", body, frames)
        assert kind == "plan"
        return response

    try:
        response = predict(2, 100.0)
        buffer.accept(response, now=100.1, tick=math.floor(0.1 * control_fps))
        observation_age_s = 0.2
        for tick in range(1, control_fps):
            now = 100.1 + tick / control_fps + 1e-6
            delay, lead = prediction_timing(
                settings, buffer.plan, fps=control_fps,
                latency_s=buffer.last_round_trip_s, observation_age_s=observation_age_s,
            )
            if buffer.remaining_s(now) <= lead:
                break
        sampled_s = now - observation_age_s
        response = predict(3, sampled_s, plan_id=buffer.accepted_plan_id, delay=delay)
        prefix = policy.predict_plan.call_args.kwargs["prefix"]
        assert prefix.inference_delay_s >= observation_age_s + 1 / control_fps - 1e-9
        assert prefix.times_s[-1] >= prefix.inference_delay_s
        accepted_at = now + 0.1
        tick = math.floor((accepted_at - 100) * control_fps)
        assert buffer.accept(response, now=accepted_at, tick=tick) > 0
        assert buffer.pop(tick, accepted_at) is not None
        assert buffer.accepted_plan_id == 3
        assert policy.predict_plan.call_count == 2
    finally:
        server.end()

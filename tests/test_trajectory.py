"""Physical-time plan fidelity and execution, independent of a robot or GPU."""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.interpolate import BSpline

from alohamini.learning.trajectory_execution import TrajectoryExecutor
from alohamini.trajectory import BSplineActionPlan, DenseActionPlan, plan_from_dict
from alohamini.trajectory_execution import ExecutionConfig, PlanCapabilities


def test_bspline_matches_scipy_with_nonuniform_repeated_knots_and_two_arms():
    knots = [0, 0, 0, 0, 0.2, 0.9, 1, 1, 1, 1]
    points = np.random.default_rng(7).normal(size=(6, 14))
    plan = BSplineActionPlan(tuple(f"joint{i}" for i in range(14)), knots, points)
    query = np.linspace(0, 1, 101, endpoint=False)
    np.testing.assert_allclose(
        [plan.action_at(float(t)) for t in query], BSpline(knots, points, 3)(query), atol=1e-12
    )
    assert plan_from_dict(plan.to_dict()).to_dict() == plan.to_dict()


def test_dense_sample_hold_does_not_interpolate_or_repeat_expired_actions():
    plan = DenseActionPlan(("q", "gripper"), [[0, 1], [5, -1]], 0.1)
    assert plan.action_at(0.099) == (0, 1)
    assert plan.action_at(0.1) == (5, -1)
    with pytest.raises(ValueError, match="outside"):
        plan.action_at(0.2)
    with pytest.raises(ValueError, match="layout"):
        plan_from_dict(plan.to_dict(), names=["gripper", "q"])


class Worker:
    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        future = Future()
        self.jobs.append((future, fn, args))
        return future

    def complete(self):
        future, fn, args = self.jobs[-1]
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)

    def shutdown(self, **kwargs):
        pass


def continuity_policy(strategy, *, names=("q",), fps=10):
    """Exercise the real continuity adapters with deterministic generated plans."""
    import torch

    from alohamini.policies.bspline.codec import align_phase
    from alohamini.policies.continuity import dense_flow_arguments

    if strategy == "rtc":
        plan = DenseActionPlan(names, [[i] * len(names) for i in range(fps)], 1 / fps)
    else:
        plan = BSplineActionPlan(names, [0] * 4 + [1] * 4, [[i] * len(names) for i in range(4)])
    policy = SimpleNamespace(
        names=list(names),
        fps=fps,
        task="pick",
        reset=Mock(),
        config=SimpleNamespace(),
        model=SimpleNamespace(parameters=lambda: iter([torch.zeros(1)])),
        processor=SimpleNamespace(action_prefix=lambda values, context: values),
        plan_capabilities=PlanCapabilities(
            kinds=(plan.to_dict()["kind"],),
            continuity_methods=("none", strategy),
            default_continuity=strategy,
        ),
    )

    def predict(snapshot, *, prefix=None, continuity="none"):
        assert continuity == strategy
        if strategy == "rtc":
            dense_flow_arguments(policy, {}, prefix, continuity)
            return plan
        return align_phase(plan, prefix) if prefix is not None else plan

    policy.predict_plan = Mock(side_effect=predict)
    return policy


@pytest.mark.parametrize("strategy", ["rtc", "phase"])
@pytest.mark.parametrize("control_fps", [10, 30])
@pytest.mark.parametrize("observation_age_s", [0.0, 0.25])
def test_default_continuity_replans_with_a_usable_prefix(
    strategy, control_fps, observation_age_s
):
    clock = SimpleNamespace(now=100.0)
    worker = Worker()
    policy = continuity_policy(strategy, fps=10)
    executor = TrajectoryExecutor(
        policy,
        ExecutionConfig(enabled=True, continuity="rtc" if strategy == "rtc" else "auto"),
        clock=lambda: clock.now,
        executor=worker,
        control_fps=control_fps,
    )
    try:
        assert executor.select_action(SimpleNamespace(request_started_s=100.0)) is None
        clock.now = 100.1
        worker.complete()
        assert executor.select_action(SimpleNamespace(request_started_s=clock.now)) is not None
        for tick in range(1, control_fps):
            clock.now = 100.1 + tick / control_fps + 1e-6
            executor.select_action(
                SimpleNamespace(request_started_s=clock.now - observation_age_s)
            )
            if len(worker.jobs) == 2:
                break
        assert len(worker.jobs) == 2
        prefix = worker.jobs[-1][2][1]
        assert prefix.inference_delay_s >= observation_age_s + 1 / control_fps - 1e-9
        assert prefix.times_s[-1] >= prefix.inference_delay_s
        second_sampled_s = executor.pending[1]
        clock.now += 0.1
        worker.complete()  # Calls dense_flow_arguments or align_phase, including strict checks.
        assert executor.select_action(
            SimpleNamespace(request_started_s=clock.now - observation_age_s)
        ) is not None
        assert executor.plan_sampled_s == second_sampled_s
        assert policy.predict_plan.call_count == 2
    finally:
        executor.close()


def setup_executor():
    clock = SimpleNamespace(now=100.0)
    worker = Worker()
    plan = DenseActionPlan(("q",), [[i] for i in range(10)], 0.1)
    policy = SimpleNamespace(
        names=["q"],
        fps=10,
        task="pick",
        reset=Mock(),
        plan_capabilities=PlanCapabilities(kinds=("dense", "bspline")),
        predict_plan=Mock(return_value=plan),
    )
    executor = TrajectoryExecutor(
        policy,
        ExecutionConfig(enabled=True, replan_interval_s=0.5),
        clock=lambda: clock.now,
        executor=worker,
    )
    return executor, policy, clock, worker


def test_delayed_predictions_are_queried_at_execution_time_and_prefixes_are_actions():
    executor, policy, clock, worker = setup_executor()

    def snapshot():
        return SimpleNamespace(request_started_s=clock.now)

    assert executor.select_action(snapshot()) is None
    clock.now = 100.2
    worker.complete()
    assert executor.select_action(snapshot()) == {"q": 2}
    clock.now = 100.5
    executor.select_action(snapshot())
    assert len(worker.jobs) == 2
    prefix = worker.jobs[-1][2][1]
    assert prefix.actions[0] == (2.0,)  # Launched early to cover measured latency.
    assert prefix.times_s[0] == 0
    worker.complete()
    assert policy.reset.call_count == 1  # Preserve multi-observation model history.
    executor.close()


def test_reset_discards_inflight_plan_and_resets_model_in_worker_only():
    executor, policy, clock, worker = setup_executor()
    snapshot = SimpleNamespace(request_started_s=100.0)
    executor.select_action(snapshot)
    executor.reset()
    policy.reset.assert_not_called()
    clock.now = 100.1
    worker.complete()
    assert executor.select_action(SimpleNamespace(request_started_s=100.1)) is None
    assert executor.plan is None and len(worker.jobs) == 2
    worker.complete()
    assert policy.reset.call_count == 2
    executor.close()


def test_expired_plan_and_unsupported_continuity_are_rejected():
    executor, policy, clock, worker = setup_executor()
    executor.select_action(SimpleNamespace(request_started_s=100.0))
    clock.now = 101.1
    worker.complete()
    with pytest.raises(RuntimeError, match="expired"):
        executor.select_action(SimpleNamespace(request_started_s=101.1))
    with pytest.raises(ValueError, match="does not support"):
        TrajectoryExecutor(policy, ExecutionConfig(enabled=True, continuity="rtc"))
    executor.close()


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_bsp_encoding_and_execution_keep_nonuniform_time_and_physical_units(degree):
    from alohamini.policies.bspline.codec import encode_sample, fit_episode, parameter_plan

    time = np.arange(50.0)
    actions = np.column_stack([np.sin(time / 10), time * 0.02] * 7)
    chunks = fit_episode(actions, degree=degree, parameter_rows=16, max_error=0.005)
    parameters = encode_sample(chunks, 0, degree)
    plan = parameter_plan(parameters, [f"q{i}" for i in range(14)], degree=degree, fps=10)
    assert plan.start_s == 0 and plan.end_s > 0
    query = np.arange(0, min(4.9, plan.end_s), 0.1)
    np.testing.assert_allclose(
        [plan.action_at(float(t)) for t in query],
        actions[(query * 10).round().astype(int)],
        atol=0.0051,
    )


def test_future_plan_keeps_executing_old_commands_until_its_domain_starts():
    executor, policy, clock, worker = setup_executor()

    def snapshot():
        return SimpleNamespace(request_started_s=clock.now)

    executor.select_action(snapshot())
    clock.now = 100.1
    worker.complete()
    assert executor.select_action(snapshot()) == {"q": 1}
    clock.now = 100.5
    executor.select_action(snapshot())
    policy.predict_plan.return_value = DenseActionPlan(("q",), [[20], [21]], 0.1, start_s=0.4)
    clock.now = 100.55
    worker.complete()
    assert executor.select_action(snapshot()) is None  # Same control tick, no repeated command.
    clock.now = 100.65
    assert executor.select_action(snapshot()) == {"q": 6}
    clock.now = 100.91
    assert executor.select_action(snapshot()) == {"q": 20}
    executor.close()


def test_phase_alignment_matches_decoded_command_at_the_handover_time():
    from alohamini.policies.bspline.codec import align_phase
    from alohamini.trajectory import ActionPrefix

    plan = BSplineActionPlan(("q",), [0, 0, 0, 0, 1, 1, 1, 1], [[0], [1], [2], [3]])
    prefix = ActionPrefix(("q",), (0.0, 0.1, 0.2, 0.3), [[0.6]] * 4, 0.2)
    aligned = align_phase(plan, prefix)
    assert aligned.action_at(0.2)[0] == pytest.approx(0.6, abs=1e-5)

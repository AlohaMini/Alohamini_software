from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_kinematic_fields import descriptor, joint_state

from alohamini.datasets.field_transforms import FieldTransforms
from alohamini.datasets.kinematic_fields import KinematicFields
from alohamini.kinematics.arm import SerialArmKinematics
from alohamini.learning.data_fields import FieldProcessor
from alohamini.learning.policy import NativePolicy
from alohamini.learning.trajectory_execution import TrajectoryExecutor
from alohamini.remote.agent import PlanBuffer
from alohamini.tcp_execution import TCPTracker
from alohamini.tcp_trajectory import TCPActionPlan
from alohamini.trajectory import plan_from_dict
from alohamini.trajectory_execution import ExecutionConfig, PlanCapabilities


def plan_for(state=None, count=60):
    state = joint_state() if state is None else state
    fields = KinematicFields(descriptor())
    actions = np.repeat(state[None], count, axis=0)
    actions[:, 0] += np.linspace(0, 0.05, count)
    actions[:, 7] -= np.linspace(0, 0.05, count)
    actions[:, [6, 13]] = [20, 80]
    return TCPActionPlan(tuple(fields.names), fields.absolute(actions), 1 / 30, descriptor())


def snapshot(state, names):
    return SimpleNamespace(payload=dict(zip(names, state, strict=True)))


def test_reference_interpolation_uses_rotations_and_never_solves_future_ik(monkeypatch):
    fields = KinematicFields(descriptor())
    actions = fields.absolute(np.repeat(joint_state()[None], 3, axis=0))
    actions[0, 3:6], actions[1, 3:6] = [0, 0, np.deg2rad(179)], [0, 0, np.deg2rad(-179)]
    actions[2, 0] += 100  # This far-future target must not prevent early execution.
    monkeypatch.setattr(SerialArmKinematics, "inverse", lambda *a, **k: pytest.fail("eager IK"))
    plan = TCPActionPlan(tuple(fields.names), actions, 0.1, descriptor())
    restored = plan_from_dict(plan.to_dict(), names=fields.names)
    reference = restored.reference_at(0.05)
    rotation = fields.rotation.decode(reference[3:6])
    np.testing.assert_allclose(rotation @ [1, 0, 0], [-1, 0, 0], atol=1e-10)
    with pytest.raises(ValueError, match="feedback tracking"):
        restored.action_at(0.05)


def test_policy_transmits_absolute_tcp_references_without_ik(monkeypatch):
    fields = FieldTransforms(
        dict(version=1, action_representation="tcp_delta", kinematic_fields=descriptor())
    )
    state = joint_state()
    actions = np.repeat(state[None], 5, axis=0)
    actions[:, 0] += np.arange(5) * 0.01
    raw = fields.encode_actions(state, actions)
    policy = NativePolicy.__new__(NativePolicy)
    policy.processor = FieldProcessor(
        SimpleNamespace(action=lambda tensor, context=None: tensor), fields, "cpu"
    )
    policy.names, policy.fps = fields.geometry.names, 30
    policy.config, policy.fixed_dimensions = SimpleNamespace(), None
    policy.algorithm = SimpleNamespace(action_offset=lambda config: 0)
    policy.model = SimpleNamespace(predict_action_chunk=lambda batch: torch.tensor(raw[None]))
    policy._observation_batch = lambda observation: {"_action_context": torch.tensor(state[None])}
    policy._require_fixed_current = lambda: None
    policy.inference_context = lambda: nullcontext()
    monkeypatch.setattr(SerialArmKinematics, "inverse", lambda *a, **k: pytest.fail("eager IK"))
    plan = policy.predict_plan(None)
    assert policy.plan_capabilities.kinds == ("tcp",)
    np.testing.assert_allclose(plan.actions, fields.geometry.absolute(actions), atol=1e-6)


def test_tracking_velocity_acceleration_and_plan_replacement_remain_continuous():
    state = joint_state()
    plan = plan_for(state)
    tracker = TCPTracker(
        fps=30, settings={"max_joint_velocity": 0.4, "max_joint_acceleration": 3.0}
    )
    joints = [*range(6), *range(7, 13)]
    previous = np.zeros(12)
    origin = 100.0
    for tick in range(30):
        if tick == 15:
            plan = plan_for(state)
            origin = 100 + tick / 30
        now = 100 + tick / 30
        target = np.asarray(tracker.step(plan, snapshot(state, plan.names), now - origin, now))
        velocity = (target[joints] - state[joints]) * 30
        assert np.max(np.abs(velocity)) <= 0.4 + 1e-8
        assert np.max(np.abs(velocity - previous)) <= 3 / 30 + 1e-7
        previous, state = velocity, target
    assert tracker.status == "tracking"


def test_both_arms_brake_keep_grippers_and_replan_after_one_arm_failure(monkeypatch):
    state = joint_state()
    plan = plan_for(state)
    tracker = TCPTracker(fps=30, settings={"max_joint_acceleration": 0.3})
    for tick in range(20):
        state = np.asarray(
            tracker.step(plan, snapshot(state, plan.names), tick / 30, 100 + tick / 30)
        )
    joint_indices = [*range(6), *range(7, 13)]
    previous = np.concatenate(
        [model.pink_solver().previous_velocity.copy() for _, _, _, model in tracker.geometry.arms]
    )
    actual = state.copy()
    actual[6], actual[13] = 35, 65  # Measured jaw position can differ under contact.
    solver = tracker.geometry.arms[1][3].pink_solver()
    original = solver.step
    monkeypatch.setattr(
        solver, "step", lambda *a, **k: (_ for _ in ()).throw(ValueError("IK failure"))
    )
    held = np.asarray(tracker.step(plan, snapshot(actual, plan.names), 20 / 30, 100 + 20 / 30))
    assert tracker.needs_replan and tracker.status == "holding"
    np.testing.assert_allclose(held[[6, 13]], state[[6, 13]])
    np.testing.assert_equal(held[-4:-1], 0)
    braking_velocity = (held[joint_indices] - state[joint_indices]) * 30
    assert np.max(np.abs(braking_velocity - previous)) <= 0.3 / 30 + 1e-8
    monkeypatch.setattr(solver, "step", original)
    replacement = plan_for(held)
    tracker.step(replacement, snapshot(held, replacement.names), 0, 100 + 21 / 30)
    assert tracker.status == "tracking" and not tracker.needs_replan


def test_expired_reference_holds_then_stops_when_recovery_budget_expires():
    state, plan = joint_state(), plan_for(count=2)
    tracker = TCPTracker(fps=30, settings={"recovery_timeout_s": 0.1})
    tracker.step(plan, snapshot(state, plan.names), plan.end_s, 100)
    assert tracker.needs_replan
    tracker.step(plan, snapshot(state, plan.names), plan.end_s, 100.05)
    with pytest.raises(RuntimeError, match="recovery exhausted"):
        tracker.step(plan, snapshot(state, plan.names), plan.end_s, 100.11)
    with pytest.raises(RuntimeError, match="controller stopped"):
        tracker.step(plan_for(state), snapshot(state, plan.names), 0, 100.14)


def test_command_reference_stays_smooth_with_delayed_noisy_feedback():
    measured = joint_state()
    plan = plan_for(measured)
    tracker = TCPTracker(
        fps=30, settings={"max_joint_velocity": 0.4, "max_joint_acceleration": 3.0}
    )
    indices = [*range(6), *range(7, 13)]
    previous, velocity = measured[indices].copy(), np.zeros(12)
    for tick in range(30):
        observed = measured.copy()
        observed[indices] += 0.002 * np.sin(tick)
        command = np.asarray(
            tracker.step(plan, snapshot(observed, plan.names), tick / 30, 100 + tick / 30)
        )
        following = (command[indices] - previous) * 30
        assert np.max(np.abs(following)) <= 0.4 + 1e-8
        assert np.max(np.abs(following - velocity)) <= 3 / 30 + 1e-7
        previous, velocity = command[indices], following
    assert tracker.status == "tracking"


def test_control_tick_gap_stops_instead_of_extrapolating_stale_velocity():
    state, plan = joint_state(), plan_for()
    tracker = TCPTracker(fps=30)
    tracker.step(plan, snapshot(state, plan.names), 0, 100)
    with pytest.raises(RuntimeError, match="control tick gap"):
        tracker.step(plan, snapshot(state, plan.names), 0.4, 100.4)


def test_fixed_arm_override_is_rejected_before_execution():
    plan = plan_for()
    body = plan.to_dict()
    body["fixed_targets"] = {plan.names[0]: 0.0}
    with pytest.raises(ValueError, match="individual arm joints"):
        plan_from_dict(body)
    body = deepcopy(plan.to_dict())
    body["descriptor"]["arms"][0]["frame"] = "different"
    tracker = TCPTracker(fps=30)
    tracker.step(plan, snapshot(joint_state(), plan.names), 0, 100)
    with pytest.raises(ValueError, match="calibration changed"):
        tracker.step(plan_from_dict(body), snapshot(joint_state(), plan.names), 0.03, 100.03)


def test_far_future_unreachable_point_does_not_discard_the_usable_reference():
    plan = plan_for(count=6)
    body = plan.to_dict()
    body["actions"][-1][0] = 100.0
    plan = plan_from_dict(body)
    state = joint_state()
    tracker = TCPTracker(fps=30)
    tracker.step(plan, snapshot(state, plan.names), 0, 100)
    assert tracker.status == "tracking"
    held = tracker.step(plan, snapshot(state, plan.names), plan.end_s - 0.001, 100.03)
    # Wall time does not skip the retimed path to a far-future target.
    assert tracker.status == "tracking"
    assert tracker.reference_governor.elapsed_s < 0.04
    body["actions"] = [body["actions"][-1]] * 6
    immediate = plan_from_dict(body)
    held = tracker.step(immediate, snapshot(state, plan.names), 0, 100.06)
    assert tracker.status == "holding" and tracker.needs_replan
    assert all(np.isfinite(held))


def test_remote_buffer_tracks_feedback_and_recovers_expired_plan():
    state, plan = joint_state(), plan_for(count=3)
    buffer = PlanBuffer("run", {"epoch": 1}, plan.names, 30, 0.12, ["tcp"])
    buffer.remember(1, 0, 100)
    buffer.accept(
        dict(
            id=1,
            session="run",
            context={"epoch": 1},
            tick=0,
            fps=30,
            names=list(plan.names),
            plan=plan.to_dict(),
            inference_s=0.01,
        ),
        now=100.01,
        tick=0,
    )
    result = buffer.pop(0, 100.01, snapshot(state, plan.names))
    assert all(type(value) is float for value in result.values())
    state = np.array([result[name] for name in plan.names])
    held = buffer.pop(4, 100.14, snapshot(state, plan.names))
    assert held is not None and buffer.tcp_tracker.needs_replan
    replacement = plan_for(state)
    buffer.remember(2, 4, 100.14)
    buffer.accept(
        dict(
            id=2,
            session="run",
            context={"epoch": 1},
            tick=4,
            fps=30,
            names=list(plan.names),
            plan=replacement.to_dict(),
            inference_s=0.01,
        ),
        now=100.15,
        tick=4,
    )
    assert buffer.pop(5, 100.17, snapshot(state, plan.names)) is not None
    assert buffer.tcp_tracker.status == "tracking"


def test_local_executor_keeps_running_control_ticks_between_predictions():
    from unittest.mock import Mock

    from test_trajectory import Worker

    clock = SimpleNamespace(now=100.0)
    worker, state = Worker(), joint_state()
    plan = plan_for(state)
    policy = SimpleNamespace(
        names=plan.names,
        fps=30,
        reset=Mock(),
        task="pick",
        plan_capabilities=PlanCapabilities(kinds=("tcp",), default_replan_interval_s=0.25),
        predict_plan=Mock(return_value=plan),
    )
    executor = TrajectoryExecutor(
        policy, ExecutionConfig(enabled=True), executor=worker, clock=lambda: clock.now
    )

    def observation():
        return SimpleNamespace(
            request_started_s=clock.now, payload=dict(zip(plan.names, state, strict=True))
        )

    try:
        assert executor.select_action(observation()) is None
        clock.now += 1 / 30
        worker.complete()
        command = executor.select_action(observation())
        assert command is not None
        for _ in range(5):
            state[:] = [command[name] for name in plan.names]
            clock.now += 1 / 30
            command = executor.select_action(observation())
            assert command is not None
        assert len(worker.jobs) >= 2
        assert worker.jobs[-1][2][1] is None  # Never pass TCP values as a joint RTC prefix.
    finally:
        executor.close()


@pytest.mark.parametrize("remote", [False, True])
def test_retiming_extends_nominal_window_but_never_extends_prediction_age(remote):
    from unittest.mock import Mock

    from test_trajectory import Worker

    clock = SimpleNamespace(now=100.0)
    worker, state = Worker(), joint_state()
    plan = plan_for(state, count=3)
    settings = {"max_tcp_linear_velocity": 0.003, "max_tcp_linear_acceleration": 0.01}
    policy = SimpleNamespace(
        names=plan.names,
        fps=30,
        reset=Mock(),
        task="pick",
        plan_capabilities=PlanCapabilities(kinds=("tcp",), default_replan_interval_s=0.25),
        predict_plan=Mock(return_value=plan),
    )
    executor = TrajectoryExecutor(
        policy,
        ExecutionConfig(enabled=True, max_prediction_age_s=0.3, tcp_tracking=settings),
        executor=worker,
        clock=lambda: clock.now,
    )
    buffer = PlanBuffer("run", {}, plan.names, 30, 0.3, ["tcp"], tcp_settings=settings)

    def observation():
        observed = snapshot(state, plan.names)
        observed.request_started_s = clock.now
        return observed

    try:
        if remote:
            buffer.remember(1, 0, 100)
            buffer.accept(
                dict(
                    id=1,
                    session="run",
                    context={},
                    tick=0,
                    fps=30,
                    names=list(plan.names),
                    plan=plan.to_dict(),
                    inference_s=0.01,
                ),
                now=100.01,
                tick=1,
            )
            tracker = buffer.tcp_tracker
        else:
            assert executor.select_action(observation()) is None
            worker.complete()
            tracker = executor.tcp_tracker
        for tick in range(1, 12):
            clock.now = 100 + (tick + 0.001) / 30
            command = (
                buffer.pop(tick, clock.now, observation())
                if remote
                else executor.select_action(observation())
            )
            assert command is not None
            state[:] = [command[name] for name in plan.names]
            if tick == 7:
                assert clock.now - 100 > plan.end_s
                assert tracker.status == "tracking"
                assert tracker.reference_governor.elapsed_s < plan.end_s
        assert tracker.status == "holding" and tracker.needs_replan
    finally:
        executor.close()

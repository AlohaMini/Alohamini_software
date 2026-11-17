from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_kinematic_fields import descriptor, joint_state

from alohamini.datasets.kinematic_fields import KinematicFields
from alohamini.tcp_execution import TCPTrackingConfig
from alohamini.tcp_reference import TCPReferenceGovernor
from alohamini.tcp_trajectory import TCPActionPlan


def reference_setup(rotation="rotvec", **settings):
    spec = descriptor()
    spec["rotation_representation"] = rotation
    geometry = KinematicFields(spec)
    initial = geometry.absolute(joint_state())
    config = replace(TCPTrackingConfig(), **settings)
    return geometry, initial, TCPReferenceGovernor(geometry, config, initial)


def poses(geometry, reference):
    result = []
    for offset in geometry.arm_offsets:
        transform = np.eye(4)
        transform[:3, 3] = reference[offset : offset + 3]
        transform[:3, :3] = geometry.rotation.decode(
            reference[offset + 3 : offset + geometry.arm_pose_width]
        )
        result.append(transform)
    return result


def constant_plan(geometry, reference):
    return TCPActionPlan(
        tuple(geometry.names),
        np.repeat(reference[None], 100, axis=0),
        0.1,
        geometry.descriptor,
    )


@pytest.mark.parametrize("rotation", ["rotvec", "rotation_6d"])
def test_linear_and_angular_bounds_survive_reversal_plan_changes_and_uneven_ticks(rotation):
    geometry, initial, governor = reference_setup(
        rotation,
        max_tcp_linear_velocity=0.12,
        max_tcp_linear_acceleration=0.25,
        max_tcp_angular_velocity=0.6,
        max_tcp_angular_acceleration=0.9,
    )
    reference, before = initial.copy(), poses(geometry, initial)
    velocity, angular = np.zeros((2, 3)), np.zeros((2, 3))
    total, origin = 0.0, 0.0
    for tick in range(180):
        dt = (0.01, 0.017, 0.033, 0.025)[tick % 4]
        total += dt
        if tick in (0, 60, 110):
            direction = 1 if tick == 0 else -1
            rotation_vector = {
                0: [1.9, 0.8, 1.0],
                60: [-1.4, 1.7, 0.2],
                110: [0.1, -0.2, -2.5],
            }[tick]
            goal = initial.copy()
            for n, offset in enumerate(geometry.arm_offsets):
                goal[offset : offset + 3] += direction * np.array([0.12, 0.06 * (-1) ** n, 0])
                goal[offset + 3 : offset + geometry.arm_pose_width] = geometry.rotation.encode(
                    Rotation.from_rotvec(rotation_vector).as_matrix() @ before[n][:3, :3]
                )
            plan, origin = constant_plan(geometry, goal), total
        step = governor.propose(plan, total - origin, dt, before)
        governor.commit(step)
        after = poses(geometry, governor.reference)
        following = np.array(
            [(b[:3, 3] - a[:3, 3]) / dt for a, b in zip(before, after, strict=True)]
        )
        following_angular = np.array(
            [
                Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).as_rotvec() / dt
                for a, b in zip(before, after, strict=True)
            ]
        )
        assert np.max(np.linalg.norm(following, axis=1)) <= 0.12 + 1e-10
        assert np.max(np.linalg.norm((following - velocity) / dt, axis=1)) <= 0.25 + 1e-8
        assert np.max(np.linalg.norm(following_angular, axis=1)) <= 0.6 + 1e-10
        assert np.max(np.linalg.norm((following_angular - angular) / dt, axis=1)) <= 0.9 + 1e-8
        before, velocity, angular = after, following, following_angular
        reference = governor.reference
    assert np.linalg.norm(reference - initial) > 0.01


def test_virtual_time_and_gripper_events_slow_down_instead_of_skipping_forward():
    geometry, initial, governor = reference_setup(
        max_tcp_linear_velocity=0.05,
        max_tcp_linear_acceleration=0.1,
    )
    actions = np.repeat(initial[None], 4, axis=0)
    actions[:, 0] += [0, 0.08, 0.08, 0.16]
    actions[:, 1] += [0, 0, 0.08, 0.08]
    actions[:, 6] = [10, 20, 30, 40]
    plan = TCPActionPlan(tuple(geometry.names), actions, 0.1, geometry.descriptor)
    for tick in range(30):
        step = governor.propose(plan, tick / 30, 1 / 30, poses(geometry, governor.reference))
        governor.commit(step)
    assert tick / 30 > plan.end_s
    assert 0 < governor.elapsed_s < plan.period_s
    assert governor.reference[6] == 10
    assert 0 <= governor.rate < 1
    assert np.linalg.norm(governor.reference[:3] - initial[:3]) <= 0.05


def test_measured_tracking_lag_reduces_virtual_progress_and_requests_a_new_plan():
    geometry, initial, fast = reference_setup(max_target_lead_m=0.005)
    _, _, slow = reference_setup(max_target_lead_m=0.005)
    actions = np.repeat(initial[None], 80, axis=0)
    actions[:, 0] += np.linspace(0, 0.2, len(actions))
    plan = TCPActionPlan(tuple(geometry.names), actions, 0.1, geometry.descriptor)
    requested = False
    for tick in range(100):
        for governor, feedback in ((fast, fast.reference), (slow, initial)):
            step = governor.propose(plan, tick / 50, 1 / 50, poses(geometry, feedback))
            governor.commit(step)
            if governor is slow:
                requested |= step.needs_replan
    assert slow.elapsed_s < fast.elapsed_s * 0.75
    assert requested


def test_braking_preserves_reference_and_velocity_until_a_replacement_is_committed():
    geometry, initial, governor = reference_setup()
    goal = initial.copy()
    goal[0] += 0.1
    plan = constant_plan(geometry, goal)
    for tick in range(10):
        step = governor.propose(plan, tick / 50, 1 / 50, poses(geometry, governor.reference))
        governor.commit(step)
    previous = governor.linear_velocity.copy()
    phase = governor.elapsed_s
    replacement = constant_plan(geometry, initial)
    governor.propose(replacement, 0, 1 / 50, poses(geometry, governor.reference))
    assert governor.plan is plan  # An unsuccessful IK step must not commit the replacement.
    governor.brake(1 / 50)
    assert governor.elapsed_s == phase
    assert np.max(np.linalg.norm(governor.linear_velocity - previous, axis=1)) <= 0.5 / 50 + 1e-10
    assert np.all(
        np.linalg.norm(governor.linear_velocity, axis=1) <= np.linalg.norm(previous, axis=1)
    )


def test_corner_acceleration_is_bounded_and_gripper_events_remain_in_order():
    geometry, initial, governor = reference_setup()
    actions = np.repeat(initial[None], 5, axis=0)
    actions[:, 0] += [0, 0.01, 0.01, 0, 0]
    actions[:, 1] += [0, 0, 0.01, 0.01, 0]
    actions[:, 6] = [10, 20, 30, 40, 50]
    plan = TCPActionPlan(tuple(geometry.names), actions, 0.2, geometry.descriptor)
    events, previous = [], np.zeros((2, 3))
    for tick in range(250):
        if governor.plan is plan and governor.elapsed_s >= plan.end_s:
            break
        step = governor.propose(plan, tick / 50, 1 / 50, poses(geometry, governor.reference))
        governor.commit(step)
        assert np.max(np.linalg.norm((step.linear_velocity - previous) * 50, axis=1)) <= 0.5 + 1e-9
        previous = step.linear_velocity
        if not events or governor.reference[6] != events[-1]:
            events.append(governor.reference[6])
    assert events == [10, 20, 30, 40, 50]
    assert governor.elapsed_s == plan.end_s

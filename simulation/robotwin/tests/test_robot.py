import os
from pathlib import Path

import numpy as np
import pytest
import sapien
from alohamini_robotwin import create_robot

from alohamini.schema import BodyVelocity
from alohamini.simulation.base import integrate_base
from alohamini.simulation.sapien import rigid_groups


@pytest.fixture
def robot(monkeypatch):
    source = os.environ.get("ROBOTWIN_SOURCE")
    if not source:
        pytest.skip("Set ROBOTWIN_SOURCE to a working RoboTwin checkout")
    source = Path(source).expanduser().resolve()
    monkeypatch.chdir(source)
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(0.001)
    robot = create_robot(scene, source=source, render=False, base_mode="kinematic")
    yield robot, scene
    scene.clear()


def test_native_load_preserves_frames_inertias_collisions_and_limits(robot):
    robot, _ = robot
    shared = robot.shared
    expected = shared.description.forward(
        dict(zip(shared.names, shared.robot.get_qpos(), strict=True))
    )
    for name, pose in expected.items():
        np.testing.assert_allclose(shared.frame(name), pose, rtol=0, atol=1e-6)
    initial = shared.robot.get_qpos().copy()
    rng = np.random.default_rng(21)
    for _ in range(3):
        sample = initial.copy()
        for i, name in enumerate(shared.names):
            joint = shared.description.joints[name]
            if joint.lower is not None:
                sample[i] = rng.uniform(joint.lower, joint.upper)
        shared.robot.set_qpos(sample)
        poses = shared.description.forward(dict(zip(shared.names, sample, strict=True)))
        for name, pose in poses.items():
            np.testing.assert_allclose(shared.frame(name), pose, rtol=0, atol=1e-6)
    shared.robot.set_qpos(initial)
    _, inertias = rigid_groups(shared.description)
    for link in shared.robot.get_links():
        _, com, tensor = inertias[link.name]
        pose = link.cmass_local_pose.to_transformation_matrix()
        np.testing.assert_allclose(pose[:3, 3], com, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            pose[:3, :3] @ np.diag(link.inertia) @ pose[:3, :3].T,
            tensor,
            rtol=0,
            atol=16 * np.finfo(np.float32).eps * np.linalg.norm(tensor),
        )
    assert sum(link.mass for link in shared.robot.get_links()) == pytest.approx(
        sum(v[0] for v in shared.description.inertials.values()), rel=1e-6
    )
    assert sum(len(link.get_collision_shapes()) for link in shared.robot.get_links()) == sum(
        len(link.findall("collision")) for link in shared.description.links.values()
    )
    for name, limits in zip(shared.names, shared.robot.get_qlimits(), strict=True):
        joint = shared.description.joints[name]
        if joint.lower is not None:
            np.testing.assert_allclose(limits, [joint.lower, joint.upper], atol=1e-6)
    np.testing.assert_allclose(
        robot.get_left_tcp_pose()[:3], shared.frame("left_tcp")[:3, 3], rtol=0, atol=1e-6
    )


def test_native_reset_rebuilds_shared_model_and_command_state(robot):
    robot, scene = robot
    initial = robot.shared.robot.get_qpos().copy()
    expected_entities = len(scene.entities)
    goal = robot.get_left_arm_jointState()[:6]
    goal[0] += 0.02
    robot.set_arm_joints(goal, [0] * 6, "left")
    robot.set_gripper(0.5, "left")
    assert robot.get_left_arm_jointState() == [*goal, 0.5]
    robot.reset(scene)
    assert len(scene.entities) == expected_entities
    np.testing.assert_allclose(robot.shared.robot.get_qpos(), initial)
    assert robot.get_left_arm_jointState()[-1] == 1
    assert robot.get_left_arm_real_jointState()[-1] == pytest.approx(1, abs=1e-6)
    with pytest.raises(ValueError, match="outside URDF limits"):
        robot.shared.set_targets({"vertical_move": 0.4})


def test_base_motion_matches_body_twist_and_motor_caps(robot):
    robot, scene = robot
    shared = robot.shared
    initial = shared.frame("base_link")
    velocity = BodyVelocity(0.02, -0.01, 0.1)
    shared.base_motion.set_velocity(velocity)
    for _ in range(200):
        robot.apply_passive_force()
        scene.step()
    np.testing.assert_allclose(
        shared.frame("base_link"), integrate_base(initial, velocity, 0.2), rtol=0, atol=1e-6
    )
    assert np.all(np.abs(shared.robot.get_qf()) <= shared.control.effort_limits + 1e-5)

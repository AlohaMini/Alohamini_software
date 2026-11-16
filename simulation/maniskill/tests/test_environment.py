import copy
import json

import alohamini_maniskill  # noqa: F401
import gymnasium as gym
import numpy as np
import pytest

from alohamini.schema import BodyVelocity
from alohamini.simulation.base import integrate_base
from alohamini.simulation.sapien import rigid_groups


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    path = tmp_path_factory.mktemp("scene") / "scene.json"
    path.write_text(json.dumps({"base_mode": "kinematic"}))
    environment = gym.make(
        "AlohaMiniResearch-v1",
        scene_config=path,
        sim_backend="physx_cpu",
        render_backend="none",
        obs_mode="state",
        control_mode="joint_position",
        reward_mode="none",
    )
    yield environment
    environment.close()


def test_native_load_preserves_frames_inertias_collisions_and_limits(env):
    obs, _ = env.reset(seed=3)
    agent = env.unwrapped.agent
    names = tuple(j.name for j in agent.robot.active_joints)
    q = agent.raw_robot.get_qpos()
    expected = agent.description.forward(dict(zip(names, q, strict=True)))
    assert obs.shape[-1] == 43  # 18 qpos, 18 qvel, XYZ + WXYZ base pose
    assert len(agent.action_joint_names) == 15
    for name, pose in expected.items():
        np.testing.assert_allclose(agent.frame(name), pose, rtol=0, atol=1e-6)
    rng = np.random.default_rng(21)
    for _ in range(3):
        sample = q.copy()
        for i, name in enumerate(names):
            joint = agent.description.joints[name]
            if joint.lower is not None:
                sample[i] = rng.uniform(joint.lower, joint.upper)
        agent.raw_robot.set_qpos(sample)
        poses = agent.description.forward(dict(zip(names, sample, strict=True)))
        for name, pose in poses.items():
            np.testing.assert_allclose(agent.frame(name), pose, rtol=0, atol=1e-6)
    agent.raw_robot.set_qpos(q)
    _, inertias = rigid_groups(agent.description)
    for name, public in agent.robot.links_map.items():
        link = public._objs[0]
        _, com, tensor = inertias[name]
        pose = link.cmass_local_pose.to_transformation_matrix()
        np.testing.assert_allclose(pose[:3, 3], com, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            pose[:3, :3] @ np.diag(link.inertia) @ pose[:3, :3].T,
            tensor,
            rtol=0,
            atol=16 * np.finfo(np.float32).eps * np.linalg.norm(tensor),
        )
    assert sum(link.mass for link in agent.raw_robot.get_links()) == pytest.approx(
        sum(v[0] for v in agent.description.inertials.values()), rel=1e-6
    )
    assert sum(len(link.get_collision_shapes()) for link in agent.raw_robot.get_links()) == sum(
        len(link.findall("collision")) for link in agent.description.links.values()
    )
    for name, limits in zip(names, agent.raw_robot.get_qlimits(), strict=True):
        joint = agent.description.joints[name]
        if joint.lower is not None:
            np.testing.assert_allclose(limits, [joint.lower, joint.upper], atol=1e-6)


def test_base_motion_and_complete_snapshot_restore(env):
    env.reset(seed=4)
    task, agent = env.unwrapped, env.unwrapped.agent
    action = agent.hold_action()
    initial = agent.frame("base_link")
    velocity = BodyVelocity(0.02, -0.01, 0.1)
    agent.set_base_velocity(velocity)
    for _ in range(3):
        env.step(action)
    np.testing.assert_allclose(
        agent.frame("base_link"),
        integrate_base(initial, velocity, 3 / task.control_freq),
        rtol=0,
        atol=1e-6,
    )
    state = copy.deepcopy(task.get_state_dict())
    phase = agent.base_motion.wheel_positions.copy()
    for _ in range(3):
        env.step(action)
    expected_qpos = agent.raw_robot.get_qpos().copy()
    expected_pose = agent.frame("base_link")
    task.set_state_dict(state)
    np.testing.assert_allclose(agent.base_motion.wheel_positions, phase)
    for _ in range(3):
        env.step(action)
    np.testing.assert_allclose(agent.raw_robot.get_qpos(), expected_qpos, rtol=0, atol=1e-5)
    np.testing.assert_allclose(agent.frame("base_link"), expected_pose, rtol=0, atol=1e-6)
    assert task.elapsed_steps.item() == 6
    assert np.all(np.abs(agent.raw_robot.get_qf()) <= agent.position_control.effort_limits + 1e-5)
    env.reset(seed=4)
    np.testing.assert_allclose(agent.frame("base_link"), initial, rtol=0, atol=1e-6)
    np.testing.assert_allclose(agent.base_motion.wheel_positions, 0)

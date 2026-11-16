import json
from pathlib import Path

import numpy as np
import pytest

from alohamini.model import get_robot_model
from alohamini.schema import BodyVelocity
from alohamini.simulation.base import BaseKinematics, BaseMotion, integrate_base
from alohamini.simulation.control import PositionControl
from alohamini.simulation.description import Description
from alohamini.simulation.frames import transform
from alohamini.simulation.sapien import rigid_groups
from alohamini.simulation.scene import load_scene


@pytest.fixture(scope="module")
def description():
    return Description(get_robot_model("alohamini2pro"))


def test_physical_root_and_legacy_fk_agree_for_all_retained_frames(description):
    assert description.root == "base_link"
    assert all(description.joints[n].kind == "continuous" for n in BaseKinematics.names)
    assert (
        not {"root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"}
        & description.joints.keys()
    )
    legacy = Description(get_robot_model("alohamini2pro"), name="planning")
    rng = np.random.default_rng(23)
    world = transform([0.4, -0.3, 0], [0, 0, 0.7])
    for _ in range(8):
        q = {
            n: rng.uniform(j.lower, j.upper)
            for n, j in description.joints.items()
            if j.lower is not None
        }
        actual, expected = description.forward(q, world), legacy.forward(q, world)
        for name in actual:
            np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=1e-12)


def test_base_inverse_and_forward_match_actual_axle_geometry(description):
    kinematics = BaseKinematics(description)
    for velocity in (
        BodyVelocity(0.1, 0, 0),
        BodyVelocity(0, 0.1, 0),
        BodyVelocity(0, 0, 0.4),
        BodyVelocity(0.1, -0.07, 0.3),
    ):
        wheels = kinematics.body_to_wheels(velocity)
        recovered = kinematics.wheels_to_body(wheels)
        np.testing.assert_allclose(
            list(vars(recovered).values()), list(vars(velocity).values()), rtol=0, atol=1e-15
        )
        for speed, geom in zip(wheels, kinematics.geometry, strict=True):
            x, y = geom["center_base_m"][:2]
            center_velocity = [
                velocity.x_m_s - y * velocity.yaw_rad_s,
                velocity.y_m_s + x * velocity.yaw_rad_s,
            ]
            rolling = np.dot(center_velocity, geom["rolling_direction_base"])
            assert speed * geom["radius_m"] == pytest.approx(rolling, abs=1e-15)


def test_body_twist_composition_and_wheel_speed_limits(description):
    pose = transform([1, 2, 0.03], [0, 0, np.pi / 2])
    np.testing.assert_allclose(
        integrate_base(pose, BodyVelocity(0.1, 0, 0), 1)[:3, 3], [1, 2.1, 0.03]
    )
    velocity = BodyVelocity(0.1, -0.03, 0.5)
    sequential = integrate_base(integrate_base(pose, velocity, 0.4), velocity, 0.6)
    np.testing.assert_allclose(sequential, integrate_base(pose, velocity, 1), atol=1e-14)
    fixed = BaseMotion(description, pose)
    with pytest.raises(ValueError, match="fixed base"):
        fixed.set_velocity(velocity)
    moving = BaseMotion(description, pose, mode="kinematic")
    with pytest.raises(ValueError, match="speed limit"):
        moving.set_velocity(BodyVelocity(10, 0, 0))


def test_fixed_groups_preserve_mass_and_world_inertia(description):
    frames, groups = rigid_groups(description)
    poses = description.forward()
    assert sum(m for m, _, _ in groups.values()) == pytest.approx(
        sum(m for m, _, _ in description.inertials.values())
    )
    source_origin_tensor = np.zeros((3, 3))
    actual_origin_tensor = np.zeros((3, 3))
    for name, (mass, com, tensor) in description.inertials.items():
        p = poses[name] @ com
        c, r = p[:3, 3], p[:3, :3]
        source_origin_tensor += r @ tensor @ r.T + mass * (c @ c * np.eye(3) - np.outer(c, c))
    for name, (mass, com, tensor) in groups.items():
        p = poses[name]
        c, r = p[:3, :3] @ com + p[:3, 3], p[:3, :3]
        actual_origin_tensor += r @ tensor @ r.T + mass * (c @ c * np.eye(3) - np.outer(c, c))
        assert mass > 0 and np.linalg.eigvalsh(tensor).min() > 0
    np.testing.assert_allclose(actual_origin_tensor, source_origin_tensor, atol=1e-12)
    assert frames["left_tcp"][0] == "left_Fixed_Jaw"
    assert frames["left_camera"][0] == "left_Fixed_Jaw"


def test_position_command_limits_include_feedforward_and_native_precision(description):
    control = PositionControl(description, ["vertical_move", "left_gripper"], [0.2, 0])
    lower = description.joints["left_gripper"].lower
    endpoint = PositionControl(description, ["left_gripper"], [np.float32(lower)])
    endpoint.restore(endpoint.state())
    assert endpoint.target[0] == lower
    control.set_goal([0.3, np.float32(lower)])
    assert control.goal[1] == lower
    with pytest.raises(ValueError, match="outside URDF limits"):
        control.set_goal([0.31, 0])
    effort = control.effort(np.array([0.2, 0]), np.zeros(2), np.array([1000, 1000]))
    np.testing.assert_allclose(effort, control.effort_limits)
    assert control.target[0] - 0.2 <= 0.05 * control.timestep + 1e-15


def test_scene_has_explicit_geometry_and_rejects_unknown_fields(tmp_path):
    source = Path(__file__).parents[1] / "examples/simulation/research.json"
    config = load_scene(source)
    assert config["T_world_base"][0][3] == -0.089
    assert next(o for o in config["objects"] if o["name"] == "table")["size"] == [0.9, 0.8, 0.05]
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({"height": 0.75}))
    with pytest.raises(ValueError, match="Unknown scene fields"):
        load_scene(path)
    path.write_text(json.dumps({"timestep_s": 0.003, "control_frequency_hz": 10}))
    with pytest.raises(ValueError, match="integer physics/control frequencies"):
        load_scene(path)

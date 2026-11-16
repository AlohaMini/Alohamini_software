import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from alohamini_mujoco.compilation import load_xml
from alohamini_mujoco.model import ModelProfile, build_model
from alohamini_mujoco.scene import ResearchScene
from alohamini_mujoco.viewer import DynamicPlayback, JointControls

from alohamini.model import get_robot_model
from alohamini.schema import BodyVelocity


@pytest.fixture(scope="module")
def physical_robot():
    return build_model()


def test_shared_urdf_loads_natively_and_all_bounded_limits_are_enabled(physical_robot):
    native = load_xml(get_robot_model("alohamini2pro").description_path("collision"))
    assert native.nq == 18
    assert physical_robot.model.nq == 15
    assert native.ngeom == physical_robot.model.ngeom == 1141
    assert not native.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    for name, joint in physical_robot.description.joints.items():
        if joint.lower is not None:
            assert bool(native.joint(name).limited[0])
            np.testing.assert_array_equal(native.joint(name).range, [joint.lower, joint.upper])
    assert physical_robot.model.body_mass.sum() == pytest.approx(11.32951, abs=1e-10)


def test_compiled_lift_limit_stops_persistent_force(physical_robot):
    import copy

    model = copy.copy(physical_robot.model)
    model.opt.gravity[:] = 0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    joint = model.joint("vertical_move")
    q, v = int(joint.qposadr[0]), int(joint.dofadr[0])
    for position, force in [(0.3, 100), (-0.3, -100)]:
        data = mujoco.MjData(model)
        data.qpos[q] = position
        data.qfrc_applied[v] = force
        for _ in range(1000):
            mujoco.mj_step(model, data)
            assert abs(data.qpos[q]) < 0.302


def test_joint_ui_changes_targets_without_teleporting_or_widening_effort(physical_robot):
    import copy

    playback = DynamicPlayback(physical_robot, paused=True)
    display_model = copy.copy(physical_robot.model)
    display = mujoco.MjData(display_model)
    mujoco.mj_copyData(display, display_model, playback.data)
    controls = JointControls(playback, display_model, display)
    index = playback.follower.names.index("vertical_move")
    initial = playback.data.qpos.copy()
    display.ctrl[index] = 0.3
    controls.read(display)
    assert playback.manual_target[index] == 0.3
    np.testing.assert_array_equal(playback.data.qpos, initial)
    assert physical_robot.model.actuator_ctrlrange[index, 1] == pytest.approx(74.2)
    for _ in range(20):
        playback.step(single=True)
    assert playback.follower.target[index] <= 0.05 * playback.data.time + 1e-12
    with pytest.raises(ValueError, match="outside URDF limits"):
        target = playback.manual_target.copy()
        target[index] = 0.8
        playback.set_manual_target(target)


def test_common_scene_kinematic_base_uses_world_pose_and_rolls_wheels(tmp_path):
    source = Path(__file__).resolve().parents[3] / "examples/simulation/research.json"
    config = json.loads(source.read_text())
    config["base_mode"] = "kinematic"
    config["objects"] = []
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(config))
    scene = ResearchScene(path)
    robot = build_model(
        profile=ModelProfile(
            locked_positions={},
            T_world_root=config["T_world_base"],
            initial_positions=config["initial_positions"],
        ),
        scene=scene,
    )
    playback = DynamicPlayback(robot)
    playback.base_motion.set_velocity(BodyVelocity(0.02, 0, 0))
    for _ in range(100):
        playback.step()
    pose = playback.data.body("base_link").xpos
    assert pose[0] == pytest.approx(config["T_world_base"][0][3] + 0.002, abs=3e-5)
    assert abs(playback.base_motion.wheel_positions).max() > 0
    playback.reset()
    np.testing.assert_allclose(playback.base_motion.pose, config["T_world_base"])

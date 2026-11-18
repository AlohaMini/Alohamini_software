import json
import xml.etree.ElementTree as ET
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
    expected_geoms = sum(
        len(link.findall("collision")) + len(link.findall("visual"))
        for link in physical_robot.description.links.values()
    )
    assert native.ngeom == physical_robot.model.ngeom == expected_geoms
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


def test_lift_internal_contact_is_excluded_while_external_contact_remains_active(physical_robot):
    xml = ET.fromstring(physical_robot.xml)
    probe = ET.SubElement(xml.find("worldbody"), "body", name="external_probe", mocap="true")
    ET.SubElement(probe, "geom", name="external_probe", type="sphere", size="0.001")
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(xml, encoding="unicode"), physical_robot.assets
    )
    assert not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    housing, sphere = model.geom("vertical_link__collision_0").id, model.geom("external_probe").id
    internal_bodies = {model.body("base_cad_link").id, model.body("vertical_link").id}
    for height in (-0.3, 0.0, 0.3):
        data = mujoco.MjData(model)
        data.qpos[:] = physical_robot.data({"vertical_move": height}).qpos
        mujoco.mj_forward(model, data)
        outward = -data.geom_xmat[housing].reshape(3, 3)[:, 1]
        data.mocap_pos[0] = data.geom_xpos[housing] + outward * (
            model.geom_size[housing, 1] + 0.001 - 0.0001
        )
        mujoco.mj_forward(model, data)
        contacts = list(data.contact)
        assert any(set(c.geom) == {housing, sphere} and c.dist < 0 for c in contacts)
        assert not any(
            {int(model.geom_bodyid[g]) for g in c.geom} == internal_bodies for c in contacts
        )


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


def test_gripper_contact_pads_keep_cad_position_and_contact_normal():
    root = get_robot_model("alohamini2pro").directory
    reference = json.loads((root / "config/gripper_reference.json").read_text())
    certificate = json.loads((root / "config/gripper_collision.json").read_text())
    for kind, record in reference["parts"].items():
        xml = ET.Element("mujoco")
        assets, world = ET.SubElement(xml, "asset"), ET.SubElement(xml, "worldbody")
        pieces = certificate["parts"][kind]["pieces"]
        for index, piece in enumerate(pieces):
            if piece.get("type") == "box":
                pose = np.asarray(piece["T_link_geometry"])
                quaternion = np.zeros(4)
                mujoco.mju_mat2Quat(quaternion, pose[:3, :3].ravel())
                ET.SubElement(
                    world,
                    "geom",
                    name=f"piece_{index}",
                    type="box",
                    size=" ".join(map(str, np.asarray(piece["parameters"]["size"]) / 2)),
                    pos=" ".join(map(str, pose[:3, 3])),
                    quat=" ".join(map(str, quaternion)),
                )
                continue
            ET.SubElement(
                assets,
                "mesh",
                name=f"mesh_{index}",
                file=str(root / piece["mesh"]),
                scale=" ".join(map(str, piece["scale"])),
            )
            ET.SubElement(world, "geom", name=f"piece_{index}", type="mesh", mesh=f"mesh_{index}")
        probe_body = ET.SubElement(world, "body", mocap="true")
        ET.SubElement(probe_body, "geom", name="probe", type="sphere", size="0.001")
        model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        data = mujoco.MjData(model)
        probe = model.geom("probe").id
        with np.load(root / record["mesh"], allow_pickle=False) as source:
            vertices = source["contact_vertices"]
        normal = np.asarray(record["contact"]["normal"])
        centre = vertices.mean(0)
        for point in [centre, (centre + vertices[0]) / 2, (centre + vertices[-1]) / 2]:
            data.mocap_pos[0] = point + normal * 0.0012
            mujoco.mj_forward(model, data)
            distances, directions = [], []
            for index in range(len(pieces)):
                line = np.zeros(6)
                distances.append(mujoco.mj_geomDistance(model, data, index, probe, 0.1, line))
                directions.append(line[3:] - line[:3])
            closest = int(np.argmin(distances))
            pad_indices = certificate["parts"][kind]["contacts"][0].get("piece_indices", [0])
            assert closest in pad_indices
            assert distances[closest] == pytest.approx(0.0002, abs=0.00001)
            direction = directions[closest] / np.linalg.norm(directions[closest])
            assert direction @ normal > 0.9999
            for penetration in (0.0001, 0.0003):
                data.mocap_pos[0] = point + normal * (0.001 - penetration)
                mujoco.mj_forward(model, data)
                touching = [
                    index
                    for index in range(len(pieces))
                    if mujoco.mj_geomDistance(model, data, index, probe, 0.1, None) < -1e-8
                ]
                assert touching and set(touching) <= set(pad_indices), (kind, penetration, touching)

import json
import xml.etree.ElementTree as ET
from dataclasses import replace

import mujoco
import numpy as np
import pytest
from alohamini_mujoco.audit import validate_model
from alohamini_mujoco.collection import collect_episode, replay_episode, restore_state
from alohamini_mujoco.control import JointTrajectory, solve_ik
from alohamini_mujoco.libero import LIBERO_COMMIT, LiberoScene, sha256
from alohamini_mujoco.model import ModelProfile, build_model


@pytest.fixture(scope="module")
def scene_robot(tmp_path_factory):
    directory = tmp_path_factory.mktemp("libero_scene")
    folder = directory / "task_000"
    folder.mkdir()
    (directory / "assets").mkdir()
    xml = """<mujoco><compiler angle="radian" inertiafromgeom="false"/>
      <worldbody>
        <geom name="support" type="box" pos=".15 .3 -.05" size=".15 .15 .05" group="3"/>
        <camera name="agentview" pos="0 0 2"/>
        <body name="source" pos=".1 .3 .05"><freejoint name="source_joint"/>
          <inertial pos="0 0 0" mass=".1" diaginertia=".001 .001 .001"/>
          <geom name="source_geom" type="box" size=".05 .05 .05" group="3"/>
        </body>
        <body name="target" pos=".2 .3 .05"><freejoint name="target_joint"/>
          <inertial pos="0 0 0" mass=".1" diaginertia=".001 .001 .001"/>
          <geom name="target_geom" type="box" size=".05 .05 .05" group="3"/>
        </body>
      </worldbody><asset/></mujoco>"""
    source = mujoco.MjModel.from_xml_string(xml)
    (folder / "scene.xml").write_text(xml)
    metadata = {
        "suite": "libero_spatial",
        "task": {"name": "unit_on", "language": "place source on target"},
        "parsed_problem": {"goal_state": [["on", "source", "target"]]},
        "source_bddl_sha256": "test",
        "source_init_sha256": "test",
        "source_timestep_s": 0.002,
        "source_gravity_m_s2": [0, 0, -9.81],
        "objects": {
            name: {"root_body": name, "contact_geoms": [f"{name}_geom"]}
            for name in ("source", "target")
        },
        "initial_states": [
            {
                "source_joint": [0.08, 0.3, 0.05, 1, 0, 0, 0],
                "target_joint": [0.2, 0.3, 0.05, 1, 0, 0, 0],
            },
            {
                "source_joint": [0.2, 0.3, 0.15, 1, 0, 0, 0],
                "target_joint": [0.2, 0.3, 0.05, 1, 0, 0, 0],
            },
        ],
        "body_inertials": {
            name: {
                k: np.atleast_1d(getattr(source.body(name), k)).tolist()
                if k != "mass"
                else float(source.body(name).mass[0])
                for k in ("mass", "ipos", "iquat", "inertia")
            }
            for name in ("source", "target")
        },
        "contact_parameters": {
            name: {
                k: np.atleast_1d(getattr(source.geom(name), k)).tolist()
                for k in ("contype", "conaffinity", "condim", "friction", "solref", "solimp")
            }
            for name in ("source_geom", "target_geom", "support")
        },
        "geom_geometry": {
            name: {
                "body": source.body(source.geom(name).bodyid[0]).name,
                **{
                    k: np.atleast_1d(getattr(source.geom(name), k)).tolist()
                    for k in ("type", "size", "pos", "quat")
                },
            }
            for name in ("source_geom", "target_geom", "support")
        },
    }
    (folder / "task.json").write_text(json.dumps(metadata))
    manifest = {
        "schema_version": 2,
        "source_commit": LIBERO_COMMIT,
        "tasks": [0],
        "files": {
            p.relative_to(directory).as_posix(): sha256(p)
            for p in directory.rglob("*")
            if p.is_file()
        },
    }
    (directory / "bundle.json").write_text(json.dumps(manifest))
    config = {"T_world_libero": [[0, -1, 0, 0.7], [1, 0, 0, -0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]]}
    scene = LiberoScene(directory, 0, config)
    return build_model(profile=ModelProfile(initial_positions={"vertical_move": 0.2}), scene=scene)


def test_scene_keeps_robot_inertia_and_named_free_object_transforms(scene_robot):
    robot = scene_robot
    report = validate_model(robot, samples=2)
    assert report["passed"], report
    assert report["source_mass_kg"] == pytest.approx(16)
    assert report["whole_scene_mass_kg"] == pytest.approx(16.2)
    assert robot.scene.validate(robot)["passed"]
    data = robot.data()
    np.testing.assert_allclose(data.body("source").xpos, [0.4, -0.12, 0.35], atol=1e-12)
    np.testing.assert_allclose(
        data.body("source").xmat.reshape(3, 3), robot.scene.T_world_libero[:3, :3], atol=1e-12
    )
    assert not robot.scene.goal_status(robot, data)["passed"]


def test_on_requires_real_contact_and_does_not_accept_hovering(scene_robot):
    scene = LiberoScene(scene_robot.scene.directory, 0, scene_robot.scene.configuration, 1)
    robot = replace(scene_robot, scene=scene)
    data = robot.data()
    assert scene.goal_status(robot, data)["passed"]
    source = robot.model.jnt_qposadr[robot.model.joint("source_joint").id]
    data.qpos[source + 2] += 0.05
    mujoco.mj_forward(robot.model, data)
    assert not scene.goal_status(robot, data)["passed"]


def test_ik_contact_diagnostics_preserve_measured_object_state(scene_robot):
    robot = scene_robot
    live = robot.data()
    address = robot.model.jnt_qposadr[robot.model.joint("source_joint").id]
    # A measured object pose deliberately differs from the scene reset bank.
    live.qpos[address : address + 3] = live.geom("right_Fixed_Jaw__collision_7").xpos
    mujoco.mj_forward(robot.model, live)
    target = np.eye(4)
    target[:3, :3] = live.body("right_tcp").xmat.reshape(3, 3)
    target[:3, 3] = live.body("right_tcp").xpos
    before = live.qpos.copy()
    result = solve_ik(robot, "right", target, reference_data=live)
    assert any("source_geom" in (c["geom1"], c["geom2"]) for c in result["contacts"])
    np.testing.assert_array_equal(live.qpos, before)


def test_scene_bundle_detects_mesh_or_metadata_changes_before_loading(scene_robot, tmp_path):
    import shutil

    directory = tmp_path / "tampered"
    shutil.copytree(scene_robot.scene.directory, directory)
    (directory / "task_000/scene.xml").write_text("<mujoco/>")
    with pytest.raises(ValueError, match="checksum"):
        LiberoScene(directory, 0, scene_robot.scene.configuration)


def test_static_geometry_intersections_are_not_hidden_by_mujoco_weld_filter(scene_robot):
    model = scene_robot.model
    original = model.geom("support").pos.copy()
    try:
        model.geom("support").pos[:2] = [0, 0]
        data = scene_robot.data()
        base = model.geom("base_cad_link__collision_0").id
        support = model.geom("support").id
        assert not any({c.geom1, c.geom2} == {base, support} for c in data.contact)
        report = scene_robot.scene.static_intersections(scene_robot)
        assert not report["passed"]
        assert any(p["robot_geom"] == "base_cad_link__collision_0" for p in report["intersections"])
    finally:
        model.geom("support").pos[:] = original


def test_collection_rejects_failure_preserves_alignment_and_replays(
    scene_robot, tmp_path, monkeypatch
):
    pytest.importorskip("h5py")
    import h5py
    from alohamini_mujoco import cli

    # Export the same compiled XML/assets without another expensive robot reload;
    # actual CLI integration separately exercises the production exporter.
    def export(robot, report, output):
        output.mkdir()
        (output / "assets").mkdir()
        for name, contents in robot.assets.items():
            (output / "assets" / name).write_bytes(contents)
        tree = ET.fromstring(robot.xml)
        tree.find("compiler").set("meshdir", "assets")
        (output / "model.xml").write_text(ET.tostring(tree, encoding="unicode"))

    monkeypatch.setattr(cli, "export_robot", export)
    trajectory = JointTrajectory(scene_robot, ["left_shoulder_pan"], [0, 0.2], [[0], [0.005]])
    output = tmp_path / "collection"
    report = collect_episode(scene_robot, trajectory, output, cameras=("agentview",))
    assert not report["passed"]
    assert not report["quality"]["checks"]["bilateral_grasp"]
    assert report["quality"]["replay"]["passed"]
    manifest = json.loads((output / "dataset.json").read_text())
    assert not (output / "episodes").exists()
    episode = output / manifest["episode"]
    with h5py.File(episode, "r") as file:
        group = file["data/demo_0"]
        assert len(group["states"]) == len(group["actions"]) + 1
        np.testing.assert_allclose(np.diff(group["simulation_time_s"]), 0.05, atol=1e-12)
        assert group["obs/agentview_rgb"].dtype == np.uint8
        assert group["obs/agentview_rgb"].shape[1:] == (128, 128, 3)
        data = scene_robot.data()
        ids = [
            scene_robot.model.jnt_qposadr[scene_robot.model.joint(n).id]
            for n in json.loads(group.attrs["joint_names"])
        ]
        for i in range(len(group["actions"])):
            restore_state(scene_robot.model, data, group["states"][i])
            np.testing.assert_array_equal(group["obs/joint_positions"][i], data.qpos[ids])
    with h5py.File(episode, "r+") as file:
        file["data/demo_0/actions"][0, 0] += 0.1
    with pytest.raises(ValueError, match="Episode checksum"):
        replay_episode(output)

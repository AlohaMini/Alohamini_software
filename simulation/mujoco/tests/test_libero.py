import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
from alohamini_mujoco.audit import validate_model
from alohamini_mujoco.collection import (
    EpisodeReplay,
    collect_episode,
    physics_state,
    replay_episode,
    restore_state,
)
from alohamini_mujoco.control import FollowerSettings, JointTrajectory, solve_ik
from alohamini_mujoco.libero import LIBERO_COMMIT, LiberoScene, sha256
from alohamini_mujoco.model import ModelProfile, build_model


@pytest.mark.parametrize(
    "packages,accepted",
    [
        ({"mujoco": "3.8.1", "robosuite": "1.4.0"}, True),
        ({"mujoco": "3.12.0", "robosuite": "1.4.0"}, False),
        ({"mujoco": "3.8.1", "robosuite": "1.5.0"}, False),
    ],
)
def test_import_pins_exporter_independently_of_runtime(tmp_path, monkeypatch, packages, accepted):
    import sys

    from alohamini_mujoco import libero

    monkeypatch.setattr(libero, "_source_revision", lambda source: None)
    monkeypatch.setattr(mujoco, "__version__", "3.12.0")

    def export(command, **kwargs):
        # The exporter receives a staging directory, not the final destination.
        directory = Path(command[command.index("--output") + 1])
        (directory / "upstream.json").write_text(json.dumps({"upstream_packages": packages}))

    monkeypatch.setattr(libero.subprocess, "run", export)
    if accepted:
        directory = libero.import_libero(tmp_path / "source", sys.executable, tmp_path / "result")
        bundle = json.loads((directory / "bundle.json").read_text())
        assert bundle["upstream_packages"] == packages
        assert bundle["files"]["upstream.json"] == sha256(directory / "upstream.json")
        exported = json.loads((directory / "upstream.json").read_text())
        assert exported["upstream_packages"] == packages
    else:
        with pytest.raises(ValueError, match="Upstream export"):
            libero.import_libero(tmp_path / "source", sys.executable, tmp_path / "result")
        assert not (tmp_path / "result").exists()


@pytest.fixture(scope="module")
def scene_robot(tmp_path_factory):
    directory = tmp_path_factory.mktemp("libero_scene")
    folder = directory / "task_000"
    folder.mkdir()
    (directory / "assets").mkdir()
    xml = """<mujoco><compiler angle="radian" inertiafromgeom="false"/>
      <worldbody>
        <geom name="floor" type="plane" pos="0 0 -.5" size="3 3 .1" group="3"/>
        <geom name="support" type="box" pos=".15 .3 -.05" size=".15 .15 .05" group="3"/>
        <camera name="agentview" pos="0 0 2"/>
        <site name="observation_debug_marker" pos=".1 .3 .16" size=".03" rgba="1 0 0 1"/>
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
        "visual_contact_geoms": ["floor", "support"],
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
            for name in ("source_geom", "target_geom", "support", "floor")
        },
        "geom_geometry": {
            name: {
                "body": source.body(source.geom(name).bodyid[0]).name,
                **{
                    k: np.atleast_1d(getattr(source.geom(name), k)).tolist()
                    for k in ("type", "size", "pos", "quat")
                },
            }
            for name in ("source_geom", "target_geom", "support", "floor")
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
    floor = robot.model.geom("floor")
    visible_floor = robot.model.geom("libero_floor_visual")
    assert floor.group[0] == 3 and floor.contype[0] == 1
    assert visible_floor.group[0] == 2
    assert visible_floor.contype[0] == visible_floor.conaffinity[0] == 0
    for field in ("type", "size", "pos", "quat", "rgba", "matid"):
        np.testing.assert_array_equal(getattr(visible_floor, field), getattr(floor, field))
    support = robot.model.geom("support")
    visible_support = robot.model.geom("libero_visual_support")
    assert visible_support.group[0] == 2
    assert visible_support.contype[0] == visible_support.conaffinity[0] == 0
    for field in ("type", "size", "pos", "quat", "rgba", "matid"):
        np.testing.assert_array_equal(getattr(visible_support, field), getattr(support, field))
    data = robot.data()
    np.testing.assert_allclose(data.body("source").xpos, [0.4, -0.12, 0.35], atol=1e-12)
    np.testing.assert_allclose(
        data.body("source").xmat.reshape(3, 3), robot.scene.T_world_libero[:3, :3], atol=1e-12
    )
    assert not robot.scene.goal_status(robot, data)["passed"]


def test_object_layout_translation_uses_world_axes_and_preserves_model(scene_robot):
    original = scene_robot.data()
    offset = np.array([-0.1, 0.02, 0.0])
    scene = LiberoScene(
        scene_robot.scene.directory, 0,
        {**scene_robot.scene.configuration, "initial_object_translation_world_m": offset.tolist()},
    )
    robot = replace(scene_robot, scene=scene)
    data = robot.data()
    for name in ("source", "target"):
        np.testing.assert_allclose(
            data.body(name).xpos, original.body(name).xpos + offset, atol=1e-12
        )
        np.testing.assert_array_equal(data.body(name).xmat, original.body(name).xmat)
    for name in ("support", "floor"):
        np.testing.assert_array_equal(data.geom(name).xpos, original.geom(name).xpos)
    np.testing.assert_array_equal(
        data.body("left_tcp").xpos, original.body("left_tcp").xpos
    )
    # Reapplying a reset does not accumulate translations.
    scene.initialize(robot, data)
    mujoco.mj_forward(robot.model, data)
    np.testing.assert_allclose(
        data.body("source").xpos, original.body("source").xpos + offset, atol=1e-12
    )
    assert scene.validate(robot)["passed"]
    assert scene.manifest["configuration"]["initial_object_translation_world_m"] == offset.tolist()


@pytest.mark.parametrize("offset", [[1, 2], [0, 0, 0.01], [float("nan"), 0, 0]])
def test_object_layout_rejects_invalid_or_vertical_translation(scene_robot, offset):
    with pytest.raises(ValueError, match="Initial object translation"):
        LiberoScene(
            scene_robot.scene.directory, 0,
            {**scene_robot.scene.configuration, "initial_object_translation_world_m": offset},
        )


@pytest.mark.parametrize("with_trajectory", [False, True])
def test_scene_view_starts_and_resets_after_physical_settling(scene_robot, with_trajectory):
    import copy

    from alohamini_mujoco.control import DynamicFollower
    from alohamini_mujoco.libero import RESET_SETTLE_S
    from alohamini_mujoco.viewer import DynamicPlayback

    scene = copy.copy(scene_robot.scene)
    scene.task = copy.deepcopy(scene.task)
    scene.task["initial_states"][0]["source_joint"][2] += 0.07
    robot = replace(scene_robot, scene=scene)
    raw_height = robot.data().body("source").xpos[2]
    expected = DynamicFollower(robot)
    for _ in range(round(RESET_SETTLE_S / robot.model.opt.timestep)):
        expected.step()
    mujoco.mj_forward(robot.model, expected.data)
    trajectory = (
        JointTrajectory(robot, ["left_shoulder_pan"], [0, 0.2], [[0], [0.01]])
        if with_trajectory
        else None
    )
    playback = DynamicPlayback(robot, trajectory, paused=True)
    assert playback.data.time == pytest.approx(RESET_SETTLE_S)
    assert playback.data.body("source").xpos[2] < raw_height - 0.06
    np.testing.assert_array_equal(playback.data.qpos, expected.data.qpos)
    np.testing.assert_array_equal(playback.data.qvel, expected.data.qvel)
    initial = physics_state(robot.model, playback.data)
    gains = playback.follower.kp.copy()
    playback.step(single=True)
    playback.reset()
    np.testing.assert_array_equal(physics_state(robot.model, playback.data), initial)
    np.testing.assert_array_equal(playback.follower.kp, gains)
    if with_trajectory:
        playback.paused = False
        while not playback.completed:
            playback.step()
        assert playback.data.time == pytest.approx(RESET_SETTLE_S + playback.duration_s)


def test_view_and_collection_reject_robot_contact_during_reset(scene_robot, tmp_path):
    import copy

    from alohamini_mujoco.viewer import DynamicPlayback

    scene = copy.copy(scene_robot.scene)
    scene.task = copy.deepcopy(scene.task)
    position = scene_robot.data().body("left_Fixed_Jaw").xpos.copy()
    local = np.linalg.inv(scene.T_world_libero) @ np.r_[position, 1]
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(scene.T_world_libero[:3, :3].T).ravel())
    scene.task["initial_states"][0]["source_joint"] = np.r_[local[:3], quaternion].tolist()
    robot = replace(scene_robot, scene=scene)
    trajectory = JointTrajectory(robot, ["left_shoulder_pan"], [0, 0.2], [[0], [0.01]])
    with pytest.raises(ValueError, match="Scene reset collision"):
        DynamicPlayback(robot, paused=True)
    output = tmp_path / "invalid_reset"
    with pytest.raises(ValueError, match="Scene reset collision"):
        collect_episode(robot, trajectory, output, cameras=("agentview",))
    assert not output.exists()


@pytest.fixture
def reference_scene(scene_robot, tmp_path):
    """Synthetic reference for frame/contact tests; task admission is tested separately."""
    import copy
    import hashlib

    import h5py
    from alohamini_mujoco.collection import STATE_SPEC
    from alohamini_mujoco.control import DynamicFollower
    from alohamini_mujoco.export import model_xml
    from alohamini_mujoco.retarget import MotionRecipe

    scene = copy.copy(scene_robot.scene)
    scene.task = copy.deepcopy(scene.task)
    scene.task["objects"]["source"]["joints"] = ["source_joint"]
    scene.task["objects"]["target"]["joints"] = ["target_joint"]
    model = copy.copy(scene_robot.model)
    model.opt.gravity[:] = 0
    xml = ET.fromstring(scene_robot.xml)
    xml.find("option").set("gravity", "0 0 0")
    initial_positions = {
        **scene_robot.initial_positions,
        "left_shoulder_lift": -np.pi / 4,
        "left_elbow_flex": np.pi / 2,
        "left_wrist_flex": -np.pi / 4,
    }
    robot = replace(
        scene_robot,
        model=model,
        xml=ET.tostring(xml, encoding="unicode"),
        scene=scene,
        initial_positions=initial_positions,
        manifest={**scene_robot.manifest, "initial_positions": initial_positions},
    )
    follower = DynamicFollower(robot)
    directory = tmp_path / "reference"
    (directory / "model").mkdir(parents=True)
    portable = model_xml(robot)
    (directory / "model/model.xml").write_text(portable)
    names = follower.names
    actions = np.tile(follower.target, (81, 1))
    # Create a nominal TCP below the measured TCP. Previewing commands as
    # measured poses would incorrectly push the supported source through its table.
    joint = names.index("vertical_move")
    actions[:, joint] -= 0.001
    states = []
    for frame in range(82):
        follower.data.time = frame / 20
        states.append(physics_state(model, follower.data))
    with h5py.File(directory / "episode.hdf5", "w") as file:
        file.attrs["env_args"] = json.dumps(robot.manifest)
        group = file.create_group("data/demo_0")
        group.attrs["joint_names"] = json.dumps(names)
        group.attrs["controller"] = json.dumps({"settings": follower.settings.to_dict()})
        group.create_dataset("actions", data=actions)
        group.create_dataset("states", data=states)
    (directory / "trajectory.json").write_text(json.dumps({"times_s": [0, 4.0]}))
    manifest = {
        "format": "alohamini-sim-v1",
        "admitted": True,
        "split": "train",
        "quality": {"replay": {"passed": True}},
        "mujoco_version": mujoco.__version__,
        "state_spec": int(STATE_SPEC),
        "collection_settings": {"fps": 20},
        "environment": robot.manifest,
        "episode": "episode.hdf5",
        "episode_sha256": sha256(directory / "episode.hdf5"),
        "model_files": {"model/model.xml": hashlib.sha256(portable.encode()).hexdigest()},
    }
    (directory / "dataset.json").write_text(json.dumps(manifest))
    return robot, directory, MotionRecipe("left", 20, 25, 30, 40, 60, 70)


def test_reference_contact_preview_accounts_for_servo_tracking(reference_scene):
    from alohamini_mujoco.retarget import ReferenceMotion

    robot, directory, recipe = reference_scene
    reference = ReferenceMotion(robot, directory, recipe)
    before = physics_state(robot.model, robot.data())
    trajectory, settings, generation = reference.plan(robot)
    assert trajectory.positions.shape == (81, robot.model.nu)
    assert generation["source_object"] == "source"
    assert settings == reference.settings
    # Neither the live reset nor the inactive arm is moved by planning.
    np.testing.assert_array_equal(physics_state(robot.model, robot.data()), before)
    for index, name in enumerate(trajectory.names):
        if name.startswith("right_"):
            np.testing.assert_array_equal(
                trajectory.positions[:, index], trajectory.positions[0, index]
            )


def test_reference_can_hold_lift_through_placement_and_retreat(reference_scene, monkeypatch):
    from alohamini_mujoco import retarget

    robot, directory, recipe = reference_scene
    reference = retarget.ReferenceMotion(robot, directory, replace(recipe, allow_lift_motion=False))
    solve = retarget.solve_ik_targets
    calls = []

    def arm_only(*args, **kwargs):
        assert "vertical_move" not in kwargs["movable_joints"]
        calls.append(tuple(kwargs["movable_joints"]))
        return solve(*args, **kwargs)

    monkeypatch.setattr(retarget, "solve_ik_targets", arm_only)
    trajectory, _, generation = reference.plan(robot)
    assert len(calls) > recipe.release_start_frame - recipe.pregrasp_end_frame
    lift = trajectory.names.index("vertical_move")
    np.testing.assert_array_equal(trajectory.positions[:, lift], trajectory.positions[0, lift])
    assert generation["recipe"]["allow_lift_motion"] is False
    data = robot.data()
    for frame in (21, 61, 80):
        data.qpos[reference.qpos_ids] = trajectory.positions[frame]
        mujoco.mj_kinematics(robot.model, data)
        np.testing.assert_allclose(
            data.body("left_tcp").xpos, reference.command_tcp[frame][:3, 3], atol=1e-5, rtol=0
        )


def test_fixed_lift_failure_does_not_fall_back_to_moving_lift(reference_scene, monkeypatch):
    from alohamini_mujoco import retarget

    robot, directory, recipe = reference_scene
    reference = retarget.ReferenceMotion(robot, directory, replace(recipe, allow_lift_motion=False))
    calls = []

    def failed_plan(robot, *, coordinated, progress):
        calls.append(coordinated)
        raise retarget._PlanningFailure("No arm-only solution")

    monkeypatch.setattr(reference, "_plan", failed_plan)
    with pytest.raises(RuntimeError, match="No arm-only solution"):
        reference.plan(robot)
    assert calls == [False]


def test_motion_recipe_reads_optional_lift_constraint(tmp_path):
    from alohamini_mujoco.retarget import MotionRecipe

    specification = {"schema_version": 1, "side": "left", "pregrasp_end_frame": 20,
                     "approach_end_frame": 25, "grasp_start_frame": 30, "source_end_frame": 40,
                     "target_start_frame": 60, "release_start_frame": 70}
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(specification))
    assert MotionRecipe.read(path).allow_lift_motion is True
    specification["allow_lift_motion"] = False
    path.write_text(json.dumps(specification))
    assert MotionRecipe.read(path).allow_lift_motion is False
    specification["allow_lift_motion"] = "false"
    path.write_text(json.dumps(specification))
    with pytest.raises(ValueError, match="allow_lift_motion"):
        MotionRecipe.read(path)


def test_reference_targets_follow_the_object_frame_in_world_coordinates(reference_scene):
    import copy

    from alohamini_mujoco.retarget import ReferenceMotion

    robot, directory, recipe = reference_scene
    reference = ReferenceMotion(robot, directory, recipe)
    scene = copy.copy(robot.scene)
    scene.task = copy.deepcopy(scene.task)
    world_shift = np.array([0.02, 0, 0])
    local_shift = scene.T_world_libero[:3, :3].T @ world_shift
    values = scene.task["initial_states"][0]["source_joint"]
    values[:3] = (np.array(values[:3]) + local_shift).tolist()
    target = replace(robot, scene=scene)
    trajectory, _, _ = reference.plan(target)
    data = target.data()
    data.qpos[reference.qpos_ids] = trajectory.positions[21]
    mujoco.mj_kinematics(target.model, data)
    np.testing.assert_allclose(
        data.body("left_tcp").xpos,
        reference.command_tcp[21][:3, 3] + world_shift,
        atol=1e-5,
        rtol=0,
    )


def test_reference_rejects_changed_physics_or_corrupted_episode(reference_scene):
    from alohamini_mujoco.retarget import ReferenceMotion

    robot, directory, recipe = reference_scene
    changed = ET.fromstring(robot.xml)
    changed.find("option").set("timestep", ".002")
    with pytest.raises(ValueError, match="physics differ"):
        ReferenceMotion(
            replace(robot, xml=ET.tostring(changed, encoding="unicode")), directory, recipe
        )
    with (directory / "episode.hdf5").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="episode checksum"):
        ReferenceMotion(robot, directory, recipe)


def test_reference_batch_records_failures_and_resumes_completed_cases(
    reference_scene, tmp_path, monkeypatch
):
    from alohamini_mujoco import collection
    from alohamini_mujoco.retarget import ReferenceMotion

    robot, directory, recipe = reference_scene
    calls = []

    def plan(reference, current, **kwargs):
        reset = current.scene.reset_index
        calls.append(reset)
        if reset == 1:
            raise RuntimeError("No collision-free approach")
        return None, None, {**reference.provenance, "reset_index": reset}

    def collect(current, trajectory, output, *, generation, **kwargs):
        output.mkdir()
        episode = output / "episode.bin"
        episode.write_bytes(b"unit storage fixture")
        manifest = {
            "generation": generation,
            "environment": current.manifest,
            "episode": episode.name,
            "episode_sha256": sha256(episode),
            "split": "train",
            "split_group": f"0:{current.scene.reset_index}",
            "admitted": True,
            "quality": {"passed": True},
        }
        (output / "dataset.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(ReferenceMotion, "plan", plan)
    monkeypatch.setattr(collection, "collect_episode", collect)
    output = tmp_path / "batch"
    report = collection.collect_reference_batch(robot, directory, recipe, [0, 1], output)
    assert not report["passed"] and report["admitted"] == 1 and report["failed_resets"] == [1]
    assert calls == [0, 1]
    assert (output / "rejected/reset_001.json").is_file()
    report = collection.collect_reference_batch(
        robot, directory, recipe, [0, 1], output, resume=True
    )
    assert calls == [0, 1]  # A failed completed attempt stays in the denominator.
    with pytest.raises(ValueError, match="unchanged reference"):
        collection.collect_reference_batch(robot, directory, recipe, [0], output, resume=True)
    (output / "reset_000/episode.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Completed batch episode changed"):
        collection.collect_reference_batch(robot, directory, recipe, [0, 1], output, resume=True)


def test_policy_playback_records_projected_targets_and_exact_failure_replay(
    reference_scene, tmp_path
):
    from alohamini_mujoco.control import DynamicFollower
    from alohamini_mujoco.evaluation import PolicyPlayback, _save_trial

    robot, _, _ = reference_scene
    follower = DynamicFollower(robot)
    command = follower.target.copy()
    index = follower.names.index("left_gripper")
    command[index] = robot.description.joints["left_gripper"].upper + 0.01
    playback = PolicyPlayback(
        robot,
        lambda data: command,
        reset_policy=lambda: None,
        follower_settings=FollowerSettings(),
        horizon_s=0.1,
    )
    while not playback.completed:
        playback.step(single=True)
    result = _save_trial(playback, tmp_path / "trial", "unit-policy", "unit-weights")
    assert result["replay"]["passed"] and result["replay"]["maximum_state_error"] == 0
    assert result["target_projection"]["left_gripper"]["maximum_violation"] == pytest.approx(0.01)
    assert all(q[index] == robot.description.joints["left_gripper"].upper for q in playback.actions)
    assert not result["success"]
    assert json.loads((tmp_path / "trial/dataset.json").read_text())["split"] == "evaluation"


def test_policy_playback_does_not_apply_nonfinite_actions(reference_scene):
    from alohamini_mujoco.evaluation import PolicyPlayback

    robot, _, _ = reference_scene
    playback = PolicyPlayback(
        robot,
        lambda data: np.full(robot.model.nu, np.nan),
        reset_policy=lambda: None,
        follower_settings=FollowerSettings(),
        horizon_s=0.1,
    )
    before = physics_state(robot.model, playback.data)
    assert not playback.step(single=True)
    np.testing.assert_array_equal(physics_state(robot.model, playback.data), before)
    assert playback.report()["failure"] == "invalid_policy_action"
    assert not playback.actions


@pytest.mark.parametrize("interactive", [False, True])
def test_evaluation_uses_si_observations_and_reports_task_failure(
    reference_scene, tmp_path, monkeypatch, interactive
):
    import hashlib
    from contextlib import nullcontext
    from dataclasses import asdict
    from types import SimpleNamespace

    import torch
    from alohamini_mujoco.collection import CollectionSettings
    from alohamini_mujoco.control import DynamicFollower
    from alohamini_mujoco.evaluation import evaluate_policy
    from alohamini_mujoco.export import model_xml

    from alohamini.learning import policy as native_policy
    from alohamini.learning.state import RecordedStateSelection

    robot, _, _ = reference_scene
    follower = DynamicFollower(robot)
    units = ["m" if k == "prismatic" else "rad" for k in follower.kinds]
    feature = {
        "dtype": "float32",
        "shape": [len(follower.names)],
        "names": list(follower.names),
        "units": units,
    }
    source = {
        "training_only": {"source": "alohamini_mujoco"},
        "features": {
            "action": feature,
            "observation.state": feature,
            "observation.images.agentview": {"shape": [32, 32, 3]},
        },
        "simulation": {
            "model_xml_sha256": [hashlib.sha256(model_xml(robot).encode()).hexdigest()],
            "initial_positions": [robot.initial_positions],
            "follower_settings": FollowerSettings().to_dict(),
            "collection_settings": asdict(CollectionSettings(image_size=32)),
        },
    }
    observed = []

    def select_action(batch):
        observed.append(batch["observation.state"].clone())
        assert batch["observation.images.agentview"].shape == (1, 3, 32, 32)
        return torch.tensor(follower.target[None], dtype=torch.float32)

    policy = SimpleNamespace(
        checkpoint_path=tmp_path / "checkpoint",
        source=source,
        names=list(follower.names),
        fps=20,
        field_transform=None,
        checkpoint_manifest_sha256="unit-policy",
        selection=RecordedStateSelection(source, "recorded"),
        manifest={"cameras": ["agentview"], "image_size": [32, 32]},
        algorithm=SimpleNamespace(include_task=False),
        model=SimpleNamespace(select_action=select_action),
        processor=lambda batch: batch,
        execution_action=lambda value, **kwargs: value,
        inference_context=nullcontext,
        reset=lambda: None,
    )
    policy.checkpoint_path.mkdir()
    (policy.checkpoint_path / "model.safetensors").write_bytes(b"unit-weights")
    monkeypatch.setattr(native_policy, "NativePolicy", lambda *args, **kwargs: policy)
    if interactive:
        from alohamini_mujoco import evaluation

        # Closing between 20 Hz boundaries still produces a replayable trace.
        monkeypatch.setattr(
            evaluation,
            "run_viewer",
            lambda robot, *, playback, **kwargs: playback.step(single=True),
        )
    report = evaluate_policy(
        robot,
        tmp_path / "checkpoint",
        [0],
        tmp_path / "evaluation",
        horizon_s=0.1,
        view=interactive,
    )
    assert report["completed_trials"] == 1 and report["success_rate"] == 0
    assert report["trials"][0]["failure"] == (
        "interactive_interrupted" if interactive else "horizon_exceeded"
    )
    assert report["trials"][0]["frames"] == (1 if interactive else 2)
    assert report["trials"][0]["replay"]["maximum_state_error"] == 0
    np.testing.assert_allclose(observed[0][0].numpy(), follower.target, atol=1e-6)
    if not interactive:
        batch = evaluate_policy(
            robot, policy.checkpoint_path, [1, 0], tmp_path / "batch", horizon_s=0.1
        )
        assert batch["complete"] and batch["requested_trials"] == 2
        assert batch["reset_failures"] == 1 and batch["success_rate"] == 0
        assert batch["trials"][0]["failure"] == "scene_reset_rejected"
        assert batch["trials"][0]["frames"] == 0
        assert batch["trials"][1]["replay"]["maximum_state_error"] == 0


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


@pytest.mark.parametrize("damage", [None, "version", "manifest"])
def test_scene_retains_verified_export_versions(scene_robot, tmp_path, damage):
    import shutil

    directory = tmp_path / "snapshot"
    shutil.copytree(scene_robot.scene.directory, directory)
    packages = {"mujoco": "3.8.1", "robosuite": "1.4.0"}
    bundle = json.loads((directory / "bundle.json").read_text())
    bundle["upstream_packages"] = dict(packages)
    if damage == "version":
        packages["mujoco"] = "3.12.0"
        bundle["upstream_packages"] = dict(packages)
    elif damage == "manifest":
        bundle["upstream_packages"]["robosuite"] = "1.5.0"
    (directory / "upstream.json").write_text(json.dumps({"upstream_packages": packages}))
    bundle["files"]["upstream.json"] = sha256(directory / "upstream.json")
    (directory / "bundle.json").write_text(json.dumps(bundle))
    if damage is None:
        scene = LiberoScene(directory, 0, scene_robot.scene.configuration)
        assert scene.manifest["upstream_packages"] == packages
    else:
        with pytest.raises(ValueError, match="export environment"):
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


def test_static_scene_rejects_invalid_distance_queries(scene_robot, monkeypatch):
    from alohamini_mujoco import libero
    from alohamini_mujoco.native_distance import InvalidNativeDistance

    query = libero.checked_geom_distance
    base = scene_robot.model.geom("base_cad_link__collision_0").id

    def invalid_base(model, data, first, second, maximum):
        if first == base:
            raise InvalidNativeDistance("Distance contradicts an independent projection")
        return query(model, data, first, second, maximum)

    monkeypatch.setattr(libero, "checked_geom_distance", invalid_base)
    report = scene_robot.scene.static_intersections(scene_robot)
    assert not report["passed"] and report["invalid_distance_queries"]
    assert report["invalid_distance_queries"][0]["robot_geom"] == "base_cad_link__collision_0"


def test_collection_rejects_failure_preserves_alignment_and_replays(
    scene_robot, tmp_path, monkeypatch
):
    pytest.importorskip("h5py")
    import h5py
    from alohamini_mujoco import export as model_export

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

    monkeypatch.setattr(model_export, "export_robot", export)
    grip = scene_robot.initial_positions["left_gripper"]
    trajectory = JointTrajectory(
        scene_robot,
        ["left_shoulder_pan", "left_gripper"],
        [0, 0.2],
        [[0, grip], [0.005, grip + 0.005]],
    )
    controller = FollowerSettings(joint_gains={"left_gripper": {"kp": 2, "kd": 0.02}})
    output = tmp_path / "collection"
    report = collect_episode(
        scene_robot, trajectory, output, follower_settings=controller, cameras=("agentview",)
    )
    assert not report["passed"]
    assert not report["quality"]["checks"]["bilateral_grasp"]
    assert report["quality"]["replay"]["passed"]
    playback = EpisodeReplay(output, paused=True)
    initial_state = physics_state(playback.robot.model, playback.data)
    assert not playback.step()
    assert not playback.report()["completed"]
    for _ in range(playback.steps_per_command):
        assert playback.step(single=True)
        np.testing.assert_array_equal(playback.follower.target, playback.actions[0])
        np.testing.assert_array_equal(playback.follower.target_velocity, 0)
    assert playback.frame_index == 1
    assert playback.maximum_error == 0
    playback.reset()
    assert playback.paused and not playback.completed
    assert playback.frame_index == playback.substep == 0
    np.testing.assert_array_equal(physics_state(playback.robot.model, playback.data), initial_state)
    np.testing.assert_array_equal(playback.follower.kp, playback.specification["kp"])
    np.testing.assert_array_equal(playback.follower.kd, playback.specification["kd"])
    manifest = json.loads((output / "dataset.json").read_text())
    assert not (output / "episodes").exists()
    episode = output / manifest["episode"]
    with h5py.File(episode, "r") as file:
        group = file["data/demo_0"]
        saved_controller = json.loads(group.attrs["controller"])
        assert saved_controller["settings"] == controller.to_dict()
        names = json.loads(group.attrs["joint_names"])
        assert saved_controller["kp"][names.index("left_gripper")] == 2
        assert len(group["states"]) == len(group["actions"]) + 1
        np.testing.assert_allclose(np.diff(group["simulation_time_s"]), 0.05, atol=1e-12)
        assert group["obs/agentview_rgb"].dtype == np.uint8
        assert group["obs/agentview_rgb"].shape[1:] == (128, 128, 3)
        # Verify actual pixels: disabling the markers must not just be metadata.
        from alohamini_mujoco.viewer import set_geometry

        initial = scene_robot.data()
        restore_state(scene_robot.model, initial, group["states"][0])
        options = mujoco.MjvOption()
        set_geometry(options, collisions=False)
        with mujoco.Renderer(scene_robot.model, height=128, width=128) as renderer:
            renderer.update_scene(initial, camera="agentview", scene_option=options)
            marked = renderer.render().copy()
            options.sitegroup[:] = 0
            renderer.update_scene(initial, camera="agentview", scene_option=options)
            clean = renderer.render().copy()
        np.testing.assert_array_equal(group["obs/agentview_rgb"][0], clean)
        assert not np.array_equal(marked, clean)
        assert manifest["observation_rendering"]["site_groups"] == []
        data = scene_robot.data()
        ids = [
            scene_robot.model.jnt_qposadr[scene_robot.model.joint(n).id]
            for n in json.loads(group.attrs["joint_names"])
        ]
        for i in range(len(group["actions"])):
            restore_state(scene_robot.model, data, group["states"][i])
            np.testing.assert_array_equal(group["obs/joint_positions"][i], data.qpos[ids])
    # Replay must carry the same fixture restrictions as live following.
    environment = manifest["environment"]
    manifest["environment"] = {**environment, "rigid_assembly_fixture": {}}
    (output / "dataset.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="environment differs"):
        replay_episode(output)
    with h5py.File(episode, "r+") as file:
        file.attrs["env_args"] = json.dumps(manifest["environment"])
    manifest["episode_sha256"] = sha256(episode)
    (output / "dataset.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cannot run dynamics"):
        replay_episode(output)
    with h5py.File(episode, "r+") as file:
        file.attrs["env_args"] = json.dumps(environment)
    manifest["environment"] = environment
    manifest["episode_sha256"] = sha256(episode)
    (output / "dataset.json").write_text(json.dumps(manifest))
    with h5py.File(episode, "r+") as file:
        file["data/demo_0/actions"][0, 0] += 0.1
    with pytest.raises(ValueError, match="Episode checksum"):
        replay_episode(output)

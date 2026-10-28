import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from alohamini.datasets.action_transforms import RelativeActions
from alohamini.datasets.field_transforms import FieldTransforms
from alohamini.datasets.kinematic_fields import (
    KinematicActions,
    KinematicFields,
    descriptor_from_spec,
)
from alohamini.kinematics import OmniBaseKinematics
from alohamini.kinematics.arm import AlohaMiniArmKinematics, SerialArmKinematics


def descriptor():
    arms, names = [], []
    for side in ("left", "right"):
        fields = [f"{side}.q{i}" for i in range(6)]
        grip = f"{side}.gripper"
        names += [*fields, grip]
        arms.append(
            dict(
                name=side,
                frame=f"{side}_Base",
                geometry=AlohaMiniArmKinematics.from_description(side).parameters(),
                coordinates=[dict(field=name, unit="rad") for name in fields],
                gripper=dict(
                    field=grip,
                    source_closed=0,
                    source_open=100,
                    closed=0,
                    open=0.1,
                    unit="canonical_0_0.1",
                ),
            )
        )
    return dict(
        version=1,
        source_names=names + ["x.vel", "y.vel", "theta.vel", "lift.height"],
        arms=arms,
        passthrough=["x.vel", "y.vel", "theta.vel", "lift.height"],
        passthrough_units=["m/s", "m/s", "deg/s", "mm"],
    )


def joint_state():
    return np.r_[
        [-0.4, -1.0, 1.0, -0.3, 0.2, -0.1, 40],
        [0.4, -1.0, 1.0, -0.3, 0.2, -0.1, 60],
        [0.1, -0.2, 10, 250],
    ]


def test_shared_geometry_matches_independent_golden_poses():
    path = (
        Path(__file__).resolve().parents[1] / "ros2/src/alohamini_validation/config/fk_golden.yaml"
    )
    golden = yaml.safe_load(path.read_text())
    inputs = np.asarray([sample["q_rad"] for sample in golden["samples"].values()])
    expected = np.asarray([sample["transform"] for sample in golden["samples"].values()])
    for side in ("left", "right"):
        model = SerialArmKinematics.from_model("alohamini2pro", side)
        for sample in golden["samples"].values():
            np.testing.assert_allclose(
                model.forward(sample["q_rad"]), sample["transform"], atol=float(golden["tolerance"])
            )
        np.testing.assert_allclose(
            model.forward_batch(inputs), expected, atol=float(golden["tolerance"])
        )
        np.testing.assert_allclose(
            model.forward_batch(inputs[None])[0], expected, atol=float(golden["tolerance"])
        )
    assert OmniBaseKinematics(0.1, 0.2).wheel_radius_m == 0.1


def test_fk_is_a_named_nonmutating_dataset_operation():
    fields = KinematicFields(descriptor())
    source = joint_state()
    row = {"observation.state": source, "timestamp": 1.25}
    result = fields.derive(row, {"observation.state": "observation.tcp"})
    assert result["observation.state"] is source and result["timestamp"] == 1.25
    assert "observation.tcp" not in row
    assert result["observation.tcp"][6] == pytest.approx(0.04)
    np.testing.assert_equal(result["observation.tcp"][-4:], source[-4:])
    feature = fields.feature()
    assert feature["frames"] == {"left": "left_Base", "right": "right_Base"}
    assert feature["units"][:7] == ["m", "m", "m", "rad", "rad", "rad", "canonical_0_0.1"]
    with pytest.raises(ValueError, match="overwrite"):
        fields.derive(row, {"observation.state": "timestamp"})


def test_relative_pose_rotation_is_composed_and_all_targets_share_anchor():
    codec = RelativeActions(7, linear_dims=[0, 1, 2], rotation_groups=[[3, 4, 5]])
    anchor = np.array([0.3, 0.1, -0.2, 0.6, -0.2, 0.1, 0.04])
    targets = np.array(
        [[0.4, 0.2, 0.1, -0.1, 0.4, 0.3, 0.07], [0.5, 0.3, 0.2, 0.3, 0.1, -0.4, 0.02]]
    )
    encoded = codec.encode(anchor, targets)
    np.testing.assert_allclose(encoded[:, :3], [[0.1, 0.1, 0.3], [0.2, 0.2, 0.4]])
    np.testing.assert_equal(encoded[:, 6], targets[:, 6])
    delta = Rotation.from_rotvec(targets[:, 3:6]) * Rotation.from_rotvec(anchor[3:6]).inv()
    np.testing.assert_allclose(Rotation.from_rotvec(encoded[:, 3:6]).as_matrix(), delta.as_matrix())
    assert not np.allclose(encoded[:, 3:6], targets[:, 3:6] - anchor[3:6])
    np.testing.assert_allclose(codec.decode(anchor, encoded), targets, atol=1e-12)
    batch = codec.encode(np.stack([anchor, anchor]), np.stack([targets, targets]))
    np.testing.assert_allclose(batch[0], encoded)


def test_geometry_snapshot_and_fk_ik_preserve_full_platform_axes(monkeypatch):
    spec = json.loads(json.dumps(descriptor()))
    fields = KinematicFields(spec)
    anchor = joint_state()
    actions = np.repeat(anchor[None], 3, axis=0)
    actions[:, 0] += np.array([0.005, 0.01, 0.02])
    actions[:, 7] += np.array([-0.005, -0.01, -0.02])
    actions[:, -4:] += [0.02, 0.03, 2.0, 10.0]
    monkeypatch.setattr(
        AlohaMiniArmKinematics,
        "from_description",
        lambda *a: pytest.fail("Live model assets must not override frozen geometry"),
    )
    frozen = KinematicFields(spec)
    restored = frozen.restore_absolute(fields.absolute(actions), anchor)
    np.testing.assert_allclose(frozen.absolute(restored), fields.absolute(actions), atol=1e-4)
    np.testing.assert_equal(restored[:, -4:], actions[:, -4:])
    np.testing.assert_allclose(restored[:, [6, 13]], actions[:, [6, 13]])


def test_ik_rejects_unreachable_target_and_bad_transform():
    model = AlohaMiniArmKinematics.from_description()
    target = model.forward(joint_state()[7:13])
    target[:3, 3] += 10
    with pytest.raises(ValueError, match="did not converge"):
        model.inverse(target, joint_state()[7:13], max_iterations=3)
    target[3, 0] = 1
    with pytest.raises(ValueError, match="rigid"):
        model.inverse(target, joint_state()[7:13])


def test_derived_contract_rejects_duplicate_fields_and_implicit_units():
    spec = descriptor()
    spec["passthrough"] += ["x.vel"]
    with pytest.raises(ValueError, match="exactly once"):
        KinematicFields(spec)
    spec = descriptor()
    spec["arms"][0]["coordinates"][0]["unit"] = "normalized"
    with pytest.raises(ValueError, match="explicitly calibrated"):
        KinematicFields(spec)
    with pytest.raises(ValueError, match="distinct"):
        RelativeActions(7, linear_dims=[0], rotation_groups=[[0, 1, 2]])
    spec = deepcopy(descriptor())
    spec["arms"][0]["gripper"]["source_open"] = 0
    with pytest.raises(ValueError, match="endpoints"):
        KinematicFields(spec)


def test_inverse_host_mapping_does_not_silently_clip_an_ik_solution():
    coordinate = dict(
        unit="host",
        host_units=dict(
            normalization="range_m100_100", range_min=2000, range_max=2100, drive_mode=0
        ),
        encoder=dict(
            ticks_per_revolution=4096,
            reference_tick=2048,
            reference_position_rad=0,
            direction=1,
            position_min_rad=-3.14,
            position_max_rad=3.14,
        ),
    )
    with pytest.raises(ValueError, match="installed Host encoder range"):
        KinematicFields.source_joint(1.0, coordinate)


def test_host_descriptor_supports_recordings_annotated_v3_and_action_only_fields(tmp_path):
    from test_dataset import metadata

    from alohamini.datasets.kinematic_fields import descriptor_from_dataset, descriptor_from_host
    from alohamini.datasets.record import dataset_features

    info = dict(robot_metadata=metadata(), features=dataset_features("alohamini2pro"))
    for motor in info["robot_metadata"]["motors"].values():
        motor.update(range_min=0, range_max=4095, drive_mode=0, homing_offset=0)
    mappings = {}
    for side in ("left", "right"):
        joints = {}
        for name in (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_yaw",
            "wrist_roll",
            "gripper",
        ):
            motor = info["robot_metadata"]["motors"][f"arm_{side}_{name}"]
            joints[name] = dict(
                reference_tick=2048,
                reference_q_rad=0.0,
                sign=1,
                motor_calibration={
                    key: motor[key]
                    for key in ("homing_offset", "range_min", "range_max", "drive_mode")
                },
            )
        joints["gripper"].update(urdf_open_rad=0.5, urdf_closed_rad=0.0)
        mappings[side] = dict(ticks_per_revolution=4096, joints=joints)
    full = descriptor_from_host(info, mappings)
    assert len(full["arms"]) == 2 and len(full["source_names"]) == 18
    assert full["arms"][0]["coordinates"][5]["encoder"]["position_min_rad"] is not None
    action_only = deepcopy(info)
    action_only["features"].pop("observation.state")
    assert descriptor_from_host(action_only, mappings) == full
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps(dict(codebase_version="v3.0", features=action_only["features"]))
    )
    (meta / "alohamini.json").write_text(json.dumps(dict(version=1, source_info=info)))
    assert descriptor_from_dataset(tmp_path, mappings) == full
    incomplete = deepcopy(mappings)
    incomplete["left"]["joints"]["shoulder_pan"].pop("motor_calibration")
    with pytest.raises(ValueError, match="lacks motor_calibration"):
        descriptor_from_host(info, incomplete)
    stale = deepcopy(mappings)
    stale["left"]["joints"]["shoulder_pan"]["motor_calibration"]["homing_offset"] += 1
    with pytest.raises(ValueError, match="differs from recording"):
        descriptor_from_host(info, stale)
    for key in ("homing_offset", "drive_mode", "range_min", "range_max"):
        changed = deepcopy(info)
        changed["robot_metadata"]["motors"]["arm_left_shoulder_pan"][key] += 1
        with pytest.raises(ValueError, match="differ"):
            KinematicFields(full).validate_source(changed)
    (meta / "alohamini.json").unlink()
    with pytest.raises(ValueError, match="explicit coordinate descriptor"):
        descriptor_from_dataset(tmp_path, mappings)


def test_ros_tracker_keeps_its_tracking_policy_and_shared_geometry():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    path = root / "ros2/src/alohamini_joycon_teleop/alohamini_joycon_teleop/kinematics.py"
    spec = importlib.util.spec_from_file_location("robot_tracking_kinematics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tracker = module.AlohaMiniArmKinematics.from_description("right")
    assert isinstance(tracker, AlohaMiniArmKinematics)
    q = joint_state()[7:13]
    target = tracker.forward(q)
    target[0, 3] += 0.005
    candidate, metrics = tracker.step(q, target, 1 / 30, max_joint_step=0.01)
    assert np.max(np.abs(candidate - q)) <= 0.01 + 1e-12
    assert metrics["position_error_m"] == pytest.approx(0.005)


def planar_spec():
    return dict(
        version=1,
        source_names=["auxiliary", "elbow", "shoulder"],
        arms=[
            dict(
                name="manipulator",
                frame="fixture",
                coordinates=[dict(field="shoulder", unit="deg"), dict(field="elbow", unit="rad")],
                geometry=dict(
                    rows=[[0.4, 0, 0, 0], [0.3, 0, 0, 0]],
                    base_transform=np.eye(4).tolist(),
                    tool_transform=np.eye(4).tolist(),
                    joint_limits=[[-2, 2], [-2, 2]],
                ),
            )
        ],
        passthrough=["auxiliary"],
        passthrough_units=["mm"],
    )


def test_field_view_projects_history_without_changing_current_action_anchor():
    spec = planar_spec()
    info = {
        "features": {
            "action": {"names": spec["source_names"]},
            "observation.state": {"names": ["shoulder", "auxiliary", "elbow"]},
        }
    }
    spec = descriptor_from_spec(info, spec)
    selected = dict(dtype="float32", shape=[3], names=["elbow", "shoulder", "current"])
    fields = FieldTransforms(
        dict(
            state_representation="tcp",
            action_representation="tcp_delta",
            kinematic_fields=spec,
            selected_state_feature=selected,
            selected_state_units=["rad", "deg", "A"],
            source_state_names=["shoulder", "auxiliary", "elbow"],
        )
    )
    fields.validate_source(info, selected)
    anchor = np.array([250, -0.6, np.rad2deg(0.25)])
    np.testing.assert_array_equal(fields.context({"observation.state": anchor[[2, 0, 1]]}), anchor)
    # Three batch members, two observation times, two future action targets.
    history = np.array([[[-0.61, np.rad2deg(0.24), 1], [-0.6, np.rad2deg(0.25), 2]]] * 3)
    anchors = np.repeat(anchor[None], 3, axis=0)
    targets = np.repeat(anchors[:, None], 2, axis=1)
    targets[..., 2] += 0.005
    states, actions = fields.encode(history, targets, anchor=anchors)
    assert states.shape == (3, 2, 7)
    np.testing.assert_array_equal(states[..., -1], history[..., -1])
    np.testing.assert_allclose(actions[:, 0], actions[:, 1])
    np.testing.assert_allclose(fields.decode(anchors, actions), targets, atol=2e-3)
    features = fields.features(
        selected, ["rad", "deg", "A"], dict(dtype="float32", shape=[3], names=spec["source_names"])
    )
    assert features["observation.state"]["names"][-1] == "current"
    assert features["observation.state"]["units"][-1] == "A"
    bad = {
        **fields.configuration,
        "selected_state_feature": {
            **selected,
            "shape": [1],
            "names": ["shoulder"],
        },
        "selected_state_units": ["deg"],
    }
    with pytest.raises(ValueError, match="complete joint-position group"):
        FieldTransforms(bad)


def test_arbitrary_revolute_chain_field_order_units_and_optional_gripper():
    codec = KinematicActions(planar_spec())
    state = np.array([250, -0.6, np.rad2deg(0.25)])
    absolute, _ = codec.encode(state)
    expected = [
        0.4 * np.cos(0.25) + 0.3 * np.cos(-0.35),
        0.4 * np.sin(0.25) + 0.3 * np.sin(-0.35),
        0,
    ]
    np.testing.assert_allclose(absolute[:3], expected, atol=1e-12)
    np.testing.assert_allclose(absolute[3:6], [0, 0, -0.35], atol=1e-12)
    assert absolute[-1] == 250 and codec.geometry.width == 7
    actions = np.repeat(state[None], 2, axis=0)
    actions[:, 2] += np.rad2deg([0.01, 0.02])
    actions[:, 1] -= [0.01, 0.02]
    actions[:, 0] += [10, 20]
    _, delta = codec.encode(state, actions)
    np.testing.assert_array_equal(delta[:, -1], actions[:, 0])
    restored = codec.decode(state, delta)
    np.testing.assert_allclose(restored, actions, atol=2e-3)
    model = codec.geometry.arms[0][3]
    assert model.jacobian([0.25, -0.6]).shape == (6, 2)
    np.testing.assert_allclose(model.forward_batch([[0.25, -0.6]])[0], model.forward([0.25, -0.6]))
    assert codec.relative.encode(absolute, np.empty((0, 7))).shape == (0, 7)


def test_equal_joint_increments_do_not_remove_unknown_physical_zero():
    model = SerialArmKinematics.from_parameters(
        dict(
            rows=[[1, 0, 0, 0]],
            base_transform=np.eye(4).tolist(),
            tool_transform=np.eye(4).tolist(),
        )
    )
    displacements = [
        (model.forward([anchor + 0.1]) - model.forward([anchor]))[:3, 3]
        for anchor in (0, np.pi / 2)
    ]
    np.testing.assert_allclose(displacements[0], [np.cos(0.1) - 1, np.sin(0.1), 0], atol=1e-12)
    np.testing.assert_allclose(displacements[1], [-np.sin(0.1), np.cos(0.1) - 1, 0], atol=1e-12)
    assert not np.allclose(*displacements)


def test_generic_descriptor_cli_accepts_unannotated_model_coordinates_without_ros_maps(tmp_path):
    from alohamini.datasets.kinematic_fields import main

    spec = planar_spec()
    info = dict(
        codebase_version="v3.0",
        features={
            key: dict(names=spec["source_names"], shape=[3], dtype="float32")
            for key in ("action", "observation.state")
        },
    )
    meta = tmp_path / "dataset/meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps(info))
    source, output = tmp_path / "spec.json", tmp_path / "frozen.json"
    source.write_text(json.dumps(spec))
    args = ["--dataset", str(meta.parent), "--spec", str(source), "--output", str(output)]
    main(args)
    frozen = json.loads(output.read_text())
    assert frozen == spec
    assert "robot_model" not in frozen and "source_calibration" not in frozen
    with pytest.raises(FileExistsError):
        main(args)
    with pytest.raises(SystemExit):
        main(args + ["--right-map", "unused.yaml"])


def test_raw_encoder_coordinates_are_explicit_and_do_not_reapply_homing_offset():
    coordinate = dict(
        unit="encoder",
        encoder=dict(
            ticks_per_revolution=4096,
            reference_tick=1000,
            reference_position_rad=0.2,
            direction=-1,
            position_min_rad=-2,
            position_max_rad=2,
        ),
    )
    angle = 0.2 - 100 * 2 * np.pi / 4096
    assert KinematicFields.joint(1100, coordinate) == pytest.approx(angle)
    assert KinematicFields.source_joint(angle, coordinate) == 1100
    with pytest.raises(ValueError, match="integer post-offset"):
        KinematicFields.joint(1100.5, coordinate)


def test_custom_model_registers_its_own_geometry_and_freezes_it(tmp_path, monkeypatch):
    import shutil

    from alohamini.model import get_robot_model, load_robot_model

    installed = get_robot_model("alohamini2pro")
    custom = tmp_path / "arm_model"
    custom.mkdir()
    for relative in (
        "config/kinematics/right_arm_kinematics.yaml",
        "config/kinematics/kinematics.yaml",
        "urdf/alohamini2pro_kinematic.urdf",
    ):
        target = custom / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(installed.asset_path(relative), target)
    chain = json.loads(installed.kinematic_chain_path("left").read_text())
    (custom / "chain.json").write_text(json.dumps(chain))
    document = json.loads((installed.directory / "model.json").read_text())
    document.pop("kinematic_presets", None)
    document.update(
        model_id="bench_arm",
        descriptions={"kinematic": document["descriptions"]["kinematic"]},
        kinematic_chains={"manipulator": "chain.json"},
    )
    (custom / "model.json").write_text(json.dumps(document))
    model = load_robot_model(custom)
    assert SerialArmKinematics.from_model(model, "manipulator").frame == "left_Base"
    with pytest.raises(ValueError, match="no kinematic chain"):
        SerialArmKinematics.from_model("alohamini1", "left")
    spec = descriptor()
    spec["arms"] = [spec["arms"][0]]
    spec["source_names"], spec["passthrough"], spec["passthrough_units"] = (
        spec["source_names"][:7],
        [],
        [],
    )
    spec["arms"][0].pop("geometry")
    spec["arms"][0]["model"] = dict(directory="arm_model", chain="manipulator")
    info = dict(features={"action": dict(names=spec["source_names"])})
    frozen = descriptor_from_spec(info, spec, directory=tmp_path)
    assert frozen["arms"][0]["model_reference"]["model_id"] == "bench_arm"
    assert "model" not in frozen["arms"][0]
    shutil.rmtree(custom)
    monkeypatch.setattr(
        SerialArmKinematics,
        "from_model",
        lambda *a: pytest.fail("Frozen geometry must not access model assets"),
    )
    assert np.isfinite(KinematicFields(frozen).absolute(joint_state()[:7])).all()

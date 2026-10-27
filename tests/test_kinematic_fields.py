import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from alohamini.datasets.action_transforms import RelativeActions
from alohamini.datasets.kinematic_fields import KinematicFields
from alohamini.kinematics import OmniBaseKinematics
from alohamini.kinematics.arm import AlohaMiniArmKinematics


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
    model = AlohaMiniArmKinematics.from_description()
    for sample in golden["samples"].values():
        np.testing.assert_allclose(
            model.forward(sample["q_rad"]), sample["transform"], atol=float(golden["tolerance"])
        )
    inputs = np.asarray([sample["q_rad"] for sample in golden["samples"].values()])
    expected = np.asarray([sample["transform"] for sample in golden["samples"].values()])
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
        motor.update(range_min=0, range_max=4095, drive_mode=0)
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
            joints[name] = dict(reference_tick=2048, reference_q_rad=0.0, sign=1)
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

import json
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_dataset import metadata

from alohamini.calibration.encoder import HostPositionUnits
from alohamini.datasets.kinematic_fields import (
    KinematicActions,
    KinematicFields,
    descriptor_from_reference_pose,
    main,
)
from alohamini.datasets.record import dataset_features
from alohamini.kinematics.arm import SerialArmKinematics
from alohamini.model import get_robot_model


def recording_info():
    info = dict(robot_metadata=metadata(), features=dataset_features("alohamini2pro"))
    for i, motor in enumerate(info["robot_metadata"]["motors"].values()):
        motor.update(homing_offset=i - 100, range_min=1000, range_max=3094, drive_mode=0)
    return info


def test_joint_zero_change_preserves_fk_and_shifts_limits_without_mutation():
    original = SerialArmKinematics.from_model("alohamini2pro", "left")
    before = original.parameters()
    reference = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
    rebased = original.with_joint_zero(reference)
    delta = np.random.default_rng(42).uniform(-0.3, 0.3, size=(20, 6))
    np.testing.assert_allclose(
        rebased.forward_batch(delta), original.forward_batch(delta + reference), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(rebased.joint_limits) + reference[:, None], original.joint_limits
    )
    assert original.parameters() == before
    with pytest.raises(ValueError, match="reference angle"):
        original.with_joint_zero([0, 0])


@pytest.mark.parametrize("side", ["left", "right"])
def test_reference_pose_matches_upright_arm_forward_tool_and_horizontal_gripper(side):
    info = recording_info()
    descriptor = descriptor_from_reference_pose(info, "upright_forward")
    arm = next(a for a in descriptor["arms"] if a["name"] == side)
    reference = dict(
        zip(
            SerialArmKinematics.from_model("alohamini2pro", side).joint_names,
            arm["model_reference"]["joint_zero"],
            strict=True,
        )
    )
    model = get_robot_model("alohamini2pro")
    joints = {
        j.find("child").get("link"): j
        for j in ET.parse(model.description_path("kinematic")).getroot().findall("joint")
    }

    # Independent URDF composition, including the CAD-to-base_link rotation.
    def transform(link):
        if link == "base_link":
            return np.eye(4)
        joint = joints[link]
        origin = joint.find("origin")
        fixed = np.eye(4)
        if origin is not None:
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            fixed[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            fixed[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        motion = np.eye(4)
        if joint.get("type") == "revolute":
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            motion[:3, :3] = Rotation.from_rotvec(
                axis * reference.get(joint.get("name"), 0)
            ).as_matrix()
        return transform(joint.find("parent").get("link")) @ fixed @ motion

    # Align the CAD members, not the line joining their offset motor centers.
    upper = transform(f"{side}_Upper_Arm")[:3, :3] @ [0, 1, 0]
    forearm = transform(f"{side}_Lower_Arm")[:3, :3] @ [0, -1, 0]
    np.testing.assert_allclose(upper / np.linalg.norm(upper), [0, 0, 1], atol=5e-6)
    np.testing.assert_allclose(forearm / np.linalg.norm(forearm), [1, 0, 0], atol=5e-6)
    tcp = transform(f"{side}_tcp")
    np.testing.assert_allclose(tcp[:3, 0], [1, 0, 0], atol=5e-6)
    jaw_joint = joints[f"{side}_Moving_Jaw"]
    jaw_axis = transform(f"{side}_Moving_Jaw")[:3, :3] @ np.fromstring(
        jaw_joint.find("axis").get("xyz"), sep=" "
    )
    assert abs(jaw_axis @ [0, 0, 1]) == pytest.approx(1, abs=1e-9)
    frozen = SerialArmKinematics.from_parameters(arm["geometry"])
    np.testing.assert_allclose(
        transform(f"{side}_Base") @ frozen.forward(np.zeros(6)), tcp, atol=1e-9
    )


def test_reference_protocol_binds_each_recording_without_old_maps_and_keeps_platform_axes():
    info = recording_info()
    descriptor = descriptor_from_reference_pose(info, "upright_forward")
    fields = KinematicFields(descriptor)
    state = np.zeros(18)
    for i, name in enumerate(fields.names[:14]):
        motor = info["robot_metadata"]["motors"][name.removesuffix(".pos")]
        state[i] = HostPositionUnits(
            **{key: motor[key] for key in ("normalization", "range_min", "range_max", "drive_mode")}
        ).from_tick(2047)
    state[-4:] = [0.1, -0.2, 10, 250]
    targets = np.repeat(state[None], 2, axis=0)
    targets[:, 0] += [0.1, 0.2]
    targets[:, -1] += [10, 20]
    codec = KinematicActions(descriptor)
    _, delta = codec.encode(state, targets)
    restored = codec.decode(state, delta)
    np.testing.assert_allclose(fields.absolute(restored), fields.absolute(targets), atol=2e-3)
    np.testing.assert_equal(restored[:, -4:], targets[:, -4:])
    assert descriptor["limit_source"] == "recorded_host_encoder_range"
    assert all(
        c["encoder"]["reference_position_rad"] == 0
        for a in descriptor["arms"]
        for c in a["coordinates"]
    )
    changed = deepcopy(info)
    for motor in changed["robot_metadata"]["motors"].values():
        motor["homing_offset"] += 10
    fresh = descriptor_from_reference_pose(changed, "upright_forward")
    assert fresh["arms"] == descriptor["arms"]
    assert fresh["source_calibration"] != descriptor["source_calibration"]
    with pytest.raises(ValueError, match="differs from recording"):
        fields.validate_source(changed)


def test_reference_pose_cli_uses_recorded_metadata_and_requires_explicit_protocol(tmp_path):
    info = recording_info()
    meta = tmp_path / "dataset/meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps(info))
    output = tmp_path / "frozen.json"
    args = [
        "--dataset",
        str(meta.parent),
        "--reference-pose",
        "upright_forward",
        "--output",
        str(output),
    ]
    main(args)
    assert json.loads(output.read_text())["reference_pose"]["name"] == "upright_forward"
    with pytest.raises(FileExistsError):
        main(args)
    with pytest.raises(ValueError, match="no kinematic preset"):
        descriptor_from_reference_pose(info, "unknown")
    info["robot_metadata"]["motors"]["arm_left_shoulder_pan"].pop("homing_offset")
    with pytest.raises(ValueError, match="lacks the homing offset"):
        descriptor_from_reference_pose(info, "upright_forward")

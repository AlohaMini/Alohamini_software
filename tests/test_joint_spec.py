"""Fixed coordinates, inward endpoint handling and unchanged device calibration."""

import math
from copy import deepcopy

import pytest

from alohamini.calibration.encoder import EncoderCalibration
from alohamini.calibration.joint_space import DeviceJointSpace, StandardJointSpace
from alohamini.calibration.lift import LiftCalibration
from alohamini.model import get_robot_model
from alohamini.model.joint_spec import ModelJointSpecification


@pytest.fixture
def specification():
    return ModelJointSpecification.from_model()


def binding(*, standard=None, reference_tick=2048, direction=1, drive_mode=0, endpoint_loss=0):
    standard = standard or StandardJointSpace.from_specification()
    encoders, motors = {}, {}
    actuators = {m.name: m for m in get_robot_model("alohamini2pro").actuators}
    for name, row in standard.joints.items():
        if row.unit != "rad":
            continue
        lower, upper = row.position_limits
        encoders[name] = EncoderCalibration(
            4096,
            reference_tick,
            (lower + upper) / 2,
            direction,
            position_min_rad=lower,
            position_max_rad=upper - endpoint_loss,
        )
        motor = actuators[row.host_field.removesuffix(".pos")]
        motors[motor.name] = dict(
            id=motor.motor_id,
            model=motor.motor_model,
            normalization="range_0_100" if name.endswith("_gripper") else "range_m100_100",
            homing_offset=0,
            drive_mode=drive_mode,
            range_min=0,
            range_max=4095,
        )
    metadata = dict(robot_model="alohamini2pro", motors=motors)
    return DeviceJointSpace(
        standard, encoders, LiftCalibration(0, 0.6, -0.3, 0.3), metadata
    ), metadata


def test_fixed_common_intervals_and_reference_pose_are_source_validated(specification):
    expected = {
        "shoulder_pan": (-125, 125),
        "shoulder_lift": (-199, 0),
        "elbow_flex": (0, 186),
        "wrist_flex": (-95, 87),
        "wrist_yaw_joint": (-82, 84),
        "wrist_roll": (-180, 180),
    }
    for suffix, degrees in expected.items():
        for side in ("left", "right"):
            row = specification.joints[f"{side}_{suffix}"]
            assert (row.lower, row.upper) == pytest.approx(tuple(map(math.radians, degrees)))
    report = specification.validate()
    assert report["passed"]
    assert len(report["checks"]) == 27
    assert report["hardware_qualified"] is False
    assert report["collision_free_workspace_verified"] is False


@pytest.mark.parametrize(
    "change", ["source_bounds", "direction", "reference", "qualification", "expansion"]
)
def test_invalid_or_unsupported_specification_claims_are_rejected(specification, change):
    data = specification.descriptor
    if change == "source_bounds":
        data["arm_joints"]["shoulder_pan"]["limits_deg"][0] = -140
    elif change == "direction":
        data["arm_joints"]["elbow_flex"]["direction"] = 1
    elif change == "reference":
        data["arm_joints"]["elbow_flex"]["reference_deg"] = 88
    elif change == "qualification":
        data["validation"]["hardware_qualified"] = True
    else:
        data["boundary_policy"]["automatic_limit_expansion"] = True
    with pytest.raises(ValueError):
        ModelJointSpecification(specification.model, data)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 4, -1])
def test_tolerance_requires_a_finite_bounded_angular_budget(specification, value):
    data = specification.descriptor
    data["boundary_policy"]["feedback_tolerance_deg"] = value
    with pytest.raises(ValueError):
        ModelJointSpecification(specification.model, data)


def test_feedback_tolerance_classifies_without_moving_the_coordinate(specification):
    row = specification.joints["left_shoulder_pan"]
    assert row.feedback_status(row.upper) == "inside"
    assert row.feedback_status(row.upper + math.radians(1.5)) == "boundary_tolerance"
    assert row.feedback_status(row.lower - math.radians(2)) == "boundary_tolerance"
    assert row.feedback_status(row.upper + math.radians(2.01)) == "outside"
    assert row.upper == pytest.approx(math.radians(125))


def test_folded_capture_is_separate_from_the_existing_model_coordinate_reference(specification):
    for side in ("left", "right"):
        assert specification.capture_positions[f"{side}_shoulder_lift"] == 0
        assert specification.capture_positions[f"{side}_elbow_flex"] == 0
        assert specification.capture_positions[f"{side}_wrist_flex"] == math.radians(87)
        assert specification.joints[f"{side}_shoulder_lift"].reference == -math.pi / 2
        assert specification.joints[f"{side}_elbow_flex"].reference == math.pi / 2
        assert specification.joints[f"{side}_wrist_flex"].reference == 0
    descriptor = specification.descriptor
    descriptor["calibration_pose"]["arm_joints"] = {
        name: "reference" for name in descriptor["arm_joints"]
    }
    other = ModelJointSpecification(specification.model, descriptor)
    assert other.joints == specification.joints


def test_new_specification_does_not_reinterpret_existing_checkpoint_ranges():
    old = StandardJointSpace.from_model()
    saved = old.descriptor
    new = StandardJointSpace.from_specification()
    assert new.signature != old.signature
    assert old.descriptor == saved
    assert StandardJointSpace(saved).signature == old.signature
    assert new.decode_targets({"left_shoulder_pan": 1})["left_shoulder_pan"] == pytest.approx(
        math.radians(125)
    )
    assert StandardJointSpace(new.descriptor).signature == new.signature


@pytest.mark.parametrize("direction,drive_mode", [(1, 0), (-1, 0), (1, 1), (-1, 1)])
def test_small_endpoint_projection_preserves_zero_limits_and_training_scale(direction, drive_mode):
    device, meta = binding(
        direction=direction, drive_mode=drive_mode, endpoint_loss=math.radians(1.5)
    )
    before, signature = deepcopy(meta), device.standard.signature
    original_encoders = dict(device.encoders)
    nominal = device.standard.joints["left_shoulder_pan"].position_at_max
    projected = device.project_positions({"left_shoulder_pan": nominal}, robot_metadata=meta)
    actual = nominal - math.radians(2)
    assert projected.positions["left_shoulder_pan"] == pytest.approx(actual)
    assert projected.adjustments["left_shoulder_pan"]["requested"] == nominal
    assert projected.to_dict()["applied"]["left_shoulder_pan"] == pytest.approx(actual)
    target = device.host_targets_from_positions(projected.positions, robot_metadata=meta)
    tick = device.units["left_shoulder_pan"].to_tick(target["arm_left_shoulder_pan.pos"])
    q = device.encoders["left_shoulder_pan"].position_from_tick(tick)
    assert q == pytest.approx(actual, abs=math.pi / 4096)
    assert meta == before
    assert dict(device.encoders) == original_encoders
    assert device.standard.signature == signature
    with pytest.raises(TypeError):
        projected.positions["left_shoulder_pan"] = 99


def test_device_ending_earlier_than_budget_is_rejected_without_learning_a_new_zero():
    device, meta = binding(endpoint_loss=math.radians(4))
    nominal = device.standard.joints["left_shoulder_pan"].position_at_max
    with pytest.raises(ValueError, match="projection tolerance"):
        device.project_positions({"left_shoulder_pan": nominal}, robot_metadata=meta)
    assert device.standard.decode_targets({"left_shoulder_pan": 1})["left_shoulder_pan"] == nominal


@pytest.mark.parametrize("direction", [-1, 1])
def test_projection_checks_installed_raw_bus_limits_including_negative_directions(direction):
    standard = StandardJointSpace.from_model()
    device, meta = binding(standard=standard, direction=direction)
    name = "left_shoulder_pan"
    upper = standard.joints[name].position_at_max
    endpoint = upper - math.radians(1.5)
    tick = device.encoders[name].position_to_tick(endpoint)
    field = "range_max" if direction == 1 else "range_min"
    meta["motors"]["arm_left_shoulder_pan"][field] = tick
    device = DeviceJointSpace(standard, device.encoders, device.lift, meta)
    projected = device.project_positions(
        {name: upper}, robot_metadata=meta, tolerance_rad=math.radians(2)
    )
    result = device.host_targets_from_positions(projected.positions, robot_metadata=meta)
    assert device.units[name].to_tick(result["arm_left_shoulder_pan.pos"]) == tick
    assert projected.positions[name] == pytest.approx(endpoint, abs=math.pi / 4096)


@pytest.mark.parametrize("direction", [-1, 1])
def test_encoder_wrap_does_not_become_a_false_endpoint(direction):
    device, meta = binding(reference_tick=4090, direction=direction)
    name = "left_shoulder_pan"
    requested = device.encoders[name].reference_position_rad + 0.2
    projected = device.project_positions({name: requested}, robot_metadata=meta)
    assert projected.positions[name] == requested
    assert projected.adjustments == {}
    device.host_targets_from_positions(projected.positions, robot_metadata=meta)


def test_angular_tolerance_cannot_expand_lift_or_gripper_semantics():
    device, meta = binding()
    with pytest.raises(ValueError, match="lift interval"):
        device.project_positions({"vertical_move": 0.301}, robot_metadata=meta)
    closed = device.standard.joints["left_gripper"].position_at_min
    result = device.project_positions({"left_gripper": closed}, robot_metadata=meta)
    assert result.positions["left_gripper"] == closed


def test_actual_feedback_beyond_nominal_bounds_remains_observable():
    device, meta = binding()
    raw = {name: encoder.reference_tick for name, encoder in device.encoders.items()}
    name = "left_shoulder_pan"
    q = device.standard.joints[name].position_at_max + math.radians(1.5)
    encoder = device.encoders[name]
    raw[name] = (
        encoder.reference_tick + round((q - encoder.reference_position_rad) * 4096 / math.tau)
    ) % 4096
    positions = device.positions_from_observation(
        {"lift_axis.height_mm": 300},
        robot_metadata=meta,
        post_offset_ticks=raw,
    )
    assert positions[name] > device.standard.joints[name].position_at_max
    assert positions[name] == pytest.approx(q, abs=math.pi / 4096)


def test_projection_retains_strict_host_identity_and_invalid_input_rejection():
    device, meta = binding()
    changed = deepcopy(meta)
    changed["motors"]["arm_left_shoulder_pan"]["homing_offset"] = 1
    with pytest.raises(ValueError, match="calibration changed"):
        device.project_positions({"left_shoulder_pan": 0}, robot_metadata=changed)
    for targets in ({}, {"unknown": 0}, {"left_shoulder_pan": float("nan")}):
        with pytest.raises(ValueError):
            device.project_positions(targets, robot_metadata=meta)
    with pytest.raises(ValueError, match="2.*degrees"):
        device.project_positions(
            {"left_shoulder_pan": 0}, robot_metadata=meta, tolerance_rad=math.radians(3)
        )


def test_installed_range_normalization_is_separate_from_frozen_policy_coordinates():
    device, meta = binding()
    meta["motors"]["arm_left_shoulder_pan"].update(range_min=1000, range_max=3000)
    first = DeviceJointSpace(device.standard, device.encoders, device.lift, meta)
    other = deepcopy(meta)
    other["motors"]["arm_left_shoulder_pan"].update(range_min=900, range_max=3100)
    second = DeviceJointSpace(device.standard, device.encoders, device.lift, other)
    commands = []
    for bound, metadata in ((first, meta), (second, other)):
        target = bound.host_targets_from_positions(
            {"left_shoulder_pan": 0.5}, robot_metadata=metadata
        )
        value = target["arm_left_shoulder_pan.pos"]
        commands.append(value)
        assert bound.units["left_shoulder_pan"].to_tick(value) == 2374
        assert bound.standard.encode_positions(
            {"left_shoulder_pan": 0.5}
        ) == first.standard.encode_positions({"left_shoulder_pan": 0.5})
    assert commands[0] != commands[1]
    assert first.standard.signature == second.standard.signature

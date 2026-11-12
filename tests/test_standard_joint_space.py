import math
from copy import deepcopy

import pytest

from alohamini.calibration.encoder import EncoderCalibration
from alohamini.calibration.joint_space import DeviceJointSpace, StandardJointSpace
from alohamini.calibration.lift import LiftCalibration


@pytest.fixture
def standard():
    return StandardJointSpace.from_model("alohamini2pro")


def machine(standard, reference_tick=2048, direction=1, drive_mode=0):
    encoders, motors = {}, {}
    for name, row in standard.joints.items():
        if row.unit != "rad":
            continue
        lower, upper = row.position_limits
        encoders[name] = EncoderCalibration(4096, reference_tick, (lower + upper) / 2, direction)
        motors[row.host_field.removesuffix(".pos")] = dict(
            id=7 if name.endswith("gripper") else 1,
            model="sts3250",
            normalization="range_0_100" if name.endswith("gripper") else "range_m100_100",
            range_min=0,
            range_max=4095,
            drive_mode=drive_mode,
            homing_offset=123,
        )
    metadata = dict(robot_model="alohamini2pro", motors=motors)
    binding = DeviceJointSpace(standard, encoders, LiftCalibration(0, 0.6, -0.3, 0.3), metadata)
    return binding, metadata


def test_fixed_model_endpoints_and_gripper_direction(standard):
    assert len(standard.joints) == 15
    q = standard.decode_targets(
        dict(left_shoulder_pan=-1, left_gripper=0, right_gripper=1, vertical_move=0)
    )
    assert q == pytest.approx(
        dict(
            left_shoulder_pan=-2.196660488252487,
            left_gripper=0.32,
            right_gripper=-1.8030294104,
            vertical_move=-0.3,
        )
    )
    assert standard.decode_targets(dict(vertical_move=1))["vertical_move"] == pytest.approx(0.3)
    assert standard.encode_positions(
        dict(left_gripper=0.32, right_gripper=-1.8030294104, vertical_move=0)
    ) == pytest.approx(dict(left_gripper=0, right_gripper=1, vertical_move=0.5))


def test_feedback_overrun_is_visible_but_cannot_become_a_target(standard):
    observed = standard.encode_positions({"vertical_move": 0.31})
    assert observed["vertical_move"] > 1
    with pytest.raises(ValueError, match="standard range"):
        standard.decode_targets(observed)
    for x in (True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            standard.decode_targets({"left_gripper": x})
    with pytest.raises(ValueError, match="Unknown"):
        standard.encode_positions({"arm_left_gripper.pos": 50})


def test_named_order_and_frozen_descriptor_round_trip(standard):
    frozen = StandardJointSpace(deepcopy(standard.descriptor))
    assert frozen.signature == standard.signature
    exported = frozen.descriptor
    exported["joints"][0]["position_at_max"] = 999
    assert frozen.signature == standard.signature
    assert list(frozen.decode_targets({"right_gripper": 0.2, "left_gripper": 0.8})) == [
        "right_gripper",
        "left_gripper",
    ]
    # Arm standard zero deliberately represents the midpoint, not CAD Home.
    assert frozen.decode_targets({"left_shoulder_lift": 0})["left_shoulder_lift"] == pytest.approx(
        -1.743369165432274
    )


@pytest.mark.parametrize("direction,drive_mode", [(1, 0), (-1, 0), (1, 1), (-1, 1)])
def test_device_zero_and_bus_inversion_do_not_change_physical_actions(
    standard, direction, drive_mode
):
    a, meta_a = machine(standard)
    b, meta_b = machine(standard, reference_tick=1700, direction=direction, drive_mode=drive_mode)
    target = {"left_shoulder_pan": 0.25, "right_gripper": 0.35, "vertical_move": 0.7}
    physical = standard.decode_targets(target)
    raw_commands = []
    for binding, metadata in ((a, meta_a), (b, meta_b)):
        commands = binding.host_targets_from_standard(target, robot_metadata=metadata)
        expected = {
            name: binding.encoders[name].position_to_tick(q)
            for name, q in physical.items()
            if name != "vertical_move"
        }
        ticks = {
            name: binding.units[name].to_tick(commands[standard.joints[name].host_field])
            for name in expected
        }
        assert ticks == expected
        for name, tick in ticks.items():
            assert binding.encoders[name].position_from_tick(tick) == pytest.approx(
                physical[name], abs=math.pi / 4096 + 1e-12
            )
        assert commands["lift_axis.height_mm"] == pytest.approx(420)
        raw_commands.append(ticks)
    assert raw_commands[0] != raw_commands[1]


def test_narrow_device_travel_rejects_instead_of_stretching_standard(standard):
    device, metadata = machine(standard)
    metadata["motors"]["arm_left_shoulder_pan"].update(range_min=1800, range_max=2300)
    narrow = DeviceJointSpace(standard, device.encoders, device.lift, metadata)
    assert narrow.standard.signature == device.standard.signature
    with pytest.raises(ValueError, match="installed Host encoder range"):
        narrow.host_targets_from_standard({"left_shoulder_pan": 0.9}, robot_metadata=metadata)


def test_lift_metres_are_not_rescaled_to_fit_a_shorter_mechanism(standard):
    device, metadata = machine(standard)
    with pytest.raises(ValueError, match="metre scale"):
        DeviceJointSpace(standard, device.encoders, LiftCalibration(0, 0.58, -0.3, 0.3), metadata)
    short = DeviceJointSpace(
        standard, device.encoders, LiftCalibration(0, 0.58, -0.3, 0.28), metadata
    )
    assert short.host_targets_from_standard({"vertical_move": 0.5}, robot_metadata=metadata)[
        "lift_axis.height_mm"
    ] == pytest.approx(300)
    with pytest.raises(ValueError, match="calibrated position limits"):
        short.host_targets_from_standard({"vertical_move": 1}, robot_metadata=metadata)


def test_live_calibration_changes_are_not_silently_rebound(standard):
    device, metadata = machine(standard)
    changed = deepcopy(metadata)
    changed["motors"]["arm_right_gripper"]["homing_offset"] += 1
    with pytest.raises(ValueError, match="calibration changed"):
        device.host_targets_from_standard({"right_gripper": 0.5}, robot_metadata=changed)


def test_raw_feedback_does_not_inherit_host_normalization_saturation(standard):
    device, metadata = machine(standard)
    narrowed = deepcopy(metadata)
    narrowed["motors"]["arm_left_shoulder_pan"].update(range_min=2000, range_max=2100)
    device = DeviceJointSpace(standard, device.encoders, device.lift, narrowed)
    raw = {name: encoder.reference_tick for name, encoder in device.encoders.items()}
    raw["left_shoulder_pan"] = 2200
    observed = device.positions_from_observation(
        {"lift_axis.height_mm": 300}, robot_metadata=narrowed, post_offset_ticks=raw
    )
    expected = 0.006135923151513 + 152 * math.tau / 4096
    assert observed["left_shoulder_pan"] == pytest.approx(expected)
    assert observed["vertical_move"] == pytest.approx(0)


def test_captured_mapping_is_required_and_preserves_device_specific_zeros(standard):
    device, metadata = machine(standard)
    arms = {}
    for side in ("left", "right"):
        joints = {}
        for name, encoder in device.encoders.items():
            if not name.startswith(side + "_"):
                continue
            suffix = name.removeprefix(side + "_")
            suffix = "wrist_yaw" if suffix == "wrist_yaw_joint" else suffix
            motor = metadata["motors"][standard.joints[name].host_field.removesuffix(".pos")]
            row = dict(
                id=motor["id"],
                model=motor["model"],
                reference_tick=encoder.reference_tick,
                reference_q_rad=encoder.reference_position_rad,
                sign=encoder.direction,
                motor_calibration={
                    k: motor[k] for k in ("homing_offset", "range_min", "range_max", "drive_mode")
                },
            )
            if suffix == "gripper":
                row.update(urdf_closed_rad=0.32, urdf_open_rad=-1.8030294104)
            joints[suffix] = row
        arms[side] = dict(
            robot_model="alohamini2pro", side=side, ticks_per_revolution=4096, joints=joints
        )
    lift = dict(
        robot_model="alohamini2pro",
        mechanism=dict(physical_min_mm=0, physical_max_mm=600),
        urdf=dict(q_at_physical_min_m=-0.3, q_at_physical_max_m=0.3),
    )
    bound = DeviceJointSpace.from_mappings(standard, arms, lift, metadata)
    assert bound.host_targets_from_standard({"left_gripper": 0.5}, robot_metadata=metadata)
    del arms["left"]["joints"]["gripper"]["motor_calibration"]
    with pytest.raises(ValueError, match="captured motor_calibration"):
        DeviceJointSpace.from_mappings(standard, arms, lift, metadata)


def test_software_normalization_inverse_can_represent_every_encoder_bin(standard):
    device, metadata = machine(standard)
    for tick in range(1000, 3000):
        encoder = device.encoders["left_shoulder_pan"]
        q = encoder.position_from_tick(tick)
        if (
            not standard.joints["left_shoulder_pan"].position_limits[0]
            <= q
            <= standard.joints["left_shoulder_pan"].position_limits[1]
        ):
            continue
        commands = device.host_targets_from_positions(
            {"left_shoulder_pan": q}, robot_metadata=metadata
        )
        assert (
            device.units["left_shoulder_pan"].to_tick(commands["arm_left_shoulder_pan.pos"]) == tick
        )

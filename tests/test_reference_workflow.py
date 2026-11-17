"""One model contract across a leader, independent followers and recorded coordinates."""

import hashlib
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
from alohamini_bridge.mapping import JointMapper

from alohamini.calibration.encoder import EncoderCalibration
from alohamini.calibration.reference import (
    PROTOCOL,
    calibration_bytes,
    json_bytes,
    save_bundle,
    side_calibration,
)
from alohamini.calibration.servo import MotorCalibration
from alohamini.datasets.kinematic_fields import KinematicFields, descriptor_from_reference_pose
from alohamini.datasets.record import dataset_features
from alohamini.hardware.leader import BimanualLeader
from alohamini.model import ActuatorSpec
from alohamini.model.joint_spec import ModelJointSpecification
from alohamini.runtime.host import NativeHost
from alohamini.runtime.startup import open_host


def device_calibration(raw, *, leader=False):
    spec = ModelJointSpecification.from_model()
    values, ticks, arms = {}, {}, {}
    for side in ("left", "right"):
        motors = tuple(m for m in spec.model.actuators if m.bus == side)
        if leader:
            motors = tuple(
                ActuatorSpec(m.name.removeprefix(f"arm_{side}_"), side, m.motor_id, "sts3215")
                for m in motors
                if m.name.startswith("arm_")
            )
        calibration, reference, mapping = side_calibration(
            spec, side, motors, {m.name: raw for m in motors}, leader=leader
        )
        values[side], arms[side] = calibration, mapping
        ticks.update(reference)
    return spec, values, ticks, arms


def encoder(entry):
    return EncoderCalibration(
        4096,
        entry["reference_tick"],
        entry["reference_q_rad"],
        entry["sign"],
        position_min_rad=entry["safe_q_min_rad"],
        position_max_rad=entry["safe_q_max_rad"],
    )


def test_one_leader_commands_two_followers_without_changing_its_calibration():
    spec, leader, _, leader_maps = device_calibration(4095, leader=True)
    original = deepcopy(leader)
    for raw in (600, 3000):
        _, followers, _, maps = device_calibration(raw)
        for side in ("left", "right"):
            for suffix, entry in leader_maps[side]["joints"].items():
                name = f"arm_{side}_{suffix}"
                source, target = leader[side][suffix], followers[side][name]
                model_name = f"{side}_{'wrist_yaw_joint' if suffix == 'wrist_yaw' else suffix}"
                row = spec.joints[model_name]
                for q in np.linspace(row.lower + 0.05, row.upper - 0.05, 11):
                    tick = encoder(entry).position_to_tick(float(q))
                    normalization = "range_0_100" if suffix == "gripper" else "range_m100_100"
                    command = source.position_units(normalization).from_tick(tick)
                    actual = target.position_units(normalization).to_tick(command)
                    measured = encoder(maps[side]["joints"][suffix]).position_from_tick(actual)
                    assert measured == pytest.approx(q, abs=1.6 * np.pi * 2 / 4096)
                assert source.homing_offset != target.homing_offset
    assert leader == original


def test_am_leader_profile_is_reusable_with_both_follower_variants(tmp_path):
    from alohamini.calibration.reference import load_reference_profile

    spec, sides, _, mappings = device_calibration(1800, leader=True)
    payloads = {
        tmp_path / f"am_leader_bi_{side}.json": calibration_bytes(values)
        for side, values in sides.items()
    }
    profile = dict(
        schema_version=1,
        protocol=PROTOCOL,
        robot_model=spec.model.model_id,
        role="leader",
        reference_pose=spec.descriptor["reference_pose"],
        capture_pose=spec.descriptor["calibration_pose"]["name"],
        specification_sha256=spec.signature,
        boundary_policy=spec.descriptor["boundary_policy"],
        reference_ticks={
            f"arm_{side}_{name}": entry["reference_tick"]
            for side, mapping in mappings.items()
            for name, entry in mapping["joints"].items()
        },
        capture_ticks={
            f"arm_{side}_{name}": entry["capture_tick"]
            for side, mapping in mappings.items()
            for name, entry in mapping["joints"].items()
        },
        arms=mappings,
        calibration_sha256={
            path.name: hashlib.sha256(data).hexdigest() for path, data in payloads.items()
        },
    )
    for path in tuple(payloads):
        payloads[path.with_name(f"{path.stem}.reference.json")] = json_bytes(profile)
    save_bundle(payloads, {})
    before = {path: path.read_bytes() for path in payloads}
    leader = BimanualLeader("alohamini2", calibration_dir=tmp_path)
    assert leader.reference_profiles["left"]["robot_model"] == "alohamini2pro"
    assert load_reference_profile(tmp_path / "am_leader_bi_left.json", "alohamini2") == profile
    assert {path: path.read_bytes() for path in payloads} == before


@pytest.mark.parametrize("raw", [0, 1, 988, 1023, 1024, 2047, 2048, 3036, 3071, 4094, 4095])
def test_folded_capture_keeps_the_full_model_travel_on_one_encoder_branch(raw):
    spec, sides, _, mappings = device_calibration(raw)
    quantum = 2 * np.pi / 4096
    for side, mapping in mappings.items():
        for suffix, entry in mapping["joints"].items():
            name = f"{side}_{'wrist_yaw_joint' if suffix == 'wrist_yaw' else suffix}"
            calibration = sides[side][f"arm_{side}_{suffix}"]
            observed = (raw - calibration.homing_offset) % 4096
            assert observed == entry["capture_tick"]
            assert calibration.range_min <= observed <= calibration.range_max
            converter = encoder(entry)
            assert converter.position_from_tick(observed) == pytest.approx(
                spec.capture_positions[name], abs=quantum + 1e-10
            )
            assert converter.position_from_tick(entry["reference_tick"]) == pytest.approx(
                spec.joints[name].reference, abs=1e-12
            )
            assert entry["safe_q_min_rad"] == pytest.approx(
                spec.joints[name].lower, abs=quantum + 1e-10
            )
            assert entry["safe_q_max_rad"] == pytest.approx(
                spec.joints[name].upper, abs=quantum + 1e-10
            )
            if suffix in ("shoulder_pan", "wrist_yaw", "wrist_roll"):
                assert calibration.range_min + calibration.range_max == 2 * entry["reference_tick"]
                assert entry["safe_q_min_rad"] == pytest.approx(-entry["safe_q_max_rad"], abs=1e-12)
            if suffix == "wrist_roll":
                assert calibration.range_max - calibration.range_min == 4094


@pytest.mark.parametrize("raw", [0, 600, 4095])
def test_generated_gripper_preserves_both_named_endpoints(raw):
    spec, sides, _, mappings = device_calibration(raw)
    for side in ("left", "right"):
        calibration = sides[side][f"arm_{side}_gripper"]
        mapping = encoder(mappings[side]["joints"]["gripper"])
        units = calibration.position_units("range_0_100")
        for q in (spec.descriptor["gripper"]["closed_rad"], spec.descriptor["gripper"]["open_rad"]):
            tick = mapping.position_to_tick(q)
            assert calibration.range_min <= tick <= calibration.range_max
            assert abs(units.to_tick(units.from_tick(tick)) - tick) <= 1
            assert mapping.position_from_tick(tick) == pytest.approx(q, abs=1e-10)


def save_robot(tmp_path, raw=4095):
    spec, sides, ticks, arms = device_calibration(raw)
    values = {name: c for side in sides.values() for name, c in side.items()}
    for motor in spec.model.actuators:
        values.setdefault(motor.name, MotorCalibration(motor.motor_id, 0, 0, 0, 4095))
    path = tmp_path / "AlohaMiniRobot.json"
    payload = calibration_bytes(values)
    profile = dict(
        schema_version=1,
        protocol=PROTOCOL,
        robot_model=spec.model.model_id,
        role="robot",
        reference_pose="upright_forward",
        capture_pose=spec.descriptor["calibration_pose"]["name"],
        capture_ticks={
            f"arm_{side}_{suffix}": entry["capture_tick"]
            for side, mapping in arms.items()
            for suffix, entry in mapping["joints"].items()
        },
        specification_sha256=spec.signature,
        boundary_policy=spec.descriptor["boundary_policy"],
        reference_ticks=ticks,
        arms=arms,
        calibration_sha256={path.name: hashlib.sha256(payload).hexdigest()},
    )
    save_bundle(
        {path: payload, path.with_suffix(".reference.json"): json_bytes(profile)},
        {path: spec.model.actuators},
    )
    return spec, path, values, profile


def test_generated_profile_reaches_host_and_records_the_execution_buffer(tmp_path, monkeypatch):
    spec, path, values, profile = save_robot(tmp_path)
    assembled = []

    def inspect(host):
        assembled.append(host)
        raise RuntimeError("inspect before I/O")

    monkeypatch.setattr(NativeHost, "start", inspect)
    with pytest.raises(RuntimeError, match="inspect before I/O"):
        open_host(spec.model.model_id, calibration_file=path, cameras={})
    host = assembled[0]
    assert host._metadata["reference_calibration"] == profile
    assert len(host.control.arms._command_bounds) == 12
    assert host.control.arms.requested_targets == {}
    for name, bounds in host.control.arms._command_bounds.items():
        calibration = values[name].encoder_calibration()
        assert bounds == pytest.approx(
            (
                calibration.position_min_rad + np.deg2rad(2),
                calibration.position_max_rad - np.deg2rad(2),
            )
        )
    profile["reference_ticks"].clear()
    assert host._metadata["reference_calibration"]["reference_ticks"]


def test_recorded_reference_uses_exact_device_ticks_and_same_physical_pose(tmp_path):
    spec, _, values, profile = save_robot(tmp_path)
    motors = {
        motor.name: dict(
            id=motor.motor_id,
            model=motor.motor_model,
            **asdict(
                values[motor.name].position_units(
                    "range_0_100" if motor.name.endswith("gripper") else "range_m100_100"
                )
            ),
            homing_offset=values[motor.name].homing_offset,
        )
        for motor in spec.model.actuators
    }
    info = dict(
        features=dataset_features(spec.model.model_id),
        robot_metadata=dict(
            robot_model=spec.model.model_id,
            motors=motors,
            reference_calibration=profile,
        ),
    )
    descriptor = descriptor_from_reference_pose(info, "upright_forward")
    fields = KinematicFields(descriptor)
    state = []
    for name in fields.names:
        motor = name.removesuffix(".pos")
        if motor not in profile["reference_ticks"]:
            state.append(0)
        else:
            normalization = "range_0_100" if motor.endswith("gripper") else "range_m100_100"
            state.append(
                values[motor]
                .position_units(normalization)
                .from_tick(profile["reference_ticks"][motor])
            )
    for arm in descriptor["arms"]:
        for coordinate in arm["coordinates"]:
            motor_name = coordinate["field"].removesuffix(".pos")
            assert coordinate["encoder"]["reference_tick"] == profile["reference_ticks"][motor_name]
            i = fields.names.index(coordinate["field"])
            assert fields.joint(state[i], coordinate) == pytest.approx(0, abs=2 * np.pi / 4096)
    assert np.isfinite(fields.absolute(state)).all()
    folded = list(state)
    for arm in descriptor["arms"]:
        for coordinate in arm["coordinates"]:
            motor_name = coordinate["field"].removesuffix(".pos")
            i = fields.names.index(coordinate["field"])
            folded[i] = (
                values[motor_name].position_units().from_tick(profile["capture_ticks"][motor_name])
            )
            model_name = motor_name.removeprefix("arm_")
            if model_name.endswith("wrist_yaw"):
                model_name += "_joint"
            assert fields.joint(folded[i], coordinate) == pytest.approx(
                spec.capture_positions[model_name] - spec.joints[model_name].reference,
                abs=4 * np.pi / 4096,
            )
    assert np.isfinite(fields.absolute(folded)).all()
    motors["arm_left_shoulder_pan"]["homing_offset"] += 1
    with pytest.raises(ValueError, match="Reference calibration differs"):
        descriptor_from_reference_pose(info, "upright_forward")


def test_generated_ros_maps_decode_the_reference_and_reject_another_calibration(tmp_path):
    spec, _, values, profile = save_robot(tmp_path)
    metadata = dict(
        motors={
            motor.name: dict(
                id=motor.motor_id,
                model=motor.motor_model,
                homing_offset=values[motor.name].homing_offset,
                **asdict(
                    values[motor.name].position_units(
                        "range_0_100" if motor.name.endswith("gripper") else "range_m100_100"
                    )
                ),
            )
            for motor in spec.model.actuators
        }
    )
    mapper = JointMapper(
        profile["arms"],
        dict(
            mechanism=dict(physical_min_mm=0, physical_max_mm=600),
            urdf=dict(q_at_physical_min_m=-0.3, q_at_physical_max_m=0.3),
        ),
    )
    observation = {"lift_axis.height_mm": 0}
    for name, tick in profile["reference_ticks"].items():
        observation[f"{name}.pos"] = mapper.tick_to_host(tick, metadata["motors"][name])
    positions = mapper.observation_to_joint_positions(observation, metadata)
    for name, row in spec.joints.items():
        assert positions[name] == pytest.approx(row.reference, abs=2 * np.pi / 4096)
    for name, tick in profile["capture_ticks"].items():
        observation[f"{name}.pos"] = mapper.tick_to_host(tick, metadata["motors"][name])
    positions = mapper.observation_to_joint_positions(observation, metadata)
    for name, expected in spec.capture_positions.items():
        assert positions[name] == pytest.approx(expected, abs=4 * np.pi / 4096)
    metadata["motors"]["arm_left_shoulder_pan"]["homing_offset"] += 1
    with pytest.raises(ValueError, match="mapping differs"):
        mapper.observation_to_joint_positions(observation, metadata)
    with pytest.raises(ValueError, match="mapping differs"):
        mapper.urdf_to_host("left_shoulder_pan", 0, metadata)


def test_generated_leader_profile_loads_through_passive_leader(tmp_path):
    spec, sides, ticks, arms = device_calibration(900, leader=True)
    payloads = {
        tmp_path / f"am_leader_bi_{side}.json": calibration_bytes(values)
        for side, values in sides.items()
    }
    profile = dict(
        schema_version=1,
        protocol=PROTOCOL,
        robot_model=spec.model.model_id,
        role="leader",
        reference_pose="upright_forward",
        capture_pose=spec.descriptor["calibration_pose"]["name"],
        capture_ticks={
            f"arm_{side}_{suffix}": entry["capture_tick"]
            for side, mapping in arms.items()
            for suffix, entry in mapping["joints"].items()
        },
        specification_sha256=spec.signature,
        boundary_policy=spec.descriptor["boundary_policy"],
        reference_ticks=ticks,
        arms=arms,
        calibration_sha256={
            path.name: hashlib.sha256(data).hexdigest() for path, data in payloads.items()
        },
    )
    for path in tuple(payloads):
        payloads[path.with_suffix(".reference.json")] = json_bytes(profile)
    save_bundle(payloads, {})
    leader = BimanualLeader(spec.model.model_id, calibration_dir=tmp_path)
    assert all(p == profile for p in leader.reference_profiles.values())
    assert leader.calibrations == sides

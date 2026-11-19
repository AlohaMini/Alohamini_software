"""Generate device calibration and model bindings from one known physical pose."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from alohamini.calibration.servo import MotorCalibration, load_motor_calibration
from alohamini.hardware.feetech_device import DeviceOperationError
from alohamini.model.joint_spec import ModelJointSpecification

PROTOCOL = "alohamini-reference-v1"


def side_calibration(spec, side, actuators, positions, *, leader):
    """Compute one side without I/O; reject a model interval that cannot fit STS counts."""
    calibrations, ticks, joints = {}, {}, {}
    for motor in actuators:
        suffix = motor.name if leader else motor.name.removeprefix(f"arm_{side}_")
        if suffix not in spec.descriptor["arm_joints"] and suffix != "gripper":
            continue
        name = f"{side}_{'wrist_yaw_joint' if suffix == 'wrist_yaw' else suffix}"
        row = spec.joints[name]
        raw = positions[motor.name]
        if type(raw) is not int or not 0 <= raw < 4096:
            raise ValueError(f"{motor.name}: invalid reference encoder reading")
        scale = row.direction * math.tau / 4096
        capture_q = spec.capture_positions[name]
        # Keep the existing model reference near the encoder centre. The
        # operator's folded pose need not be at that centre; otherwise the
        # shoulder's 199-degree travel would cross the single-turn branch.
        # Decimal model endpoints can differ from an exact encoder count by
        # ~1e-8 ticks; retain that count without extending travel by a full tick.
        for reference in (2047, 2048):
            endpoints = [reference + (q - row.reference) / scale for q in (row.lower, row.upper)]
            lower = max(0, math.ceil(min(endpoints) - 1e-7))
            upper = min(4095, math.floor(max(endpoints) + 1e-7))
            if math.isclose(row.reference - row.lower, row.upper - row.reference, abs_tol=1e-10):
                radius = min(reference - lower, upper - reference)
                lower, upper = reference - radius, reference + radius
            capture_value = reference + (capture_q - row.reference) / scale
            pose = spec.descriptor["calibration_pose"]["arm_joints"].get(suffix)
            rounding = pose["encoder_rounding"] if isinstance(pose, dict) else "nearest"
            # Explicit Home angles retain the original encoder quantization even
            # when a travel endpoint moves beyond Home.
            captured = {
                "nearest": round(capture_value),
                "floor": math.floor(capture_value + 1e-7),
                "ceil": math.ceil(capture_value - 1e-7),
            }[rounding]
            capture_tick = max(lower, min(upper, captured))
            offset = (raw - capture_tick + 2048) % 4096 - 2048
            if offset != -2048:
                break
        else:
            raise ValueError(f"{motor.name}: reference offset is not representable")
        if not lower <= reference <= upper or lower == upper:
            raise ValueError(f"{motor.name}: model interval cannot fit the encoder reference")
        q_limits = sorted(row.reference + (tick - reference) * scale for tick in (lower, upper))
        quantum = math.tau / 4096
        if max(q_limits[0] - row.lower, row.upper - q_limits[1]) > quantum + 1e-10:
            raise ValueError(f"{motor.name}: model travel crosses an unsupported encoder branch")
        if abs(row.reference + (capture_tick - reference) * scale - capture_q) > quantum + 1e-10:
            raise ValueError(f"{motor.name}: folded reference cannot fit the encoder branch")
        entry = MotorCalibration(motor.motor_id, 0, offset, lower, upper)
        calibrations[motor.name] = entry
        full_name = f"arm_{side}_{suffix}"
        ticks[full_name] = reference
        joints[suffix] = dict(
            id=motor.motor_id,
            model=motor.motor_model,
            reference_tick=reference,
            reference_q_rad=row.reference,
            capture_tick=capture_tick,
            capture_q_rad=capture_q,
            sign=row.direction,
            joint_per_encoder_ratio=1.0,
            safe_q_min_rad=q_limits[0],
            safe_q_max_rad=q_limits[1],
            motor_calibration={
                key: asdict(entry)[key]
                for key in ("homing_offset", "range_min", "range_max", "drive_mode")
            },
        )
        if suffix == "gripper":
            joints[suffix].update(
                urdf_closed_rad=spec.descriptor["gripper"]["closed_rad"],
                urdf_open_rad=spec.descriptor["gripper"]["open_rad"],
            )
    if set(joints) != {*spec.descriptor["arm_joints"], "gripper"}:
        raise ValueError(f"{side}: reference calibration requires every named arm joint")
    return (
        calibrations,
        ticks,
        dict(
            schema_version=1,
            robot_model=spec.model.model_id,
            side=side,
            ticks_per_revolution=4096,
            status="generated_from_reference_pose",
            capture_pose=spec.descriptor["calibration_pose"]["name"],
            joints=joints,
        ),
    )


def reference_coordinates(metadata):
    """Validate the recorded model binding before comparing device-independent units."""
    profile = metadata.get("reference_calibration")
    if not isinstance(profile, dict):
        raise ValueError("Model-coordinate binding requires both reference profiles")
    spec = ModelJointSpecification.from_signature(
        metadata["robot_model"], profile.get("specification_sha256")
    )
    if (
        type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != 1
        or type(profile.get("schema_version")) is not int
        or profile["schema_version"] != 1
        or profile.get("role") != "robot"
        or profile.get("protocol") != PROTOCOL
        or profile.get("robot_model") != spec.model.model_id
        or profile.get("specification_sha256") != spec.signature
        or profile.get("reference_pose") != spec.descriptor["reference_pose"]
        or profile.get("capture_pose") != spec.descriptor["calibration_pose"]["name"]
        or profile.get("boundary_policy") != spec.descriptor["boundary_policy"]
    ):
        raise ValueError("Reference profile differs from the model coordinate specification")
    coordinates, references, captures = {}, {}, {}
    try:
        for side in ("left", "right"):
            motors = tuple(
                m for m in spec.model.actuators if m.bus == side and m.name.startswith("arm_")
            )
            values, positions = {}, {}
            for motor in motors:
                item = metadata["motors"][motor.name]
                if item["model"] != motor.motor_model:
                    raise ValueError(f"Reference motor identity differs: {motor.name}")
                values[motor.name] = MotorCalibration(
                    **{
                        key: item[key]
                        for key in ("id", "drive_mode", "homing_offset", "range_min", "range_max")
                    }
                )
                positions[motor.name] = (
                    profile["capture_ticks"][motor.name] + item["homing_offset"]
                ) % 4096
            expected, ticks, mapping = side_calibration(spec, side, motors, positions, leader=False)
            if values != expected or profile["arms"][side] != mapping:
                raise ValueError(
                    f"Recorded {side} motor parameters differ from their reference map"
                )
            references.update(ticks)
            for suffix, entry in mapping["joints"].items():
                name = f"arm_{side}_{suffix}"
                unit = values[name].position_units(metadata["motors"][name]["normalization"])
                captures[name] = entry["capture_tick"]
                coordinates[name] = (
                    unit.normalization,
                    entry["sign"],
                    entry["safe_q_min_rad"],
                    entry["safe_q_max_rad"],
                )
        if references != profile["reference_ticks"] or captures != profile["capture_ticks"]:
            raise ValueError("Reference encoder ticks differ from the generated model binding")
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete model reference metadata") from error
    return spec.signature, coordinates


def json_bytes(value):
    return (json.dumps(value, indent=4, allow_nan=False) + "\n").encode()


def calibration_bytes(values):
    return json_bytes({name: asdict(entry) for name, entry in values.items()})


def save_bundle(payloads, motor_files):
    """Prepare every file first; roll back installed files if any replacement fails."""
    temporaries, originals, installed = {}, {}, []
    try:
        for path, payload in payloads.items():
            if path.is_symlink():
                raise ValueError("Reference calibration files must not be symbolic links")
            path.parent.mkdir(parents=True, exist_ok=True)
            originals[path] = path.read_bytes() if path.exists() else None
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as stream:
                temporaries[path] = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if path in motor_files:
                load_motor_calibration(temporaries[path], motor_files[path])
        for path, original in originals.items():
            if original is not None:
                backup = path.with_name(f"{path.stem}.{time.time_ns()}.backup{path.suffix}")
                with backup.open("xb") as stream:
                    stream.write(original)
        for path, temporary in temporaries.items():
            os.replace(temporary, path)
            installed.append(path)
    except BaseException as primary:
        failures = {}
        for path in reversed(installed):
            try:
                if originals[path] is None:
                    path.unlink()
                else:
                    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                        recovery = Path(stream.name)
                        stream.write(originals[path])
                        stream.flush()
                        os.fsync(stream.fileno())
                    try:
                        os.replace(recovery, path)
                    finally:
                        recovery.unlink(missing_ok=True)
            except BaseException as error:
                failures[str(path)] = str(error)
        if failures:
            raise DeviceOperationError({"save": str(primary), **failures}) from primary
        raise
    finally:
        for path in temporaries.values():
            path.unlink(missing_ok=True)


def load_reference_profile(calibration_path, robot_model, *, allow_previous=False):
    """An older motor JSON remains readable; an existing stale profile is rejected."""
    calibration_path = Path(calibration_path)
    path = calibration_path.with_name(f"{calibration_path.stem}.reference.json")
    if not path.exists():
        return None
    profile = json.loads(path.read_text())
    target = ModelJointSpecification.from_model(robot_model)
    source_model = profile.get("robot_model")
    leader = profile.get("role") == "leader"
    spec = target
    if allow_previous:
        spec = ModelJointSpecification.from_signature(
            source_model, profile.get("specification_sha256")
        )
        target = ModelJointSpecification.from_model(
            robot_model, **({"version": 1} if spec.descriptor["specification_version"] == 1 else {})
        )
    if leader and source_model != robot_model:
        if not allow_previous:
            spec = ModelJointSpecification.from_model(source_model)
        # Leaders encode joint coordinates, not the follower's asset identity.
        # EDU has its own sensor-equipped description but shares Pro's arm
        # coordinates and Home pose. Validate the saved profile against its
        # original model below; never rewrite its model name or signature.
        if spec.joints != target.joints or spec.capture_positions != target.capture_positions:
            raise ValueError("Reference profile uses incompatible leader joint coordinates")
    if not leader and source_model != robot_model:
        raise ValueError("Reference profile differs from the requested robot model")
    digest = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    if (
        profile.get("schema_version") != 1
        or profile.get("protocol") != PROTOCOL
        or profile.get("robot_model") != spec.model.model_id
        or profile.get("specification_sha256") != spec.signature
        or profile.get("reference_pose") != spec.descriptor["reference_pose"]
        or profile.get("capture_pose") != spec.descriptor["calibration_pose"]["name"]
        or profile.get("boundary_policy") != spec.descriptor["boundary_policy"]
        or profile.get("calibration_sha256", {}).get(calibration_path.name) != digest
    ):
        raise ValueError(
            "Reference profile differs from the installed calibration or model; "
            "run alohamini calibrate and reuse to upgrade a supported old profile, "
            "or recalibrate"
        )
    if profile.get("role") not in ("leader", "robot"):
        raise ValueError("Invalid reference calibration role")
    references, captures = {}, {}
    for side in ("left", "right"):
        motors = tuple(m for m in spec.model.actuators if m.bus == side)
        if leader:
            from alohamini.model import ActuatorSpec

            motors = tuple(
                ActuatorSpec(m.name.removeprefix(f"arm_{side}_"), side, m.motor_id, "sts3215")
                for m in motors
                if m.name.startswith("arm_")
            )
            files = [
                name for name in profile["calibration_sha256"] if name.endswith(f"_{side}.json")
            ]
            if len(files) != 1 or Path(files[0]).name != files[0]:
                raise ValueError("Reference profile must identify both leader files")
            source = calibration_path.parent / files[0]
        else:
            source = calibration_path
        if (
            hashlib.sha256(source.read_bytes()).hexdigest()
            != profile["calibration_sha256"][source.name]
        ):
            raise ValueError("Reference profile differs from the paired calibration file")
        values = load_motor_calibration(source, motors if leader else spec.model.actuators)
        positions = {}
        for motor in motors:
            if leader or motor.name.startswith("arm_"):
                full_name = f"arm_{side}_{motor.name}" if leader else motor.name
                captured = profile["capture_ticks"][full_name]
                positions[motor.name] = (captured + values[motor.name].homing_offset) % 4096
        expected, ticks, mapping = side_calibration(spec, side, motors, positions, leader=leader)
        if (
            expected != {name: values[name] for name in expected}
            or profile["arms"][side] != mapping
        ):
            raise ValueError(
                "Reference mapping differs from model coordinates or motor calibration"
            )
        references.update(ticks)
        captures.update(
            {
                f"arm_{side}_{suffix}": entry["capture_tick"]
                for suffix, entry in mapping["joints"].items()
            }
        )
    if profile["reference_ticks"] != references:
        raise ValueError("Reference encoder ticks differ from their generated mappings")
    if profile["capture_ticks"] != captures:
        raise ValueError("Captured encoder ticks differ from the folded pose")
    return profile

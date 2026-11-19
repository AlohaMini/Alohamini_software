"""Prepare a verified v1-to-v2 wrist-limit update without opening hardware."""

import hashlib
from copy import deepcopy
from dataclasses import replace

from alohamini.calibration.reference import (
    calibration_bytes,
    json_bytes,
    load_reference_profile,
    side_calibration,
)
from alohamini.calibration.servo import load_motor_calibration
from alohamini.model import ActuatorSpec
from alohamini.model.joint_spec import ModelJointSpecification


def prepare_reference_upgrade(paths, robot_model, *, leader):
    """Return new values/files, or None; only the wrist upper limits may change.

    The caller writes and verifies hardware inside a calibration transaction,
    then installs the prepared files with save_bundle before committing it.
    """
    profiles = [
        load_reference_profile(path, robot_model, allow_previous=True)
        for path in dict.fromkeys(paths.values())
    ]
    if all(profile is None for profile in profiles):
        return None
    profile = profiles[0]
    if profile is None or any(item != profile for item in profiles):
        raise ValueError("Both calibration files must have the same reference profile")
    if profile["role"] != ("leader" if leader else "robot"):
        raise ValueError("Reference profile role differs from the calibration target")
    if set(profile["calibration_sha256"]) != {path.name for path in paths.values()}:
        raise ValueError("Reference profile identifies different calibration files")
    spec = ModelJointSpecification.from_model(profile["robot_model"])
    target = ModelJointSpecification.from_model(robot_model)
    if spec.joints != target.joints or spec.capture_positions != target.capture_positions:
        raise ValueError("Reference profile uses incompatible leader joint coordinates")
    if profile["specification_sha256"] == spec.signature:
        return None
    old_spec = ModelJointSpecification.from_signature(
        profile["robot_model"], profile["specification_sha256"]
    )
    if (
        old_spec.descriptor["specification_version"] != 1
        or spec.descriptor["specification_version"] != 2
    ):
        raise ValueError("Unsupported reference calibration upgrade")
    updated, motor_files, payloads = {}, {}, {}
    result = deepcopy(profile)
    for side in ("left", "right"):
        motors = tuple(m for m in spec.model.actuators if m.bus == side)
        if leader:
            motors = tuple(
                ActuatorSpec(m.name.removeprefix(f"arm_{side}_"), side, m.motor_id, "sts3215")
                for m in motors
                if m.name.startswith("arm_")
            )
        expected = motors if leader else spec.model.actuators
        values = load_motor_calibration(paths[side], expected)
        raw = {
            m.name: (
                profile["capture_ticks"][f"arm_{side}_{m.name}" if leader else m.name]
                + values[m.name].homing_offset
            )
            % 4096
            for m in motors
            if leader or m.name.startswith("arm_")
        }
        generated, ticks, mapping = side_calibration(spec, side, motors, raw, leader=leader)
        for name, value in generated.items():
            suffix = name if leader else name.removeprefix(f"arm_{side}_")
            previous = values[name]
            allowed = (
                replace(previous, range_max=value.range_max) if suffix == "wrist_flex" else previous
            )
            old_entry = profile["arms"][side]["joints"][suffix]
            new_entry = mapping["joints"][suffix]
            if (
                value != allowed
                or value.range_max < previous.range_max
                or any(
                    old_entry[key] != new_entry[key]
                    for key in (
                        "reference_tick",
                        "reference_q_rad",
                        "capture_tick",
                        "capture_q_rad",
                        "sign",
                    )
                )
                or ticks[f"arm_{side}_{suffix}"]
                != profile["reference_ticks"][f"arm_{side}_{suffix}"]
            ):
                raise ValueError(
                    f"{name}: upgrade would change an existing encoder zero or mapping"
                )
        updated[side] = {**values, **generated}
        result["arms"][side] = mapping
        motor_files[paths[side]] = expected
    if leader:
        payloads = {paths[side]: calibration_bytes(updated[side]) for side in updated}
    else:
        # Each side was loaded from the same whole-robot JSON.
        merged = {m.name: updated[m.bus][m.name] for m in spec.model.actuators}
        updated = {side: merged for side in updated}
        payloads = {paths["left"]: calibration_bytes(merged)}
    result["specification_sha256"] = spec.signature
    result["calibration_sha256"] = {
        path.name: hashlib.sha256(data).hexdigest() for path, data in payloads.items()
    }
    for path in tuple(payloads):
        payloads[path.with_name(f"{path.stem}.reference.json")] = json_bytes(result)
    return updated, payloads, motor_files

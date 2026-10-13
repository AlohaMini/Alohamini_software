"""Explicit transfer of normalized arm coordinates between calibrated robots."""

from copy import deepcopy

from alohamini.apps.replay import check_calibration
from alohamini.calibration.encoder import HostPositionUnits
from alohamini.datasets.record import state_names
from alohamini.model import get_robot_model


def legacy_normalized_binding(source, robot_model, snapshot):
    """Bind explicitly declared old action units, without inventing trained calibration.

    The deployment option declares normalized arm positions, m/s translation,
    deg/s rotation and mm lift targets. Host calibration maps these values; this
    does not recover the old robot's joint angles or lift origin.
    """
    if snapshot.robot_model != robot_model:
        raise ValueError("Declared deployment robot model differs from Host")
    names = source["features"]["action"]["names"]
    if names != state_names(robot_model):
        raise ValueError("Legacy action names/order differ from the declared Host model")
    live = snapshot.payload["_robot_metadata"]
    check_calibration(live, snapshot)
    for actuator in get_robot_model(robot_model).actuators:
        if not actuator.name.startswith("arm_"):
            continue
        entry = live["motors"][actuator.name]
        expected = "range_0_100" if actuator.name.endswith("_gripper") else "range_m100_100"
        if entry["normalization"] != expected:
            raise ValueError(f"Legacy normalized deployment requires {expected}: {actuator.name}")
        HostPositionUnits(
            **{key: entry[key] for key in ("normalization", "range_min", "range_max", "drive_mode")}
        )
        offset = entry["homing_offset"]
        if type(offset) is not int or not -2047 <= offset <= 2047:
            raise ValueError(f"Invalid Host homing offset: {actuator.name}")
    return deepcopy(live), dict(
        mode="legacy_normalized",
        meaning="Explicit recorded-unit declaration mapped by current Host calibration; "
        "old physical joint poses and lift origin are not reconstructed",
        declared_units={
            "arm_joints": "range_m100_100",
            "grippers": "range_0_100",
            "base_linear": "m/s",
            "base_angular": "deg/s",
            "lift": "mm",
        },
        trained_metadata=None,
        training_provenance=deepcopy(source.get("training_only")),
        host_metadata=deepcopy(live),
    )


def normalized_transfer(trained, snapshot):
    """Keep normalized values; use the new robot's own ranges and encoder offsets.

    This transfers fractional joint positions, not Cartesian poses. Identities,
    units, inversion, base and lift calibration retain the strict contract.
    No calibration file, register, checkpoint or model statistic is modified.
    """
    live = snapshot.payload["_robot_metadata"]
    candidate = deepcopy(trained)
    differences = {}
    allowed = ("range_min", "range_max", "homing_offset")
    unit_fields = ("normalization", "range_min", "range_max", "drive_mode")
    for actuator in get_robot_model(snapshot.robot_model).actuators:
        name = actuator.name
        if not name.startswith("arm_"):
            continue
        old = candidate.get("motors", {}).get(name, {})
        new = live.get("motors", {}).get(name, {})
        for entry in (old, new):
            if entry.get("normalization") not in ("range_m100_100", "range_0_100"):
                raise ValueError(f"Normalized transfer requires normalized arm positions: {name}")
            if any(key not in entry for key in (*allowed, *unit_fields)):
                raise ValueError(f"Incomplete calibration for normalized transfer: {name}")
            HostPositionUnits(**{key: entry[key] for key in unit_fields})
            if (
                type(entry["homing_offset"]) is not int
                or not -2047 <= entry["homing_offset"] <= 2047
            ):
                raise ValueError(f"Invalid homing offset for normalized transfer: {name}")
        changes = {
            key: {"trained": old[key], "Host": new[key]} for key in allowed if old[key] != new[key]
        }
        if changes:
            differences[name] = changes
        old.update({key: new[key] for key in allowed})
    check_calibration(candidate, snapshot)
    return deepcopy(live), dict(
        mode="normalized",
        meaning=(
            "Same normalized arm values in the destination joint ranges; "
            "not Cartesian pose preservation"
        ),
        trained_metadata=deepcopy(trained),
        host_metadata=deepcopy(live),
        differences=differences,
    )

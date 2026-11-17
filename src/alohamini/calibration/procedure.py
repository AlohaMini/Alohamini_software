"""One reference-pose calibration; no range sweep, motion goal or torque enable."""

import hashlib
import logging
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from uuid import uuid4

from alohamini.calibration.reference import (
    PROTOCOL,
    calibration_bytes,
    json_bytes,
    load_reference_profile,
    save_bundle,
    side_calibration,
)
from alohamini.calibration.servo import MotorCalibration, load_motor_calibration
from alohamini.hardware.feetech_device import (
    CalibrationMismatchError,
    DeviceOperationError,
    FeetechBusDevice,
)
from alohamini.model import ActuatorSpec
from alohamini.model.joint_spec import ModelJointSpecification
from alohamini.paths import WorkspacePaths

logger = logging.getLogger(__name__)


def ensure_calibration(buses, calibrations, paths):
    """Restore a saved file on explicit confirmation; new zeroing has one CLI entry."""
    mismatches = {}
    for side, bus in buses.items():
        try:
            bus.verify_calibration(calibrations[side])
        except CalibrationMismatchError as error:
            mismatches.setdefault(paths[side], error)
    if not mismatches:
        return False
    if not sys.stdin.isatty():
        path, error = next(iter(mismatches.items()))
        raise CalibrationMismatchError(
            f"{error}\nCalibration file: {path}\n"
            "Interactive confirmation required to restore calibration."
        ) from error
    print("Support both arms and the lift; torque is disabled.", flush=True)
    for path in mismatches:
        response = input(
            f"Calibration file: {path}\nPress ENTER to restore this saved calibration, "
            "or enter any text to cancel and run alohamini calibrate: "
        )
        if response.strip():
            raise InterruptedError("Calibration restore cancelled; use alohamini calibrate to zero")
    with ExitStack() as transactions:
        for bus in buses.values():
            transactions.enter_context(bus.calibration_restore_session())
        for side, bus in buses.items():
            bus.write_calibration(calibrations[side])
        for side, bus in buses.items():
            bus.verify_calibration(calibrations[side])
    return False


@contextmanager
def _connected_bus(bus, session):
    primary = None
    try:
        bus.connect(session)
        yield bus
    except BaseException as error:
        primary = error
        raise
    finally:
        failures = {}
        for label, operation in (("disable torque", bus.disable_torque), ("close", bus.close)):
            try:
                operation()
            except BaseException as error:
                failures[label] = f"{type(error).__name__}: {error}"
        if failures:
            if primary is not None and not isinstance(primary, Exception):
                primary.add_note(str(failures))
                logger.error("Calibration cleanup failed: %s", failures)
            else:
                if primary is not None:
                    failures["calibration"] = str(primary)
                raise DeviceOperationError(failures) from primary


def _capture(bus, names, policy):
    samples = [bus.read_positions(names) for _ in range(policy["samples"])]
    result = {}
    for name in names:
        if any(type(sample[name]) is not int or not 0 <= sample[name] < 4096 for sample in samples):
            raise ValueError(f"{name}: reference encoder is outside the single-turn range")
        first = samples[0][name]
        offsets = [(sample[name] - first + 2048) % 4096 - 2048 for sample in samples]
        if (max(offsets) - min(offsets)) * 360 / 4096 > policy["max_span_deg"]:
            raise ValueError(f"{name}: reference pose moved; hold it still and retry calibration")
        result[name] = round(first + sum(offsets) / len(offsets)) % 4096
    return result


def calibrate(
    target,
    robot_model,
    *,
    device_id=None,
    calibration_dir=None,
    left_port=None,
    right_port=None,
    arm_profile=None,
):
    """Zero both arms in one known pose and generate all dependent device mappings."""
    if target not in ("leader", "robot"):
        raise ValueError("Calibration target must be leader or robot")
    spec = ModelJointSpecification.from_model(robot_model)
    model, leader = spec.model, target == "leader"
    if arm_profile is not None and (not leader or arm_profile != "am-leader-6dof"):
        raise ValueError("Reference calibration requires the model's am-leader-6dof profile")
    if not sys.stdin.isatty():
        raise RuntimeError("Calibration requires an interactive terminal")
    device_id = device_id or ("am_leader_bi" if leader else "AlohaMiniRobot")
    workspace = WorkspacePaths()
    role = "teleoperators" if leader else "robots"
    directory = (
        Path(calibration_dir).expanduser()
        if calibration_dir is not None
        else workspace.calibration / role
    )
    ports = (
        left_port or f"/dev/am_arm_{'leader' if leader else 'follower'}_left",
        right_port or f"/dev/am_arm_{'leader' if leader else 'follower'}_right",
    )
    if Path(ports[0]).resolve() == Path(ports[1]).resolve():
        raise ValueError("Left and right buses must use different serial devices")
    paths, actuators, existing = {}, {}, {}
    for side in ("left", "right"):
        file_id = f"{device_id}_{side}" if leader else device_id
        default = workspace.calibration_file(role, file_id)
        paths[side] = directory / default.name
        if paths[side].is_symlink():
            raise ValueError("Use a shared calibration directory, not a linked JSON file")
        motors = tuple(m for m in model.actuators if m.bus == side)
        if leader:
            motors = tuple(
                ActuatorSpec(m.name.removeprefix(f"arm_{side}_"), side, m.motor_id, "sts3215")
                for m in motors
                if m.name.startswith("arm_")
            )
        actuators[side] = motors
        expected = motors if leader else model.actuators
        existing[side] = (
            load_motor_calibration(paths[side], expected) if paths[side].exists() else {}
        )
    reuse = False
    if all(existing.values()):
        response = input(f"{device_id} 已有标定：按 Enter 复用，输入 c 重新摆位标定：").strip()
        if response not in ("", "c"):
            raise InterruptedError("Calibration cancelled; no serial port opened")
        reuse = response == ""
        if reuse:
            for path in set(paths.values()):
                load_reference_profile(path, robot_model)
    if leader and left_port is None and right_port is None:
        from alohamini.hardware.usb_setup import ensure_arm_ports

        ports = ensure_arm_ports("leader", workspace=workspace)
    with ExitStack() as cleanup:
        buses = {}
        for side, port in zip(("left", "right"), ports, strict=True):
            bus = FeetechBusDevice(
                port, actuators[side], position_calibrations={}, velocity_limits={}
            )
            buses[side] = cleanup.enter_context(_connected_bus(bus, uuid4().hex))
        for bus in buses.values():
            bus.disable_torque()
        with ExitStack() as transactions:
            for bus in buses.values():
                transactions.enter_context(bus.calibration_restore_session())
            if reuse:
                for side, bus in buses.items():
                    values = {m.name: existing[side][m.name] for m in bus.actuators}
                    bus.write_calibration(values)
                return
            label = "主臂" if leader else "从臂"
            response = input(
                f"将双{label}摆到 Home：1/5/6 对齐中位，2/3/4 蜷缩轻靠接触面，夹爪合拢。\n"
                "不要挤压外壳；确认双臂稳固、升降已支撑。\n"
                "双臂保持静止，按 Enter 标定（输入其他内容取消）："
            )
            if response.strip():
                raise InterruptedError("Reference capture cancelled")
            original, names, values, reference_ticks, mappings = {}, {}, {}, {}, {}
            for side, bus in buses.items():
                original[side] = bus.read_calibration()
                names[side] = [m.name for m in bus.actuators if leader or m.name.startswith("arm_")]
                bus.configure_calibration(names[side], preserve_other_motors=True)
                bus.write_calibration(
                    {
                        m.name: MotorCalibration(m.motor_id, 0, 0, 0, 4095)
                        for m in bus.actuators
                        if m.name in names[side]
                    }
                )
                arm, ticks, mapping = side_calibration(
                    spec,
                    side,
                    bus.actuators,
                    _capture(bus, names[side], spec.descriptor["reference_capture"]),
                    leader=leader,
                )
                values[side] = arm if leader else {**original[side], **arm}
                reference_ticks.update(ticks)
                mappings[side] = mapping
                bus.write_calibration({name: values[side][name] for name in names[side]})
                readback = _capture(bus, names[side], spec.descriptor["reference_capture"])
                for name, actual in readback.items():
                    suffix = name if leader else name.removeprefix(f"arm_{side}_")
                    expected = mapping["joints"][suffix]["capture_tick"]
                    error_deg = abs(actual - expected) * 360 / 4096
                    if error_deg > spec.descriptor["reference_capture"]["max_span_deg"]:
                        raise CalibrationMismatchError(
                            f"{name}: reference position readback differs after offset write "
                            f"(expected={expected}, actual={actual}); hold the supported arm "
                            "still and retry; if this persists, power-cycle servo power"
                        )
                print(f"{'左' if side == 'left' else '右'}{label}已写入并核对。")
            if leader:
                motor_payloads = {paths[side]: calibration_bytes(values[side]) for side in buses}
                motor_files = {paths[side]: actuators[side] for side in buses}
            else:
                all_values = {name: value for side in buses for name, value in values[side].items()}
                motor_payloads = {paths["left"]: calibration_bytes(all_values)}
                motor_files = {paths["left"]: model.actuators}
            profile = dict(
                schema_version=1,
                protocol=PROTOCOL,
                robot_model=robot_model,
                role=target,
                reference_pose=spec.descriptor["reference_pose"],
                capture_pose=spec.descriptor["calibration_pose"]["name"],
                capture_ticks={
                    f"arm_{side}_{suffix}": entry["capture_tick"]
                    for side, mapping in mappings.items()
                    for suffix, entry in mapping["joints"].items()
                },
                specification_sha256=spec.signature,
                boundary_policy=spec.descriptor["boundary_policy"],
                reference_ticks=reference_ticks,
                arms=mappings,
                calibration_sha256={
                    path.name: hashlib.sha256(data).hexdigest()
                    for path, data in motor_payloads.items()
                },
            )
            payloads = dict(motor_payloads)
            for path in motor_payloads:
                payloads[path.with_name(f"{path.stem}.reference.json")] = json_bytes(profile)
            save_bundle(payloads, motor_files)
            for path in motor_payloads:
                print(f"标定已保存：{path}")
            print("双臂保持扭矩关闭；标定掉电保留。")

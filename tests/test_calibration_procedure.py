import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_feetech_device import DelayedRegisterSerial, RegisterSerial

from alohamini.calibration.procedure import calibrate
from alohamini.calibration.interaction import CalibrationInteraction
from alohamini.calibration.reference import load_reference_profile
from alohamini.calibration.servo import load_motor_calibration
from alohamini.model import ActuatorSpec, get_robot_model


class CalibrationSerial(RegisterSerial):
    """Retain mechanical position when the EEPROM homing offset changes."""

    def write(self, packet):
        motor_id, instruction, address = packet[2], packet[4], packet[5]
        if instruction == 3 and address == 31:
            previous = self.get(motor_id, 31, 2)
            offset = -(previous & 0x7FF) if previous & 0x800 else previous
            physical = (self.get(motor_id, 56, 2) + offset) % 4096
            result = super().write(packet)
            current = self.get(motor_id, 31, 2)
            offset = -(current & 0x7FF) if current & 0x800 else current
            self.set(motor_id, 56, 2, (physical - offset) % 4096)
            return result
        return super().write(packet)


class DelayedCalibrationSerial(DelayedRegisterSerial, CalibrationSerial):
    pass


@unittest.skipUnless(importlib.util.find_spec("scservo_sdk"), "Feetech SDK unavailable")
class CalibrationProcedureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.serials = {}
        factory_patch = patch("serial.Serial", side_effect=self.serial_factory)
        factory_patch.start()
        self.addCleanup(factory_patch.stop)
        tty_patch = patch("sys.stdin.isatty", return_value=True)
        tty_patch.start()
        self.addCleanup(tty_patch.stop)

    def serial_factory(self, **_kwargs):
        serials = self.serials

        class PortSelection:
            def __setattr__(self, name, value):
                if name == "port":
                    object.__setattr__(self, "selected", serials[value])
                else:
                    setattr(self.selected, name, value)

            def __getattr__(self, name):
                return getattr(self.selected, name)

        return PortSelection()

    def configure(
        self, target="leader", model_name="alohamini2pro", *, serial_type=CalibrationSerial
    ):
        self.target, self.model_name = target, model_name
        self.actuators = {}
        self.original = {}
        for side in ("left", "right"):
            motors = [m for m in get_robot_model(model_name).actuators if m.bus == side]
            if target == "leader":
                motors = [
                    ActuatorSpec(m.name.removeprefix(f"arm_{side}_"), side, m.motor_id, "sts3215")
                    for m in motors
                    if m.name.startswith("arm_")
                ]
            self.actuators[side] = motors
            serial = serial_type()
            template = serial.registers[1].copy()
            serial.registers = {m.motor_id: template.copy() for m in motors}
            for motor in motors:
                for address, width, value in (
                    (3, 2, {"sts3215": 777, "sts3250": 2825, "sts3095": 2569}[motor.motor_model]),
                    (7, 1, 250),
                    (9, 2, 200),
                    (11, 2, 3500),
                    (18, 1, 0x1C),
                    (31, 2, 0x800 | 123),
                    (33, 1, 0),
                    (55, 1, 1),
                    (56, 2, 1700),
                ):
                    serial.set(motor.motor_id, address, width, value)
            self.serials[f"/dev/test-{side}"] = serial
            self.original[side] = {i: bytes(values) for i, values in serial.registers.items()}

    def run_calibration(self, response=""):
        responses = response if isinstance(response, list) else [response, response]
        with (
            patch("builtins.input", side_effect=responses) as prompt,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            calibrate(
                self.target,
                self.model_name,
                calibration_dir=self.root,
                left_port="/dev/test-left",
                right_port="/dev/test-right",
            )
        return prompt

    def assert_passive_and_closed(self):
        for serial in self.serials.values():
            self.assertFalse(serial.is_open)
            for motor_id in serial.registers:
                self.assertEqual(serial.get(motor_id, 40, 1), 0)
                self.assertEqual(serial.get(motor_id, 55, 1), 1)
            for packet in serial.requests:
                if packet[4] == 3:
                    self.assertNotIn(packet[5], (42, 46))  # No motion targets, including homing.
                    if packet[5] == 40:
                        self.assertEqual(packet[6], 0)

    def assert_restored(self, sides=("left", "right")):
        for side in sides:
            serial = self.serials[f"/dev/test-{side}"]
            for motor_id, expected in self.original[side].items():
                for address, width in ((7, 1), (9, 2), (11, 2), (18, 1), (31, 2), (33, 1)):
                    self.assertEqual(
                        bytes(serial.registers[motor_id][address : address + width]),
                        expected[address : address + width],
                        (side, motor_id, address),
                    )

    @contextlib.contextmanager
    def gui_interaction(self, *, cancel_at=None, wrong_prompt=False):
        read_fd, write_fd = os.pipe()
        events = []
        with os.fdopen(read_fd) as source, os.fdopen(write_fd, "w") as control:
            class Output(io.StringIO):
                def flush(output):
                    event = json.loads(output.getvalue().splitlines()[-1])
                    events.append(event)
                    if event["step"] == cancel_at:
                        control.write('{"action":"cancel"}\n')
                        control.flush()
                        self.assertTrue(interaction.cancelled.wait(1), "取消信号未送达")
                        return
                    if event["prompt_id"] is not None:
                        control.write(json.dumps(dict(action="confirm", prompt_id=(
                            event["prompt_id"] + 1 if wrong_prompt else event["prompt_id"]
                        ))) + "\n")
                        control.flush()

            interaction = CalibrationInteraction(source, Output())
            with patch("sys.stdin.isatty", return_value=False), patch(
                "builtins.input", side_effect=AssertionError("桌面标定不得读取终端")
            ), contextlib.redirect_stdout(io.StringIO()):
                yield interaction, events

    def test_gui_recalibration_uses_existing_procedure_without_terminal_input(self):
        self.configure()
        self.run_calibration()
        with self.gui_interaction() as (interaction, events):
            calibrate("leader", self.model_name, calibration_dir=self.root,
                      left_port="/dev/test-left", right_port="/dev/test-right", interaction=interaction)
        self.assertEqual([e["step"] for e in events], [
            "connecting", "home", "writing_left", "writing_right", "saving"
        ])
        self.assertIsNotNone(load_reference_profile(self.root / "am_leader_bi_left.json", self.model_name))
        self.assert_passive_and_closed()

    def test_gui_cancel_during_second_arm_rolls_back_both_arms_and_files(self):
        self.configure()
        with self.gui_interaction(cancel_at="writing_right") as (interaction, _):
            with self.assertRaises(InterruptedError):
                calibrate("leader", self.model_name, calibration_dir=self.root,
                          left_port="/dev/test-left", right_port="/dev/test-right", interaction=interaction)
        self.assert_restored()
        self.assert_passive_and_closed()
        self.assertFalse(list(self.root.glob("*.json")))

    def test_gui_replaces_unreadable_saved_calibration(self):
        self.configure()
        (self.root / "am_leader_bi_left.json").write_text("损坏的旧文件", encoding="utf-8")
        with self.gui_interaction() as (interaction, _):
            calibrate("leader", self.model_name, calibration_dir=self.root,
                      left_port="/dev/test-left", right_port="/dev/test-right", interaction=interaction)
        self.assertIsNotNone(load_reference_profile(self.root / "am_leader_bi_left.json", self.model_name))
        self.assert_passive_and_closed()

    def test_gui_linux_registration_is_reused_by_default_runtime_without_aliases(self):
        from alohamini.hardware import leader, usb_setup

        self.configure()
        directory = self.root / "calibration/devices"
        directory.mkdir(parents=True)
        bindings = {side: usb_setup.USBController(f"GUI-{side}", 1, 2) for side in ("left", "right")}
        (directory / "leader.json").write_text(json.dumps(dict(
            schema_version=1, role="leader", sides={side: asdict(board) for side, board in bindings.items()}
        )))
        with patch.dict(os.environ, ALOHAMINI_WORKSPACE=str(self.root)), patch.object(
            usb_setup, "usb_ports", return_value={
                f"/dev/test-{side}": SimpleNamespace(**asdict(board)) for side, board in bindings.items()
            }
        ), patch.object(leader.sys, "platform", "linux"), self.gui_interaction() as (interaction, _):
            calibrate("leader", self.model_name, interaction=interaction)
            runtime = leader.BimanualLeader(self.model_name)
            self.assertEqual(runtime._ports, {side: f"/dev/test-{side}" for side in bindings})
            runtime.connect()
            runtime.close()
        self.assert_passive_and_closed()

    def test_gui_stale_confirmation_cannot_zero_motors(self):
        self.configure()
        with self.gui_interaction(wrong_prompt=True) as (interaction, _):
            with self.assertRaisesRegex(ValueError, "步骤已变化"):
                calibrate("leader", self.model_name, calibration_dir=self.root,
                          left_port="/dev/test-left", right_port="/dev/test-right", interaction=interaction)
        self.assert_restored()
        self.assert_passive_and_closed()
        self.assertFalse(list(self.root.glob("*.json")))

    def test_one_pose_writes_both_leaders_and_generates_matching_profiles(self):
        self.configure()
        prompt = self.run_calibration()
        self.assertEqual(prompt.call_count, 1)
        for side in ("left", "right"):
            path = self.root / f"am_leader_bi_{side}.json"
            values = load_motor_calibration(path, self.actuators[side])
            profile = load_reference_profile(path, "alohamini2pro")
            self.assertEqual(len(values), 7)
            self.assertEqual(len(profile["reference_ticks"]), 14)
            for motor in self.actuators[side]:
                entry = values[motor.name]
                self.assertEqual(entry.drive_mode, 0)
                full_name = f"arm_{side}_{motor.name}"
                captured = profile["capture_ticks"][full_name]
                self.assertEqual((1577 - entry.homing_offset) % 4096, captured)
                self.assertEqual(
                    self.serials[f"/dev/test-{side}"].get(motor.motor_id, 56, 2), captured
                )
                self.assertEqual(
                    profile["arms"][side]["joints"][motor.name]["reference_tick"], 2047
                )
        self.assert_passive_and_closed()

    def test_alohamini2_uses_its_own_follower_motors_with_the_shared_home(self):
        self.configure("robot", "alohamini2")
        prompt = self.run_calibration()
        self.assertEqual(prompt.call_count, 1)
        path = self.root / "AlohaMiniRobot.json"
        profile = load_reference_profile(path, "alohamini2")
        self.assertEqual(profile["robot_model"], "alohamini2")
        for side in ("left", "right"):
            mapping = profile["arms"][side]["joints"]
            self.assertEqual(mapping["shoulder_pan"]["model"], "sts3095")
            self.assertEqual(mapping["wrist_flex"]["model"], "sts3215")
        self.assert_passive_and_closed()

    def test_one_confirmation_captures_both_arms(self):
        self.configure("robot")
        prompts = []

        def place(prompt):
            prompts.append(prompt)
            self.assertFalse(list(self.root.rglob("*.json")))
            left, right = self.serials["/dev/test-left"], self.serials["/dev/test-right"]
            self.assertIn("both follower arms", prompt)
            left.set(1, 56, 2, 1930)
            right.set(1, 56, 2, 2400)
            return ""

        with patch("builtins.input", side_effect=place), contextlib.redirect_stdout(io.StringIO()):
            calibrate(
                "robot",
                self.model_name,
                calibration_dir=self.root,
                left_port="/dev/test-left",
                right_port="/dev/test-right",
            )
        values = load_motor_calibration(
            self.root / "AlohaMiniRobot.json", get_robot_model(self.model_name).actuators
        )
        self.assertEqual(len(prompts), 1)
        self.assertEqual(values["arm_left_shoulder_pan"].homing_offset, -240)
        self.assertEqual(values["arm_right_shoulder_pan"].homing_offset, 230)
        self.assert_passive_and_closed()

    def test_second_side_failure_restores_the_already_zeroed_first_side(self):
        from alohamini.calibration.reference import side_calibration

        self.configure("robot")

        def fail_right(spec, side, *args, **kwargs):
            if side == "right":
                raise RuntimeError("Right reference failed")
            return side_calibration(spec, side, *args, **kwargs)

        with patch("alohamini.calibration.procedure.side_calibration", side_effect=fail_right):
            with self.assertRaisesRegex(RuntimeError, "Right reference failed"):
                self.run_calibration()
        self.assert_restored()
        self.assert_passive_and_closed()
        self.assertFalse(list(self.root.rglob("*.json")))

    def test_incorrect_post_offset_position_is_rejected_and_rolled_back(self):
        from alohamini.hardware.feetech_device import FeetechBusDevice

        self.configure("robot")
        actual_read = FeetechBusDevice.read_positions
        count = 0

        def shifted(bus, names):
            nonlocal count
            count += 1
            positions = actual_read(bus, names)
            return {name: value + (20 if count > 3 else 0) for name, value in positions.items()}

        with patch.object(FeetechBusDevice, "read_positions", new=shifted):
            with self.assertRaisesRegex(ValueError, "readback differs"):
                self.run_calibration()
        self.assert_restored()
        self.assert_passive_and_closed()
        self.assertFalse(list(self.root.rglob("*.json")))

    def test_full_turn_feedback_is_not_silently_wrapped_during_capture(self):
        self.configure("robot")
        with patch(
            "alohamini.hardware.feetech_device.FeetechBusDevice.read_positions",
            side_effect=lambda names: dict.fromkeys(names, 6144),
        ):
            with self.assertRaisesRegex(ValueError, "outside the single-turn range"):
                self.run_calibration()
        self.assert_restored()
        self.assert_passive_and_closed()

    def test_default_ports_run_usb_setup_then_continue_the_same_reference_capture(self):
        self.configure()
        with (
            patch(
                "alohamini.hardware.usb_setup.ensure_arm_ports",
                return_value=("/dev/test-left", "/dev/test-right"),
            ) as setup,
            patch("builtins.input", return_value="") as prompt,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            calibrate("leader", self.model_name, calibration_dir=self.root)
        self.assertEqual(setup.call_args.args, ("leader",))
        self.assertEqual(prompt.call_count, 1)
        self.assertIsNotNone(
            load_reference_profile(self.root / "am_leader_bi_left.json", self.model_name)
        )
        self.assert_passive_and_closed()

    def test_usb_setup_failure_never_opens_or_writes_motor_ports(self):
        self.configure()
        with (
            patch(
                "alohamini.hardware.usb_setup.ensure_arm_ports",
                side_effect=PermissionError("sudo cancelled"),
            ),
            self.assertRaises(PermissionError),
        ):
            calibrate("leader", self.model_name, calibration_dir=self.root)
        self.assertFalse(any(serial.requests for serial in self.serials.values()))
        self.assertFalse(list(self.root.rglob("*.json")))

    def test_closed_port_failure_preserves_permission_or_lock_error(self):
        from serial import SerialException

        for code in (13, 11):
            for side in ("left", "right"):
                with self.subTest(errno=code, side=side):
                    self.configure()
                    failed = self.serials[f"/dev/test-{side}"]
                    error = SerialException(
                        code, "Permission denied" if code == 13 else "Port busy"
                    )
                    with patch.object(failed, "open", side_effect=error):
                        with self.assertRaises(SerialException) as raised:
                            self.run_calibration()
                    self.assertIs(raised.exception, error)
                    self.assertFalse(failed.requests)
                    for bus_side in ("left", "right"):
                        serial = self.serials[f"/dev/test-{bus_side}"]
                        self.assertFalse(serial.is_open)
                        if side == "right" and bus_side == "left":
                            # Opening the second bus failed; the first must still unload.
                            for motor_id in serial.registers:
                                self.assertEqual(serial.get(motor_id, 40, 1), 0)
                            self.assert_restored(("left",))
                        else:
                            self.assertEqual(
                                {i: bytes(v) for i, v in serial.registers.items()},
                                self.original[bus_side],
                            )
                    self.assertFalse(list(self.root.rglob("*.json")))

    def test_partial_motor_verification_still_unloads_verified_motors(self):
        from alohamini.hardware.feetech_device import DeviceOperationError

        self.configure()
        serial = self.serials["/dev/test-left"]
        serial.set(1, 40, 1, 1)
        serial.set(2, 3, 2, 0)  # The first motor verifies, the second has the wrong model.
        with self.assertRaisesRegex(DeviceOperationError, "Motor model mismatch"):
            self.run_calibration()
        self.assertEqual(serial.get(1, 40, 1), 0)
        writes = [packet for packet in serial.requests if packet[4] == 3]
        self.assertTrue(writes)
        self.assertTrue(all(packet[2] == 1 and packet[5] == 40 for packet in writes))
        self.assertFalse(serial.is_open)
        self.assertFalse(self.serials["/dev/test-right"].requests)
        self.assertFalse(list(self.root.rglob("*.json")))

    def test_default_storage_and_runtime_follow_the_connected_leader_set(self):
        from alohamini.hardware import usb_setup
        from alohamini.hardware.leader import BimanualLeader
        from alohamini.paths import WorkspacePaths

        self.configure()
        workspace = WorkspacePaths(self.root)
        bindings = {
            name: {
                side: usb_setup.USBController(f"{name}-{side}", 1, 2) for side in ("left", "right")
            }
            for name in ("A", "B")
        }
        for name, pair in bindings.items():
            # The most recently selected pair is B, even while A is attached.
            path = (
                workspace.calibration
                / "devices"
                / (f"leaders/{usb_setup._pair_key(pair)}.json" if name == "A" else "leader.json")
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    dict(
                        schema_version=1,
                        role="leader",
                        sides={side: asdict(board) for side, board in pair.items()},
                    )
                )
            )
        current = "A"

        def available():
            return {
                f"/dev/test-{side}": SimpleNamespace(**asdict(board))
                for side, board in bindings[current].items()
            }

        ports = dict(left_port="/dev/test-left", right_port="/dev/test-right")
        with (
            patch.dict(os.environ, ALOHAMINI_WORKSPACE=str(self.root)),
            patch.object(usb_setup, "usb_ports", side_effect=available),
            patch("builtins.input", return_value="") as prompt,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            calibrate("leader", self.model_name, **ports)
            first_serials = self.serials
            first = BimanualLeader(self.model_name, **ports)
            first_files = {p: p.read_bytes() for p in first._calibration_paths.values()}
            first_offset = first.calibrations["left"]["shoulder_pan"].homing_offset

            current = "B"
            self.configure()
            for serial in self.serials.values():
                for motor_id in serial.registers:
                    serial.set(motor_id, 56, 2, 2000)
            calibrate("leader", self.model_name, **ports)
            second = BimanualLeader(self.model_name, **ports)
            self.assertNotEqual(
                second.calibrations["left"]["shoulder_pan"].homing_offset, first_offset
            )
            self.assertNotEqual(first._calibration_paths, second._calibration_paths)
            self.assertEqual(prompt.call_count, 2)  # One Home capture per new pair.

            current, self.serials = "A", first_serials
            with patch(
                "alohamini.calibration.procedure._capture",
                side_effect=AssertionError("must reuse A without Home capture"),
            ):
                calibrate("leader", self.model_name, **ports)
            selected = BimanualLeader(self.model_name, **ports)
            self.assertEqual(selected.calibrations, first.calibrations)
            self.assertEqual({p: p.read_bytes() for p in first_files}, first_files)
            self.assertEqual(prompt.call_count, 3)
        self.assert_passive_and_closed()

    def test_explicit_directory_does_not_touch_default_calibration_storage(self):
        from alohamini.hardware.usb_setup import USBController

        self.configure()
        workspace = self.root / "unused-workspace"
        flat = workspace / "calibration/teleoperators"
        flat.mkdir(parents=True)
        for side in ("left", "right"):
            (flat / f"am_leader_bi_{side}.json").write_text("unrelated legacy calibration")
        binding = workspace / "calibration/devices/leader.json"
        binding.parent.mkdir()
        binding.write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    role="leader",
                    sides={
                        side: asdict(USBController(f"old-{side}", 1, 2))
                        for side in ("left", "right")
                    },
                )
            )
        )
        before = {p: p.read_bytes() for p in workspace.rglob("*.json")}
        with patch.dict(os.environ, ALOHAMINI_WORKSPACE=str(workspace)):
            self.run_calibration()
        self.assertEqual({p: p.read_bytes() for p in workspace.rglob("*.json")}, before)

    def test_follower_uses_existing_ports_without_usb_binding(self):
        self.configure("robot")
        for side in ("left", "right"):
            self.serials[f"/dev/am_arm_follower_{side}"] = self.serials[f"/dev/test-{side}"]
        with (
            patch("alohamini.hardware.usb_setup.ensure_arm_ports") as setup,
            patch("builtins.input", return_value="") as prompt,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            calibrate("robot", self.model_name, calibration_dir=self.root)
        setup.assert_not_called()
        self.assertEqual(prompt.call_count, 1)
        self.assertIsNotNone(
            load_reference_profile(self.root / "AlohaMiniRobot.json", self.model_name)
        )
        self.assert_passive_and_closed()

    def test_robot_one_pose_preserves_velocity_axes_and_saves_shared_bindings(self):
        self.configure("robot")
        self.run_calibration()
        path = self.root / "AlohaMiniRobot.json"
        values = load_motor_calibration(path, get_robot_model(self.model_name).actuators)
        profile = load_reference_profile(path, self.model_name)
        self.assertEqual(len(values), 18)
        for side, motors in self.actuators.items():
            serial = self.serials[f"/dev/test-{side}"]
            for motor in motors:
                if not motor.name.startswith("arm_"):
                    for address in range(96):
                        if address != 40:
                            self.assertEqual(
                                serial.registers[motor.motor_id][address],
                                self.original[side][motor.motor_id][address],
                            )
            self.assertEqual(
                set(profile["arms"][side]["joints"]),
                {
                    motor.name.removeprefix(f"arm_{side}_")
                    for motor in motors
                    if motor.name.startswith("arm_")
                },
            )
        self.assert_passive_and_closed()

    def test_counter_endpoints_can_be_zeroed_without_an_offset_overflow(self):
        self.configure()
        self.serials["/dev/test-left"].set(1, 56, 2, 122)
        self.serials["/dev/test-right"].set(1, 56, 2, 123)
        self.run_calibration()
        left = load_reference_profile(self.root / "am_leader_bi_left.json", self.model_name)
        self.assertEqual(left["reference_ticks"]["arm_left_shoulder_pan"], 2048)
        self.assertEqual(left["reference_ticks"]["arm_right_shoulder_pan"], 2047)
        self.assert_passive_and_closed()

    def test_reuse_does_not_capture_or_rewrite_files(self):
        self.configure("robot")
        self.run_calibration()
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch(
            "alohamini.calibration.procedure._capture", side_effect=AssertionError("new capture")
        ):
            prompt = self.run_calibration()
        self.assertEqual(prompt.call_count, 1)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assert_passive_and_closed()

    def test_profile_drift_is_rejected_before_ports_are_opened(self):
        self.configure("robot")
        self.run_calibration()
        path = self.root / "AlohaMiniRobot.json"
        data = json.loads(path.read_text())
        data["arm_left_shoulder_pan"]["homing_offset"] += 1
        path.write_text(json.dumps(data))
        with patch("alohamini.calibration.procedure.FeetechBusDevice") as bus:
            with self.assertRaisesRegex(ValueError, "Reference profile"):
                self.run_calibration()
            bus.assert_not_called()

    def test_cancel_leaves_all_calibration_registers_unchanged(self):
        self.configure("robot")
        with self.assertRaises(InterruptedError):
            self.run_calibration("cancel")
        self.assert_restored()
        self.assert_passive_and_closed()
        self.assertFalse(list(self.root.rglob("*.json")))

    def test_file_failure_rolls_back_both_eeprom_and_all_installed_files(self):
        self.configure("robot")
        self.run_calibration()
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.original = {
            side: {i: bytes(v) for i, v in self.serials[f"/dev/test-{side}"].registers.items()}
            for side in ("left", "right")
        }
        actual_replace = os.replace
        count = 0

        def fail_once(source, target):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("injected file replacement failure")
            return actual_replace(source, target)

        with patch("alohamini.calibration.reference.os.replace", side_effect=fail_once):
            with self.assertRaises(OSError):
                self.run_calibration(["c", "", ""])
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.assert_restored()
        self.assert_passive_and_closed()
        load_reference_profile(self.root / "AlohaMiniRobot.json", self.model_name)

    def test_motor_failure_rolls_back_both_sides_and_writes_no_files(self):
        self.configure("robot")
        self.serials["/dev/test-right"].errors[(1, 0x82, 56)] = 4
        with self.assertRaises(ConnectionError):
            self.run_calibration()
        self.assert_restored()
        self.assert_passive_and_closed()
        self.assertFalse(list(self.root.rglob("*.json")))

    def test_moving_reference_pose_is_rejected_and_restored(self):
        self.configure()
        counter = 0

        def moving(bus, names):
            nonlocal counter
            counter += 1
            return dict.fromkeys(names, 1000 + counter * 20)

        with patch("alohamini.hardware.feetech_device.FeetechBusDevice.read_positions", new=moving):
            with self.assertRaisesRegex(ValueError, "moved"):
                self.run_calibration()
        self.assert_restored()
        self.assert_passive_and_closed()

    def test_delayed_fragmented_replies_still_calibrate_one_pose(self):
        self.configure("robot", serial_type=DelayedCalibrationSerial)
        self.run_calibration()
        self.assert_passive_and_closed()

    def test_no_arms_sweep_target_or_headless_calibration(self):
        self.configure()
        with patch("alohamini.calibration.procedure.FeetechBusDevice") as bus:
            with self.assertRaises(ValueError):
                calibrate("arms", "alohamini2pro")
            with patch("sys.stdin.isatty", return_value=False):
                with self.assertRaises(RuntimeError):
                    calibrate("leader", "alohamini2pro")
            bus.assert_not_called()

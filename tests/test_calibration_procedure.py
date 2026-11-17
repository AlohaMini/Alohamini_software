import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_feetech_device import DelayedRegisterSerial, RegisterSerial

from alohamini.calibration.procedure import calibrate
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

    def test_one_pose_writes_both_leaders_and_generates_matching_profiles(self):
        self.configure()
        prompt = self.run_calibration()
        self.assertEqual(prompt.call_count, 2)
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

    def test_each_side_is_sampled_at_its_own_confirmation(self):
        self.configure("robot")
        prompts = []

        def place(prompt):
            prompts.append(prompt)
            self.assertFalse(list(self.root.rglob("*.json")))
            left, right = self.serials["/dev/test-left"], self.serials["/dev/test-right"]
            if len(prompts) == 1:
                self.assertIn("LEFT", prompt)
                left.set(1, 56, 2, 1930)
            else:
                self.assertIn("RIGHT", prompt)
                self.assertEqual(left.get(1, 31, 2), 0x800 | 240)
                left.set(1, 56, 2, 3123)  # Moving the completed arm must not resample it.
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
        self.assertEqual(len(prompts), 2)
        self.assertEqual(values["arm_left_shoulder_pan"].homing_offset, -240)
        self.assertEqual(values["arm_right_shoulder_pan"].homing_offset, 230)
        self.assert_passive_and_closed()

    def test_cancel_second_side_restores_the_already_zeroed_first_side(self):
        self.configure("robot")
        with self.assertRaises(InterruptedError):
            self.run_calibration(["", "cancel"])
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
        self.assertEqual(prompt.call_count, 2)
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

    def test_robot_one_pose_preserves_velocity_axes_and_generates_ros_bindings(self):
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
            generated = json.loads(
                (self.root / "hardware" / f"hardware_joint_map_{side}.yaml").read_text()
            )
            self.assertEqual(generated, profile["arms"][side])
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
            if count == 4:
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

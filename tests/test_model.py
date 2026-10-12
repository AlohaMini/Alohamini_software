import unittest
from dataclasses import FrozenInstanceError

from alohamini.model import get_robot_model, robot_models


class ModelTests(unittest.TestCase):
    def test_supported_layouts_and_bus_addresses(self):
        self.assertEqual(robot_models(), ("alohamini1", "alohamini2", "alohamini2pro"))
        for name, count in (("alohamini1", 16), ("alohamini2", 18), ("alohamini2pro", 18)):
            with self.subTest(name=name):
                model = get_robot_model(name)
                self.assertEqual(len(model.actuators), count)
                self.assertEqual(len({a.name for a in model.actuators}), count)
                self.assertEqual(len({(a.bus, a.motor_id) for a in model.actuators}), count)
                self.assertEqual(
                    [(a.name, a.motor_id) for a in model.actuators if a.name.startswith("base_")],
                    [("base_left_wheel", 8), ("base_back_wheel", 9), ("base_right_wheel", 10)],
                )
                self.assertEqual(model.actuators[-1].name, "lift_axis")
                self.assertEqual(model.actuators[-1].motor_id, 11)
                self.assertEqual(model.actuators[-1].bus, "left")

    def test_pro_motors_match_corrected_hardware(self):
        model = get_robot_model("alohamini2pro")
        for motor in model.actuators:
            expected = (
                "sts3095"
                if motor.name == "lift_axis" or motor.name.endswith(("shoulder_lift", "elbow_flex"))
                else "sts3250"
            )
            self.assertEqual(motor.motor_model, expected, motor.name)

    def test_standard_models_are_not_silently_changed_to_pro(self):
        first = get_robot_model("alohamini1")
        self.assertTrue(all(a.motor_model == "sts3215" for a in first.actuators))
        self.assertFalse(any("wrist_yaw" in a.name for a in first.actuators))
        second = get_robot_model("alohamini2")
        self.assertEqual(second.actuators[0].motor_model, "sts3095")
        self.assertEqual(second.actuators[3].motor_model, "sts3215")

    def test_geometry_is_explicitly_in_meters(self):
        for name, radius, base, lead in (
            ("alohamini1", 0.05, 0.125, 0.084),
            ("alohamini2", 0.063, 0.195, 0.131),
            ("alohamini2pro", 0.063, 0.195, 0.131),
        ):
            model = get_robot_model(name)
            self.assertEqual(
                (model.wheel_radius_m, model.base_radius_m, model.lift_lead_m_per_rev),
                (radius, base, lead),
            )

    def test_catalog_cannot_be_changed_through_returned_objects(self):
        model = get_robot_model("alohamini2pro")
        with self.assertRaises(FrozenInstanceError):
            model.model_id = "different"
        with self.assertRaises(FrozenInstanceError):
            model.actuators[0].motor_model = "different"

    def test_unknown_model_has_no_fallback(self):
        for name in ("unknown", "", None, [], True):
            with self.subTest(name=name), self.assertRaises(ValueError):
                get_robot_model(name)


class ArmRomTests(unittest.TestCase):
    """Model-level expected arm travel, used to reject implausible calibration."""

    def test_models_with_geometry_declare_every_arm_joint(self):
        for name in ("alohamini2", "alohamini2pro"):
            with self.subTest(model=name):
                model = get_robot_model(name)
                arms = {m.name for m in model.actuators if m.name.startswith("arm_")}
                self.assertEqual(set(model.arm_rom.joints), arms)

    def test_model_without_geometry_declares_no_rom(self):
        # alohamini1 has no installed-geometry asset, so calibration stays unchecked
        # rather than being checked against an invented range.
        model = get_robot_model("alohamini1")
        self.assertIsNone(model.arm_rom)
        self.assertIsNone(model.rom_band("arm_left_shoulder_pan"))

    def test_rom_records_its_provenance(self):
        rom = get_robot_model("alohamini2").arm_rom
        self.assertEqual(rom.calibrated, "accepted_hardware_joint_map")
        self.assertIn("joint_limits.yaml", rom.source)

    def test_shoulder_lift_band_matches_installed_geometry(self):
        band = get_robot_model("alohamini2").rom_band("arm_left_shoulder_lift")
        # Installed geometry allows ~202 degrees, not the ~359 a full turn implies.
        self.assertAlmostEqual(band.span_tick / 4096 * 360, 201.9, places=1)
        self.assertEqual(band.type, "revolute")
        self.assertFalse(band.continuous)

    def test_continuous_joints_accept_a_full_turn(self):
        for name in ("arm_left_wrist_roll", "arm_right_wrist_roll"):
            with self.subTest(joint=name):
                band = get_robot_model("alohamini2").rom_band(name)
                self.assertTrue(band.continuous)
                self.assertTrue(band.accepts(4095))

    def test_measured_wrap_around_spans_are_rejected(self):
        model = get_robot_model("alohamini2")
        # The 2026-10-12 field failure: every one of these measured a near-full
        # turn because the encoder wrapped during sampling.
        for joint, measured in (
            ("arm_left_shoulder_lift", 4085),
            ("arm_left_elbow_flex", 4095),
            ("arm_left_wrist_flex", 4082),
            ("arm_right_shoulder_lift", 4094),
        ):
            with self.subTest(joint=joint):
                self.assertFalse(model.rom_band(joint).accepts(measured))

    def test_valid_measured_spans_are_accepted(self):
        model = get_robot_model("alohamini2")
        for joint, measured in (
            ("arm_left_shoulder_pan", 2515),
            ("arm_left_gripper", 1092),
            ("arm_left_wrist_yaw", 1892),
            ("arm_right_gripper", 1466),
        ):
            with self.subTest(joint=joint):
                self.assertTrue(model.rom_band(joint).accepts(measured))

    def test_band_rejects_inconsistent_values(self):
        from alohamini.model.types import ArmRomBand

        with self.assertRaises(ValueError):
            ArmRomBand("revolute", 3000, 1500, 2500)  # span outside the band
        with self.assertRaises(ValueError):
            ArmRomBand("revolute", 2000, 2000, 4096)  # max above one revolution
        with self.assertRaises(ValueError):
            ArmRomBand("revolute", 2000, -1, 2500)  # negative lower bound
        with self.assertRaises(ValueError):
            ArmRomBand("spinning", 2000, 1500, 2500)  # unknown joint type


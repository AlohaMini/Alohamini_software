import math
import unittest
from dataclasses import replace
from unittest.mock import patch

from alohamini.calibration.encoder import EncoderCalibration, HostPositionUnits
from alohamini.calibration.servo import MotorCalibration
from alohamini.hardware.feedback import FeedbackBatch, MotorFeedback
from alohamini.hardware.feetech_device import FeetechBusDevice
from alohamini.hardware.motors_bus import MotorsBus
from alohamini.model import ActuatorSpec


class MotorsBusTests(unittest.TestCase):
    def bus(self, normalization="range_m100_100", drive=0):
        installed = MotorCalibration(1, drive, -123, 1000, 3000)
        return MotorsBus(
            position_calibrations={"joint": installed.encoder_calibration()},
            position_units={"joint": installed.position_units(normalization)},
        )

    def test_normalized_saturation_and_direction_preserve_exact_register_endpoints(self):
        for mode in ("range_m100_100", "range_0_100"):
            for drive in (0, 1):
                bus = self.bus(mode, drive)
                for value, expected in ((-1000, 1000), (1000, 3000)):
                    expected = 4000 - expected if drive else expected
                    with self.subTest(mode=mode, drive=drive, value=value):
                        physical = bus.prepare_position_targets({"joint": value})
                        self.assertEqual(bus.encode_positions(physical), {"joint": expected})

    def test_input_truncation_is_preserved_before_physical_rounding(self):
        bus = self.bus()
        # This request is 2000.99 ticks. Encoding an unquantized physical target
        # would round it to 2001, changing the deployed input mapping.
        physical = bus.prepare_position_targets({"joint": 0.099})
        self.assertEqual(bus.encode_positions(physical), {"joint": 2000})

    def test_degree_requests_and_direct_physical_targets_remain_strict(self):
        bus = self.bus("degrees", 1)
        physical = bus.prepare_position_targets({"joint": 12.5})
        self.assertEqual(bus.encode_positions(physical), {"joint": int(12.5 * 4095 / 360 + 2000)})
        with self.assertRaisesRegex(ValueError, "installed encoder range"):
            bus.prepare_position_targets({"joint": 300})
        with self.assertRaisesRegex(ValueError, "calibrated position limits"):
            bus.encode_positions({"joint": math.tau})
        for value in (True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                bus.prepare_position_targets({"joint": value})

    def test_read_normalization_selection_does_not_change_command_units(self):
        bus = self.bus(drive=1)
        self.assertEqual(bus.normalize_positions({"joint": 1000}), {"joint": 100})
        self.assertEqual(
            bus.normalize_positions({"joint": 1000}, normalizations={"joint": "degrees"}),
            {"joint": -1000 * 360 / 4095},
        )
        self.assertEqual(bus.position_units["joint"].normalization, "range_m100_100")
        self.assertEqual(
            bus.encode_positions(bus.prepare_position_targets({"joint": 100})), {"joint": 1000}
        )

    def test_mapping_owns_immutable_configuration_and_rejects_incompatible_ranges(self):
        calibration = MotorCalibration(1, 0, 0, 1000, 3000)
        physical = {"joint": calibration.encoder_calibration()}
        units = {"joint": calibration.position_units()}
        bus = MotorsBus(position_calibrations=physical, position_units=units)
        physical.clear()
        units.clear()
        with self.assertRaises(TypeError):
            bus.position_units["joint"] = HostPositionUnits("degrees", 1000, 3000, 0)
        self.assertEqual(bus.encode_positions(bus.prepare_position_targets({"joint": 0})),
                         {"joint": 2000})
        with self.assertRaisesRegex(ValueError, "calibrated position limits"):
            MotorsBus(
                position_calibrations={"joint": replace(
                    calibration.encoder_calibration(), position_min_rad=-0.1, position_max_rad=0.1
                )},
                position_units={"joint": calibration.position_units()},
            )

    def test_physical_only_bus_does_not_invent_normalized_units(self):
        bus = MotorsBus(position_calibrations={
            "joint": EncoderCalibration(4096, 2048, 0, 1, position_min_rad=-1, position_max_rad=1)
        })
        self.assertEqual(bus.encode_positions({"joint": 0}), {"joint": 2048})
        with self.assertRaisesRegex(ValueError, "missing declared"):
            bus.prepare_position_targets({"joint": 0})

    def test_feedback_missing_or_out_of_turn_stays_absent_and_raw_is_preserved(self):
        bus = self.bus()
        for tick, expected in ((500, {"joint": -100}), (4096, {})):
            batch = FeedbackBatch(
                "arm", "session", 1, 1.0, 1.01,
                {"joint": MotorFeedback({"position_raw": tick}, current_a=0.1)}, {},
            )
            self.assertEqual(bus.feedback_positions(batch), expected)
            self.assertEqual(batch.samples["joint"].registers["position_raw"], tick)
        batch = FeedbackBatch("arm", "session", 1, 1.0, 1.01, {}, {})
        self.assertEqual(bus.feedback_positions(batch), {})

    def test_device_mapping_never_opens_or_writes_hardware(self):
        installed = MotorCalibration(1, 0, 0, 1000, 3000)
        with patch("serial.Serial") as serial:
            device = FeetechBusDevice(
                "/dev/test-only", (ActuatorSpec("joint", "arm", 1, "sts3250"),),
                position_calibrations={"joint": installed.encoder_calibration()},
                position_units={"joint": installed.position_units()}, velocity_limits={},
            )
            with patch.object(device, "_sync_write") as write:
                targets = device.prepare_position_targets({"joint": 101})
                self.assertEqual(device.encode_positions(targets), {"joint": 3000})
                self.assertEqual(device.position_targets_to_units(targets), {"joint": 100})
                serial.assert_not_called()
                write.assert_not_called()

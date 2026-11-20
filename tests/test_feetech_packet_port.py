import unittest
from unittest.mock import patch

from alohamini.hardware.feetech import _PacketPort


class ReconfiguringSerial:
    """A reply arrives during TX; changing timeout reconfigures and clears RX."""

    def __init__(self, timeout=0.005):
        self._write_timeout = timeout
        self.pending = b""
        self.writes = []
        self.timeouts = []

    @property
    def write_timeout(self):
        return self._write_timeout

    @write_timeout.setter
    def write_timeout(self, value):
        self.timeouts.append(value)
        self._write_timeout = value
        self.pending = b""

    def write(self, data):
        self.writes.append((data, self.write_timeout))
        self.pending = b"\xff\xff\x01\x04\x00\x15\x0c\xd9"
        return len(data)

    def read(self, size):
        result, self.pending = self.pending[:size], self.pending[size:]
        return result


class PacketPortTests(unittest.TestCase):
    def test_macos_keeps_reply_without_reconfiguring_write_timeout(self):
        for override in (None, 0.05):
            with self.subTest(override=override):
                serial = ReconfiguringSerial()
                with (
                    patch("alohamini.hardware.feetech._PLATFORM", "darwin"),
                    patch("alohamini.hardware.feetech.time.monotonic", return_value=1.0),
                ):
                    port = _PacketPort(serial, 1.05, write_timeout_s=override)
                    self.assertEqual(port.writePort([1, 2]), 2)
                    self.assertEqual(port.readPort(8), b"\xff\xff\x01\x04\x00\x15\x0c\xd9")
                self.assertEqual(serial.writes, [(b"\x01\x02", 0.005)])
                self.assertEqual(serial.timeouts, [])

    def test_macos_rejects_short_deadline_before_transmitting(self):
        for override in (None, 0.05):
            with self.subTest(override=override):
                serial = ReconfiguringSerial()
                with (
                    patch("alohamini.hardware.feetech._PLATFORM", "darwin"),
                    patch("alohamini.hardware.feetech.time.monotonic", return_value=1.0),
                ):
                    port = _PacketPort(serial, 1.003, write_timeout_s=override)
                    with self.assertRaisesRegex(TimeoutError, "configured serial write timeout"):
                        port.writePort([1])
                self.assertEqual(serial.writes, [])
                self.assertEqual(serial.timeouts, [])

    def test_macos_requires_finite_positive_configured_write_timeout(self):
        for timeout in (None, 0, -1, True, float("nan"), float("inf")):
            with self.subTest(timeout=timeout):
                serial = ReconfiguringSerial(timeout)
                with (
                    patch("alohamini.hardware.feetech._PLATFORM", "darwin"),
                    patch("alohamini.hardware.feetech.time.monotonic", return_value=1.0),
                ):
                    with self.assertRaises((TypeError, ValueError)):
                        _PacketPort(serial, 1.05, write_timeout_s=0.05).writePort([1])
                self.assertEqual(serial.writes, [])
                self.assertEqual(serial.timeouts, [])

    def test_macos_write_failure_does_not_reconfigure_port(self):
        serial = ReconfiguringSerial()
        with (
            patch("alohamini.hardware.feetech._PLATFORM", "darwin"),
            patch("alohamini.hardware.feetech.time.monotonic", return_value=1.0),
            patch.object(serial, "write", side_effect=OSError("disconnected")),
        ):
            with self.assertRaisesRegex(OSError, "disconnected"):
                _PacketPort(serial, 1.05, write_timeout_s=0.05).writePort([1])
        self.assertEqual(serial.timeouts, [])

    def test_other_platforms_keep_bounded_write_timeout_and_restore_it(self):
        for platform in ("linux", "win32"):
            with self.subTest(platform=platform):
                serial = ReconfiguringSerial()
                with (
                    patch("alohamini.hardware.feetech._PLATFORM", platform),
                    patch("alohamini.hardware.feetech.time.monotonic", return_value=1.0),
                ):
                    _PacketPort(serial, 1.03, write_timeout_s=0.05).writePort([1])
                self.assertAlmostEqual(serial.writes[0][1], 0.03)
                self.assertAlmostEqual(serial.timeouts[0], 0.03)
                self.assertEqual(serial.timeouts[1], 0.005)
                self.assertEqual(serial.write_timeout, 0.005)

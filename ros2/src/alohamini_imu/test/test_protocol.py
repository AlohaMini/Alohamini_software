import math
import struct

import pytest
from alohamini_imu.device import read_settings
from alohamini_imu.protocol import CONFIG, FrameParser, ImuDecoder, read_command


# Protocol examples and hand-selected physical values; no implementation encoder
# is used to build the report frames consumed by the decoder.
def report(identifier, values, fmt):
    data = struct.pack(fmt, *values)
    packet = bytes((0x55, 0x55, identifier, len(data))) + data
    return packet + bytes((sum(packet) % 256,))


MOTION = report(3, (8192, -8192, 0, 2880, -2880, 0, 6400), "<7h")
QUATERNION = report(2, (32767, 0, 0, 0), "<4h")


def test_parser_handles_fragmented_and_concatenated_frames():
    parser = FrameParser()
    data = b"noise\x55" + QUATERNION + MOTION + bytes.fromhex("55 af 84 01 01 8a")
    frames = []
    for byte in data:
        frames.extend(parser.feed(bytes((byte,))))
    assert [frame.identifier for frame in frames] == [2, 3, 0x84]
    assert frames[-1].kind == CONFIG
    assert frames[-1].payload == b"\x01"


def test_parser_recovers_from_bad_checksum_and_bad_motion_length():
    parser = FrameParser()
    bad_checksum = MOTION[:-1] + bytes((MOTION[-1] ^ 1,))
    frames = parser.feed(b"\x55\x55\x03\xff" + bad_checksum + MOTION)
    assert len(frames) == 1
    assert frames[0].payload == MOTION[4:-1]
    assert parser.checksum_errors >= 1
    assert parser.length_errors >= 1


def test_default_ranges_give_one_g_and_ninety_degrees_per_second():
    decoder = ImuDecoder(1, 2)
    sample = decoder.feed(QUATERNION + MOTION, 10.0)
    assert sample.acceleration == pytest.approx((9.80665, -9.80665, 0.0))
    assert sample.angular_velocity == pytest.approx((math.pi / 2, -math.pi / 2, 0.0))
    assert sample.temperature == 25.0
    assert sample.quaternion == (1.0, 0.0, 0.0, 0.0)


def test_other_ranges_use_actual_device_scale():
    decoder = ImuDecoder(3, 0)  # 16G, 256 dps
    sample = decoder.feed(report(3, (2048, 0, 0, 0, 0, 11520, -256), "<7h"), 0.0)
    assert sample.acceleration == pytest.approx((9.80665, 0.0, 0.0))
    assert sample.angular_velocity[2] == pytest.approx(math.pi / 2)
    assert sample.temperature == -1.0
    assert sample.quaternion is None


def test_no_new_motion_produces_no_new_sample_and_stale_orientation_is_unavailable():
    decoder = ImuDecoder(1, 2, orientation_timeout=0.2)
    assert decoder.feed(QUATERNION, 1.0) is None
    assert decoder.feed(MOTION, 1.1).quaternion is not None
    assert decoder.feed(b"", 1.15) is None
    assert decoder.feed(MOTION, 1.4).quaternion is None
    assert decoder.feed(report(2, (0, 0, 0, 0), "<4h") + MOTION, 1.5).quaternion is None


def test_backlog_keeps_latest_motion_and_counts_superseded_samples():
    decoder = ImuDecoder(1, 2)
    final = report(3, (0, 0, 8192, 0, 0, 0, 6400), "<7h")
    sample = decoder.feed(MOTION + MOTION + final, 5.0)
    assert sample.acceleration == pytest.approx((0.0, 0.0, 9.80665))
    assert sample.angular_velocity == (0.0, 0.0, 0.0)
    assert decoder.motion_frames == 3
    assert decoder.skipped_motion_frames == 2


def test_read_command_matches_manual_example():
    assert read_command(0x04) == bytes.fromhex("55 af 84 00 88")


class ConfigurationPort:
    """A serial peer that answers only read requests with documented replies."""

    def __init__(self, replies=None):
        self.replies = replies or {
            0x84: bytes.fromhex("55 af 84 01 01 8a"),
            0x83: bytes.fromhex("55 af 83 01 02 8a"),
            0x8A: bytes.fromhex("55 af 8a 01 03 92"),
            0x88: bytes.fromhex("55 af 88 01 1f ac"),
            0x8F: bytes.fromhex("55 af 8f 01 01 95"),
        }
        self.buffer = b""
        self.requests = []

    def write(self, data):
        assert data[:2] == b"\x55\xaf"
        assert data[2] & 0x80 and data[3] == 0  # no writes or calibration
        assert data[-1] == sum(data[:-1]) % 256
        self.requests.append(data)
        self.buffer += MOTION + self.replies.get(data[2], b"")

    @property
    def in_waiting(self):
        return min(len(self.buffer), 3)  # fragmented replies, interleaved reports

    def read(self, count):
        result, self.buffer = self.buffer[:count], self.buffer[count:]
        return result

    def reset_input_buffer(self):
        self.buffer = b""


def test_settings_are_read_not_overwritten_and_streaming_frames_are_tolerated():
    port = ConfigurationPort()
    settings = read_settings(port)
    assert (settings.acc_code, settings.gyro_code, settings.rate) == (1, 2, 100)
    assert settings.report_flags == 0x1F
    assert len(port.requests) == 5


def test_missing_range_reply_fails_instead_of_guessing():
    with pytest.raises(RuntimeError, match="Refusing to guess"):
        read_settings(ConfigurationPort({0x84: b""}), timeout=0.02, attempts=1)


def test_disabled_motion_reports_fail_before_publishing():
    port = ConfigurationPort()
    port.replies[0x88] = bytes.fromhex("55 af 88 01 03 90")
    with pytest.raises(RuntimeError, match="Enable automatic reporting"):
        read_settings(port)

"""Malformed servo traffic must not consume a following complete legal reply."""

import importlib.util
from unittest.mock import patch

import pytest
from test_feedback import MemorySerial, motors, packet

from alohamini.hardware.feetech import FeetechFeedbackReader, _PacketPort

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("scservo_sdk") is None, reason="optional Feetech SDK not installed"
)


class FragmentedSerial(MemorySerial):
    def __init__(self, wire, size):
        super().__init__([wire])
        self.size = size

    def read(self, size):
        return super().read(min(size, self.size))


def read(serial, *, positions=False):
    reader = FeetechFeedbackReader(serial, motors(1, 2), clock_id="offline-test")
    return reader.read_positions() if positions else reader.read()


def test_truncated_reply_does_not_swallow_the_next_complete_motor_packet():
    complete = packet(2)
    serial = MemorySerial([packet(1)[:6] + complete])
    result = read(serial)
    assert set(result.samples) == {"joint_2"}
    assert set(result.failures) == {"joint_1"}
    assert result.samples["joint_2"].registers["position_raw"] == 1234
    assert len(serial.writes) == 1


@pytest.mark.parametrize("length", [0, 1])
def test_impossible_status_length_does_not_abort_before_a_complete_legal_reply(length):
    # These six bytes cannot be a status packet: LENGTH must include ERROR and CHECKSUM.
    invalid = bytes([255, 255, 1, length, 0, 0])
    serial = MemorySerial([invalid + packet(2)])
    result = read(serial)
    assert set(result.samples) == {"joint_2"}
    assert set(result.failures) == {"joint_1"}
    assert result.samples["joint_2"].current_a == pytest.approx(0.117)
    assert len(serial.writes) == 1


@pytest.mark.parametrize("fragment", [1, 2, 7])
def test_complete_replies_split_across_reads_are_not_mistaken_for_broken_packets(fragment):
    serial = FragmentedSerial(b"noise\xff" + packet(2) + packet(1), fragment)
    result = read(serial)
    assert set(result.samples) == {"joint_1", "joint_2"}
    assert not result.failures
    assert len(serial.writes) == 1


def test_checksum_error_does_not_discard_the_complete_reply_which_follows_it():
    bad = bytearray(packet(1))
    bad[-1] ^= 1
    result = read(MemorySerial([bytes(bad) + packet(2)]))
    assert set(result.samples) == {"joint_2"}
    assert set(result.failures) == {"joint_1"}


def test_resynchronized_packet_retains_servo_alarm_instead_of_becoming_healthy():
    result = read(MemorySerial([packet(1)[:6] + packet(2, error=0x01)]))
    assert set(result.samples) == {"joint_2"}
    assert result.samples["joint_2"].packet_error == 0x01
    assert set(result.failures) == {"joint_1", "joint_2"}
    assert "Servo packet error 0x01" in result.failures["joint_2"]


def test_position_only_replies_retain_two_byte_values_under_fragmentation():
    wire = packet(2, data=(3210).to_bytes(2, "little")) + packet(
        1, data=(1234).to_bytes(2, "little")
    )
    result = read(FragmentedSerial(wire, 1), positions=True)
    assert not result.failures
    assert result.samples["joint_1"].registers == {"position_raw": 1234}
    assert result.samples["joint_2"].registers == {"position_raw": 3210}


def test_resynchronizing_continuous_noise_does_not_extend_the_transaction_deadline():
    serial = MemorySerial()
    clock = [1.0]

    def noise(size):
        clock[0] += 0.0001
        return b"\xff" * size

    reader = FeetechFeedbackReader(serial, motors(1, 2), clock_id="offline-test")
    with (
        patch("alohamini.hardware.feetech.time.monotonic", side_effect=lambda: clock[0]),
        patch.object(serial, "read", side_effect=noise),
    ):
        result = reader.read()
    assert not result.samples
    assert set(result.failures) == {"joint_1", "joint_2"}
    assert 0 < result.received_s - result.request_started_s < 0.009
    assert len(serial.writes) == 1


@pytest.mark.parametrize("use_sdk", [False, True], ids=["resynchronizing", "legacy-sdk"])
@pytest.mark.parametrize("error", [0, 0x01])
def test_last_complete_reply_matches_sdk_when_read_returns_after_deadline(use_sdk, error):
    from scservo_sdk.protocol_packet_handler import protocol_packet_handler

    serial = MemorySerial([packet(1, error=error) + packet(2)])
    clock, read_starts = [1.0], []
    read_bytes = serial.read

    def descheduled_read(size):
        read_starts.append(clock[0])
        # The header arrives promptly. While reading its remaining payload the
        # Host loses scheduling time, then receives a complete checksum-valid packet.
        clock[0] += 0.02 if len(read_starts) == 2 else 0.0001
        return read_bytes(size)

    reader = FeetechFeedbackReader(serial, motors(1, 2), clock_id="offline-test")
    receive = (
        (lambda port: protocol_packet_handler().rxPacket(port))
        if use_sdk
        else _PacketPort.read_packet
    )
    with (
        patch("alohamini.hardware.feetech.time.monotonic", side_effect=lambda: clock[0]),
        patch.object(serial, "read", side_effect=descheduled_read),
        patch.object(_PacketPort, "read_packet", receive),
    ):
        result = reader.read()
    assert len(read_starts) == 2
    assert all(start < 1.00852 for start in read_starts)
    assert result.received_s > 1.00852
    assert set(result.samples) == {"joint_1"}
    assert result.samples["joint_1"].registers["position_raw"] == 1234
    assert result.samples["joint_1"].packet_error == error
    assert set(result.failures) == ({"joint_1", "joint_2"} if error else {"joint_2"})
    assert serial.pending == packet(2)
    assert len(serial.writes) == 1


@pytest.mark.parametrize("operation", ["packet", "read", "write", "clear"])
def test_expired_transaction_starts_no_more_serial_io(operation):
    serial = MemorySerial()
    port = _PacketPort(serial, 1.0)
    with (
        patch("alohamini.hardware.feetech.time.monotonic", return_value=1.0),
        patch.object(serial, "read", wraps=serial.read) as read_bytes,
        patch.object(serial, "write", wraps=serial.write) as write_bytes,
        patch.object(serial, "reset_input_buffer", wraps=serial.reset_input_buffer) as clear,
    ):
        with pytest.raises(TimeoutError, match="deadline"):
            if operation == "packet":
                port.read_packet()
            elif operation == "read":
                port.readPort(6)
            elif operation == "write":
                port.writePort(list(packet(1)))
            else:
                port.clearPort()
    read_bytes.assert_not_called()
    write_bytes.assert_not_called()
    clear.assert_not_called()


@pytest.mark.parametrize("malformed", ["partial", "checksum"])
def test_invalid_last_reply_cannot_continue_receiving_after_deadline(malformed):
    wire = packet(1)
    if malformed == "partial":
        wire = wire[:-1]
    else:
        wire = wire[:-1] + bytes([wire[-1] ^ 1]) + packet(2)
    serial = MemorySerial([wire])
    clock, read_starts = [1.0], []
    read_bytes = serial.read

    def descheduled_read(size):
        read_starts.append(clock[0])
        clock[0] += 0.02 if len(read_starts) == 2 else 0.0001
        return read_bytes(size)

    reader = FeetechFeedbackReader(serial, motors(1, 2), clock_id="offline-test")
    with (
        patch("alohamini.hardware.feetech.time.monotonic", side_effect=lambda: clock[0]),
        patch.object(serial, "read", side_effect=descheduled_read),
    ):
        result = reader.read()
    assert len(read_starts) == 2
    assert all(start < 1.00852 for start in read_starts)
    assert result.received_s > 1.00852
    assert not result.samples
    assert set(result.failures) == {"joint_1", "joint_2"}
    assert len(serial.writes) == 1

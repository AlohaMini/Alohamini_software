"""Critical short-read compatibility, with unchanged shared recovery deadlines."""

import importlib.util
from copy import deepcopy
from unittest.mock import patch

import pytest
from test_feetech_device import RegisterSerial, SimulatedClock

from alohamini.calibration import EncoderCalibration
from alohamini.hardware.feetech_device import FeetechBusDevice
from alohamini.model import ActuatorSpec

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("scservo_sdk") is None, reason="optional Feetech SDK not installed"
)


class UsbLatencySerial(RegisterSerial):
    """Synthetic per-request USB latency, not a model of measured hardware speed."""

    def __init__(self):
        super().__init__()
        self.clock = SimulatedClock()
        self.usb_latency = 0.0
        self.scheduled = []

    def reply(self, motor_id, instruction, address, data=b""):
        previous = self.pending
        self.pending = b""
        super().reply(motor_id, instruction, address, data)
        if self.pending:
            delay = (
                self.usb_latency + (8 + 6 + len(data)) * 10 / self.baudrate
                if instruction == 2
                else 0
            )
            self.scheduled.append((self.clock() + delay, self.pending))
        self.pending = previous

    def _deliver(self):
        ready = sorted(item for item in self.scheduled if item[0] <= self.clock() + 1e-12)
        self.scheduled = [item for item in self.scheduled if item[0] > self.clock() + 1e-12]
        self.pending += b"".join(data for _, data in ready)

    def read(self, size):
        self._deliver()
        if not self.pending:
            self.clock.now += 0.0001
            self._deliver()
        return super().read(size)

    def reset_input_buffer(self):
        self._deliver()
        super().reset_input_buffer()


@pytest.fixture
def make_device():
    devices = []

    def create(serial=None):
        serial = RegisterSerial() if serial is None else serial
        serial.registers[3] = deepcopy(serial.registers[2])
        serial.set(3, 46, 2, 1000)
        actuators = [
            ActuatorSpec("joint", "left", 1, "sts3250"),
            ActuatorSpec("wheel", "left", 2, "sts3250"),
            ActuatorSpec("lift_axis", "left", 3, "sts3250"),
        ]
        device = FeetechBusDevice(
            "/dev/in-memory",
            actuators,
            position_calibrations={
                "joint": EncoderCalibration(
                    4096, 2048, 0, 1, position_min_rad=-1, position_max_rad=1
                )
            },
            velocity_limits={"wheel": 3000, "lift_axis": 1300},
        )
        with patch("serial.Serial", return_value=serial):
            device.connect("offline-session")
        devices.append(device)
        serial.requests.clear()
        return device, serial

    yield create
    for device in devices:
        device.close()


def singles(serial):
    return [(request[2], request[5], request[6]) for request in serial.requests if request[4] == 2]


def assert_no_writes(serial):
    assert not any(request[4] in (3, 0x83) for request in serial.requests)


@pytest.mark.parametrize(
    "name,motor_id,required",
    [
        ("joint", 1, {"position_raw": 2500, "current_raw": 0x8001}),
        ("wheel", 2, {"velocity_raw": -5, "current_raw": 0x8001}),
        (
            "lift_axis",
            3,
            {"position_raw": 2500, "velocity_raw": -5, "moving": 1, "current_raw": 0x8001},
        ),
    ],
)
def test_previously_read_full_block_does_not_make_long_replies_required(
    make_device, monkeypatch, name, motor_id, required
):
    device, serial = make_device()
    assert not device.read_feedback().failures
    serial.set(motor_id, 56, 2, 2500)
    serial.set(motor_id, 58, 2, 0x8005)
    serial.set(motor_id, 66, 1, 1)
    serial.set(motor_id, 69, 2, 0x8001)
    serial.drop.add((motor_id, 0x82, 56))
    reply = serial.reply

    def short_only_reply(reply_id, instruction, address, data=b""):
        if (reply_id, instruction, address, len(data)) != (motor_id, 2, 56, 15):
            reply(reply_id, instruction, address, data)

    monkeypatch.setattr(serial, "reply", short_only_reply)
    serial.requests.clear()
    result = device.read_feedback()
    assert not result.failures
    sample = result.samples[name]
    assert sample.registers == required
    assert sample.current_a == pytest.approx(0x8001 * 0.0065)
    assert sample.field_errors["temperature_raw"] == "Full feedback block unavailable"
    assert all(width < 15 for _, _, width in singles(serial))
    assert_no_writes(serial)


@pytest.mark.parametrize("prior_full_success", [False, True])
def test_all_missing_motors_retain_the_required_short_reads(make_device, prior_full_success):
    device, serial = make_device()
    if prior_full_success:
        assert not device.read_feedback().failures
    serial.drop.update((motor_id, 0x82, 56) for motor_id in (1, 2, 3))
    serial.requests.clear()
    result = device.read_feedback()
    assert not result.failures
    assert set(result.samples) == {"joint", "wheel", "lift_axis"}
    assert singles(serial) == [
        (1, 69, 2),
        (1, 56, 2),
        (2, 69, 2),
        (2, 58, 2),
        (3, 69, 2),
        (3, 56, 2),
        (3, 58, 2),
        (3, 66, 1),
    ]
    assert_no_writes(serial)


@pytest.mark.parametrize("failure", ["servo", "length", "checksum", "timeout"])
def test_failed_required_read_does_not_hide_faults_or_reuse_old_feedback(
    make_device, monkeypatch, failure
):
    device, serial = make_device()
    assert not device.read_feedback().failures
    serial.set(1, 56, 2, 2222)
    serial.set(1, 69, 2, 200)
    serial.drop.add((1, 0x82, 56))
    serial.requests.clear()
    if failure == "servo":
        serial.errors[1, 2, 69] = 0x01
    elif failure == "timeout":
        serial.drop.add((1, 2, 69))
    else:
        reply = serial.reply

        def malformed(motor_id, instruction, address, data=b""):
            required = (motor_id, instruction, address) == (1, 2, 69)
            reply(
                motor_id,
                instruction,
                address,
                data[:-1] if required and failure == "length" else data,
            )
            if required and failure == "checksum":
                serial.pending = serial.pending[:-1] + bytes([serial.pending[-1] ^ 1])

        monkeypatch.setattr(serial, "reply", malformed)
    result = device.read_feedback()
    assert "joint" in result.failures
    assert "joint" not in result.samples
    assert set(result.samples) == {"wheel", "lift_axis"}
    # A cyclic fallback has one attempt; failing the first required field must
    # not reuse the prior full sample or continue to read a misleading position.
    assert singles(serial) == [(1, 69, 2)]
    assert "attempts=1" in result.failures["joint"]
    if failure == "servo":
        assert "servo_error=0x01" in result.failures["joint"]
    assert_no_writes(serial)


def test_primary_servo_alarm_is_not_hidden_by_a_successful_short_retry(make_device):
    device, serial = make_device()
    device.read_feedback()
    serial.errors[1, 0x82, 56] = 0x20
    serial.requests.clear()
    result = device.read_feedback()
    assert result.samples["joint"].packet_error == 0x20
    assert "joint" in result.failures
    assert not singles(serial)
    assert_no_writes(serial)


def test_multiple_motors_and_fields_keep_one_shared_absolute_deadline(make_device):
    serial = UsbLatencySerial()
    device, serial = make_device(serial)
    with patch("alohamini.hardware.feetech.time.monotonic", side_effect=serial.clock):
        device.read_feedback()
        serial.drop.update((motor_id, 0x82, 56) for motor_id in (1, 2, 3))
        serial.requests.clear()
        serial.usb_latency = 0.0005
        started, transaction = [], device._register_transaction

        def read_register(*args, **kwargs):
            started.append(serial.clock())
            return transaction(*args, **kwargs)

        with patch.object(device, "_register_transaction", side_effect=read_register) as io:
            result = device.read_feedback()
    assert not result.failures
    assert io.call_count == 8
    deadlines = [call.kwargs["deadline"] for call in io.call_args_list]
    assert len(set(deadlines)) == 1
    # 8 ms + 1 ms per missing motor; never a new budget for each short read.
    assert deadlines[0] - started[0] == pytest.approx(0.011)
    assert result.received_s <= deadlines[0]
    assert_no_writes(serial)


def test_slow_short_reads_do_not_extend_the_shared_budget_or_reuse_cached_fields(make_device):
    serial = UsbLatencySerial()
    device, serial = make_device(serial)
    with patch("alohamini.hardware.feetech.time.monotonic", side_effect=serial.clock):
        device.read_feedback()
        serial.drop.update({(1, 0x82, 56), (2, 0x82, 56)})
        serial.requests.clear()
        serial.usb_latency = 0.003
        started, transaction = [], device._register_transaction

        def read_register(*args, **kwargs):
            started.append(serial.clock())
            return transaction(*args, **kwargs)

        with patch.object(device, "_register_transaction", side_effect=read_register) as io:
            result = device.read_feedback()
    assert set(result.samples) == {"joint", "lift_axis"}
    assert set(result.failures) == {"wheel"}
    assert singles(serial) == [(1, 69, 2), (1, 56, 2), (2, 69, 2), (2, 58, 2)]
    deadlines = [call.kwargs["deadline"] for call in io.call_args_list]
    assert len(set(deadlines)) == 1
    assert deadlines[0] - started[0] == pytest.approx(0.010)
    assert result.received_s - started[0] == pytest.approx(0.010, abs=0.0001)
    assert "attempts=1" in result.failures["wheel"]
    assert_no_writes(serial)

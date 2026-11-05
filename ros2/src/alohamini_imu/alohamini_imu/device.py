"""Read device settings without changing its running or saved configuration."""

import time
from dataclasses import dataclass

from .protocol import CONFIG, RETURN_RATES, FrameParser, read_command


@dataclass(frozen=True)
class Settings:
    acc_code: int
    gyro_code: int
    rate: int
    report_flags: int


def read_settings(port, timeout=0.5, attempts=3):
    parser = FrameParser()

    def read_register(register):
        for _ in range(attempts):
            port.write(read_command(register))
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                for frame in parser.feed(port.read(min(port.in_waiting, 4096))):
                    if frame.kind == CONFIG and frame.identifier == (register | 0x80):
                        if len(frame.payload) != 1:
                            raise RuntimeError(f"Invalid reply for IMU register 0x{register:02x}")
                        return frame.payload[0]
                time.sleep(0.002)
        raise RuntimeError(
            f"No reply for IMU register 0x{register:02x}; check baud rate, "
            "55 55 protocol, power and exclusive serial access. Refusing to guess the range."
        )

    acc_code = read_register(0x04)
    gyro_code = read_register(0x03)
    rate_code = read_register(0x0A)
    flags = read_register(0x08)
    enabled = read_register(0x0F)
    if acc_code not in range(4) or gyro_code not in range(4):
        raise RuntimeError("Device returned an unsupported accelerometer/gyroscope range")
    if rate_code not in range(len(RETURN_RATES)):
        raise RuntimeError("Device returned an unsupported report rate")
    if enabled != 1 or not flags & 0x04:
        raise RuntimeError(
            "Enable automatic reporting and acc/gyro (0x03) in the IMU PC tool first"
        )
    # Configuration transactions may have queued old motion reports.
    port.reset_input_buffer()
    return Settings(acc_code, gyro_code, RETURN_RATES[rate_code], flags)

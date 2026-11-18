"""Hiwonder 55 55 / 55 AF protocol from the supplied ten-axis IMU manual.

The physical conversions follow the supplied imu_usb example. This implements
only acquisition and configuration reads; it never calibrates or saves settings.
"""

import math
import struct
from dataclasses import dataclass

REPORT = 0x55
CONFIG = 0xAF
REPORT_LENGTHS = {1: 6, 2: 8, 3: 14, 4: 6, 5: 10}
ACC_LSB_PER_G = (16384.0, 8192.0, 4096.0, 2048.0)
GYRO_LSB_PER_DPS = (128.0, 64.0, 32.0, 16.0)
RETURN_RATES = (250, 200, 125, 100, 50, 20, 10, 5, 2, 1)


@dataclass(frozen=True)
class Frame:
    kind: int
    identifier: int
    payload: bytes


def read_command(register):
    """Build a read request, including the checksum of BOTH header bytes."""
    if not 0 <= register < 0x80:
        raise ValueError("register must be between 0 and 127")
    data = bytes((0x55, CONFIG, register | 0x80, 0))
    return data + bytes((sum(data) & 0xFF,))


class FrameParser:
    """Incremental, bounded parser that recovers from noise and broken frames."""

    def __init__(self):
        self.buffer = bytearray()
        self.checksum_errors = 0
        self.length_errors = 0

    def feed(self, data):
        self.buffer.extend(data)
        frames = []
        while len(self.buffer) >= 2:
            if self.buffer[0] != 0x55 or self.buffer[1] not in (REPORT, CONFIG):
                del self.buffer[0]
                continue
            if len(self.buffer) < 4:
                break
            kind, identifier, length = self.buffer[1:4]
            expected = REPORT_LENGTHS.get(identifier) if kind == REPORT else None
            if length > 64 or (expected is not None and length != expected):
                self.length_errors += 1
                del self.buffer[0]
                continue
            size = length + 5
            if len(self.buffer) < size:
                break
            if (sum(self.buffer[: size - 1]) & 0xFF) != self.buffer[size - 1]:
                self.checksum_errors += 1
                # Drop one byte, not the claimed frame length: that length may
                # have been corrupted, with a good frame starting inside it.
                del self.buffer[0]
                continue
            frames.append(Frame(kind, identifier, bytes(self.buffer[4 : size - 1])))
            del self.buffer[:size]
        return frames


@dataclass(frozen=True)
class Sample:
    acceleration: tuple
    angular_velocity: tuple
    temperature: float
    quaternion: tuple | None  # w, x, y, z; unavailable when missing or stale


class ImuDecoder:
    """Convert new motion frames to SI using ranges read from the device."""

    def __init__(self, acc_code, gyro_code, orientation_timeout=0.2):
        if acc_code not in range(4) or gyro_code not in range(4):
            raise ValueError("Unsupported IMU range code")
        self.acc_scale = 9.80665 / ACC_LSB_PER_G[acc_code]
        self.gyro_scale = math.pi / 180.0 / GYRO_LSB_PER_DPS[gyro_code]
        self.orientation_timeout = orientation_timeout
        self.parser = FrameParser()
        self.quaternion = None
        self.quaternion_time = None
        self.motion_frames = 0
        self.skipped_motion_frames = 0

    def feed(self, data, received_at):
        """Return only the newest motion sample in a read; never replay caches.

        The wire format has no sample timestamp. Collapse accumulated motion
        frames instead of publishing a backlog as many simultaneous samples.
        """
        sample = None
        for frame in self.parser.feed(data):
            if frame.kind != REPORT:
                continue
            if frame.identifier == 2:
                values = tuple(value / 32768.0 for value in struct.unpack("<4h", frame.payload))
                norm = math.sqrt(sum(value * value for value in values))
                self.quaternion = (
                    tuple(value / norm for value in values) if 0.9 <= norm <= 1.1 else None
                )
                self.quaternion_time = received_at
            elif frame.identifier == 3:
                values = struct.unpack("<7h", frame.payload)
                quaternion = self.quaternion
                if (
                    self.quaternion_time is None
                    or not 0 <= received_at - self.quaternion_time <= self.orientation_timeout
                ):
                    quaternion = None
                if sample is not None:
                    self.skipped_motion_frames += 1
                sample = Sample(
                    tuple(value * self.acc_scale for value in values[:3]),
                    tuple(value * self.gyro_scale for value in values[3:6]),
                    values[6] / 256.0,
                    quaternion,
                )
                self.motion_frames += 1
        return sample

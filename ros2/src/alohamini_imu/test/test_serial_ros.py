"""Exercise the installed ROS launch against a PTY, never a physical IMU."""

import os
import pty
import select
import signal
import struct
import subprocess
import sys
import threading
import time
import tty
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("serial")
from rclpy.context import Context  # noqa: E402 — optional ROS imports follow importorskip.
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import Imu  # noqa: E402


def wire_frame(kind, identifier, payload):
    packet = bytes((0x55, kind, identifier, len(payload))) + payload
    return packet + bytes((sum(packet) % 256,))


class SerialImu:
    def __init__(self, acc_code=1):
        self.master, self.slave = pty.openpty()
        tty.setraw(self.slave)
        self.path = os.ttyname(self.slave)
        self.registers = {0x84: acc_code, 0x83: 2, 0x8A: 3, 0x88: 0x1F, 0x8F: 1}
        self.stop = threading.Event()
        self.pause = threading.Event()
        self.requests = []
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        buffer = bytearray()
        next_report = float("inf")
        try:
            while not self.stop.is_set():
                if select.select([self.master], [], [], 0.002)[0]:
                    buffer.extend(os.read(self.master, 4096))
                    while len(buffer) >= 5:
                        request = bytes(buffer[:5])
                        del buffer[:5]
                        assert request[:2] == b"\x55\xaf"
                        assert request[3] == 0 and request[2] & 0x80
                        assert request[-1] == sum(request[:-1]) % 256
                        self.requests.append(request)
                        reply = wire_frame(0xAF, request[2], bytes((self.registers[request[2]],)))
                        # Split the reply to exercise real serial buffering.
                        os.write(self.master, reply[:3])
                        os.write(self.master, reply[3:])
                        if request[2] == 0x8F:
                            next_report = time.monotonic() + 0.05
                if time.monotonic() >= next_report and not self.pause.is_set():
                    next_report = time.monotonic() + 0.01
                    quaternion = wire_frame(0x55, 2, struct.pack("<4h", 32767, 0, 0, 0))
                    motion = wire_frame(0x55, 3, struct.pack("<7h", 0, 0, 8192, 0, 0, 2880, 6400))
                    os.write(self.master, quaternion + motion)
        except BaseException as exc:
            self.error = exc

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        os.close(self.master)
        os.close(self.slave)
        assert self.error is None, self.error


def terminate(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)


@pytest.mark.parametrize("stop_mode", ["stream_timeout", "interrupt"])
def test_launch_publishes_real_ros_messages_and_stops(monkeypatch, tmp_path, stop_mode):
    # A private DDS domain keeps the test away from running robot topics.
    monkeypatch.setenv("ROS_DOMAIN_ID", str(110 + os.getpid() % 90))
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "ros_logs"))
    source_launch = Path(__file__).resolve().parents[1] / "launch" / "test_imu.launch.py"
    device = SerialImu()
    context = Context()
    rclpy.init(context=context)
    listener = Node("imu_test_listener", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(listener)
    messages = []
    listener.create_subscription(Imu, "/imu/data_raw", messages.append, qos_profile_sensor_data)
    process = subprocess.Popen(
        [
            "ros2",
            "launch",
            str(source_launch),
            f"serial_port:={device.path}",
            "data_timeout_sec:=0.5",
            "report_period_sec:=0.2",
            "startup_timeout_sec:=8.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while len(messages) < 40 and time.monotonic() < deadline and process.poll() is None:
            executor.spin_once(timeout_sec=0.05)
        assert len(messages) >= 40, "Launch did not publish sufficient IMU samples"
        message = messages[-1]
        assert message.header.frame_id == "imu_link"
        assert message.linear_acceleration.z == pytest.approx(9.80665)
        assert message.angular_velocity.z == pytest.approx(1.5707963267948966)
        assert message.orientation.w == pytest.approx(1.0)
        stamps = [msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec for msg in messages]
        assert all(b > a for a, b in zip(stamps, stamps[1:], strict=False))
        if stop_mode == "stream_timeout":
            device.pause.set()
        else:
            os.killpg(process.pid, signal.SIGINT)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired as exc:
            pytest.fail(f"Launch did not stop after data timeout: {exc.output!r}")
        assert "imu ok:" in output
        if stop_mode == "stream_timeout":
            assert "No new valid IMU motion frame" in output or "IMU stream stopped" in output
            assert "exit code 1" in output
        else:
            assert process.returncode == 0
            assert "process has finished cleanly" in output
        assert "Traceback" not in output
        assert len(device.requests) == 5
    finally:
        terminate(process)
        executor.shutdown()
        listener.destroy_node()
        context.try_shutdown()
        device.close()


def test_driver_rejects_unknown_device_range(tmp_path):
    device = SerialImu(acc_code=9)
    environment = dict(
        os.environ,
        ROS_LOG_DIR=str(tmp_path / "logs"),
        ROS_LOCALHOST_ONLY="1",
        ROS_DOMAIN_ID=str(110 + os.getpid() % 90),
    )
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "alohamini_imu.imu_node",
                "--ros-args",
                "-p",
                f"serial_port:={device.path}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env=environment,
        )
        assert result.returncode == 1
        assert "unsupported accelerometer/gyroscope range" in result.stderr + result.stdout
    finally:
        device.close()

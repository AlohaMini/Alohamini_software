"""Publish new Hiwonder motion samples, with explicit startup/stream failures."""

import math
import signal
import sys
import time

import rclpy
import serial
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, Temperature

from .device import read_settings
from .protocol import ImuDecoder


def diagonal(variance):
    return [variance, 0.0, 0.0, 0.0, variance, 0.0, 0.0, 0.0, variance]


class ImuNode(Node):
    def __init__(self):
        super().__init__("imu_node")
        self.port = None
        self.failed = False
        self.declare_parameter("serial_port", "/dev/imu_usb")
        self.declare_parameter("serial_baudrate", 115200)
        self.declare_parameter("frame_id", "imu_link")
        self.declare_parameter("imu_topic", "/imu/data_raw")
        self.declare_parameter("temperature_topic", "/imu/temperature")
        self.declare_parameter("data_timeout_sec", 2.0)
        self.declare_parameter("orientation_timeout_sec", 0.2)
        self.declare_parameter("angular_velocity_variance", 0.01)
        self.declare_parameter("linear_acceleration_variance", 0.04)
        self.declare_parameter("orientation_variance", 0.01)
        self.frame_id = self.get_parameter("frame_id").value
        self.data_timeout = self.positive("data_timeout_sec")
        self.orientation_timeout = self.positive("orientation_timeout_sec")
        self.gyro_cov = diagonal(self.positive("angular_velocity_variance"))
        self.acc_cov = diagonal(self.positive("linear_acceleration_variance"))
        self.orientation_cov = diagonal(self.positive("orientation_variance"))
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        self.publisher = self.create_publisher(
            Imu, self.get_parameter("imu_topic").value, qos_profile_sensor_data
        )
        self.temperature_publisher = self.create_publisher(
            Temperature, self.get_parameter("temperature_topic").value, qos_profile_sensor_data
        )

    def positive(self, name):
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    def start(self):
        path = self.get_parameter("serial_port").value
        baud = self.get_parameter("serial_baudrate").value
        self.port = serial.Serial(path, baudrate=baud, timeout=0, write_timeout=0.5, exclusive=True)
        self.get_logger().info(f"Opened {path} at {baud}; reading actual device configuration")
        settings = read_settings(self.port)
        self.decoder = ImuDecoder(settings.acc_code, settings.gyro_code, self.orientation_timeout)
        self.get_logger().info(
            f"IMU verified: acc={2 << settings.acc_code}G "
            f"gyro={256 << settings.gyro_code}dps report={settings.rate}Hz "
            f"flags=0x{settings.report_flags:02x}; frame={self.frame_id}; no TF broadcast"
        )
        if not settings.report_flags & 0x02:
            self.get_logger().warn(
                "Quaternion reporting is off; orientation will be marked unavailable"
            )
        self.last_motion = time.monotonic()
        self.last_diagnostics = self.last_motion
        self.last_errors = (0, 0, 0)
        self.timer = self.create_timer(0.002, self.poll)

    def fail(self, message):
        self.failed = True
        # Leave the executor before shutting its context down. Shutting down
        # the global executor inside its timer callback can wait on itself.
        raise RuntimeError(message)

    def poll(self):
        current = time.monotonic()
        try:
            waiting = self.port.in_waiting
            if waiting > 16384:
                self.fail("IMU serial backlog exceeds 16 KiB; refusing delayed samples")
                return
            data = self.port.read(waiting)
            sample = self.decoder.feed(data, current)
        except (serial.SerialException, OSError) as exc:
            self.fail(f"IMU serial read failed: {exc}")
            return
        if sample is not None:
            self.last_motion = current
            msg = Imu()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z) = (
                sample.acceleration
            )
            (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z) = (
                sample.angular_velocity
            )
            msg.linear_acceleration_covariance = self.acc_cov
            msg.angular_velocity_covariance = self.gyro_cov
            if sample.quaternion is None:
                msg.orientation_covariance[0] = -1.0
            else:
                (msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z) = (
                    sample.quaternion
                )
                msg.orientation_covariance = self.orientation_cov
            self.publisher.publish(msg)
            temperature = Temperature()
            temperature.header = msg.header
            temperature.temperature = sample.temperature
            self.temperature_publisher.publish(temperature)
        if current - self.last_motion > self.data_timeout:
            self.fail(f"No new valid IMU motion frame for {self.data_timeout:.1f}s; stream stopped")
            return
        if current - self.last_diagnostics >= 5.0:
            errors = (
                self.decoder.parser.checksum_errors,
                self.decoder.parser.length_errors,
                self.decoder.skipped_motion_frames,
            )
            if errors != self.last_errors:
                self.get_logger().warn(
                    f"IMU totals: checksum_errors={errors[0]} length_errors={errors[1]} "
                    f"superseded_motion_frames={errors[2]}"
                )
            self.last_errors = errors
            self.last_diagnostics = current

    def destroy_node(self):
        if self.port is not None:
            self.port.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    result = 0
    try:
        node = ImuNode()
        node.start()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except (RuntimeError, ValueError, serial.SerialException, OSError) as exc:
        rclpy.logging.get_logger("imu_node").error(str(exc))
        result = 1
    finally:
        # A terminal/timeout may signal the process group, then launch signals
        # each child again. Let serial and ROS cleanup finish after the first.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if node is not None:
            result = 1 if node.failed else result
            node.destroy_node()
        rclpy.try_shutdown()
    return result


if __name__ == "__main__":
    sys.exit(main())

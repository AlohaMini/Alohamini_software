"""Check the IMU ROS interface and report live physical measurements."""

import math
import signal
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


def euler_degrees(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


class ImuDataTest(Node):
    def __init__(self):
        super().__init__("imu_data_test")
        defaults = {
            "imu_topic": "/imu/data_raw",
            "frame_id": "imu_link",
            "startup_timeout_sec": 10.0,
            "data_timeout_sec": 2.0,
            "report_period_sec": 1.0,
            "max_message_age_sec": 0.5,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.frame_id = self.get_parameter("frame_id").value
        self.startup_timeout = self.positive("startup_timeout_sec")
        self.data_timeout = self.positive("data_timeout_sec")
        self.report_period = self.positive("report_period_sec")
        self.max_age = self.positive("max_message_age_sec")
        self.started = time.monotonic()
        self.last_report = self.started
        self.last_received = None
        self.last_stamp = None
        self.latest = None
        self.count = 0
        self.rejected = 0
        self.failed = False
        self.subscription = self.create_subscription(
            Imu, self.get_parameter("imu_topic").value, self.on_imu, qos_profile_sensor_data
        )
        self.timer = self.create_timer(min(0.1, self.report_period), self.check)
        self.get_logger().info(f"Monitoring {self.get_parameter('imu_topic').value}")

    def positive(self, name):
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    def on_imu(self, msg):
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        age = (self.get_clock().now().nanoseconds - stamp) / 1e9
        values = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ]
        valid = (
            msg.header.frame_id == self.frame_id
            and stamp > 0
            and -0.1 <= age <= self.max_age
            and (self.last_stamp is None or stamp > self.last_stamp)
            and all(math.isfinite(value) for value in values)
            and msg.angular_velocity_covariance[0] >= 0
            and msg.linear_acceleration_covariance[0] >= 0
        )
        if msg.orientation_covariance[0] != -1:
            q = msg.orientation
            norm = math.sqrt(sum(value * value for value in (q.w, q.x, q.y, q.z)))
            valid = valid and math.isfinite(norm) and abs(norm - 1.0) <= 0.01
        if not valid:
            self.rejected += 1
            if self.rejected == 1 or self.rejected % 100 == 0:
                self.get_logger().warn(
                    f"Rejected IMU message: check frame, finite values, quaternion and "
                    f"ordered/current timestamps (age={age:.3f}s, rejected={self.rejected})"
                )
            return
        self.last_received = time.monotonic()
        self.last_stamp = stamp
        self.latest = msg
        self.count += 1

    def check(self):
        current = time.monotonic()
        failure = None
        if self.last_received is None and current - self.started > self.startup_timeout:
            failure = "No valid IMU message received before startup timeout"
        elif self.last_received is not None and current - self.last_received > self.data_timeout:
            failure = "IMU stream stopped or only invalid/stale messages are arriving"
        if failure:
            self.failed = True
            raise RuntimeError(failure)
        elapsed = current - self.last_report
        if elapsed < self.report_period:
            return
        if self.latest is not None:
            msg = self.latest
            a, g = msg.linear_acceleration, msg.angular_velocity
            norm = math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
            orientation = "unavailable"
            if msg.orientation_covariance[0] != -1:
                orientation = "[{:.1f}, {:.1f}, {:.1f}] deg".format(*euler_degrees(msg.orientation))
            self.get_logger().info(
                f"imu ok: frame={msg.header.frame_id} rate={self.count / elapsed:.1f} Hz "
                f"age={(self.get_clock().now().nanoseconds - self.last_stamp) / 1e6:.1f} ms "
                f"acc=[{a.x:.3f}, {a.y:.3f}, {a.z:.3f}] m/s^2 |a|={norm:.3f} "
                f"gyro=[{g.x:.4f}, {g.y:.4f}, {g.z:.4f}] rad/s "
                f"wz={math.degrees(g.z):.2f} deg/s rpy={orientation} rejected={self.rejected}"
            )
        self.count = 0
        self.last_report = current


def main(args=None):
    rclpy.init(args=args)
    node = None
    result = 0
    try:
        node = ImuDataTest()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except (RuntimeError, ValueError) as exc:
        rclpy.logging.get_logger("imu_data_test").error(str(exc))
        result = 1
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if node is not None:
            result = 1 if node.failed else result
            node.destroy_node()
        rclpy.try_shutdown()
    return result


if __name__ == "__main__":
    sys.exit(main())

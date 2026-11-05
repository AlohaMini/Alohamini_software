"""Mask a scan sector without changing acquisition timestamps or scan geometry."""

import copy
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def mask_scan(message, lower, upper):
    if (
        not math.isfinite(message.angle_min)
        or not math.isfinite(message.angle_increment)
        or message.angle_increment <= 0
    ):
        raise ValueError("Invalid scan angle geometry")
    if message.intensities and len(message.intensities) != len(message.ranges):
        raise ValueError("Scan intensity/range sizes differ")
    result = copy.deepcopy(message)
    for index in range(len(result.ranges)):
        angle = message.angle_min + index * message.angle_increment
        # Keep +/-pi endpoints; modulo changes neither wrapped sector membership.
        angle = math.atan2(math.sin(angle), math.cos(angle))
        keep = lower <= angle <= upper if lower <= upper else angle >= lower or angle <= upper
        if not keep:
            result.ranges[index] = math.inf
            if result.intensities:
                result.intensities[index] = 0.0
    return result


class ScanFilter(Node):
    def __init__(self, **kwargs):
        super().__init__("scan_filter", **kwargs)
        defaults = {
            "input_topic": "/scan",
            "output_topic": "/scan_filtered",
            "min_angle": math.pi / 2,
            "max_angle": -math.pi / 2,
            "max_age_sec": 0.5,
            "max_future_sec": 0.05,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.lower, self.upper, self.max_age, self.max_future = (
            float(self.get_parameter(key).value)
            for key in ("min_angle", "max_angle", "max_age_sec", "max_future_sec")
        )
        if not all(
            math.isfinite(x) for x in (self.lower, self.upper, self.max_age, self.max_future)
        ):
            raise ValueError("Scan filter parameters must be finite")
        if (
            max(abs(self.lower), abs(self.upper)) > math.pi
            or self.max_age <= 0
            or self.max_future < 0
        ):
            raise ValueError("Invalid scan sector/time bounds")
        self.last_stamp = self.last_clock = None
        self.dropped = 0
        source, target = (self.get_parameter(k).value for k in ("input_topic", "output_topic"))
        if source == target:
            raise ValueError("Scan filter input and output must differ")
        self.publisher = self.create_publisher(LaserScan, target, qos_profile_sensor_data)
        self.subscription = self.create_subscription(
            LaserScan, source, self.on_scan, qos_profile_sensor_data
        )

    def on_scan(self, msg):
        now = self.get_clock().now().nanoseconds
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        # Allow a new playback epoch after a ROS clock reset.
        if self.last_clock is not None and now < self.last_clock:
            self.last_stamp = None
        self.last_clock = now
        age = (now - stamp) / 1e9
        if (
            not msg.header.frame_id
            or stamp <= 0
            or not -self.max_future <= age <= self.max_age
            or (self.last_stamp is not None and stamp <= self.last_stamp)
        ):
            self.dropped += 1
            self.get_logger().warning(
                "Rejected stale, future or unordered scan; check sensor clocks",
                throttle_duration_sec=5.0,
            )
            return
        try:
            filtered = mask_scan(msg, self.lower, self.upper)
        except ValueError as exc:
            self.dropped += 1
            self.get_logger().warning(str(exc), throttle_duration_sec=5.0)
            return
        self.last_stamp = stamp
        self.publisher.publish(filtered)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ScanFilter()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()

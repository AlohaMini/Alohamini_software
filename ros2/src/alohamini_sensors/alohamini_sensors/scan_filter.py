"""Mask a scan sector without changing acquisition timestamps or scan geometry."""

import copy
import math

import rclpy
from geometry_msgs.msg import Point
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker


def sector_marker(header, lower, upper, radius=1.0):
    """Show the same wrapped angular sector that mask_scan keeps."""
    marker = Marker()
    marker.header = copy.deepcopy(header)
    marker.ns = "scan_sector"
    marker.id = 0
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = marker.scale.y = marker.scale.z = 1.0
    marker.color.g, marker.color.b, marker.color.a = 0.8, 1.0, 0.18
    end = upper if upper >= lower else upper + 2 * math.pi
    for index in range(48):
        a = lower + (end - lower) * index / 48
        b = lower + (end - lower) * (index + 1) / 48
        marker.points.extend(
            [
                Point(),
                Point(x=radius * math.cos(a), y=radius * math.sin(a)),
                Point(x=radius * math.cos(b), y=radius * math.sin(b)),
            ]
        )
    return marker


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
        self.marker_publisher = self.create_publisher(Marker, "/scan_sector_marker", 1)
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
        self.marker_publisher.publish(sector_marker(msg.header, self.lower, self.upper))


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

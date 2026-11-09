"""Motion gate for SLAM: preserve pure yaw updates and original scan timestamps."""

import math
from collections import deque

import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def moved(previous, pose, distance=0.05, heading=0.05):
    if previous is None:
        return True
    yaw = math.atan2(math.sin(pose[2] - previous[2]), math.cos(pose[2] - previous[2]))
    return (
        math.hypot(pose[0] - previous[0], pose[1] - previous[1]) >= distance or abs(yaw) >= heading
    )


class ScanGate(Node):
    def __init__(self):
        super().__init__("slam_scan_gate")
        self.declare_parameter("distance", 0.05)
        self.declare_parameter("heading", 0.05)
        self.distance, self.heading = (
            float(self.get_parameter(k).value) for k in ("distance", "heading")
        )
        if not all(math.isfinite(v) and v > 0 for v in (self.distance, self.heading)):
            raise ValueError("Motion gate thresholds must be finite and positive")
        self.poses = deque(maxlen=100)
        self.previous = self.last_stamp = None
        self.mapping_ready = False
        self.publisher = self.create_publisher(LaserScan, "/scan_slam", qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self.odom, 10)
        self.create_subscription(LaserScan, "/scan_filtered", self.scan, qos_profile_sensor_data)
        self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_ready,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

    def map_ready(self, msg):
        self.mapping_ready = msg.info.width > 0 and msg.info.height > 0

    def odom(self, msg):
        if msg.child_frame_id != "base_link":
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        pose = (p.x, p.y, yaw)
        if all(math.isfinite(v) for v in pose):
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if self.poses and stamp < self.poses[-1][0]:
                self.poses.clear()
                self.previous = self.last_stamp = None
                self.mapping_ready = False
            self.poses.append((stamp, pose))

    def scan(self, msg):
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        if not self.poses or (self.last_stamp is not None and stamp <= self.last_stamp):
            return
        closest, pose = min(self.poses, key=lambda item: abs(item[0] - stamp))
        if abs(closest - stamp) > 100_000_000:
            return
        # Retry during startup: the first scan can arrive before SLAM or its TF
        # listener is ready. Once a map exists, stationary scans are suppressed.
        if not self.mapping_ready or moved(self.previous, pose, self.distance, self.heading):
            self.publisher.publish(msg)
            self.previous, self.last_stamp = pose, stamp


def main():
    rclpy.init()
    node = ScanGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

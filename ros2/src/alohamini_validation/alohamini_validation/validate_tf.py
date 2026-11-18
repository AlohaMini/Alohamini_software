# SPDX-License-Identifier: Apache-2.0
# Migrated from alohamini_ros2/alohamini_validation.
from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener


def translation(transform) -> tuple[float, float, float]:
    value = transform.transform.translation
    return float(value.x), float(value.y), float(value.z)


def description_frames(description):
    tree = ET.fromstring(description)
    frames = {link.get("name") for link in tree.findall("link")}
    children = {joint.find("child").get("link") for joint in tree.findall("joint")}
    roots = frames - children
    if not frames or None in frames or len(roots) != 1:
        raise ValueError("Robot description must have one named root frame")
    return roots.pop(), frames


def main() -> None:
    rclpy.init()
    node = Node(
        "alohamini_tf_validation",
        namespace="/alohamini_plan_only",
        cli_args=["--ros-args", "-r", "/tf:=tf", "-r", "/tf_static:=tf_static"],
    )
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    default_topic = (
        "robot_description" if node.get_namespace() == "/alohamini_plan_only"
        else "/alohamini/robot_description"
    )
    topic = node.declare_parameter("robot_description_topic", default_topic).value
    descriptions = []
    node.create_subscription(
        String, topic, lambda message: descriptions.append(message.data),
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )
    deadline = time.monotonic() + 8.0
    try:
        while not descriptions and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        assert descriptions, f"robot description unavailable on {topic}"
        root, frames = description_frames(descriptions[-1])
        missing = frames - {root}
        while missing and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            missing = {
                frame
                for frame in missing
                if not buffer.can_transform(
                    root, frame, rclpy.time.Time(), timeout=Duration(seconds=0.0)
                )
            }
        assert not missing, f"missing TF frames: {sorted(missing)}"
        left = translation(buffer.lookup_transform(root, "left_tcp", rclpy.time.Time()))
        right = translation(buffer.lookup_transform(root, "right_tcp", rclpy.time.Time()))
        assert all(math.isfinite(value) for value in (*left, *right))
        # Live poses may be asymmetric, including the model's folded Home.
        for frame in frames - {root}:
            transform = buffer.lookup_transform(root, frame, rclpy.time.Time())
            assert all(math.isfinite(value) for value in translation(transform)), frame
            q = transform.transform.rotation
            assert math.isclose(
                sum(value * value for value in (q.x, q.y, q.z, q.w)), 1.0, abs_tol=1e-6
            ), frame
        print(f"[PASS] TF tree: {len(frames)} frames, {root} -> both TCPs available")
        print(f"       left_tcp={left}, right_tcp={right}")
    finally:
        del listener
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

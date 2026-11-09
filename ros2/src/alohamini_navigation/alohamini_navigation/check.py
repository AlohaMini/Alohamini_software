"""Read-only EDU topic health check; does not enable motion or open serial devices."""

import argparse
import json
import math
import os
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--require-wheel", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be positive and finite")
    import rclpy
    from geometry_msgs.msg import TwistWithCovarianceStamped
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, LaserScan

    rclpy.init()
    node = rclpy.create_node(
        "alohamini_edu_check",
        parameter_overrides=[Parameter("use_sim_time", value=args.simulation)],
    )
    topics = {"/scan": LaserScan, "/imu/data_raw": Imu}
    if args.require_wheel:
        topics["/wheel/twist"] = TwistWithCovarianceStamped
    results = {name: dict(samples=0, valid=0) for name in topics}
    last_received = dict.fromkeys(topics, float("-inf"))

    def received(name, msg):
        item = results[name]
        last_received[name] = time.monotonic()
        item["samples"] += 1
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        age = (node.get_clock().now().nanoseconds - stamp) / 1e9
        item.update(frame=msg.header.frame_id, age_s=age)
        valid = bool(msg.header.frame_id) and -0.05 <= age <= 0.5
        if isinstance(msg, LaserScan):
            item["valid_ranges"] = sum(
                math.isfinite(x) and msg.range_min <= x <= msg.range_max for x in msg.ranges
            )
            valid &= item["valid_ranges"] > 10
        elif isinstance(msg, Imu):
            item["gyro_z_rad_s"] = msg.angular_velocity.z
            item["acceleration_norm_m_s2"] = math.sqrt(
                sum(
                    x * x
                    for x in (
                        msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z,
                    )
                )
            )
            valid &= math.isfinite(msg.angular_velocity.z) and math.isfinite(
                item["acceleration_norm_m_s2"]
            )
        else:
            velocity = msg.twist.twist
            valid &= all(
                math.isfinite(v) for v in (velocity.linear.x, velocity.linear.y, velocity.angular.z)
            )
        item["valid"] += int(valid)

    for name, cls in topics.items():
        node.create_subscription(
            cls, name, lambda msg, name=name: received(name, msg), qos_profile_sensor_data
        )
    deadline = time.monotonic() + args.duration
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    passed = all(
        item["valid"] >= 3
        and item["valid"] == item["samples"]
        and time.monotonic() - last_received[name] <= 0.5
        for name, item in results.items()
    )
    print(
        json.dumps(
            dict(passed=passed, ros_domain_id=os.environ.get("ROS_DOMAIN_ID", "0"), topics=results),
            indent=2,
        )
    )
    return 0 if passed else 1

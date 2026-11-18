"""Keyboard-only base control through /cmd_vel and the existing ROS Bridge."""

import argparse
import math
import select
import sys
import termios
import time
import tty

KEYS = {
    "w": (1, 0, 0),
    "s": (-1, 0, 0),
    "a": (0, 0, 1),
    "d": (0, 0, -1),
    "z": (0, 1, 0),
    "x": (0, -1, 0),
}


def main(argv=None):
    import rclpy
    from geometry_msgs.msg import Twist

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=0.10)
    parser.add_argument("--yaw-speed", type=float, default=0.35)
    args = parser.parse_args(argv)
    if not all(math.isfinite(x) and 0 < x <= 1 for x in (args.speed, args.yaw_speed)):
        parser.error("Speeds must be finite, positive and <= 1")
    if not sys.stdin.isatty():
        parser.error("A terminal is required")
    rclpy.init()
    node = rclpy.create_node("alohamini_base_teleop")
    publisher = node.create_publisher(Twist, "/cmd_vel", 1)
    settings = termios.tcgetattr(sys.stdin)
    print(
        "w/s: forward/back  a/d: yaw  z/x: lateral  space: stop  q: exit\n"
        "Enable the Bridge separately. Movement expires 0.35 s after the last key repeat."
    )
    current = Twist()
    last_key = 0.0
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            if select.select([sys.stdin], [], [], 0.05)[0]:
                key = sys.stdin.read(1)
                if key in ("q", "\x03"):
                    break
                current = Twist()
                if key in KEYS:
                    x, y, yaw = KEYS[key]
                    current.linear.x, current.linear.y = x * args.speed, y * args.speed
                    current.angular.z = yaw * args.yaw_speed
                last_key = time.monotonic()
            if time.monotonic() - last_key > 0.35:
                current = Twist()
            publisher.publish(current)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        if rclpy.ok():
            for _ in range(3):
                publisher.publish(Twist())
                rclpy.spin_once(node, timeout_sec=0.02)
        node.destroy_node()
        rclpy.try_shutdown()

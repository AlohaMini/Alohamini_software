"""Exercise EDU terminal controls, recording and optional RViz without hardware."""

import argparse
import io
import json
import os
import pty
import re
import signal
import subprocess
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from PIL import Image, ImageDraw, ImageGrab
from rclpy.action import ActionClient
from rclpy.clock import Clock, ClockType
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CompressedImage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--navigation-map", required=True)
    parser.add_argument("--screenshot", action="store_true", help="Requires an X display")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["ALOHAMINI_WORKSPACE"] = str(output / "workspace")
    rclpy.init()
    node = rclpy.create_node(
        "edu_session_check", parameter_overrides=[Parameter("use_sim_time", value=True)]
    )
    maps = []
    node.create_subscription(
        OccupancyGrid,
        "/map",
        maps.append,
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )
    initial = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)
    camera = node.create_publisher(
        CompressedImage, "/alohamini/cameras/forward/image_raw/compressed", 1
    )
    picture = Image.new("RGB", (480, 270), (30, 65, 100))
    draw = ImageDraw.Draw(picture)
    draw.rectangle((60, 70, 180, 210), fill=(230, 90, 40))
    draw.rectangle((270, 70, 410, 210), fill=(50, 190, 120))
    draw.text((24, 24), "OFFLINE CAMERA TEST", fill="white")
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG")

    def camera_frame():
        message = CompressedImage()
        message.header.stamp = node.get_clock().now().to_msg()
        message.header.frame_id = "forward_camera_optical"
        message.format, message.data = "jpeg", buffer.getvalue()
        camera.publish(message)

    node.create_timer(0.1, camera_frame, clock=Clock(clock_type=ClockType.STEADY_TIME))
    process, terminal = None, None
    terminal_text = ""
    result = {"result": "incomplete"}

    def spin(predicate, timeout=30):
        nonlocal terminal_text
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.03)
            if terminal is not None:
                try:
                    data = os.read(terminal, 65536).decode(errors="replace")
                    terminal_text += data
                except (BlockingIOError, OSError):
                    pass
            if predicate():
                return True
        return False

    def start(*words):
        nonlocal process, terminal
        terminal, slave = pty.openpty()
        os.set_blocking(terminal, False)
        command = [
            "ros2",
            "run",
            "alohamini_navigation",
            "edu_session",
            "--simulation",
            "--no-camera",
            *words,
        ]
        if not args.screenshot:
            command.append("--no-rviz")
        process = subprocess.Popen(
            command, stdin=slave, stdout=slave, stderr=slave, start_new_session=True
        )
        os.close(slave)
        assert spin(lambda: "Motion starts disabled" in terminal_text), terminal_text

    def key(text):
        os.write(terminal, text.encode())

    def lifecycle(name):
        client = node.create_client(GetState, f"/{name}/get_state")
        try:
            assert client.wait_for_service(timeout_sec=10), name
            end = time.monotonic() + 30
            while time.monotonic() < end:
                future = client.call_async(GetState.Request())
                assert spin(future.done, 5), name
                if future.result().current_state.id == 3:
                    return
                spin(lambda: False, 0.2)
            raise AssertionError(f"{name} did not activate")
        finally:
            node.destroy_client(client)

    def finish(interrupt=False):
        nonlocal process, terminal, terminal_text
        if interrupt:
            os.killpg(process.pid, signal.SIGINT)
        else:
            key("q")
        assert spin(lambda: process.poll() is not None, 40), terminal_text
        assert process.returncode == 0, terminal_text
        with (output / "terminal.log").open("a") as stream:
            stream.write(terminal_text)
        os.close(terminal)
        process, terminal, terminal_text = None, None, ""

    try:
        start("--map-name", "keyboard", "--record", "keyboard")
        assert spin(lambda: bool(maps)), "No mapping data"
        with (output / "sensor-check.json").open("w") as stream:
            health = subprocess.run(
                [
                    "ros2",
                    "run",
                    "alohamini_navigation",
                    "edu_check",
                    "--simulation",
                    "--require-wheel",
                    "--duration",
                    "2",
                ],
                stdout=stream,
                check=False,
            )
        assert health.returncode == 0, (output / "sensor-check.json").read_text()
        key("e")
        assert spin(lambda: "Simulation control enabled" in terminal_text, 8), terminal_text
        for _ in range(15):
            key("a")
            spin(lambda: False, 0.15)
        key("S")
        assert spin(lambda: "Saved. Press e" in terminal_text, 40), terminal_text
        result["keyboard_save"] = True
        finish()
        manifest = json.loads((output / "workspace/runs/keyboard/manifest.json").read_text())
        assert manifest["status"] == "complete", manifest
        assert manifest["message_counts"]["/scan"] > 0
        result["recording_finalized"] = True

        start("--mode", "navigation", "--map", args.navigation_map)
        lifecycle("amcl")
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = node.get_clock().now().to_msg()
        pose.pose.pose.orientation.w = 1.0
        pose.pose.covariance[0] = pose.pose.covariance[7] = pose.pose.covariance[35] = 0.01
        initial.publish(pose)
        lifecycle("bt_navigator")
        key("e")
        assert spin(lambda: "Simulation control enabled" in terminal_text, 8), terminal_text
        action = ActionClient(node, NavigateToPose, "/navigate_to_pose")
        assert action.wait_for_server(timeout_sec=10)
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = node.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.orientation.w = 0.7, 1.0
        future = action.send_goal_async(goal)
        assert spin(future.done, 10) and future.result().accepted
        reached = future.result().get_result_async()
        spin(lambda: False, 6)
        if args.screenshot:
            ImageGrab.grab().save(output / "navigation-rviz.png")
        assert spin(reached.done, 45), "Session blocked Nav2 velocity commands"
        assert reached.result().status == 4, reached.result()
        result["navigation_status"] = reached.result().status
        action.destroy()
        key("0")
        spin(lambda: False, 2)
        finish(interrupt=True)
        if args.screenshot:
            logs = list((output / "workspace/logs/navigation").glob("*/rviz.log"))
            result["rviz_shutdown_clean"] = len(logs) == 2 and all(
                "process has finished cleanly" in path.read_text()
                and not re.search(r"process has died|failed to terminate", path.read_text())
                for path in logs
            )
            assert result["rviz_shutdown_clean"], "RViz shutdown failed; inspect rviz.log"
        result["result"] = "passed"
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            if not spin(lambda: process.poll() is not None, 45):
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if terminal is not None:
            os.close(terminal)
        if terminal_text:
            (output / "failed-terminal.log").write_text(terminal_text)
        node.destroy_node()
        rclpy.try_shutdown()
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Exercise real SLAM, recording, replay and Nav2 against the planar simulator."""

import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_srvs.srv import SetBool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["ALOHAMINI_WORKSPACE"] = str(output / "workspace")
    rclpy.init()
    node = rclpy.create_node(
        "edu_stack_check", parameter_overrides=[Parameter("use_sim_time", value=True)]
    )
    maps, poses = [], []
    node.create_subscription(
        OccupancyGrid,
        "/map",
        maps.append,
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )
    node.create_subscription(Odometry, "/simulation/ground_truth", poses.append, 10)
    velocity = node.create_publisher(Twist, "/cmd_vel", 1)
    initial = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)
    processes, streams, result = [], [], {"result": "incomplete"}

    def spin(predicate, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.03)
            if predicate():
                return True
        return False

    def start(command, name):
        stream = (output / name).open("w")
        streams.append(stream)
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        processes.append(process)
        return process

    def finish(process):
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            if not spin(lambda: process.poll() is not None, 15):
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def cli(words, name, timeout=40):
        process = start(["ros2", "run", "alohamini_navigation", "map_session", *words], name)
        assert spin(lambda: process.poll() is not None, timeout), name
        assert process.returncode == 0, (name, (output / name).read_text()[-3000:])

    def service(kind, name, request):
        client = node.create_client(kind, name)
        try:
            assert client.wait_for_service(timeout_sec=10), name
            future = client.call_async(request)
            assert spin(future.done, 10), name
            return future.result()
        finally:
            node.destroy_client(client)

    try:
        mapping = start(
            ["ros2", "launch", "alohamini_nav_sim", "simulation.launch.py"], "mapping.log"
        )
        assert spin(lambda: bool(maps) and bool(poses), 40), "mapping produced no map/pose"
        assert service(
            SetBool, "/alohamini_nav_sim/command_enable", SetBool.Request(data=True)
        ).success
        recorder = start(
            [
                "ros2",
                "run",
                "alohamini_navigation",
                "map_session",
                "record",
                "mapping",
                "--runtime-mode",
                "simulation",
                "--duration",
                "13",
                "--require-topics",
                "/scan",
                "/imu/data_raw",
                "/wheel/twist",
                "/tf_static",
            ],
            "record.log",
        )
        end = time.monotonic() + 12
        while time.monotonic() < end:
            twist = Twist()
            twist.angular.z = 0.6
            velocity.publish(twist)
            rclpy.spin_once(node, timeout_sec=0.03)
            time.sleep(0.02)
        velocity.publish(Twist())
        assert spin(lambda: recorder.poll() is not None, 20)
        assert recorder.returncode == 0, (output / "record.log").read_text()[-2000:]
        assert maps[-1].info.width >= 140, "Pure rotation did not expand the map"
        cli(["save", "room", "--revision", "v1", "--runtime-mode", "simulation"], "save.log")
        cli(["stop", "--runtime-mode", "simulation"], "mapping-stop.log")
        finish(mapping)
        revision = output / "workspace/maps/room/v1"
        bag = output / "workspace/runs/mapping/bag"
        assert (revision / "slam.posegraph").is_file()
        before = len(maps)
        replay = start(
            [
                "ros2",
                "launch",
                "alohamini_navigation",
                "replay.launch.py",
                f"bag:={bag}",
                "mode:=mapping",
                "use_rviz:=false",
            ],
            "replay.log",
        )
        assert spin(lambda: len(maps) > before, 25), "replay did not produce a fresh map"
        cli(["save", "replayed", "--revision", "v1", "--runtime-mode", "replay"], "replay-save.log")
        cli(["stop", "--runtime-mode", "replay"], "replay-stop.log")
        finish(replay)
        navigation = start(
            [
                "ros2",
                "launch",
                "alohamini_nav_sim",
                "simulation.launch.py",
                "mode:=navigation",
                f"map:={revision}/map.yaml",
            ],
            "navigation.log",
        )
        for name in (
            "amcl",
            "map_server",
            "planner_server",
            "controller_server",
            "bt_navigator",
            "collision_monitor",
        ):
            end = time.monotonic() + 30
            while time.monotonic() < end:
                if (
                    service(GetState, f"/{name}/get_state", GetState.Request()).current_state.id
                    == 3
                ):
                    if name == "amcl":
                        pose = PoseWithCovarianceStamped()
                        pose.header.frame_id = "map"
                        pose.header.stamp = node.get_clock().now().to_msg()
                        pose.pose.pose.orientation.w = 1.0
                        pose.pose.covariance[0] = pose.pose.covariance[7] = pose.pose.covariance[
                            35
                        ] = 0.01
                        initial.publish(pose)
                    break
                spin(lambda: False, 0.2)
            else:
                raise AssertionError(f"{name} did not activate")
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = node.get_clock().now().to_msg()
        pose.pose.pose.orientation.w = 1.0
        pose.pose.covariance[0] = pose.pose.covariance[7] = pose.pose.covariance[35] = 0.01
        initial.publish(pose)
        spin(lambda: False, 2)
        assert service(
            SetBool, "/alohamini_nav_sim/command_enable", SetBool.Request(data=True)
        ).success
        action = ActionClient(node, NavigateToPose, "/navigate_to_pose")
        assert action.wait_for_server(timeout_sec=10)
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = node.get_clock().now().to_msg()
        goal.pose.pose.position.x = 0.7
        goal.pose.pose.orientation.w = 1.0
        sent = action.send_goal_async(goal)
        assert spin(sent.done, 10) and sent.result().accepted
        reached = sent.result().get_result_async()
        assert spin(reached.done, 45), "navigation goal timed out"
        result["navigation_status"] = reached.result().status
        position = poses[-1].pose.pose.position
        result["ground_truth_xy"] = [position.x, position.y]
        assert reached.result().status == 4, result
        assert math.hypot(position.x - 0.7, position.y) < 0.3, result
        action.destroy()
        cli(["stop", "--runtime-mode", "simulation"], "navigation-stop.log")
        finish(navigation)
        result.update(
            result="passed",
            map_messages=len(maps),
            recorded_bag=str(bag),
            maps=[str(revision), str(output / "workspace/maps/replayed/v1")],
        )
    finally:
        for process in reversed(processes):
            finish(process)
        for stream in streams:
            stream.close()
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        node.destroy_node()
        rclpy.try_shutdown()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

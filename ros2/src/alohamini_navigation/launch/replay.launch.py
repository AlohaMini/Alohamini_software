"""Replay selected observations only; never launch hardware or replay motion commands."""

import math
from pathlib import Path

import yaml
from alohamini_navigation.launching import include
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def replay_topics(available, mode, use_imu):
    if mode not in ("view", "mapping"):
        raise ValueError("Replay mode must be view or mapping")
    common = {
        "/scan",
        "/imu/data_raw",
        "/imu/temperature",
        "/wheel/twist",
        "/tf_static",
        "/joint_states",
        "/alohamini/robot_description",
    }
    common.update(t for t in available if t.startswith("/alohamini/cameras/"))
    if mode == "view":
        common.update(
            {
                "/scan_filtered",
                "/scan_sector_marker",
                "/tf",
                "/odom",
                "/map",
                "/plan",
                "/local_plan",
                "/global_costmap/costmap",
                "/local_costmap/costmap",
            }
        )
    else:
        required = {"/scan", "/wheel/twist", "/tf_static"}
        if use_imu:
            required.add("/imu/data_raw")
        missing = required - available
        if missing:
            raise ValueError(f"Mapping replay is missing topics: {sorted(missing)}")
    result = sorted(common & available)
    if not result:
        raise ValueError("Bag contains no supported observation topics")
    return result


def _nodes(context):
    def value(key):
        return LaunchConfiguration(key).perform(context)

    bag = Path(value("bag")).expanduser().resolve()
    metadata = yaml.safe_load((bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    available = {
        item["topic_metadata"]["name"]
        for item in metadata["topics_with_message_count"]
        if item["message_count"] > 0
    }
    mode, rate = value("mode"), float(value("rate"))
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Replay rate must be positive and finite")
    topics = replay_topics(available, mode, value("use_imu") == "true")
    actions = []
    if mode == "mapping":
        actions.append(
            include(
                "alohamini_navigation",
                "mapping.launch.py",
                runtime_mode="replay",
                use_sim_time="true",
                robot_model=value("robot_model"),
                use_imu=value("use_imu"),
                sensor_profile=value("sensor_profile"),
                use_rviz=value("use_rviz"),
                slam_params=value("slam_params"),
            )
        )
    elif value("use_rviz") == "true":
        actions.append(include("alohamini_navigation", "rviz.launch.py", use_sim_time="true"))
    actions.append(
        ExecuteProcess(
            cmd=[
                "ros2",
                "bag",
                "play",
                str(bag),
                "--clock",
                "50",
                "--delay",
                "2",
                "--rate",
                str(rate),
                "--disable-keyboard-controls",
                "--topics",
                *topics,
            ],
            output="screen",
        )
    )
    return actions


def generate_launch_description():
    defaults = dict(
        mode="view",
        rate="1.0",
        robot_model="alohamini2edu",
        use_imu="true",
        use_rviz="true",
        sensor_profile="",
        slam_params="",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("bag"),
            *[DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()],
            OpaqueFunction(function=_nodes),
        ]
    )

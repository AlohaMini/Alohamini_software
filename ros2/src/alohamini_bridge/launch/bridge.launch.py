from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=str(
                    Path(get_package_share_directory("alohamini_bridge")) / "config/bridge.yaml"
                ),
            ),
            DeclareLaunchArgument("host", default_value="127.0.0.1"),
            DeclareLaunchArgument(
                "robot_model",
                default_value="alohamini2pro",
                choices=["alohamini2", "alohamini2pro"],
            ),
            DeclareLaunchArgument("observation_port", default_value="5556"),
            DeclareLaunchArgument("command_port", default_value="5555"),
            DeclareLaunchArgument("state_timestamp_mode", default_value="receipt"),
            DeclareLaunchArgument("max_state_response_age_sec", default_value="0.25"),
            Node(
                package="alohamini_bridge",
                executable="bridge_node",
                name="alohamini_lerobot_bridge",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "host": LaunchConfiguration("host"),
                        "expected_robot_model": LaunchConfiguration("robot_model"),
                        "state_timestamp_mode": LaunchConfiguration("state_timestamp_mode"),
                        "max_state_response_age_sec": ParameterValue(
                            LaunchConfiguration("max_state_response_age_sec"), value_type=float
                        ),
                        "observation_port": ParameterValue(
                            LaunchConfiguration("observation_port"), value_type=int
                        ),
                        "command_port": ParameterValue(
                            LaunchConfiguration("command_port"), value_type=int
                        ),
                    },
                ],
            ),
        ]
    )

"""Wheel/IMU odometry and gated base control, without SLAM or Nav2 planning."""

from alohamini_navigation.launching import runtime
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _nodes(context):
    def value(key):
        return LaunchConfiguration(key).perform(context)

    return runtime(
        value("robot_model"),
        value("host"),
        start_sensors=value("start_sensors").lower() == "true",
        use_imu=value("use_imu").lower() == "true",
        base_only=value("base_only").lower() == "true",
        sensor_profile=value("sensor_profile"),
        arm_mapping_dir=value("arm_mapping_dir"),
        state_timestamp_mode=value("state_timestamp_mode"),
        observation_port=value("observation_port"),
        command_port=value("command_port"),
        collision_params=value("collision_params"),
    )


def generate_launch_description():
    defaults = dict(
        robot_model="alohamini2edu",
        host="127.0.0.1",
        start_sensors="false",
        use_imu="true",
        base_only="true",
        sensor_profile="",
        arm_mapping_dir="",
        state_timestamp_mode="host_wall",
        observation_port="5556",
        command_port="5555",
        collision_params="",
    )
    return LaunchDescription(
        [
            *[DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()],
            OpaqueFunction(function=_nodes),
        ]
    )

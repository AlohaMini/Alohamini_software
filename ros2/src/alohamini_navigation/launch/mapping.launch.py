"""One runtime plus asynchronous slam_toolbox; saving never owns motor hardware."""

from pathlib import Path

from alohamini_navigation.launching import include, model_config
from alohamini_navigation.maps import validate_map_revision
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    def value(key):
        return LaunchConfiguration(key).perform(context)

    params = {"use_sim_time": False}
    graph = value("pose_graph")
    if graph:
        path = Path(graph).expanduser().resolve()
        if not all(Path(str(path) + ext).is_file() for ext in (".posegraph", ".data")):
            raise ValueError("pose_graph must name an existing .posegraph/.data basename")
        validate_map_revision(path.parent / "map.yaml", value("robot_model"))
        params.update(map_file_name=str(path), map_start_at_dock=True)
    return [
        include("alohamini_navigation", "runtime.launch.py"),
        Node(
            package="slam_toolbox",
            executable="async_slam_toolbox_node",
            name="slam_toolbox",
            parameters=[model_config(value("robot_model"), "slam"), params],
            output="screen",
        ),
        Node(
            package="nav2_map_server",
            executable="map_saver_server",
            name="map_saver",
            parameters=[{"save_map_timeout": 10.0, "map_subscribe_transient_local": True}],
            output="screen",
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_map_saver",
            parameters=[{"autostart": True, "node_names": ["map_saver"]}],
            output="screen",
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_model", default_value="alohamini2edu"),
            DeclareLaunchArgument("pose_graph", default_value=""),
            # Expose child arguments to ros2 launch --show-args and CLI overrides.
            DeclareLaunchArgument("host", default_value="127.0.0.1"),
            DeclareLaunchArgument("start_sensors", default_value="false"),
            DeclareLaunchArgument("use_imu", default_value="true"),
            DeclareLaunchArgument("base_only", default_value="true"),
            DeclareLaunchArgument("sensor_profile", default_value=""),
            DeclareLaunchArgument("arm_mapping_dir", default_value=""),
            DeclareLaunchArgument("state_timestamp_mode", default_value="host_wall"),
            DeclareLaunchArgument("observation_port", default_value="5556"),
            DeclareLaunchArgument("command_port", default_value="5555"),
            DeclareLaunchArgument("collision_params", default_value=""),
            OpaqueFunction(function=_nodes),
        ]
    )

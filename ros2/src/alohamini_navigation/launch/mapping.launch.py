"""One runtime plus asynchronous slam_toolbox; saving never owns motor hardware."""

from pathlib import Path

from alohamini_navigation.launching import include, model_config, runtime_defaults
from alohamini_navigation.maps import validate_map_revision
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    def value(key):
        return LaunchConfiguration(key).perform(context)

    params = {"use_sim_time": value("use_sim_time") == "true"}
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
            package="alohamini_navigation",
            executable="scan_gate",
            name="slam_scan_gate",
            parameters=[{"use_sim_time": params["use_sim_time"]}],
            output="screen",
        ),
        Node(
            package="slam_toolbox",
            executable="async_slam_toolbox_node",
            name="slam_toolbox",
            parameters=[value("slam_params") or model_config(value("robot_model"), "slam"), params],
            output="screen",
        ),
        Node(
            package="nav2_map_server",
            executable="map_saver_server",
            name="map_saver",
            parameters=[
                {"save_map_timeout": 10.0, "map_subscribe_transient_local": True},
                {"use_sim_time": params["use_sim_time"]},
            ],
            output="screen",
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_map_saver",
            parameters=[
                {
                    "autostart": True,
                    "node_names": ["map_saver"],
                    "use_sim_time": params["use_sim_time"],
                }
            ],
            output="screen",
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("pose_graph", default_value=""),
            DeclareLaunchArgument("slam_params", default_value=""),
            # Expose child arguments to ros2 launch --show-args and CLI overrides.
            *[DeclareLaunchArgument(k, default_value=v) for k, v in runtime_defaults().items()],
            OpaqueFunction(function=_nodes),
        ]
    )

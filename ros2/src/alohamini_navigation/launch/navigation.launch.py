"""AMCL owns map -> odom; SLAM is deliberately absent from this graph."""

from pathlib import Path

from alohamini_navigation.launching import configured_file, include, model_config, runtime_defaults
from alohamini_navigation.maps import validate_map_revision
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile


def _nodes(context):
    def value(key):
        return LaunchConfiguration(key).perform(context)

    path = Path(value("map")).expanduser().resolve()
    if not path.is_file() or path.suffix not in (".yaml", ".yml"):
        raise ValueError("map must be an existing saved map YAML")
    validate_map_revision(path, value("robot_model"))
    params, cleanup = configured_file(
        value("nav2_params") or model_config(value("robot_model"), "nav2"), "nav2"
    )
    simulation = value("use_sim_time") == "true"
    configured = ParameterFile(params, allow_substs=True)
    components = [
        ("nav2_controller", "controller_server", [("cmd_vel", "cmd_vel_nav")]),
        ("nav2_smoother", "smoother_server", []),
        ("nav2_planner", "planner_server", []),
        ("nav2_behaviors", "behavior_server", [("cmd_vel", "cmd_vel_nav")]),
        ("nav2_bt_navigator", "bt_navigator", []),
        ("nav2_waypoint_follower", "waypoint_follower", []),
        (
            "nav2_velocity_smoother",
            "velocity_smoother",
            [("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")],
        ),
    ]
    return [
        cleanup,
        include("alohamini_navigation", "runtime.launch.py"),
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            parameters=[configured, {"yaml_filename": str(path), "use_sim_time": simulation}],
            output="screen",
        ),
        Node(
            package="nav2_amcl",
            executable="amcl",
            name="amcl",
            parameters=[configured, {"use_sim_time": simulation}],
            output="screen",
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_localization",
            output="screen",
            parameters=[
                {
                    "autostart": True,
                    "node_names": ["map_server", "amcl"],
                    "use_sim_time": simulation,
                }
            ],
        ),
        *[
            Node(
                package=package,
                executable=name,
                name=name,
                output="screen",
                parameters=[configured, {"use_sim_time": simulation}],
                remappings=remappings,
            )
            for package, name, remappings in components
        ],
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            output="screen",
            parameters=[
                {
                    "autostart": True,
                    "node_names": [name for _, name, _ in components],
                    "use_sim_time": simulation,
                }
            ],
        ),
    ]


def generate_launch_description():
    defaults = dict(runtime_defaults(), nav2_params="")
    return LaunchDescription(
        [
            DeclareLaunchArgument("map"),
            *[DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()],
            OpaqueFunction(function=_nodes),
        ]
    )

from pathlib import Path

from alohamini_navigation.launching import include
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    def value(key):
        return LaunchConfiguration(key).perform(context)

    mode = value("mode")
    if mode not in ("mapping", "navigation"):
        raise ValueError("Simulation mode must be mapping or navigation")
    arguments = dict(
        runtime_mode="simulation",
        use_sim_time="true",
        start_sensors="false",
        enable_cameras="false",
        use_rviz=value("use_rviz"),
        robot_model=value("robot_model"),
    )
    if mode == "navigation":
        arguments["map"] = value("map")
    return [
        Node(
            package="alohamini_nav_sim",
            executable="simulator",
            parameters=[
                {
                    "world": value("world"),
                    "robot_model": value("robot_model"),
                    "use_sim_time": False,
                }
            ],
            output="screen",
        ),
        include("alohamini_navigation", mode + ".launch.py", **arguments),
    ]


def generate_launch_description():
    defaults = dict(
        mode="mapping",
        map="",
        robot_model="alohamini2edu",
        use_rviz="false",
        world=str(Path(get_package_share_directory("alohamini_nav_sim")) / "worlds/room.yaml"),
    )
    return LaunchDescription(
        [
            *[DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()],
            OpaqueFunction(function=_nodes),
        ]
    )

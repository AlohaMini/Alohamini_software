from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from alohamini.model import get_robot_model


def robot_description(robot_model="alohamini2pro", description="collision"):
    return get_robot_model(robot_model).description_xml(description)


def _nodes(context):
    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="alohamini_robot_state_publisher",
            parameters=[
                {
                    "robot_description": robot_description(
                        LaunchConfiguration("robot_model").perform(context),
                        LaunchConfiguration("description").perform(context),
                    ),
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                }
            ],
            remappings=[("robot_description", "/alohamini/robot_description")],
            output="screen",
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_model", default_value="alohamini2pro"),
            DeclareLaunchArgument("description", default_value="collision"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            OpaqueFunction(function=_nodes),
        ]
    )

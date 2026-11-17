"""Joy-Con hardware input component using externally owned ROS interfaces."""

from pathlib import Path

from alohamini_joycon_teleop.ik_launch import local_endpoint, start_worker
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node

from alohamini.model import get_robot_model


def _start_reader(context, package):
    python = LaunchConfiguration("native_python").perform(context)
    command = (
        [python]
        if python
        else [
            EnvironmentVariable("CONDA_EXE", default_value="conda"),
            "run",
            "--no-capture-output",
            "-n",
            "alohamini",
            "python",
        ]
    )
    return [
        ExecuteProcess(
            cmd=[
                *command,
                str(package / "scripts" / "joycon_native_reader.py"),
                "--endpoint",
                "tcp://127.0.0.1:5567",
            ],
            output="screen",
        )
    ]


def generate_launch_description() -> LaunchDescription:
    package = Path(get_package_share_directory("alohamini_joycon_teleop"))
    urdf = get_robot_model("alohamini2pro").description_xml("collision")
    ik_endpoint = local_endpoint()
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("start_native_reader", default_value="true"),
            DeclareLaunchArgument(
                "ik_python",
                default_value="",
                description="Pink worker Python; empty uses conda run -n alohamini",
            ),
            DeclareLaunchArgument(
                "native_python",
                default_value="",
                description="Joy-Con reader Python; empty uses conda run -n alohamini",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="joycon_hardware_rviz",
                output="log",
                arguments=["-d", str(package / "config" / "joycon_hardware.rviz")],
                parameters=[{"robot_description": urdf}],
                condition=IfCondition(LaunchConfiguration("use_rviz")),
            ),
            OpaqueFunction(
                function=start_worker,
                args=[ik_endpoint],
            ),
            OpaqueFunction(
                function=_start_reader,
                args=[package],
                condition=IfCondition(LaunchConfiguration("start_native_reader")),
            ),
            Node(
                package="alohamini_joycon_teleop",
                executable="teleop_node",
                name="alohamini_joycon_teleop",
                output="screen",
                parameters=[
                    str(package / "config" / "joycon.yaml"),
                    {
                        "hardware_mode": True,
                        "input_endpoint": "tcp://127.0.0.1:5567",
                        "ik_endpoint": ik_endpoint,
                    },
                ],
            ),
        ]
    )

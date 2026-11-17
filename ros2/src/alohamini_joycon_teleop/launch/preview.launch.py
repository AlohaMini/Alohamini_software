from pathlib import Path

from alohamini_joycon_teleop.ik_launch import local_endpoint, start_worker
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


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
                "tcp://127.0.0.1:5568",
            ],
            output="screen",
        )
    ]


def generate_launch_description() -> LaunchDescription:
    package = Path(get_package_share_directory("alohamini_joycon_teleop"))
    moveit = Path(get_package_share_directory("alohamini_moveit_config"))
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
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(moveit / "launch" / "plan_only.launch.py")),
                launch_arguments={
                    "use_rviz": LaunchConfiguration("use_rviz"),
                    "joycon_preview": "true",
                }.items(),
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
                    {"hardware_mode": False, "ik_endpoint": ik_endpoint},
                ],
            ),
        ]
    )

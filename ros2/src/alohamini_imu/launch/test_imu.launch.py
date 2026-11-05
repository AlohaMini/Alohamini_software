from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("startup_timeout_sec", default_value="10.0"),
            DeclareLaunchArgument("report_period_sec", default_value="1.0"),
            # Stop the entire standalone session if either reader or checker exits.
            RegisterEventHandler(
                OnProcessExit(
                    on_exit=[EmitEvent(event=Shutdown(reason="IMU reader or checker exited"))]
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("alohamini_imu"),
                            "launch",
                            "imu.launch.py",
                        ]
                    )
                )
            ),
            Node(
                package="alohamini_imu",
                executable="imu_data_test",
                name="imu_data_test",
                output="screen",
                parameters=[
                    {
                        "imu_topic": ParameterValue(
                            LaunchConfiguration("imu_topic"), value_type=str
                        ),
                        "frame_id": ParameterValue(LaunchConfiguration("frame_id"), value_type=str),
                        **{
                            name: ParameterValue(LaunchConfiguration(name), value_type=float)
                            for name in (
                                "startup_timeout_sec",
                                "data_timeout_sec",
                                "report_period_sec",
                            )
                        },
                    }
                ],
            ),
        ]
    )

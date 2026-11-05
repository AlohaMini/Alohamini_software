from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    lidar_launch = PathJoinSubstitution(
        [FindPackageShare("alohamini_rplidar_c1"), "launch", "rplidar_c1.launch.py"]
    )
    serial_port = LaunchConfiguration("serial_port")
    scan_topic = LaunchConfiguration("scan_topic")
    startup_timeout_sec = LaunchConfiguration("startup_timeout_sec")
    report_period_sec = LaunchConfiguration("report_period_sec")

    return LaunchDescription(
        [
            DeclareLaunchArgument("serial_port", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            DeclareLaunchArgument("startup_timeout_sec", default_value="10.0"),
            DeclareLaunchArgument("report_period_sec", default_value="1.0"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(lidar_launch),
                launch_arguments={
                    "serial_port": serial_port,
                    "scan_topic": scan_topic,
                }.items(),
            ),
            Node(
                package="alohamini_rplidar_c1",
                executable="rplidar_c1_scan_test",
                name="rplidar_c1_scan_test",
                output="screen",
                parameters=[
                    {
                        "scan_topic": scan_topic,
                        "startup_timeout_sec": startup_timeout_sec,
                        "report_period_sec": report_period_sec,
                    }
                ],
            ),
        ]
    )

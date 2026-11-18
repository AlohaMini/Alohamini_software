from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("serial_port", default_value="/dev/lidar_c1"),
            DeclareLaunchArgument("serial_baudrate", default_value="460800"),
            DeclareLaunchArgument("frame_id", default_value="laser_frame"),
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            DeclareLaunchArgument("inverted", default_value="false"),
            DeclareLaunchArgument("angle_compensate", default_value="true"),
            DeclareLaunchArgument("scan_frequency", default_value="10.0"),
            DeclareLaunchArgument("scan_mode", default_value="Standard"),
            DeclareLaunchArgument("initial_reset", default_value="false"),
            Node(
                package="alohamini_rplidar_c1",
                executable="rplidar_c1_node",
                name="rplidar_c1_node",
                output="screen",
                parameters=[
                    {
                        "serial_port": LaunchConfiguration("serial_port"),
                        "serial_baudrate": LaunchConfiguration("serial_baudrate"),
                        "frame_id": LaunchConfiguration("frame_id"),
                        "scan_topic": LaunchConfiguration("scan_topic"),
                        "inverted": LaunchConfiguration("inverted"),
                        "angle_compensate": LaunchConfiguration("angle_compensate"),
                        "scan_frequency": LaunchConfiguration("scan_frequency"),
                        "scan_mode": LaunchConfiguration("scan_mode"),
                        "initial_reset": LaunchConfiguration("initial_reset"),
                    }
                ],
            ),
        ]
    )

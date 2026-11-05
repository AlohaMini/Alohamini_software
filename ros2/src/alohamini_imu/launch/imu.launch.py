from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    defaults = {
        "serial_port": ("/dev/imu_usb", str),
        "serial_baudrate": ("115200", int),
        "frame_id": ("imu_link", str),
        "imu_topic": ("/imu/data_raw", str),
        "temperature_topic": ("/imu/temperature", str),
        "data_timeout_sec": ("2.0", float),
        "orientation_timeout_sec": ("0.2", float),
        "angular_velocity_variance": ("0.01", float),
        "linear_acceleration_variance": ("0.04", float),
        "orientation_variance": ("0.01", float),
    }
    return LaunchDescription(
        [
            *[
                DeclareLaunchArgument(name, default_value=default)
                for name, (default, _) in defaults.items()
            ],
            Node(
                package="alohamini_imu",
                executable="imu_node",
                name="imu_node",
                output="screen",
                parameters=[
                    {
                        name: ParameterValue(LaunchConfiguration(name), value_type=kind)
                        for name, (_, kind) in defaults.items()
                    }
                ],
            ),
        ]
    )

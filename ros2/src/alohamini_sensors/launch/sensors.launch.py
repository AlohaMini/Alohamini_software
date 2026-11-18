"""Sensor-only graph: no HostClient, motor serial port or motion command publisher."""

from alohamini_sensors.profile import sensor_profile
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    def value(key):
        return LaunchConfiguration(key).perform(context)

    profile = sensor_profile(value("robot_model"), value("sensor_profile"))
    nodes = []
    for kind, package, executable in (
        ("lidar", "alohamini_rplidar_c1", "rplidar_c1_node"),
        ("imu", "alohamini_imu", "imu_node"),
    ):
        if value(f"enable_{kind}").lower() != "true":
            continue
        sensor = profile[kind]
        parameters = {key: sensor[key] for key in ("serial_port", "serial_baudrate", "frame_id")}
        parameters["serial_port"] = value(f"{kind}_port") or sensor["serial_port"]
        if kind == "lidar":
            parameters.update(
                {key: sensor[key] for key in ("scan_mode", "scan_frequency", "inverted")}
            )
        if value("start_drivers").lower() == "true":
            nodes.append(
                Node(
                    package=package,
                    executable=executable,
                    name=executable,
                    parameters=[parameters, {"use_sim_time": value("use_sim_time") == "true"}],
                    output="screen",
                    respawn=True,
                    respawn_delay=2.0,
                )
            )
        if value("publish_mounts").lower() == "true":
            x, y, z = sensor["xyz"]
            roll, pitch, yaw = sensor["rpy"]
            nodes.append(
                Node(
                    package="tf2_ros",
                    executable="static_transform_publisher",
                    name=f"{kind}_mount",
                    arguments=[
                        "--x",
                        str(x),
                        "--y",
                        str(y),
                        "--z",
                        str(z),
                        "--roll",
                        str(roll),
                        "--pitch",
                        str(pitch),
                        "--yaw",
                        str(yaw),
                        "--frame-id",
                        sensor["parent_frame"],
                        "--child-frame-id",
                        sensor["frame_id"],
                    ],
                    parameters=[{"use_sim_time": value("use_sim_time") == "true"}],
                    output="screen",
                )
            )
    if value("enable_lidar").lower() == "true":
        nodes.append(
            Node(
                package="alohamini_sensors",
                executable="scan_filter",
                name="scan_filter",
                parameters=[
                    profile["scan_filter"],
                    {"use_sim_time": value("use_sim_time") == "true"},
                ],
                output="screen",
            )
        )
    return nodes


def generate_launch_description():
    defaults = {
        "robot_model": "alohamini2edu",
        "sensor_profile": "",
        "lidar_port": "",
        "imu_port": "",
        "enable_lidar": "true",
        "enable_imu": "true",
        "publish_mounts": "true",
        "start_drivers": "true",
        "use_sim_time": "false",
    }
    return LaunchDescription(
        [
            *[DeclareLaunchArgument(key, default_value=value) for key, value in defaults.items()],
            OpaqueFunction(function=_nodes),
        ]
    )

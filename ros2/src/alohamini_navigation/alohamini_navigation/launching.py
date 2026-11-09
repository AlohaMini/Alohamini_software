"""Compose ROS navigation around one existing native Host client."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from alohamini.model import get_robot_model


def include(package, filename, **arguments):
    path = Path(get_package_share_directory(package)) / "launch" / filename
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(path)), launch_arguments=arguments.items()
    )


def model_config(model, name):
    return str(get_robot_model(model).asset_path(f"config/navigation/{name}.yaml"))


def ekf_parameters(use_imu, use_sim_time=False):
    # Input order: xyz, rpy, vxyz, vrpy, axyz. Do not fuse an integrated wheel
    # pose a second time or use gravity-containing acceleration as planar motion.
    config = dict(
        use_sim_time=use_sim_time,
        frequency=30.0,
        sensor_timeout=0.2,
        two_d_mode=True,
        transform_time_offset=0.0,
        transform_timeout=0.05,
        print_diagnostics=True,
        publish_tf=True,
        publish_acceleration=False,
        map_frame="map",
        odom_frame="odom",
        base_link_frame="base_link",
        world_frame="odom",
        twist0="/wheel/twist",
        twist0_queue_size=5,
        twist0_config=[False] * 6 + [True, True, False, False, False, not use_imu] + [False] * 3,
    )
    if use_imu:
        config.update(
            imu0="/imu/data_raw",
            imu0_queue_size=10,
            imu0_config=[False] * 11 + [True] + [False] * 3,
            imu0_differential=False,
            imu0_relative=False,
        )
    return config


def runtime_defaults():
    return dict(
        robot_model="alohamini2edu",
        host="127.0.0.1",
        start_sensors="false",
        use_imu="true",
        base_only="true",
        sensor_profile="",
        arm_mapping_dir="",
        state_timestamp_mode="host_wall",
        observation_port="5556",
        command_port="5555",
        collision_params="",
        ekf_params="",
        runtime_mode="hardware",
        use_sim_time="false",
        enable_cameras="false",
        cameras="[auto]",
        camera_stream_port="5557",
        use_rviz="false",
    )


def runtime(
    model,
    host,
    *,
    start_sensors=False,
    use_imu=True,
    sensor_profile="",
    base_only=True,
    arm_mapping_dir="",
    state_timestamp_mode="host_wall",
    observation_port="5556",
    command_port="5555",
    collision_params="",
    ekf_params="",
    runtime_mode="hardware",
    use_sim_time=False,
    enable_cameras=False,
    cameras="[auto]",
    camera_stream_port="5557",
    use_rviz=False,
):
    if runtime_mode not in ("hardware", "simulation", "replay"):
        raise ValueError("runtime_mode must be hardware, simulation or replay")
    if runtime_mode != "hardware" and (not use_sim_time or start_sensors or enable_cameras):
        raise ValueError("Offline modes require sim time and forbid hardware sensors/cameras")
    if runtime_mode == "hardware" and use_sim_time:
        raise ValueError("Hardware mode requires wall time")
    actions = []
    if runtime_mode != "replay":
        actions.append(
            include(
                "alohamini_description",
                "description.launch.py",
                robot_model=model,
                description="base" if base_only else "navigation",
                use_sim_time=str(use_sim_time).lower(),
            )
        )
    if runtime_mode == "hardware":
        actions.append(
            include(
                "alohamini_bridge",
                "bridge.launch.py",
                host=host,
                robot_model=model,
                base_only=str(base_only).lower(),
                arm_mapping_dir=arm_mapping_dir,
                state_timestamp_mode=state_timestamp_mode,
                cmd_vel_topic="/cmd_vel_safe",
                observation_port=observation_port,
                command_port=command_port,
            )
        )
    actions.extend(
        [
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node",
                parameters=[
                    ekf_parameters(use_imu, use_sim_time),
                    *([ekf_params] if ekf_params else []),
                    {"use_sim_time": use_sim_time},
                ],
                remappings=[("odometry/filtered", "/odom")],
                output="screen",
            ),
            Node(
                package="nav2_collision_monitor",
                executable="collision_monitor",
                name="collision_monitor",
                parameters=[
                    collision_params or model_config(model, "collision"),
                    {"use_sim_time": use_sim_time},
                ],
                output="screen",
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_collision_monitor",
                parameters=[
                    {
                        "autostart": True,
                        "node_names": ["collision_monitor"],
                        "use_sim_time": use_sim_time,
                    }
                ],
                output="screen",
            ),
        ]
    )
    if start_sensors or runtime_mode == "simulation":
        actions.append(
            include(
                "alohamini_sensors",
                "sensors.launch.py",
                robot_model=model,
                sensor_profile=sensor_profile,
                enable_imu=str(use_imu).lower(),
                start_drivers=str(runtime_mode == "hardware").lower(),
                use_sim_time=str(use_sim_time).lower(),
            )
        )
    elif runtime_mode == "replay":
        from alohamini_sensors.profile import sensor_profile as load_profile

        actions.append(
            Node(
                package="alohamini_sensors",
                executable="scan_filter",
                parameters=[
                    load_profile(model, sensor_profile)["scan_filter"],
                    {"use_sim_time": True},
                ],
            )
        )
    if enable_cameras:
        actions.append(
            include(
                "alohamini_camera",
                "camera.launch.py",
                host=host,
                port=camera_stream_port,
                cameras=cameras,
                publish_raw="false",
                timestamp_mode="host_wall",
            )
        )
    if use_rviz:
        actions.append(
            include(
                "alohamini_navigation", "rviz.launch.py", use_sim_time=str(use_sim_time).lower()
            )
        )
    return actions

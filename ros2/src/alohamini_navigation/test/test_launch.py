"""Expand actual installed launch components without starting any hardware."""

import importlib.util
from pathlib import Path

import pytest
from alohamini_navigation.launching import ekf_parameters

spec = importlib.util.spec_from_file_location(
    "launch_graph_helpers",
    Path(__file__).parents[2] / "alohamini_bringup/test/test_launch_graph.py",
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
launch_graph = helpers.launch_graph


def test_ekf_uses_independent_sources_without_double_counting_pose():
    params = ekf_parameters(True)
    assert [i for i, x in enumerate(params["twist0_config"]) if x] == [6, 7]
    assert [i for i, x in enumerate(params["imu0_config"]) if x] == [11]
    assert params["base_link_frame"] == "base_link" and params["world_frame"] == "odom"
    no_imu = ekf_parameters(False)
    assert "imu0" not in no_imu
    assert [i for i, x in enumerate(no_imu["twist0_config"]) if x] == [6, 7, 11]


@pytest.mark.parametrize("local_sensors", ["false", "true"])
def test_mapping_has_one_host_client_one_description_and_one_odom_owner(local_sensors):
    nodes, processes = launch_graph(
        "alohamini_navigation", "mapping.launch.py", start_sensors=local_sensors
    )
    packages = [p for p, _, _ in nodes]
    assert not processes
    for name in (
        "alohamini_bridge",
        "robot_state_publisher",
        "robot_localization",
        "slam_toolbox",
        "nav2_collision_monitor",
    ):
        assert packages.count(name) == 1
    assert "nav2_amcl" not in packages and "controller_manager" not in packages
    bridge = next(params for _, exe, params in nodes if exe == "bridge_node")
    assert bridge["expected_robot_model"] == "alohamini2edu"
    assert bridge["base_only"] is True
    assert bridge["cmd_vel_topic"] == "/cmd_vel_safe"
    assert packages.count("alohamini_rplidar_c1") == int(local_sensors == "true")
    assert packages.count("alohamini_imu") == int(local_sensors == "true")


def test_sensor_only_graph_cannot_command_or_open_motor_bus():
    nodes, processes = launch_graph("alohamini_sensors", "sensors.launch.py")
    assert not processes
    assert {p for p, _, _ in nodes} == {
        "alohamini_rplidar_c1",
        "alohamini_imu",
        "alohamini_sensors",
        "tf2_ros",
    }
    assert all("arm_follower" not in str(params) for _, _, params in nodes)


def test_navigation_rejects_missing_map_before_starting_runtime():
    with pytest.raises(ValueError, match="existing saved"):
        launch_graph("alohamini_navigation", "navigation.launch.py", map="/missing/map.yaml")


def test_edu_full_hardware_and_moveit_follow_selected_model():
    nodes, _ = launch_graph(
        "alohamini_bringup",
        "hardware.launch.py",
        robot_model="alohamini2edu",
        enable_cameras="false",
        enable_moveit="true",
        use_rviz="false",
    )
    for package, _, params in nodes:
        if package in ("robot_state_publisher", "moveit_ros_move_group"):
            assert "alohamini2edu_moveit" in params["robot_description"]
        if package == "alohamini_bridge":
            assert params["expected_robot_model"] == "alohamini2edu"

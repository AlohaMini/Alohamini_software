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


def test_simulation_graph_has_no_hardware_clients_and_all_consumers_use_sim_clock():
    nodes, _ = launch_graph(
        "alohamini_navigation", "mapping.launch.py", runtime_mode="simulation", use_sim_time="true"
    )
    assert not (
        {"alohamini_bridge", "alohamini_camera", "alohamini_imu", "alohamini_rplidar_c1"}
        & {p for p, _, _ in nodes}
    )
    assert all(params.get("use_sim_time") is True for _, _, params in nodes)


def test_camera_and_navigation_rviz_are_composed_without_another_bridge():
    nodes, _ = launch_graph(
        "alohamini_navigation", "mapping.launch.py", enable_cameras="true", use_rviz="true"
    )
    packages = [p for p, _, _ in nodes]
    assert packages.count("alohamini_bridge") == 1
    assert packages.count("alohamini_camera") == 1
    assert packages.count("rviz2") == 1
    camera = next(params for package, _, params in nodes if package == "alohamini_camera")
    assert camera["timestamp_mode"] == "host_wall" and not camera["publish_raw"]


@pytest.mark.parametrize("mode", ["simulation", "replay"])
def test_offline_modes_reject_hardware_start(mode):
    with pytest.raises(ValueError, match="forbid hardware"):
        launch_graph(
            "alohamini_navigation",
            "mapping.launch.py",
            runtime_mode=mode,
            use_sim_time="true",
            start_sensors="true",
        )


def test_replay_excludes_motor_commands_and_duplicate_odom_tf(tmp_path):
    import yaml

    bag = tmp_path / "bag"
    bag.mkdir()
    topics = [
        "/scan",
        "/imu/data_raw",
        "/wheel/twist",
        "/tf_static",
        "/tf",
        "/odom",
        "/cmd_vel",
        "/cmd_vel_safe",
    ]
    (bag / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "rosbag2_bagfile_information": {
                    "topics_with_message_count": [
                        {"topic_metadata": {"name": topic}, "message_count": 10} for topic in topics
                    ]
                }
            }
        )
    )
    nodes, processes = launch_graph(
        "alohamini_navigation", "replay.launch.py", bag=str(bag), mode="mapping", use_rviz="false"
    )
    assert not any(package == "alohamini_bridge" for package, _, _ in nodes)
    command = processes[0]
    selected = command[command.index("--topics") + 1 :]
    assert not {"/cmd_vel", "/cmd_vel_safe", "/tf", "/odom"} & set(selected)
    assert "/tf_static" in selected and "/scan" in selected

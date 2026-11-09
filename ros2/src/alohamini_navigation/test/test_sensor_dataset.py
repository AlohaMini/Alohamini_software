import json

import pytest
from alohamini_navigation.recording import recording_topics
from alohamini_navigation.sensor_dataset import align_frame, attach_episode, bag_messages, collect
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String

from alohamini.datasets.ros_sensors import iter_sensor_frames


def native_frame():
    return dict(
        safety={"host_session_id": "host-a"},
        host_timing={"state_sample_monotonic_s": 10.1},
        robot_metadata={"robot_model": "alohamini2edu"},
        episode_index=0,
        frame_index=0,
    )


def messages():
    scan = LaserScan()
    scan.header.stamp.sec = 1000
    scan.header.stamp.nanosec = 50_000_000
    scan.header.frame_id = "laser_frame"
    scan.range_min, scan.range_max, scan.angle_increment = 0.15, 8.0, 0.1
    scan.ranges = [1.0, float("inf"), float("nan")]
    imu = Imu()
    imu.header.stamp.sec, imu.header.stamp.nanosec = 1000, 90_000_000
    imu.header.frame_id = "imu_link"
    imu.angular_velocity.z = 0.2
    imu.orientation_covariance[0] = -1.0
    anchor = String(
        data=json.dumps(
            dict(
                host_session_id="host-a", capture_monotonic_s=10.0, capture_unix_ns=1000_000_000_000
            )
        )
    )
    return [("/scan", scan), ("/imu/data_raw", imu), ("/alohamini/camera_clock", anchor)]


def test_native_extension_matches_session_and_physical_time(tmp_path):
    path = tmp_path / "meta/safety/episode_000000.jsonl"
    path.parent.mkdir(parents=True)
    (tmp_path / "meta/info.json").write_text(json.dumps({"total_episodes": 1}))
    path.write_text(json.dumps(native_frame()) + "\n")
    target = attach_episode(tmp_path / "bag", tmp_path, 0, "alohamini2edu", messages=messages())
    result = list(iter_sensor_frames(tmp_path, 0))[0]
    assert result["sample_unix_ns"] == 1000_100_000_000
    assert result["sensors"]["scan"]["sample"]["ranges"] == [1.0, None, None]
    assert result["sensors"]["imu"]["sample"]["orientation_xyzw"] is None
    assert result["sensors"]["imu"]["sample"]["angular_velocity"][2] == pytest.approx(0.2)
    with pytest.raises(FileExistsError):
        attach_episode(tmp_path / "bag", tmp_path, 0, "alohamini2edu", messages=messages())
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        list(iter_sensor_frames(tmp_path, 0))
    assert json.loads((target / "manifest.json").read_text())["valid_frames"] == {
        "scan": 1,
        "imu": 1,
    }


def test_alignment_rejects_other_host_epoch_and_clock_steps():
    samples = {"scan": [], "imu": []}
    with pytest.raises(ValueError, match="No clock"):
        align_frame(native_frame(), {"host-b": [(10.0, 1000_000_000_000)]}, samples)
    with pytest.raises(ValueError, match="jumped"):
        align_frame(
            native_frame(),
            {"host-a": [(10.0, 1000_000_000_000), (10.2, 1000_300_000_000)]},
            samples,
        )


def test_no_future_or_stale_sample_is_fabricated():
    row = align_frame(
        native_frame(),
        {"host-a": [(10.0, 1000_000_000_000)]},
        {"scan": [{"stamp_ns": 999_000_000_000}], "imu": [{"stamp_ns": 1000_101_000_000}]},
    )
    assert all(not item["valid"] and item["sample"] is None for item in row["sensors"].values())


def test_camera_recording_includes_capture_clock_without_raw_image_duplication():
    topics = recording_topics(["forward"])
    assert "/alohamini/cameras/forward/image_raw/compressed" in topics
    assert "/alohamini/camera_clock" in topics
    assert "/alohamini/cameras/forward/image_raw" not in topics


def test_real_rosbag_reader_preserves_sensor_values_and_host_clock(tmp_path):
    import rosbag2_py
    from rclpy.serialization import serialize_message

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(tmp_path / "bag"), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for index, (topic, message) in enumerate(messages()):
        kind = type(message).__name__
        package = "std_msgs" if kind == "String" else "sensor_msgs"
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=topic, type=f"{package}/msg/{kind}", serialization_format="cdr"
            )
        )
        writer.write(topic, serialize_message(message), 1000_100_000_000 + index)
    del writer
    anchors, samples = collect(bag_messages(tmp_path / "bag"))
    row = align_frame(native_frame(), anchors, samples)
    assert row["sensors"]["scan"]["sample"]["ranges"] == [1.0, None, None]
    assert row["sensors"]["imu"]["valid"]


def test_unfinished_native_episode_is_rejected(tmp_path):
    path = tmp_path / "meta/safety/episode_000000.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(native_frame()) + "\n")
    (tmp_path / "meta/info.json").write_text(json.dumps({"total_episodes": 0}))
    with pytest.raises(ValueError, match="not been committed"):
        attach_episode(tmp_path / "bag", tmp_path, 0, "alohamini2edu", messages=messages())

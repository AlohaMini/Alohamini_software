import math
from unittest.mock import Mock

import pytest
import rclpy
from alohamini_sensors.scan_filter import ScanFilter, mask_scan
from sensor_msgs.msg import LaserScan


def scan(stamp=100.0):
    msg = LaserScan()
    msg.header.frame_id = "laser_frame"
    msg.header.stamp.sec = int(stamp)
    msg.header.stamp.nanosec = round((stamp % 1) * 1e9)
    msg.angle_min = -math.pi
    msg.angle_max = math.pi
    msg.angle_increment = math.pi / 2
    msg.scan_time = 0.1
    msg.time_increment = 0.025
    msg.range_min, msg.range_max = 0.15, 8.0
    msg.ranges = [1.0] * 5
    msg.intensities = [10.0] * 5
    return msg


def test_mask_preserves_acquisition_time_geometry_and_input():
    original = scan()
    result = mask_scan(original, math.pi / 2 - 1e-6, -math.pi / 2 + 1e-6)
    assert result.header == original.header
    assert (
        result.scan_time == original.scan_time and result.time_increment == original.time_increment
    )
    assert math.isinf(result.ranges[2]) and result.intensities[2] == 0
    assert result.ranges[0] == result.ranges[4] == 1
    assert list(original.ranges) == [1.0] * 5


def test_stale_future_and_repeated_scans_are_not_restamped(monkeypatch):
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("ROS_DOMAIN_ID", "84")
    rclpy.init()
    node = ScanFilter()
    try:
        node.publisher = Mock()
        node.get_clock = lambda: Mock(now=lambda: Mock(nanoseconds=100_400_000_000))
        node.on_scan(scan(100.0))
        node.publisher.publish.assert_called_once()
        assert node.publisher.publish.call_args.args[0].header.stamp.sec == 100
        assert node.publisher.publish.call_args.args[0].header.stamp.nanosec == 0
        for msg in (scan(99.0), scan(101.0), scan(100.0)):
            node.on_scan(msg)
        assert node.dropped == 3
        assert node.publisher.publish.call_count == 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_invalid_scan_layout_rejected():
    msg = scan()
    msg.intensities = [1.0]
    with pytest.raises(ValueError, match="sizes"):
        mask_scan(msg, -1, 1)

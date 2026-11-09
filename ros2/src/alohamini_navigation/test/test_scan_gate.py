import math

from alohamini_navigation.scan_gate import ScanGate, moved


def test_initial_scan_is_retried_until_slam_acknowledges_a_map():
    from collections import deque
    from types import SimpleNamespace
    from unittest.mock import Mock

    from sensor_msgs.msg import LaserScan

    gate = SimpleNamespace(
        poses=deque([(1_000_000_000, (0.0, 0.0, 0.0))]),
        previous=None,
        last_stamp=None,
        mapping_ready=False,
        publisher=Mock(),
        distance=0.05,
        heading=0.05,
    )
    scan = LaserScan()
    scan.header.stamp.sec = 1
    ScanGate.scan(gate, scan)
    scan.header.stamp.nanosec = 20_000_000
    ScanGate.scan(gate, scan)
    assert gate.publisher.publish.call_count == 2
    gate.mapping_ready = True
    scan.header.stamp.nanosec = 40_000_000
    ScanGate.scan(gate, scan)
    assert gate.publisher.publish.call_count == 2


def test_motion_gate_keeps_pure_rotation_without_stationary_duplicate_scans():
    assert moved(None, (0, 0, 0))
    assert moved((0, 0, 0), (0, 0, 0.06))
    assert moved((0, 0, 0), (0.06, 0, 0))
    assert not moved((0, 0, 0), (0.01, 0.01, 0.01))
    assert not moved((0, 0, math.pi - 0.01), (0, 0, -math.pi + 0.01))

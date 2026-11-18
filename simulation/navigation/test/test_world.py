import math

import pytest
from alohamini_nav_sim.world import World


def world():
    return World(
        dict(
            bounds=[-4.0, -3.0, 4.0, 3.0],
            obstacles=[[1.0, -0.4, 1.5, 0.4]],
            initial_pose=[0.0, 0.0, 0.0],
            robot_radius=0.3,
        )
    )


def test_ray_intersects_obstacle_and_cable_rear_scan_frame():
    sim = world()
    front = sim.scan([0.2, 0.0, 0.12], [0.0, 0.0, math.pi], count=5)
    assert front[0] == pytest.approx(0.8)
    assert front[2] == pytest.approx(4.2)


def test_measured_motion_stops_at_footprint_and_rotates_in_body_frame():
    sim = world()
    for _ in range(200):
        velocity = sim.step(0.2, 0.0, 0.0, 0.02)
    assert sim.x < 0.7 and velocity == (0.0, 0.0, 0.0)
    sim = world()
    sim.yaw = math.pi / 2
    sim.step(0.1, 0.0, 0.0, 0.1)
    assert sim.y == pytest.approx(0.01)


def test_rejects_overlapping_start_or_nonfinite_command():
    with pytest.raises(ValueError):
        World(dict(bounds=[-1, -1, 1, 1], initial_pose=[0.9, 0, 0], robot_radius=0.3))
    with pytest.raises(ValueError):
        world().step(float("nan"), 0, 0, 0.02)

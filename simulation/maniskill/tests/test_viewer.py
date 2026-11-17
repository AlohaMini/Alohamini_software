from types import SimpleNamespace

import pytest
from alohamini_maniskill import viewer


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def perf_counter(self):
        return self.now

    def sleep(self, duration):
        assert duration > 0
        self.sleeps.append(duration)
        self.now += duration


def test_render_refreshes_between_control_steps_and_physics_keeps_real_time(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(viewer, "time", clock)
    frames = []

    def render():
        frames.append(clock.now)
        clock.now += 0.003
        return SimpleNamespace(closed=False)

    schedule = viewer.ViewerClock(render, timestep=0.001, refresh_hz=60, control_period=0.05)
    for _ in range(500):
        clock.now += 0.0002
        schedule.step()
    assert 28 <= len(frames) <= 31
    assert clock.now == pytest.approx(0.5, abs=0.004)
    assert sum(t < 0.05 for t in frames) >= 3


def test_slow_physics_retains_responsive_rendering_without_extra_steps(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(viewer, "time", clock)
    frames = []

    def render():
        frames.append(clock.now)
        return SimpleNamespace(closed=False)

    schedule = viewer.ViewerClock(render, timestep=0.001, refresh_hz=60, control_period=0.05)
    for _ in range(100):
        clock.now += 0.003
        schedule.step()
    assert clock.now == pytest.approx(0.3)
    assert 16 <= len(frames) <= 18
    assert not clock.sleeps


def test_pause_resumes_wall_clock_and_close_stops_scheduling(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(viewer, "time", clock)
    native = SimpleNamespace(closed=False)
    blocked = []

    def render():
        if not blocked:
            clock.now += 2
            blocked.append(True)
        return native

    schedule = viewer.ViewerClock(render, timestep=0.001, refresh_hz=60, control_period=0.05)
    schedule.step()
    assert clock.now == 2
    schedule.step()
    assert clock.now == pytest.approx(2.001)
    native.closed = True
    clock.now += 0.02
    schedule.step()
    assert schedule.closed
    before = clock.now
    schedule.step()
    assert clock.now == before


def test_render_overrun_does_not_starve_physics(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(viewer, "time", clock)
    frames = []

    def render():
        frames.append(clock.now)
        clock.now += 0.04
        return SimpleNamespace(closed=False)

    schedule = viewer.ViewerClock(render, timestep=0.001, refresh_hz=60, control_period=0.05)
    for _ in range(500):
        clock.now += 0.0002
        schedule.step()
    assert 5 <= len(frames) <= 12
    assert clock.now < 0.7


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_refresh_rate_is_rejected(value):
    with pytest.raises(ValueError):
        viewer.ViewerClock(lambda: None, timestep=0.001, refresh_hz=value, control_period=0.05)

from contextlib import nullcontext

import mujoco
import numpy as np
import pytest
from alohamini_mujoco import viewer as inspection
from alohamini_mujoco.control import DynamicFollower, JointTrajectory


def test_paused_single_step_reset_and_playback_match_offline_dynamics(robot):
    trajectory = JointTrajectory(robot, ["right_shoulder_pan"], [0, 0.2], [[0], [0.01]])
    offline = DynamicFollower(robot)
    offline.follow(trajectory)
    playback = inspection.DynamicPlayback(robot, trajectory, paused=True)
    initial = playback.data.qpos.copy()
    for _ in range(3):
        assert not playback.step()
    assert playback.data.time == 0
    np.testing.assert_array_equal(playback.data.qpos, initial)
    assert playback.step(single=True)
    assert playback.data.time == pytest.approx(robot.model.opt.timestep)
    assert playback.paused
    playback.paused = False
    while not playback.completed:
        assert playback.step()
    assert playback.paused
    assert not playback.step(single=True)
    assert playback.data.time == pytest.approx(offline.data.time)
    np.testing.assert_allclose(playback.data.qpos, offline.data.qpos, atol=1e-12)
    np.testing.assert_allclose(playback.data.qvel, offline.data.qvel, atol=1e-12)
    playback.reset()
    np.testing.assert_array_equal(playback.data.qpos, initial)
    np.testing.assert_array_equal(playback.data.qvel, 0)
    assert playback.data.time == 0
    assert playback.paused and not playback.completed


def test_playback_rejects_a_discontinuous_trajectory_start(robot):
    trajectory = JointTrajectory(robot, ["right_shoulder_pan"], [0, 1], [[0.1], [0.2]])
    with pytest.raises(ValueError, match="measured joint positions"):
        inspection.DynamicPlayback(robot, trajectory)


def test_window_edits_cannot_change_physics_and_controls_use_simulation_time(robot, monkeypatch):
    from mujoco import viewer

    snapshots = []
    gravity = robot.model.opt.gravity.copy()
    timestep = robot.model.opt.timestep

    class Window:
        def __init__(self, model, data, key_callback, **kwargs):
            self.model, self.data, self.callback = model, data, key_callback
            self.opt, self.cam = mujoco.MjvOption(), mujoco.MjvCamera()
            self.text = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def lock(self):
            return nullcontext()

        def is_running(self):
            return len(snapshots) < 4

        def set_texts(self, texts):
            self.text = texts

        def sync(self, **kwargs):
            snapshots.append(
                (self.data.time, self.data.qpos.copy(), self.opt.geomgroup.copy(), self.text)
            )
            # Simulate native GUI edits: neither positions nor physical options
            # may leak back into the controller's model and live state.
            self.model.opt.gravity[:] = 0
            self.model.opt.timestep = 0.2
            self.data.qpos[:] = 100
            if len(snapshots) == 1:
                self.callback(297)  # F8
                self.callback(262)  # Right: one physics step while paused
            elif len(snapshots) == 2:
                self.callback(32)  # Space: resume
            elif len(snapshots) == 3:
                self.callback(32)
                self.callback(298)  # F9: reset and pause

    monkeypatch.setattr(inspection, "check_display", lambda: None)
    monkeypatch.setattr(viewer, "launch_passive", Window)
    playback = inspection.run_viewer(robot, paused=True)
    assert snapshots[0][0] == 0
    assert snapshots[1][0] == pytest.approx(timestep)
    assert snapshots[2][0] >= snapshots[1][0]
    assert snapshots[3][0] == 0
    assert snapshots[0][2][2] == 1 and snapshots[0][2][3] == 0
    assert snapshots[1][2][2] == 0 and snapshots[1][2][3] == 1
    for _, positions, _, texts in snapshots:
        np.testing.assert_allclose(positions, snapshots[0][1], atol=1e-4)
        assert "Time / speed" in texts[2]
    np.testing.assert_array_equal(robot.model.opt.gravity, gravity)
    assert robot.model.opt.timestep == timestep
    assert playback.paused and playback.data.time == 0


def test_missing_display_fails_before_launching_a_render_thread(monkeypatch):
    import glfw

    monkeypatch.setattr(glfw, "init", lambda: False)
    with pytest.raises(RuntimeError, match="desktop session"):
        inspection.check_display()

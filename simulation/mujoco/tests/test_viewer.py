from contextlib import contextmanager, nullcontext

import mujoco
import numpy as np
import pytest
from alohamini_mujoco import viewer as inspection
from alohamini_mujoco.control import DynamicFollower, FollowerSettings, JointTrajectory


@pytest.mark.parametrize("override_gains", [False, True])
def test_paused_single_step_reset_and_playback_match_offline_dynamics(robot, override_gains):
    trajectory = JointTrajectory(robot, ["right_shoulder_pan"], [0, 0.2], [[0], [0.01]])
    settings = FollowerSettings(
        joint_gains={"right_shoulder_pan": {"kp": 4, "kd": 0.5}} if override_gains else {}
    )
    offline = DynamicFollower(robot, settings=settings)
    offline.follow(trajectory)
    playback = inspection.DynamicPlayback(
        robot, trajectory, paused=True, follower_settings=settings
    )
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
    np.testing.assert_array_equal(playback.follower.kp, offline.kp)
    np.testing.assert_array_equal(playback.follower.kd, offline.kd)


def test_playback_rejects_a_discontinuous_trajectory_start(robot):
    trajectory = JointTrajectory(robot, ["right_shoulder_pan"], [0, 1], [[0.1], [0.2]])
    with pytest.raises(ValueError, match="measured joint positions"):
        inspection.DynamicPlayback(robot, trajectory)


def test_window_edits_cannot_change_physics_and_controls_use_simulation_time(robot, monkeypatch):
    snapshots = []
    gravity = robot.model.opt.gravity.copy()
    timestep = robot.model.opt.timestep

    class Window:
        def __init__(self, model, data, key_callback, **kwargs):
            self.model, self.data, self.callback = model, data, key_callback
            self.opt, self.cam = mujoco.MjvOption(), mujoco.MjvCamera()
            self.perturb = mujoco.MjvPerturb()
            self.text = None
            self.lock_held = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        @contextmanager
        def lock(self):
            assert not self.lock_held
            self.lock_held = True
            try:
                yield
            finally:
                self.lock_held = False

        def is_running(self):
            return len(snapshots) < 4

        def set_texts(self, texts):
            assert not self.lock_held, "Overlay updates must allow the render thread to lock"
            self.text = texts

        def sync(self, **kwargs):
            assert not self.lock_held, "Native sync locks internally"
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
    monkeypatch.setattr(inspection, "native_window", Window)
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


@pytest.mark.parametrize("mode", ["live", "trajectory", "recorded"])
def test_mouse_forces_move_live_joints_and_clear_on_release_without_affecting_playback(
    robot, monkeypatch, mode
):
    from mujoco.viewer import Handle

    trajectory = (
        JointTrajectory(robot, ["right_shoulder_pan"], [0, 1], [[0], [0.01]])
        if mode == "trajectory"
        else None
    )
    playback = inspection.DynamicPlayback(robot, trajectory, paused=True)
    if mode == "recorded":
        playback.is_recorded = True
    offline = inspection.DynamicPlayback(robot, trajectory, paused=True)
    offline.step(single=True)
    snapshots = []

    class NativeState:
        pass

    class Window(Handle):
        def __init__(self, model, data, key_callback, **kwargs):
            self.model, self.data, self.callback = model, data, key_callback
            self.native_state = NativeState()
            self.native_state.m, self.native_state.d = model, data
            # Exercise the installed viewer handle's public properties; a
            # hand-written perturbation attribute hid an API mismatch here.
            super().__init__(
                self.native_state,
                mujoco.MjvCamera(),
                mujoco.MjvOption(),
                mujoco.MjvPerturb(),
                None,
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def lock(self):
            return nullcontext()

        def is_running(self):
            return len(snapshots) < 3

        def set_texts(self, texts):
            pass

        def sync(self, **kwargs):
            snapshots.append((self.data.qpos.copy(), self.data.xfrc_applied.copy()))
            if len(snapshots) == 1:
                self.perturb.select = self.model.body("right_Fixed_Jaw").id
                scene = mujoco.MjvScene(self.model, maxgeom=1)
                mujoco.mjv_updateCamera(self.model, self.data, self.cam, scene)
                mujoco.mjv_initPerturb(self.model, self.data, scene, self.perturb)
                self.perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
                self.perturb.refpos[1] += 0.03
                self.perturb.refselpos[1] += 0.03
                self.callback(262)
            elif len(snapshots) == 2:
                self.perturb.select = self.model.body("left_Fixed_Jaw").id
                self.perturb.active = 0
                self.callback(262)

    monkeypatch.setattr(inspection, "check_display", lambda: None)
    monkeypatch.setattr(inspection, "native_window", Window)
    inspection.run_viewer(robot, trajectory, playback=playback)
    if mode == "live":
        assert np.linalg.norm(snapshots[1][1]) > 0
        assert not np.array_equal(snapshots[1][0], offline.data.qpos)
    else:
        np.testing.assert_array_equal(snapshots[1][0], offline.data.qpos)
        np.testing.assert_array_equal(snapshots[1][1], 0)
    # Releasing the mouse and switching selection must leave no persistent load.
    np.testing.assert_array_equal(snapshots[2][1], 0)


def test_missing_display_fails_before_launching_a_render_thread(monkeypatch):
    import glfw

    monkeypatch.setattr(glfw, "init", lambda: False)
    with pytest.raises(RuntimeError, match="desktop session"):
        inspection.check_display()


def test_native_window_joins_native_cleanup_and_propagates_launch_errors(monkeypatch):
    import threading

    from mujoco import viewer

    closed, destroyed = threading.Event(), threading.Event()

    class Handle:
        def close(self):
            closed.set()

    def native(model, data, *, handle_return, **kwargs):
        handle_return.put(Handle())
        assert closed.wait(2)
        destroyed.set()

    monkeypatch.setattr(inspection.sys, "platform", "linux")
    monkeypatch.setattr(viewer, "_launch_internal", native)
    with inspection.native_window(None, None):
        assert not destroyed.is_set()
    assert destroyed.is_set()
    closed.clear()
    destroyed.clear()
    with pytest.raises(KeyboardInterrupt):
        with inspection.native_window(None, None):
            raise KeyboardInterrupt
    assert destroyed.is_set()

    def failed(*args, **kwargs):
        raise ValueError("Native launch failed")

    monkeypatch.setattr(viewer, "_launch_internal", failed)
    with pytest.raises(RuntimeError, match="Cannot launch") as error:
        with inspection.native_window(None, None):
            pass
    assert isinstance(error.value.__cause__, ValueError)

    def cleanup_failed(model, data, *, handle_return, **kwargs):
        handle_return.put(Handle())
        assert closed.wait(2)
        raise ValueError("Native cleanup failed")

    monkeypatch.setattr(viewer, "_launch_internal", cleanup_failed)
    with pytest.raises(RuntimeError, match="viewer failed"):
        with inspection.native_window(None, None):
            pass

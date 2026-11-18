"""Interactive inspection and physics-based playback, isolated from GUI edits."""

from __future__ import annotations

import copy
import queue
import sys
import threading
import time
from contextlib import contextmanager

import mujoco
import numpy as np

from alohamini.simulation.base import BaseMotion

from .control import DynamicFollower
from .libero import settle_scene


def set_geometry(options, *, collisions):
    """Show complete visual meshes or the convex shapes used by collision detection."""
    options.geomgroup[:] = 0
    options.geomgroup[3 if collisions else 2] = 1
    options.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = int(collisions)


def set_camera(robot, camera):
    settings = (
        robot.scene.configuration.get("inspection_camera", {}) if robot.scene is not None else {}
    )
    camera.lookat[:] = settings.get("lookat_world_m", robot.model.stat.center)
    camera.distance = settings.get("distance_m", 1.3 * robot.model.stat.extent)
    camera.azimuth = settings.get("azimuth_deg", 45)
    camera.elevation = settings.get("elevation_deg", -20)


def check_display():
    # launch_passive waits for a render thread: preflight here so an unavailable
    # display/context produces an error instead of waiting on that thread forever.
    import glfw

    if not glfw.init():
        raise RuntimeError("Cannot open GLFW display; run view in a desktop session with DISPLAY")
    try:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(64, 64, "AlohaMini display check", None, None)
        if not window:
            raise RuntimeError("Cannot create an OpenGL window on the current display")
        glfw.destroy_window(window)
    finally:
        glfw.default_window_hints()


@contextmanager
def native_window(model, data, **kwargs):
    """Join the native render thread before Python tears down GLFW.

    MuJoCo 3.12.0 launch_passive creates a daemon and close only requests exit.
    Own the same pinned native entry point so window closure waits for resource
    destruction. macOS retains the mjpython main-thread launcher requirement.
    """
    from mujoco import viewer

    if sys.platform == "darwin":
        with viewer.launch_passive(model, data, **kwargs) as window:
            yield window
        return
    handles, errors = queue.Queue(1), []

    def render():
        try:
            viewer._launch_internal(
                model, data, run_physics_thread=False, handle_return=handles, **kwargs
            )
        except Exception as exc:
            errors.append(exc)
            try:
                handles.put_nowait(exc)
            except queue.Full:
                pass

    thread = threading.Thread(target=render, name="AlohaMini viewer", daemon=False)
    thread.start()
    try:
        window = handles.get()
        if isinstance(window, Exception):
            raise RuntimeError("Cannot launch the native viewer") from window
        try:
            yield window
        finally:
            window.close()
    finally:
        thread.join()
    if errors:
        raise RuntimeError("Native viewer failed") from errors[0]


class DynamicPlayback:
    """Pauseable simulation; reset is the only operation that replaces physics state."""

    def __init__(self, robot, trajectory=None, *, paused=False, follower_settings=None):
        self.robot = robot
        self.trajectory = trajectory
        self.follower = DynamicFollower(robot, settings=follower_settings)
        self.follower_settings = self.follower.settings
        self.initial_target = self.follower.target.copy()
        self.manual_target = self.initial_target.copy()
        if robot.scene is not None:
            if hasattr(robot.scene, "settle"):
                robot.scene.settle(self.follower)
            else:
                settle_scene(self.follower)
        self.base_motion = None
        if robot.scene is not None and robot.scene.configuration.get("base_mode") == "kinematic":
            self.base_motion = BaseMotion(
                robot.description, robot.profile.T_world_root, mode="kinematic"
            )
        self.indices = (
            self.follower.trajectory_indices(trajectory)
            if trajectory is not None
            else np.empty(0, dtype=int)
        )
        self.initial_data = mujoco.MjData(robot.model)
        mujoco.mj_copyData(self.initial_data, robot.model, self.data)
        self.start_time_s = float(self.data.time)
        self.paused = paused
        self.completed = False
        self.saturated_steps = 0

    @property
    def data(self):
        return self.follower.data

    @property
    def duration_s(self):
        if self.trajectory is None:
            return None
        return float(self.trajectory.times[-1]) + self.follower.settings.settle_time_s

    def reset(self):
        mujoco.mj_copyData(self.data, self.robot.model, self.initial_data)
        self.follower.target[:] = self.initial_target
        self.manual_target[:] = self.initial_target
        self.follower.target_velocity[:] = 0
        self.paused = True
        self.completed = False
        self.saturated_steps = 0
        if self.base_motion is not None:
            self.base_motion = BaseMotion(
                self.robot.description, self.robot.profile.T_world_root, mode="kinematic"
            )

    def set_manual_target(self, target):
        target = np.asarray(target, dtype=float)
        if target.shape != self.manual_target.shape:
            raise ValueError("Manual targets must match the driven joint order")
        for name, value in zip(self.follower.names, target, strict=True):
            self.robot.description.joints[name].check_position(float(value))
        self.manual_target[:] = target

    def step(self, *, single=False):
        if self.completed or (self.paused and not single):
            return False
        if self.trajectory is not None:
            q, velocity = self.trajectory.reference(float(self.data.time) - self.start_time_s)
            self.follower.target[self.indices] = q
            self.follower.target_velocity[self.indices] = velocity
        else:
            if self.base_motion is not None:
                from .model import pose_attributes

                pose = pose_attributes(self.base_motion.step(self.robot.model.opt.timestep))
                mocap = self.robot.model.body_mocapid[
                    self.robot.model.body(self.robot.description.root).id
                ]
                self.data.mocap_pos[mocap] = np.fromstring(pose["pos"], sep=" ")
                self.data.mocap_quat[mocap] = np.fromstring(pose["quat"], sep=" ")
                for name, q in zip(
                    self.base_motion.kinematics.names, self.base_motion.wheel_positions, strict=True
                ):
                    self.manual_target[self.follower.names.index(name)] = q
            velocity = np.array(
                [self.robot.description.joints[n].velocity for n in self.follower.names]
            )
            delta = np.clip(
                self.manual_target - self.follower.target,
                -velocity * self.robot.model.opt.timestep,
                velocity * self.robot.model.opt.timestep,
            )
            self.follower.target[:] += delta
            self.follower.target_velocity[:] = delta / self.robot.model.opt.timestep
        self.saturated_steps += int(self.follower.step().any())
        if (
            self.duration_s is not None
            and self.data.time - self.start_time_s + 1e-12 >= self.duration_s
        ):
            self.completed = True
            self.paused = True
        return True


class JointControls:
    """Translate native UI edits into bounded, speed-limited position commands.

    The display copy never integrates dynamics. Its Control fields expose
    rad/metre targets instead of raw actuator torque, and Joint slider edits
    request the same targets without overwriting the physical state.
    """

    def __init__(self, playback, display_model, display_data):
        self.playback = playback
        self.qpos_ids = playback.follower.qpos_ids
        self.limits = np.array(
            [
                (
                    [joint.lower, joint.upper]
                    if joint.lower is not None
                    else [playback.manual_target[i] - np.pi, playback.manual_target[i] + np.pi]
                )
                for i, n in enumerate(playback.follower.names)
                for joint in [playback.robot.description.joints[n]]
            ]
        )
        display_model.actuator_ctrlrange[:] = self.limits
        display_model.actuator_ctrllimited[:] = 1
        self.publish(display_data)

    def publish(self, display_data):
        display_data.ctrl[:] = self.playback.manual_target
        self.last_positions = display_data.qpos[self.qpos_ids].copy()
        self.last_controls = display_data.ctrl.copy()

    def read(self, display_data):
        goal = self.playback.manual_target.copy()
        for candidate, previous in (
            (display_data.qpos[self.qpos_ids], self.last_positions),
            (display_data.ctrl, self.last_controls),
        ):
            changed = candidate != previous
            valid = (
                np.isfinite(candidate)
                & (candidate >= self.limits[:, 0])
                & (candidate <= self.limits[:, 1])
            )
            goal[changed & valid] = candidate[changed & valid]
        self.playback.set_manual_target(goal)


def run_viewer(
    robot,
    trajectory=None,
    *,
    collisions=False,
    paused=False,
    follower_settings=None,
    playback=None,
):
    """Open a native window, preserving every physics step even below real time."""
    check_display()
    if playback is None:
        playback = DynamicPlayback(
            robot, trajectory, paused=paused, follower_settings=follower_settings
        )
    mouse_forces = (
        getattr(playback, "trajectory", trajectory) is None
        and not getattr(playback, "is_recorded", False)
        and not getattr(playback, "is_policy", False)
    )
    # Native joint sliders and physics controls can write model/data during sync.
    # A separate presentation copy keeps those writes out of dynamic playback.
    display_model = copy.copy(robot.model)
    display_data = mujoco.MjData(display_model)
    mujoco.mj_copyData(display_data, display_model, playback.data)
    joint_controls = JointControls(playback, display_model, display_data) if mouse_forces else None
    events = queue.SimpleQueue()
    with native_window(
        display_model,
        display_data,
        key_callback=events.put,
        show_left_ui=True,
        show_right_ui=True,
    ) as window:
        with window.lock():
            set_geometry(window.opt, collisions=collisions)
            set_camera(robot, window.cam)
        published_time = float(display_data.time)

        perturbation = None

        def step(*, single=False):
            if mouse_forces and perturbation is not None:
                # Native perturbation writes only the selected body. Clear old
                # mouse forces when releasing or selecting a different body.
                playback.data.xfrc_applied[:] = 0
                mujoco.mjv_applyPerturbForce(robot.model, playback.data, perturbation)
            return playback.step(single=single)

        frame_period_s = 1 / 30
        wall_anchor = rate_wall = time.monotonic()
        sim_anchor = rate_sim = float(playback.data.time)
        realtime_factor = 0.0
        print(
            "AlohaMini viewer: Space play/pause; F8 visual/collision; "
            "F9 reset and pause; Right single step; F6 frames; "
            + ("Ctrl+drag applies body forces; " if mouse_forces else "")
            + "close window to exit.",
            flush=True,
        )
        while window.is_running():
            frame_start = time.monotonic()
            with window.lock():
                if display_data.time < published_time:
                    playback.reset()
            if joint_controls is not None:
                with window.lock():
                    joint_controls.read(display_data)
            if mouse_forces:
                # Pointer events update at display cadence. Read their target
                # once per frame; recompute the spring force at each physics
                # step without repeatedly contending with the render thread.
                with window.lock():
                    perturbation = copy.copy(window.perturb)
            while not events.empty():
                key = events.get()
                if key == 32 and not playback.completed:  # Space
                    # Passive MuJoCo windows leave playback to the application;
                    # Simulate.run is read-only and its native Run UI is disabled.
                    playback.paused = not playback.paused
                elif key == 297:  # GLFW_KEY_F8
                    collisions = not collisions
                    with window.lock():
                        set_geometry(window.opt, collisions=collisions)
                elif key == 298:  # GLFW_KEY_F9
                    playback.reset()
                elif key == 262 and playback.paused:  # GLFW_KEY_RIGHT
                    step(single=True)
                # Re-anchor after interaction: paused wall time never advances the
                # reference, and reset never causes an automatic catch-up burst.
                wall_anchor = rate_wall = frame_start
                sim_anchor = rate_sim = float(playback.data.time)
                realtime_factor = 0.0
            if playback.paused:
                wall_anchor = frame_start
                sim_anchor = float(playback.data.time)
            else:
                target_time = sim_anchor + frame_start - wall_anchor + frame_period_s
                # Bound wall time per frame, not physics accuracy or geometry.
                # Slow models fall behind wall time; no integration steps are skipped.
                while (
                    not playback.paused
                    and playback.data.time + 1e-12 < target_time
                    and time.monotonic() < frame_start + frame_period_s
                ):
                    step()
            now = time.monotonic()
            if now - rate_wall >= 0.5:
                realtime_factor = (float(playback.data.time) - rate_sim) / (now - rate_wall)
                rate_wall, rate_sim = now, float(playback.data.time)
            mode = (
                "Finished (F9 to reset)"
                if playback.completed
                else "Paused (Space to run)"
                if playback.paused
                else "Replaying recorded commands"
                if getattr(playback, "is_recorded", False)
                else "Running policy"
                if getattr(playback, "is_policy", False)
                else "Following trajectory"
                if trajectory is not None
                else "Following joint targets"
            )
            with window.lock():
                mujoco.mj_copyData(display_data, display_model, playback.data)
                for field in dir(robot.model.opt):
                    if field.startswith("_"):
                        continue
                    value = getattr(robot.model.opt, field)
                    if isinstance(value, np.ndarray):
                        getattr(display_model.opt, field)[:] = value
                    elif not callable(value):
                        setattr(display_model.opt, field, value)
                published_time = float(playback.data.time)
                if joint_controls is not None:
                    joint_controls.publish(display_data)
            # Overlay updates wait for the render thread. It must be able to
            # acquire the viewer lock while processing the request.
            window.set_texts(
                (
                    mujoco.mjtFont.mjFONT_NORMAL,
                    mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    "AlohaMini\nTime / speed\nGeometry\nContacts / saturated steps\nControls",
                    f"{mode}\n{playback.data.time:.3f} s / {realtime_factor:.2f}x\n"
                    f"{'Collision convex hulls' if collisions else 'Visual meshes'}\n"
                    f"{playback.data.ncon} / {playback.saturated_steps}\n"
                    "Space: pause  F8: geometry  F9: reset  Right: step  F6: frames"
                    + (
                        "  Sliders: joint targets (rad/m)  Ctrl+drag: force" if mouse_forces else ""
                    ),
                )
            )
            window.sync(state_only=True)
            time.sleep(max(0, frame_start + frame_period_s - time.monotonic()))
    return playback

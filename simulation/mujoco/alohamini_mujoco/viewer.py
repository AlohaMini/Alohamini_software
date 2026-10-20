"""Interactive inspection and physics-based playback, isolated from GUI edits."""

from __future__ import annotations

import copy
import queue
import time

import mujoco
import numpy as np

from .control import DynamicFollower


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


class DynamicPlayback:
    """Pauseable simulation; reset is the only operation that replaces physics state."""

    def __init__(self, robot, trajectory=None, *, paused=False):
        self.robot = robot
        self.trajectory = trajectory
        self.follower = DynamicFollower(robot)
        self.indices = (
            self.follower.trajectory_indices(trajectory)
            if trajectory is not None
            else np.empty(0, dtype=int)
        )
        self.initial_data = robot.data()
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
        self.follower = DynamicFollower(self.robot, data=self.data)
        self.paused = True
        self.completed = False
        self.saturated_steps = 0

    def step(self, *, single=False):
        if self.completed or (self.paused and not single):
            return False
        if self.trajectory is not None:
            q, velocity = self.trajectory.reference(float(self.data.time))
            self.follower.target[self.indices] = q
            self.follower.target_velocity[self.indices] = velocity
        self.saturated_steps += int(self.follower.step().any())
        if self.duration_s is not None and self.data.time + 1e-12 >= self.duration_s:
            self.completed = True
            self.paused = True
        return True


def run_viewer(robot, trajectory=None, *, collisions=False, paused=False, links=None):
    """Open a native window, preserving every physics step even below real time."""
    from mujoco import viewer

    check_display()
    playback = DynamicPlayback(robot, trajectory, paused=paused)
    # Native joint sliders and physics controls can write model/data during sync.
    # A separate presentation copy keeps those writes out of dynamic playback.
    display_model = copy.copy(robot.model)
    if links is not None:
        unknown = set(links) - robot.description.links.keys()
        selected = [
            display_model.geom(name).id for name, g in robot.geometry.items() if g["link"] in links
        ]
        if unknown or not selected:
            raise ValueError(
                f"Select existing robot links with geometry; unknown={sorted(unknown)}"
            )
        hidden = np.ones(display_model.ngeom, dtype=bool)
        hidden[selected] = False
        display_model.geom_group[hidden] = 0
    display_data = mujoco.MjData(display_model)
    mujoco.mj_copyData(display_data, display_model, playback.data)
    events = queue.SimpleQueue()
    with viewer.launch_passive(
        display_model,
        display_data,
        key_callback=events.put,
        show_left_ui=False,
        show_right_ui=False,
    ) as window:
        with window.lock():
            set_geometry(window.opt, collisions=collisions)
            set_camera(robot, window.cam)
            if links is not None:
                centers = display_data.geom_xpos[selected]
                radii = display_model.geom_rbound[selected, None]
                lower, upper = (centers - radii).min(0), (centers + radii).max(0)
                window.cam.lookat[:] = (lower + upper) / 2
                window.cam.distance = max(0.1, float(np.linalg.norm(upper - lower)) * 1.2)
        frame_period_s = 1 / 30
        wall_anchor = rate_wall = time.monotonic()
        sim_anchor = rate_sim = float(playback.data.time)
        realtime_factor = 0.0
        print(
            "AlohaMini viewer: Space play/pause; F8 visual/collision; "
            "F9 reset and pause; Right single step; F6 frames; close window to exit.",
            flush=True,
        )
        while window.is_running():
            frame_start = time.monotonic()
            while not events.empty():
                key = events.get()
                if key == 32 and not playback.completed:  # Space
                    playback.paused = not playback.paused
                elif key == 297:  # GLFW_KEY_F8
                    collisions = not collisions
                    with window.lock():
                        set_geometry(window.opt, collisions=collisions)
                elif key == 298:  # GLFW_KEY_F9
                    playback.reset()
                elif key == 262 and playback.paused:  # GLFW_KEY_RIGHT
                    playback.step(single=True)
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
                    playback.step()
            now = time.monotonic()
            if now - rate_wall >= 0.5:
                realtime_factor = (float(playback.data.time) - rate_sim) / (now - rate_wall)
                rate_wall, rate_sim = now, float(playback.data.time)
            mode = (
                "Finished (F9 to reset)"
                if playback.completed
                else "Paused"
                if playback.paused
                else "Following trajectory"
                if trajectory is not None
                else "Holding initial pose"
            )
            with window.lock():
                mujoco.mj_copyData(display_data, display_model, playback.data)
                window.set_texts(
                    (
                        mujoco.mjtFont.mjFONT_NORMAL,
                        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                        "AlohaMini\nTime / speed\nGeometry\nContacts / saturated steps\nControls",
                        f"{mode}\n{playback.data.time:.3f} s / {realtime_factor:.2f}x\n"
                        f"{'Collision convex hulls' if collisions else 'Visual meshes'}\n"
                        f"{playback.data.ncon} / {playback.saturated_steps}\n"
                        "Space: pause  F8: geometry  F9: reset  Right: step  F6: frames",
                    )
                )
            window.sync(state_only=True)
            time.sleep(max(0, frame_start + frame_period_s - time.monotonic()))
    return playback

"""Refresh the native viewer between physics steps with a separate wall clock."""

import math
import time


class ViewerClock:
    def __init__(self, render, *, timestep, refresh_hz, control_period):
        if any(not math.isfinite(v) or v <= 0 for v in (timestep, refresh_hz, control_period)):
            raise ValueError("Viewer and simulation periods must be finite and positive")
        self.render = render
        self.timestep = timestep
        self.frame_period = 1 / refresh_hz
        self.control_period = control_period
        self.closed = False
        self._step_deadline = self._frame_deadline = time.perf_counter()

    def step(self):
        if self.closed:
            return
        now = time.perf_counter()
        if now >= self._frame_deadline:
            viewer = self.render()
            self.closed = viewer.closed
            if self.closed:
                return
            self._frame_deadline = now + self.frame_period
            finished = time.perf_counter()
            if finished >= self._frame_deadline:
                # Slow presentation must leave a physics interval before the
                # next frame, rather than rendering after every 1 ms step.
                self._frame_deadline = finished + self.frame_period
            now = finished
        self._step_deadline += self.timestep
        if now - self._step_deadline > self.control_period:
            # Resuming a paused or blocked window starts a fresh wall-clock
            # schedule. Physics still advances exactly one timestep per call.
            self._step_deadline = now
        delay = self._step_deadline - now
        if delay > 0:
            time.sleep(delay)

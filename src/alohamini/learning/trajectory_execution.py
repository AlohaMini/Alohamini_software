"""Single-worker execution of physical-time plans; policy adapters own continuity."""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from alohamini.trajectory import committed_prefix, plan_from_dict
from alohamini.trajectory_execution import prediction_timing


class TrajectoryExecutor:
    asynchronous = True

    def __init__(self, policy, config, *, clock=time.monotonic, executor=None, control_fps=None):
        if not config.enabled:
            raise ValueError("TrajectoryExecutor requires enabled asynchronous execution")
        config = policy.plan_capabilities.resolve(config)
        self.policy, self.settings, self.clock = policy, config, clock
        self.fps = policy.fps if control_fps is None else control_fps
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Trajectory control FPS must be finite and positive")
        self.executor, self.future = executor, None
        self.generation, self.closed = 0, False
        self.worker_generation = None
        self.execution = config.to_dict()
        self.reset()

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def reset(self):
        self.generation += 1
        self.origin = self.clock()
        self.plan = self.plan_sampled_s = None
        self.staged = None
        self.last_tick = -1
        self.latency_s = 0.0
        self.execution_timing = None
        self.tcp_tracker = None
        if self.policy.plan_capabilities.kinds == ("tcp",):
            from alohamini.tcp_execution import TCPTracker

            self.tcp_tracker = TCPTracker(fps=self.fps, settings=self.settings.tcp_tracking)
        self.task_at_reset = getattr(self.policy, "task", None)

    def close(self):
        self.closed = True
        self.reset()
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)

    def _predict(self, snapshot, prefix, generation):
        if self.worker_generation != generation:
            self.policy.reset()
            self.worker_generation = generation
        plan = self.policy.predict_plan(
            snapshot, prefix=prefix, continuity=self.settings.continuity
        )
        return plan, self.clock()

    def select_action(self, snapshot):
        if self.closed:
            raise RuntimeError("Trajectory executor is closed")
        if getattr(self.policy, "task", None) != self.task_at_reset:
            self.reset()
        now = self.clock()
        tick = max(0, math.floor((now - self.origin) * self.fps))
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            generation, sampled_s = self.pending
            if generation == self.generation:
                plan, finished_s = future.result()
                # Serialization validation also ensures a worker-owned, immutable CPU plan.
                plan = plan_from_dict(plan.to_dict(), names=self.names)
                if plan.to_dict()["kind"] not in self.policy.plan_capabilities.kinds:
                    raise ValueError("Policy returned an undeclared trajectory type")
                self.latency_s = max(0, finished_s - sampled_s)
                age = now - sampled_s
                expired = age < 0 or age >= min(plan.end_s, self.settings.max_prediction_age_s)
                if expired and getattr(plan, "kind", None) != "tcp":
                    raise RuntimeError("Trajectory prediction expired before execution")
                if expired:
                    if self.tcp_tracker is not None:
                        self.tcp_tracker.needs_replan = True
                elif age < plan.start_s:
                    self.staged = (plan, sampled_s)
                else:
                    self.plan, self.plan_sampled_s = plan, sampled_s

        if self.staged is not None:
            candidate, sampled_s = self.staged
            if now >= sampled_s + candidate.start_s:
                self.plan, self.plan_sampled_s = candidate, sampled_s
                self.staged = None

        sampled_s = snapshot.request_started_s
        if not math.isfinite(sampled_s) or sampled_s > now:
            raise ValueError("Observation has an invalid local monotonic timestamp")
        delay, lead = prediction_timing(
            self.settings,
            self.plan,
            fps=self.fps,
            latency_s=self.latency_s,
            observation_age_s=now - sampled_s,
        )
        due = self.plan is None
        if self.plan is not None:
            deadline = min(self.plan.end_s, self.settings.max_prediction_age_s)
            if self.settings.replan_interval_s is not None:
                deadline = min(deadline, self.settings.replan_interval_s)
            due = now - self.plan_sampled_s >= deadline - lead
        due = due or (self.tcp_tracker is not None and self.tcp_tracker.needs_replan)
        newer = self.plan_sampled_s is None or sampled_s > self.plan_sampled_s
        if self.future is None and self.staged is None and due and newer:
            prefix = None
            if self.plan is not None and getattr(self.plan, "kind", None) != "tcp":
                prefix = committed_prefix(
                    self.plan, sampled_s - self.plan_sampled_s, self.fps, delay
                )
            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trajectory")
            self.pending = (self.generation, sampled_s)
            self.future = self.executor.submit(
                self._predict, deepcopy(snapshot), prefix, self.generation
            )

        if tick <= self.last_tick:
            return None
        if self.plan is None:
            if now - self.origin > self.settings.startup_timeout_s:
                raise RuntimeError("Trajectory prediction startup timed out")
            return None
        elapsed = now - self.plan_sampled_s
        if now < self.plan_sampled_s + self.plan.start_s:
            return None
        tcp = getattr(self.plan, "kind", None) == "tcp"
        if elapsed >= min(self.plan.end_s, self.settings.max_prediction_age_s) and not tcp:
            raise RuntimeError("Trajectory plan exhausted or expired")
        # Subtracting a large clock origin can round just below a valid start.
        if tcp:
            from alohamini.tcp_execution import TCPTracker

            if self.tcp_tracker is None:
                self.tcp_tracker = TCPTracker(fps=self.fps, settings=self.settings.tcp_tracking)
            values = self.tcp_tracker.step(
                self.plan,
                snapshot,
                elapsed,
                now,
                expired=elapsed >= self.settings.max_prediction_age_s,
            )
        else:
            values = self.plan.action_at(max(self.plan.start_s, elapsed))
        self.last_tick = tick
        self.execution_timing = dict(
            observation_request_started_s=self.plan_sampled_s,
            execution_elapsed_s=elapsed,
            prediction_age_s=elapsed,
            inference_delay_s=self.latency_s,
        )
        if tcp:
            governor = self.tcp_tracker.reference_governor
            self.execution_timing.update(
                tcp_reference_elapsed_s=governor.elapsed_s,
                tcp_reference_rate=governor.rate,
            )
        return dict(zip(self.names, values, strict=True))

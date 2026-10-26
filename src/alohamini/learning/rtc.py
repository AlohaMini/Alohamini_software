"""Single-worker RTC execution on a monotonic action timeline."""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import torch


class RTCPolicy:
    asynchronous = True

    def __init__(self, policy, config, *, clock=time.monotonic, executor=None):
        if not config.enabled:
            raise ValueError("RTCPolicy requires enabled RTC")
        self.horizon = policy.validate_rtc(config)
        self.policy, self.settings, self.clock = policy, config, clock
        self.rtc = config.to_dict()
        self.executor = executor
        self.future = None
        self.generation = 0
        self.closed = False
        self.reset()

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def reset(self):
        # A running CUDA call cannot be cancelled. Discard its result without
        # mutating model state from the control thread.
        self.generation += 1
        self.origin = self.clock()
        self.plan = None
        self.plan_tick = self.last_tick = -1
        self.plan_sampled_s = None
        self.delay_steps = self.settings.inference_delay_steps
        self.rtc_timing = None
        self.task_at_reset = getattr(self.policy, "task", None)

    def close(self):
        self.closed = True
        self.reset()
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)

    def _predict(self, snapshot, previous, delay):
        self.policy.reset()
        result = self.policy.predict_action_chunk(
            snapshot, rtc=self.settings, previous_actions=previous, inference_delay=delay,
        )
        return result, self.clock()

    def select_action(self, snapshot):
        if self.closed:
            raise RuntimeError("RTC executor is closed")
        if getattr(self.policy, "task", None) != self.task_at_reset:
            self.reset()
        now = self.clock()
        tick = max(0, math.floor((now - self.origin) * self.fps))
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            generation, source_tick, sampled_s = self.pending
            if generation == self.generation:
                chunk, finished_s = future.result()
                chunk = torch.as_tensor(chunk).detach().float().cpu()
                if (
                    chunk.shape != (self.horizon, len(self.names))
                    or not torch.isfinite(chunk).all()
                ):
                    raise ValueError("RTC policy returned invalid absolute targets")
                self.delay_steps = max(
                    self.settings.inference_delay_steps,
                    math.ceil(max(0, finished_s - sampled_s) * self.fps) + 1,
                )
                if (
                    now - sampled_s >= self.settings.max_prediction_age_s
                    or tick - source_tick >= len(chunk)
                ):
                    raise RuntimeError("RTC prediction expired before it could be executed")
                self.plan, self.plan_tick, self.plan_sampled_s = chunk, source_tick, sampled_s

        if self.future is None and (
            self.plan is None or tick >= self.plan_tick + self.settings.replan_steps
        ):
            sampled_s = snapshot.request_started_s
            source_tick = math.floor((sampled_s - self.origin) * self.fps)
            previous = None
            delay = self.delay_steps
            if self.plan is not None:
                offset = source_tick - self.plan_tick
                if offset <= 0:
                    source_tick = None  # Wait for a newer Host sample.
                else:
                    previous = self.plan[offset:].clone()
                    delay = max(delay, tick - source_tick + 1)
                    if delay >= len(previous):
                        raise RuntimeError(
                            "RTC inference delay exceeds the remaining action overlap"
                        )
            if source_tick is not None:
                if self.executor is None:
                    self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc")
                self.pending = (self.generation, source_tick, sampled_s)
                self.future = self.executor.submit(
                    self._predict, deepcopy(snapshot), previous, delay
                )

        if tick <= self.last_tick:
            return None
        if self.plan is None:
            if now - self.origin > 15.0:
                raise RuntimeError("RTC startup timed out")
            return None
        index = tick - self.plan_tick
        if (
            not 0 <= index < len(self.plan)
            or now - self.plan_sampled_s >= self.settings.max_prediction_age_s
        ):
            raise RuntimeError("RTC action buffer exhausted or expired; stopping control")
        self.last_tick = tick
        self.rtc_timing = dict(
            source_tick=self.plan_tick, execution_tick=tick,
            observation_request_started_s=self.plan_sampled_s,
            prediction_age_s=now - self.plan_sampled_s,
            inference_delay_steps=self.delay_steps,
        )
        return dict(zip(self.names, self.plan[index].tolist(), strict=True))

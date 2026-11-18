"""Model-side continuity adapters; the executor only passes physical-time prefixes."""

import math
from bisect import bisect_right

import torch

from alohamini.learning.processor import scale_action
from alohamini.policies.rtc import RTCGuidance, RTCGuidanceConfig
from alohamini.trajectory_execution import PlanCapabilities

DENSE_FLOW_CAPABILITIES = PlanCapabilities(
    continuity_methods=("none", "rtc"), gradient_methods=("rtc",)
)


def dense_flow_arguments(policy, batch, prefix, continuity):
    if continuity == "none":
        return {}
    if continuity != "rtc":
        raise ValueError("Unsupported dense-flow continuity method")
    encoded = None
    delay = 0
    if prefix is not None:
        if prefix.names != tuple(policy.names):
            raise ValueError("Continuity prefix has different action coordinates")
        if prefix.times_s[0] > 1e-6:
            raise ValueError("Dense flow RTC requires a prefix covering t=0")
        times = [i / policy.fps for i in range(int(prefix.times_s[-1] * policy.fps) + 1)]
        rows = [prefix.actions[max(0, bisect_right(prefix.times_s, t + 1e-9) - 1)] for t in times]
        previous = torch.tensor(
            rows, dtype=torch.float32, device=next(policy.model.parameters()).device
        )
        previous = scale_action(
            previous,
            getattr(policy.config, "inference_action_scale_dims", ()),
            1 / getattr(policy.config, "inference_action_scale", 1.0),
        )
        encoded = policy.processor.action_prefix(
            previous[None], context=batch.get("_action_context")
        )
        delay = math.ceil(prefix.inference_delay_s * policy.fps)
        if delay >= len(rows):
            raise ValueError("RTC inference delay exceeds its committed action prefix")
    return dict(
        rtc_processor=RTCGuidance(RTCGuidanceConfig()),
        prev_chunk_left_over=encoded,
        inference_delay=delay,
    )

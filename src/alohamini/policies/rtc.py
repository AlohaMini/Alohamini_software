"""Inference-time RTC guidance, with the OpenPI noise-at-t=1 convention.

Algorithm: Physical Intelligence, real-time-chunking-kinetix/src/model.py,
realtime_action. The VJP includes the denoiser Jacobian, not only the identity.
"""

import math

import torch


def prefix_weights(length, delay, overlap, *, device, schedule="linear"):
    end = min(overlap, length)
    start = min(delay, end)
    index = torch.arange(length, device=device)
    if schedule == "ones":
        weights = torch.ones(length, device=device)
    elif schedule == "zeros":
        weights = (index < start).float()
    else:
        weights = ((start - 1 - index) / (end - start + 1) + 1).clamp(0, 1)
        if schedule == "exp":
            weights = weights * weights.expm1() / math.expm1(1)
        elif schedule != "linear":
            raise ValueError("Unknown RTC prefix attention schedule")
    return weights.masked_fill(index >= end, 0)


class RTCGuidance:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def is_debug_enabled():
        return False

    def denoise_step(
        self, *, x_t, prev_chunk_left_over, inference_delay, time,
        original_denoise_step_partial, execution_horizon=None,
    ):
        previous = prev_chunk_left_over
        if previous is None or previous.shape[-2] == 0:
            return original_denoise_step_partial(x_t)
        if (
            previous.ndim != 3 or previous.shape[0] != x_t.shape[0]
            or not 0 < previous.shape[2] <= x_t.shape[2]
            or type(inference_delay) is not int or inference_delay < 0
        ):
            raise ValueError("Invalid RTC prefix dimensions or inference delay")
        count = min(previous.shape[1], x_t.shape[1])
        if execution_horizon is not None:
            count = min(count, execution_horizon)
        width = previous.shape[2]
        weights = prefix_weights(
            count, inference_delay, count, device=x_t.device,
            schedule=self.config.prefix_attention_schedule,
        )[None, :, None]
        # The caller prepares prefix caches outside inference_mode. Only this
        # latent needs a gradient; autograd.grad never populates weight .grad.
        with torch.inference_mode(False), torch.enable_grad():
            latent = x_t.detach().clone().requires_grad_(True)
            velocity = original_denoise_step_partial(latent)
            clean = latent - time * velocity
            error = (previous[:, :count].to(clean) - clean[:, :count, :width]) * weights
            correction = torch.autograd.grad(
                clean[:, :count, :width], latent, grad_outputs=error.detach(),
            )[0]
        denominator = time * (1 - time)
        strength = self.config.max_guidance_weight if denominator <= 0 else min(
            (time * time + (1 - time) ** 2) / denominator,
            self.config.max_guidance_weight,
        )
        return (velocity - strength * correction).detach()

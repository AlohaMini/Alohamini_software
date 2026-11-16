"""RTC math, coordinate frames and deterministic asynchronous execution."""

from unittest.mock import Mock

import pytest
import torch

from alohamini.policies.rtc import RTCGuidance, prefix_weights
from alohamini.policies.rtc import RTCGuidanceConfig as RTCConfig


@pytest.mark.parametrize("schedule", ["linear", "exp", "zeros", "ones"])
def test_prefix_weights_commit_delay_and_ignore_nonoverlap(schedule):
    result = prefix_weights(10, 2, 6, device="cpu", schedule=schedule)
    torch.testing.assert_close(result[:2], torch.ones(2))
    assert not result[6:].any()
    if schedule == "linear":
        torch.testing.assert_close(result[2:6], torch.tensor([0.8, 0.6, 0.4, 0.2]))
    if schedule == "zeros":
        assert not result[2:].any()


@pytest.mark.parametrize("time", [1.0, 0.5, 0.1, 0.0])
def test_guidance_uses_full_denoiser_vjp_without_parameter_gradients(time):
    config = RTCConfig()
    weight = torch.nn.Parameter(torch.tensor(0.3))
    latent = torch.ones(1, 4, 3)
    previous = torch.full((1, 2, 2), 2.0)
    result = RTCGuidance(config).denoise_step(
        x_t=latent, prev_chunk_left_over=previous, inference_delay=1, time=time,
        original_denoise_step_partial=lambda x: weight * x,
    )
    strength = 10 if time in (0, 1) else min(
        (time**2 + (1-time)**2) / (time * (1-time)), 10
    )
    jacobian = 1 - time * weight.detach()
    error = (2 - jacobian) * torch.tensor([1.0, 0.5])
    expected = torch.full_like(latent, 0.3)
    expected[:, :2, :2] -= (strength * jacobian * error)[None, :, None]
    torch.testing.assert_close(result, expected)
    assert weight.grad is None and not result.requires_grad
    assert result.isfinite().all()


def test_no_prefix_keeps_original_velocity():
    latent = torch.randn(1, 4, 3)
    fn = Mock(return_value=latent * 2)
    actual = RTCGuidance(RTCConfig()).denoise_step(
        x_t=latent, prev_chunk_left_over=None, inference_delay=2, time=1.0,
        original_denoise_step_partial=fn,
    )
    torch.testing.assert_close(actual, latent * 2)
    fn.assert_called_once_with(latent)


def test_pi05_prefix_is_reencoded_relative_to_new_state():
    from test_pi05 import processor

    transform = processor()
    old_state = torch.arange(18.0)[None]
    new_state = old_state + 5
    original = torch.randn(1, 6, 32)
    absolute = transform.restore_actions(original, old_state)
    encoded = transform.encode_actions(absolute, new_state)
    assert encoded.shape == (1, 6, 18)
    restored = transform.restore_actions(torch.nn.functional.pad(encoded, (0, 14)), new_state)
    torch.testing.assert_close(restored, absolute, atol=2e-6, rtol=1e-6)
    assert not torch.allclose(encoded, original[..., :18])

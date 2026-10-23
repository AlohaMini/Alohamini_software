"""RTC math, coordinate frames and deterministic asynchronous execution."""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from alohamini.learning.rtc import RTCPolicy
from alohamini.policies.rtc import RTCGuidance, prefix_weights
from alohamini.rtc import RTCConfig


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
    config = RTCConfig(enabled=True)
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
    actual = RTCGuidance(RTCConfig(enabled=True)).denoise_step(
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


class ManualExecutor:
    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        future = Future()
        self.jobs.append((future, fn, args))
        return future

    def complete(self):
        future, fn, args = self.jobs[-1]
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)

    def shutdown(self, **kwargs):
        self.shutdown_options = kwargs


def execution():
    clock = SimpleNamespace(now=100.0)
    executor = ManualExecutor()
    policy = SimpleNamespace(
        fps=10, names=["joint.pos"], task="pick",
        validate_rtc=Mock(return_value=8), reset=Mock(),
        predict_action_chunk=Mock(return_value=torch.arange(8.0)[:, None]),
    )
    config = RTCConfig(enabled=True, replan_steps=3, inference_delay_steps=1)
    rtc = RTCPolicy(policy, config, clock=lambda: clock.now, executor=executor)
    return rtc, policy, clock, executor


def select(rtc, clock, now):
    clock.now = now
    return rtc.select_action(SimpleNamespace(request_started_s=now))


def test_async_skips_elapsed_actions_and_guides_from_executing_plan():
    rtc, policy, clock, executor = execution()
    assert select(rtc, clock, 100.0) is None
    assert select(rtc, clock, 100.11) is None
    assert len(executor.jobs) == 1
    clock.now = 100.16
    executor.complete()
    assert select(rtc, clock, 100.21) == {"joint.pos": 2.0}
    assert select(rtc, clock, 100.22) is None
    assert select(rtc, clock, 100.31) == {"joint.pos": 3.0}
    assert len(executor.jobs) == 2
    assert policy.reset.call_count == 1  # No model access from the control thread.
    prefix = executor.jobs[-1][2][1]
    torch.testing.assert_close(prefix[:, 0], torch.arange(3.0, 8.0))
    assert executor.jobs[-1][2][2] == 3  # ceil(0.16 * 10) + one tick margin
    policy.predict_action_chunk.return_value = torch.arange(100.0, 108.0)[:, None]
    clock.now = 100.46
    executor.complete()
    assert select(rtc, clock, 100.51) == {"joint.pos": 102.0}
    rtc.close()
    assert executor.shutdown_options == dict(wait=False, cancel_futures=True)


def test_reset_discards_running_result_without_concurrent_model_reset():
    rtc, policy, clock, executor = execution()
    select(rtc, clock, 100.0)
    rtc.reset()
    policy.reset.assert_not_called()
    assert select(rtc, clock, 100.11) is None
    assert len(executor.jobs) == 1
    executor.complete()
    assert select(rtc, clock, 100.21) is None
    assert len(executor.jobs) == 2
    assert rtc.plan is None
    rtc.close()


def test_expired_result_and_worker_error_do_not_produce_actions():
    rtc, policy, clock, executor = execution()
    select(rtc, clock, 100.0)
    clock.now = 101.0
    executor.complete()
    with pytest.raises(RuntimeError, match="expired"):
        select(rtc, clock, 101.0)
    rtc.reset()
    policy.predict_action_chunk.side_effect = RuntimeError("failed sampler")
    select(rtc, clock, 101.0)
    executor.complete()
    with pytest.raises(RuntimeError, match="failed sampler"):
        select(rtc, clock, 101.1)
    rtc.close()


def test_exhausted_buffer_never_repeats_last_action():
    rtc, _, clock, executor = execution()
    select(rtc, clock, 100.0)
    clock.now = 100.01
    executor.complete()
    assert select(rtc, clock, 100.11)
    assert select(rtc, clock, 100.31)
    with pytest.raises(RuntimeError, match="exhausted"):
        select(rtc, clock, 100.81)
    rtc.close()


@pytest.mark.parametrize("values", [
    {"replan_steps": 0}, {"inference_delay_steps": -1},
    {"max_guidance_weight": float("nan")}, {"max_prediction_age_s": 0},
    {"prefix_attention_schedule": "bad"}, {"enabled": 1},
])
def test_invalid_rtc_settings(values):
    with pytest.raises(ValueError):
        RTCConfig(**values)


def test_cached_observation_before_reset_keeps_its_original_action_time():
    rtc, _, clock, executor = execution()
    rtc.select_action(SimpleNamespace(request_started_s=99.89))
    clock.now = 100.01
    executor.complete()
    assert select(rtc, clock, 100.11) == {"joint.pos": 3.0}
    rtc.close()


def test_rtc_settings_are_recorded_as_deployment_options(tmp_path):
    import json

    from alohamini.analysis.provenance import write_evaluation

    rtc, _, _, _ = execution()
    write_evaluation(
        tmp_path, rtc, {"schema_version": 1}, fps=10, task="pick", num_episodes=1,
    )
    saved = json.loads((tmp_path / "meta/evaluation.json").read_text())
    assert saved["deployment"]["rtc"] == rtc.settings.to_dict()
    rtc.close()


def test_cli_wraps_checkpoint_and_rejects_rtc_options_without_enable(tmp_path):
    from unittest.mock import patch

    from alohamini.cli import main

    (tmp_path / "policy.json").write_text("{}")
    policy = SimpleNamespace(fps=30, task="pick", validate_rtc=Mock(return_value=50))
    argv = [
        "evaluate", "--host", "pi", "--robot_model", "alohamini2pro",
        "--policy.path", str(tmp_path),
    ]
    with (
        patch("alohamini.learning.policy.NativePolicy", return_value=policy),
        patch("alohamini.apps.evaluation.evaluate") as evaluate,
    ):
        assert main([*argv, "--rtc", "--rtc.replan_steps", "8"]) == 0
        wrapper = evaluate.call_args.kwargs["policy_factory"]()
        assert isinstance(wrapper, RTCPolicy)
        assert wrapper.settings.replan_steps == 8 and wrapper.settings.enabled
        wrapper.close()
        evaluate.reset_mock()
        assert main([*argv, "--rtc.replan_steps", "8"]) == 1
        evaluate.assert_not_called()


def test_unsupported_policy_and_short_horizon_are_rejected_before_execution():
    from alohamini.learning.policy import NativePolicy

    policy = NativePolicy.__new__(NativePolicy)
    policy.manifest = {"kind": "act"}
    with pytest.raises(ValueError, match="pi05 and smolvla"):
        RTCPolicy(policy, RTCConfig(enabled=True))
    policy.manifest = {"kind": "smolvla"}
    policy.config = SimpleNamespace(chunk_size=8)
    with pytest.raises(ValueError, match="chunk must exceed"):
        RTCPolicy(policy, RTCConfig(enabled=True))

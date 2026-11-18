"""Validate MPS device and shared trainer contracts without requiring a GPU."""

from unittest.mock import Mock

import pytest
import torch

from alohamini.learning.execution import Execution, validate_local_workers
from alohamini.learning.loading import make_loader
from alohamini.learning.metrics import MetricAccumulator, MetricSpec, loss_count
from alohamini.learning.precision import reduction_device, tf32_context
from alohamini.learning.training_state import restore_rng, rng_state


def test_mps_launch_rejects_invalid_device_and_distributed_before_run_files(tmp_path, monkeypatch):
    from alohamini.learning.train import launch_training

    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="MPS unavailable"):
        launch_training({"device": "mps", "run_name": "unavailable"})
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    with pytest.raises(ValueError, match="num_processes=1"):
        validate_local_workers(2, "mps")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="WORLD_SIZE=1"):
        Execution("mps")
    assert not (tmp_path / "logs/training").exists()


def test_mps_accounting_preserves_double_precision_on_cpu():
    spec = MetricSpec("mean", "loss", "samples", loss_count("_weight"))
    metrics = MetricAccumulator([spec], device="mps")
    assert metrics.totals.device.type == "cpu" and metrics.totals.dtype == torch.float64
    metrics.add({"loss": torch.tensor(0.5)}, {}, {"_weight": 2**25 + 1})
    assert metrics.result()["metric_totals"]["mean"] == {
        "sum": (2**25 + 1) / 2, "count": 2**25 + 1,
    }
    assert reduction_device("cuda:1") == torch.device("cuda:1")
    precise = MetricAccumulator([spec], device="mps")
    precise.add({"loss": 1.0 + 2**-30}, {}, {"_weight": 1})
    assert precise.result()["metrics"]["mean"] == 1.0 + 2**-30


def test_mps_rng_round_trip_and_old_checkpoint_compatibility(monkeypatch):
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: torch.tensor([1, 2], dtype=torch.uint8))
    saved = rng_state("mps")
    restored = Mock()
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "set_rng_state", restored)
    restore_rng(saved)
    torch.testing.assert_close(restored.call_args.args[0], saved["mps"])
    restore_rng({k: v for k, v in saved.items() if k != "mps"})
    assert restored.call_count == 1


def test_mps_loader_rejects_explicit_fork_and_keeps_zero_workers():
    with pytest.raises(ValueError, match="spawn or forkserver"):
        make_loader([1], device="mps", num_workers=1, multiprocessing_context="fork")
    loader = make_loader([1], device="mps", num_workers=0)
    assert not loader.pin_memory and loader.multiprocessing_context is None


def test_tf32_configuration_is_noop_without_cuda_build(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda, "is_built", lambda: False)
    with tf32_context(True):
        assert torch.ones(1).item() == 1


@pytest.mark.parametrize("precision,mode", [("none", "no"), ("float16", "fp16"),
                                           ("bfloat16", "bf16")])
def test_mps_uses_accelerate_precision_without_rewriting_settings(monkeypatch, precision, mode):
    from types import SimpleNamespace

    from accelerate.utils import DistributedType

    import alohamini.learning.execution as execution

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_macos_or_newer", lambda *a: True)
    accelerator = Mock(return_value=SimpleNamespace(
        distributed_type=DistributedType.NO, device=torch.device("mps"),
        num_processes=1, process_index=0, scaler=Mock() if precision == "float16" else None,
        free_memory=lambda: None,
    ))
    monkeypatch.setattr(execution, "Accelerator", accelerator)
    runtime = Execution("mps", precision)
    assert runtime.precision == precision
    assert accelerator.call_args.kwargs["mixed_precision"] == mode
    assert accelerator.call_args.kwargs["gradient_accumulation_plugin"].num_steps == 1
    assert (runtime.scaler is not None) == (precision == "float16")
    runtime.close()

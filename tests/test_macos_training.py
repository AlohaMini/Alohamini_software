"""Run real Metal accumulation, AMP, validation and checkpoint resume when available."""

import pytest
import torch
from test_native_learning import recording as recording
from test_training import test_periodic_checkpoint_resume_exact_and_no_data_mutation as resume_case


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires a Mac with MPS access")
@pytest.mark.parametrize("kind", ["act", "am_act"])
@pytest.mark.parametrize("precision", ["none", "bfloat16", "float16"])
def test_mps_trainer_checkpoint_resume(recording, tmp_path, monkeypatch, precision, kind):
    resume_case(recording, tmp_path, monkeypatch, "mps", precision, 1, 2, 0, kind=kind)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires a Mac with MPS access")
def test_mps_trainer_workers_and_ema_resume(recording, tmp_path, monkeypatch):
    resume_case(recording, tmp_path, monkeypatch, "mps", "none", 1, 2, 2, kind="act",
                ema_settings={"enable": True, "decay": 0.9})

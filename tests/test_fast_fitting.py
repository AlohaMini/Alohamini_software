"""Native FAST fitting preserves coefficients and validates held-out chunks."""

import json

import numpy as np
import pytest

from alohamini.policies.pi05.fast_fitting import fit_fast, validate_fast


def test_fit_and_validate_dct_bpe_without_remote_code(tmp_path):
    generator = np.random.default_rng(2)
    actions = [generator.uniform(-1, 1, (8, 3)).astype(np.float32) for _ in range(100)]
    config = fit_fast(actions[:80], tmp_path, scale=10, vocab_size=1024)
    report = validate_fast(tmp_path, actions[80:], max_tokens=256)
    assert config["vocab_size"] <= 1024 and config["time_horizon"] == 8
    assert report["reconstruction_rmse"] < 0.04
    assert report["validation_chunks"] == 20
    assert json.loads((tmp_path / "processor_config.json").read_text())["scale"] == 10


def test_fitting_and_validation_reject_unrepresentable_or_truncated_chunks(tmp_path):
    with pytest.raises(ValueError, match="alphabet"):
        fit_fast([np.array([[-1000.0], [1000.0]], np.float32)], tmp_path, vocab_size=256)
    generator = np.random.default_rng(7)
    actions = [generator.normal(size=(16, 3)).astype(np.float32) for _ in range(20)]
    fit_fast(actions, tmp_path, vocab_size=1024)
    with pytest.raises(ValueError, match="exceed.*max_action_tokens"):
        validate_fast(tmp_path, actions, max_tokens=1)


def test_unseen_held_out_coefficient_characters_are_not_dropped(tmp_path):
    chunks = [
        np.repeat(np.asarray(values, np.float32)[:, None], 18, axis=1)
        for values in ([-1, -1 / 3], [-1 / 3, 1 / 3], [1, 1])
    ]
    held_out = np.repeat(np.array([[1 / 3], [1]], np.float32), 18, axis=1)
    fit_fast(chunks, tmp_path, vocab_size=512)
    report = validate_fast(tmp_path, [held_out], max_tokens=256)
    assert report["reconstruction_rmse"] < 0.04

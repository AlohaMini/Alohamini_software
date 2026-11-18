"""Temporal statistics retain row offsets, exclude padding and decode chunks before queuing."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.processor import Processor, quantile_statistics, validate_statistics
from alohamini.learning.window_statistics import window_statistics


def test_window_statistics_fit_real_rows_without_images_and_invert_execution_prefix(
    recording, monkeypatch
):
    samples = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="none",
        image_size=(32, 32),
        delta_indices={"action": [0, 1, 2]},
    )
    original = samples._value

    def numeric_only(index, key):
        assert not key.startswith("observation.images.")
        return original(index, key)

    monkeypatch.setattr(samples, "_value", numeric_only)
    stats = window_statistics(
        samples, {"action": [0, 1, 2]}, base=quantile_statistics(samples), batch_size=2
    )
    np.testing.assert_allclose(np.array(stats["action"]["mean"])[:, 0], [1.5, 2.0, 2.5])
    np.testing.assert_allclose(np.array(stats["action"]["count"])[:, 0], [4, 3, 2])
    processor = Processor(
        stats, modes={"action": "MEAN_STD"}, normalization_windows={"action": [0, 1, 2]}
    )
    target = torch.stack([samples.numeric_sample(i, ["action"])["action"] for i in [0, 1]])
    normalized = processor({"action": target})["action"]
    assert processor.decode_before_queue
    torch.testing.assert_close(processor.action(normalized[:, :2]), target[:, :2])
    with pytest.raises(ValueError, match="batched chunk"):
        processor.action(normalized[:, 0])
    expected = {k: (v.type, v.shape) for k, v in samples.input_features.items()}
    manifest = dict(config=dict(normalization_windows=dict(action=[0, 1, 2])))
    validate_statistics(stats, expected, list(range(18)), manifest=manifest)
    with pytest.raises(ValueError, match="offsets"):
        validate_statistics(
            stats,
            expected,
            list(range(18)),
            manifest=dict(config=dict(normalization_windows=dict(action=[-1, 0, 1]))),
        )


def test_temporal_quantile_prefix_preserves_outliers_and_rejects_unseen_offsets(recording):
    samples = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="none",
        image_size=(32, 32),
        delta_indices={"action": [0, 1, 2]},
    )
    stats = window_statistics(samples, {"action": [0, 1, 2]}, base=quantile_statistics(samples))
    config = SimpleNamespace(
        input_features=samples.input_features,
        output_features=samples.output_features,
        normalization_mapping=dict(VISUAL="IDENTITY", ACTION="QUANTILES"),
        normalization_windows=dict(action=[0, 1, 2]),
    )
    processor = Processor.from_config(config, stats)
    physical = torch.full((1, 3, 18), 100.0)
    normalized = processor({"action": physical})["action"]
    assert normalized.min() > 1
    torch.testing.assert_close(processor.action(normalized[:, :1]), physical[:, :1])
    samples.delta_indices["action"] = [0, 10]
    with pytest.raises(ValueError, match="no real training targets"):
        window_statistics(samples, {"action": [0, 10]}, base=quantile_statistics(samples))

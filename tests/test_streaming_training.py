"""The bounded data backend must preserve actual checkpoint continuation."""

import pytest
import test_training as training_tests
from test_native_learning import recording as recording


@pytest.mark.parametrize("processes,workers", [(1, 2), (2, 0)])
def test_streaming_checkpoint_resume_is_exact(recording, tmp_path, monkeypatch, processes, workers):
    original = training_tests.launch_training

    def launch(settings):
        return original(
            {
                **settings,
                "dataset_streaming": True,
                "streaming_episode_pool_size": 2,
                "dataset_cache_size_mb": 1,
                "dataset_cache_block_rows": 2,
                "dataset_cache_dir": str(tmp_path / "cache"),
                # Four selected frames leave a nonzero epoch offset after step 1
                # for both one-rank and two-rank runs.
                "batch_size": 1,
            }
        )

    monkeypatch.setattr(training_tests, "launch_training", launch)
    training_tests.test_periodic_checkpoint_resume_exact_and_no_data_mutation(
        recording,
        tmp_path,
        monkeypatch,
        device="cpu",
        precision="none",
        process_count=processes,
        accumulation=1,
        workers=workers,
        kind="act",
    )

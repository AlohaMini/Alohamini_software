"""Resume opens the committed index; statistical reconstruction waits for saving."""

import contextlib
import json
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import pytest
from test_recording_storage import add_episode, create, metadata, rows

from alohamini.datasets.lerobotv3 import recording_statistics, recover_recording
from alohamini.datasets.record import LocalDataset


def resume(root):
    return LocalDataset(root, fps=30, task="pick", robot_metadata=metadata(), resume=True)


def test_resume_and_empty_close_do_not_decode_or_restore_statistics(tmp_path):
    root = create(tmp_path / "record", episodes=2, cameras=("forward",))
    with (
        patch("av.open", side_effect=AssertionError("Resume opened a video")),
        patch(
            "alohamini.datasets.lerobot_tools.IntegrityChecker.run",
            side_effect=AssertionError("Resume performed a full dataset check"),
        ),
        patch(
            "alohamini.datasets.lerobotv3.recording_statistics",
            side_effect=AssertionError("Resume scanned historical statistics"),
        ),
        contextlib.closing(resume(root)) as dataset,
    ):
        assert (dataset.num_episodes, dataset.total_frames) == (2, 6)
        assert dataset._v3_stats is None
    assert not (root / ".recording").exists()


def test_discard_does_not_restore_stats_and_first_save_restores_exactly_once(tmp_path):
    root = create(tmp_path / "record", episodes=1, cameras=("forward",))
    original_open = av.open

    def no_historical_video(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).is_relative_to(root / "videos"):
            raise AssertionError("Resume decoded a historical video with saved image statistics")
        return original_open(path, *args, **kwargs)

    with (
        patch("av.open", side_effect=no_historical_video),
        patch(
            "alohamini.datasets.lerobotv3.recording_statistics", wraps=recording_statistics
        ) as restore,
        contextlib.closing(resume(root)) as dataset,
    ):
        restore.assert_not_called()
        add_episode(dataset, 2)
        dataset.discard_episode()
        restore.assert_not_called()
        add_episode(dataset, 2)
        dataset.save_episode()
        restore.assert_called_once_with(root)
        assert dataset.last_save_timings["resume_statistics"] >= 0
        add_episode(dataset, 4)
        dataset.save_episode()
        restore.assert_called_once_with(root)

    actions = np.asarray([row["action"] for row in rows(root)], dtype=np.float64)
    stats = json.loads((root / "meta/stats.json").read_text())["action"]
    assert stats["count"] == [9]
    np.testing.assert_allclose(stats["mean"], actions.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(stats["std"], actions.std(axis=0), atol=1e-12)
    for q in (0.01, 0.1, 0.5, 0.9, 0.99):
        np.testing.assert_allclose(
            stats[f"q{int(q * 100):02d}"], np.quantile(actions, q, axis=0), atol=1e-12
        )


def test_failed_lazy_statistics_retains_episode_for_recovery(tmp_path):
    root = create(tmp_path / "record", episodes=1, cameras=("forward",))
    committed_stats = (root / "meta/stats.json").read_bytes()
    dataset = resume(root)
    add_episode(dataset, 2)
    with patch(
        "alohamini.datasets.lerobotv3.recording_statistics", side_effect=OSError("read failed")
    ) as restore:
        with pytest.raises(OSError, match="read failed"):
            dataset.save_episode()
        dataset.close()
        restore.assert_called_once_with(root)
    assert dataset._failed and dataset._writer.save_failed
    assert (root / "meta/stats.json").read_bytes() == committed_stats
    assert (root / ".recording/episodes/episode_000001/frames.parquet").exists()
    with pytest.raises(RuntimeError, match="Interrupted recording retained"):
        LocalDataset(
            root,
            fps=30,
            task="pick",
            robot_metadata=metadata(),
            resume=True,
            discard_interrupted=False,
        )
    recovered = tmp_path / "recovered"
    assert recover_recording(root, recovered)["valid"]
    assert len(rows(recovered)) == 5

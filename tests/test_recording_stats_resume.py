"""Exact saved statistics avoid history reconstruction across recorder restarts."""

import contextlib
import json
from pathlib import Path
from unittest.mock import Mock, patch

import av
import numpy as np
import pyarrow.parquet as pq
from test_recording_storage import add_episode, create, metadata, rows

from alohamini.datasets.lerobotv3 import recording_statistics
from alohamini.datasets.record import LocalDataset


@contextlib.contextmanager
def no_history_content(root):
    original_parquet, original_read, original_video = pq.ParquetFile, pq.read_table, av.open

    def historical(path, directory):
        return isinstance(path, (str, Path)) and Path(path).is_relative_to(root / directory)

    def parquet(path, *args, **kwargs):
        result = original_parquet(path, *args, **kwargs)
        if historical(path, "data"):
            for name in ("read", "read_row_group", "read_row_groups", "iter_batches"):
                setattr(result, name, Mock(side_effect=AssertionError("Historical numeric read")))
        return result

    def read(path, *args, **kwargs):
        if historical(path, "data"):
            raise AssertionError("Historical numeric read")
        return original_read(path, *args, **kwargs)

    def video(path, *args, **kwargs):
        if historical(path, "videos"):
            raise AssertionError("Historical video decode")
        return original_video(path, *args, **kwargs)

    with (
        patch("pyarrow.parquet.ParquetFile", side_effect=parquet),
        patch("pyarrow.parquet.read_table", side_effect=read),
        patch("av.open", side_effect=video),
    ):
        yield


def resume(root):
    return LocalDataset(root, fps=30, task="pick", robot_metadata=metadata(), resume=True)


def assert_exact_numeric_stats(root):
    values = rows(root)
    info = json.loads((root / "meta/info.json").read_text())
    actual = json.loads((root / "meta/stats.json").read_text())
    for key, feature in info["features"].items():
        if feature["dtype"] == "video":
            continue
        vectors = np.asarray([row[key] for row in values], np.float64).reshape(len(values), -1)
        assert actual[key]["count"] == [len(values)]
        for name, expected in (
            ("min", vectors.min(axis=0)),
            ("max", vectors.max(axis=0)),
            ("mean", vectors.mean(axis=0)),
            ("std", vectors.std(axis=0)),
        ):
            np.testing.assert_allclose(actual[key][name], expected, atol=1e-12)
        for q in (0.01, 0.1, 0.5, 0.9, 0.99):
            np.testing.assert_array_equal(
                actual[key][f"q{int(q * 100):02d}"], np.quantile(vectors, q, axis=0)
            )


def test_each_resume_first_save_uses_exact_saved_state_without_history_reads(tmp_path):
    root = create(tmp_path / "record", episodes=2, cameras=("forward",))
    for length in (5, 7):
        with no_history_content(root), contextlib.closing(resume(root)) as dataset:
            add_episode(dataset, length)
            dataset.save_episode()
        assert_exact_numeric_stats(root)


def test_old_dataset_without_state_rebuilds_once_then_resumes_without_history_reads(tmp_path):
    root = create(tmp_path / "record", episodes=1, cameras=("forward",))
    cache = root / "meta/recording_stats.npz"
    cache.unlink()
    with (
        patch(
            "alohamini.datasets.lerobotv3.recording_statistics", wraps=recording_statistics
        ) as restore,
        contextlib.closing(resume(root)) as dataset,
    ):
        restore.assert_not_called()
        add_episode(dataset, 5)
        dataset.save_episode()
        restore.assert_called_once_with(root)
    assert cache.is_file()
    with no_history_content(root), contextlib.closing(resume(root)) as dataset:
        add_episode(dataset, 2)
        dataset.save_episode()
    assert_exact_numeric_stats(root)


def test_changed_historical_values_invalidate_saved_state(tmp_path):
    root = create(tmp_path / "record", episodes=1, cameras=("forward",))
    path = next((root / "data").rglob("*.parquet"))
    table = pq.read_table(path)
    edited = table.to_pylist()
    edited[0]["action"][0] = 500
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(edited, schema=table.schema), path)
    with contextlib.closing(resume(root)) as dataset:
        add_episode(dataset, 2)
        dataset.save_episode()
    assert_exact_numeric_stats(root)


def test_optional_state_write_failure_keeps_committed_episode_and_exact_stats(tmp_path, caplog):
    root = create(tmp_path / "record", episodes=1, cameras=("forward",))
    with contextlib.closing(resume(root)) as dataset:
        add_episode(dataset, 2)
        with patch(
            "alohamini.datasets.recording_stats_cache.save_recording_statistics",
            side_effect=OSError("cache disk full"),
        ):
            dataset.save_episode()
        assert not dataset._failed
        assert dataset.num_episodes == 2
    assert "Episode saved, but recording statistics state" in caplog.text
    assert_exact_numeric_stats(root)
    # The previous cache belongs to the old commit and must not be reused.
    with contextlib.closing(resume(root)) as dataset:
        add_episode(dataset, 4)
        dataset.save_episode()
    assert_exact_numeric_stats(root)

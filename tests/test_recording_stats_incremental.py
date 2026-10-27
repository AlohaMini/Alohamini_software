"""Saved distributions grow by episode without exporting or rewriting history."""

import contextlib
import json
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import pyarrow.parquet as pq
import pytest
from test_recording_stats_resume import no_history_content, resume
from test_recording_storage import add_episode, create, metadata, rows

from alohamini.datasets import recording_stats_cache as cache
from alohamini.datasets.exact_distribution import ExactDistributionStats
from alohamini.datasets.image_statistics import ImageStatistics
from alohamini.datasets.record import LocalDataset


def manifest(root):
    with np.load(root / cache._CACHE, allow_pickle=False) as archive:
        return json.loads(archive["manifest"].tobytes())


def resume_without_history(root):
    with no_history_content(root):
        return resume(root)


def immutable_archives(root):
    directory = root / "meta/recording_stats"
    return {
        path: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in directory.glob("*.npz")
    }


def assert_archives_unchanged(previous):
    for path, signature in previous.items():
        assert (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes()) == signature


def assert_all_statistics(root):
    values = rows(root)
    info = json.loads((root / "meta/info.json").read_text())
    actual = json.loads((root / "meta/stats.json").read_text())
    images = ImageStatistics()
    for path in (root / "meta/episodes").rglob("*.parquet"):
        for episode in pq.read_table(path, columns=["image_statistics"]).to_pylist():
            images.merge(ImageStatistics(json.loads(episode["image_statistics"])))
    image_stats = images.result()
    assert set(actual) == set(info["features"])
    quantiles = (0.01, 0.1, 0.5, 0.9, 0.99)
    expected_fields = {"count", "min", "max", "mean", "std"} | {
        f"q{int(q * 100):02d}" for q in quantiles
    }
    for key, feature in info["features"].items():
        assert set(actual[key]) == expected_fields
        if feature["dtype"] in ("image", "video"):
            for name, expected in image_stats[key].items():
                np.testing.assert_array_equal(actual[key][name], expected)
            continue
        vectors = np.asarray([row[key] for row in values], np.float64).reshape(len(values), -1)
        assert actual[key]["count"] == [len(values)]
        for name, expected in (
            ("min", vectors.min(axis=0)),
            ("max", vectors.max(axis=0)),
            ("mean", vectors.mean(axis=0)),
            ("std", vectors.std(axis=0)),
        ):
            np.testing.assert_allclose(actual[key][name], expected, rtol=1e-12, atol=1e-12)
        for q in quantiles:
            np.testing.assert_array_equal(
                actual[key][f"q{int(q * 100):02d}"], np.quantile(vectors, q, axis=0)
            )


def save_delta(root, dataset, count):
    previous = immutable_archives(root)
    old_manifest = manifest(root)
    add_episode(dataset, count)
    with no_history_content(root):
        dataset.save_episode()
    assert_archives_unchanged(previous)
    current = manifest(root)
    assert current["version"] == 2
    assert current["frames"] == old_manifest["frames"] + count
    assert current["parent"]["frames"] == old_manifest["frames"]
    parent = root / current["parent"]["path"]
    assert parent.is_file() and parent.parent == root / "meta/recording_stats"
    with np.load(root / cache._CACHE, allow_pickle=False) as archive:
        # Cumulative moments are small; distributions contain only this new episode.
        index = current["trackers"]["action"]
        assert archive[f"tracker_{index}__count"].item() == current["frames"]
        assert archive[f"tracker_{index}__counts_0"].sum() == count
    assert_all_statistics(root)


def test_continuous_saves_and_v2_resume_keep_exact_stats_without_full_exports(tmp_path):
    root = tmp_path / "record"
    with contextlib.closing(
        LocalDataset(root, fps=30, task="pick", robot_metadata=metadata())
    ) as dataset:
        add_episode(dataset, 3)
        dataset.save_episode()
        assert manifest(root)["version"] == 1
        assert_all_statistics(root)
        with patch.object(
            ExactDistributionStats,
            "export_state",
            side_effect=AssertionError("Normal saves must not export the full distribution"),
        ) as full_export:
            for count in (5, 1):
                save_delta(root, dataset, count)
            full_export.assert_not_called()

    for count in (7, 2):
        with (
            patch.object(
                ExactDistributionStats,
                "export_state",
                side_effect=AssertionError("Resumed saves must not export history"),
            ) as full_export,
            contextlib.closing(resume_without_history(root)) as dataset,
        ):
            save_delta(root, dataset, count)
            full_export.assert_not_called()


def test_v1_checkpoint_is_used_as_parent_without_reexporting_it(tmp_path):
    root = create(tmp_path / "record", episodes=1, cameras=("forward",))
    old = root / cache._CACHE
    original = (old.stat().st_ino, old.stat().st_mtime_ns, old.read_bytes())
    assert manifest(root)["version"] == 1
    with (
        patch.object(
            ExactDistributionStats,
            "export_state",
            side_effect=AssertionError("A valid v1 checkpoint must not be reexported"),
        ) as full_export,
        contextlib.closing(resume_without_history(root)) as dataset,
    ):
        save_delta(root, dataset, 5)
        parent = root / manifest(root)["parent"]["path"]
        assert (parent.stat().st_ino, parent.stat().st_mtime_ns, parent.read_bytes()) == original
        full_export.assert_not_called()


@pytest.mark.parametrize("failure", ["archive_write", "latest_publish"])
@pytest.mark.parametrize("restart", [False, True])
def test_failed_incremental_checkpoint_preserves_episode_and_rebases_without_lost_frames(
    tmp_path, caplog, failure, restart
):
    root = create(tmp_path / "record", episodes=2, cameras=("forward",))
    original_replace, original_video = cache.os.replace, av.open

    def reject_latest(source, target):
        if Path(target) == root / cache._CACHE:
            raise OSError("interrupted incremental cache publication")
        return original_replace(source, target)

    def no_old_video(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).is_relative_to(root / "videos"):
            raise AssertionError("Historical video must not be decoded for cache repair")
        return original_video(path, *args, **kwargs)

    with contextlib.closing(resume(root)) as dataset:
        previous = immutable_archives(root)
        latest = (root / cache._CACHE).read_bytes()
        add_episode(dataset, 2)
        target = "np.savez_compressed" if failure == "archive_write" else "os.replace"
        replacement = (
            {"side_effect": OSError("interrupted incremental cache write")}
            if failure == "archive_write"
            else {"side_effect": reject_latest}
        )
        with patch(f"alohamini.datasets.recording_stats_cache.{target}", **replacement):
            dataset.save_episode()
        assert not dataset._failed and dataset.num_episodes == 3
        assert dataset.total_frames == 8
        assert (root / cache._CACHE).read_bytes() == latest
        assert_archives_unchanged(previous)
        assert "Episode saved, but recording statistics state" in caplog.text
        assert_all_statistics(root)

        # Repair both in the same recorder and after a restart. The formally
        # committed episode whose cache write failed must never be omitted.
        if restart:
            dataset.close()
        with patch("av.open", side_effect=no_old_video):
            repair = (
                contextlib.closing(resume(root)) if restart else contextlib.nullcontext(dataset)
            )
            with repair as recording:
                add_episode(recording, 4)
                recording.save_episode()
                assert recording.num_episodes == 4 and recording.total_frames == 12
        assert manifest(root)["frames"] == 12
        assert_archives_unchanged(previous)
        assert_all_statistics(root)

    with (
        patch.object(
            ExactDistributionStats,
            "export_state",
            side_effect=AssertionError("Repaired checkpoints must resume incrementally"),
        ) as full_export,
        contextlib.closing(resume_without_history(root)) as dataset,
    ):
        save_delta(root, dataset, 1)
        full_export.assert_not_called()

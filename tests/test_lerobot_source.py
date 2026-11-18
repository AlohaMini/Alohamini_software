"""Shared-shard reads preserve physical image offsets and requested episode order."""

import io
import json
import pickle
import subprocess
import sys
from tempfile import TemporaryFile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from alohamini.learning.lerobot import LeRobotSource


@pytest.fixture
def source(tmp_path):
    (tmp_path / "meta/episodes").mkdir(parents=True)
    (tmp_path / "meta/safety").mkdir()
    features = {
        "action": {"dtype": "float32", "shape": [1]},
        "observation.images.forward": {"dtype": "image"},
    }
    info = {
        "features": features,
        "fps": 30,
        "robot_type": "alohamini2pro",
        "data_path": "data_{file_index}.parquet",
    }
    original = {"features": features, "fps": 30, "robot_metadata": {"robot_model": "alohamini2pro"}}
    (tmp_path / "meta/alohamini.json").write_text(json.dumps({"source_info": original}))
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame({"task_index": [0]}, index=["pick"])),
        tmp_path / "meta/tasks.parquet",
    )
    metadata = []
    for ep in range(4):
        metadata.append(
            {"episode_index": ep, "data/chunk_index": 0, "data/file_index": ep % 2, "length": 2}
        )
        (tmp_path / f"meta/safety/episode_{ep:06d}.jsonl").write_text("")
    pq.write_table(pa.Table.from_pylist(metadata), tmp_path / "meta/episodes/episodes.parquet")
    for shard in range(2):
        rows = []
        for frame in range(2):
            for ep in (shard, shard + 2):
                value = ep * 20 + frame
                image = io.BytesIO()
                Image.fromarray(np.full((2, 3, 3), value, np.uint8)).save(image, format="PNG")
                rows.append(
                    {
                        "episode_index": ep,
                        "frame_index": frame,
                        "action": [value],
                        "task_index": 0,
                        "observation.images.forward": {"bytes": image.getvalue()},
                    }
                )
        pq.write_table(
            pa.Table.from_pylist(rows), tmp_path / f"data_{shard}.parquet", row_group_size=3
        )
    return LeRobotSource(tmp_path, info)


def test_bulk_reads_each_shard_once_and_keeps_image_offsets(source):
    opened = []
    parquet = pq.ParquetFile

    def tracked(path):
        file = parquet(path)
        spy = MagicMock(wraps=file)
        spy.num_row_groups = file.num_row_groups
        spy.__enter__.return_value = spy
        spy.__exit__.side_effect = lambda *args: file.close()
        opened.append(spy)
        return spy

    # Alternate physical shards and request reverse episode order.
    with patch("alohamini.learning.lerobot.pq.ParquetFile", side_effect=tracked):
        results = list(source.read_episodes([3, 2, 1, 0], ["action", "task"]))
    assert len(opened) == 2
    assert [file.read_row_group.call_count for file in opened] == [2, 2]
    for episode, (rows, safety, paths) in zip([3, 2, 1, 0], results, strict=True):
        assert [row["action"] for row in rows] == [[episode * 20], [episode * 20 + 1]]
        assert [row["task"] for row in rows] == ["pick", "pick"]
        assert safety.name == f"episode_{episode:06d}.jsonl"
        assert source.root / "meta/tasks.parquet" in paths
        for frame, row in enumerate(rows):
            np.testing.assert_array_equal(
                source.image(row["observation.images.forward"], "forward"),
                np.full((2, 3, 3), episode * 20 + frame, np.uint8),
            )


def test_duplicate_requests_own_rows_and_new_traversal_reads_fresh_files(source):
    results = source.read_episodes([0, 0], ["action"])
    first = next(results)[0]
    first[0]["action"][0] = -1
    assert next(results)[0][0]["action"] == [0]
    path = source.root / "data_0.parquet"
    table = pq.read_table(path).to_pylist()
    table[0]["action"] = [99]
    pq.write_table(pa.Table.from_pylist(table), path)
    assert source.read_episode(0, ["action"])[0][0]["action"] == [99]


def test_bulk_rejects_missing_sidecar_and_reordered_frames(source):
    safety = source.root / "meta/safety/episode_000002.jsonl"
    safety.unlink()
    with pytest.raises(ValueError, match="sidecar"):
        list(source.read_episodes([0, 2], ["action"]))
    safety.touch()
    path = source.root / "data_0.parquet"
    table = pq.read_table(path).to_pylist()
    table[0]["frame_index"] = 1
    with pytest.raises(ValueError, match="row order or boundary"):
        pq.write_table(pa.Table.from_pylist(table), path)
        list(source.read_episodes([0, 2], ["action"]))


def test_stream_batches_shards_once_and_preserves_physical_image_offsets(source, tmp_path):
    opened = []
    parquet = pq.ParquetFile

    def tracked(path):
        file = parquet(path)
        spy = MagicMock(wraps=file)
        spy.num_row_groups = file.num_row_groups
        spy.__enter__.return_value = spy
        spy.__exit__.side_effect = lambda *args: file.close()
        spy.read_row_group.side_effect = AssertionError("Streaming must use bounded batches")
        opened.append(spy)
        return spy

    cache = tmp_path / "cache"
    with patch("alohamini.learning.lerobot.pq.ParquetFile", side_effect=tracked):
        results = [
            (list(rows), safety, paths)
            for rows, safety, paths in source.stream_episodes(
                [3, 2, 1, 0],
                ["action", "task", "observation.images.forward"],
                batch_size=1,
                cache_dir=cache,
            )
        ]
    assert len(opened) == 2
    assert [file.iter_batches.call_count for file in opened] == [2, 2]
    for file in opened:
        file.read_row_group.assert_not_called()
        for call in file.iter_batches.call_args_list:
            assert call.kwargs["batch_size"] == 1
            assert not any(key.startswith("observation.images.") for key in call.kwargs["columns"])
    assert not list(cache.iterdir())
    for episode, (rows, safety, paths) in zip([3, 2, 1, 0], results, strict=True):
        assert [row["action"] for row in rows] == [[episode * 20], [episode * 20 + 1]]
        assert [row["task"] for row in rows] == ["pick", "pick"]
        assert safety.name == f"episode_{episode:06d}.jsonl"
        assert source.root / "meta/tasks.parquet" in paths
        for frame, row in enumerate(rows):
            np.testing.assert_array_equal(
                source.image(row["observation.images.forward"], "forward"),
                np.full((2, 3, 3), episode * 20 + frame, np.uint8),
            )


def test_stream_duplicate_requests_own_rows_and_rescan_only_on_new_traversal(source, tmp_path):
    cache = tmp_path / "cache"
    traversal = source.stream_episodes([0, 0], ["action"], cache_dir=cache)
    first = list(next(traversal)[0])
    first[0]["action"][0] = -1
    assert list(next(traversal)[0])[0]["action"] == [0]
    traversal.close()
    assert not list(cache.iterdir())
    path = source.root / "data_0.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["action"] = [99]
    pq.write_table(pa.Table.from_pylist(rows), path)
    traversal = source.stream_episodes([0], ["action"], cache_dir=cache)
    try:
        assert list(next(traversal)[0])[0]["action"] == [99]
    finally:
        traversal.close()
    assert not list(cache.iterdir())


@pytest.mark.parametrize("failure", ["sidecar", "order", "short", "long"])
def test_stream_rejects_bad_episode_bounds_and_cleans_private_spool(source, tmp_path, failure):
    cache = tmp_path / "cache"
    if failure == "sidecar":
        (source.root / "meta/safety/episode_000000.jsonl").unlink()
    else:
        path = source.root / "data_0.parquet"
        rows = pq.read_table(path).to_pylist()
        if failure == "order":
            rows[0]["frame_index"], rows[2]["frame_index"] = 1, 0
        elif failure == "short":
            del rows[2]
        else:
            rows.append({**rows[2], "frame_index": 2})
        pq.write_table(pa.Table.from_pylist(rows), path)
    match = "sidecar" if failure == "sidecar" else "row order or boundary"
    traversal = source.stream_episodes([0], ["action"], batch_size=1, cache_dir=cache)
    try:
        with pytest.raises(ValueError, match=match):
            for rows, _, _ in traversal:
                list(rows)
    finally:
        traversal.close()
    assert not list(cache.iterdir())


def test_stream_early_close_releases_disk_spool_and_default_uses_workspace(
    source, tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(workspace))
    opened = []

    def tracked(*args, **kwargs):
        file = TemporaryFile(*args, **kwargs)
        opened.append(file)
        return file

    with patch("alohamini.learning.lerobot.TemporaryFile", side_effect=tracked):
        traversal = source.stream_episodes([0, 2], ["action"], batch_size=1)
        rows = next(traversal)[0]
        assert next(rows)["action"] == [0]
        assert len(opened) == 1 and not opened[0].closed
        traversal.close()
    cache = workspace / "cache/datasets"
    assert opened[0].closed
    assert not list(cache.iterdir())


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_stream_requires_a_positive_batch_size(source, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        next(source.stream_episodes([0], ["action"], batch_size=batch_size))


def test_source_pickling_drops_arrow_image_cache_and_keeps_lazy_pixels(source):
    reference = source.read_episode(0, ["action"])[0][0]["observation.images.forward"]
    expected = source.image(reference, "forward")
    assert source._image_cache is not None
    restored = pickle.loads(pickle.dumps(source))
    assert restored._image_cache is None
    assert restored._image_cache_key is None
    assert source._image_cache is not None
    np.testing.assert_array_equal(restored.image(reference, "forward"), expected)


def test_stream_imports_after_torch_and_av_without_sqlite(source, tmp_path):
    script = """
import builtins
import json
import sys
from pathlib import Path

import torch
import av

original_import = builtins.__import__
def without_sqlite(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'sqlite3', '_sqlite3'}:
        raise AssertionError('Local row spooling must not import native SQLite')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_sqlite

from alohamini.learning.lerobot import LeRobotSource
root = Path(sys.argv[1])
source = LeRobotSource(root, json.loads((root / 'meta/info.json').read_text()))
results = [
    [row['action'] for row in rows]
    for rows, _, _ in source.stream_episodes(
        [2, 0], ['action'], batch_size=1, cache_dir=sys.argv[2]
    )
]
assert results == [[[40], [41]], [[0], [1]]]
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(source.root), str(tmp_path / "cache")],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr

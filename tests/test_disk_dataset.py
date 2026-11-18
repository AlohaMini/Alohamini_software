"""Disk-backed preparation preserves training values and multiprocessing behavior."""

import hashlib
import json
import pickle

import pytest
import torch
from test_native_learning import recording as recording

from alohamini.datasets.lerobotv3 import export_lerobot
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.data_index import EpisodeLayout
from alohamini.learning.data_store import FileBackedIntSequence, IndexSet, RowStore
from alohamini.learning.execution import RankBatchSampler
from alohamini.learning.loading import make_loader
from alohamini.learning.statistics import _sample_indices_digest
from alohamini.learning.training_state import EpisodeAwareSampler


@pytest.mark.parametrize("v3", [False, True])
def test_disk_backed_samples_keep_actions_padding_records_and_locations(recording, tmp_path, v3):
    root = recording
    if v3:
        root = tmp_path / "v3"
        export_lerobot(recording, root)
    data = AlohaMiniDataset(
        root,
        episodes=[1, 0],
        cameras=[],
        state="none",
        chunk_size=3,
        cache_block_rows=2,
        cache_size_mb=0.1,
        cache_dir=tmp_path / "cache",
    )
    assert isinstance(data.rows, RowStore) and isinstance(data.records, RowStore)
    assert isinstance(data.sample_indices, FileBackedIntSequence)
    assert isinstance(data._used_rows["action"], IndexSet)
    assert data.locations == [(ep, i) for ep in [1, 0] for i in range(4)]
    assert data.segment_starts == [0] * 4 + [4] * 4
    assert data.segment_ends == [4] * 4 + [8] * 4
    for index in range(len(data)):
        ep, frame = data.locations[index]
        expected = [ep * 10 + min(frame + offset, 3) for offset in range(3)]
        actual = data[index]
        assert actual["action"][:, 0].tolist() == expected
        assert actual["action_is_pad"].tolist() == [frame + j >= 4 for j in range(3)]
        assert data.records[index]["safety"]["feedback_valid"]
    assert data.rows.currbytes <= data.rows.cache_bytes
    assert data.records.currbytes <= data.records.cache_bytes
    assert data.statistics()["action"]["mean"] == [6.5] * 18


@pytest.mark.parametrize("method", ["spawn", "forkserver"])
def test_workers_reopen_dataset_without_shipping_row_cache(recording, tmp_path, method):
    data = AlohaMiniDataset(
        recording,
        episodes=[0, 1],
        cameras=[],
        state="none",
        chunk_size=3,
        cache_dir=tmp_path / "cache",
        cache_block_rows=2,
    )
    expected = list(make_loader(data, batch_size=2))
    assert data.rows.currbytes > 0
    restored = pickle.loads(pickle.dumps(data))
    assert restored.rows.currbytes == restored.records.currbytes == 0
    loader = make_loader(
        data,
        batch_size=2,
        num_workers=2,
        multiprocessing_context=method,
        persistent_workers=False,
    )
    actual = list(loader)
    for wanted, received in zip(expected, actual, strict=True):
        for key in wanted:
            torch.testing.assert_close(wanted[key], received[key], rtol=0, atol=0)


@pytest.mark.parametrize("size", [0, 1, 1024, 1025, 2049])
def test_compact_sample_index_digest_matches_existing_statistics_contract(size, tmp_path):
    values = [index * 3 for index in range(size)]
    indices = FileBackedIntSequence(root=tmp_path)
    indices.extend(values)
    expected = hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()
    assert _sample_indices_digest(indices) == expected


def test_episode_layout_handles_empty_spans_negative_indices_and_slices():
    layout = EpisodeLayout([(8, 0, 0), (3, 0, 2), (9, 2, 2), (4, 2, 5)])
    locations, starts, ends = (layout.view(key) for key in ("location", "start", "end"))
    assert locations == [(3, 0), (3, 1), (4, 0), (4, 1), (4, 2)]
    assert locations[-1] == (4, 2) and locations[::-2] == [(4, 2), (4, 0), (3, 0)]
    assert starts == [0, 0, 2, 2, 2] and ends == [2, 2, 5, 5, 5]
    with pytest.raises(IndexError):
        _ = locations[5]


@pytest.mark.parametrize("size", [1, 2, 7, 31])
@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("drop", [False, True])
def test_default_sampler_keeps_original_global_shuffle_and_resume(size, world, drop):
    batch_size, seed = 2, 13
    width = batch_size * world
    count = size // width if drop else (size + width - 1) // width
    if not count:
        return
    for epoch in [0, 1]:
        original = EpisodeAwareSampler([0], [size], shuffle=True, seed=seed)
        original.set_epoch(epoch)
        order = list(original)
        if world > 1 and not drop:
            order = (order * ((count * width + size - 1) // size))[: count * width]
        for offset in [0, count - 1]:
            for rank in range(world):
                expected = [
                    order[i * width + rank * batch_size : i * width + (rank + 1) * batch_size]
                    for i in range(offset, count)
                ]
                sampler = RankBatchSampler(
                    size, batch_size, seed, rank, world, drop, consumed=epoch * count + offset
                )
                assert list(sampler) == expected


def test_row_cache_cannot_write_into_source_dataset(recording):
    with pytest.raises(ValueError, match="outside the source dataset"):
        AlohaMiniDataset(recording, episodes=[0], state="none", cache_dir=recording / "cache")


def test_large_contiguous_and_sparse_used_index_updates_stay_exact(tmp_path):
    used = IndexSet(20000, root=tmp_path)
    used.update(range(1, 15000))
    used.update([0, 2, 2, 16000, 19000])
    assert len(used) == 15002
    assert list(used) == [*range(15000), 16000, 19000]


def test_constructor_failure_releases_cache_while_traceback_is_retained(
    recording, tmp_path, monkeypatch
):
    cache = tmp_path / "cache"

    def fail_append(*args):
        raise OSError("simulated cache write failure")

    monkeypatch.setattr(RowStore, "append", fail_append)
    with pytest.raises(OSError, match="cache write failure") as failure:
        AlohaMiniDataset(recording, episodes=[0], state="none", cache_dir=cache)
    assert failure.value.__traceback__ is not None
    assert list(cache.iterdir()) == []

"""Simulation numeric streaming preserves samples without episode-sized reads."""

from collections.abc import Iterator

import h5py
import numpy as np
import pytest
from test_simulation_dataset import batch, episode

from alohamini.datasets.simulation import SimulationSource
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.streaming import StreamingBatchSampler

_COLUMNS = (
    "action",
    "observation.state",
    "observation.joint_velocity",
    "observation.images.agentview",
    "task",
)


def watch_files(monkeypatch):
    opened = []
    original = h5py.File

    def open_file(*args, **kwargs):
        file = original(*args, **kwargs)
        opened.append(file)
        return file

    monkeypatch.setattr(h5py, "File", open_file)
    return opened


@pytest.mark.parametrize("batch_size", [1, 2, 1024])
def test_simulation_stream_matches_old_rows_types_and_references_with_bounded_reads(
    tmp_path, monkeypatch, batch_size
):
    _, _, rgb = episode(tmp_path / "sim")
    source = SimulationSource(tmp_path / "sim")
    expected, expected_sidecar, expected_paths = next(source.read_episodes([0], _COLUMNS))
    opened = watch_files(monkeypatch)
    reads = []
    original = h5py.Dataset.__getitem__

    def bounded_read(dataset, selection, *args, **kwargs):
        assert not dataset.name.endswith("_rgb"), "Creating image references must not read pixels"
        assert isinstance(selection, slice)
        assert selection.start is not None and selection.stop is not None
        assert 0 < selection.stop - selection.start <= batch_size
        reads.append((dataset.name, selection.start, selection.stop))
        return original(dataset, selection, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", bounded_read)
    stream = source.stream_episodes(
        [0], [*_COLUMNS, "action"], batch_size=batch_size, cache_dir=tmp_path / "unused"
    )
    rows, sidecar, paths = next(stream)
    assert isinstance(rows, Iterator)
    assert not opened, "The episode file must stay closed until rows are consumed"
    assert sidecar is expected_sidecar is None
    assert paths == expected_paths
    actual = list(rows)
    assert len(opened) == 1 and not opened[0].id.valid
    assert len(reads) == 3 * ((len(expected) + batch_size - 1) // batch_size)
    assert len(actual) == len(expected)
    for row, old in zip(actual, expected, strict=True):
        assert row.keys() == old.keys()
        for key in row:
            if isinstance(old[key], np.ndarray):
                np.testing.assert_array_equal(row[key], old[key])
                assert row[key].dtype == old[key].dtype == np.float32
                assert row[key].shape == old[key].shape == (2,)
            else:
                assert row[key] == old[key]
    stream.close()
    # The existing image interface still returns the lossless HDF5 RGB sample.
    monkeypatch.setattr(h5py.Dataset, "__getitem__", original)
    np.testing.assert_array_equal(
        source.image(actual[1]["observation.images.agentview"], "agentview"), rgb[1]
    )
    source._local.files.popitem()[1].close()
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("close_target", ["rows", "outer"])
def test_partial_consumption_closes_hdf5_through_either_iterator(
    tmp_path, monkeypatch, close_target
):
    episode(tmp_path / "sim")
    source = SimulationSource(tmp_path / "sim")
    opened = watch_files(monkeypatch)
    stream = source.stream_episodes([0], _COLUMNS, batch_size=2)
    rows, _, _ = next(stream)
    next(rows)
    assert len(opened) == 1 and opened[0].id.valid
    (rows if close_target == "rows" else stream).close()
    assert not opened[0].id.valid
    assert list(rows) == []
    stream.close()


def test_advancing_outer_stream_closes_previous_partial_episode(tmp_path, monkeypatch):
    episode(tmp_path / "a", reset=0)
    episode(tmp_path / "b", reset=1)
    batch(tmp_path, [("a", True), ("b", True)])
    source = SimulationSource(tmp_path)
    opened = watch_files(monkeypatch)
    stream = source.stream_episodes([1, 0], ["action"], batch_size=1)
    first, _, _ = next(stream)
    assert next(first)["episode_index"] == 1
    second, _, _ = next(stream)
    assert len(opened) == 1 and not opened[0].id.valid
    assert list(first) == []
    assert [row["episode_index"] for row in second] == [0, 0, 0]
    assert len(opened) == 2 and not any(file.id.valid for file in opened)
    assert list(stream) == []


def test_closing_unconsumed_rows_does_not_open_hdf5(tmp_path, monkeypatch):
    episode(tmp_path / "sim")
    source = SimulationSource(tmp_path / "sim")
    opened = watch_files(monkeypatch)
    stream = source.stream_episodes([0], ["action"])
    rows, _, _ = next(stream)
    stream.close()
    assert not opened
    assert list(rows) == []


def test_hdf5_read_failure_closes_file_without_yielding_incomplete_row(tmp_path, monkeypatch):
    episode(tmp_path / "sim")
    source = SimulationSource(tmp_path / "sim")
    opened = watch_files(monkeypatch)
    original = h5py.Dataset.__getitem__

    def fail_second_batch(dataset, selection, *args, **kwargs):
        if selection.start == 2:
            raise OSError("injected HDF5 read failure")
        return original(dataset, selection, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", fail_second_batch)
    stream = source.stream_episodes([0], ["action", "observation.state"], batch_size=2)
    rows, _, _ = next(stream)
    assert next(rows)["frame_index"] == 0
    assert next(rows)["frame_index"] == 1
    with pytest.raises(OSError, match="injected"):
        next(rows)
    assert not any(file.id.valid for file in opened)
    stream.close()


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size_is_rejected_without_opening_episode(tmp_path, monkeypatch, batch_size):
    episode(tmp_path / "sim")
    source = SimulationSource(tmp_path / "sim")
    opened = watch_files(monkeypatch)
    with pytest.raises(ValueError, match="batch_size"):
        next(source.stream_episodes([0], ["action"], batch_size=batch_size))
    assert not opened


def test_unknown_field_is_rejected_without_opening_episode(tmp_path, monkeypatch):
    episode(tmp_path / "sim")
    source = SimulationSource(tmp_path / "sim")
    opened = watch_files(monkeypatch)
    with pytest.raises(ValueError, match="not recorded"):
        next(source.stream_episodes([0], ["action", "unknown"]))
    assert not opened


def test_disk_backed_training_reads_simulation_through_streaming_source(tmp_path, monkeypatch):
    actions, positions, rgb = episode(tmp_path / "sim")

    def forbid_materialization(*args, **kwargs):
        raise AssertionError("The default training loader must stream HDF5 rows")

    monkeypatch.setattr(SimulationSource, "read_episodes", forbid_materialization)
    data = AlohaMiniDataset(
        tmp_path / "sim",
        episodes=[0],
        state="recorded",
        chunk_size=4,
        image_size=(32, 32),
        return_uint8=True,
        include_task=True,
        cache_block_rows=1,
    )
    assert data.records == [{}, {}, {}]
    sampler = StreamingBatchSampler(data, 2, 11, pool_size=1)
    assert sorted(index for indices in sampler for index in indices) == [0, 1, 2]
    first, last = data[0], data[2]
    np.testing.assert_array_equal(first["observation.state"], positions[0].astype(np.float32))
    np.testing.assert_array_equal(first["action"][0], actions[0].astype(np.float32))
    np.testing.assert_array_equal(first["observation.images.agentview"].permute(1, 2, 0), rgb[0])
    assert first["task"] == "place the bowl on the plate"
    assert last["action_is_pad"].tolist() == [False, True, True, True]
    np.testing.assert_array_equal(last["action"], np.tile(actions[-1].astype(np.float32), (4, 1)))

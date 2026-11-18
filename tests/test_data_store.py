"""Private file-backed training data stays bounded and can reopen in workers."""

import gc
import multiprocessing
import os
import pickle
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

from alohamini.learning.data_store import FileBackedIntSequence, IndexSet, RowStore


def _worker_read(connection, rows, indices, used):
    try:
        connection.send(
            {
                "row": rows[-1],
                "indices": list(indices),
                "used": list(used),
                "cache_bytes": rows.currbytes,
                "misses": rows.misses,
                "pid": rows._pid,
            }
        )
    finally:
        rows.close()
        indices.close()
        used.close()
        connection.close()


def _worker_reject_building(connection, *stores):
    errors = []
    for store in stores:
        for _ in range(2):
            try:
                store.finalize()
            except RuntimeError as exc:
                errors.append(str(exc))
        store.close()
    connection.send(errors)
    connection.close()


def test_rows_snapshot_values_views_and_sequence_semantics(tmp_path):
    store = RowStore(block_rows=2, root=tmp_path)
    first = ({"action": [1.0, 2.0]}, {"safety": {"valid": True}})
    store.append(first)
    first[0]["action"][0] = 99
    store.extend([({"action": [3.0]}, {}), ({"action": [4.0]}, {"timing": 0.1})])
    assert len(store) == 3
    assert store.finalize() is store
    rows, records = store.view(0), store.view(1)
    assert rows == [{"action": [1.0, 2.0]}, {"action": [3.0]}, {"action": [4.0]}]
    assert records[::-1] == [{"timing": 0.1}, {}, {"safety": {"valid": True}}]
    assert store[-1] == ({"action": [4.0]}, {"timing": 0.1})
    assert store[::2] == [store[0], store[2]]
    assert store != []
    assert rows != [{"action": [99]}]
    for index in (3, -4):
        with pytest.raises(IndexError):
            _ = store[index]
    with pytest.raises(TypeError):
        _ = store[1.5]
    with pytest.raises(ValueError, match="finalized"):
        store.append(first)
    with pytest.raises(TypeError):
        rows[0] = {}


def test_views_share_one_decoded_cache_and_pickle_one_store():
    store = RowStore(block_rows=4)
    store.extend(({"action": [i]}, {"timing": i / 30}) for i in range(4))
    rows, records = store.view(0), store.view(1)
    assert rows[0] == {"action": [0]}
    assert records[1] == {"timing": 1 / 30}
    assert store.misses == 1
    assert store.hits == 1
    reopened_rows, reopened_records = pickle.loads(pickle.dumps((rows, records)))
    assert reopened_rows.store is reopened_records.store
    assert reopened_rows.store.currbytes == 0
    assert list(reopened_rows) == list(rows)
    assert list(reopened_records) == list(records)


def test_lru_budget_evicts_old_blocks_and_does_not_cache_oversized_blocks():
    store = RowStore(cache_bytes=4096, block_rows=1)
    store.extend({"value": str(i) + "x" * 1024} for i in range(3))
    assert store[0]["value"].startswith("0")
    first_size = store.currbytes
    assert 0 < first_size <= store.cache_bytes
    _ = store[0]
    assert store.hits == 1
    _ = store[1], store[2]
    assert store.currbytes <= store.cache_bytes
    misses = store.misses
    _ = store[0]
    assert store.misses == misses + 1

    oversized = RowStore(cache_bytes=32, block_rows=2)
    oversized.extend({"value": "z" * 2048} for _ in range(3))
    assert len(list(oversized)) == 3
    assert oversized.currbytes == 0
    assert oversized.misses == 2  # one working block, not one decode per row


def test_cache_can_be_disabled_without_changing_values():
    store = RowStore(cache_bytes=0, block_rows=1)
    store.extend([{"value": 1}, {"value": 2}])
    assert store == [{"value": 1}, {"value": 2}]
    assert store.currbytes == 0
    assert store[0] == store[0]
    assert store.currbytes == 0


def test_spawn_payload_does_not_include_growing_rows_or_cache():
    small, large = RowStore(block_rows=8), RowStore(block_rows=8)
    small.append({"value": "x" * 1024})
    large.extend({"value": "x" * 1024, "index": i} for i in range(4096))
    _ = large[2048]  # populate a cache that must not enter the pickle
    small_payload, large_payload = pickle.dumps(small), pickle.dumps(large)
    assert len(large_payload) < 1024
    assert abs(len(large_payload) - len(small_payload)) < 64
    reopened = pickle.loads(large_payload)
    assert reopened.currbytes == 0
    assert reopened[4095] == {"value": "x" * 1024, "index": 4095}
    assert large._owner is not None
    assert reopened._owner is None


@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_workers_reopen_data_and_cannot_delete_owner_files(method, tmp_path):
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{method} is not supported")
    store = RowStore(block_rows=2, root=tmp_path)
    store.extend({"value": i} for i in range(3))
    indices = FileBackedIntSequence(root=tmp_path)
    indices.extend([0, 2, 9])
    used = IndexSet(10, root=tmp_path)
    used.update([9, 2, 2, 0])
    store.finalize()
    indices.finalize()
    used.finalize()
    _ = store[-1], indices[0], 2 in used  # inherited open handles/cache in fork
    owner_paths = [value.path for value in (store, indices, used)]
    context = multiprocessing.get_context(method)
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_worker_read, args=(writer, store, indices, used))
    process.start()
    writer.close()
    try:
        assert reader.poll(20), "worker did not return data"
        result = reader.recv()
        process.join(20)
        assert process.exitcode == 0
        assert result["row"] == {"value": 2}
        assert result["indices"] == [0, 2, 9]
        assert result["used"] == [0, 2, 9]
        assert result["misses"] == 1
        assert result["pid"] != os.getpid()
        assert result["cache_bytes"] <= store.cache_bytes
        assert all(path.is_dir() for path in owner_paths)
        assert store[-1] == {"value": 2}
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        reader.close()


def test_view_keeps_owner_alive_until_dataset_lifetime_ends(tmp_path):
    store = RowStore(root=tmp_path)
    store.append(({"value": 1}, {}))
    store.finalize()
    path, reference = store.path, weakref.ref(store)
    view = store.view(0)
    del store
    gc.collect()
    assert reference() is not None
    assert path.is_dir()
    assert view[0] == {"value": 1}
    del view
    gc.collect()
    assert reference() is None
    assert not path.exists()


def test_fork_cannot_finalize_or_flush_an_unfinished_parent_store():
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork is not supported")
    rows = RowStore(block_rows=1)
    rows.append({"value": 0})
    indices = FileBackedIntSequence()
    indices.append(0)
    used = IndexSet(3)
    used.add(0)
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_worker_reject_building, args=(writer, rows, indices, used))
    process.start()
    writer.close()
    try:
        assert reader.poll(10)
        errors = reader.recv()
        process.join(10)
        assert process.exitcode == 0
        assert len(errors) == 6
        assert all("finalize" in message for message in errors)
        rows.append({"value": 1})
        indices.append(1)
        used.add(2)
        assert list(rows) == [{"value": 0}, {"value": 1}]
        assert list(indices) == [0, 1]
        assert list(used) == [0, 2]
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        reader.close()


def test_close_is_idempotent_and_ownerless_reopen_does_not_own_directory(tmp_path):
    store = RowStore(root=tmp_path)
    store.append({"value": 1})
    reopened = pickle.loads(pickle.dumps(store))
    path = store.path
    reopened.close()
    reopened.close()
    assert path.is_dir()
    assert store[0] == {"value": 1}
    with pytest.raises(ValueError, match="closed"):
        _ = reopened[0]
    store.close()
    store.close()
    assert not path.exists()


def test_compact_integer_sequence_search_slices_empty_and_reopen(tmp_path):
    indices = FileBackedIntSequence(root=tmp_path)
    indices.extend([-3, 0, 4, 4, 1000000000000])
    assert len(indices) == 5
    assert indices.finalize() is indices
    assert (indices.path / "values").stat().st_size == 5 * 8
    assert indices[1::2] == [0, 4]
    assert indices[-1] == 1000000000000
    assert indices.searchsorted(4) == 2
    assert indices.searchsorted(4, side="right") == 4
    assert indices.searchsorted(-10) == 0
    assert indices.searchsorted(1000000000001) == 5
    reopened = pickle.loads(pickle.dumps(indices))
    assert reopened == [-3, 0, 4, 4, 1000000000000]
    with pytest.raises(ValueError, match="side"):
        indices.searchsorted(0, side="middle")
    with pytest.raises(ValueError, match="finalized"):
        indices.append(9)
    empty = FileBackedIntSequence()
    assert list(empty) == []
    assert empty[:] == []
    assert empty.searchsorted(1) == 0


def test_index_set_deduplicates_and_iterates_sorted_without_python_set(tmp_path):
    used = IndexSet(131075, root=tmp_path)
    used.add(131074)
    used.update([65536, 0, 8, 8, 65535, 0])
    assert len(used) == 5
    assert 8 in used
    assert -1 not in used
    assert 131075 not in used
    assert "8" not in used
    assert (used.path / "values").stat().st_size == 131075
    assert list(used) == [0, 8, 65535, 65536, 131074]
    reopened = pickle.loads(pickle.dumps(used))
    assert len(reopened) == 5
    assert list(reopened) == list(used)
    assert len(pickle.dumps(used)) < 512
    with pytest.raises(ValueError, match="finalized"):
        used.add(1)
    empty = IndexSet(0)
    assert list(empty) == []
    assert 0 not in empty


def test_concurrent_reads_do_not_race_file_positions_or_cached_blocks():
    rows = RowStore(cache_bytes=4096, block_rows=3)
    rows.extend({"value": i, "text": str(i) * 200} for i in range(60))
    indices = FileBackedIntSequence()
    indices.extend(range(60))
    used = IndexSet(60)
    used.update(range(0, 60, 2))
    rows.finalize()
    indices.finalize()
    used.finalize()

    def read(i):
        return rows[i]["value"], indices[i], i in used

    requested = [(i * 17) % 60 for i in range(600)]
    with ThreadPoolExecutor(max_workers=6) as workers:
        actual = list(workers.map(read, requested))
    assert actual == [(i, i, i % 2 == 0) for i in requested]
    assert rows.currbytes <= rows.cache_bytes


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RowStore(cache_bytes=-1),
        lambda: RowStore(cache_bytes=True),
        lambda: RowStore(block_rows=0),
        lambda: RowStore(block_rows=1.5),
        lambda: IndexSet(-1),
        lambda: IndexSet(True),
    ],
)
def test_invalid_configuration_does_not_create_partial_owners(factory):
    with pytest.raises(ValueError):
        factory()


def test_invalid_integer_values_and_set_bounds_are_not_silently_coerced():
    indices = FileBackedIntSequence()
    for value in (True, 1.2):
        with pytest.raises(TypeError):
            indices.append(value)
    with pytest.raises(OverflowError):
        indices.append(1 << 63)
    assert len(indices) == 0
    used = IndexSet(3)
    for index in (-1, 3):
        with pytest.raises(IndexError):
            used.add(index)
    for index in (True, 1.2):
        with pytest.raises(TypeError):
            used.add(index)
    assert len(used) == 0

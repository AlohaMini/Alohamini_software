"""Committed statistics resume without scanning historical rows or decoding video."""

import errno
import json
import os
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from alohamini.datasets import recording_stats_cache as cache
from alohamini.datasets.lerobotv3 import _Stats

CAMERA = "observation.images.forward"
SOURCES = (
    "data/chunk-000/file-000.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
    f"videos/{CAMERA}/chunk-000/file-000.mp4",
)


def append(stats, start, count):
    rows = [
        {"action": [1e12 + index / 1024, 0.1], "timestamp": index / 30}
        for index in range(start, start + count)
    ]
    images = [np.full((2, 3, 3), start % 255, dtype=np.uint8)]
    stats.update(rows, decoded_images={CAMERA: images})


def publish_metadata(root, stats):
    values = {
        "meta/info.json": {
            "codebase_version": "v3.0",
            "total_frames": stats.frames,
            "total_episodes": 1,
            "fps": 30,
            "features": stats.features,
        },
        "meta/alohamini.json": {"source_info": {"fps": 30}},
        "meta/stats.json": stats.result(),
    }
    for relative, value in values.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, default=lambda item: item.tolist()))


@pytest.fixture
def recorded(tmp_path):
    features = {
        "action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        CAMERA: {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        },
    }
    stats = _Stats(features, recording=True)
    append(stats, 0, 3)
    append(stats, 3, 4)
    publish_metadata(tmp_path, stats)
    for relative in SOURCES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"Historical contents must not be read by the cache")
    return tmp_path, stats


def assert_states(actual, expected):
    assert actual.frames == expected.frames
    assert actual.features == expected.features
    assert actual.image_statistics.payload() == expected.image_statistics.payload()
    for key, tracker in expected.trackers.items():
        if tracker is None:
            assert actual.trackers[key] is None
            continue
        left, right = actual.trackers[key].export_state(), tracker.export_state()
        assert left.keys() == right.keys()
        for name, value in right.items():
            assert left[name].dtype == value.dtype
            assert left[name].shape == value.shape
            assert left[name].tobytes() == value.tobytes(), (key, name)


def rewrite_cache(root, change):
    path = root / cache._CACHE
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    manifest = json.loads(arrays["manifest"].tobytes())
    change(manifest, arrays)
    arrays["manifest"] = np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def test_round_trip_is_bitwise_and_continues_original_batches_without_history_reads(
    recorded, monkeypatch
):
    root, original = recorded
    cache.save_recording_statistics(root, original)
    read_bytes = Path.read_bytes

    def metadata_only(path):
        assert path.relative_to(root).as_posix() in cache._METADATA
        return read_bytes(path)

    def forbidden(*args, **kwargs):
        pytest.fail("Valid caches must not scan data, decode images or recompute quantiles")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_bytes", metadata_only)
        scoped.setattr(_Stats, "result", forbidden)
        scoped.setattr("alohamini.datasets.lerobotv3.pq.ParquetFile", forbidden)
        scoped.setattr("alohamini.datasets.lerobotv3.pq.read_table", forbidden)
        scoped.setattr("alohamini.datasets.lerobotv3.Image.open", forbidden)
        restored = cache.load_recording_statistics(root)
    assert restored is not None
    assert_states(restored, original)
    assert restored._recording_source_binding == cache.recording_statistics_binding(root)
    append(original, 7, 5)
    append(restored, 7, 5)
    assert_states(restored, original)
    for key, expected in original.result().items():
        for name, value in expected.items():
            np.testing.assert_array_equal(restored.result()[key][name], value)


def test_missing_cache_is_legacy_fallback_without_warning(recorded, caplog):
    root, _ = recorded
    assert cache.load_recording_statistics(root) is None
    assert not caplog.records


@pytest.mark.parametrize("relative", [*SOURCES, *cache._METADATA])
def test_any_formal_source_or_committed_metadata_change_invalidates_cache(recorded, relative):
    root, stats = recorded
    cache.save_recording_statistics(root, stats)
    with (root / relative).open("ab") as stream:
        stream.write(b" ")
    assert cache.load_recording_statistics(root) is None


def test_same_size_replacement_with_restored_mtime_is_still_invalid(recorded):
    root, stats = recorded
    cache.save_recording_statistics(root, stats)
    source = root / SOURCES[0]
    details, content = source.stat(), source.read_bytes()
    replacement = source.with_suffix(".replacement")
    replacement.write_bytes(content)
    os.utime(replacement, ns=(details.st_atime_ns, details.st_mtime_ns))
    replacement.replace(source)
    assert cache.load_recording_statistics(root) is None


@pytest.mark.parametrize(
    "damage", ["truncated", "compressed", "count", "float32", "object", "image", "extra"]
)
def test_truncated_or_malformed_cache_falls_back(recorded, damage, caplog):
    root, stats = recorded
    cache.save_recording_statistics(root, stats)
    if damage == "truncated":
        (root / cache._CACHE).write_bytes(b"PK\x03\x04truncated")
    elif damage == "compressed":
        path = root / cache._CACHE
        with zipfile.ZipFile(path) as archive:
            header = archive.getinfo("manifest.npy").header_offset
        contents = bytearray(path.read_bytes())
        name_size = int.from_bytes(contents[header + 26 : header + 28], "little")
        extra_size = int.from_bytes(contents[header + 28 : header + 30], "little")
        contents[header + 30 + name_size + extra_size] = 255
        path.write_bytes(contents)
    else:

        def corrupt(manifest, arrays):
            prefix = f"tracker_{manifest['trackers']['action']}__"
            if damage == "count":
                arrays[prefix + "counts_0"][:] = 0
            elif damage == "float32":
                arrays[prefix + "mean"] = arrays[prefix + "mean"].astype(np.float32)
            elif damage == "object":
                arrays[prefix + "mean"] = np.array([object(), object()], dtype=object)
            elif damage == "image":
                manifest["image_stats"]["cameras"][CAMERA]["histogram"][0][0] = -1
            else:
                arrays["unexpected"] = np.array([1])

        rewrite_cache(root, corrupt)
    assert cache.load_recording_statistics(root) is None
    assert "Recording statistics cache ignored" in caplog.text


@pytest.mark.parametrize("relative", cache._METADATA)
def test_invalid_metadata_structure_falls_back_instead_of_raising(recorded, relative):
    root, stats = recorded
    cache.save_recording_statistics(root, stats)
    (root / relative).write_text("[]")
    assert cache.load_recording_statistics(root) is None


def test_append_updates_binding_but_does_not_reauthorize_changed_history(recorded):
    root, stats = recorded
    cache.save_recording_statistics(root, stats)
    old = deepcopy(stats._recording_source_binding)
    new_source = root / "data/chunk-000/file-001.parquet"
    new_source.write_bytes(b"new committed episode")
    append(stats, 7, 3)
    publish_metadata(root, stats)
    cache.save_recording_statistics(root, stats)
    assert stats._recording_source_binding != old
    assert len(stats._recording_source_binding["files"]) == len(old["files"]) + 1
    restored = cache.load_recording_statistics(root)
    assert restored is not None
    assert_states(restored, stats)
    (root / SOURCES[0]).write_bytes(b"changed old data")
    with pytest.raises(ValueError, match="Previously accumulated recording file changed"):
        cache.save_recording_statistics(root, restored)


@pytest.mark.parametrize("operation", ["replace", "encode"])
def test_failed_atomic_save_preserves_old_cache_and_removes_tempfile(
    recorded, monkeypatch, operation
):
    root, stats = recorded
    cache.save_recording_statistics(root, stats)
    original = (root / cache._CACHE).read_bytes()
    original_binding = deepcopy(stats._recording_source_binding)

    def fail(*args, **kwargs):
        raise OSError("injected cache write failure")

    if operation == "replace":
        monkeypatch.setattr(cache.os, "replace", fail)
    else:
        monkeypatch.setattr(cache.np, "savez_compressed", fail)
    with pytest.raises(OSError, match="injected"):
        cache.save_recording_statistics(root, stats)
    assert (root / cache._CACHE).read_bytes() == original
    assert stats._recording_source_binding == original_binding
    assert not list((root / "meta").glob(".recording_stats-*.npz"))


def test_change_during_load_or_save_is_rejected(recorded, monkeypatch):
    root, stats = recorded
    cache.save_recording_statistics(root, stats)
    original = (root / cache._CACHE).read_bytes()
    changed = cache.recording_statistics_binding(root)
    changed["files"][0]["mtime_ns"] += 1
    monkeypatch.setattr(cache, "recording_statistics_binding", lambda _: changed)
    assert cache.load_recording_statistics(root) is None
    with pytest.raises(ValueError, match="changed while saving"):
        cache.save_recording_statistics(root, stats)
    assert (root / cache._CACHE).read_bytes() == original
    assert not list((root / "meta").glob(".recording_stats-*.npz"))


def test_archive_size_budget_grows_with_committed_frames_without_fixed_gib_limit():
    features = {"action": {"dtype": "float32", "shape": [36]}}
    arrays, size = cache._archive_limits(features, 10_000_000)
    assert arrays == 1 + 5 + 2 * 36
    assert size > 5 * 1024**3


def test_incremental_restore_preserves_moments_and_signed_zero_bit_for_bit(recorded):
    root, template = recorded
    stats = _Stats(template.features, recording=True)
    batches = (
        [{"action": [0.0, -0.0], "timestamp": 0.0}, {"action": [2.0, 3.0], "timestamp": 1.0}],
        [{"action": [-0.0, 0.0], "timestamp": 2.0}, {"action": [-2.0, -3.0], "timestamp": 3.0}],
        [{"action": [0.0, -0.0], "timestamp": 4.0}],
    )
    for batch in batches:
        stats.update(batch, {CAMERA: [np.zeros((2, 3, 3), np.uint8)]})
        publish_metadata(root, stats)
        cache.save_recording_statistics(root, stats)
        restored = cache.load_recording_statistics(root)
        assert restored is not None
        assert_states(restored, stats)


@pytest.mark.parametrize("damage", ["missing", "truncated", "outside", "cycle"])
def test_broken_incremental_parent_is_rejected_without_returning_partial_stats(
    recorded, damage, caplog
):
    root, stats = recorded
    cache.save_recording_statistics(root, stats)
    append(stats, 7, 3)
    publish_metadata(root, stats)
    cache.save_recording_statistics(root, stats)
    with np.load(root / cache._CACHE, allow_pickle=False) as archive:
        current = json.loads(archive["manifest"].tobytes())
    parent = root / current["parent"]["path"]
    if damage == "missing":
        parent.unlink()
    elif damage == "truncated":
        parent.write_bytes(b"broken NPZ")
    else:

        def corrupt(manifest, arrays):
            manifest["parent"]["path"] = "../outside.npz" if damage == "outside" else cache._CACHE

        rewrite_cache(root, corrupt)
    assert cache.load_recording_statistics(root) is None
    assert "Recording statistics cache ignored" in caplog.text


def test_filesystem_without_hardlinks_can_continue_incrementally(recorded, monkeypatch):
    root, stats = recorded
    cache.save_recording_statistics(root, stats)

    def unsupported(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "hard links unsupported")

    monkeypatch.setattr("alohamini.datasets.recording_stats_incremental.os.link", unsupported)
    append(stats, 7, 3)
    publish_metadata(root, stats)
    cache.save_recording_statistics(root, stats)
    restored = cache.load_recording_statistics(root)
    assert restored is not None
    assert_states(restored, stats)

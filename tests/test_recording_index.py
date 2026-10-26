"""Resume checks read episode indexes and data footers, never historical frames."""

import json
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_recording_storage import create

from alohamini.datasets.recording_index import read_recording_index


@pytest.fixture
def recording(tmp_path):
    root = create(tmp_path / "record", cameras=("forward",))
    source = json.loads((root / "meta/alohamini.json").read_text())["source_info"]
    return root, source


def change_index(root, episode, **changes):
    path = root / f"meta/episodes/chunk-000/file-{episode:03d}.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0].update(changes)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_index_reads_only_selected_metadata_columns_and_data_footers(recording, tmp_path):
    root, source = recording
    expected = json.loads((root / "meta/info.json").read_text())
    original_file, original_text = pq.ParquetFile, Path.read_text
    selected, footer_files = [], []

    class IndexOrFooter:
        def __init__(self, path):
            self.path = Path(path)
            self.is_data = self.path.is_relative_to(root / "data")
            self.file = original_file(path)
            if self.is_data:
                footer_files.append(self.path)

        def __getattr__(self, name):
            return getattr(self.file, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.file.close()

        def read(self, *, columns):
            assert not self.is_data, "historical numeric rows must not be read"
            assert columns and not any(
                name.startswith("stats/") or name == "image_statistics" for name in columns
            )
            selected.append(columns)
            return self.file.read(columns=columns)

        def iter_batches(self, *args, **kwargs):
            raise AssertionError("historical numeric rows must not be iterated")

    def metadata_text(path, *args, **kwargs):
        assert path == root / "meta/info.json", "sidecars and other JSON must not be opened"
        return original_text(path, *args, **kwargs)

    # An unrelated subtree is deliberately outside the startup check's scope.
    unrelated = root / "discarded"
    unrelated.mkdir()
    (unrelated / "old-link").symlink_to(tmp_path / "unrelated-missing-target")
    with (
        patch("pyarrow.parquet.ParquetFile", side_effect=IndexOrFooter),
        patch("pyarrow.parquet.read_table", side_effect=AssertionError("full table read")),
        patch("pathlib.Path.read_text", new=metadata_text),
        patch("av.open", side_effect=AssertionError("video must not be opened")),
        patch("alohamini.datasets.images.video_rgb", side_effect=AssertionError("video decode")),
    ):
        info, shapes = read_recording_index(root, source)
    assert info == expected and shapes == {"forward": [32, 32, 3]}
    assert len(selected) == 2 and len(set(footer_files)) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"total_frames": 7},
        {"total_episodes": 3},
        {"total_episodes": True},
        {"total_frames": 0},
        {"fps": 25},
        {"codebase_version": "v2.1"},
        {"data_path": "../file-{file_index}.parquet"},
        {"total_tasks": 2},
    ],
)
def test_rejects_inconsistent_global_index_metadata(recording, changes):
    root, source = recording
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info.update(changes)
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError):
        read_recording_index(root, source)


@pytest.mark.parametrize(
    "changes",
    [
        {"episode_index": 2},
        {"length": 0},
        {"dataset_from_index": 4},
        {"dataset_to_index": 8},
        {"tasks": ["a different task"]},
        {"data/file_index": -1},
        {"data/file_index": 1.5},
        {"meta/episodes/file_index": 0},
        {"videos/observation.images.forward/chunk_index": -1},
        {"videos/observation.images.forward/from_timestamp": float("nan")},
        {"videos/observation.images.forward/to_timestamp": 0.2},
    ],
)
def test_rejects_episode_gaps_bad_references_and_video_intervals(recording, changes):
    root, source = recording
    change_index(root, 1, **changes)
    with pytest.raises(ValueError):
        read_recording_index(root, source)


@pytest.mark.parametrize(
    "relative",
    [
        "data/chunk-000/file-001.parquet",
        "videos/observation.images.forward/chunk-000/file-001.mp4",
        "meta/safety/episode_000001.jsonl",
    ],
)
def test_missing_or_empty_referenced_files_are_rejected_without_opening_them(recording, relative):
    root, source = recording
    path = root / relative
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="nonempty"):
        read_recording_index(root, source)
    path.unlink()
    with pytest.raises(ValueError, match="Missing recording"):
        read_recording_index(root, source)


@pytest.mark.parametrize(
    "relative", ["meta/recovery/interrupted.json", ".recording/episodes/episode_000002.pending"]
)
def test_pending_recovery_prevents_resume_without_reading_marker_contents(recording, relative):
    root, source = recording
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_text("not JSON and must not be parsed")
    with pytest.raises(ValueError, match="recover before resuming"):
        read_recording_index(root, source)


def test_shared_data_file_checks_the_sum_of_its_episode_lengths(recording):
    root, source = recording
    first = root / "data/chunk-000/file-000.parquet"
    second = root / "data/chunk-000/file-001.parquet"
    table = pa.concat_tables([pq.read_table(first), pq.read_table(second)])
    pq.write_table(table, first)
    second.unlink()
    change_index(root, 1, **{"data/file_index": 0})
    assert read_recording_index(root, source)[0]["total_frames"] == 6
    pq.write_table(table.slice(0, 5), first)
    with pytest.raises(ValueError, match="footer row count"):
        read_recording_index(root, source)


def test_data_footer_schema_must_match_the_declared_feature_types(recording):
    root, source = recording
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    field = pa.field("action", pa.list_(pa.float64(), 18))
    table = table.set_column(
        table.schema.get_field_index("action"), field, table["action"].cast(field.type)
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="data schema"):
        read_recording_index(root, source)


@pytest.mark.parametrize("kind", ["reference", "parent", "future"])
def test_internal_reference_and_future_write_symlinks_are_rejected(recording, tmp_path, kind):
    root, source = recording
    if kind == "parent":
        path = root / "data/chunk-000"
        destination = tmp_path / "external-shard"
        path.rename(destination)
    elif kind == "reference":
        path = root / "videos/observation.images.forward/chunk-000/file-001.mp4"
        destination = tmp_path / "external.mp4"
        path.rename(destination)
    else:
        path = root / "data/chunk-000/file-002.parquet"
        destination = tmp_path / "external-future.parquet"
        destination.write_bytes(b"untouched")
    path.symlink_to(destination, target_is_directory=kind == "parent")
    with pytest.raises(ValueError, match="symbolic link"):
        read_recording_index(root, source)


def test_explicit_root_symlink_retains_existing_resume_behavior(recording, tmp_path):
    root, source = recording
    alias = tmp_path / "selected-recording"
    alias.symlink_to(root, target_is_directory=True)
    assert read_recording_index(alias, source)[0]["total_episodes"] == 2


def test_equivalent_float_fps_and_extra_arrow_metadata_remain_compatible(recording):
    root, source = recording
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info["fps"] = 30.0
    path.write_text(json.dumps(info))
    data = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(data)
    table = table.replace_schema_metadata({**table.schema.metadata, b"custom": b"preserved"})
    pq.write_table(table, data)
    assert read_recording_index(root, source)[0]["fps"] == 30

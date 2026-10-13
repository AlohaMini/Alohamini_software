import json

import pytest
import torch
from test_dataset_lerobot_tools import _add_video, _make_gapped_dataset
from test_native_learning import recording as recording

from alohamini.datasets.lerobotv3 import export_lerobot
from alohamini.datasets.tools import check_dataset
from alohamini.learning.data import AlohaMiniDataset


def test_original_v3_opt_in_preserves_windows_and_missing_provenance(recording, tmp_path):
    root = tmp_path / "v3"
    export_lerobot(recording, root)
    args = dict(episodes=[0, 1], state="none", chunk_size=3, image_size=(32, 32))
    native = AlohaMiniDataset(root, **args)
    (root / "meta/alohamini.json").unlink()
    for sidecar in (root / "meta/safety").glob("*.jsonl"):
        sidecar.unlink()
    with pytest.raises(ValueError, match="allow_unannotated"):
        AlohaMiniDataset(root, **args)
    original = AlohaMiniDataset(root, allow_unannotated=True, **args)
    assert "robot_metadata" not in original.info
    assert original.info["training_only"]["calibration"] == "unavailable"
    assert original.records == [{}] * len(original.rows)
    assert native.locations == original.locations
    assert native.statistics() == original.statistics()
    for i in range(len(native)):
        for key in native[i]:
            torch.testing.assert_close(native[i][key], original[i][key], rtol=0, atol=0)
    assert not (root / "meta/alohamini.json").exists()
    recorded = AlohaMiniDataset(root, episodes=[0], state="recorded", allow_unannotated=True)
    torch.testing.assert_close(
        recorded[0]["observation.state"], torch.tensor(recorded.rows[0]["observation.state"])
    )
    assert set(recorded.selection.units) == {"recorded (unit unverified)"}


def test_original_v3_opt_in_does_not_relax_annotated_export_checks(recording, tmp_path):
    root = tmp_path / "v3"
    export_lerobot(recording, root)
    (root / "meta/safety/episode_000000.jsonl").unlink()
    with pytest.raises(ValueError, match="sidecar"):
        AlohaMiniDataset(root, episodes=[0], state="none", allow_unannotated=True)
    metadata = root / "meta/alohamini.json"
    content = json.loads(metadata.read_text())
    content["source_info"]["features"]["action"]["names"][0] = "wrong_axis"
    metadata.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="coordinates"):
        AlohaMiniDataset(root, episodes=[0], state="none", allow_unannotated=True)


def test_only_unused_video_tails_are_allowed_and_recorded(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    _make_gapped_dataset(tmp_path)
    path = tmp_path / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    for i, row in enumerate(rows):
        row.update(index=i, episode_index=i // 2)
    pq.write_table(pa.Table.from_pylist(rows), path)
    info_path = tmp_path / "meta/info.json"
    info = json.loads(info_path.read_text())
    info.update(total_episodes=2, total_frames=4, splits={"train": "0:2"})
    info_path.write_text(json.dumps(info))
    _add_video(tmp_path, starts=(0, 3), count=6)
    path = tmp_path / "meta/episodes/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    for i, row in enumerate(rows):
        row["episode_index"] = i
        row["videos/observation.images.forward/to_timestamp"] = (i + 1) * 3 / 25
    pq.write_table(pa.Table.from_pylist(rows), path)
    assert not check_dataset(tmp_path)["valid"]
    report = check_dataset(tmp_path, allow_unused_video_tail=True)
    assert report["valid"], report["issues"]
    assert sum(i["code"] == "VIDEO_UNUSED_EPISODE_TAIL" for i in report["issues"]) == 2
    # Timestamp mapping and undersized video ranges are never waived.
    rows[0]["videos/observation.images.forward/to_timestamp"] = 1 / 25
    pq.write_table(pa.Table.from_pylist(rows), path)
    report = check_dataset(tmp_path, allow_unused_video_tail=True)
    assert not report["valid"]
    assert "VIDEO_EPISODE_LENGTH_MISMATCH" in {i["code"] for i in report["issues"]}

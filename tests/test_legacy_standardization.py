"""Legacy calibration conversion preserves source evidence and training contracts."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_model_coordinate_binding import standard_robot
from test_native_learning import recording as recording

from alohamini.datasets.lerobotv3 import export_lerobot
from alohamini.datasets.record import StateSelection, state_names
from alohamini.datasets.tools import check_dataset
from alohamini.learning.calibration import reference_binding
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.lerobot import LeRobotSource
from alohamini.model import get_robot_model

_path = Path(__file__).parents[1] / "scripts/standardize_legacy_dataset.py"
_spec = importlib.util.spec_from_file_location("standardize_legacy_dataset", _path)
conversion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conversion)
JOINT = "arm_left_shoulder_pan"


def plan_for(source):
    return dict(
        format=conversion.FORMAT,
        version=1,
        method="same_encoder",
        basis="Test operator confirms unchanged motors and mechanics",
        source_robot_metadata=deepcopy(source),
        target_robot_metadata=standard_robot(1300).payload["_robot_metadata"],
    )


def complete_fixture_metadata(recording):
    path = recording / "meta/info.json"
    info = json.loads(path.read_text())
    template = standard_robot(1300).payload["_robot_metadata"]["motors"]
    for motor in get_robot_model("alohamini2pro").actuators:
        row = info["robot_metadata"]["motors"].setdefault(
            motor.name, deepcopy(template[motor.name])
        )
        row.update(id=motor.motor_id, model=motor.motor_model, homing_offset=0)
    path.write_text(json.dumps(info))
    for path in recording.glob("episodes/*/safety.jsonl"):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for record in records:
            if "robot_metadata" in record:
                record["robot_metadata"] = info["robot_metadata"]
        path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_same_encoder_undoes_asymmetric_range_and_preserves_non_arm_units():
    metadata = deepcopy(standard_robot(1300).payload["_robot_metadata"])
    metadata.pop("reference_calibration")
    plan = plan_for(metadata)
    old, new = (p["motors"][JOINT] for p in (metadata, plan["target_robot_metadata"]))
    old.update(range_min=1000, range_max=3000, homing_offset=new["homing_offset"] + 20)
    plan["source_robot_metadata"] = metadata
    value = np.zeros((3, 18))
    value[:, 0] = [-50, 0, 50]
    value[:, -4:] = [0.1, -0.2, 15, 320]
    result = conversion.Conversion(plan, state_names("alohamini2pro")).transform(value)
    # Old zero is tick 2000, not the manually zeroed tick 2047.
    expected_ticks = np.array([1500, 2000, 2500]) + 20
    expected = (expected_ticks - new["range_min"]) / (
        new["range_max"] - new["range_min"]
    ) * 200 - 100
    np.testing.assert_allclose(result[:, 0], expected, atol=2e-5)
    np.testing.assert_allclose(result[:, -4:], value[:, -4:])


def test_raw_feedback_restores_saturation_and_keeps_out_of_range_measurements():
    metadata = deepcopy(standard_robot(1300).payload["_robot_metadata"])
    plan = plan_for(metadata)
    source = plan["source_robot_metadata"]["motors"][JOINT]
    target = plan["target_robot_metadata"]["motors"][JOINT]
    source.update(range_max=target["range_max"] - 10)
    values = np.zeros((1, 18))
    values[0, 0] = 100
    raw = np.zeros((1, 18))
    raw[0, 0] = target["range_max"] + 8
    valid = np.zeros_like(raw)
    valid[0, 0] = 1
    result = conversion.Conversion(plan, state_names("alohamini2pro")).transform(
        values, raw=raw, raw_valid=valid
    )
    assert result[0, 0] > 100
    raw[0, 0] = 2000
    with pytest.raises(ValueError, match="disagree"):
        conversion.Conversion(plan, state_names("alohamini2pro")).transform(
            values, raw=raw, raw_valid=valid
        )


def test_wrap_and_software_inversion_do_not_apply_zero_twice():
    metadata = deepcopy(standard_robot(1300).payload["_robot_metadata"])
    plan = plan_for(metadata)
    old = plan["source_robot_metadata"]["motors"][JOINT]
    new = plan["target_robot_metadata"]["motors"][JOINT]
    old.update(range_min=0, range_max=4095, drive_mode=1, homing_offset=2047)
    new.update(range_min=0, range_max=4095, drive_mode=0, homing_offset=-2047)
    value = np.zeros((1, 18))
    value[0, 0] = 100
    actual = conversion.Conversion(plan, state_names("alohamini2pro")).transform(value)
    assert actual[0, 0] == pytest.approx(4094 / 4095 * 200 - 100, abs=1e-5)


@pytest.mark.parametrize("annotated", [True, False])
def test_dataset_copy_stats_training_and_cross_robot_binding(recording, tmp_path, annotated):
    complete_fixture_metadata(recording)
    source = tmp_path / "original"
    export_lerobot(recording, source)
    annotation_path = source / "meta/alohamini.json"
    original_annotation = json.loads(annotation_path.read_text())
    plan = plan_for(original_annotation["source_info"]["robot_metadata"])
    if not annotated:
        annotation_path.unlink()
        for path in (source / "meta/safety").glob("*.jsonl"):
            path.unlink()
    before = {
        str(p.relative_to(source)): conversion.file_digest(p)
        for p in source.rglob("*")
        if p.is_file()
    }
    output = tmp_path / "standard"
    report = conversion.convert(source, plan, output=output)
    assert report["status"] == "complete"
    assert before == {
        str(p.relative_to(source)): conversion.file_digest(p)
        for p in source.rglob("*")
        if p.is_file()
    }
    assert check_dataset(output)["valid"]
    path = next(output.glob("data/**/*.parquet"))
    result = pq.read_table(path)
    original = pq.read_table(source / path.relative_to(output))
    assert result["source_action"] == original["action"]
    assert result["observation.source_state"] == original["observation.state"]
    assert "observation.source_motor_position_raw" in result.column_names
    assert "observation.motor_position_raw" not in result.column_names
    assert "observation.motor_velocity_raw" in result.column_names
    assert "observation.motor_current_ma" in result.column_names
    stats = json.loads((output / "meta/stats.json").read_text())
    tables = [
        pq.read_table(p, columns=["action"]) for p in sorted(output.glob("data/**/*.parquet"))
    ]
    actions = np.concatenate([np.asarray(t["action"].to_pylist()) for t in tables])
    np.testing.assert_allclose(stats["action"]["mean"], actions.mean(0))
    samples = AlohaMiniDataset(
        output,
        episodes=[0, 1],
        state="joint_position",
        chunk_size=2,
        image_size=(32, 32),
        cache_dir=tmp_path / "cache",
    )
    assert len(samples) > 0
    assert "training_only" not in samples.info
    assert samples._storage.has_safety_metadata is annotated
    if not annotated:
        assert samples.records == [{}] * len(samples.rows)
        assert not list((output / "meta/safety").glob("*.jsonl"))
    assert samples[0]["action"].shape == (2, 18)
    derived = StateSelection(samples.info, "joint_velocity,joint_current")
    assert derived.feature["shape"] == [28]
    reference_binding(samples.info["robot_metadata"], standard_robot(4095))
    with pytest.raises(FileExistsError):
        conversion.convert(source, plan, output=output)


def test_missing_safety_requires_bound_conversion_manifest(recording, tmp_path):
    source = tmp_path / "source"
    export_lerobot(recording, source)
    annotation = json.loads((source / "meta/alohamini.json").read_text())
    annotation["capture_metadata"] = {"available": False, "reason": "not_recorded_in_source"}
    (source / "meta/alohamini.json").write_text(json.dumps(annotation))
    with pytest.raises(ValueError, match="provenance"):
        LeRobotSource(source, json.loads((source / "meta/info.json").read_text()))


def test_missing_video_is_rejected_before_creating_output(recording, tmp_path):
    complete_fixture_metadata(recording)
    source = tmp_path / "source"
    export_lerobot(recording, source)
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.forward"]["dtype"] = "video"
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    info_path.write_text(json.dumps(info))
    for path in source.glob("meta/episodes/**/*.parquet"):
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            references = {"chunk_index": 0, "file_index": 999, "from_timestamp": 0.0}
            for field, value in references.items():
                row[f"videos/observation.images.forward/{field}"] = value
        pq.write_table(pa.Table.from_pylist(rows), path)
    annotation = json.loads((source / "meta/alohamini.json").read_text())
    output = tmp_path / "converted"
    with pytest.raises(ValueError, match="Source media is incomplete"):
        conversion.convert(
            source, plan_for(annotation["source_info"]["robot_metadata"]), output=output
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".pending-*"))


def test_contact_reference_keeps_full_angular_span_instead_of_rescaling_endpoints():
    metadata = deepcopy(standard_robot(1300).payload["_robot_metadata"])
    plan = plan_for(metadata)
    plan["method"] = "legacy_home_contacts"
    anchors = {}
    for name in metadata["motors"]:
        if name.startswith("arm_"):
            anchors[name] = dict(source_tick=2047, target_tick=2047)
    anchors[JOINT] = dict(source_tick=2500, target_tick=2400)
    plan["reference_anchors"] = anchors
    old = plan["source_robot_metadata"]["motors"][JOINT]
    new = plan["target_robot_metadata"]["motors"][JOINT]
    old.update(range_min=1000, range_max=3000)
    value = np.zeros((2, 18))
    value[:, 0] = [50, -50]
    actual = conversion.Conversion(plan, state_names("alohamini2pro")).transform(value)
    ticks = (actual[:, 0] + 100) / 200 * (new["range_max"] - new["range_min"]) + new["range_min"]
    np.testing.assert_allclose(ticks, [2400, 1400], atol=2e-4)

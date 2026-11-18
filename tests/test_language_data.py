"""Shared optional annotation columns stay frame-aligned and preserve ragged batches."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_native_learning import model_options
from test_native_learning import recording as recording

from alohamini.datasets.language import (
    language_arrow_type,
    language_feature_info,
    validate_language_rows,
)
from alohamini.datasets.lerobotv3 import export_lerobot
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.loading import make_loader


def annotated_v3(recording, path):
    export_lerobot(recording, path)
    (path / "meta/alohamini.json").unlink()
    for file in (path / "meta/safety").glob("*.jsonl"):
        file.unlink()
    return add_annotations(path)


def add_annotations(path):
    info = json.loads((path / "meta/info.json").read_text())
    info["features"].update(language_feature_info())
    (path / "meta/info.json").write_text(json.dumps(info))
    for file in (path / "data").rglob("*.parquet"):
        table = pq.read_table(file)
        persistent = []
        events = []
        for row in table.to_pylist():
            persistent.append(
                [
                    dict(
                        role="assistant",
                        style="subtask",
                        content="approach",
                        timestamp=0.0,
                        camera=None,
                        tool_calls=None,
                    ),
                    dict(
                        role="assistant",
                        style="subtask",
                        content="grasp",
                        timestamp=0.1,
                        camera=None,
                        tool_calls=None,
                    ),
                ]
            )
            events.append(
                [
                    dict(
                        role="assistant",
                        style="vqa",
                        content="blue",
                        camera="observation.images.forward",
                        tool_calls=None,
                    )
                ]
                if row["frame_index"] == 1
                else []
            )
        table = table.append_column(
            "language_persistent", pa.array(persistent, type=language_arrow_type(persistent=True))
        )
        table = table.append_column(
            "language_events", pa.array(events, type=language_arrow_type(persistent=False))
        )
        pq.write_table(table, file)
    return path


@pytest.mark.parametrize("workers", [0, 1])
def test_annotated_v3_loads_without_framework_and_collates_by_sample(recording, tmp_path, workers):
    root = annotated_v3(recording, tmp_path / "annotated")
    fields = dict(language_persistent=True, language_events=False, optional_missing=False)
    data = AlohaMiniDataset(
        root,
        episodes=[0],
        allow_unannotated=True,
        state="recorded",
        image_size=(32, 32),
        chunk_size=2,
        include_task=True,
        annotation_fields=fields,
        cache_dir=tmp_path / "cache",
    )
    assert data.annotation_fields == ("language_persistent", "language_events")
    assert data.input_features["observation.state"].shape == (18,)
    first = data[0]["_annotations"]
    assert first["episode_index"] == first["frame_index"] == 0
    assert first["timestamp"] == 0
    assert len(first["fields"]["language_persistent"]) == 2
    assert first["fields"]["language_events"] == []
    first["fields"]["language_persistent"][0]["content"] = "changed"
    assert data[0]["_annotations"]["fields"]["language_persistent"][0]["content"] == "approach"
    loader = make_loader(data, batch_size=2, num_workers=workers, persistent_workers=False)
    batch = next(iter(loader))
    assert batch["action"].shape == (2, 2, 18)
    assert isinstance(batch["_annotations"], list) and len(batch["_annotations"]) == 2
    assert [len(sample["fields"]["language_events"]) for sample in batch["_annotations"]] == [0, 1]
    assert batch["_annotations"][1]["frame_index"] == 1
    from alohamini.learning.processor import Processor

    normalized = Processor({"action": {"mean": [0.0] * 18, "std": [1.0] * 18}})(batch)
    assert normalized["_annotations"] is batch["_annotations"]
    assert normalized["task"] == batch["task"]
    plain = AlohaMiniDataset(
        root,
        episodes=[0],
        allow_unannotated=True,
        state="recorded",
        image_size=(32, 32),
        chunk_size=2,
    )
    assert "_annotations" not in plain[0]


def test_required_annotation_columns_fail_explicitly(recording, tmp_path):
    root = annotated_v3(recording, tmp_path / "annotated")
    with pytest.raises(ValueError, match="Required annotation fields"):
        AlohaMiniDataset(
            root, episodes=[0], allow_unannotated=True, annotation_fields={"missing": True}
        )


def test_annotation_camera_and_column_semantics_are_validated():
    with pytest.raises(ValueError, match="recorded camera"):
        validate_language_rows(
            [dict(role="assistant", style="vqa", content="blue")], persistent=False
        )
    with pytest.raises(ValueError, match="another column"):
        validate_language_rows(
            [dict(role="assistant", style="subtask", timestamp=0)], persistent=False
        )
    with pytest.raises(ValueError, match="must not bind"):
        validate_language_rows(
            [
                dict(
                    role="assistant",
                    style="motion",
                    timestamp=0,
                    camera="observation.images.forward",
                )
            ],
            persistent=True,
        )


def test_annotation_contract_saved_in_native_checkpoint(recording, tmp_path):
    from alohamini.learning.policy import save_checkpoint
    from alohamini.policies.registry import algorithm

    root = annotated_v3(recording, tmp_path / "annotated")
    data = AlohaMiniDataset(
        root,
        episodes=[0],
        state="recorded",
        image_size=(32, 32),
        chunk_size=3,
        allow_unannotated=True,
        annotation_fields={"language_persistent": True},
    )
    recipe = algorithm("act")
    options = model_options()
    options.update(input_features=data.input_features, output_features=data.output_features)
    model = recipe.build(options)
    path = tmp_path / "checkpoint"
    save_checkpoint(path, model, recipe.statistics(data), data, training={})
    manifest = json.loads((path / "policy.json").read_text())
    preprocessing = json.loads((path / "preprocessing.json").read_text())
    assert manifest["annotations"] == preprocessing["annotations"]
    assert manifest["annotations"]["fields"] == ["language_persistent"]


def test_loss_count_context_is_explicit_and_independent_of_language_annotations(
    recording, tmp_path
):
    import torch

    root = annotated_v3(recording, tmp_path / "annotated")
    options = dict(
        root=root,
        episodes=[0],
        state="recorded",
        allow_unannotated=True,
        include_task=True,
        image_size=(32, 32),
        annotation_fields={"language_persistent": True},
        delta_indices={"observation.state": [-1, 0], "action": [0, 1]},
    )
    minimal = AlohaMiniDataset(**options)
    metadata = minimal.action_metadata(0)
    assert "_annotations" in metadata
    assert "observation.state" not in metadata and "task" not in metadata
    requested = AlohaMiniDataset(**options, loss_metadata_fields=("observation.state", "task"))
    metadata = requested.action_metadata(0)
    sample = requested[0]
    torch.testing.assert_close(metadata["observation.state"], sample["observation.state"])
    assert metadata["task"] == sample["task"]

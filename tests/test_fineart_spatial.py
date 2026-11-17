"""Camera-bound normalized VQA uses reserved loc tokens and real VLM gradients."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from test_fineart_vla import tiny_options
from test_language_data import annotated_v3
from test_native_learning import recording as recording

from alohamini.datasets.language import language_arrow_type
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.loading import make_loader
from alohamini.policies.fineart_vla.adapter import FineARTVLAAlgorithm
from alohamini.policies.fineart_vla.spatial import location, spatial_answer


def test_spatial_coordinates_require_declared_format_and_units():
    assert spatial_answer(dict(label="cup", point_format="xy", point=[0.25, 0.5])) == (
        "cup <loc0512><loc0256>"
    )
    assert spatial_answer(
        dict(detections=[dict(label="cup", bbox_format="xyxy", bbox=[0, 0.25, 0.5, 1])])
    ) == ("cup <loc0256><loc0000><loc1023><loc0512>")
    for value in (float("nan"), float("inf"), -0.1, 320, True):
        with pytest.raises(ValueError, match="coordinates|image fractions"):
            location(value)
    with pytest.raises(ValueError, match="bbox_format"):
        spatial_answer(dict(detections=[dict(label="cup", bbox=[0, 0, 1, 1])]))
    with pytest.raises(ValueError, match="ordered"):
        spatial_answer(dict(detections=[dict(label="cup", bbox_format="xyxy", bbox=[1, 0, 0, 1])]))


def test_vqa_routes_by_camera_and_trains_loc_tokens(recording, tmp_path, monkeypatch):
    recipe = FineARTVLAAlgorithm()
    options = tiny_options(tmp_path / "assets", monkeypatch)
    options["vqa_cameras"] = ["forward"]
    root = annotated_v3(recording, tmp_path / "data")
    for path in (root / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        events = table["language_events"].to_pylist()
        for i, row in enumerate(table.to_pylist()):
            if row["frame_index"] == 1:
                events[i] = [
                    dict(
                        role=role,
                        style="vqa",
                        content=content,
                        camera="observation.images.forward",
                        tool_calls=None,
                    )
                    for role, content in [
                        ("user", "Where is the cup?"),
                        (
                            "assistant",
                            json.dumps(dict(label="cup", point_format="xy", point=[0.25, 0.5])),
                        ),
                    ]
                ]
        table = table.set_column(
            table.schema.get_field_index("language_events"),
            "language_events",
            pa.array(events, type=language_arrow_type(persistent=False)),
        )
        pq.write_table(table, path)
    data = AlohaMiniDataset(
        root,
        episodes=[0],
        allow_unannotated=True,
        state="recorded",
        image_size=(32, 32),
        **recipe.sample_spec(options),
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    model = recipe.build(options)
    stats = recipe.statistics(data)
    processor = recipe.processor(model, stats, "cpu")
    raw = next(iter(make_loader(data, batch_size=len(data))))
    prepared = processor(raw)
    labels = prepared["text_labels"][1]
    target = labels[labels != -100].tolist()
    tokenizer = processor.text.tokenizer
    assert tokenizer.convert_tokens_to_ids("<loc0512>") in target
    assert tokenizer.convert_tokens_to_ids("<loc0256>") in target
    assert target[-1] == tokenizer.eos_token_id
    assert not prepared["predict_actions"][1]
    loss, metrics = model(prepared)
    loss.backward()
    assert torch.isfinite(loss) and metrics["text_loss"] > 0
    head = model.network.paligemma_with_expert.paligemma.lm_head
    assert head.weight.grad[1024:2048].abs().sum() > 0
    processor.config.fast_skip_tokens = 128
    with pytest.raises(ValueError, match="fast_skip_tokens"):
        processor.text(raw)

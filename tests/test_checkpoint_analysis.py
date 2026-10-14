"""Checkpoint lineage, immutable split selection and aligned demonstration analysis."""

import json
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from alohamini.analysis.comparison import prepare
from alohamini.analysis.metrics import analyze
from alohamini.analysis.provenance import (
    capture_policy,
    digest,
    reference_episode,
    resolve_context,
    verify_reference,
    write_evaluation,
)
from alohamini.analysis.source import load_episode


@pytest.fixture
def example(tmp_path):
    root = tmp_path / "demonstrations"
    (root / "meta").mkdir(parents=True)
    feature = {"dtype": "float32", "shape": [1], "names": ["arm_left_joint.pos"]}
    info = {
        "format": "alohamini-episodes",
        "fps": 30,
        "features": {"action": feature},
        "cameras": ["wrist_left"],
        "robot_metadata": {
            "robot_model": "alohamini2pro",
            "motors": {"arm_left_joint": {"normalization": "range_m100_100"}},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    for ep in range(2):
        path = root / f"episodes/episode_{ep:06d}"
        (path / "images/wrist_left").mkdir(parents=True)
        rows = []
        for i in range(4):
            relative = f"images/wrist_left/frame_{i:06d}.png"
            Image.new("RGB", (8, 8), (i * 20, ep * 20, 0)).save(path / relative)
            rows.append(
                {
                    "episode_index": ep,
                    "frame_index": i,
                    "timestamp": i / 30,
                    "action": [float(i)],
                    "observation.images.wrist_left": relative,
                }
            )
        pq.write_table(pa.Table.from_pylist(rows), path / "frames.parquet")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    hashes = {
        str(p.relative_to(root)): digest(p)
        for p in root.rglob("*")
        if p.is_file() and p.suffix in (".parquet", ".json")
    }
    manifest = {
        "format": "alohamini-policy",
        "version": 1,
        "kind": "act",
        "config": {"n_action_steps": 10, "chunk_size": 3, "temporal_ensemble_coeff": None},
        "state": "none",
        "cameras": ["wrist_left"],
        "image_size": [8, 8],
        "source_info": info,
        "training": {"dataset": str(root), "train_episodes": [0], "val_episodes": [1]},
        "table_sha256": hashes,
    }
    (checkpoint / "policy.json").write_text(json.dumps(manifest))
    (checkpoint / "model.safetensors").write_bytes(b"test-only-weight-identity")
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    return SimpleNamespace(
        root=root, checkpoint=checkpoint, manifest=manifest, evaluation=evaluation
    )


def test_recorded_checkpoint_and_effective_overrides_are_automatic(example):
    @dataclass
    class Config:
        n_action_steps: int = 1
        temporal_ensemble_coeff: float = 0.01

    policy = SimpleNamespace(
        checkpoint_path=example.checkpoint,
        config=Config(),
        fixed_dimensions={"targets": {"arm_left_joint.pos": 4}},
        calibration_mode="strict",
    )
    captured = capture_policy(policy)
    write_evaluation(example.evaluation, policy, captured, fps=30, task="pick", num_episodes=2)
    context = resolve_context(example.evaluation)
    assert context["checkpoint_source"] == "recorded"
    assert context["effective_policy_config"]["n_action_steps"] == 1
    assert context["checkpoint"]["config"]["n_action_steps"] == 10
    assert context["training_root"] == str(example.root)
    assert context["training_available"]
    assert reference_episode(context) == 0
    assert reference_episode(resolve_context(example.evaluation, reference_split="val")) == 1
    with pytest.raises(FileExistsError):
        write_evaluation(example.evaluation, policy, captured, fps=30, task="pick", num_episodes=2)


def test_legacy_context_does_not_invent_runtime_defaults(example):
    context = resolve_context(example.evaluation, checkpoint=example.checkpoint)
    assert context["effective_policy_config"] is None
    assert context["checkpoint_source"] == "user_supplied_for_legacy_recording"
    assert context["verified_metadata"] == ["meta/info.json"]
    with pytest.raises(ValueError, match="train_episodes"):
        reference_episode(context, 1)


def test_moved_dataset_is_verified_without_media_reads(example, tmp_path):
    moved = tmp_path / "moved"
    example.root.rename(moved)
    context = resolve_context(
        example.evaluation, checkpoint=example.checkpoint, training_root=moved
    )
    assert context["training_available"]
    reference = load_episode(moved, 0)
    verify_reference(context, reference)
    assert "episodes/episode_000000/frames.parquet" in context["verified_reference_files"]
    path = moved / "episodes/episode_000000/frames.parquet"
    table = pq.read_table(path)
    pq.write_table(
        table.set_column(table.schema.get_field_index("action"), "action", pa.array([[99.0]] * 4)),
        path,
    )
    with pytest.raises(ValueError, match="指纹不符"):
        verify_reference(context, load_episode(moved, 0))


def test_recorded_checkpoint_cannot_be_silently_replaced(example):
    policy = SimpleNamespace(checkpoint_path=example.checkpoint, config={})
    write_evaluation(
        example.evaluation, policy, capture_policy(policy), fps=30, task="pick", num_episodes=1
    )
    (example.checkpoint / "model.safetensors").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="weights_sha256"):
        resolve_context(example.evaluation, checkpoint=example.checkpoint)


def test_default_profile_aligns_stages_and_hides_control(example):
    context = resolve_context(example.evaluation, checkpoint=example.checkpoint)
    evaluation, reference = (analyze(load_episode(example.root, i)) for i in (0, 1))
    prepare(evaluation, context, reference=reference, anchors=(0.02, 0.03))
    assert evaluation.summary["comparison"]["alignment"] == "user_stage_anchors"
    assert evaluation.summary["comparison"]["pointwise_error"] is None
    assert evaluation.summary["comparison"]["comparable_fields"] == ["arm_left_joint.pos"]
    assert not any(k.startswith(("recording/", "currents_ma/")) for k in evaluation.series)
    assert all(list(v) == ["recorded_action"] for v in evaluation.series.values())
    assert "command_hz_estimate" not in evaluation.summary
    assert evaluation.anchor_s == 0.02
    assert reference.anchor_s == 0.03


def test_unknown_units_never_become_comparable(example):
    context = resolve_context(example.evaluation, checkpoint=example.checkpoint)
    evaluation, reference = (analyze(load_episode(example.root, i)) for i in (0, 1))
    reference.summary["actions"]["arm_left_joint.pos"]["unit"] = "recorded units (unverified)"
    prepare(evaluation, context, reference=reference)
    assert evaluation.summary["comparison"]["comparable_fields"] == []


def test_cli_generates_comparison_from_saved_checkpoint_without_model_load(example, tmp_path):
    from alohamini.analysis.cli import main

    policy = SimpleNamespace(checkpoint_path=example.checkpoint, config={"n_action_steps": 1})
    write_evaluation(
        example.root, policy, capture_policy(policy), fps=30, task="pick", num_episodes=1
    )
    output = tmp_path / "report"
    with patch(
        "alohamini.analysis.probe.probe_reference", side_effect=AssertionError("not opt-in")
    ):
        main([str(example.root), "--viewer", "none", "--output", str(output)])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["model_context"]["checkpoint_source"] == "recorded"
    assert summary["comparison"]["episode"] == 0
    assert summary["performance"]["video_keyframe_reads"] == 0


def test_rerun_and_html_reference_export(example, tmp_path):
    import rerun as rr

    from alohamini.analysis.media import MediaReader
    from alohamini.analysis.report import export_html
    from alohamini.analysis.viewer import show

    context = resolve_context(example.evaluation, checkpoint=example.checkpoint)
    evaluation, reference = (analyze(load_episode(example.root, i)) for i in (0, 1))
    prepare(evaluation, context, reference=reference, anchors=(0.02, 0.03))
    reader = MediaReader(tmp_path / "cache")
    output = tmp_path / "export"
    output.mkdir()
    rr.set_strict_mode(True)
    try:
        show(evaluation, reader, spawn=False, rrd=output / "recording.rrd")
        export_html(evaluation, output, reader, keyframe_times=[0.04])
        assert {x["run"] for x in evaluation.summary["keyframes"]} == {"reference", "evaluation"}
        assert (output / "recording.rrd").stat().st_size > 1000
        assert "demonstration" in (output / "report.html").read_text()
    finally:
        rr.set_strict_mode(False)
        reader.close()


def test_probe_uses_future_demonstration_labels_and_masks_episode_end(example, tmp_path):
    import torch

    from alohamini.analysis.media import MediaReader
    from alohamini.analysis.probe import probe_reference

    context = resolve_context(example.evaluation, checkpoint=example.checkpoint)
    reference = analyze(load_episode(example.root, 0))
    calls = []

    class Policy:
        def __init__(self, *args, **kwargs):
            self.selection = None
            self.manifest = example.manifest
            self.names = ["arm_left_joint.pos"]
            self.fixed_dimensions = None
            self.fps = 30

        def reset(self):
            calls.append("reset")

        def predict(self, sample):
            assert "action" not in sample
            assert tuple(sample["observation.images.wrist_left"].shape) == (3, 8, 8)
            return torch.zeros((3, 1))

    output = tmp_path / "probe"
    output.mkdir()
    reader = MediaReader(tmp_path / "cache")
    try:
        with patch("alohamini.learning.policy.NativePolicy", Policy):
            result = probe_reference(context, reference, reader, output, samples=2, device="cpu")
        assert calls == ["reset", "reset"]
        assert result["frame_indices"] == [0, 3]
        assert result["metrics"]["arm_left_joint.pos"]["valid_action_steps"] == 4
        assert result["metrics"]["arm_left_joint.pos"]["chunk_mae"] == 1.5
        assert result["metrics"]["arm_left_joint.pos"]["first_action_mae"] == 1.5
        with np.load(output / "demonstration_predictions.npz") as saved:
            assert saved["valid"].tolist() == [[True, True, True], [True, False, False]]
    finally:
        reader.close()

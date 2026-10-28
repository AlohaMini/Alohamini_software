"""Offline probe contracts: real small policies and a language-model boundary stub."""

import json
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image
from test_diffusion_fastwam import diffusion_options
from test_native_learning import model_options
from test_native_learning import recording as recording

from alohamini.analysis.media import MediaReader
from alohamini.analysis.metrics import analyze
from alohamini.analysis.probe import probe_reference
from alohamini.analysis.provenance import describe_checkpoint
from alohamini.analysis.source import load_episode
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, make_policy, save_checkpoint
from alohamini.learning.state import RecordedStateSelection
from alohamini.policies.configuration import PolicyFeature
from alohamini.policies.registry import algorithm
from alohamini.policies.smolvla.processor_smolvla import SmolVLAProcessor


def invoke(context, reference, tmp_path, *, samples=2, seed=31):
    output = tmp_path / "probe"
    output.mkdir()
    reader = MediaReader(tmp_path / "keyframes")
    try:
        result = probe_reference(
            context, reference, reader, output, samples=samples, device="cpu", seed=seed
        )
    finally:
        reader.close()
    with np.load(output / "demonstration_predictions.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    assert all(not value.dtype.hasobject for value in arrays.values())
    assert json.loads(arrays["metadata_json"].item()) == result
    assert json.loads((output / "demonstration_probe.json").read_text()) == result
    return result, arrays


def tiny_checkpoint(recording, tmp_path, kind, *, state="joint_position,base_velocity,lift_height"):
    adapter = algorithm(kind)
    options = diffusion_options() if kind == "diffusion" else model_options()
    data = AlohaMiniDataset(
        recording,
        episodes=[0],
        state=state,
        image_size=(32, 32),
        **adapter.sample_spec(options, cameras=["forward"]),
    )
    options["input_features"] = data.input_features
    options["output_features"] = data.output_features
    stats = adapter.statistics(data, options)
    model = make_policy(kind, options, stats)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model,
        stats,
        data,
        training={"dataset": str(recording), "train_episodes": [0], "val_episodes": [1]},
    )
    context = {
        "checkpoint": describe_checkpoint(checkpoint, hash_weights=True),
        "reference_split": "train",
    }
    reference = analyze(load_episode(recording, 0, all_numeric=True, include_task=True))
    return context, reference, data


@pytest.mark.parametrize("kind", ["act", "am_act", "diffusion"])
def test_real_tiny_checkpoint_probe_matches_deployment_predict_and_declared_offsets(
    recording, tmp_path, kind
):
    context, reference, data = tiny_checkpoint(recording, tmp_path, kind)
    result, arrays = invoke(context, reference, tmp_path)
    deployed = NativePolicy(context["checkpoint"]["path"], offline=True)
    offsets = np.asarray(deployed.manifest["sample_windows"]["action"])
    assert result["schema_version"] == 2 and result["kind"] == kind
    assert (
        result["checkpoint_identity"]["weights_sha256"] == context["checkpoint"]["weights_sha256"]
    )
    assert result["window"]["action_offsets"] == offsets.tolist()
    assert result["window"]["predicted_steps"] == len(offsets)
    np.testing.assert_array_equal(arrays["action_offsets"], offsets)
    assert arrays["frame_index"].tolist() == [0, 3]
    assert arrays["sample_seed"].tolist() == [31, 34]
    assert arrays["action_names"].tolist() == deployed.names
    assert arrays["action_units"].tolist() == result["action_units"]
    for sample, index in enumerate(arrays["row_index"]):
        random.seed(int(arrays["sample_seed"][sample]))
        np.random.seed(int(arrays["sample_seed"][sample]))
        torch.manual_seed(int(arrays["sample_seed"][sample]))
        deployed.reset()
        expected = deployed.predict(data[int(index)]).numpy()
        np.testing.assert_array_equal(arrays["prediction"][sample], expected)
        for step, delta in enumerate(offsets):
            target = int(index) + delta
            inside = 0 <= target < len(reference.data.rows)
            assert arrays["target_row_index"][sample, step] == (target if inside else -1)
            assert arrays["valid"][sample, step] == (inside and delta >= 0)
            if inside:
                np.testing.assert_array_equal(
                    arrays["target"][sample, step], reference.data.rows[target]["action"]
                )
    first = offsets.tolist().index(0)
    for column, name in enumerate(deployed.names):
        error = np.abs(arrays["prediction"][:, :, column] - arrays["target"][:, :, column])
        valid = arrays["coordinate_valid"][:, :, column]
        metric = result["metrics"][name]
        assert metric["chunk_mae"] == pytest.approx(float(error[valid].mean()))
        assert metric["first_action_mae"] == pytest.approx(float(error[:, first].mean()))
        assert metric["valid_action_steps"] == int(valid.sum())


def test_real_velocity_state_requires_all_numeric_source_columns(recording, tmp_path, monkeypatch):
    context, reference, _ = tiny_checkpoint(recording, tmp_path, "act", state="joint_velocity")
    assert "observation.motor_velocity_raw" in reference.data.rows[0]
    loaded, predict, observed = (
        NativePolicy(context["checkpoint"]["path"], offline=True),
        NativePolicy.predict,
        [],
    )

    def capture(self, sample):
        observed.append(sample["observation.state"].clone())
        return predict(self, sample)

    monkeypatch.setattr(NativePolicy, "predict", capture)
    invoke(context, reference, tmp_path)
    for actual, index in zip(observed, (0, 3), strict=True):
        np.testing.assert_array_equal(
            actual.numpy(), loaded.selection.frame(reference.data.rows[index])
        )


@pytest.fixture
def language_stub(tmp_path, monkeypatch):
    """Use real NativePolicy.predict and SmolVLA preprocessing, never a VLM download."""
    root = tmp_path / "demonstration"
    (root / "meta").mkdir(parents=True)
    episode = root / "episodes/episode_000000"
    (episode / "images/forward").mkdir(parents=True)
    names = ["joint_a.pos", "joint_b.pos"]
    vector = {
        "dtype": "float32",
        "shape": [2],
        "names": names,
        "units": dict.fromkeys(names, "rad"),
    }
    info = {
        "format": "alohamini-episodes",
        "fps": 30,
        "cameras": ["forward"],
        "training_only": {"source": "original_lerobot_v3"},
        "features": {
            "action": vector,
            "observation.state": deepcopy(vector),
            "observation.images.forward": {"dtype": "image", "shape": [4, 4, 3]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    rows = []
    for index, frame in enumerate([0, 2, 5]):
        relative = f"images/forward/frame_{frame:06d}.png"
        Image.new("RGB", (4, 4), (index * 30, 0, 0)).save(episode / relative)
        rows.append(
            {
                "episode_index": 0,
                "frame_index": frame,
                "timestamp": frame / 30,
                "observation.state": [10.0 + index * 2, 100.0 + index * 5],
                "action": [20.0 + index * 10, 200.0 + index * 20],
                "observation.images.forward": relative,
                "task": "pick",
                "task_index": 0,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), episode / "frames.parquet")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = {
        "format": "alohamini-policy",
        "version": 1,
        "kind": "smolvla",
        "config": {"chunk_size": 3, "n_action_steps": 2},
        "state": "recorded",
        "cameras": ["forward"],
        "image_size": [4, 4],
        "source_info": info,
        "sample_windows": {"action": [0, 1, 2]},
        "training": {},
    }
    (checkpoint / "policy.json").write_text(json.dumps(manifest))
    (checkpoint / "model.safetensors").write_bytes(b"language-boundary-stub-not-model-weights")
    events = SimpleNamespace(samples=[], normalized=[], contexts=[], prompts=[], resets=0)
    stats = {
        "observation.state": {"mean": [10.0, 100.0], "std": [2.0, 5.0]},
        "action": {"mean": [1.0, 2.0], "std": [3.0, 4.0]},
    }
    config = SimpleNamespace(
        input_features={
            "observation.state": PolicyFeature("STATE", (2,)),
            "observation.images.forward": PolicyFeature("VISUAL", (3, 4, 4)),
        },
        output_features={"action": PolicyFeature("ACTION", (2,))},
        image_features={"observation.images.forward": PolicyFeature("VISUAL", (3, 4, 4))},
        normalization_mapping={"STATE": "MEAN_STD", "VISUAL": "IDENTITY", "ACTION": "MEAN_STD"},
        tokenizer_max_length=8,
        pad_language_to="max_length",
        inference_action_scale_dims=(),
        inference_action_scale=1.0,
    )

    def tokenizer(prompts, **kwargs):
        events.prompts.extend(prompts)
        return {"input_ids": torch.tensor([[4, 5]]), "attention_mask": torch.tensor([[1, 1]])}

    class ContextProcessor(SmolVLAProcessor):
        def __call__(self, batch):
            result = super().__call__(batch)
            result["_action_context"] = batch["observation.state"].clone()
            return result

        def action(self, tensor, *, context=None):
            assert context is not None
            return self.numeric.action(tensor) + context[:, None, :]

    class Model(torch.nn.Module):
        def reset(self):
            events.resets += 1

        def predict_action_chunk(self, batch):
            assert "action" not in batch and "task" not in batch
            events.normalized.append(batch["observation.state"].clone())
            events.contexts.append(batch["_action_context"].clone())
            assert batch["observation.language.tokens"].tolist() == [[4, 5]]
            return torch.zeros(1, 3, 2)

    class Policy(NativePolicy):
        def __init__(self, path, *, device, offline):
            assert Path(path) == checkpoint and device == "cpu" and offline
            self.manifest, self.source = manifest, info
            self.names, self.fps, self.config = names, 30, config
            self.selection = RecordedStateSelection(info, "recorded")
            self.fixed_dimensions, self.fixed_state_references, self.fixed_current = None, {}, None
            self.algorithm = algorithm("smolvla")
            self.processor, self.model = ContextProcessor(config, stats, tokenizer), Model()

        def predict(self, sample):
            assert "action" not in sample
            events.samples.append(
                {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in sample.items()
                }
            )
            return super().predict(sample)

    monkeypatch.setattr("alohamini.learning.policy.NativePolicy", Policy)
    context = {
        "checkpoint": describe_checkpoint(checkpoint, hash_weights=True),
        "reference_split": "val",
        "deployment": {"task": "wrong evaluation task"},
    }
    reference = analyze(load_episode(root, 0, all_numeric=True, include_task=True))
    return SimpleNamespace(
        context=context,
        reference=reference,
        events=events,
        manifest=manifest,
        root=root,
        checkpoint=checkpoint,
        policy_class=Policy,
    )


def test_language_stub_uses_row_task_normalization_relative_context_and_row_offsets(
    language_stub, tmp_path
):
    case = language_stub
    case.reference.data.rows[-1]["task"] = "place  here\n"
    result, arrays = invoke(case.context, case.reference, tmp_path)
    assert case.events.prompts == ["pick\n", "place  here\n"]
    assert case.events.resets == 2
    np.testing.assert_array_equal(case.events.normalized[0].numpy(), [[0.0, 0.0]])
    np.testing.assert_array_equal(case.events.normalized[1].numpy(), [[2.0, 2.0]])
    np.testing.assert_array_equal(arrays["prediction"][0], [[11.0, 102.0]] * 3)
    np.testing.assert_array_equal(arrays["prediction"][1], [[15.0, 112.0]] * 3)
    assert arrays["target_frame_index"].tolist() == [[0, 2, 5], [5, -1, -1]]
    assert arrays["valid"].tolist() == [[True, True, True], [True, False, False]]
    assert arrays["sample_seed"].tolist() == [31, 36]
    assert result["metrics"]["joint_a.pos"]["chunk_mae"] == pytest.approx(20.5)
    assert result["metrics"]["joint_a.pos"]["first_action_mae"] == pytest.approx(17.0)
    assert result["metrics"]["joint_a.pos"]["persistence_chunk_mae"] == pytest.approx(21.5)
    assert result["baseline_source"] == "current_observation_same_named_coordinates"
    assert json.loads(arrays["observation_frame_indices"][1])["observation.state"] == [5]


@pytest.mark.parametrize("task", [None, "", "   "])
def test_language_probe_rejects_missing_original_task_before_prediction(
    language_stub, tmp_path, task
):
    case = language_stub
    case.reference.data.rows[0]["task"] = task
    with pytest.raises(ValueError, match="任务"):
        invoke(case.context, case.reference, tmp_path)
    assert not case.events.samples and not case.events.prompts
    assert not (tmp_path / "probe/demonstration_predictions.npz").exists()


def test_coordinate_masks_do_not_discard_other_finite_action_dimensions(language_stub, tmp_path):
    case = language_stub
    case.reference.data.rows[1]["action"][0] = float("nan")
    result, arrays = invoke(case.context, case.reference, tmp_path)
    assert arrays["coordinate_valid"][0, 1].tolist() == [False, True]
    assert not arrays["valid"][0, 1]
    assert result["metrics"]["joint_a.pos"]["valid_action_steps"] == 3
    assert result["metrics"]["joint_b.pos"]["valid_action_steps"] == 4


@pytest.mark.parametrize("field", ["action", "observation.state"])
def test_reordered_named_columns_are_rejected_instead_of_silently_reinterpreted(
    language_stub, tmp_path, field
):
    case = language_stub
    case.reference.data.info = deepcopy(case.reference.data.info)
    case.reference.data.info["features"][field]["names"].reverse()
    for row in case.reference.data.rows:
        row[field].reverse()
    with pytest.raises(ValueError, match="字段|顺序"):
        invoke(case.context, case.reference, tmp_path)
    assert not case.events.samples


def test_display_clock_segment_selects_queries_without_changing_episode_training_windows(
    language_stub, tmp_path
):
    case = language_stub
    case.reference.data.sidecars = [
        {"host_timing": {"state_sample_monotonic_s": stamp, "clock_id": "clock"}}
        for stamp in (100.0, 100.1, 1.0)
    ]
    reference = analyze(case.reference.data, segment=0)
    _, arrays = invoke(case.context, reference, tmp_path)
    assert arrays["frame_index"].tolist() == [0, 2]
    assert arrays["target_frame_index"].tolist() == [[0, 2, 5], [2, 5, -1]]
    assert arrays["valid"].tolist() == [[True, True, True], [True, True, False]]


def test_changed_checkpoint_is_rejected_before_loading_language_model(language_stub, tmp_path):
    case = language_stub
    (case.checkpoint / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint.*改变"):
        invoke(case.context, case.reference, tmp_path)
    assert not case.events.samples and not case.events.prompts


@pytest.mark.parametrize("state_unit", [None, "deg"])
def test_persistence_baseline_does_not_assume_unknown_or_different_state_units(
    language_stub, tmp_path, state_unit
):
    case = language_stub
    units = {"joint_b.pos": "rad"}
    if state_unit is not None:
        units["joint_a.pos"] = state_unit
    case.manifest["source_info"]["features"]["observation.state"]["units"] = units
    case.reference.data.info["features"]["observation.state"]["units"] = deepcopy(units)
    (case.checkpoint / "policy.json").write_text(json.dumps(case.manifest))
    case.context["checkpoint"] = describe_checkpoint(case.checkpoint, hash_weights=True)
    result, arrays = invoke(case.context, case.reference, tmp_path)
    assert np.isnan(arrays["baseline"][:, :, 0]).all()
    assert np.isfinite(arrays["baseline"][:, :, 1]).all()
    first = result["metrics"]["joint_a.pos"]
    assert first["chunk_mae"] is not None
    assert first["persistence_first_action_mae"] is None
    assert first["persistence_chunk_mae"] is None
    assert result["metrics"]["joint_b.pos"]["persistence_chunk_mae"] is not None


def test_future_observation_window_is_rejected_before_reading_pixels(
    language_stub, tmp_path, monkeypatch
):
    case = language_stub
    windows = {"action": [0, 1, 2], "observation.state": [0, 1]}
    case.manifest["sample_windows"] = windows
    (case.checkpoint / "policy.json").write_text(json.dumps(case.manifest))
    case.context["checkpoint"] = describe_checkpoint(case.checkpoint, hash_weights=True)
    monkeypatch.setattr(
        type(algorithm("smolvla")),
        "sample_spec",
        lambda *args, **kwargs: {"delta_indices": windows, "include_task": True},
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Reject a noncausal window before loading any observation")

    monkeypatch.setattr(MediaReader, "keyframe", forbidden)
    with pytest.raises(ValueError, match="未来观测"):
        invoke(case.context, case.reference, tmp_path)
    assert not case.events.samples and not case.events.prompts


@pytest.mark.parametrize("task_layout", ["pandas_index", "task_column"])
def test_v3_reader_resolves_original_task_index_and_keeps_extra_numeric_fields(
    tmp_path, task_layout
):
    root = tmp_path / "v3"
    for relative in ("meta/episodes/chunk-000", "data/chunk-000"):
        (root / relative).mkdir(parents=True)
    vector = {"dtype": "float32", "shape": [1], "names": ["joint.pos"]}
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": 3,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {
            "action": vector,
            "observation.state": vector,
            "motor_feedback.sample_started_s": {"dtype": "float64", "shape": [1]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    rows = [
        {
            "episode_index": 0,
            "frame_index": i,
            "timestamp": i / 30,
            "action": [float(i)],
            "observation.state": [float(i + 10)],
            "motor_feedback.sample_started_s": [100.0 + i / 30],
            "task_index": i % 2,
        }
        for i in range(3)
    ]
    pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [{"episode_index": 0, "length": 3, "data/chunk_index": 0, "data/file_index": 0}]
        ),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    tasks = root / "meta/tasks.parquet"
    if task_layout == "pandas_index":
        pd.DataFrame({"task_index": [0, 1]}, index=["pick", "place"]).to_parquet(tasks)
    else:
        pq.write_table(
            pa.Table.from_pylist(
                [{"task_index": 0, "task": "pick"}, {"task_index": 1, "task": "place"}]
            ),
            tasks,
        )
    original = load_episode(root, 0)
    assert all("task" not in row for row in original.rows)
    assert all("motor_feedback.sample_started_s" not in row for row in original.rows)
    selected = load_episode(root, 0, all_numeric=True, include_task=True)
    assert [row["task"] for row in selected.rows] == ["pick", "place", "pick"]
    assert [row["motor_feedback.sample_started_s"] for row in selected.rows] == [
        row["motor_feedback.sample_started_s"] for row in rows
    ]
    assert tasks in selected.signatures


def test_cli_real_act_exports_probe_identity_and_links_matching_artifacts(
    recording, tmp_path, monkeypatch
):
    from alohamini.analysis import cli, source

    context, _, _ = tiny_checkpoint(recording, tmp_path, "act")
    output = tmp_path / "report"
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    read_episode, reads = source.load_episode, []

    def tracked(root, episode, **kwargs):
        data = read_episode(root, episode, **kwargs)
        reads.append((kwargs, data))
        return data

    monkeypatch.setattr(source, "load_episode", tracked)
    assert (
        cli.main(
            [
                str(recording),
                "--checkpoint",
                context["checkpoint"]["path"],
                "--viewer",
                "none",
                "--export-html",
                "--probe-samples",
                "2",
                "--output",
                str(output),
            ]
        )
        == output
    )
    reference_reads = [
        data for kwargs, data in reads if kwargs.get("all_numeric") and kwargs.get("include_task")
    ]
    assert len(reference_reads) == 1
    assert reference_reads[0].rows[0]["task"] == "test"
    assert "observation.motor_velocity_raw" in reference_reads[0].rows[0]
    summary = json.loads((output / "summary.json").read_text())
    result = json.loads((output / "demonstration_probe.json").read_text())
    assert summary["demonstration_probe"] == result
    assert summary["model_context"]["verified_reference_files"]
    assert (
        result["checkpoint_identity"]["manifest_sha256"] == context["checkpoint"]["manifest_sha256"]
    )
    with np.load(output / "demonstration_predictions.npz", allow_pickle=False) as archive:
        assert json.loads(archive["metadata_json"].item()) == result
        assert archive["prediction"].shape == (2, 3, 18)
    html = (output / "report.html").read_text()
    for artifact in ("demonstration_predictions.npz", "demonstration_probe.json"):
        assert (output / artifact).is_file()
        assert any(f"href={quote}{artifact}{quote}" in html for quote in ('"', "'"))


def test_cdp_probe_uses_only_teacher_forced_past_actions_with_padding_and_sequence_index(
    language_stub, tmp_path, monkeypatch
):
    from alohamini.policies.cdp.adapter import CDPProcessor

    case = language_stub
    case.manifest["kind"] = "cdp"
    case.manifest["config"] = {"n_obs_steps": 2, "horizon": 5, "n_action_steps": 3}
    adapter = algorithm("cdp")
    windows = adapter.sample_spec(case.manifest["config"], cameras=["forward"])["delta_indices"]
    case.manifest["sample_windows"] = windows
    (case.checkpoint / "policy.json").write_text(json.dumps(case.manifest))
    case.context["checkpoint"] = describe_checkpoint(case.checkpoint, hash_weights=True)
    batches, samples = [], []

    class Model(torch.nn.Module):
        def reset(self):
            pass

        def predict_action_chunk(self, batch):
            assert "action" not in batch and "task" not in batch
            batches.append({key: value.clone() for key, value in batch.items()})
            return torch.zeros(1, 3, 2)

    class Policy(NativePolicy):
        def __init__(self, path, *, device, offline):
            assert Path(path) == case.checkpoint and device == "cpu" and offline
            self.manifest, self.source = case.manifest, case.manifest["source_info"]
            self.names = self.source["features"]["action"]["names"]
            self.fps, self.config = 30, SimpleNamespace()
            self.selection = RecordedStateSelection(self.source, "recorded")
            self.fixed_dimensions, self.fixed_state_references, self.fixed_current = None, {}, None
            self.algorithm, self.model = adapter, Model()
            self.processor = CDPProcessor(
                {"action": {"min": [0.0, 0.0], "max": [100.0, 1000.0]}},
                modes={"action": "MIN_MAX"},
            )

        def predict(self, sample):
            samples.append({key: value.clone() for key, value in sample.items()})
            return super().predict(sample)

    monkeypatch.setattr("alohamini.learning.policy.NativePolicy", Policy)
    result, arrays = invoke(case.context, case.reference, tmp_path)
    assert result["kind"] == "cdp"
    assert result["history_action_offsets"] == [-2, -1]
    assert any("示范中严格过去的动作作为条件" in note for note in result["notes"])
    assert [sample["sequence_index"].item() for sample in samples] == [0, 2]
    assert [batch["sequence_index"].tolist() for batch in batches] == [[0], [2]]
    assert samples[0]["past_action_is_pad"].tolist() == [True, True]
    assert samples[1]["past_action_is_pad"].tolist() == [False, False]
    assert batches[0]["past_action"].tolist() == [[[0.0, 0.0], [0.0, 0.0]]]
    # Row 2 is frame 5: history is rows 0/1, excluding the current [40, 240] label.
    assert samples[1]["past_action"].tolist() == [[20.0, 200.0], [30.0, 220.0]]
    torch.testing.assert_close(
        batches[1]["past_action"], torch.tensor([[[-0.6, -0.6], [-0.4, -0.56]]])
    )
    assert json.loads(arrays["observation_frame_indices"][1])["past_action"] == [0, 2]
    assert json.loads(arrays["observation_padding"][0])["past_action"] == [True, True]
    assert arrays["target_frame_index"].tolist() == [[0, 2, 5], [5, -1, -1]]

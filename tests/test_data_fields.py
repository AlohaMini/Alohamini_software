import json
import shutil
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from test_dataset import jpeg
from test_kinematic_fields import planar_spec
from test_native_learning import model_options
from test_native_learning import recording as recording
from test_spirit import tiny_spirit_options

from alohamini.datasets.lerobotv3 import export_lerobot
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.data_fields import prepare_fields
from alohamini.learning.policy import NativePolicy, make_processor, save_checkpoint
from alohamini.learning.statistics import training_statistics
from alohamini.policies.registry import algorithm


def source_dataset(recording, tmp_path, *, position_names=False):
    root = tmp_path / "model-angles"
    export_lerobot(recording, root)
    (root / "meta/alohamini.json").unlink()
    for path in (root / "meta/safety").glob("*.jsonl"):
        path.unlink()
    spec = planar_spec()
    if position_names:
        names = {"elbow": "elbow.pos", "shoulder": "shoulder.pos"}
        spec["source_names"] = [names.get(name, name) for name in spec["source_names"]]
        for coordinate in spec["arms"][0]["coordinates"]:
            coordinate["field"] = names[coordinate["field"]]
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    for key in ("observation.state", "action"):
        info["features"][key].update(names=spec["source_names"], shape=[3])
    info_path.write_text(json.dumps(info))
    for path in (root / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            state = np.array([250, -0.6, np.rad2deg(0.25)])
            state[2] += 0.002 * row["frame_index"]
            target = state.copy()
            target[2] += 0.005
            row["observation.state"], row["action"] = state.tolist(), target.tolist()
        for key in ("observation.state", "action"):
            table = table.set_column(
                table.schema.get_field_index(key),
                key,
                pa.array([row[key] for row in rows], type=pa.list_(pa.float32(), 3)),
            )
        pq.write_table(table, path)
    specification = tmp_path / "coordinates.json"
    specification.write_text(json.dumps(spec))
    source = AlohaMiniDataset(
        root,
        episodes=[0],
        state="recorded",
        allow_unannotated=True,
        include_task=True,
        chunk_size=3,
        image_size=(32, 32),
        exclude_fixed_state=False,
    )
    settings = dict(
        data=dict(
            state_representation="tcp",
            action_representation="tcp_delta",
            kinematics=dict(spec=str(specification)),
        )
    )
    return source, settings, specification


@pytest.mark.parametrize("kind", ["act", "spirit"])
def test_shared_fields_train_both_algorithms_and_reload_without_source_geometry(
    recording, tmp_path, kind
):
    torch.set_num_threads(2)
    source, settings, specification = source_dataset(recording, tmp_path)
    recipe = algorithm(kind)
    source.include_task = recipe.include_task
    options = model_options() if kind == "act" else tiny_spirit_options(tmp_path / "qwen")
    samples = prepare_fields(source, settings, recipe, options=options)
    assert source[0]["observation.state"].shape == (3,)
    assert samples[0]["observation.state"].shape == (7,)
    assert samples[0]["action"].shape == (3, 7)
    assert samples[0]["_recorded_action"].shape == (3, 3)
    assert "kinematics" not in settings["data"]
    options.update(input_features=samples.input_features, output_features=samples.output_features)
    artifact = training_statistics(recipe, samples, options, kind)
    assert artifact["contract"]["data_fields"] == settings["data"]
    stats = artifact["stats"]
    model = recipe.build(options, stats)
    processor = make_processor(model, stats, "cpu", data_fields=settings["data"])
    batch = next(iter(torch.utils.data.DataLoader(samples, batch_size=2)))
    prepared = processor(batch)
    assert prepared["observation.state"].shape == (2, 7)
    assert prepared["_action_context"].shape == (2, 3)
    loss, _ = model(prepared)
    assert torch.isfinite(loss)
    loss.backward()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, stats, samples, training={})
    specification.unlink()
    if kind == "spirit":
        shutil.rmtree(tmp_path / "qwen")
    loaded = NativePolicy(checkpoint, offline=True)
    assert loaded.manifest["data_fields"] == settings["data"]
    assert loaded.config.action_feature.shape == (7,)
    assert loaded.manifest["source_info"]["features"]["action"]["shape"] == [3]
    assert loaded.manifest["statistics_features"]["action"]["names"][0] == "delta.manipulator.x"
    assert len(loaded.manifest["physical_action_summary"]["mean"]) == 3
    assert len(loaded.manifest["recorded_state_stats"]["mean"]) == 3
    # ACT returns normalized targets; Spirit returns physical targets. Their
    # algorithm processors produce the matching train-time target in both cases.
    predicted = prepared["action"][:1].detach()
    loaded.model.predict_action_chunk = lambda batch: predicted
    result = loaded.predict(samples[0])
    np.testing.assert_allclose(result.numpy(), samples[0]["_recorded_action"].numpy(), atol=2e-3)
    raw_result = loaded.predict(source[0])
    np.testing.assert_allclose(result.numpy(), raw_result.numpy(), atol=1e-6)
    # Metrics must compare decoded physical targets with the recorded labels.
    from alohamini.learning.train import offline_evaluate

    loaded.model.predict_action_chunk = lambda batch: batch["action"]
    metrics = offline_evaluate(loaded, samples, batch_size=2, num_workers=0)
    assert max(metrics["mae_by_action"].values()) < 2e-3


@pytest.mark.parametrize(
    "selection,representation,state_width",
    [
        ("joint_position", "tcp", 6),
        ("none", "recorded", None),
    ],
)
def test_model_input_selection_is_independent_from_full_joint_action_context(
    recording, tmp_path, selection, representation, state_width, monkeypatch
):
    from alohamini.learning.evaluate import evaluate_checkpoint

    torch.set_num_threads(2)
    original, settings, specification = source_dataset(recording, tmp_path, position_names=True)
    source = AlohaMiniDataset(
        original.root,
        episodes=[0],
        state=selection,
        context_fields=["observation.state"],
        allow_unannotated=True,
        chunk_size=3,
        image_size=(32, 32),
        exclude_fixed_state=False,
    )
    settings["data"]["state_representation"] = representation
    recipe, options = algorithm("act"), model_options()
    samples = prepare_fields(source, settings, recipe, options=options)
    sample = samples[0]
    assert sample["_action_context"].shape == (3,)
    assert sample["action"].shape == (3, 7)
    if state_width is None:
        assert "observation.state" not in sample
        assert "observation.state" not in samples.input_features
    else:
        assert source[0]["observation.state"].shape == (2,)
        assert sample["observation.state"].shape == (state_width,)
    options.update(input_features=samples.input_features, output_features=samples.output_features)
    stats = training_statistics(recipe, samples, options, "act")["stats"]
    assert ("observation.state" in stats) == (state_width is not None)
    model = recipe.build(options, stats)
    processor = make_processor(model, stats, "cpu", data_fields=settings["data"])
    batch = next(iter(torch.utils.data.DataLoader(samples, batch_size=2)))
    prepared = processor(batch)
    loss, _ = model(prepared)
    assert torch.isfinite(loss)
    loss.backward()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, stats, samples, training={})
    specification.unlink()
    loaded = NativePolicy(checkpoint, offline=True)
    loaded.model.predict_action_chunk = lambda batch: prepared["action"][:1]
    expected = sample["_recorded_action"].numpy()
    np.testing.assert_allclose(loaded.predict(sample).numpy(), expected, atol=2e-3)
    raw_sample = {**source[0], "_action_context": sample["_action_context"]}
    np.testing.assert_allclose(loaded.predict(raw_sample).numpy(), expected, atol=2e-3)
    if state_width is not None:
        with pytest.raises(ValueError, match="context|shape"):
            loaded.predict(source[0])
        assert len(loaded.manifest["recorded_state_stats"]["mean"]) == 2
        # Fixed-input controls operate on raw selected fields before FK. They must
        # leave the real observation anchor and action labels intact.
        loaded.fixed_state_references = {0: -0.59}
        adapted = loaded.fixed_input(sample)
        assert adapted["observation.state"][0] == pytest.approx(-0.59)
        assert "_fields_encoded" not in adapted
        torch.testing.assert_close(adapted["_action_context"], sample["_action_context"])
        torch.testing.assert_close(adapted["action"], sample["_recorded_action"])
        expected_state = loaded.field_transform.geometry.absolute(
            [250, -0.59, source[0]["observation.state"][1].item()]
        )[:6]
        transformed, _ = loaded.field_transform.encode(
            adapted["observation.state"].numpy(), anchor=adapted["_action_context"].numpy()
        )
        np.testing.assert_allclose(transformed, expected_state, atol=1e-7)
    # The public evaluator reconstructs the same field view from the checkpoint.
    loaded.model.predict_action_chunk = lambda batch: batch["action"]
    monkeypatch.setattr("alohamini.learning.evaluate.NativePolicy", lambda *a, **kw: loaded)
    metrics = evaluate_checkpoint(
        checkpoint,
        source.root,
        episodes=[0],
        batch_size=2,
        num_workers=0,
    )
    assert max(metrics["mae_by_action"].values()) < 2e-3
    # Exercise live observation assembly and the physical action queue using
    # synthetic snapshots; no Host connection or hardware control occurs.
    loaded.fixed_state_references = {}
    monkeypatch.setattr(loaded, "_check_calibration", lambda snapshot: None)
    names = source.info["features"]["observation.state"]["names"]
    snapshot = SimpleNamespace(
        payload=dict(zip(names, sample["_action_context"].tolist(), strict=True)),
        images={"forward": jpeg()},
    )
    calls = []

    def predict(batch):
        calls.append(batch["_action_context"].clone())
        return prepared["action"][:1]

    loaded.model.predict_action_chunk = predict
    loaded.reset()
    first = loaded.select_action(snapshot)
    snapshot.payload[names[1]] += 0.05
    second = loaded.select_action(snapshot)
    np.testing.assert_allclose([first[name] for name in loaded.names], expected[0], atol=2e-3)
    np.testing.assert_allclose([second[name] for name in loaded.names], expected[1], atol=2e-3)
    assert len(calls) == 1


def test_vision_only_tcp_actions_still_validate_recorded_joint_context(recording, tmp_path):
    source, settings, _ = source_dataset(recording, tmp_path)
    settings["data"]["state_representation"] = "tcp"
    vision = AlohaMiniDataset(
        source.root,
        episodes=[0],
        state="none",
        context_fields=["observation.state"],
        allow_unannotated=True,
        chunk_size=3,
        image_size=(32, 32),
    )
    with pytest.raises(ValueError, match="selected numeric inputs"):
        prepare_fields(vision, settings, algorithm("act"), options=model_options())
    path = next((source.root / "data").rglob("*.parquet"))
    table = pq.read_table(path)
    values = table["observation.state"].to_pylist()
    values[0][1] = float("nan")
    table = table.set_column(
        table.schema.get_field_index("observation.state"),
        "observation.state",
        pa.array(values, type=pa.list_(pa.float32(), 3)),
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="observation.state.*NaN|observation context"):
        AlohaMiniDataset(
            source.root,
            episodes=[0],
            state="none",
            context_fields=["observation.state"],
            allow_unannotated=True,
            chunk_size=3,
            image_size=(32, 32),
        )


def test_state_and_action_representation_are_independent_and_unsupported_adapters_reject(
    recording, tmp_path
):
    source, settings, _ = source_dataset(recording, tmp_path)
    options = model_options()
    settings["data"]["state_representation"] = "recorded"
    samples = prepare_fields(source, settings, algorithm("act"), options=options)
    assert samples.input_features["observation.state"].shape == (3,)
    assert samples.output_features["action"].shape == (7,)
    np.testing.assert_array_equal(samples[0]["observation.state"], source[0]["observation.state"])
    with pytest.raises(ValueError, match="not registered"):
        prepare_fields(source, settings, algorithm("pi05"), options={})
    with pytest.raises(ValueError, match="temporal blending"):
        prepare_fields(
            source, settings, algorithm("act"), options={"temporal_ensemble_coeff": 0.01}
        )


@pytest.mark.parametrize("representation", ["recorded", "tcp_absolute"])
def test_absolute_statistics_preserve_unique_rows_and_relative_windows_are_explicit(
    recording, tmp_path, representation
):
    source, settings, _ = source_dataset(recording, tmp_path)
    settings["data"]["action_representation"] = representation
    samples = prepare_fields(source, settings, algorithm("act"), options=model_options())
    rows = sorted(source._used_rows["action"])
    expected = np.asarray([source.rows[i]["action"] for i in rows])
    if representation == "tcp_absolute":
        expected = samples.field_transform.geometry.absolute(expected)
    stats = samples.statistics(keys=["action"])["action"]
    np.testing.assert_allclose(stats["mean"], expected.mean(0), atol=1e-8)
    np.testing.assert_allclose(stats["std"], expected.std(0), atol=1e-8)
    values = samples[0]["action"].numpy()
    restored = samples.field_transform.decode(samples[0]["_action_context"].numpy(), values)
    np.testing.assert_allclose(restored, samples[0]["_recorded_action"].numpy(), atol=2e-3)


def test_field_cli_configuration_is_at_data_level(monkeypatch):
    from alohamini.learning.train_config import parse_training_args

    with monkeypatch.context() as context:
        context.setattr("alohamini.learning.train_config.resolve_pretrained", lambda x: x)
        cfg, _ = parse_training_args(
            [
                "--policy.type=spirit",
                "--dataset.root=/dataset",
                "--output_dir=/output",
                "--state=joint_position,base_velocity,lift_height",
                "--data.kinematics.reference_pose=upright_forward",
            ]
        )
    assert cfg["data"] == dict(
        state_representation="tcp",
        action_representation="tcp_delta",
        kinematics=dict(reference_pose="upright_forward"),
    )
    assert cfg["model"]["action_representation"] == "recorded"
    assert not cfg["model"].get("kinematics_path")


def test_training_entry_resolves_fields_and_resumes_without_input_description(recording, tmp_path):
    from alohamini.learning.checkpoint import read_checkpoint
    from alohamini.learning.train import train

    source, settings, specification = source_dataset(recording, tmp_path)
    output = tmp_path / "run"
    options = model_options()
    cfg = dict(
        policy="act",
        dataset=str(source.root),
        output_dir=str(output),
        state="recorded",
        allow_unannotated=True,
        train_episodes=[0],
        val_episodes=[],
        device="cpu",
        image_size=[32, 32],
        steps=1,
        batch_size=2,
        num_workers=0,
        cpu_threads=2,
        model=options,
        data=settings["data"],
        save_freq=1,
    )
    checkpoint = train(cfg)
    model_path, manifest = read_checkpoint(checkpoint)
    assert manifest["data_fields"]["action_representation"] == "tcp_delta"
    assert "kinematics" not in manifest["data_fields"]
    specification.unlink()
    saved = json.loads((model_path / "train_config.json").read_text())
    saved.update(resume=True, _checkpoint=str(model_path.parent), steps=2)
    resumed = train(saved)
    _, restored = read_checkpoint(resumed)
    assert restored["data_fields"] == manifest["data_fields"]
    assert restored["training"]["steps"] == 2

"""Named relative joint targets across training, statistics and portable inference."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from test_gr00t_lewm import groot_assets as groot_assets
from test_gr00t_lewm import options as groot_options
from test_native_learning import recording as recording

from alohamini.datasets.field_transforms import FieldTransforms
from alohamini.learning.checkpoint import resolve_pretrained
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.data_fields import prepare_fields
from alohamini.learning.policy import NativePolicy, make_processor, save_checkpoint
from alohamini.learning.statistics import training_statistics
from alohamini.policies.registry import algorithm


def test_named_joint_deltas_keep_gripper_and_extra_axes_absolute():
    names = ["gripper", "joint_b", "base", "joint_a", "lift"]
    fields = FieldTransforms(
        dict(
            action_representation="joint_delta",
            source_action_names=names,
            source_state_names=["observed_a", "lift", "observed_b"],
            joint_delta_fields={"joint_a": "observed_a", "joint_b": "observed_b"},
        )
    )
    anchor = np.array([30.0, 250.0, -10.0])
    actions = np.array([[70.0, -7.0, 0.2, 35.0, 260.0], [0.0, -12.0, -0.3, 31.0, 240.0]])
    _, encoded = fields.encode(anchor, actions)
    np.testing.assert_array_equal(
        encoded, [[70.0, 3.0, 0.2, 5.0, 260.0], [0.0, -2.0, -0.3, 1.0, 240.0]]
    )
    np.testing.assert_array_equal(fields.decode(anchor, encoded), actions)
    metadata = fields.features(None, [], dict(shape=[5], names=names))["action"]
    assert metadata["names"] == ["gripper", "delta.joint_b", "base", "delta.joint_a", "lift"]
    metadata = fields.features(
        dict(shape=[3], names=["observed_a", "lift", "observed_b"]),
        ["deg", "mm", "rad"],
        dict(shape=[5], names=names),
    )["action"]
    assert metadata["units"][1] == "rad" and metadata["units"][3] == "deg"
    np.testing.assert_array_equal(
        fields.decode(np.stack([anchor, anchor]), np.stack([encoded, encoded])),
        np.stack([actions, actions]),
    )


@pytest.mark.parametrize("mapping", [{}, {"missing": "q"}, {"q": "missing"}])
def test_joint_delta_requires_declared_fields(mapping):
    with pytest.raises(ValueError, match="joint_delta"):
        FieldTransforms(
            dict(
                action_representation="joint_delta",
                source_action_names=["q", "gripper"],
                source_state_names=["q", "gripper"],
                joint_delta_fields=mapping,
            )
        )


@pytest.mark.parametrize("representation", ["joint_delta", "recorded"])
def test_gr00t_joint_actions_statistics_checkpoint_and_legacy_reload(
    recording, groot_assets, tmp_path, representation
):
    recipe = algorithm("gr00t_n17")
    opts = groot_options("gr00t_n17", groot_assets)
    for key in ("state_names", "action_names", "state_slot_map", "action_slot_map"):
        opts.pop(key)
    opts["use_percentiles"] = False
    source = AlohaMiniDataset(
        recording,
        episodes=[0],
        cameras=["forward"],
        image_size=(32, 32),
        state="joint_position",
        **recipe.sample_spec(opts, cameras=["forward"]),
    )
    settings = recipe.apply_preset({"data": {"action_representation": representation}})
    samples = prepare_fields(source, settings, recipe, options=opts)
    opts.update(input_features=samples.input_features, output_features=samples.output_features)
    recipe.prepare_options(opts, samples)
    if representation == "joint_delta":
        mapping = settings["data"]["joint_delta_fields"]
        assert len(mapping) == 12 and not any("gripper" in key for key in mapping)
        assert samples.field_transform.geometry is None  # No FK/calibration required.
        expected = np.asarray(source[0]["action"], dtype=float)
        names = source.info["features"]["action"]["names"]
        anchor = np.asarray(source.rows[source.sample_indices[0]]["observation.state"])
        for name in mapping:
            index = names.index(name)
            expected[:, index] -= anchor[index]
        np.testing.assert_allclose(samples[0]["action"], expected, atol=1e-6)
    artifact = training_statistics(recipe, samples, opts, "gr00t_n17")
    stats = artifact["stats"]
    expected = (
        torch.cat([samples[i]["action"] for i in range(len(samples))]).double()
        if representation == "joint_delta"
        else np.asarray([source.rows[i]["action"] for i in sorted(source._used_rows["action"])])
    )
    np.testing.assert_allclose(stats["action"]["mean"], expected.mean(0), atol=1e-6)
    if representation == "joint_delta":
        quantiles = recipe.statistics(samples, {"use_percentiles": True})
        np.testing.assert_allclose(
            quantiles["action"]["q01"], np.quantile(expected, 0.01, axis=0), atol=1e-6
        )
    model = recipe.build(opts, stats)
    processor = make_processor(model, stats, "cpu", data_fields=settings.get("data"))
    batch = processor(next(iter(torch.utils.data.DataLoader(samples, batch_size=2))))
    loss, _ = model(batch)
    assert torch.isfinite(loss)
    loss.backward()
    checkpoint = tmp_path / "policy"
    save_checkpoint(checkpoint, model, stats, samples, training={})
    cfg = recipe.apply_preset(resolve_pretrained({"pretrained_path": str(checkpoint)}))
    for resume in (False, True):
        cfg = recipe.apply_preset({**cfg, "resume": resume})
        assert cfg["data"]["action_representation"] == representation
    loaded = NativePolicy(checkpoint, offline=True)
    target = batch["action"][:1].detach()
    loaded.model.predict_action_chunk = lambda _: target
    actual = loaded.predict(samples[0]).numpy()
    reference = samples[0].get("_recorded_action", samples[0]["action"]).numpy()
    np.testing.assert_allclose(actual, reference, atol=1e-5)


def test_gr00t_default_and_explicit_tcp_override():
    recipe = algorithm("gr00t_n17")
    assert recipe.apply_preset({})["data"]["action_representation"] == "joint_delta"
    settings = {"data": {"state_representation": "tcp", "action_representation": "tcp_delta"}}
    original = deepcopy(settings)
    assert recipe.apply_preset(settings)["data"] == settings["data"]
    assert settings == original
    legacy = {"resume": True, "paper_preset": {"name": "gr00t-n17-recorded-v1"}}
    assert recipe.apply_preset(legacy) == legacy


def test_external_joint_fields_require_explicit_binding(recording, tmp_path):
    from test_data_fields import source_dataset

    source, _, _ = source_dataset(recording, tmp_path)
    recipe = algorithm("gr00t_n17")
    settings = recipe.apply_preset({})
    with pytest.raises(ValueError, match="explicit data.joint_delta_fields"):
        prepare_fields(source, settings, recipe)
    settings["data"]["joint_delta_fields"] = {"elbow": "elbow", "shoulder": "shoulder"}
    samples = prepare_fields(source, settings, recipe)
    raw = source[0]["action"].numpy()
    encoded = samples[0]["action"].numpy()
    np.testing.assert_array_equal(encoded[:, 0], raw[:, 0])
    np.testing.assert_allclose(
        samples.field_transform.decode(samples[0]["_action_context"].numpy(), encoded),
        raw,
        atol=1e-6,
    )


def test_joint_delta_cli_records_mapping(tmp_path):
    from alohamini.learning.train_config import parse_training_args

    cfg, _ = parse_training_args(
        [
            "--policy.type=gr00t_n17",
            f"--dataset.root={tmp_path}",
            "--run_name=joint_delta_cli",
            "--data.action_representation=joint_delta",
            '--data.joint_delta_fields={"joint_target":"joint_state"}',
        ]
    )
    assert cfg["data"]["joint_delta_fields"] == {"joint_target": "joint_state"}

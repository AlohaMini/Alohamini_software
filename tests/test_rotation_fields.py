"""Rotation conventions, reference frames and complete policy data round trips."""

import json
import shutil
from contextlib import nullcontext

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from test_data_fields import source_dataset
from test_gr00t_lewm import groot_assets as groot_assets
from test_gr00t_lewm import options as groot_options
from test_hyvla_openwam import tiny_options, tokenizer_assets
from test_kinematic_fields import descriptor, joint_state
from test_native_learning import recording as recording

from alohamini.datasets.action_transforms import RelativeActions
from alohamini.datasets.field_transforms import FieldTransforms
from alohamini.datasets.rotations import RotationEncoding
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.data_fields import prepare_fields
from alohamini.learning.policy import NativePolicy, make_processor, save_checkpoint
from alohamini.learning.statistics import training_statistics
from alohamini.policies.registry import algorithm


def test_rotation_6d_rows_roundtrip_and_projection_near_pi():
    codec = RotationEncoding("rotation_6d")
    matrices = Rotation.from_rotvec(
        [[0.3, -0.7, 0.8], [np.pi - 1e-8, 0, 0], [np.pi + 1e-8, 0, 0], [0, 0, 0]]
    ).as_matrix()
    encoded = codec.encode(matrices)
    np.testing.assert_allclose(encoded, matrices[:, :2, :].reshape(-1, 6))
    np.testing.assert_allclose(codec.decode(encoded), matrices, atol=1e-14)
    assert np.linalg.norm(encoded[1] - encoded[2]) < 1e-7
    # Unnormalized, nonorthogonal model output must still decode onto SO(3).
    distorted = encoded.copy()
    distorted[:, :3] *= 2
    distorted[:, 3:] = 3 * encoded[:, 3:] + 0.4 * encoded[:, :3]
    np.testing.assert_allclose(codec.decode(distorted), matrices, atol=1e-14)
    np.testing.assert_allclose(codec.decode(distorted * 1e200), matrices, atol=1e-14)
    assert codec.decode(np.empty((2, 0, 6))).shape == (2, 0, 3, 3)


@pytest.mark.parametrize("values", [[0] * 6, [1, 0, 0, 2, 0, 0], [np.nan] * 6])
def test_invalid_rotation_6d_cannot_reach_ik(values):
    with pytest.raises(ValueError, match="Degenerate|finite"):
        RotationEncoding("rotation_6d").decode(values)


@pytest.mark.parametrize("representation", ["rotvec", "rotation_6d"])
@pytest.mark.parametrize("frame", ["base", "tool"])
def test_relative_pose_matches_homogeneous_transform_and_fixed_anchor(representation, frame):
    rotation = RotationEncoding(representation)
    anchor_matrix = np.eye(4)
    anchor_matrix[:3, :3] = Rotation.from_rotvec([0.6, -0.2, 0.1]).as_matrix()
    anchor_matrix[:3, 3] = [0.3, 0.1, -0.2]
    targets = np.repeat(np.eye(4)[None], 2, axis=0)
    targets[:, :3, :3] = Rotation.from_rotvec([[-0.1, 0.4, 0.3], [0.3, 0.1, -0.4]]).as_matrix()
    targets[:, :3, 3] = [[0.4, 0.2, 0.1], [0.5, 0.3, 0.2]]
    anchor = np.r_[anchor_matrix[:3, 3], rotation.encode(anchor_matrix[:3, :3]), 0.04]
    actions = np.concatenate(
        [targets[:, :3, 3], rotation.encode(targets[:, :3, :3]), [[0.07], [0.02]]], axis=-1
    )
    codec = RelativeActions(
        len(anchor),
        linear_dims=[0, 1, 2],
        rotation_groups=[list(range(3, len(anchor) - 1))],
        rotation_representation=representation,
        reference_frame=frame,
        position_groups=[[0, 1, 2]] if frame == "tool" else (),
    )
    delta = codec.encode(anchor, actions)
    expected = np.linalg.inv(anchor_matrix) @ targets
    if frame == "base":
        expected[:, :3, 3] = targets[:, :3, 3] - anchor_matrix[:3, 3]
        expected[:, :3, :3] = targets[:, :3, :3] @ anchor_matrix[:3, :3].T
    np.testing.assert_allclose(delta[:, :3], expected[:, :3, 3], atol=1e-14)
    np.testing.assert_allclose(rotation.decode(delta[:, 3:-1]), expected[:, :3, :3], atol=1e-14)
    np.testing.assert_array_equal(delta[:, -1], actions[:, -1])
    np.testing.assert_allclose(codec.decode(anchor, delta), actions, atol=1e-14)
    batched = codec.encode(np.stack([anchor] * 3), np.stack([actions] * 3))
    np.testing.assert_allclose(batched, np.stack([delta] * 3), atol=1e-14)
    restored = RelativeActions(**json.loads(json.dumps(codec.configuration())))
    np.testing.assert_allclose(restored.decode(anchor, delta), actions, atol=1e-14)
    assert codec.encode(anchor, np.empty((0, len(anchor)))).shape == (0, len(anchor))


@pytest.mark.parametrize("action", ["recorded", "tcp_absolute", "tcp_delta"])
def test_dual_arm_24d_metadata_grippers_passthrough_and_ik(action):
    fields = FieldTransforms(
        dict(
            state_representation="tcp",
            action_representation=action,
            rotation_representation="rotation_6d",
            kinematic_fields=descriptor(),
            tcp_delta_frame="tool" if action == "tcp_delta" else "base",
        )
    )
    state = joint_state()
    actions = np.repeat(state[None], 2, axis=0)
    actions[:, 0] += [0.005, 0.01]
    actions[:, 7] += [-0.005, -0.01]
    actions[:, [6, 13]] += [1, -2]
    actions[:, -4:] += [0.02, 0.03, 2, 10]
    observed, encoded = fields.encode(state, actions)
    assert observed.shape == (24,)
    assert encoded.shape == (2, 18 if action == "recorded" else 24)
    np.testing.assert_allclose(observed[[9, 19]], [0.04, 0.06])
    np.testing.assert_array_equal(observed[-4:], state[-4:])
    np.testing.assert_array_equal(encoded[:, -4:], actions[:, -4:])
    feature = fields.geometry.feature()
    assert feature["names"][3:9] == [f"left.{n}" for n in RotationEncoding("rotation_6d").names]
    assert feature["units"][3:9] == ["dimensionless"] * 6
    restored = fields.decode(state, encoded)
    np.testing.assert_allclose(
        fields.geometry.absolute(restored), fields.geometry.absolute(actions), atol=1e-4
    )
    np.testing.assert_allclose(restored[:, [6, 13]], actions[:, [6, 13]])
    np.testing.assert_array_equal(restored[:, -4:], actions[:, -4:])


@pytest.mark.parametrize("kind", ["hyvla05", "gr00t_n17"])
def test_vla_tool_6d_training_statistics_checkpoint_and_physical_decode(
    recording, groot_assets, tmp_path, kind
):
    torch.set_num_threads(2)
    source, settings, specification = source_dataset(recording, tmp_path)
    settings["data"].update(rotation_representation="rotation_6d", tcp_delta_frame="tool")
    recipe = algorithm(kind)
    if kind == "hyvla05":
        assets = tmp_path / "hyvla"
        tokenizer_assets(assets)
        options = tiny_options(kind, assets)
        options.update(chunk_size=3)
        options["network"].update(max_state_dim=32, max_action_dim=32)
    else:
        assets = groot_assets
        options = groot_options(kind, assets)
        for key in ("state_names", "action_names", "state_slot_map", "action_slot_map"):
            options.pop(key)
        options["use_percentiles"] = False  # Preserve test labels at the distribution tails.
    source = AlohaMiniDataset(
        source.root,
        episodes=[0],
        state="recorded",
        allow_unannotated=True,
        image_size=(32, 32),
        exclude_fixed_state=False,
        **recipe.sample_spec(options, cameras=source.cameras),
    )
    samples = prepare_fields(source, settings, recipe, options=options)
    options.update(input_features=samples.input_features, output_features=samples.output_features)
    if kind == "gr00t_n17":
        recipe.prepare_options(options, samples)
        assert options["action_names"][3] == "delta.manipulator.r00"
    artifact = training_statistics(recipe, samples, options, kind)
    assert artifact["contract"]["data_fields"]["tcp_delta_frame"] == "tool"
    assert samples[0]["action"].shape == (3, 10)
    stats = artifact["stats"]
    # Compare fitted moments with all encoded action windows, not recorded joints.
    actions = torch.cat([samples[i]["action"] for i in range(len(samples))]).double()
    np.testing.assert_allclose(stats["action"]["mean"], actions.mean(0), atol=1e-7)
    if kind == "gr00t_n17":
        quantiles = recipe.statistics(samples, {"use_percentiles": True})
        np.testing.assert_allclose(
            quantiles["action"]["q01"], np.quantile(actions.numpy(), 0.01, axis=0), atol=1e-7
        )
        assert quantiles["action"]["min"] == quantiles["action"]["q01"]
    model = recipe.build(options, stats)
    processor = make_processor(model, stats, "cpu", data_fields=settings["data"])
    prepared = processor(next(iter(torch.utils.data.DataLoader(samples, batch_size=2))))
    with torch.autocast("cpu", dtype=torch.bfloat16) if kind == "hyvla05" else nullcontext():
        loss, _ = model(prepared)
    assert torch.isfinite(loss)
    loss.backward()
    model.eval()
    with (
        torch.no_grad(),
        torch.autocast("cpu", dtype=torch.bfloat16) if kind == "hyvla05" else nullcontext(),
    ):
        prediction = model.predict_action_chunk(prepared)
    assert prediction.shape == (2, 3, 10) and torch.isfinite(prediction).all()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, stats, samples, training={})
    specification.unlink()
    shutil.rmtree(assets)
    loaded = NativePolicy(checkpoint, offline=True)
    assert loaded.manifest["data_fields"] == settings["data"]
    assert loaded.config.action_feature.shape == (10,)
    predicted = prepared["action"][:1].detach()
    loaded.model.predict_action_chunk = lambda batch: predicted
    result = loaded.predict(samples[0])
    np.testing.assert_allclose(result.numpy(), samples[0]["_recorded_action"].numpy(), atol=2e-3)
    raw = {**source[0], "_action_context": samples[0]["_action_context"]}
    np.testing.assert_allclose(loaded.predict(raw).numpy(), result.numpy(), atol=1e-6)


@pytest.mark.parametrize(
    "config",
    [
        {"rotation_representation": "rotation_6d"},
        {"rotation_representation": "quaternion"},
        {"tcp_delta_frame": "tool"},
        {"state_representation": "tcp", "tcp_delta_frame": "tool"},
    ],
)
def test_inactive_or_invalid_rotation_and_frame_options_fail(config):
    with pytest.raises(ValueError, match="representation|tcp_delta"):
        FieldTransforms(config)

"""Small real AC-DiT networks, stage transfer and declared input contracts."""

import numpy as np
import pytest
import torch
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, make_policy, save_checkpoint
from alohamini.policies.acdit.adapter import ACDiTAlgorithm
from alohamini.policies.acdit.preset import apply_preset
from alohamini.policies.configuration import PolicyFeature

torch.set_num_threads(2)


@pytest.fixture
def siglip_assets(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, SiglipConfig, SiglipImageProcessor

    path = tmp_path / "siglip"
    config = SiglipConfig(
        vision_config={
            "hidden_size": 24,
            "intermediate_size": 48,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 32,
            "patch_size": 4,
        },
        text_config={
            "hidden_size": 24,
            "intermediate_size": 48,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "max_position_embeddings": 16,
            "vocab_size": 32,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "bos_token_id": 1,
        },
    )
    config.save_pretrained(path)
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "test": 3, "pick": 4}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        model_max_length=16,
    ).save_pretrained(path)
    SiglipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(path)
    return path


def tiny_options(assets, *, stage="mobility", points=True):
    from alohamini.datasets.record import state_names

    names = state_names("alohamini2pro")
    features = {
        "observation.state": PolicyFeature("STATE", (18,)),
        "observation.images.forward": PolicyFeature("VISUAL", (3, 32, 32)),
    }
    if points:
        features["observation.point_cloud"] = PolicyFeature("ENV", (32, 3))
    return dict(
        assets_path=str(assets),
        load_pretrained_encoders=False,
        stage=stage,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        mobility_hidden_dim=32,
        mobility_num_layers=2,
        mobility_num_heads=4,
        padded_dim=32,
        tokenizer_max_length=8,
        num_train_timesteps=10,
        num_inference_steps=2,
        pointcloud_frame="forward_camera",
        pointcloud_units="m",
        pointcloud_points=32,
        pointcloud_knn=4,
        pointcloud_stages=2,
        pointcloud_embed_dim=6,
        pointcloud_sampling="fps",
        pointcloud_projection_size=32,
        lora_dropout=0.0,
        image_augmentation=False,
        state_noise_snr=None,
        condition_mask_prob=0.0,
        use_pointcloud=points,
        state_names=names,
        action_names=names,
        state_slot_map={name: i for i, name in enumerate(names)},
        action_slot_map={name: i for i, name in enumerate(names)},
        input_features=features,
        output_features={"action": PolicyFeature("ACTION", (18,))},
    )


def raw_batch(points=True, batch_size=2):
    batch = {
        "observation.state": torch.randn(batch_size, 1, 18),
        "observation.images.forward": torch.rand(batch_size, 2, 3, 32, 32),
        "observation.language.tokens": torch.tensor([[3, 4, 0, 0]]).expand(batch_size, -1),
        "observation.language.attention_mask": torch.tensor([[True, True, False, False]]).expand(
            batch_size, -1
        ),
        "action": torch.randn(batch_size, 2, 18),
        "action_is_pad": torch.zeros(batch_size, 2, dtype=torch.bool),
    }
    if points:
        batch["observation.point_cloud"] = torch.rand(batch_size, 2, 32, 3)
    return batch


@pytest.mark.parametrize("points", [False, True])
@pytest.mark.parametrize("stage", ["mobility", "full"])
def test_real_network_gradients_modes_and_sampling(siglip_assets, points, stage):
    torch.manual_seed(11)
    model = make_policy("acdit", tiny_options(siglip_assets, stage=stage, points=points))
    batch = raw_batch(points)
    # Open a gradient path beyond the zero-initialized diffusion output layer.
    with torch.no_grad():
        active = model.mobility if stage == "mobility" else model.body
        active.model.final_layer.ffn_final.fc2.weight.normal_(std=0.02)
    loss, metrics = model(batch)
    loss.backward()
    assert loss.isfinite() and metrics["diffusion_loss"].isfinite()
    assert all(p.grad is None for p in model.siglip.parameters())
    if stage == "full":
        assert not model.mobility.training
        assert all(not p.requires_grad and p.grad is None for p in model.mobility.parameters())
        assert model.perception.image_proj[0].weight.grad.abs().sum() > 0
    if points:
        assert model.point_encoder.token_dim_transform.weight.grad.abs().sum() > 0
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for name, p in model.point_encoder.named_parameters()
            if "lora_B" in name
        )
    model.eval()
    if stage == "mobility":
        with pytest.raises(ValueError, match="pretraining checkpoint"):
            model.predict_action_chunk(batch)
    else:
        result = model.predict_action_chunk(batch)
        assert result.shape == (2, 2, 18) and result.isfinite().all()


def test_stage_one_supervises_only_named_base_controls(siglip_assets):
    model = make_policy("acdit", tiny_options(siglip_assets, points=False)).eval()
    batch = raw_batch(False)
    noise, timestep = torch.randn(2, 2, 32), torch.tensor([2, 7])
    first = model(batch, noise=noise, timestep=timestep)[0]
    batch["action"][:, :, :14] += 123
    second = model(batch, noise=noise, timestep=timestep)[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_paper_and_released_training_presets_are_distinct():
    paper = apply_preset({"policy": "acdit", "model": {"stage": "mobility"}})
    released = apply_preset(
        {"policy": "acdit", "model": {"stage": "mobility", "recipe": "released"}}
    )
    assert paper["steps"] == 20_000 and released["steps"] == 30_000
    assert paper["batch_size"] == 16 and released["batch_size"] == 24
    assert apply_preset({"model": {"stage": "full"}})["steps"] == 30_000
    assert apply_preset({"model": {"stage": "full", "recipe": "released"}})["steps"] == 20_000
    assert paper["image_resize_mode"] == "pad" and paper["image_pad_value"] == 0.5
    assert paper["trainable_dtype"] == "float32"


def test_shared_image_padding_preserves_geometry_and_tensor_ownership():
    from alohamini.learning.processor import image_tensor

    rgb = np.full((32, 64, 3), 255, np.uint8)
    padded = image_tensor(rgb, (64, 64), resize_mode="pad", pad_value=0.5)
    assert padded.shape == (3, 64, 64)
    assert (padded[:, :16] == 0.5).all() and (padded[:, 16:48] == 1).all()
    assert (image_tensor(rgb, (64, 64)) == 1).all()
    assert (rgb == 255).all()


@pytest.mark.parametrize("points", [False, True])
def test_zero_point_cloud_and_single_batch_shape(siglip_assets, points):
    model = make_policy("acdit", tiny_options(siglip_assets, stage="full", points=points)).eval()
    batch = raw_batch(points, batch_size=1)
    if points:
        batch["observation.point_cloud"].zero_()
    assert model.predict_action_chunk(batch).isfinite().all()


def test_rgb_checkpoint_transfers_complete_mobility_branch(recording, siglip_assets, tmp_path):
    from alohamini.learning.checkpoint import initialize_policy, resolve_pretrained

    component = ACDiTAlgorithm()
    opts = tiny_options(siglip_assets, points=False)
    samples = AlohaMiniDataset(
        recording,
        episodes=[0],
        image_size=(32, 32),
        image_resize_mode="pad",
        image_pad_value=0.5,
        **component.sample_spec(opts, cameras=["forward"]),
    )
    opts.update(input_features=samples.input_features, output_features=samples.output_features)
    opts["control_frequency"] = samples.info["fps"]
    component.prepare_options(opts, samples)
    stats = component.statistics(samples, opts)
    model = make_policy("acdit", opts)
    with torch.no_grad():
        model.mobility.img_adaptor[0].weight.fill_(0.123)
    source = tmp_path / "mobility"
    save_checkpoint(source, model, stats, samples, training={"trainable_dtype": "preserve"})
    settings = resolve_pretrained({"pretrained_path": str(source), "model": {"stage": "full"}})
    settings = apply_preset(settings)
    full_opts = component.options(settings, "cpu")
    full = make_policy("acdit", full_opts)
    initialize_policy(component, full, settings)
    for key, expected in model.mobility.state_dict().items():
        torch.testing.assert_close(full.mobility.state_dict()[key], expected, rtol=0, atol=0)
    assert all(not p.requires_grad for p in full.mobility.parameters())
    path = tmp_path / "full"
    save_checkpoint(path, full, stats, samples, training={"mixed_precision": "bfloat16"})
    import shutil

    shutil.rmtree(siglip_assets)
    shutil.rmtree(source)
    restored = NativePolicy(path, offline=True)
    for name, expected in full.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[name], expected, rtol=0, atol=0)
    assert restored.manifest["image_resize_mode"] == "pad"
    prepared = component.processor(restored.model, stats, "cpu")(
        {
            "observation.state": torch.zeros(1, 1, 18),
            "observation.images.forward": torch.zeros(1, 2, 3, 32, 32),
            "task": ["test"],
        }
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert restored.model.predict_action_chunk(prepared).shape == (1, 2, 18)
    from alohamini.learning.evaluate import evaluate_checkpoint

    report = evaluate_checkpoint(path, recording, episodes=[1], batch_size=2)
    assert report["valid_action_steps"] > 0


def test_dynamic_views_horizon_boolean_context_and_gradient_checkpointing(siglip_assets):
    opts = tiny_options(siglip_assets, stage="full", points=False)
    opts.update(
        chunk_size=3,
        n_action_steps=3,
        gradient_checkpointing=True,
        context_fields=["observation.is_grasped"],
        context_dim=1,
    )
    opts["input_features"].update(
        {
            "observation.images.wrist": PolicyFeature("VISUAL", (3, 32, 32)),
            "observation.is_grasped": PolicyFeature("ENV", (1,)),
        }
    )
    model = make_policy("acdit", opts)
    batch = raw_batch(False)
    batch["observation.images.wrist"] = torch.rand(2, 2, 3, 32, 32)
    batch["observation.is_grasped"] = torch.ones(2, 1, 1, dtype=torch.bool)
    batch["action"] = torch.randn(2, 3, 18)
    batch["action_is_pad"] = torch.zeros(2, 3, dtype=torch.bool)
    model(batch)[0].backward()
    model.eval()
    assert model.predict_action_chunk(batch).shape == (2, 3, 18)
    batch["observation.images.forward"] = batch["observation.images.forward"][:, -1]
    with pytest.raises(ValueError, match="explicit RGB history"):
        model.predict_action_chunk(batch)


def test_auxiliary_sensor_contract_never_synthesizes_missing_values():
    from alohamini.learning.auxiliary import auxiliary_features, snapshot_auxiliary

    key = "observation.point_cloud"
    info = {"features": {key: {"dtype": "float32", "shape": [4, 3]}}}
    assert auxiliary_features(info, {key: [-1, 0]}, {}) == {key: ("ENV", (4, 3))}
    features = {key: PolicyFeature("ENV", (4, 3))}
    with pytest.raises(ValueError, match="Missing required"):
        snapshot_auxiliary({}, features, info)
    with pytest.raises(ValueError, match="invalid shape"):
        snapshot_auxiliary({key: [[0, 0, 0]]}, features, info)
    assert snapshot_auxiliary({key: [[0, 0, 0]] * 4}, features, info)[key].shape == (4, 3)


@pytest.mark.parametrize(
    "invalid",
    [
        {"use_pointcloud": "false"},
        {"num_heads": 7},
        {"n_action_steps": 3},
        {"num_inference_steps": 11},
        {"base_action_names": ["unknown_control"]},
        {"pointcloud_frame": ""},
    ],
)
def test_invalid_configuration_is_rejected(siglip_assets, invalid):
    from alohamini.policies.acdit.configuration_acdit import ACDiTConfig

    with pytest.raises(ValueError):
        ACDiTConfig(**{**tiny_options(siglip_assets), **invalid})

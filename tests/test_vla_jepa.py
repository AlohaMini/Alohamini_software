"""Real reduced Qwen3, V-JEPA2, latent predictor and flow DiT integration."""

import json
from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
from test_gr00t_lewm import groot_assets as groot_assets
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.loading import collate_samples
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.policies.vla_jepa.adapter import VLAJEPAAlgorithm


def tiny_options(qwen):
    from transformers import VJEPA2Config, VJEPA2VideoProcessor

    jepa = qwen.parent / "vjepa2"
    config = VJEPA2Config(
        crop_size=16,
        patch_size=4,
        frames_per_clip=4,
        tubelet_size=2,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        mlp_ratio=2,
        pred_hidden_size=16,
        pred_num_attention_heads=4,
        pred_num_hidden_layers=1,
        num_pooler_layers=1,
    )
    config.save_pretrained(jepa)
    VJEPA2VideoProcessor(crop_size=16).save_pretrained(jepa)
    return dict(
        assets_path=str(qwen.parent),
        qwen_config=json.loads((qwen / "config.json").read_text()),
        jepa_config=config.to_dict(),
        dtype="float32",
        chunk_size=2,
        n_action_steps=1,
        num_video_frames=4,
        jepa_tubelet_size=2,
        num_action_tokens_per_timestep=2,
        num_embodied_action_tokens_per_instruction=2,
        num_inference_timesteps=2,
        action_model_type="DiT-test",
        action_hidden_size=32,
        action_num_layers=2,
        action_dropout=0.0,
        predictor_depth=2,
        predictor_num_heads=4,
        repeated_diffusion_steps=2,
        resize_images_to=None,
    )


def build_data(recording, qwen):
    recipe = VLAJEPAAlgorithm()
    options = tiny_options(qwen)
    data = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **recipe.sample_spec(options, cameras=("forward",)),
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    recipe.prepare_options(options, data)
    return recipe, options, data


@pytest.mark.parametrize(
    "device,dtype,amp,storage",
    [
        ("cpu", "float32", False, "preserve"),
        ("cuda", "bfloat16", True, "preserve"),
        ("cuda", "bfloat16", True, "float32"),
    ],
)
def test_native_world_action_backward_and_checkpoint(
    recording, groot_assets, tmp_path, monkeypatch, device, dtype, amp, storage
):
    from alohamini.learning.precision import apply_trainable_dtype
    from alohamini.policies import registry

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    recipe, options, data = build_data(recording, groot_assets)
    options["dtype"] = dtype
    stats = recipe.statistics(data)
    model = apply_trainable_dtype(recipe.build(options), storage).to(device)
    processor = recipe.processor(model, stats, device)
    raw = collate_samples([data[0], data[1]])
    batch = processor(raw)
    with torch.autocast(device, dtype=torch.bfloat16) if amp else nullcontext():
        loss, terms = model(batch)
    loss.backward()
    assert torch.isfinite(loss) and set(terms) == {"action_loss", "wm_loss"}
    assert model.model.action_model.action_encoder.layer1.weight.grad is not None
    assert model.model.video_predictor.predictor_embed.weight.grad is not None
    assert (
        model.model.qwen.model.model.language_model.layers[0].self_attn.q_proj.weight.grad
        is not None
    )
    assert all(
        p.grad is None and not p.requires_grad for p in model.model.video_encoder.parameters()
    )
    model.train()
    assert not model.model.video_encoder.training
    assert len(model.model.action_model.model.transformer_blocks) == 2
    assert len(model.model.video_predictor.predictor_blocks) == 2
    counts = recipe.make_loss_counter(model.config, stats)
    before = counts(collate_samples([data.action_metadata(i) for i in (0, 1)]))
    assert before == counts(raw) and before["_world_weight"] == 1
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16) if amp else nullcontext():
        prediction = model.predict_action_chunk(batch)
    assert prediction.shape == (2, 2, 18) and torch.isfinite(prediction).all()
    upper = processor.action(torch.full_like(prediction, 2))
    torch.testing.assert_close(
        upper, torch.as_tensor(stats["action"]["max"], device=device).expand_as(upper)
    )
    monkeypatch.setitem(
        registry.ALGORITHMS, "vla_jepa", ("alohamini.policies.vla_jepa.adapter", "VLAJEPAAlgorithm")
    )
    path = tmp_path / "checkpoint"
    save_checkpoint(
        path,
        model,
        stats,
        data,
        training={"mixed_precision": "bfloat16" if amp else "none", "trainable_dtype": storage},
    )
    loaded = NativePolicy(path, device=device, offline=True)
    assert loaded.predict(data[0]).shape == (2, 18)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)


def test_no_future_observation_leak_and_view_order(recording, groot_assets):
    recipe, options, data = build_data(recording, groot_assets)
    model = recipe.build(options).eval()
    processor = recipe.processor(model, recipe.statistics(data), "cpu")
    raw = collate_samples([data[0], data[1]])
    changed = deepcopy(raw)
    changed["observation.images.forward"][:, 1:] = 1 - changed["observation.images.forward"][:, 1:]
    first, second = processor(raw), processor(changed)
    with torch.no_grad():
        a, b = model.model.readouts(first), model.model.readouts(second)
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
    flat = torch.tensor([1.0, 2.0, 10.0, 20.0]).reshape(4, 1, 1)
    torch.testing.assert_close(
        model.model.merge_views(flat, 2, 2), torch.tensor([[[1.0, 2.0]], [[10.0, 20.0]]])
    )
    assert VLAJEPAAlgorithm.inference_spec(options, cameras=("forward",))["delta_indices"] == {
        "action": [0, 1]
    }


def test_metadata_masks_do_not_decode_video(recording, groot_assets, monkeypatch):
    recipe, options, data = build_data(recording, groot_assets)
    full = data[1]

    def no_decode(*args, **kwargs):
        raise AssertionError("metadata must not decode images")

    monkeypatch.setattr(
        data,
        "_value",
        lambda row, key: no_decode() if key.startswith("observation.images.") else torch.zeros(18),
    )
    meta = data.action_metadata(1)
    torch.testing.assert_close(
        meta["observation.images.forward_is_pad"], full["observation.images.forward_is_pad"]
    )


def test_causal_teacher_context_and_pre_norm_readout(recording, groot_assets, monkeypatch):
    recipe, options, data = build_data(recording, groot_assets)
    model = recipe.build(options).eval()
    processor = recipe.processor(model, recipe.statistics(data), "cpu")
    batch = processor(collate_samples([data[0], data[1]]))
    with torch.no_grad():
        embodied, readouts = model.model.readouts(batch)
        model.model.qwen.model.model.language_model.norm.weight.mul_(3)
        altered, _ = model.model.readouts(batch)
    torch.testing.assert_close(embodied, altered, rtol=0, atol=0)
    contexts = []
    original = model.model.video_predictor.forward

    def capture(context, *args, **kwargs):
        contexts.append(context.detach().clone())
        return original(context, *args, **kwargs)

    monkeypatch.setattr(model.model.video_predictor, "forward", capture)
    changed = {**batch, "video_pixels": batch["video_pixels"].clone()}
    changed["video_pixels"][:, 2:] += 2
    with torch.no_grad():
        model.model.world_loss(batch, readouts)
        model.model.world_loss(changed, readouts)
        torch.testing.assert_close(contexts[0], contexts[1], rtol=0, atol=0)
        model.config.world_model_context = "shared"
        model.model.world_loss(batch, readouts)
        model.model.world_loss(changed, readouts)
    assert not torch.allclose(contexts[2], contexts[3])


def test_shared_training_and_resume(recording, groot_assets, tmp_path, monkeypatch):
    from safetensors.torch import save_model

    from alohamini.learning.train import train
    from alohamini.learning.train_config import parse_training_args
    from alohamini.policies import registry

    recipe, options, data = build_data(recording, groot_assets)
    seed = recipe.build(options)
    save_model(seed.model.qwen.model, groot_assets / "model.safetensors")
    save_model(seed.model.video_encoder, groot_assets.parent / "vjepa2/model.safetensors")
    monkeypatch.setitem(
        registry.ALGORITHMS, "vla_jepa", ("alohamini.policies.vla_jepa.adapter", "VLAJEPAAlgorithm")
    )
    del seed
    output = tmp_path / "run"
    options.pop("input_features")
    options.pop("output_features")
    result = train(
        dict(
            policy="vla_jepa",
            pretrained_path=str(groot_assets.parent),
            dataset=str(recording),
            output_dir=str(output),
            device="cpu",
            cpu_threads=2,
            num_workers=0,
            train_episodes=[0],
            val_episodes=[1],
            state="joint_position,base_velocity,lift_height",
            image_size=[32, 32],
            steps=1,
            batch_size=1,
            gradient_accumulation_steps=2,
            mixed_precision="none",
            eval_steps=1,
            eval_batch_size=2,
            save_freq=1,
            log_freq=0,
            scheduler={"type": "none"},
            model=options,
        )
    )
    assert (result / "policy.json").is_file()
    metrics = json.loads((output / "validation-metrics.jsonl").read_text().splitlines()[0])
    assert metrics["metric_totals"]["wm_loss"]["count"] == 1
    resumed, _ = parse_training_args(
        ["--config_path", str(result / "train_config.json"), "--resume=true", "--steps=2"]
    )
    assert (train(resumed) / "policy.json").is_file()
    assert (output / "checkpoints/000002/training_state/state.pt").is_file()


def test_policy_loading_resets_declared_robot_and_view_heads(recording, groot_assets, tmp_path):
    from safetensors.torch import save_model

    recipe, options, data = build_data(recording, groot_assets)
    options.update(
        base_weights_mode="policy", retain_unused_predictor_heads=True, world_model_rope_legacy=True
    )
    source = recipe.build(options)
    processor = recipe.processor(source, recipe.statistics(data), "cpu")
    loss, _ = source(processor(collate_samples([data[0], data[1]])))
    loss.backward()
    assert source.model.video_predictor.extrinsics_encoder.weight.grad is None
    base = tmp_path / "base"
    base.mkdir()
    save_model(source, base / "model.safetensors")
    options["world_model_cameras"].append("observation.images.wrist")
    options["input_features"]["observation.images.wrist"] = options["input_features"][
        "observation.images.forward"
    ]
    target = recipe.build(options)
    with pytest.raises(ValueError, match="incompatible"):
        recipe.initialize(target, dict(pretrained_path=str(base)))
    options["reinitialize_world_heads"] = True
    target = recipe.build(options)
    recipe.initialize(target, dict(pretrained_path=str(base)))
    assert target.model.video_predictor.predictor_embed.in_features == 64
    assert target.model.video_predictor.predictor_proj.out_features == 64
    assert not target.model.video_predictor.state_encoder.weight.requires_grad
    assert not target.model.video_predictor.extrinsics_encoder.weight.requires_grad
    torch.testing.assert_close(
        target.model.action_model.model.proj_out_2.weight,
        source.model.action_model.model.proj_out_2.weight,
    )
    assert not torch.equal(
        target.model.action_model.action_encoder.layer1.weight,
        source.model.action_model.action_encoder.layer1.weight,
    )


def test_policy_can_drop_teacher_and_freeze_backbone(recording, groot_assets, tmp_path):
    from safetensors.torch import save_model

    recipe, options, data = build_data(recording, groot_assets)
    source = recipe.build(options)
    base = tmp_path / "base"
    base.mkdir()
    save_model(source, base / "model.safetensors")
    options.update(enable_world_model=False, freeze_qwen=True, base_weights_mode="policy")
    target = recipe.build(options)
    recipe.initialize(target, dict(pretrained_path=str(base)))
    recipe.load(base, target)
    assert target.model.video_encoder is None and target.model.video_predictor is None
    assert all(not p.requires_grad for p in target.model.qwen.parameters())
    processor = recipe.processor(target, recipe.statistics(data), "cpu")
    batch = processor(collate_samples([data[0], data[1]]))
    loss, metrics = target(batch)
    loss.backward()
    assert set(metrics) == {"action_loss"} and torch.isfinite(loss)
    assert target.model.action_model.action_encoder.layer1.weight.grad is not None


def test_released_geometry_and_protocol_do_not_override_new_optimizer(groot_assets, tmp_path):
    base = tmp_path / "policy-base"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps(
            dict(
                type="vla_jepa", optimizer_lr=999.0, scheduler_warmup_steps=999, action_num_layers=3
            )
        )
    )
    options = VLAJEPAAlgorithm.options(
        dict(
            pretrained_path=str(base),
            model=dict(assets_path=str(groot_assets.parent), action_num_layers=2),
        ),
        "cpu",
    )
    assert options["action_num_layers"] == 2
    assert options["base_weights_mode"] == "policy"
    assert options["retain_unused_predictor_heads"] and options["world_model_rope_legacy"]
    assert "optimizer_lr" not in options and "scheduler_warmup_steps" not in options

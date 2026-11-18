"""Real tiny Wan VAE, UMT5, causal streams and recorded-action alignment."""

from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.loading import collate_samples
from alohamini.policies.lingbot_va.adapter import LingBotVAAlgorithm


@pytest.fixture
def streaming_recording(tmp_path):
    return recording.__wrapped__(tmp_path, frames=12)


@pytest.fixture
def fresh_execution():
    from accelerate.state import AcceleratorState, GradientState

    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    yield
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


def tiny_options(path):
    from diffusers import AutoencoderKLWan
    from test_longwam import tiny_options as backbone_options
    from transformers import UMT5Config, UMT5EncoderModel

    base = backbone_options(path / "tokenizer")
    AutoencoderKLWan(**base["vae_config"]).save_pretrained(path / "vae")
    UMT5EncoderModel(UMT5Config(**base["text_encoder_config"])).save_pretrained(
        path / "text_encoder"
    )
    return dict(
        wan_pretrained_path=str(path),
        dtype="float32",
        attn_mode="torch",
        patch_size=(1, 1, 1),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        action_dim=30,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=2,
        height=16,
        width=16,
        vae_spatial_factor=8,
        frame_chunk_size=1,
        action_per_frame=4,
        attn_window=6,
        num_inference_steps=1,
        action_num_inference_steps=1,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        max_sequence_length=8,
    )


def assembled(root, path):
    recipe = LingBotVAAlgorithm()
    options = tiny_options(path)
    data = AlohaMiniDataset(
        root,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **recipe.sample_spec(options, cameras=("forward",)),
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    recipe.prepare_options(options, data)
    options["action_slots"] = {name: 29 - i for i, name in enumerate(options["action_names"])}
    stats = recipe.statistics(data)
    model = recipe.build(options)
    processor = recipe.processor(model, stats, "cpu")
    return recipe, options, data, stats, model, processor


def test_causal_masks_keep_samples_and_teacher_targets_separate():
    from alohamini.policies.lingbot_va.core import training_masks

    own, cross = training_masks(
        (2, 4, 2, 1, 1), (2, 30, 2, 1, 1), 0, 1, 30, (1, 1, 1), "cpu", 3, "torch"
    )
    assert own.shape == (16, 16) and cross.shape == (16, 6)
    # Layout: noisy video b0(t0,t1),b1(t0,t1), clean video, noisy action, clean action.
    assert own[0, 0] and not own[0, 1] and not own[0, 4]
    assert own[1, 4] and own[1, 12]  # next noisy video reads earlier clean video/action.
    assert not own[1, 5] and not own[1, 13]
    assert not own[0, 2] and not own[1, 6]
    assert cross[0, :3].all() and not cross[0, 3:].any()
    assert cross[2, 3:].all() and not cross[2, :3].any()


@pytest.mark.parametrize(
    "device,storage", [("cpu", "preserve"), ("cuda", "preserve"), ("cuda", "float32")]
)
def test_real_training_normalization_and_checkpoint(
    streaming_recording, tmp_path, monkeypatch, device, storage
):
    from alohamini.learning.policy import NativePolicy, save_checkpoint
    from alohamini.learning.precision import apply_trainable_dtype
    from alohamini.policies import registry

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    recipe, options, data, stats, model, processor = assembled(
        streaming_recording, tmp_path / "assets"
    )
    if device == "cuda":
        options["dtype"] = "bfloat16"
        model = recipe.build(options)
    model = apply_trainable_dtype(model, storage).to(device)
    processor = recipe.processor(model, stats, device)
    raw = collate_samples([data[0], data[1]])
    batch = processor(raw)
    model._ensure_frozen_modules()
    assert not model._vae.training
    assert all(not p.requires_grad for p in model._vae.parameters())
    assert next(model._vae.parameters()).dtype == (
        torch.bfloat16 if device == "cuda" else torch.float32
    )
    assert torch.get_default_dtype() == torch.float32
    with torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else nullcontext():
        loss, terms = model(batch)
    loss.backward()
    assert torch.isfinite(loss) and set(terms) == {"latent_loss", "action_loss"}
    assert model.transformer.action_embedder.weight.grad is not None
    assert model.transformer.patch_embedding_mlp.weight.grad is not None
    assert model.transformer.blocks[0].attn1.to_q.weight.grad is not None
    assert len(model.transformer.blocks) == 2
    assert all(
        p.grad is None
        for p in model.transformer.condition_embedder_action.text_embedder.parameters()
    )
    with torch.no_grad():
        latents, actions, mask, _ = model._build_training_streams(batch)
    assert latents.shape[2] == 2 and actions.shape == (2, 30, 2, 4, 1)
    slots = model.config.used_action_channel_ids
    bootstrap = actions[:, slots, 0, :, 0].transpose(1, 2)
    expected = processor.numeric({"action": torch.zeros(2, 4, 18)})["action"]
    torch.testing.assert_close(bootstrap.float(), expected.float(), rtol=0.01, atol=0.01)
    torch.testing.assert_close(
        actions[:, slots, 1, :, 0].transpose(1, 2).float(),
        batch["action"].float(),
        rtol=0.01,
        atol=0.01,
    )
    assert mask[:, slots].all() and mask.sum() == 2 * 18 * 8
    path = tmp_path / "checkpoint"
    monkeypatch.setitem(
        registry.ALGORITHMS,
        "lingbot_va",
        ("alohamini.policies.lingbot_va.adapter", "LingBotVAAlgorithm"),
    )
    save_checkpoint(
        path,
        model,
        stats,
        data,
        training=dict(
            mixed_precision="bfloat16" if device == "cuda" else "none", trainable_dtype=storage
        ),
    )
    loaded = NativePolicy(path, device=device, offline=True)
    assert loaded.predict(data[0]).shape == (4, 18)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)


def test_prepared_text_features_remove_runtime_encoder_dependency(
    streaming_recording, tmp_path, monkeypatch
):
    import shutil

    from alohamini.learning.policy import NativePolicy, save_checkpoint
    from alohamini.policies import registry
    from alohamini.policies.lingbot_va import core, modeling_lingbot_va

    recipe, options, data, stats, seed, _ = assembled(streaming_recording, tmp_path / "assets")
    options["text_feature_cache_dir"] = str(tmp_path / "features")
    calls = []
    original = core.load_text_encoder

    def load(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "load_text_encoder", load)
    monkeypatch.setattr(modeling_lingbot_va, "load_text_encoder", load)
    state = torch.get_rng_state()
    prepared = recipe.prepare_training_assets(options, data, stats)
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
    assert len(calls) == 1
    model = recipe.build(prepared)
    processor = recipe.processor(model, stats, "cpu")
    batch = processor(collate_samples([data[0], data[1]]))
    loss, _ = model(batch)
    loss.backward()
    assert len(calls) == 1 and model._frozen["text_encoder"] is None
    path = tmp_path / "checkpoint"
    monkeypatch.setitem(
        registry.ALGORITHMS,
        "lingbot_va",
        ("alohamini.policies.lingbot_va.adapter", "LingBotVAAlgorithm"),
    )
    save_checkpoint(path, model, stats, data, training={"mixed_precision": "none"})
    shutil.rmtree(tmp_path / "features")
    shutil.rmtree(tmp_path / "assets/text_encoder")
    loaded = NativePolicy(path, offline=True)
    assert loaded.predict(data[0]).shape == (4, 18)
    assert len(calls) == 1 and loaded.model._frozen["text_encoder"] is None
    from alohamini.learning.checkpoint import resolve_pretrained

    fine_tuning = resolve_pretrained(dict(pretrained_path=str(path)))
    assert fine_tuning["model"]["text_feature_store_path"] == str(path / "assets/text_features")
    reused = recipe.prepare_training_assets(fine_tuning["model"], data, stats)
    assert reused["text_feature_store_path"] == str(path / "assets/text_features")
    assert len(calls) == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_trainer_prepares_features_and_resumes(
    streaming_recording, tmp_path, monkeypatch, device, fresh_execution
):
    from safetensors.torch import load_file, save_file, save_model

    from alohamini.learning.train import train
    from alohamini.learning.train_config import parse_training_args
    from alohamini.policies import registry

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    recipe, options, data, stats, model, _ = assembled(streaming_recording, tmp_path / "assets")
    base = tmp_path / "assets/transformer"
    base.mkdir()
    model.transformer.save_config(base)
    file = base / "diffusion_pytorch_model.safetensors"
    save_model(model.transformer, file)
    weights = load_file(file)
    # The public base retains this unused predecessor of patch_embedding_mlp.
    weights.update(
        {
            "patch_embedding.weight": torch.zeros(16, 4, 1, 1, 1),
            "patch_embedding.bias": torch.zeros(16),
        }
    )
    save_file(weights, file)
    del model
    options.pop("input_features")
    options.pop("output_features")
    if device == "cuda":
        options["dtype"] = "bfloat16"
    options["text_feature_cache_dir"] = str(tmp_path / "feature-cache")
    monkeypatch.setitem(
        registry.ALGORITHMS,
        "lingbot_va",
        ("alohamini.policies.lingbot_va.adapter", "LingBotVAAlgorithm"),
    )
    output = tmp_path / "run"
    result = train(
        dict(
            policy="lingbot_va",
            pretrained_path=str(tmp_path / "assets"),
            dataset=str(streaming_recording),
            output_dir=str(output),
            device=device,
            cpu_threads=2,
            num_workers=0,
            train_episodes=[0],
            val_episodes=[1],
            state="joint_position,base_velocity,lift_height",
            image_size=[32, 32],
            steps=1,
            batch_size=1,
            gradient_accumulation_steps=2,
            **({"mixed_precision": "none"} if device == "cpu" else {}),
            eval_steps=1,
            eval_batch_size=2,
            save_freq=1,
            log_freq=0,
            scheduler={"type": "none"},
            model=options,
        )
    )
    assert (result / "assets/text_features/features.json").is_file()
    import json

    manifest = json.loads((result / "policy.json").read_text())
    assert manifest["training"]["trainable_dtype"] == "float32"
    assert manifest["training"]["mixed_precision"] == ("bfloat16" if device == "cuda" else "none")
    assert (
        load_file(result / "model.safetensors")["transformer.action_embedder.weight"].dtype
        == torch.float32
    )
    saved_state = torch.load(
        output / "checkpoints/000001/training_state/state.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert all(
        state["exp_avg"].dtype == torch.float32
        for state in saved_state["optimizer"]["state"].values()
    )
    import shutil

    shutil.rmtree(tmp_path / "feature-cache")
    shutil.rmtree(tmp_path / "assets/text_encoder")
    resumed, _ = parse_training_args(
        ["--config_path", str(result / "train_config.json"), "--resume=true", "--steps=2"]
    )
    assert (train(resumed) / "assets/text_features/features.json").is_file()
    assert (output / "checkpoints/000002/training_state/state.pt").is_file()


def test_streaming_observations_and_independent_offline_chunks(streaming_recording, tmp_path):
    recipe, options, data, stats, model, processor = assembled(
        streaming_recording, tmp_path / "assets"
    )
    raw = collate_samples([data[0]])
    current = {**raw, "observation.images.forward": raw["observation.images.forward"][:, 0]}
    for tick in range(9):
        action = model.select_action(processor(current))
        assert action.shape == (1, 18) and torch.isfinite(action).all()
    assert model._frame_st_id == 3
    assert model._executed_actions.shape[2] == 1
    model.reset()
    assert not model._started and model._frame_st_id == 0 and not model._action_queue
    assert model.transformer.blocks[0].attn1.attn_caches["pos"] is None
    initial = processor(raw)
    changed = deepcopy(raw)
    changed["observation.images.forward"][:, 1:] = 1 - changed["observation.images.forward"][:, 1:]
    noise = torch.randn(1, 30, 2, 4, 1)
    video = torch.randn(1, 4, 2, 2, 2)
    first = model.predict_action_chunk(initial, noise=noise, video_noise=video)
    second = model.predict_action_chunk(processor(changed), noise=noise, video_noise=video)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (1, 4, 18)


def test_checkpoint_backward_keeps_each_microbatch_visibility(streaming_recording, tmp_path):
    recipe, options, data, stats, model, processor = assembled(
        streaming_recording, tmp_path / "assets"
    )
    model._ensure_frozen_modules()
    reference = recipe.build(options)
    reference.load_state_dict(model.state_dict())
    reference._ensure_frozen_modules()
    first = processor(collate_samples([data[0]]))
    second = processor(collate_samples([data[0], data[1]]))
    torch.manual_seed(11)
    a, _ = model(first)
    torch.manual_seed(22)
    b, _ = model(second)
    (a + b).backward()
    torch.manual_seed(11)
    x, _ = reference(first)
    x.backward()
    torch.manual_seed(22)
    y, _ = reference(second)
    y.backward()
    for name, value in model.named_parameters():
        if value.grad is not None:
            torch.testing.assert_close(
                value.grad, dict(reference.named_parameters())[name].grad, rtol=1e-4, atol=1e-5
            )


def test_cuda_flex_matches_sdpa_and_runs_real_policy(streaming_recording, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from alohamini.policies.lingbot_va.core import masked_attention, training_masks

    args = ((2, 4, 2, 2, 2), (2, 30, 2, 4, 1), 64, 1, 30, (1, 1, 1), "cuda", 8)
    dense, _ = training_masks(*args, "torch")
    sparse, _ = training_masks(*args, "flex")
    values = [
        torch.randn(1, 128, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    copies = [v.detach().clone().requires_grad_() for v in values]
    first = masked_attention(*values, dense)
    second = masked_attention(*copies, sparse)
    torch.testing.assert_close(first, second, rtol=0.02, atol=0.02)
    first.float().square().mean().backward()
    second.float().square().mean().backward()
    for a, b in zip(values, copies, strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=0.08, atol=0.002)
    recipe, options, data, stats, _, _ = assembled(streaming_recording, tmp_path / "assets")
    options.update(attn_mode="flex", dtype="bfloat16", attention_head_dim=32, ffn_dim=96)
    model = recipe.build(options).to("cuda")
    processor = recipe.processor(model, stats, "cuda")
    batch = processor(collate_samples([data[0], data[1]]))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.transformer.blocks[0].attn1.to_q.weight.grad is not None

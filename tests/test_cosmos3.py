"""Real tiny Nano/Edge MoT, Wan VAE, native normalization and checkpoint paths."""

import json
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
from test_lingbot_va import fresh_execution as fresh_execution
from test_lingbot_va import streaming_recording as streaming_recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.loading import collate_samples
from alohamini.policies.cosmos3.adapter import Cosmos3EdgeAlgorithm, Cosmos3NanoAlgorithm


def tiny_options(path, variant="nano"):
    from test_longwam import tiny_options as backbone_options
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    path.mkdir(parents=True, exist_ok=True)
    special = ["[PAD]", "[UNK]", "[EOS]", "<|vision_start|>"]
    backend = Tokenizer(
        models.WordLevel(
            {t: i for i, t in enumerate(special + ["test", "user", "assistant"])}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token=special[0],
        unk_token=special[1],
        eos_token=special[2],
        additional_special_tokens=[special[3]],
        chat_template=(
            "{% for m in messages %}{{m['role']}} {% for c in m['content'] %}"
            "{{c['text']}} {% endfor %}{% endfor %}"
            "{% if add_generation_prompt %}assistant {% endif %}"
        ),
    )
    tokenizer.save_pretrained(path / "tokenizer")
    vae = backbone_options(path / "umt5")["vae_config"]
    vae["decoder_base_dim"] = 4
    text = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        pad_token_id=0,
        max_position_embeddings=20000,
        rope_parameters=dict(rope_type="default", rope_theta=10000.0, mrope_section=[1, 1, 2]),
    )
    if variant == "nano":
        backbone = dict(
            model_type="qwen3_vl",
            text_config=text,
            vision_config=dict(
                depth=1,
                hidden_size=32,
                intermediate_size=64,
                num_heads=4,
                patch_size=4,
                spatial_merge_size=2,
                temporal_patch_size=2,
                out_hidden_size=32,
                deepstack_visual_indexes=[],
            ),
            image_token_id=7,
            video_token_id=8,
            vision_start_token_id=3,
        )
    else:
        backbone = dict(
            model_type="nemotron_3_dense_vl",
            text_config=text,
            vision_config=dict(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                patch_size=4,
                spatial_merge_size=2,
            ),
            projector_config=dict(
                input_hidden_size=32,
                merger_intermediate_size=64,
                out_hidden_size=32,
                spatial_merge_size=2,
            ),
            image_token_id=7,
            video_token_id=8,
            vision_start_token_id=3,
        )
    return dict(
        variant=variant,
        mot_backbone=backbone,
        vae=vae,
        tokenizer_path=str(path / "tokenizer"),
        dtype="float32",
        chunk_size=4,
        n_action_steps=2,
        view_size=(32, 32),
        canvas_hw=(32, 32),
        max_action_dim=32,
        num_embodiment_domains=2,
        domain_id=0,
        latent_patch_size=1,
        timestep_embedding_dim=16,
        num_inference_steps=2,
        cfg_dropout_rate=0.0,
        guidance_scale=1.0,
    )


def assembled(root, path, variant="nano"):
    recipe = Cosmos3NanoAlgorithm() if variant == "nano" else Cosmos3EdgeAlgorithm()
    options = tiny_options(path, variant)
    data = AlohaMiniDataset(
        root,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **recipe.sample_spec(options, cameras=("forward",)),
    )
    recipe.prepare_options(options, data)
    options.update(input_features=data.input_features, output_features=data.output_features)
    stats = recipe.statistics(data)
    model = recipe.build(options)
    return recipe, options, data, stats, model


@pytest.mark.parametrize("variant", ["nano", "edge"])
@pytest.mark.parametrize("device,storage", [("cpu", "preserve"), ("cuda", "float32")])
def test_real_mot_forward_backward_checkpoint(
    streaming_recording, tmp_path, monkeypatch, variant, device, storage
):
    from alohamini.learning.policy import NativePolicy, save_checkpoint
    from alohamini.learning.precision import apply_trainable_dtype
    from alohamini.policies import registry

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    recipe, options, data, stats, model = assembled(
        streaming_recording, tmp_path / "assets", variant
    )
    if device == "cuda":
        options["dtype"] = "bfloat16"
        model = recipe.build(options)
    model = apply_trainable_dtype(model, storage).to(device)
    processor = recipe.processor(model, stats, device)
    batch = processor(collate_samples([data[0], data[1]]))
    model.train()
    assert not model.model.network.vae.training
    with torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else nullcontext():
        loss, terms = model(batch)
    loss.backward()
    assert torch.isfinite(loss) and set(terms) == {"video_loss", "action_loss"}
    assert model.model.network.action2llm.fc.weight.grad is not None
    layers = model.model.network.mot_backbone.model.language_model.layers
    assert len(layers) == 2 and layers[0].self_attn.q_proj_gen.weight.grad is not None
    assert layers[0].self_attn.q_proj.weight.grad is None
    assert all(p.grad is None and not p.requires_grad for p in model.model.network.vae.parameters())
    path = tmp_path / "checkpoint"
    monkeypatch.setitem(
        registry.ALGORITHMS,
        model.name,
        ("alohamini.policies.cosmos3.adapter", type(recipe).__name__),
    )
    save_checkpoint(
        path,
        model,
        stats,
        data,
        training=dict(
            trainable_dtype=storage,
            mixed_precision="bfloat16" if device == "cuda" else "none",
        ),
    )
    loaded = NativePolicy(path, device=device, offline=True)
    prediction = loaded.predict(data[0])
    assert prediction.shape == (4, 18) and torch.isfinite(prediction).all()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["nano", "edge"])
def test_inference_reads_current_views_and_state_uses_action_coordinates(
    streaming_recording, tmp_path, variant
):
    from alohamini.policies.cosmos3.core import FlowState

    recipe, options, data, stats, model = assembled(
        streaming_recording, tmp_path / "assets", variant
    )
    options.update(
        use_state=True,
        state_action_names=options["action_feature_names"],
        normalization_mapping=dict(VISUAL="IDENTITY", STATE="QUANTILES", ACTION="QUANTILES"),
    )
    model = recipe.build(options).eval()
    processor = recipe.processor(model, stats, "cpu")
    raw = collate_samples([data[0]])
    prepared = processor(raw)
    expected = processor.numeric({"action": raw["observation.state"]})["action"]
    torch.testing.assert_close(expected, prepared["observation.state"], rtol=0, atol=0)
    noise = FlowState(torch.randn(1, 4, 2, 2, 2), torch.randn(1, 5, 32))
    first = model.predict_action_chunk(prepared, noise=noise)
    changed = deepcopy(raw)
    changed["observation.images.forward"][:, 1:] = 1 - changed["observation.images.forward"][:, 1:]
    changed["action"] += 100
    second = model.predict_action_chunk(processor(changed), noise=noise)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    bad = deepcopy(options)
    bad["state_action_names"] = list(reversed(bad["state_action_names"]))
    with pytest.raises(ValueError, match="state coordinates"):
        recipe.build(bad)


@pytest.mark.parametrize("variant", ["nano", "edge"])
def test_native_trainer_and_resume(
    streaming_recording, tmp_path, monkeypatch, fresh_execution, variant
):
    from safetensors.torch import save_model

    from alohamini.learning.train import train
    from alohamini.learning.train_config import parse_training_args
    from alohamini.policies import registry

    base = tmp_path / "assets"
    recipe, options, data, stats, model = assembled(streaming_recording, base, variant)
    save_model(model, base / "model.safetensors")
    raw_config = asdict(model.config)
    raw_config.update(optimizer_lr=100.0, domain_id=8, use_state=True)
    (base / "config.json").write_text(json.dumps(raw_config))
    del model
    name = "cosmos3_" + variant
    monkeypatch.setitem(
        registry.ALGORITHMS, name, ("alohamini.policies.cosmos3.adapter", type(recipe).__name__)
    )
    options.pop("input_features")
    options.pop("output_features")
    output = tmp_path / "run"
    result = train(
        dict(
            policy=name,
            pretrained_path=str(base),
            dataset=str(streaming_recording),
            output_dir=str(output),
            device="cpu",
            cpu_threads=2,
            num_workers=0,
            train_episodes=[0],
            val_episodes=[1],
            image_size=[32, 32],
            steps=1,
            batch_size=1,
            gradient_accumulation_steps=2,
            mixed_precision="none",
            eval_steps=1,
            eval_batch_size=2,
            save_freq=1,
            log_freq=0,
            scheduler=dict(type="none"),
            model=options,
        )
    )
    manifest = json.loads((result / "policy.json").read_text())
    assert manifest["training"]["trainable_dtype"] == "float32"
    assert manifest["config"]["domain_id"] == 0 and not manifest["config"]["use_state"]
    saved = json.loads((result / "train_config.json").read_text())
    assert saved["optimizer"]["lr"] == 5e-5
    resumed, _ = parse_training_args(
        [
            "--config_path",
            str(result / "train_config.json"),
            "--resume=true",
            "--steps=2",
        ]
    )
    train(resumed)
    assert (output / "checkpoints/000002/training_state/state.pt").is_file()


@pytest.mark.parametrize("variant", ["nano", "edge"])
def test_compatible_robot_heads_are_still_reset_when_requested(
    streaming_recording, tmp_path, variant
):
    from safetensors.torch import save_model

    recipe, options, data, stats, model = assembled(
        streaming_recording, tmp_path / "assets", variant
    )
    network = model.model.network
    with torch.no_grad():
        for name in network._action_modules:
            for value in getattr(network, name).parameters():
                value.fill_(42)
    base = tmp_path / "base"
    base.mkdir()
    save_model(model, base / "model.safetensors")
    original = network.vae2llm.weight.detach().clone()
    recipe.initialize(model, {"pretrained_path": str(base)})
    torch.testing.assert_close(network.vae2llm.weight, original, rtol=0, atol=0)
    assert all(
        not torch.all(value == 42)
        for name in network._action_modules
        for value in getattr(network, name).parameters()
    )
    model.config.reinitialize_robot_heads = False
    recipe.initialize(model, {"pretrained_path": str(base)})
    assert all(
        torch.all(value == 42)
        for name in network._action_modules
        for value in getattr(network, name).parameters()
    )

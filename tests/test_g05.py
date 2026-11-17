"""Actual tiny Qwen3.5 linear/full attention and G0.5 expert through native interfaces."""

from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
from test_lingbot_va import fresh_execution as fresh_execution
from test_lingbot_va import streaming_recording as streaming_recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.loading import collate_samples
from alohamini.policies.g05.adapter import G05Algorithm


def tiny_options(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    special = [
        "[PAD]",
        "[UNK]",
        "<|endoftext|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
    ]
    backend = Tokenizer(
        models.WordLevel(
            {
                v: i
                for i, v in enumerate(
                    special + ["test", "pick", "red", "block", "Subtask:", "approach", "grasp"]
                )
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token=special[0],
        unk_token=special[1],
        eos_token=special[2],
        additional_special_tokens=special[3:],
    )
    tokenizer.save_pretrained(path / "hf_processor")
    text = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_parameters=dict(
            rope_type="default",
            rope_theta=10000.0,
            mrope_section=[2, 2, 4],
            mrope_interleaved=True,
            partial_rotary_factor=1.0,
        ),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
        pad_token_id=0,
    )
    core = dict(
        pad_token_id=0,
        eos_token_id=2,
        image_token_index=5,
        proprio_dim=20,
        attn_implementation="eager",
        position_ids_type="pi0fast",
        vlm=text,
        action_expert={
            **text,
            "hidden_size": 16,
            "input_dim": 20,
            "output_dim": 20,
            "layer_types": ["full_attention"] * 2,
        },
        vision=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            patch_size=4,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=64,
            temporal_freq=0,
        ),
        fm=dict(
            time_convention="pi_convention",
            flow_sig_min=0.001,
            padding_action_weight=0.0,
            zero_pad_action_target=False,
            final_action_clip_value=None,
            action_dim=20,
        ),
        AT_CONFIG=dict(parts_meta={"control": 20}, model_arch=dict(codebook_size=8, n_codebooks=1)),
    )
    return dict(
        author_model_config=core,
        assets_path=str(path),
        dtype="float32",
        chunk_size=4,
        n_action_steps=2,
        num_flow_samples=2,
        num_inference_steps=2,
        proprio_dropout_p=0.0,
        camera_sizes={"observation.images.forward": [16, 16]},
    )


def assembled(root, path, *, allow_unannotated=False, **overrides):
    recipe = G05Algorithm()
    options = {**tiny_options(path), **overrides}
    data = AlohaMiniDataset(
        root,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        allow_unannotated=allow_unannotated,
        **recipe.sample_spec(options, cameras=("forward",)),
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    recipe.prepare_options(options, data)
    stats = recipe.statistics(data, options)
    model = recipe.build(options)
    return recipe, options, data, stats, model


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actual_flow_precision_checkpoint(streaming_recording, tmp_path, monkeypatch, device):
    from alohamini.learning.policy import NativePolicy, save_checkpoint
    from alohamini.learning.precision import apply_trainable_dtype
    from alohamini.policies import registry

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    recipe, options, data, stats, model = assembled(streaming_recording, tmp_path / "assets")
    if device == "cuda":
        options["dtype"] = "bfloat16"
        model = recipe.build(options)
    model = apply_trainable_dtype(model, "float32").to(device)
    processor = recipe.processor(model, stats, device)
    raw = collate_samples([data[0], data[-1]])
    batch = processor(raw)
    counts = recipe.make_loss_counter(model.config, stats)(
        collate_samples([data.action_metadata(0), data.action_metadata(-1)])
    )
    expected = (~raw["action_is_pad"]).sum().item() * 18 * 2
    assert counts == {"_flow_weight": expected, "_text_weight": 0}
    assert processor.decode_before_queue
    with torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else nullcontext():
        loss, terms = model(batch)
    loss.backward()
    assert torch.isfinite(loss) and set(terms) == {"fm_loss"}
    assert model.backend.model.proprio_embedder.mlp[0].weight.grad.abs().sum() > 0
    assert model.backend.model.vlm.layers[0].linear_attn.in_proj_qkv.weight.grad.abs().sum() > 0
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    path = tmp_path / "checkpoint"
    monkeypatch.setitem(
        registry.ALGORITHMS, "g05", ("alohamini.policies.g05.adapter", "G05Algorithm")
    )
    save_checkpoint(
        path,
        model,
        stats,
        data,
        training=dict(
            trainable_dtype="float32", mixed_precision="bfloat16" if device == "cuda" else "none"
        ),
    )
    loaded = NativePolicy(path, device=device, offline=True)
    prediction = loaded.predict(data[0])
    assert prediction.shape == (4, 18) and torch.isfinite(prediction).all()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.model.state_dict()[key], rtol=0, atol=0)


def test_joint_gradients_and_checkpoint_recomputation(streaming_recording, tmp_path):
    recipe, options, data, stats, model = assembled(streaming_recording, tmp_path / "assets")
    processor = recipe.processor(model, stats, "cpu")
    batch = processor(collate_samples([data[0], data[-1]]))
    state = deepcopy(model.state_dict())
    gradients = []
    for enabled in (False, True):
        candidate = recipe.build({**options, "gradient_checkpointing": enabled})
        candidate.load_state_dict(state)
        torch.manual_seed(42)
        loss, _ = candidate(batch)
        loss.backward()
        gradients.append(
            {
                key: value.grad.clone()
                for key, value in candidate.named_parameters()
                if value.grad is not None
            }
        )
    assert gradients[0].keys() == gradients[1].keys()
    for key in gradients[0]:
        torch.testing.assert_close(gradients[0][key], gradients[1][key], rtol=1e-5, atol=1e-6)
    insulated = recipe.build(
        {**options, "joint_training": False, "reinitialize_robot_heads": False}
    )
    loss, _ = insulated(batch)
    loss.backward()
    assert all(value.grad is None for value in insulated.backend.model.vlm.parameters())
    assert all(
        value.grad is None for value in insulated.backend.model.proprio_embedder.parameters()
    )
    assert insulated.backend.model.action_expert.input_proj.weight.grad.abs().sum() > 0


def test_language_counter_and_insulated_text_gradient(streaming_recording, tmp_path):
    from test_language_data import annotated_v3

    from alohamini.learning.validation import loss_denominators
    from alohamini.policies.g05.tokenizer_g05 import G05_LABELS

    root = annotated_v3(streaming_recording, tmp_path / "annotated")
    recipe, options, data, stats, model = assembled(
        root,
        tmp_path / "assets",
        allow_unannotated=True,
        predict_cot=True,
        text_loss_weight=1.0,
        use_language_recipe=True,
        recipe=dict(
            messages=[
                dict(role="user", content="${task}", stream="high_level"),
                dict(
                    role="assistant",
                    content="Subtask: ${subtask}",
                    stream="high_level",
                    target=True,
                ),
            ]
        ),
    )
    model = recipe.build({**options, "joint_training": False, "reinitialize_robot_heads": False})
    raw = collate_samples([data[i] for i in range(len(data))])
    counter = recipe.make_loss_counter(model.config, stats)
    count = counter(raw)
    assert loss_denominators(data, counter) == count
    batch = recipe.processor(model, stats, "cpu")(raw)
    assert count["_text_weight"] == int((batch[G05_LABELS][:, 1:] != -100).sum()) > 0
    loss, terms = model(batch)
    loss.backward()
    assert torch.isfinite(loss) and terms["ce_loss"] > 0
    assert model.backend.model.vlm.input_proj.weight.grad.abs().sum() > 0


def test_native_trainer_and_resume(streaming_recording, tmp_path, monkeypatch, fresh_execution):
    import json

    from safetensors.torch import save_model

    from alohamini.learning.train import train
    from alohamini.learning.train_config import parse_training_args
    from alohamini.policies import registry

    base = tmp_path / "assets"
    recipe, options, data, stats, model = assembled(streaming_recording, base)
    save_model(model, base / "model.safetensors")
    source = model.backend.model.vlm.input_proj.weight.detach().clone()
    heads = model.backend.model.action_expert.input_proj.weight.detach().clone()
    recipe.initialize(model, {"pretrained_path": str(base)})
    torch.testing.assert_close(source, model.backend.model.vlm.input_proj.weight, rtol=0, atol=0)
    assert not torch.equal(heads, model.backend.model.action_expert.input_proj.weight)
    monkeypatch.setitem(
        registry.ALGORITHMS, "g05", ("alohamini.policies.g05.adapter", "G05Algorithm")
    )
    options.pop("input_features")
    options.pop("output_features")
    output = tmp_path / "run"
    result = train(
        dict(
            policy="g05",
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
    assert manifest["config"]["normalization_windows"]["action"] == [0, 1, 2, 3]
    saved = json.loads((result / "train_config.json").read_text())
    assert saved["optimizer"]["lr"] == 1e-5
    resumed, _ = parse_training_args(
        ["--config_path", str(result / "train_config.json"), "--resume=true", "--steps=2"]
    )
    train(resumed)
    assert (output / "checkpoints/000002/training_state/state.pt").is_file()


def test_temporal_camera_history_and_live_spec(streaming_recording, tmp_path):
    recipe, options, _, _, _ = assembled(streaming_recording, tmp_path / "assets")
    options["n_obs_steps"] = 3
    options["author_model_config"]["vision"].update(depth=2, temporal_freq=1)
    recipe, options, data, stats, model = assembled(
        streaming_recording, tmp_path / "assets", **options
    )
    history = recipe.live_history_spec(options, cameras=("forward",))
    assert history == {"observation.state": [-2, -1, 0], "observation.images.forward": [-2, -1, 0]}
    processor = recipe.processor(model, stats, "cpu")
    batch = processor(collate_samples([data[0], data[3]]))
    loss, _ = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    result = processor.action(model.predict_action_chunk(batch))
    assert result.shape == (2, 4, 18) and torch.isfinite(result).all()


def test_actual_frozen_codec_text_count_and_flow_inference(
    streaming_recording, tmp_path, monkeypatch
):
    from safetensors.torch import save_model
    from transformers.modeling_utils import local_torch_dtype

    from alohamini.policies.g05.action_codec_g05 import G05NativeActionCodec
    from alohamini.policies.g05.tokenizer_g05 import G05_LABELS, G05Tokenizer

    path = tmp_path / "assets"
    recipe, options, _, _, _ = assembled(streaming_recording, path)
    architecture = dict(
        max_component_dim=20,
        horizon=4,
        horizon_patch_size=2,
        conv_in_action_kernel=2,
        encoder_channels=8,
        latent_dim=8,
        c_mults=[1],
        strides=[[1, 1]],
        transformer_depths=[1],
        num_heads=1,
        dim_heads=32,
        ffn_mult=2.0,
        rope_base=10000,
        use_block_dct=True,
        block_dct_block_size=2,
        n_codebooks=1,
        codebook_size=8,
        codebook_dim=4,
    )
    at_config = dict(parts_meta={"control": 20}, model_arch=architecture)
    options["author_model_config"]["AT_CONFIG"] = at_config
    tokenizer = G05Tokenizer(path / "hf_processor", options["author_model_config"])
    codec = G05NativeActionCodec(at_config, action_token_begin=tokenizer.action_token_begin)
    for name, buffer in codec.module.named_buffers():
        if name.endswith("codebook"):
            buffer.normal_()
    save_model(codec.module, path / "action_tokenizer.safetensors")
    with local_torch_dtype(torch.bfloat16):
        loaded = G05NativeActionCodec.load(
            at_config,
            action_token_begin=tokenizer.action_token_begin,
            ckpt_path=path / "action_tokenizer.safetensors",
        )
    for key, value in codec.module.state_dict().items():
        torch.testing.assert_close(value, loaded.module.state_dict()[key], rtol=0, atol=0)
    options.update(
        train_discrete_actions=True, reinitialize_robot_heads=False, text_loss_weight=1.0
    )
    recipe, options, data, stats, model = assembled(streaming_recording, path, **options)
    processor = recipe.processor(model, stats, "cpu")
    raw = collate_samples([data[0], data[1]])
    batch = processor(raw)
    count = recipe.make_loss_counter(model.config, stats)(
        collate_samples([data.action_metadata(0), data.action_metadata(1)])
    )
    assert count["_text_weight"] == int((batch[G05_LABELS][:, 1:] != -100).sum()) > 0
    loss, terms = model(batch)
    loss.backward()
    assert torch.isfinite(loss) and terms["ce_loss"] > 0
    assert not model.backend.action_tokenizer.module.training
    assert all(
        not p.requires_grad and p.grad is None
        for p in model.backend.action_tokenizer.module.parameters()
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Flow inference must not run the autoregressive action decoder")

    generate = model.backend._generate_text
    monkeypatch.setattr(model.backend, "_generate_text", forbidden)
    assert model.predict_action_chunk(batch).shape == (2, 4, 20)
    monkeypatch.setattr(model.backend, "_generate_text", generate)
    model.config.action_head = "actioncodec"
    monkeypatch.setattr(model.backend, "_infer_flow", forbidden)
    decoded = model.predict_action_chunk(batch)
    assert decoded.shape == (2, 4, 20) and torch.isfinite(decoded).all()

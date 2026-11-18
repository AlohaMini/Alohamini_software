"""Real reduced WALL experts, named slots, cache and native checkpoint lifecycle."""

from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.policies.slots import NamedSlots
from alohamini.policies.wall_oss_05.adapter import WallOSS05Algorithm


def tiny_options(path):
    from test_eo1 import tiny_options as qwen_options
    from transformers import Qwen2_5_VLProcessor

    from alohamini.policies.wall_oss_05.processor import ACTION_TOKEN, PROPRIO_TOKEN

    options = qwen_options(path)
    processor = Qwen2_5_VLProcessor.from_pretrained(path, local_files_only=True)
    processor.tokenizer.add_tokens([PROPRIO_TOKEN, ACTION_TOKEN], special_tokens=True)
    processor.save_pretrained(path)
    core = options["vlm_config"]
    core.update(
        action_hidden_size=16,
        dim_inputs=[32, 16],
        experts=[
            dict(hidden_size=32, intermediate_size=64),
            dict(hidden_size=16, intermediate_size=32),
        ],
        noise_scheduler=dict(beta_alpha=1.5, beta_beta=1.0, s=0.999),
        use_x_pred=False,
        use_x_loss=True,
        causal_action_attention_mask=True,
        action_token_id=processor.tokenizer.convert_tokens_to_ids(ACTION_TOKEN),
    )
    core["text_config"]["num_hidden_layers"] = 2
    core["text_config"]["layer_types"] = ["full_attention"] * 2
    return dict(
        assets_path=str(path),
        core_config=core,
        dtype="float32",
        chunk_size=2,
        n_action_steps=1,
        num_inference_timesteps=2,
        image_long_edge=32,
        tokenizer_max_length=512,
    )


def build_data(recording, tmp_path):
    recipe = WallOSS05Algorithm()
    options = tiny_options(tmp_path / "assets")
    data = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **recipe.sample_spec(options),
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    recipe.prepare_options(options, data)
    # Deliberately use disjoint, reversed state/action mappings, not leading slots.
    options["state_slots"] = {name: 25 - i for i, name in enumerate(options["state_names"])}
    options["action_slots"] = {name: 24 - i for i, name in enumerate(options["action_names"])}
    return recipe, options, data


def test_named_slots_roundtrip_mask_and_validation():
    slots = NamedSlots(["x", "y"], {"x": 3, "y": 0}, 5)
    values = torch.randn(2, 3, 2, requires_grad=True)
    packed = slots.pack(values)
    assert packed.shape == (2, 3, 5)
    assert slots.mask(2, 3, device="cpu").sum() == 12
    assert (packed[..., [1, 2, 4]] == 0).all()
    torch.testing.assert_close(slots.unpack(packed), values)
    slots.unpack(packed).sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
    for mapping in ({"x": 1}, {"x": 1, "y": 1}, {"x": 5, "y": 0}):
        with pytest.raises(ValueError):
            NamedSlots(["x", "y"], mapping, 5)


@pytest.mark.parametrize(
    "device,dtype,storage",
    [
        ("cpu", "float32", "preserve"),
        ("cuda", "bfloat16", "preserve"),
        ("cuda", "bfloat16", "float32"),
    ],
)
def test_native_flow_masks_precision_and_checkpoint(recording, tmp_path, device, dtype, storage):
    from alohamini.learning.precision import apply_trainable_dtype
    from alohamini.policies import registry

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    recipe, options, data = build_data(recording, tmp_path)
    options["dtype"] = dtype
    stats = recipe.statistics(data)
    model = apply_trainable_dtype(recipe.build(options), storage).to(device)
    processor = recipe.processor(model, stats, device)
    raw = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    raw["task"] = ["test", "test longer task"]
    batch = processor(raw)
    batch["action_is_pad"] = torch.tensor([[False, True], [False, False]], device=device)
    assert batch["dof_mask"].shape == (2, 2, 26)
    assert batch["dof_mask"][0, 0].nonzero().flatten().tolist() == sorted(
        options["action_slots"].values()
    )
    state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None

    def amp():
        return (
            torch.autocast(device, dtype=torch.bfloat16) if storage == "float32" else nullcontext()
        )

    with amp():
        loss, terms = model(batch)
    loss.backward()
    assert torch.isfinite(loss) and set(terms) == {"flow_loss"}
    assert len(model.model.model.layers) == 2
    assert model.model.model.layers[0].self_attn.qkv_proj_experts[0].weight.grad is not None
    assert model.model.model.layers[0].self_attn.qkv_proj_experts[1].weight.grad is not None
    assert model.model.action_preprocessor.w1.weight.grad is not None
    modified = {**batch, "action": batch["action"].clone()}
    modified["action"][0, 1] = 1e6
    torch.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    with amp():
        expected, _ = model(modified)
    # A padded final action cannot influence valid earlier causal queries.
    torch.testing.assert_close(loss, expected)
    model.eval()
    with amp():
        actions = processor.action(model.predict_action_chunk(batch))
    assert actions.shape == (2, 2, 18) and torch.isfinite(actions).all()
    torch.testing.assert_close(
        processor.action(processor.action_slots.unpack(batch["action"])), raw["action"].to(device)
    )
    checkpoint = tmp_path / "checkpoint"
    registry.ALGORITHMS.setdefault(
        "wall_oss_05", ("alohamini.policies.wall_oss_05.adapter", "WallOSS05Algorithm")
    )
    save_checkpoint(
        checkpoint,
        model,
        stats,
        data,
        training={"mixed_precision": "bfloat16" if storage == "float32" else "none", "tf32": False},
    )
    loaded = NativePolicy(checkpoint, offline=True, device=device)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
    assert loaded.predict(data[0]).shape == (2, 18)


def test_mappings_and_base_geometry(recording, tmp_path):
    recipe, options, data = build_data(recording, tmp_path)
    invalid = deepcopy(options)
    invalid.pop("state_slots")
    invalid["reinitialize_robot_heads"] = False
    with pytest.raises(ValueError, match="explicit named"):
        recipe.prepare_options(invalid, data)
    for change in (
        dict(use_x_loss=False),
        dict(use_x_pred=True),
        dict(causal_action_attention_mask=False),
    ):
        invalid = deepcopy(options)
        invalid["core_config"].update(change)
        with pytest.raises(ValueError):
            recipe.config_class(**invalid)


def test_cached_flow_matches_full_prefix(recording, tmp_path, monkeypatch):
    recipe, options, data = build_data(recording, tmp_path)
    model = recipe.build(options).eval()
    processor = recipe.processor(model, recipe.statistics(data), "cpu")
    raw = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    raw["task"] = ["test", "test longer task"]
    batch = processor(raw)
    core = model.model
    noise = torch.randn(2, 2, 26)
    original_vision = core.visual.forward
    encoded = []

    def vision(*args, **kwargs):
        encoded.append(1)
        return original_vision(*args, **kwargs)

    monkeypatch.setattr(core.visual, "forward", vision)
    with torch.no_grad():
        cached = core.generate_flow_action(
            **model.inputs(batch),
            action_horizon=2,
            action_dim=26,
            num_inference_timesteps=2,
            noise=noise,
        )["predict_action"]
        assert len(encoded) == 1
        actual = noise.clone()
        inputs = {key: value for key, value in model.inputs(batch).items() if key != "dof_mask"}
        times = core.action_preprocessor.inference_times(2, torch.device("cpu"))
        for index in range(2):
            actions = core.action_preprocessor.embed(
                times[index].expand(2), actual, batch["dof_mask"]
            )
            hidden = core._hidden_states(**inputs, action_embeddings=actions)
            mask = batch["input_ids"] == core.action_token_id
            velocity = core.action_preprocessor.action_proj_back(hidden[mask, :16].float()).view(
                2, 2, 26
            )
            actual = actual + (times[index + 1] - times[index]) * velocity
    torch.testing.assert_close(cached, actual, rtol=2e-5, atol=2e-6)


def test_shared_trainer_and_resume(recording, tmp_path, monkeypatch):
    import json

    from safetensors.torch import save_model

    from alohamini.learning.train import train
    from alohamini.learning.train_config import parse_training_args
    from alohamini.policies import registry

    monkeypatch.setitem(
        registry.ALGORITHMS,
        "wall_oss_05",
        ("alohamini.policies.wall_oss_05.adapter", "WallOSS05Algorithm"),
    )
    recipe, options, data = build_data(recording, tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    save_model(recipe.build(options), base / "model.safetensors")
    output = tmp_path / "run"
    # Let the trainer resolve real names and save the coordinate maps itself.
    options.pop("input_features")
    options.pop("output_features")
    result = train(
        dict(
            policy="wall_oss_05",
            pretrained_path=str(base),
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
            eval_steps=1,
            eval_batch_size=2,
            save_freq=1,
            log_freq=0,
            scheduler={"type": "none"},
            model=options,
        )
    )
    config = json.loads((result / "policy.json").read_text())["config"]
    assert config["state_slots"] == options["state_slots"]
    assert config["action_slots"] == options["action_slots"]
    assert (output / "checkpoints/000001/training_state/state.pt").is_file()
    resumed, _ = parse_training_args(
        ["--config_path", str(result / "train_config.json"), "--resume=true", "--steps=2"]
    )
    assert (train(resumed) / "policy.json").is_file()
    assert (output / "checkpoints/000002/training_state/state.pt").is_file()

"""Reduced real Qwen/DiT/MoE models and named data protocol regression checks."""

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
import torch
from test_kinematic_fields import descriptor, joint_state
from transformers import Qwen3VLConfig

from alohamini.policies.registry import algorithm


def tiny_assets(path):
    path.mkdir(exist_ok=True, parents=True)
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=151936,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000,
                "mrope_section": [2, 1, 1],
            },
            rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
            pad_token_id=0,
            eos_token_id=151645,
        ),
        vision_config=dict(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            deepstack_visual_indexes=[],
        ),
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
    )
    config.save_pretrained(path)
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessorFast,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    vocab = {f"t{i}": i for i in range(151669)}
    special = {
        "[PAD]": 0,
        "[UNK]": 1,
        "<|endoftext|>": 151643,
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|vision_start|>": 151652,
        "<|vision_end|>": 151653,
        "<|image_pad|>": 151655,
        "<|video_pad|>": 151656,
    }
    for token, index in special.items():
        del vocab[f"t{index}"]
        vocab[token] = index
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="<|im_end|>",
        additional_special_tokens=list(special),
        extra_special_tokens={
            "image_token": "<|image_pad|>",
            "video_token": "<|video_pad|>",
            "vision_start_token": "<|vision_start|>",
            "vision_end_token": "<|vision_end|>",
        },
    )
    template = (
        "{% for message in messages %}{{ '<|im_start|>' + message"
        "['role'] + '\\n' }}{% if message['content'] is string %}{"
        "{ message['content'] }}{% else %}{% for item in message["
        "'content'] %}{% if item['type'] == 'image' %}{{ '<|visio"
        "n_start|><|image_pad|><|vision_end|>' }}{% else %}{{ ite"
        "m['text'] }}{% endif %}{% endfor %}{% endif %}{{ '<|im_e"
        "nd|>\\n' }}{% endfor %}"
    )
    tokenizer.chat_template = template
    tokenizer.add_tokens(list(special), special_tokens=True)
    processor = Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessorFast(
            patch_size=16, temporal_patch_size=2, merge_size=2, min_pixels=1024, max_pixels=1024
        ),
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=template,
    )
    processor.save_pretrained(path)
    return config.to_dict()


def options(kind, path):
    network = tiny_assets(path)
    shared = dict(
        input_features={
            "observation.state": {"type": "STATE", "shape": [18]},
            **{
                f"observation.images.{c}": {"type": "VISUAL", "shape": [3, 32, 32]}
                for c in ("forward", "wrist_left", "wrist_right")
            },
        },
        output_features={"action": {"type": "ACTION", "shape": [18]}},
        chunk_size=2,
        n_action_steps=2,
        gradient_checkpointing=False,
        dtype="float32",
    )
    if kind == "xiaomi_robotics1":
        return dict(
            **shared,
            assets_path=str(path),
            vlm_config=network,
            kinematic_fields=descriptor(),
            hidden_size=32,
            num_layers=2,
            training_repeat=1,
            num_steps=1,
        )
    from alohamini.policies.lingbot_vla_v2.semantic import JOINTS, NAMES, make_layout

    return dict(
        **shared,
        tokenizer_path=str(path),
        layout=make_layout(NAMES, ["declared test units"] * 18),
        canonical_joints=dict(JOINTS),
        expert_hidden_size=32,
        expert_intermediate_size=64,
        action_num_attention_heads=4,
        action_num_key_value_heads=2,
        action_head_dim=8,
        token_num_experts=2,
        token_top_k=1,
        token_moe_layers=[0, 1],
        token_moe_intermediate_size=32,
        token_shared_intermediate_size=32,
        num_steps=1,
        tokenizer_max_length=8,
        image_max_pixels=1024,
        image_min_pixels=1024,
        resize_imgs_with_padding=(32, 32),
    )


def model_batch(kind):
    if kind == "lingbot_vla_v2":
        return {
            "images": torch.randn(2, 3, 4, 1536),
            "img_masks": torch.ones(2, 3, dtype=torch.bool),
            "image_grid_thw": torch.tensor([[[1, 2, 2]] * 3] * 2),
            "observation.language.tokens": torch.tensor([[12, 14, 16, 0]] * 2),
            "observation.language.attention_mask": torch.tensor(
                [[1, 1, 1, 0]] * 2, dtype=torch.bool
            ),
            "observation.state": torch.zeros(2, 55),
            "action": torch.randn(2, 2, 55),
            "joint_mask": torch.ones(2, 2, 55, dtype=torch.bool),
            "action_is_pad": torch.zeros(2, 2, dtype=torch.bool),
        }
    prompt = [
        151644,
        12,
        151652,
        151655,
        151653,
        151652,
        151655,
        151653,
        151652,
        151655,
        151653,
        151645,
    ]
    suffix = [151644, 151670, 151645, 151644, 151671, 151672, 151669, 151645]
    image = {"pixel_values": torch.randn(24, 1536), "image_grid_thw": torch.tensor([[1, 2, 2]] * 6)}
    action = torch.randn(2, 2, 60)
    mask = torch.zeros_like(action, dtype=torch.bool)
    mask[..., :7] = True
    mask[..., 8:15] = True
    mask[..., 17:20] = True
    return {
        **image,
        "input_ids": torch.tensor([prompt + suffix] * 2),
        "attention_mask": torch.ones(2, len(prompt + suffix), dtype=torch.long),
        "condition_lengths": torch.tensor([len(prompt)] * 2),
        "inference_inputs": {
            **image,
            "input_ids": torch.tensor([prompt] * 2),
            "attention_mask": torch.ones(2, len(prompt), dtype=torch.long),
        },
        "action": action,
        "action_mask": mask,
        "state": torch.zeros(2, 1, 60),
        "vlm_action_target": action.flatten(0, 1),
        "vlm_action_mask": mask.flatten(0, 1),
        "vlm_action_actual_length": torch.tensor([2, 2]),
        "action_is_pad": torch.zeros(2, 2, dtype=torch.bool),
    }


@pytest.mark.parametrize("kind", ["xiaomi_robotics1", "lingbot_vla_v2"])
def test_real_model_update_eval_and_strict_reload(tmp_path, kind):
    torch.set_num_threads(2)
    component = algorithm(kind)
    cfg = options(kind, tmp_path / "assets")
    model = component.build(cfg)
    batch = model_batch(kind)
    if kind == "xiaomi_robotics1":
        batch.pop("action_is_pad")
    optimizer = torch.optim.AdamW(model.get_optim_params(), lr=1e-4)
    loss, metrics = model(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(batch)[0])
        action = model.predict_action_chunk(batch)
    assert action.shape == (2, 2, 60 if kind == "xiaomi_robotics1" else 55)
    assert action.isfinite().all()
    restored = component.build(cfg)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.eval()
    torch.manual_seed(44)
    expected = model.predict_action_chunk(batch)
    torch.manual_seed(44)
    actual = restored.predict_action_chunk(batch)
    torch.testing.assert_close(actual, expected)


def test_xiaomi_state_action_slots_and_gripper_roundtrip():
    from alohamini.datasets.field_transforms import FieldTransforms
    from alohamini.policies.xiaomi_robotics1.semantic import XiaomiFields

    desc = descriptor()
    codec = XiaomiFields(desc)
    transform = FieldTransforms(
        dict(
            state_representation="recorded",
            action_representation="tcp_delta",
            tcp_delta_frame="tool",
            rotation_representation="rotvec",
            kinematic_fields=desc,
        )
    )
    raw = joint_state()
    targets = np.stack([raw, raw.copy()])
    targets[1, 6] += 10
    _, delta = transform.encode(raw, targets)
    packed = codec.action(raw, delta)
    np.testing.assert_allclose(packed[0, :15], 0, atol=1e-6)
    assert packed[1, 6] == pytest.approx(0.01)
    state = codec.state(raw)
    assert state[6] == 0 and state[7] == pytest.approx(0.04)
    assert state[15] == pytest.approx(0.06)
    assert np.count_nonzero(packed[:, [7, 15, 16]]) == 0
    np.testing.assert_allclose(codec.decode(raw, packed), delta, atol=1e-6)


def test_lingbot_slot_order_and_explicit_lift_extension():
    from alohamini.policies.lingbot_vla_v2.semantic import (
        NAMES,
        make_layout,
        relative_actions,
        validate_layout,
    )

    layout = make_layout(NAMES, ["declared"] * 18)
    assert layout["slots"][14:] == [36, 37, 38, None]
    state = np.arange(18, dtype=float)
    action = relative_actions(state, np.stack([state + 1]), layout)
    np.testing.assert_allclose(action[0, :6], 1)
    assert action[0, 6] == 7 and action[0, 14] == 15
    broken = deepcopy(layout)
    broken["canonical_joints"].reverse()
    with pytest.raises(ValueError):
        validate_layout(broken)
    assert make_layout(NAMES, ["declared"] * 18, lift_mode="reserved")["slots"][-1] == 51


def raw_batch(kind):
    state = torch.tensor(np.stack([joint_state()] * 2), dtype=torch.float32)
    action = state[:, None].repeat(1, 2, 1)
    action[:, 1, 6] += 1
    if kind == "xiaomi_robotics1":
        from alohamini.datasets.field_transforms import FieldTransforms

        codec = FieldTransforms(
            dict(
                state_representation="recorded",
                action_representation="tcp_delta",
                tcp_delta_frame="tool",
                kinematic_fields=descriptor(),
            )
        )
        action = torch.tensor(codec.encode(state.numpy(), action.numpy())[1], dtype=torch.float32)
    return {
        "observation.state": state,
        "action": action,
        "task": ["move cube", "move cube"],
        "action_is_pad": torch.tensor([[False, False], [False, True]]),
        **{
            f"observation.images.{c}": torch.randint(0, 255, (2, 3, 32, 32), dtype=torch.uint8)
            for c in ("forward", "wrist_left", "wrist_right")
        },
    }


def numeric_stats(kind):
    stats = {k: {"mean": [0.0] * 18, "std": [1.0] * 18} for k in ("observation.state", "action")}
    if kind == "xiaomi_robotics1":
        stats["xiaomi"] = {
            "state": {"q01": [[-1.0] * 60], "q99": [[1.0] * 60]},
            "action": {"mean": [[0.0] * 60] * 2, "std": [[1.0] * 60] * 2},
        }
    return stats


@pytest.mark.parametrize("kind", ["xiaomi_robotics1", "lingbot_vla_v2"])
def test_real_processor_and_portable_assets(tmp_path, kind):
    torch.set_num_threads(2)
    component = algorithm(kind)
    model = component.build(options(kind, tmp_path / "assets"))
    raw = raw_batch(kind)
    stats = numeric_stats(kind)
    processor = component.processor(model, stats, "cpu")
    batch = processor(raw)
    loss, _ = model(batch)
    assert torch.isfinite(loss)
    model.eval()
    torch.manual_seed(63)
    expected = model.predict_action_chunk(batch)
    altered = {**raw, "action": raw["action"] + 100}
    torch.manual_seed(63)
    actual = model.predict_action_chunk(processor(altered))
    torch.testing.assert_close(actual, expected)
    path = tmp_path / "checkpoint"
    path.mkdir()
    manifest = {"config": asdict(model.config)}
    component.save(path, model, manifest)
    restored = component.build(
        component.checkpoint_options(path, manifest["config"], "cpu", None, "checkpoint")
    )
    component.load(path, restored)
    restored.eval()
    rebound = component.processor(restored, stats, "cpu")
    torch.manual_seed(63)
    torch.testing.assert_close(restored.predict_action_chunk(rebound(raw)), expected)


def check_native_dataset_checkpoint(tmp_path, kind):
    import json

    from test_dataset import frame, jpeg, metadata

    from alohamini.datasets.record import _EpisodeWriter
    from alohamini.learning.data import AlohaMiniDataset
    from alohamini.learning.data_fields import prepare_fields
    from alohamini.learning.policy import NativePolicy, make_processor, save_checkpoint
    from alohamini.learning.statistics import training_statistics
    from alohamini.policies.lingbot_vla_v2.semantic import NAMES

    cameras = ["forward", "wrist_left", "wrist_right"]
    root = tmp_path / "three-view-recording"
    writer = _EpisodeWriter(root, fps=30, task="move cube", robot_metadata=metadata(cameras))
    for _ in range(2):
        writer.begin_episode()
        for i in range(3):
            value = frame(writer)
            value["observation.state"] = joint_state().astype(np.float32)
            value["action"] = value["observation.state"].copy()
            value["action"][6] += i
            writer.add_frame(
                value, {c: jpeg() for c in cameras}, {"safety": {"feedback_valid": True}}
            )
        writer.save_episode()
    writer.close()
    from alohamini.datasets.lerobotv3 import export_lerobot

    exported = tmp_path / "exported"
    export_lerobot(root, exported)
    (exported / "meta/alohamini.json").unlink()
    root = exported
    desc = descriptor()
    rename = dict(zip(desc["source_names"], NAMES, strict=True))
    desc["source_names"] = list(NAMES)
    desc["passthrough"] = [rename[n] for n in desc["passthrough"]]
    for arm in desc["arms"]:
        for coordinate in arm["coordinates"]:
            coordinate["field"] = rename[coordinate["field"]]
        arm["gripper"]["field"] = rename[arm["gripper"]["field"]]
    (root / "meta/kinematics.json").write_text(json.dumps(desc))
    component = algorithm(kind)
    cfg = options(kind, tmp_path / "assets")
    cfg.pop("kinematic_fields", None)
    cfg.pop("layout", None)
    settings = component.apply_preset({})
    source = AlohaMiniDataset(
        root,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        cameras=cameras,
        image_size=(32, 32),
        allow_unannotated=True,
        **component.sample_spec(cfg, cameras=cameras),
    )
    samples = prepare_fields(source, settings, component, options=cfg)
    cfg.update(input_features=samples.input_features, output_features=samples.output_features)
    component.prepare_options(cfg, samples)
    stats = training_statistics(component, samples, cfg, kind)["stats"]
    model = component.build(cfg)
    processor = make_processor(model, stats, "cpu", data_fields=settings.get("data"))
    batch = next(iter(torch.utils.data.DataLoader(samples, batch_size=2)))
    prepared = processor(batch)
    loss, _ = model(prepared)
    loss.backward()
    assert torch.isfinite(loss)
    target = tmp_path / "checkpoint"
    save_checkpoint(target, model, stats, samples, training={})
    loaded = NativePolicy(target, device="cpu", offline=True)
    predicted = loaded.predict(samples[0])
    assert predicted.shape == (2, 18) and torch.isfinite(predicted).all()
    from alohamini.learning.evaluate import evaluate_checkpoint

    report = evaluate_checkpoint(target, root, episodes=[0], batch_size=2)
    assert report["valid_action_steps"] > 0 and len(report["mae_by_action"]) == 18
    # Feed encoded labels through the real platform decoding/IK path.
    loaded.model.predict_action_chunk = lambda b: prepared["action"][:1].detach()
    decoded = loaded.predict(samples[0])
    torch.testing.assert_close(decoded, source[0]["action"], rtol=1e-4, atol=2e-3)
    assert (
        loaded.config.kinematic_fields == cfg["kinematic_fields"]
        if kind == "xiaomi_robotics1"
        else loaded.config.layout["slots"][-1] is None
    )

    from alohamini.learning.checkpoint import read_checkpoint
    from alohamini.learning.train import train

    run = train(
        dict(
            policy=kind,
            pretrained_path=str(target),
            dataset=str(root),
            allow_unannotated=True,
            train_episodes=[0],
            val_episodes=[1],
            device="cpu",
            mixed_precision="none",
            steps=1,
            scheduler={"type": "none"},
            batch_size=2,
            eval_steps=1,
            eval_batch_size=2,
            save_freq=1,
            num_workers=0,
            cpu_threads=2,
            image_size=[32, 32],
            output_dir=str(tmp_path / "run"),
            model={"gradient_checkpointing": False},
        )
    )
    model_path, _ = read_checkpoint(run)
    saved = json.loads((model_path / "train_config.json").read_text())
    saved.update(resume=True, _checkpoint=str(model_path.parent), steps=2)
    _, resumed = read_checkpoint(train(saved))
    assert resumed["training"]["steps"] == 2


def test_optional_empty_loss_keeps_valid_objective():
    from torch.utils.data import DataLoader

    from alohamini.learning.execution import Execution
    from alohamini.learning.validation import evaluate_loss, loss_denominators

    class Data:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {"action": torch.tensor(float(index + 1))}

        action_metadata = __getitem__

    def counts(batch):
        return {"_primary": len(batch["action"]), "_optional": 0}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, b):
            return self.weight * b["action"].mean() * b["_primary"] + self.weight * 99 * b[
                "_optional"
            ], {}

    data = Data()
    totals = loss_denominators(data, counts)
    assert totals == {"_primary": 3, "_optional": 0}
    model = Model()
    loader = DataLoader(data, batch_size=2)
    assert evaluate_loss(model, dict, loader, counts, totals, device="cpu") == pytest.approx(2)
    ex = Execution("cpu", "none", "ddp")
    model, opt = ex.prepare(model, torch.optim.SGD(model.parameters(), lr=0.1))
    result = ex.update(model, list(loader), opt, 0, dict, reduction=counts)
    assert result["grad_norm"] == pytest.approx(2)
    assert model.weight.item() == pytest.approx(0.8)
    with pytest.raises(ValueError, match="positive"):
        loss_denominators(data, lambda b: {"_empty": 0})
    with pytest.raises(ValueError, match="nonnegative"):
        loss_denominators(data, lambda b: {"_bad": -1})


@pytest.mark.parametrize("kind", ["xiaomi_robotics1", "lingbot_vla_v2"])
def test_missing_runtime_assets_fail_without_network(tmp_path, monkeypatch, kind):
    import httpx
    import requests

    def forbidden(*args, **kwargs):
        pytest.fail("Local policy attempted a network request")

    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(requests.Session, "send", forbidden)
    component = algorithm(kind)
    cfg = options(kind, tmp_path / "assets")
    model = component.build(cfg)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        (tmp_path / "assets" / name).unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError):
        component.processor(model, numeric_stats(kind), "cpu")


def test_lingbot_inference_never_requests_teacher_frames():
    adapter = algorithm("lingbot_vla_v2")
    cfg = {"chunk_size": 50, "align_params": {"mode": "query"}}
    cameras = ["forward", "wrist_left", "wrist_right"]
    assert adapter.sample_spec(cfg, cameras=cameras)["delta_indices"][
        "observation.images.forward"
    ] == [0, 49]
    assert list(adapter.inference_spec(cfg, cameras=cameras)["delta_indices"]) == ["action"]
    assert adapter.evaluation_sample_spec(cfg, cameras=cameras) == adapter.inference_spec(
        cfg, cameras=cameras
    )


def test_lingbot_depth_teacher_loads_fused_upstream_attention(tmp_path, monkeypatch):
    from alohamini.policies.lingbot_vla_v2._vendor.model_core import depth_teachers as depth

    monkeypatch.setitem(depth.DINOV2_ARCHS, "tiny", (32, 2, 4))
    config = {
        "encoder": {
            "backbone": "tiny",
            "intermediate_layers": [0, 1],
            "dim_out": 8,
            "depth_emb_mode": "conv_1c",
            "img_depth_fuse_mode": "cat_token",
        }
    }
    expected = depth.MoRGBDEncoder(**config).eval()
    state = expected.state_dict()
    released = {}
    for key, value in state.items():
        if ".attention." not in key:
            key = key.replace(".layer_scale1.lambda1", ".ls1.gamma").replace(
                ".layer_scale2.lambda1", ".ls2.gamma"
            )
            released[key] = value
    for i in range(2):
        prefix = f"encoder.backbone.blocks.{i}."
        names = (
            ("attention.q_proj", "attention.k_proj", "attention.v_proj")
            if prefix + "attention.q_proj.weight" in state
            else (
                "attention.attention.query",
                "attention.attention.key",
                "attention.attention.value",
            )
        )
        output = (
            "attention.o_proj"
            if prefix + "attention.o_proj.weight" in state
            else "attention.output.dense"
        )
        for suffix in ("weight", "bias"):
            released[prefix + "attn.qkv." + suffix] = torch.cat(
                [state[prefix + n + "." + suffix] for n in names]
            )
            released[prefix + "attn.proj." + suffix] = state[prefix + output + "." + suffix]
    path = tmp_path / "teacher.pt"
    torch.save({"model_config": config, "model": released}, path)
    loaded = depth.load_depth_teacher(depth.MoRGBDEncoder, str(path)).eval()
    for key, value in state.items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    image = torch.rand(1, 3, 28, 28)
    depth_map = torch.ones(1, 1, 28, 28)
    with torch.no_grad():
        actual = loaded.encoder(image, 2, 2, depth=depth_map)[0]
        reference = expected.encoder(image, 2, 2, depth=depth_map)[0]
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert actual.shape == (1, 8, 2, 2) and actual.isfinite().all()

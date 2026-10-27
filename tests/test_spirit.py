import json

import numpy as np
import pytest
import torch
from test_kinematic_fields import descriptor, joint_state
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.policies.registry import algorithm
from alohamini.policies.spirit.configuration_spirit import SpiritConfig
from alohamini.policies.spirit.processor_spirit import SpiritFields


def tiny_spirit_options(path, *, tied=False):
    """Real local tokenizer, Qwen vision/text layers and DiT; no mock network or Hub."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLConfig,
        Qwen3VLProcessor,
    )
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    vocab = {
        token: i
        for i, token in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[EOS]",
                "<|vision_start|>",
                "<|image_pad|>",
                "<|vision_end|>",
                "<|video_pad|>",
                "task",
                "pick",
                "user",
                "assistant",
                "system",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
        model_max_length=512,
        additional_special_tokens=list(vocab)[3:7],
        chat_template=(
            "{% for message in messages %}"
            "{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}"
        ),
    )
    image = Qwen2VLImageProcessor(
        min_pixels=16, max_pixels=16, patch_size=2, temporal_patch_size=2, merge_size=2
    )
    processor = Qwen3VLProcessor(
        image_processor=image, tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor()
    )
    processor.save_pretrained(path)
    config = Qwen3VLConfig(
        text_config=dict(
            tie_word_embeddings=tied,
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=1024,
            pad_token_id=0,
            rope_parameters=dict(
                rope_type="default",
                rope_theta=10000,
                mrope_section=[1, 1, 2],
                mrope_interleaved=True,
            ),
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            deepstack_visual_indexes=[],
        ),
        image_token_id=4,
        video_token_id=6,
        vision_start_token_id=3,
        vision_end_token_id=5,
        tie_word_embeddings=tied,
    )
    config.save_pretrained(path)
    return dict(
        backbone_path=str(path),
        action_representation="recorded",
        gradient_checkpointing=False,
        n_action_steps=2,
        network=dict(
            dit_hidden_size=32,
            dit_num_heads=4,
            dit_num_layers=2,
            dit_cross_attention_dim=32,
            chunk_size=3,
            num_steps=2,
        ),
    )


@pytest.mark.parametrize("tied", [False, True])
def test_real_spirit_forward_backward_checkpoint_and_inference(recording, tmp_path, tied):
    torch.set_num_threads(2)
    recipe = algorithm("spirit")
    options = tiny_spirit_options(tmp_path / "qwen", tied=tied)
    data = AlohaMiniDataset(
        recording,
        episodes=[0],
        include_task=True,
        chunk_size=3,
        image_size=(32, 32),
        exclude_fixed_state=False,
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    stats = recipe.statistics(data, options)
    model = recipe.build(options, stats)
    processor = recipe.processor(model, stats, "cpu")
    assert model.network.qwen.get_output_embeddings().weight.requires_grad == tied
    assert model.network.qwen.get_input_embeddings().weight.requires_grad
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    prepared = processor(batch)
    assert set(model.network.state_dict())
    assert (~prepared["action_mask"]).any() == batch["action_is_pad"].any()
    loss, metrics = model(prepared)
    assert torch.isfinite(loss) and np.isfinite(metrics["flow_loss"])
    loss.backward()
    for module in (model.network.qwen, model.network.dit, model.network.action_out_proj):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    prediction = model.predict_action_chunk(prepared)
    assert prediction.shape == (2, 3, 18) and torch.isfinite(prediction).all()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model,
        stats,
        data,
        training={"paper_preset": recipe.apply_preset({})["paper_preset"]},
    )
    loaded = NativePolicy(checkpoint, offline=True)
    assert loaded.predict(data[0]).shape == (3, 18)
    assert loaded.manifest["assets"]["backbone_path"]["storage"] == "bundled"
    assert loaded.manifest["config"]["action_representation"] == "recorded"
    assert loaded.config.network["dit_num_layers"] == 2
    if tied:
        gathered = {key: value.detach().clone() for key, value in model.state_dict().items()}
        exported = tmp_path / "gathered-checkpoint"
        save_checkpoint(exported, model, stats, data, training={}, model_state=gathered)
        restored = NativePolicy(exported, offline=True)
        assert (
            restored.model.network.qwen.get_input_embeddings().weight
            is restored.model.network.qwen.get_output_embeddings().weight
        )
        torch.testing.assert_close(
            restored.model.network.qwen.get_input_embeddings().weight,
            model.network.qwen.get_input_embeddings().weight,
        )


def test_cartesian_pipeline_keeps_base_lift_and_uses_shared_data_codecs():
    spec = descriptor()
    state = joint_state()
    action = np.repeat(state[None], 2, axis=0)
    action[:, 0] += [0.005, 0.01]
    action[:, 7] -= [0.005, 0.01]
    action[:, -4:] += [0.1, 0.2, 0.3, 4]
    config = type("Config", (), {"action_representation": "eef_delta", "kinematic_fields": spec})()
    fields = SpiritFields(config)
    encoded_state, encoded_action = fields.encode(state, action)
    np.testing.assert_equal(encoded_action[:, -4:], action[:, -4:])
    np.testing.assert_equal(encoded_state[-4:], state[-4:])
    restored = fields.decode(state, encoded_action)
    np.testing.assert_allclose(
        fields.geometry.absolute(restored), fields.geometry.absolute(action), atol=1e-4
    )


def test_preset_separates_reference_protocol_and_recorded_requests():
    cfg = algorithm("spirit").apply_preset(
        {"model": {"action_representation": "recorded"}, "image_size": [64, 64]}
    )
    assert cfg["image_size"] == [64, 64]
    assert cfg["paper_preset"]["overrides"]["image_size"] == [64, 64]
    assert "measured" in cfg["paper_preset"]["reference_recipe"]["frames"]
    assert "requests" in cfg["paper_preset"]["adaptation"]["source"]
    assert cfg["paper_preset"]["adaptation"]["action_representation"] == "recorded"


def test_cartesian_requires_explicit_semantics_and_camera_mapping(tmp_path):
    options = dict(
        backbone_path=str(tmp_path),
        input_features={
            "observation.state": dict(type="STATE", shape=[18]),
            "observation.images.unknown": dict(type="VISUAL", shape=[3, 32, 32]),
        },
        output_features={"action": dict(type="ACTION", shape=[18])},
    )
    with pytest.raises(ValueError, match="descriptor"):
        SpiritConfig(**options)
    with pytest.raises(ValueError, match="camera_mapping"):
        SpiritConfig(**options, action_representation="recorded")
    config = SpiritConfig(
        **options, action_representation="recorded", camera_mapping={"cam_high": "unknown"}
    )
    assert config.chunk_size == 60


@pytest.mark.parametrize(
    "settings",
    [
        {"network": {"chunk_size": True}},
        {"network": {"max_state_dim": 0}},
        {"network": {"chunk_size": 512}},
        {"n_action_steps": 1.5},
        {"robot_type": 42},
    ],
)
def test_spirit_rejects_invalid_public_settings(settings):
    options = dict(
        backbone_path="local-assets",
        action_representation="recorded",
        input_features={
            "observation.state": dict(type="STATE", shape=[18]),
            "observation.images.forward": dict(type="VISUAL", shape=[3, 32, 32]),
        },
        output_features={"action": dict(type="ACTION", shape=[18])},
    )
    with pytest.raises(ValueError):
        SpiritConfig(**options, **settings)


def test_external_base_loading_preserves_target_statistics(recording, tmp_path):
    from safetensors.torch import save_file

    recipe = algorithm("spirit")
    options = tiny_spirit_options(tmp_path / "qwen")
    data = AlohaMiniDataset(
        recording,
        episodes=[0],
        include_task=True,
        chunk_size=3,
        image_size=(32, 32),
        exclude_fixed_state=False,
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    stats = recipe.statistics(data, options)
    model = recipe.build(options, stats)
    base = tmp_path / "base"
    base.mkdir()
    weights = {key: value.detach().clone() for key, value in model.network.state_dict().items()}
    for key in weights:
        if key.startswith(("normalize_", "unnormalize_")):
            weights[key] = torch.zeros(14)
    save_file(weights, base / "model.safetensors")
    recipe.initialize(model, {"pretrained_path": str(base)})
    np.testing.assert_allclose(
        model.network.normalize_targets.buffer_action["min"].numpy(), stats["action"]["min"]
    )
    bad = dict(weights)
    bad.pop("action_out_proj.weight")
    save_file(bad, base / "model.safetensors")
    with pytest.raises(RuntimeError, match="action_out_proj"):
        recipe.initialize(model, {"pretrained_path": str(base)})


def test_cartesian_native_checkpoint_keeps_geometry_and_restores_host_field_order(
    recording, tmp_path
):
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    from alohamini.datasets.lerobotv3 import export_lerobot

    root = tmp_path / "explicit-radian-v3"
    export_lerobot(recording, root)
    (root / "meta/alohamini.json").unlink()
    for path in (root / "meta/safety").glob("*.jsonl"):
        path.unlink()
    spec = json.loads(json.dumps(descriptor()))
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    for key in ("observation.state", "action"):
        info["features"][key]["names"] = spec["source_names"]
    info_path.write_text(json.dumps(info))
    for path in (root / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            state = joint_state()
            state[0] += 0.002 * row["frame_index"]
            target = state.copy()
            target[0] += 0.005
            row["observation.state"], row["action"] = state.tolist(), target.tolist()
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    data = AlohaMiniDataset(
        root,
        episodes=[0],
        allow_unannotated=True,
        state="recorded",
        include_task=True,
        chunk_size=3,
        image_size=(32, 32),
        exclude_fixed_state=False,
    )
    options = tiny_spirit_options(tmp_path / "qwen")
    options.update(
        action_representation="eef_delta",
        kinematic_fields=spec,
        input_features=data.input_features,
        output_features=data.output_features,
    )
    recipe = algorithm("spirit")
    stats = recipe.statistics(data, options)
    model = recipe.build(options, stats)
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    loss, _ = model(recipe.processor(model, stats, "cpu")(batch))
    loss.backward()
    assert torch.isfinite(loss)
    checkpoint = tmp_path / "cartesian-checkpoint"
    save_checkpoint(checkpoint, model, stats, data, training={})
    shutil.rmtree(tmp_path / "qwen")
    loaded = NativePolicy(checkpoint, offline=True)
    assert loaded.config.kinematic_fields == spec
    preprocessing = json.loads((checkpoint / "preprocessing.json").read_text())
    derived = preprocessing["statistics_features"]
    assert derived["action"]["names"][0] == "delta.left.x"
    assert derived["action"]["units"][0] == "m"
    assert derived["action"]["names"][6] == "left.gripper"
    assert derived["action"]["units"][-4:] == spec["passthrough_units"]
    assert preprocessing["field_transforms"][1]["anchor"] == "observation.state"
    resumed = recipe.options(
        {
            "resume": True,
            "_checkpoint": str(checkpoint),
            "model": {
                "backbone_path": "/removed/source/assets",
                "kinematics_path": "/removed/fields.json",
            },
        },
        "cpu",
    )
    assert resumed["kinematic_fields"] == spec
    assert str(checkpoint / "qwen_assets") == resumed["backbone_path"]
    with pytest.raises(ValueError, match="unchanged action_representation"):
        recipe.options(
            {
                "pretrained_path": str(checkpoint),
                "normalization": "checkpoint",
                "model": {"action_representation": "recorded"},
            },
            "cpu",
        )
    # An independently specified zero displacement must preserve the observation
    # joints; gripper and base/lift predictions are absolute, not increments.
    target = torch.zeros(1, 3, 18)
    target[..., 6], target[..., 13] = 0.04, 0.06
    target[..., -4:] = torch.tensor([0.1, -0.2, 10.0, 250.0])
    loaded.model.predict_action_chunk = lambda batch: target
    actual = loaded.predict(data[0])
    expected = data[0]["observation.state"].expand(3, -1)
    torch.testing.assert_close(actual, expected)

"""Real reduced GR00T/LeWM networks, goal contracts and portable checkpoints."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from test_native_learning import recording as recording

from alohamini.datasets.record import state_names
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, make_policy, save_checkpoint
from alohamini.policies.registry import algorithm

torch.set_num_threads(2)


@pytest.fixture
def groot_assets(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLConfig,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    root = tmp_path / "qwen"
    special = [
        "[PAD]",
        "[EOS]",
        "[UNK]",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "[BOS]",
    ]
    vocab = {v: i for i, v in enumerate(special + ["test", "pick", "place", "red", "blue"])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        additional_special_tokens=special[3:7],
    )
    template = (
        "{% for message in messages %}{% for item in message['content"
        "'] %}{% if item['type'] == 'image' %}<|vision_start|><|image"
        "_pad|><|vision_end|> {% else %}{{ item['text'] }}{% endif %}"
        "{% endfor %}{% endfor %}"
    )
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=4,
            temporal_patch_size=2,
            merge_size=2,
            min_pixels=64,
            max_pixels=1024,
        ),
        tokenizer=tokenizer,
        chat_template=template,
        video_processor=Qwen3VLVideoProcessor(patch_size=4, temporal_patch_size=2, merge_size=2),
    )
    processor.save_pretrained(root)
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            pad_token_id=0,
            max_position_embeddings=256,
            rope_parameters=dict(
                rope_type="default",
                rope_theta=10000.0,
                mrope_section=[1, 1, 2],
                mrope_interleaved=True,
            ),
        ),
        vision_config=dict(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=4,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=64,
            deepstack_visual_indexes=[0],
        ),
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )
    config.save_pretrained(root)
    return root


def options(kind, assets=None, *, cameras=("forward",), state=True):
    features = {
        f"observation.images.{c}": {"type": "VISUAL", "shape": [3, 32, 32]} for c in cameras
    }
    if state:
        features["observation.state"] = {"type": "STATE", "shape": [18]}
    result = dict(
        input_features=features, output_features={"action": {"type": "ACTION", "shape": [18]}}
    )
    if kind == "lewm":
        result.update(
            history_size=2,
            action_block=1,
            planning_horizon=2,
            chunk_size=2,
            n_action_steps=1,
            embed_dim=24,
            encoder_depth=1,
            encoder_heads=3,
            encoder_mlp_dim=48,
            patch_size=4,
            predictor_depth=1,
            predictor_heads=3,
            predictor_head_dim=8,
            predictor_mlp_dim=48,
            projector_hidden_dim=48,
            dropout=0.0,
            sigreg_knots=5,
            sigreg_projections=8,
            cem_samples=8,
            cem_elites=2,
            cem_iterations=2,
        )
    else:
        import json

        names = state_names("alohamini2pro")
        result.update(
            assets_path=str(assets),
            backbone_config=json.loads((assets / "config.json").read_text()),
            network=dict(
                max_state_dim=32,
                max_action_dim=32,
                max_num_embodiments=2,
                hidden_size=32,
                input_embedding_dim=32,
                backbone_embedding_dim=32,
                select_layer=2,
                action_horizon=3,
                max_seq_len=16,
                state_dropout_prob=0.0,
                strict_action_padding_mask=True,
                strict_vl_padding_mask=True,
                diffusion_model_cfg=dict(
                    num_layers=4,
                    num_attention_heads=4,
                    attention_head_dim=8,
                    norm_type="ada_norm",
                    dropout=0.0,
                    final_dropout=False,
                    output_dim=32,
                    interleave_self_attention=True,
                    positional_embeddings=None,
                ),
                vl_self_attention_cfg=dict(
                    num_layers=1,
                    num_attention_heads=4,
                    attention_head_dim=8,
                    dropout=0.0,
                    output_dim=32,
                    positional_embeddings=None,
                ),
                num_inference_timesteps=2,
            ),
            chunk_size=3,
            n_action_steps=2,
            embodiment_id=1,
            initialize_from_base=False,
            state_names=names,
            action_names=names,
            state_slot_map={n: i for i, n in enumerate(names)},
            action_slot_map={n: i for i, n in enumerate(names)},
            backbone_dtype="float32",
            shortest_image_edge=32,
            image_augmentation=False,
            crop_fraction=1.0,
        )
    return result


def stats(model):
    result = {
        "action": {"mean": [0.0] * 18, "std": [1.0] * 18, "min": [-2.0] * 18, "max": [2.0] * 18}
    }
    for key, feature in model.config.input_features.items():
        if feature.type == "VISUAL":
            result[key] = {"mean": [[[0.5]]] * 3, "std": [[[0.5]]] * 3}
        else:
            result[key] = deepcopy(result["action"])
    return result


def raw_batch(kind, cameras=("forward",), batch_size=2):
    result = {
        f"observation.images.{c}": torch.rand(batch_size, 3 if kind == "lewm" else 1, 3, 32, 32)
        for c in cameras
    }
    result.update(
        action=torch.randn(batch_size, 3, 18),
        action_is_pad=torch.zeros(batch_size, 3, dtype=torch.bool),
    )
    if kind != "lewm":
        result.update(
            {"observation.state": torch.randn(batch_size, 1, 18), "task": ["pick red"] * batch_size}
        )
    return result


@pytest.mark.parametrize("kind", ["gr00t_n17", "lewm"])
@pytest.mark.parametrize("cameras", [("forward",), ("forward", "wrist_left")])
def test_real_network_gradients_and_actions(groot_assets, kind, cameras):
    torch.manual_seed(3)
    model = make_policy(kind, options(kind, groot_assets, cameras=cameras, state=kind != "lewm"))
    component = algorithm(kind)
    processor = component.processor(model, stats(model), "cpu")
    batch = processor(raw_batch(kind, cameras))
    loss, diagnostics = model(batch)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(v) for v in diagnostics.values())
    if kind == "lewm":
        assert (
            model.world.encoder.embeddings.patch_embeddings.projection.weight.grad.abs().sum() > 0
        )
        assert model.world.predictor.pos_embedding.grad.abs().sum() > 0
        with pytest.raises(ValueError, match="goal"):
            model.eval().predict_action_chunk(batch)
        goal = processor.prepare_goal(
            {c: np.full((32, 32, 3), 128, dtype=np.uint8) for c in cameras}
        )
        model.set_goal({k: goal[k] for k in model.config.image_features})
    else:
        assert all(p.grad is None for p in model.backbone.parameters())
        assert model.action_head.action_decoder.layer2.W.grad.abs().sum() > 0
    model.eval()
    action = model.predict_action_chunk(batch)
    assert action.shape == (2, model.config.chunk_size, 18) and torch.isfinite(action).all()
    model.reset()
    if kind == "lewm":
        assert model._goal is not None


def test_groot_masked_targets_and_supervision_free_inference(groot_assets):
    model = make_policy("gr00t_n17", options("gr00t_n17", groot_assets)).eval()
    processor = algorithm("gr00t_n17").processor(model, stats(model), "cpu")
    batch = processor(raw_batch("gr00t_n17"))
    batch["action_is_pad"][:, -1] = True
    torch.manual_seed(9)
    reference = model(batch)[0]
    batch["action"][:, -1] = 1e6
    torch.manual_seed(9)
    torch.testing.assert_close(model(batch)[0], reference, rtol=0, atol=0)
    torch.manual_seed(8)
    reference = model.predict_action_chunk(batch)
    del batch["action"], batch["action_is_pad"]
    torch.manual_seed(8)
    torch.testing.assert_close(model.predict_action_chunk(batch), reference, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["gr00t_n17", "lewm"])
def test_native_checkpoint_and_goal_api(recording, groot_assets, tmp_path, kind):
    component = algorithm(kind)
    opts = options(kind, groot_assets, state=kind != "lewm")
    dataset = AlohaMiniDataset(
        recording,
        episodes=[0],
        cameras=["forward"],
        image_size=(32, 32),
        state="none" if kind == "lewm" else "joint_position,base_velocity,lift_height",
        image_resize_mode="pad" if kind == "lewm" else "stretch",
        image_pad_value=0.5 if kind == "lewm" else 0.0,
        **component.sample_spec(opts, cameras=["forward"]),
    )
    opts.update(input_features=dataset.input_features, output_features=dataset.output_features)
    if kind == "gr00t_n17":
        component.prepare_options(opts, dataset)
    model = make_policy(kind, opts).eval()
    statistics = component.statistics(dataset, opts)
    checkpoint = tmp_path / "policy"
    save_checkpoint(checkpoint, model, statistics, dataset, training={})
    if kind == "gr00t_n17":
        import shutil

        shutil.rmtree(groot_assets)
    restored = NativePolicy(checkpoint, offline=True)
    if kind == "lewm":
        from alohamini.learning.processor import image_tensor

        goal = np.zeros((24, 32, 3), dtype=np.uint8)
        restored.set_goal({"forward": goal})
        expected = restored.processor.numeric(
            {
                "observation.images.forward": image_tensor(
                    goal, (32, 32), resize_mode="pad", pad_value=0.5
                )[None]
            }
        )
        torch.testing.assert_close(
            restored.model._goal["observation.images.forward"],
            expected["observation.images.forward"],
        )
    else:
        with pytest.raises(ValueError, match="visual goals"):
            restored.set_goal({"forward": np.zeros((32, 32, 3), dtype=np.uint8)})
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[key], value, rtol=0, atol=0)
    action = restored.predict(dataset[0])
    assert action.shape == (model.config.chunk_size, 18) and torch.isfinite(action).all()


def test_lewm_episode_windows_and_counts(recording):
    component = algorithm("lewm")
    opts = options("lewm", state=False)
    component.options({"model": opts}, "cpu")
    samples = AlohaMiniDataset(
        recording,
        episodes=[0],
        image_size=(32, 32),
        state="none",
        **component.sample_spec(opts, cameras=["forward"]),
    )
    assert len(samples) == 3
    sample = samples[0]
    assert sample["observation.images.forward"].shape[0] == 3
    assert (
        component.loss_counts({k: v[None] for k, v in samples.action_metadata(0).items()})[
            "_prediction_weight"
        ]
        == 1
    )
    assert (
        component.loss_counts({k: v[None] for k, v in samples.action_metadata(1).items()})[
            "_prediction_weight"
        ]
        == 2
    )


@pytest.mark.parametrize("kind", ["gr00t_n17", "lewm"])
def test_numeric_limits_follow_the_transformed_field_units(recording, tmp_path, kind):
    from test_data_fields import source_dataset

    from alohamini.learning.data_fields import prepare_fields

    source, cfg, _ = source_dataset(recording, tmp_path)
    cfg["data"]["action_representation"] = "tcp_absolute"
    component = algorithm(kind)
    samples = prepare_fields(source, cfg, component, options={})
    fitted = component.statistics(samples, {"use_percentiles": False})
    values = torch.cat([samples[i]["action"] for i in range(len(samples))])
    assert values.shape[-1] == 7 and source.output_features["action"].shape == (3,)
    np.testing.assert_allclose(fitted["action"]["min"], values.min(0).values, atol=1e-7)
    np.testing.assert_allclose(fitted["action"]["max"], values.max(0).values, atol=1e-7)
    if kind == "gr00t_n17":
        quantiles = component.statistics(samples, {"use_percentiles": True})
        assert len(quantiles["action"]["min"]) == 7
        np.testing.assert_array_equal(quantiles["action"]["min"], quantiles["action"]["q01"])


def test_lewm_rejects_ambiguous_cross_time_relative_action_blocks():
    with pytest.raises(ValueError, match="transition action blocks"):
        algorithm("lewm").options({"data": {"action_representation": "tcp_delta"}}, "cpu")


def test_lewm_planning_does_not_read_future_supervision():
    model = make_policy("lewm", options("lewm", state=False)).eval()
    processor = algorithm("lewm").processor(model, stats(model), "cpu")
    prepared_goal = processor.prepare_goal({"forward": np.zeros((32, 32, 3), dtype=np.uint8)})
    model.set_goal({k: prepared_goal[k] for k in model.config.image_features})
    batch = processor(raw_batch("lewm"))
    torch.manual_seed(6)
    reference = model.predict_action_chunk(batch)
    batch["observation.images.forward"][:, -1] = 1e6
    batch["action"][:] = -1e6
    torch.manual_seed(6)
    torch.testing.assert_close(model.predict_action_chunk(batch), reference, rtol=0, atol=0)


@pytest.mark.parametrize("llm,vision", [(True, False), (False, True)])
def test_gr00t_independent_module_finetuning(groot_assets, llm, vision):
    cfg = options("gr00t_n17", groot_assets)
    cfg.update(tune_llm=llm, tune_visual=vision)
    model = make_policy("gr00t_n17", cfg)
    processor = algorithm("gr00t_n17").processor(model, stats(model), "cpu")
    model(processor(raw_batch("gr00t_n17")))[0].backward()
    for module, enabled in [
        (model.backbone.model.model.language_model, llm),
        (model.backbone.model.model.visual, vision),
    ]:
        nonzero = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        assert bool(nonzero) == enabled


def test_gr00t_partial_network_overrides_preserve_the_base_architecture(groot_assets, tmp_path):
    import json

    cfg = options("gr00t_n17", groot_assets)
    root = tmp_path / "base"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "Gr00tN1d7", **cfg["network"]}))
    resolved = algorithm("gr00t_n17").options(
        dict(
            pretrained_path=str(root),
            model=dict(
                assets_path=str(groot_assets),
                network=dict(
                    num_inference_timesteps=7,
                    diffusion_model_cfg={"dropout": 0.05},
                ),
            ),
        ),
        "cpu",
    )
    assert resolved["network"]["diffusion_model_cfg"]["num_layers"] == 4
    assert resolved["network"]["diffusion_model_cfg"]["dropout"] == 0.05
    assert resolved["network"]["num_inference_timesteps"] == 7
    assert resolved["network"]["select_layer"] == 2


def test_gr00t_augmentation_replays_across_each_samples_views(groot_assets):
    cfg = options("gr00t_n17", groot_assets, cameras=("forward", "wrist_left"))
    cfg.update(image_augmentation=True, crop_fraction=0.95)
    model = make_policy("gr00t_n17", cfg)
    processor = algorithm("gr00t_n17").processor(model, stats(model), "cpu")
    batch = raw_batch("gr00t_n17", cameras=("forward", "wrist_left"))
    batch["observation.images.wrist_left"] = batch["observation.images.forward"].clone()
    prepared = processor(batch)
    counts = prepared["image_grid_thw"].prod(-1).tolist()
    patches = prepared["pixel_values"].split(counts)
    torch.testing.assert_close(patches[0], patches[1], rtol=0, atol=0)
    torch.testing.assert_close(patches[2], patches[3], rtol=0, atol=0)


@pytest.mark.parametrize(
    "goal",
    [
        {},
        {"unknown": np.zeros((32, 32, 3), dtype=np.uint8)},
        {"forward": np.zeros((32, 32, 3), dtype=np.float32)},
    ],
)
def test_goal_inputs_are_explicit_and_validated(goal):
    model = make_policy("lewm", options("lewm", state=False))
    processor = algorithm("lewm").processor(model, stats(model), "cpu")
    with pytest.raises(ValueError, match="Goal|goal"):
        processor.prepare_goal(goal)


def test_default_recipes_and_explicit_overrides_are_recorded():
    groot = algorithm("gr00t_n17").apply_preset({})
    assert groot["steps"] == 10000 and groot["scheduler"]["warmup_steps"] == 500
    lewm = algorithm("lewm").apply_preset({})
    assert lewm["state"] == "none" and lewm["ema"]["enable"] is False
    cfg = algorithm("gr00t_n17").apply_preset({"steps": 2000, "batch_size": 2})
    assert cfg["steps"] == 2000 and cfg["scheduler"]["warmup_steps"] == 100
    assert cfg["paper_preset"]["overrides"]["batch_size"] == 2

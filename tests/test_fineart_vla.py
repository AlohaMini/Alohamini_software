"""Real tiny FineART math, gradient isolation and platform annotation/checkpoint paths."""

import json
from copy import deepcopy

import pytest
import torch
from test_language_data import annotated_v3
from test_native_learning import recording as recording
from test_pi0_fast import tiny_options as fast_options

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.loading import make_loader
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.learning.validation import loss_denominators
from alohamini.policies.fineart_vla.adapter import FineARTVLAAlgorithm


def tiny_options(path, monkeypatch):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    base = fast_options(path, monkeypatch)
    # Exercise the author's released nested codec layout, including native repackaging.
    codec = path / "fast"
    (codec / "bpe_tokenizer").mkdir()
    (codec / "tokenizer.json").rename(codec / "bpe_tokenizer/tokenizer.json")
    vocabulary = {f"unused{i}": i for i in range(2048)}
    for i in range(1024):
        vocabulary.pop(f"unused{1024 + i}")
        vocabulary[f"<loc{i:04d}>"] = 1024 + i
    for i, token in enumerate(
        [
            "<unk>",
            "<s>",
            "</s>",
            "<pad>",
            "User",
            "Assistant",
            "approach",
            "grasp",
            "State",
            "Action",
            ":",
            ";",
        ]
    ):
        vocabulary.pop(f"unused{i}")
        vocabulary[token] = i
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 1)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(path / "text")
    return dict(
        assets_path=str(path / "text"),
        fast_tokenizer_path=base["fast_tokenizer_path"],
        auto_fit_fast_tokenizer=False,
        network=dict(
            dtype="float32",
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_horizon=2,
            max_token_len=200,
        ),
        flow_num_repeats=2,
        n_action_steps=1,
        num_inference_steps=2,
    )


@pytest.mark.parametrize(
    "device,dtype", [("cpu", "float32"), ("cuda", "float32"), ("cuda", "bfloat16")]
)
def test_native_fineart_objectives_ki_and_checkpoint(
    recording, tmp_path, monkeypatch, device, dtype
):
    check_native_dataset_checkpoint(recording, tmp_path, monkeypatch, device, dtype)


def check_native_dataset_checkpoint(
    recording, tmp_path, monkeypatch, device="cpu", dtype="float32"
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    recipe = FineARTVLAAlgorithm()
    options = tiny_options(tmp_path / "assets", monkeypatch)
    options["network"]["dtype"] = dtype
    root = annotated_v3(recording, tmp_path / "annotated")
    data = AlohaMiniDataset(
        root,
        episodes=[0],
        allow_unannotated=True,
        state="recorded",
        image_size=(32, 32),
        **recipe.sample_spec(options),
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    model = recipe.build(options).to(device)
    stats = recipe.statistics(data)
    processor = recipe.processor(model, stats, device)
    raw = next(iter(make_loader(data, batch_size=len(data))))
    counter = recipe.make_loss_counter(model.config, stats)
    counts = counter(raw)
    assert counts["_flow_weight"] > 0 and counts["_text_weight"] > 0
    assert loss_denominators(data, counter) == counts
    batch = processor(raw)
    loss, metrics = model(batch)
    assert torch.isfinite(loss) and set(metrics) == {"flow_loss", "text_loss", "fast_loss"}
    loss.backward()
    vlm = model.network.paligemma_with_expert.paligemma
    query = vlm.language_model.layers[0].self_attn.q_proj.weight
    assert query.grad is not None and query.grad.abs().sum() > 0
    assert vlm.lm_head.weight.grad is not None
    assert model.network.action_out_proj.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    flow, _, _ = model.network.objective_outputs(batch, model.config)
    flow.mean().backward()
    assert query.grad is None or not query.grad.any()
    assert vlm.lm_head.weight.grad is None or not vlm.lm_head.weight.grad.any()
    model.eval()
    inference_raw = {k: v for k, v in raw.items() if k not in ("action", "action_is_pad")}
    inference = processor(inference_raw)
    prediction = model.predict_action_chunk(inference)
    assert prediction.shape == (len(data), 2, 32) and torch.isfinite(prediction).all()
    from alohamini.policies import registry

    monkeypatch.setitem(
        registry.ALGORITHMS,
        "fineart_vla",
        ("alohamini.policies.fineart_vla.adapter", "FineARTVLAAlgorithm"),
    )
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, stats, data, training={"tf32": False})
    loaded = NativePolicy(checkpoint, device=device, offline=True)
    assert loaded.predict(data[0]).shape == (2, 18)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], value, rtol=0, atol=0)


def test_fineart_independent_flow_blocks_never_read_fast_targets(recording, tmp_path, monkeypatch):
    recipe = FineARTVLAAlgorithm()
    options = tiny_options(tmp_path / "assets", monkeypatch)
    data = AlohaMiniDataset(
        annotated_v3(recording, tmp_path / "data"),
        episodes=[0],
        allow_unannotated=True,
        state="recorded",
        image_size=(32, 32),
        **recipe.sample_spec(options),
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    model = recipe.build(options).eval()
    processor = recipe.processor(model, recipe.statistics(data), "cpu")
    batch = processor(next(iter(make_loader(data, batch_size=2))))
    noise = torch.randn(4, 2, 32)
    time = torch.tensor([0.2, 0.4, 0.6, 0.8])
    with torch.no_grad():
        flow, _, _ = model.network.objective_outputs(batch, model.config, noise=noise, time=time)
        changed = dict(batch, fast_tokens=batch["fast_tokens"].roll(1, 1))
        altered, _, _ = model.network.objective_outputs(
            changed, model.config, noise=noise, time=time
        )
        torch.testing.assert_close(flow, altered, rtol=0, atol=0)
        single = deepcopy(model.config)
        single.flow_num_repeats = 1
        independent = [
            model.network.objective_outputs(
                batch, single, noise=noise[i : i + 2], time=time[i : i + 2]
            )[0]
            for i in (0, 2)
        ]
        torch.testing.assert_close(flow, torch.stack(independent).mean(0), rtol=2e-5, atol=2e-6)


def test_text_token_mean_matches_unequal_microbatches(recording, tmp_path, monkeypatch):
    recipe = FineARTVLAAlgorithm()
    options = tiny_options(tmp_path / "assets", monkeypatch)
    options.update(
        flow_loss_weight=0,
        fast_action_loss_weight=0,
        recipe={
            "messages": [
                {"role": "user", "content": "${task}", "stream": "high_level"},
                {
                    "role": "assistant",
                    "content": "${subtask}",
                    "stream": "high_level",
                    "target": True,
                },
            ]
        },
    )
    data = AlohaMiniDataset(
        annotated_v3(recording, tmp_path / "data"),
        episodes=[0],
        allow_unannotated=True,
        state="recorded",
        image_size=(32, 32),
        **recipe.sample_spec(options),
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    model = recipe.build(options)
    stats = recipe.statistics(data)
    processor = recipe.processor(model, stats, "cpu")
    from alohamini.learning.loading import collate_samples

    samples = [data[i] for i in range(len(data))]
    for i, sample in enumerate(samples):
        for row in sample["_annotations"]["fields"]["language_persistent"]:
            row["content"] = "approach " + "grasp " * i
    raw = collate_samples(samples)
    counts = recipe.make_loss_counter(model.config, stats)
    full_counts = counts(raw)
    whole, _ = model(processor(raw))
    whole.backward()
    weight = model.network.paligemma_with_expert.paligemma.language_model.layers[
        0
    ].self_attn.q_proj.weight
    gradient = weight.grad.clone()
    model.zero_grad(set_to_none=True)
    partial = 0
    for chunk in (samples[:1], samples[1:]):
        raw_chunk = collate_samples(chunk)
        prepared = processor(raw_chunk)
        prepared.update(
            {
                key: value / full_counts[key] if full_counts[key] else 0
                for key, value in counts(raw_chunk).items()
            }
        )
        loss, _ = model(prepared)
        loss.backward()
        partial += loss.detach()
    torch.testing.assert_close(partial, whole.detach(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(weight.grad, gradient, rtol=2e-5, atol=2e-6)
    assert model.network.action_out_proj.weight.grad is None


def test_released_base_configuration_is_inherited_with_explicit_overrides(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "action_tokenizer").mkdir()
    (base / "config.json").write_text(
        json.dumps(
            dict(
                chunk_size=50,
                tokenizer_max_length=128,
                max_action_dim=32,
                fast_skip_tokens=128,
                flow_num_repeats=1,
                max_action_tokens=384,
            )
        )
    )
    options = FineARTVLAAlgorithm.options(
        dict(
            pretrained_path=str(base),
            model=dict(
                assets_path=str(base),
                network=dict(max_token_len=256),
                flow_num_repeats=2,
            ),
        ),
        "cpu",
    )
    assert options["fast_skip_tokens"] == 128 and options["max_action_tokens"] == 384
    assert options["flow_num_repeats"] == 2 and options["network"]["max_token_len"] == 256
    assert options["network"]["action_horizon"] == 50


def test_fitted_codec_cache_reuses_verified_assets(recording, tmp_path, monkeypatch):
    from alohamini.policies.fineart_vla import assets

    recipe = FineARTVLAAlgorithm()
    options = tiny_options(tmp_path / "assets", monkeypatch)
    options.update(
        auto_fit_fast_tokenizer=True,
        fast_fit_samples=3,
        fast_validation_samples=1,
        fast_fit_vocab_size=512,
        fast_tokenizer_cache_dir=str(tmp_path / "cache"),
    )
    data = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        exclude_fixed_state=False,
        image_size=(32, 32),
        chunk_size=2,
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    stats = recipe.statistics(data)
    first = recipe.prepare_training_assets(options, data, stats)
    monkeypatch.setattr(
        assets, "fit_fast", lambda *args, **kwargs: pytest.fail("cache was refitted")
    )
    cached = recipe.prepare_training_assets(options, data, stats)
    assert cached["fast_fit_report"] == first["fast_fit_report"]
    from pathlib import Path

    file = Path(cached["fast_tokenizer_path"]) / "tokenizer.json"
    file.write_text(file.read_text() + " ")
    with pytest.raises(ValueError, match="cache changed"):
        recipe.prepare_training_assets(options, data, stats)


def test_task_only_recipe_counts_without_language_columns(recording, tmp_path, monkeypatch):
    from alohamini.datasets.lerobotv3 import export_lerobot
    from alohamini.policies.fineart_vla.configuration_fineart_vla import FineARTVLAConfig

    recipe = FineARTVLAAlgorithm()
    options = tiny_options(tmp_path / "assets", monkeypatch)
    options.update(
        text_loss_weight=0,
        fast_action_loss_weight=0,
        recipe={"messages": [{"role": "user", "content": "${task}", "stream": "low_level"}]},
    )
    root = tmp_path / "v3"
    export_lerobot(recording, root)
    (root / "meta/alohamini.json").unlink()
    for path in (root / "meta/safety").glob("*.jsonl"):
        path.unlink()
    data = AlohaMiniDataset(
        root,
        episodes=[0],
        state="recorded",
        allow_unannotated=True,
        image_size=(32, 32),
        **recipe.sample_spec(options),
    )
    assert data[0]["_annotations"]["fields"] == {}
    options.update(input_features=data.input_features, output_features=data.output_features)
    counter = recipe.make_loss_counter(FineARTVLAConfig(**options), recipe.statistics(data))
    assert loss_denominators(data, counter) == {
        "_flow_weight": 4,
        "_text_weight": 0,
        "_fast_weight": 0,
    }


def test_language_generation_runs_real_cache_with_padding(recording, tmp_path, monkeypatch):
    import math

    recipe = FineARTVLAAlgorithm()
    options = tiny_options(tmp_path / "assets", monkeypatch)
    data = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        exclude_fixed_state=False,
        image_size=(32, 32),
        chunk_size=2,
        include_task=True,
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    model = recipe.build(options).eval()
    processor = recipe.processor(model, recipe.statistics(data), "cpu")
    prepared = processor(
        {
            "observation.state": torch.zeros(2, 18),
            "observation.images.forward": torch.zeros(2, 3, 32, 32, dtype=torch.uint8),
            "task": ["approach", "approach grasp"],
        },
        text_query=True,
    )
    backbone = model.network.paligemma_with_expert
    original = backbone.paligemma.lm_head.forward
    choices = iter([[7, 6], [2, 7], [2, 2]])
    captured = []

    def scripted_head(hidden):
        captured.append(hidden.detach().clone())
        logits = original(hidden)
        logits.fill_(-1000)
        logits[torch.arange(len(hidden)), torch.tensor(next(choices))] = 0
        return logits

    monkeypatch.setattr(backbone.paligemma.lm_head, "forward", scripted_head)
    assert model.generate_text_tokens(prepared, eos_id=2) == [[7], [6, 7]]
    with torch.no_grad():
        images, masks, tokens, valid, _ = model.network._preprocess_observation(
            prepared["observation"], train=False
        )
        prefix, valid, _ = model.network.embed_prefix(images, masks, tokens, valid)
        extra = torch.tensor([[7], [6]])
        embedded = backbone.embed_language_tokens(extra) * math.sqrt(prefix.shape[-1])
        extended = torch.cat((valid, torch.ones(2, 1, dtype=torch.bool)), dim=1)
        attention = extended[:, :, None] & extended[:, None, :]
        attention[:, :-1, -1] = False
        (full, _), _ = backbone(
            inputs_embeds=[torch.cat((prefix, embedded), dim=1), None],
            attention_mask=model.network._prepare_attention_masks_4d(attention),
            position_ids=extended.long().cumsum(-1) - 1,
            use_cache=False,
        )
    torch.testing.assert_close(captured[1], full[:, -1], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("auto_fit", [False, True])
def test_shared_trainer_counts_annotation_objectives_and_saves_state(
    recording, tmp_path, monkeypatch, auto_fit
):
    from safetensors.torch import save_model

    from alohamini.learning.train import train
    from alohamini.policies import registry

    monkeypatch.setitem(
        registry.ALGORITHMS,
        "fineart_vla",
        ("alohamini.policies.fineart_vla.adapter", "FineARTVLAAlgorithm"),
    )
    options = tiny_options(tmp_path / "assets", monkeypatch)
    options["base_weights_prefix"] = ""
    options.update(
        auto_fit_fast_tokenizer=auto_fit,
        fast_tokenizer_cache_dir=str(tmp_path / "codec-cache"),
        fast_fit_samples=3,
        fast_validation_samples=1,
        fast_fit_vocab_size=512,
    )
    options["recipe"] = {
        "messages": [
            {"role": "user", "content": "${task}", "stream": "high_level"},
            {"role": "assistant", "content": "${subtask}", "stream": "high_level", "target": True},
            {"role": "user", "content": "${subtask}", "stream": "low_level"},
        ]
    }
    root = annotated_v3(recording, tmp_path / "data")
    recipe = FineARTVLAAlgorithm()
    data = AlohaMiniDataset(
        root,
        episodes=[0],
        allow_unannotated=True,
        state="recorded",
        image_size=(32, 32),
        **recipe.sample_spec(options),
    )
    seed = recipe.build(
        dict(options, input_features=data.input_features, output_features=data.output_features)
    )
    base = tmp_path / "base"
    base.mkdir()
    save_model(seed.network, base / "model.safetensors")
    del seed
    output = tmp_path / "run"
    result = train(
        dict(
            policy="fineart_vla",
            pretrained_path=str(base),
            dataset=str(root),
            output_dir=str(output),
            allow_unannotated=True,
            device="cpu",
            cpu_threads=2,
            num_workers=0,
            train_episodes=[0],
            val_episodes=[1],
            state="recorded",
            image_size=[32, 32],
            steps=1,
            batch_size=1,
            gradient_accumulation_steps=4,
            eval_steps=1,
            eval_batch_size=2,
            save_freq=1,
            log_freq=0,
            scheduler={"type": "none"},
            model=options,
        )
    )
    assert (result / "policy.json").is_file()
    if auto_fit:
        fitted = json.loads((result / "policy.json").read_text())["config"]["fast_fit_report"]
        assert fitted["fit_chunks"] == 3 and fitted["validation"]["validation_chunks"] == 1
        assert fitted["validation"]["reconstruction_rmse"] <= 0.1
    assert (output / "checkpoints/000001/training_state/state.pt").is_file()
    validation = json.loads((output / "validation-metrics.jsonl").read_text().splitlines()[0])
    assert validation["metric_totals"]["loss_text"]["count"] > 0
    assert validation["metric_totals"]["loss_flow"]["count"] > 0
    assert "loss_fast" in validation["metrics"]
    from alohamini.learning.train_config import parse_training_args

    resumed_cfg, _ = parse_training_args(
        [
            "--config_path",
            str(result / "train_config.json"),
            "--resume=true",
            "--steps=2",
        ]
    )
    resumed = train(resumed_cfg)
    assert (resumed / "policy.json").is_file()
    assert (output / "checkpoints/000002/training_state/state.pt").is_file()

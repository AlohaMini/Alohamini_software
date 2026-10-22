"""Real small transformers: objective isolation, adapters, and offline assets."""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_model

from alohamini.policies.pi05.adapter import PI05Algorithm, PI05TrainingConfig, PI05TrainModel
from alohamini.policies.pi05.finetuning import load_base_weights
from alohamini.policies.pi05.knowledge_insulation import attention_masks, token_cross_entropy


def options(**overrides):
    result = dict(
        input_features={
            "observation.state": {"type": "STATE", "shape": [18]},
            "observation.images.forward": {"type": "VISUAL", "shape": [3, 224, 224]},
        },
        output_features={"action": {"type": "ACTION", "shape": [18]}},
        network={
            "dtype": "float32",
            "paligemma_variant": "dummy",
            "action_expert_variant": "dummy",
            "action_horizon": 2,
            "pytorch_compile_mode": None,
        },
        n_action_steps=2,
        lora_rank=2,
        expert_lora_rank=2,
        fast_tokenizer_path="/unused/fast",
    )
    result.update(overrides)
    return result


@pytest.fixture
def tiny_model(monkeypatch):
    from alohamini.policies.pi05 import configuration_pi05, gemma_pytorch

    original_pg = gemma_pytorch.PaliGemmaForConditionalGeneration
    original_expert = gemma_pytorch.GemmaForCausalLM

    def paligemma(config):
        config.text_config.vocab_size = 128
        config.vision_config.hidden_size = 16
        config.vision_config.intermediate_size = 32
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 2
        config.vision_config.patch_size = 112
        model = original_pg(config)
        model.model.multi_modal_projector.linear = torch.nn.Linear(16, 32)
        return model

    def expert(config):
        config.vocab_size = 128
        return original_expert(config)

    monkeypatch.setattr(gemma_pytorch, "PaliGemmaForConditionalGeneration", paligemma)
    monkeypatch.setattr(gemma_pytorch, "GemmaForCausalLM", expert)
    monkeypatch.setattr(
        configuration_pi05,
        "get_config",
        lambda _: configuration_pi05.GemmaConfig(32, 2, 64, 4, 1, 8),
    )
    torch.set_num_threads(2)
    return lambda **kwargs: PI05TrainModel(PI05TrainingConfig(**options(**kwargs)))


def batch():
    return dict(
        observation=SimpleNamespace(
            images={
                "base_0_rgb": torch.rand(2, 3, 224, 224) * 2 - 1,
                "left_wrist_0_rgb": torch.zeros(2, 3, 224, 224),
                "right_wrist_0_rgb": torch.zeros(2, 3, 224, 224),
            },
            image_masks={
                "base_0_rgb": torch.ones(2, dtype=torch.bool),
                "left_wrist_0_rgb": torch.zeros(2, dtype=torch.bool),
                "right_wrist_0_rgb": torch.zeros(2, dtype=torch.bool),
            },
            state=torch.zeros(2, 32),
            tokenized_prompt=torch.tensor([[2, 3, 4, 0], [2, 5, 0, 0]]),
            tokenized_prompt_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
            token_ar_mask=None,
            token_loss_mask=None,
        ),
        action=torch.rand(2, 2, 32),
        fast_tokens=torch.tensor([[20, 21, 1], [22, 1, 0]]),
        fast_mask=torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
    )


def nonzero_grad(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


@pytest.mark.parametrize("checkpointing", [False, True])
def test_ki_detaches_keys_and_values_without_freezing_vlm(tiny_model, checkpointing):
    model = tiny_model(training_recipe="ki", gradient_checkpointing=checkpointing)
    inputs = batch()
    flow, fast = model.network(
        inputs["observation"],
        inputs["action"],
        knowledge_insulation=True,
        fast_tokens=inputs["fast_tokens"],
        fast_mask=inputs["fast_mask"],
        ce_chunk_size=2,
    )
    flow.mean().backward(retain_graph=True)
    combined = model.network.paligemma_with_expert
    assert nonzero_grad(combined.gemma_expert)
    assert not nonzero_grad(combined.paligemma)
    model.zero_grad(set_to_none=True)
    fast.mean().backward()
    assert nonzero_grad(combined.paligemma.language_model)
    assert nonzero_grad(combined.paligemma.vision_tower)
    assert not nonzero_grad(combined.gemma_expert)
    model.zero_grad(set_to_none=True)
    model.network(inputs["observation"], inputs["action"]).mean().backward()
    assert nonzero_grad(combined.paligemma)  # Original flow gradient path remains intact.
    assert model.network.is_gradient_checkpointing_enabled() == checkpointing
    assert (
        combined.paligemma.vision_tower.vision_model.encoder.layers[0].gradient_checkpointing
        == checkpointing
    )


def test_masks_and_action_positions_do_not_depend_on_teacher_forcing():
    context = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    targets = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    actions = torch.ones(2, 2, dtype=torch.bool)
    mask, positions = attention_masks(context, targets, actions)
    assert not mask[:, :3, 3:].any()
    assert not mask[:, 3:6, 6:].any()
    assert not mask[:, 6:, 3:6].any()
    assert not mask[0, :, 2].any()
    assert not mask[0, 3, 4:6].any()
    assert mask[0, 4, 3]
    assert positions[:, 6:].tolist() == [[2, 3], [3, 4]]


def test_fast_labels_never_change_flow_or_inference(tiny_model):
    model = tiny_model(training_recipe="ki").eval()
    inputs = batch()
    noise, time = torch.zeros_like(inputs["action"]), torch.full((2,), 0.5)
    baseline = model.network(inputs["observation"], inputs["action"], noise=noise, time=time)
    first, _ = model.network(
        inputs["observation"],
        inputs["action"],
        noise=noise,
        time=time,
        knowledge_insulation=True,
        fast_tokens=inputs["fast_tokens"],
        fast_mask=inputs["fast_mask"],
    )
    changed, _ = model.network(
        inputs["observation"],
        inputs["action"],
        noise=noise,
        time=time,
        knowledge_insulation=True,
        fast_tokens=inputs["fast_tokens"] + 7,
        fast_mask=inputs["fast_mask"],
    )
    torch.testing.assert_close(first, baseline, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(first, changed, atol=0, rtol=0)
    prediction = model.network.sample_actions(
        torch.device("cpu"), inputs["observation"], noise=noise, num_steps=2
    )
    assert prediction.shape == inputs["action"].shape and torch.isfinite(prediction).all()


@pytest.mark.parametrize(
    "mode,vision_mode,vision_updates",
    [
        ("a", "frozen", False),
        ("a", "full", True),
        ("b", "frozen", False),
        ("c", "frozen", True),
        ("d", "frozen", True),
    ],
)
def test_each_finetuning_mode_updates_exactly_selected_parameters(
    tiny_model, mode, vision_mode, vision_updates
):
    model = tiny_model(
        training_recipe="flow" if mode == "b" else "ki",
        finetuning_mode=mode,
        a_vision_mode=vision_mode,
    )
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.get_optim_params(), lr=0.01, weight_decay=0)
    loss, outputs = model(batch())
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    combined = model.network.paligemma_with_expert
    assert bool(nonzero_grad(combined.paligemma.vision_tower)) == vision_updates
    assert bool(nonzero_grad(combined.paligemma.language_model)) == (mode != "b")
    assert nonzero_grad(combined.gemma_expert)
    assert not combined.embed_language_tokens(torch.tensor([[2, 3]])).requires_grad
    changed = [name for name, p in model.named_parameters() if not torch.equal(before[name], p)]
    assert changed
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(before[name], p), name
    if mode != "b":
        assert all(
            "lora_" in name for name in changed if ".paligemma.model.language_model." in name
        )
        assert "fast_loss" in outputs
    if mode == "a":
        # Action/time projections remain trainable; transformer base weights stay frozen.
        assert all("lora_" in name for name in changed if ".gemma_expert.model." in name)
        assert "network.action_out_proj.weight" in changed
    if mode == "d":
        assert all("lora_" in name for name in changed if ".paligemma." in name)


@pytest.mark.parametrize("recipe", ["flow", "ki"])
def test_lora_base_parity_and_self_contained_checkpoint_resume(tiny_model, tmp_path, recipe):
    base = tiny_model(gradient_checkpointing=False).eval()
    weights = tmp_path / "base.safetensors"
    save_model(base.network, weights)
    model = tiny_model(
        finetuning_mode="a", training_recipe=recipe, gradient_checkpointing=False
    ).eval()
    load_base_weights(model.network, weights)
    inputs = batch()
    noise, time = torch.zeros_like(inputs["action"]), torch.full((2,), 0.5)
    torch.testing.assert_close(
        base.network(inputs["observation"], inputs["action"], noise=noise, time=time),
        model.network(inputs["observation"], inputs["action"], noise=noise, time=time),
        rtol=1e-5,
        atol=5e-7,
    )
    optimizer = torch.optim.AdamW(model.get_optim_params(), lr=0.001)
    model(inputs)[0].backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    # Actual native adapter export, relocation, strict reload, then an optimizer step.
    sp = tmp_path / "original.model"
    sp.write_bytes(b"tokenizer resource")
    model.config.tokenizer_path = str(sp)
    if recipe == "ki":
        fast_dir = tmp_path / "fast"
        fast_dir.mkdir()
        for filename in ("tokenizer.json", "processor_config.json"):
            (fast_dir / filename).write_text("{}")
        model.config.fast_tokenizer_path = str(fast_dir)
    else:
        model.config.fast_tokenizer_path = ""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = {"config": asdict(model.config)}
    component = PI05Algorithm()
    component.save(checkpoint, model, manifest)
    optimizer_state = deepcopy(optimizer.state_dict())
    moved = checkpoint.rename(tmp_path / "moved")
    restored_options = component.checkpoint_options(
        moved, manifest["config"], "cpu", None, "checkpoint"
    )
    if recipe == "ki":
        assert Path(restored_options["fast_tokenizer_path"]).is_relative_to(moved)
    restored = component.build(restored_options).eval()
    component.load(moved, restored)
    resumed_optimizer = torch.optim.AdamW(restored.get_optim_params(), lr=0.001)
    resumed_optimizer.load_state_dict(optimizer_state)
    for candidate, optim in ((model, optimizer), (restored, resumed_optimizer)):
        torch.manual_seed(87)
        candidate(inputs)[0].backward()
        optim.step()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


def test_token_loss_ignores_padding_and_uses_shifted_targets():
    hidden = torch.randn(2, 7, 5, requires_grad=True)
    context = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    targets = torch.tensor([[0, 1, 2], [3, 4, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    head = torch.nn.Linear(5, 6)
    actual = token_cross_entropy(hidden, context, targets, mask, head, lambda f, *a: f(*a), 2)
    expected = torch.stack(
        [
            torch.nn.functional.cross_entropy(head(hidden[0, [1, 4, 5]]), targets[0]),
            torch.nn.functional.cross_entropy(head(hidden[1, [2, 4]]), targets[1, :2]),
        ]
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert not hidden.grad[:, 3].any()


def test_recipe_and_mode_are_independent_cli_fields():
    from alohamini.learning.train_config import parse_training_args

    for mode in ("a", "b", "c", "d"):
        for recipe in ("flow", "ki"):
            cfg, _ = parse_training_args(
                [
                    "--config",
                    f"examples/learning/pi05/{mode}.json",
                    f"--policy.training_recipe={recipe}",
                ]
            )
            assert cfg["model"]["finetuning_mode"] == mode
            assert cfg["model"]["training_recipe"] == recipe
            if mode == "b" and recipe == "ki":
                with pytest.raises(ValueError, match="freezes the VLM"):
                    PI05TrainingConfig(**options(finetuning_mode=mode, training_recipe=recipe))
            else:
                PI05TrainingConfig(**options(finetuning_mode=mode, training_recipe=recipe))
    preset = PI05Algorithm.apply_preset({})
    assert preset["save_freq"] == 1000 and preset["log_freq"] == 100
    assert preset["ema"]["enable"] is False
    assert preset["model"]["gradient_checkpointing"] is True
    assert preset["model"]["network"]["pytorch_compile_mode"] == "max-autotune"


def test_camera_order_survives_sorted_json_and_legacy_checkpoint_loading(tmp_path):
    names = ("wrist_right", "forward", "wrist_left")
    features = {f"observation.images.{name}": {"type": "VISUAL", "shape": [3, 224, 224]}
                for name in names}
    features["observation.state"] = {"type": "STATE", "shape": [18]}
    config = PI05TrainingConfig(**options(input_features=features, fast_tokenizer_path=""))
    assert config.camera_order == names
    restored = PI05TrainingConfig(**json.loads(json.dumps(asdict(config), sort_keys=True)))
    assert restored.camera_order == names

    (tmp_path / "tokenizer.model").write_bytes(b"test")
    old = asdict(config)
    old.pop("camera_order")
    old["tokenizer_path"] = "tokenizer.model"
    options_from_checkpoint = PI05Algorithm().checkpoint_options(
        tmp_path, old, "cpu", None, "checkpoint"
    )
    legacy = PI05TrainingConfig(**options_from_checkpoint)
    assert legacy.camera_order == ("forward", "wrist_left", "wrist_right")
    with pytest.raises(ValueError, match="camera_order"):
        PI05TrainingConfig(**options(input_features=features, camera_order=("chest",)))
    with pytest.raises(ValueError, match="camera"):
        PI05TrainingConfig(**options(input_features={
            "observation.state": {"type": "STATE", "shape": [18]}
        }))


def test_language_only_cotraining_does_not_update_expert(tiny_model):
    model = tiny_model(training_recipe="ki", finetuning_mode="d")
    inputs = batch()
    loss = model.network.forward_language(
        inputs["observation"], inputs["fast_tokens"], inputs["fast_mask"], ce_chunk_size=2
    )
    loss.mean().backward()
    combined = model.network.paligemma_with_expert
    assert nonzero_grad(combined.paligemma.language_model)
    assert nonzero_grad(combined.paligemma.vision_tower)
    assert not nonzero_grad(combined.gemma_expert)


def test_fast_tokenizer_matches_official_dct_bpe_and_rejects_truncation(tmp_path):
    from scipy.fft import dct
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    from alohamini.policies.pi05.fast_tokenizer import FASTTokenizer

    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(["".join(map(chr, range(256)))], vocab_size=256)
    bpe._tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"scale": 10, "min_token": -100, "vocab_size": bpe.get_vocab_size()})
    )
    sp = SimpleNamespace(vocab_size=lambda: 2048, encode=lambda *a, **kw: [10, 1])
    local = FASTTokenizer(tmp_path, sp, 300)
    actions = np.random.default_rng(1).uniform(-1, 1, (2, 5, 3)).astype(np.float32)
    official_bpe = PreTrainedTokenizerFast(tokenizer_object=bpe, clean_up_tokenization_spaces=False)
    expected = []
    for chunk in np.around(dct(actions, axis=1, norm="ortho") * 10):
        text = "".join(map(chr, np.maximum(chunk.flatten() + 100, 0).astype(int)))
        expected.append(official_bpe(text)["input_ids"])
    assert local.encode_actions(actions) == expected
    tokens, mask = local.targets(actions, device="cpu")
    assert tokens[0, : len(expected[0])].tolist() == [2048 - 1 - 128 - n for n in expected[0]]
    assert mask.sum(1).tolist() == [len(row) + 2 for row in expected]
    local.max_tokens = 1
    with pytest.raises(ValueError, match="never truncate actions"):
        local.targets(actions, device="cpu")


def test_vlm_dataset_sampling_is_reproducible_and_rejects_changed_manifest(tmp_path):
    from PIL import Image

    from alohamini.policies.pi05.vlm_data import VLMCoTrainingData, manifest_digest

    Image.new("RGB", (32, 24), (127, 128, 129)).save(tmp_path / "image.png")
    path = tmp_path / "vlm.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"images": {"forward": "image.png"}, "prompt": str(i), "answer": "move"})
            for i in range(5)
        )
    )
    config = PI05TrainingConfig(
        **options(
            training_recipe="ki", vlm_data_path=str(path), vlm_data_sha256=manifest_digest(path)
        )
    )
    tokenizer = SimpleNamespace(
        _tokenizer=SimpleNamespace(encode=lambda text, **kwargs: [2] + [ord(c) for c in text])
    )
    data = VLMCoTrainingData(config, tokenizer)
    rng = torch.get_rng_state()
    first = data.sample(3, device="cpu")
    torch.set_rng_state(rng)
    second = data.sample(3, device="cpu")
    torch.testing.assert_close(
        first["observation"].tokenized_prompt, second["observation"].tokenized_prompt
    )
    assert list(first["observation"].images) == ["forward"]
    assert first["observation"].images["forward"].shape == (3, 3, 224, 224)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest changed"):
        VLMCoTrainingData(config, tokenizer)


def test_vlm_dataset_infers_its_own_views_and_masks_missing_views(tmp_path, monkeypatch):
    from PIL import Image

    from alohamini.policies.pi05.vlm_data import VLMCoTrainingData, manifest_digest

    Image.new("RGB", (32, 24), (127, 128, 129)).save(tmp_path / "image.png")
    path = tmp_path / "vlm.jsonl"
    views = [("side",), ("overhead", "side", "chest", "wrist_left")]
    path.write_text("\n".join(json.dumps({
        "images": dict.fromkeys(names, "image.png"), "prompt": "pick", "answer": "move"
    }) for names in views))
    config = PI05TrainingConfig(**options(
        training_recipe="ki", vlm_data_path=str(path), vlm_data_sha256=manifest_digest(path)
    ))
    tokenizer = SimpleNamespace(_tokenizer=SimpleNamespace(encode=lambda *a, **kw: [2, 3]))
    data = VLMCoTrainingData(config, tokenizer)
    monkeypatch.setattr(torch, "randint", lambda *a, **kw: torch.tensor([0, 1]))
    observation = data.sample(2, device="cpu")["observation"]
    assert tuple(observation.images) == ("side", "overhead", "chest", "wrist_left")
    assert observation.image_masks["side"].tolist() == [True, True]
    for name in ("overhead", "chest", "wrist_left"):
        assert observation.image_masks[name].tolist() == [False, True]
        assert torch.all(observation.images[name][0] == -1)


@pytest.mark.parametrize("recipe", ["flow", "ki"])
def test_four_camera_training_and_inference_use_every_view(tiny_model, monkeypatch, recipe):
    from test_pi05 import processor

    from alohamini.policies.pi05 import tokenizer
    from alohamini.policies.pi05.adapter import PI05BatchProcessor

    names = ("wrist_right", "overhead", "wrist_left", "chest")
    features = {f"observation.images.{name}": {"type": "VISUAL", "shape": [3, 32, 32]}
                for name in names}
    features["observation.state"] = {"type": "STATE", "shape": [18]}
    model = tiny_model(input_features=features, training_recipe=recipe)
    reference = processor()
    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", lambda *args: reference.tokenizer)
    prepare = PI05BatchProcessor(model.config, reference.stats, "cpu")
    raw = {f"observation.images.{name}": torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
           for name in reversed(names)}
    raw.update({"observation.state": torch.zeros(2, 18), "task": ["pick", "pick"]})
    prepared = prepare(raw)
    prepared.update({key: value for key, value in batch().items() if key != "observation"})
    observation = prepared["observation"]
    assert tuple(observation.images) == names
    for image in observation.images.values():
        image.requires_grad_()
    loss, _ = model(prepared)
    assert torch.isfinite(loss)
    loss.backward()
    for image in observation.images.values():
        assert image.grad is not None and torch.isfinite(image.grad).all()
        assert image.grad.abs().sum() > 0  # Includes the fourth view, through real transformers.
    model.eval()
    noise = torch.zeros(2, 2, 32)
    expected = model.network.sample_actions("cpu", observation, noise=noise, num_steps=2)
    changed = SimpleNamespace(**vars(observation))
    changed.images = {**observation.images, "chest": -observation.images["chest"]}
    actual = model.network.sample_actions("cpu", changed, noise=noise, num_steps=2)
    assert torch.isfinite(expected).all() and expected.shape == (2, 2, 32)
    assert (actual - expected).abs().max() > 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["a", "d"])
@pytest.mark.parametrize("autocast", [False, True])
def test_cuda_bfloat16_ki_lora_step_and_flow_inference(tiny_model, mode, autocast):

    model = tiny_model(training_recipe="ki", finetuning_mode=mode).cuda()
    model.network.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    # Model precision conversion precedes adapter insertion in production.
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()
    inputs = batch()
    for key, value in vars(inputs["observation"]).items():
        if isinstance(value, dict):
            setattr(inputs["observation"], key, {k: v.cuda() for k, v in value.items()})
        elif torch.is_tensor(value):
            setattr(inputs["observation"], key, value.cuda())
    for key in ("action", "fast_tokens", "fast_mask"):
        inputs[key] = inputs[key].cuda()
    optimizer = torch.optim.AdamW(model.get_optim_params(), lr=1e-4)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        loss, _ = model(inputs)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        prediction = model.predict_action_chunk(inputs)
    assert torch.isfinite(prediction).all()

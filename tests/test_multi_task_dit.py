"""Real tiny CLIP/DiT models: conditioning, objectives, portable assets and trainer EMA."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_diffusion_fastwam import check_named_action, features
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.ema import EMAConfig, EMAManager
from alohamini.learning.optim import make_optimizer_and_scheduler, resolve_optimization
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.policies.registry import algorithm


@pytest.mark.parametrize("window,expected", [({}, 7), ({"horizon": 64, "n_action_steps": 32}, 31)])
def test_tail_exclusion_tracks_window_and_preserves_overrides(window, expected):
    recipe = algorithm("multi_task_dit")
    for override, result in (
        ({}, expected),
        ({"drop_n_last_frames": None}, expected),
        ({"drop_n_last_frames": 0}, 0),
        ({"drop_n_last_frames": 7}, 7),
    ):
        options = {**window, **override}
        config = recipe.config_class(**features(), **options)
        assert config.drop_n_last_frames == result
        assert recipe.sample_spec(options)["drop_n_last_frames"] == result


def make_clip_assets(root):
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import (
        CLIPTextConfig,
        CLIPTextModel,
        CLIPTokenizer,
        CLIPVisionConfig,
        CLIPVisionModel,
    )

    vision, text = root / "clip_vision", root / "clip_text"
    alphabet = sorted(ByteLevel.alphabet())
    vocab = {token: i for i, token in enumerate(alphabet + [c + "</w>" for c in alphabet])}
    vocab.update({"<|startoftext|>": len(vocab), "<|endoftext|>": len(vocab) + 1})
    tokenizer = CLIPTokenizer(vocab=vocab, merges=[], model_max_length=32)
    tokenizer.save_pretrained(text)
    CLIPTextModel(
        CLIPTextConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    ).save_pretrained(text)
    CLIPVisionModel(
        CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=16,
            patch_size=8,
        )
    ).save_pretrained(vision)
    return vision, text


def dit_options(root):
    vision, text = make_clip_assets(root)
    return dict(
        vision_encoder_name=str(vision),
        text_encoder_name=str(text),
        hidden_dim=16,
        num_heads=2,
        num_layers=2,
        timestep_embed_dim=8,
        horizon=4,
        n_obs_steps=2,
        n_action_steps=2,
        drop_n_last_frames=0,
        image_resize_shape=[16, 16],
        image_crop_shape=None,
        dropout=0.0,
        tokenizer_max_length=32,
        num_train_timesteps=4,
        num_inference_steps=2,
        num_integration_steps=2,
    )


def data_and_model(recording, options):
    recipe = algorithm("multi_task_dit")
    recipe.options({"model": options}, "cpu")
    data = AlohaMiniDataset(
        recording,
        episodes=[0, 1],
        image_size=(32, 32),
        **recipe.sample_spec(options, cameras=["forward"]),
    )
    options = {
        **options,
        "input_features": data.input_features,
        "output_features": data.output_features,
    }
    stats = recipe.statistics(data, options)
    model = recipe.build(options)
    processor = recipe.processor(model, stats, "cpu")
    raw = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    return recipe, data, model, stats, processor(raw)


@pytest.mark.parametrize(
    "objective,sampler",
    [
        ("diffusion", "DDPM"),
        ("diffusion", "DDIM"),
        ("flow_matching", "euler"),
        ("flow_matching", "rk4"),
    ],
)
def test_real_clip_training_sampling_and_portable_checkpoint(
    recording, tmp_path, objective, sampler
):
    options = dit_options(tmp_path / "assets")
    options.update(objective=objective, do_mask_loss_for_padding=True)
    if objective == "diffusion":
        options.update(
            noise_scheduler_type=sampler,
            prediction_type="sample" if sampler == "DDIM" else "epsilon",
        )
    else:
        options.update(
            integration_method=sampler,
            timestep_sampling_strategy="uniform" if sampler == "rk4" else "beta",
        )
    recipe, data, model, stats, batch = data_and_model(recording, options)
    model.train()
    assert not model.observation_encoder.text_encoder.text_encoder.training
    cfg = resolve_optimization(
        {"steps": 3, "optimizer": {"type": "adam", "lr": 1e-3}}, model.config
    )
    optimizer, _ = make_optimizer_and_scheduler(cfg, model)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 1e-4]
    ema = EMAManager(model, EMAConfig(enable=True))
    assert not any("text_encoder.text_encoder" in name for name in ema.state_dict()["parameters"])
    for _ in range(2):
        optimizer.zero_grad()
        loss, metrics = model(batch)
        assert torch.isfinite(loss)
        torch.testing.assert_close(loss, metrics["objective_loss"])
        loss.backward()
        optimizer.step()
        ema.update(model)
    vision = model.observation_encoder.vision_encoder
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in vision.parameters())
    assert all(
        p.grad is None for p in model.observation_encoder.text_encoder.text_encoder.parameters()
    )
    model.eval()
    encoded = model.observation_encoder.encode(model._prepare_batch(batch))
    other = dict(batch)
    processor = recipe.processor(model, stats, "cpu")
    raw = next(iter(torch.utils.data.DataLoader(data, batch_size=2)))
    other.update(processor({**raw, "task": ["close", "open"]}))
    assert not torch.allclose(
        encoded, model.observation_encoder.encode(model._prepare_batch(other))
    )
    torch.manual_seed(9)
    expected = model.predict_action_chunk(batch)
    assert expected.shape == (2, 4, 18) and torch.isfinite(expected).all()
    target = tmp_path / "checkpoint"
    save_checkpoint(target, model, stats, data, training={})
    (tmp_path / "assets").rename(tmp_path / "old-assets")
    loaded = NativePolicy(target)
    torch.manual_seed(9)
    torch.testing.assert_close(loaded.model.predict_action_chunk(batch), expected, rtol=0, atol=0)
    assert loaded.manifest["assets"]["vision_encoder_name"]["storage"] == "bundled"
    assert loaded.predict(data[0]).shape == (4, 18)
    check_named_action(loaded, data)
    # A rollout must not replace an explicitly supplied offline history with queued observations.
    torch.manual_seed(9)
    torch.testing.assert_close(loaded.model.predict_action_chunk(batch), expected, rtol=0, atol=0)


def test_history_action_alignment_and_task_switch(recording, tmp_path, monkeypatch):
    _, _, model, _, batch = data_and_model(recording, dit_options(tmp_path))
    chunk = torch.arange(4.0)[None, :, None].expand(2, 4, 18)
    histories = []

    def predict(history):
        histories.append(history)
        return chunk

    monkeypatch.setattr(model, "predict_action_chunk", predict)
    current = {
        key: value[:, -1]
        if key == "observation.state" or key.startswith("observation.images.")
        else value
        for key, value in batch.items()
        if key not in ("action", "action_is_pad")
    }
    assert model.select_action(current).eq(1).all()
    torch.testing.assert_close(
        histories[0]["observation.state"][:, 0], current["observation.state"]
    )
    assert model.select_action(current).eq(2).all()
    assert len(histories) == 1
    assert model.select_action(current).eq(1).all()
    changed = {
        **current,
        "observation.language.tokens": current["observation.language.tokens"].roll(1, dims=1),
    }
    assert model.select_action(changed).eq(1).all()  # Flush pending actions for the previous task.
    assert len(histories) == 3


@pytest.mark.parametrize("objective", ["diffusion", "flow_matching"])
def test_objective_mask_and_accumulation_weight(recording, tmp_path, objective):
    opts = dit_options(tmp_path)
    opts.update(objective=objective, do_mask_loss_for_padding=True)
    recipe, _, model, _, batch = data_and_model(recording, opts)
    batch["action_is_pad"][:] = True
    batch["action_is_pad"][0, 1] = False
    assert recipe.loss_counts(batch) == {"_loss_weight": 1}
    # Compare the actual objective against an independently reduced error with fixed RNG.
    prepared = model._prepare_batch(batch)
    cond = model.observation_encoder.encode(prepared)
    seen = []

    class Zero(torch.nn.Module):
        def forward(self, x, t, conditioning_vec):
            seen.append((x, t))
            return torch.zeros_like(x)

    torch.manual_seed(7)
    noise = torch.randn_like(batch["action"])
    torch.manual_seed(7)
    actual = model.objective.compute_loss(Zero(), batch, cond)
    target = noise if objective == "diffusion" else batch["action"] - noise
    torch.testing.assert_close(actual, target[0, 1].square().mean())
    torch.manual_seed(12)
    plain, _ = model(batch)
    torch.manual_seed(12)
    scaled, _ = model({**batch, "_loss_weight": 0.25})
    torch.testing.assert_close(scaled, plain * 0.25)
    batch["action_is_pad"][:] = True
    zero, _ = model(batch)
    assert zero == 0
    zero.backward()


def test_preset_validation_and_multicamera(recording, tmp_path):
    recipe = algorithm("multi_task_dit")
    cfg = recipe.apply_preset({"policy": "multi_task_dit", "ema": {"enable": True}})
    assert recipe.apply_preset(cfg) == cfg
    assert cfg["paper_preset"]["reference_revision"] and cfg["ema"]["enable"]
    assert recipe.sample_spec({}, cameras=["front"])["delta_indices"]["action"] == list(
        range(-1, 31)
    )
    options = dit_options(tmp_path)
    _, data, _, _, batch = data_and_model(recording, options)
    options.update(
        input_features={
            **data.input_features,
            "observation.images.side": data.input_features["observation.images.forward"],
        },
        output_features=data.output_features,
        use_separate_rgb_encoder_per_camera=True,
        use_rope=False,
        use_positional_encoding=True,
    )
    model = recipe.build(options)
    batch["observation.images.side"] = batch["observation.images.forward"].flip(-1)
    loss, _ = model(batch)
    loss.backward()
    assert len(model.observation_encoder.vision_encoders) == 2
    for bad in (
        {"n_action_steps": 4},
        {"num_heads": 3},
        {"timestep_embed_dim": 2},
        {"tokenizer_padding_side": "left"},
        {"num_inference_steps": 0},
    ):
        with pytest.raises(ValueError):
            recipe.build({**options, **bad})
    stats = recipe.statistics(data)
    stats["observation.images.side"] = stats["observation.images.forward"]
    processor = recipe.processor(model, stats, "cpu")
    with pytest.raises(ValueError, match="task instruction"):
        processor({"observation.state": torch.zeros(2, 18), "task": ["", "pick"]})


def _identical(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _identical(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _identical(x, y)
    else:
        assert a == b


@pytest.mark.parametrize(
    "objective,processes", [("diffusion", 1), ("flow_matching", 1), ("diffusion", 2)]
)
def test_detached_training_exact_resume_with_ema(recording, tmp_path, objective, processes):
    from safetensors.torch import load_file

    from alohamini.learning.train_config import parse_training_args

    options = dit_options(tmp_path / "assets")
    options.update(objective=objective, do_mask_loss_for_padding=True, dropout=0.1)
    cfg = dict(
        policy="multi_task_dit",
        dataset=str(recording),
        model=options,
        device="cpu",
        mixed_precision="none",
        steps=2,
        train_episodes=[0],
        val_episodes=[1],
        image_size=[32, 32],
        batch_size=3,
        gradient_accumulation_steps=2,
        num_workers=0,
        save_freq=1,
        eval_steps=1,
        seed=31,
        deterministic_algorithms=True,
        scheduler={"type": "none"},
        ema={"enable": True},
        cpu_threads=2,
        num_processes=processes,
    )

    def run(settings, label, stop=False):
        path, log, pid = (tmp_path / f"{label}{suffix}" for suffix in (".json", ".log", ".pid"))
        path.write_text(json.dumps(settings))
        command = [sys.executable]
        if processes > 1:
            command += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={processes}",
            ]
        command += [str(Path(__file__).resolve()), "--worker", str(path)]
        with log.open("wb") as stream:
            proc = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, "DIT_TEST_STOP": str(int(stop))},
            )
        pid.write_text(str(proc.pid))
        try:
            assert proc.wait(timeout=120) == 0, log.read_text()
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)

    run({**cfg, "output_dir": str(tmp_path / "full")}, "full")
    run(
        {**cfg, "output_dir": str(tmp_path / "resumed"), "steps": 1 if processes > 1 else 2},
        "interrupted",
        stop=processes == 1,
    )
    resumed_cfg, _ = parse_training_args(
        [
            "--config_path",
            str(tmp_path / "resumed/checkpoint/train_config.json"),
            "--resume=true",
            "--steps=2",
        ]
    )
    (tmp_path / "assets").rename(tmp_path / "old-assets")
    run(resumed_cfg, "resume")
    paths = [tmp_path / name / "checkpoint" for name in ("full", "resumed")]
    _identical(*(load_file(path / "model.safetensors") for path in paths))
    states = [
        torch.load(path.resolve().parent / "training_state/state.pt", weights_only=True)
        for path in paths
    ]
    _identical(*states)
    assert states[0]["ema"]["num_updates"] == 2
    for path in paths:
        policy = NativePolicy(path.parent / "checkpoint-ema")
        assert policy.manifest["training"]["weights"] == "ema"
    assert (
        json.loads((tmp_path / "resumed/offline-evaluation.json").read_text())["valid_action_steps"]
        > 0
    )
    for line in (tmp_path / "full/metrics.jsonl").read_text().splitlines():
        entry = json.loads(line)
        assert entry["metrics"]["loss_" + objective] == pytest.approx(entry["loss"], rel=1e-5)


@pytest.mark.parametrize("objective", ["diffusion", "flow_matching"])
def test_cuda_bfloat16_training_and_sampling(recording, tmp_path, objective):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # Exercise the installed kernels, including the actual GPU's architecture.
    x = torch.ones(16, 16, device="cuda")
    assert (x @ x).eq(16).all()
    options = dit_options(tmp_path)
    options["objective"] = objective
    _, _, model, _, batch = data_and_model(recording, options)
    model.to("cuda").train()
    batch = {key: value.to("cuda") for key, value in batch.items()}
    ema = EMAManager(model, EMAConfig(enable=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = model(batch)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    ema.update(model)
    with ema.average_parameters(model), torch.autocast("cuda", dtype=torch.bfloat16):
        predicted = model.predict_action_chunk(batch)
    assert predicted.shape == (2, 4, 18) and torch.isfinite(predicted).all()


if __name__ == "__main__":
    import alohamini.learning.train as training

    torch.set_num_threads(2)
    if os.environ.get("DIT_TEST_STOP") == "1":
        save = training.save_training_checkpoint

        def save_and_stop(*args, **kwargs):
            save(*args, **kwargs)
            raise SystemExit(0)

        training.save_training_checkpoint = save_and_stop
    training.train(json.loads(Path(sys.argv[-1]).read_text()))

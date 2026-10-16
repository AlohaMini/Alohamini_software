"""Real reduced CDP2 and DiT-Block networks, dataset contracts and native training."""

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
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.policies.registry import algorithm


def small_options(kind):
    options = dict(
        **features(),
        pretrained_backbone_weights=None,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        crop_shape=None,
        horizon=4,
        n_obs_steps=2,
        n_action_steps=1,
        num_train_timesteps=4,
        num_inference_steps=4,
    )
    if kind == "dit_policy":
        options.update(feedforward_dim=64, time_dim=16, gaussian_blur=False)
    return options


def data_and_model(root, kind):
    recipe = algorithm(kind)
    options = small_options(kind)
    data = AlohaMiniDataset(
        root,
        episodes=[0, 1],
        image_size=(32, 32),
        state="joint_position,base_velocity,lift_height",
        **recipe.sample_spec(options, cameras=["forward"]),
    )
    stats = recipe.statistics(data, options)
    model = recipe.build(options, stats)
    processor = recipe.processor(model, stats, "cpu")
    batch = processor(next(iter(torch.utils.data.DataLoader(data, batch_size=2))))
    return recipe, data, model, stats, batch


@pytest.mark.parametrize("kind", ["cdp", "dit_policy"])
def test_backward_checkpoint_and_named_actions(recording, tmp_path, kind):
    recipe, data, model, stats, batch = data_and_model(recording, kind)
    loss, _ = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
    model.eval()
    torch.manual_seed(25)
    expected = model.predict_action_chunk(batch)
    changed = {**batch, "action": torch.randn_like(batch["action"]) * 1e6}
    torch.manual_seed(25)
    torch.testing.assert_close(model.predict_action_chunk(changed), expected, rtol=0, atol=0)
    assert expected.shape == batch["action"].shape
    path = tmp_path / kind
    save_checkpoint(path, model, stats, data, training={})
    loaded = NativePolicy(path)
    torch.manual_seed(25)
    restored = loaded.model.predict_action_chunk(batch)
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    assert loaded.predict(data[0]).shape == expected.shape[1:]
    check_named_action(loaded, data)
    # Exercise more than one chunk: this enters CDP's persistent KV-cache path.
    single = {k: v[:1, -1] for k, v in batch.items() if k in model.config.input_features}
    model.reset()
    torch.manual_seed(92)
    rollout = torch.stack([model.select_action(single) for _ in range(5)])
    assert torch.isfinite(rollout).all()
    model.reset()
    torch.manual_seed(92)
    torch.testing.assert_close(
        torch.stack([model.select_action(single) for _ in range(5)]), rollout, rtol=0, atol=0
    )


def test_cdp_past_actions_are_episode_bounded_and_normalized(recording):
    recipe, data, model, stats, batch = data_and_model(recording, "cdp")
    assert batch["past_action_is_pad"][0].all()
    assert batch["past_action"][0].eq(0).all()
    sample = data[2]
    assert sample["past_action_is_pad"].tolist() == [False, False]
    torch.testing.assert_close(sample["past_action"][0], data[0]["action"][0])
    torch.testing.assert_close(sample["past_action"][1], data[1]["action"][0])
    start = next(i for i, row in enumerate(data.sample_indices) if data.locations[row][0] == 1)
    assert data[start]["sequence_index"] == 0
    assert data[start]["past_action_is_pad"].all()
    with pytest.raises(ValueError, match="negative row offsets"):
        AlohaMiniDataset(recording, episodes=[0], delta_indices={"past_action": [-1, 0]})


def test_cdp_causal_mask_and_incremental_cache_equivalence():
    from alohamini.policies.cdp.configuration_cdp import CDPConfig
    from alohamini.policies.cdp.transformer import CausalTransformer

    config = CDPConfig(**small_options("cdp"))
    net = CausalTransformer(config, 5).eval()
    action, obs = torch.randn(2, 4, 18), torch.randn(2, 2, 5)
    time, start = torch.tensor([[2, 3], [1, 2]]), torch.tensor([0, 1])
    result, cache = net(action, time, obs, start)
    changed = action.clone()
    changed[:, -1] += 100
    predicted, _ = net(changed, time, obs, start)
    torch.testing.assert_close(predicted[:, :-1], result[:, :-1], rtol=0, atol=0)
    # Frozen history K/V at a fixed diffusion time gives the same future prediction.
    cached, _ = net(action[:, 2:], time, obs[:, 2:], start + 2, history_size=0, caches=cache)
    torch.testing.assert_close(cached, result[:, 2:], rtol=1e-5, atol=1e-6)


def test_cdp_ddpm_step_matches_diffusers(recording):
    from diffusers import DDPMScheduler

    _, _, model, _, _ = data_and_model(recording, "cdp")
    scheduler = DDPMScheduler(
        num_train_timesteps=4, beta_schedule="squaredcos_cap_v2", prediction_type="sample"
    )
    x, prediction = torch.randn(2, 2, 18), torch.randn(2, 2, 18)
    for step in (0, 1, 3):
        torch.manual_seed(8)
        actual = model._step(prediction, torch.full((2, 2), step), x)
        torch.manual_seed(8)
        expected = scheduler.step(prediction, step, x).prev_sample
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model._step(prediction, torch.full((2, 2), -1), x), x)


def test_presets_preserve_explicit_overrides_and_distinct_windows():
    cdp, dit = algorithm("cdp"), algorithm("dit_policy")
    assert cdp.apply_preset({})["ema"]["enable"]
    assert not cdp.apply_preset({"ema": {"enable": False}})["ema"]["enable"]
    assert not dit.apply_preset({}).get("ema", {}).get("enable", False)
    assert cdp.sample_spec({})["delta_indices"]["action"] == list(range(8))
    assert dit.sample_spec({})["delta_indices"]["action"] == list(range(100))
    for recipe in (cdp, dit):
        from alohamini.learning.optim import resolve_optimization

        config = recipe.config_class(
            **{
                **features(),
                "crop_shape": None,
            }
        )
        resolved = resolve_optimization(recipe.apply_preset({}), config)
        assert resolved["steps"] > resolved["scheduler"]["warmup_steps"]
        assert resolved["optimizer"]["grad_clip_norm"] == 0
        settings = {"batch_size": 2, "model": {"drop_n_last_frames": 3}}
        preset = recipe.apply_preset(settings)
        assert preset["batch_size"] == 2
        assert preset["paper_preset"]["overrides"] == settings
        assert recipe.sample_spec(preset["model"])["drop_n_last_frames"] == 3


def test_cdp_optimizer_override_applies_to_transformer():
    from alohamini.learning.optim import make_optimizer_and_scheduler, resolve_optimization

    model = algorithm("cdp").build(small_options("cdp"))
    cfg = resolve_optimization({"steps": 2, "optimizer": {"weight_decay": 0.02}}, model.config)
    optimizer, _ = make_optimizer_and_scheduler(cfg, model)
    assert [g["weight_decay"] for g in optimizer.param_groups] == [0.02, 0.0, 1e-6]


@pytest.mark.parametrize("kind", ["cdp", "dit_policy"])
@pytest.mark.parametrize("shared", [False, True])
def test_multicamera_augmentation_and_encoder_gradients(kind, shared):
    options = small_options(kind)
    options["share_rgb_encoder"] = shared
    options["input_features"]["observation.images.wrist"] = {
        "type": "VISUAL",
        "shape": [3, 32, 32],
    }
    options["crop_shape"] = [28, 28]
    if kind == "dit_policy":
        options.update(gaussian_blur=True, pooled_vision=shared)
    model = algorithm(kind).build(options)
    batch = {key: torch.rand(2, 2, 3, 32, 32) for key in model.config.image_features}
    batch["observation.state"] = torch.rand(2, 2, 18)
    encoded = model.encode(batch)
    encoded.square().mean().backward()
    assert torch.isfinite(encoded).all()
    assert len(model.vision.encoders) == (1 if shared else 2)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.vision.parameters()
    )
    model.eval()
    torch.testing.assert_close(model.encode(batch), model.encode(batch), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["cdp", "dit_policy"])
def test_cuda_bfloat16_forward_and_sampling(recording, kind):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    tensor = torch.ones(16, 16, device="cuda")
    assert (tensor @ tensor).eq(16).all()
    _, _, model, _, batch = data_and_model(recording, kind)
    model.cuda()
    batch = {k: v.cuda() for k, v in batch.items()}
    ema = EMAManager(model, EMAConfig(enable=True))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = model(batch)
    loss.backward()
    torch.optim.AdamW(model.parameters()).step()
    ema.update(model)
    model.eval()
    with ema.average_parameters(model), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model.predict_action_chunk(batch)
        model.reset()
        snapshot = {k: v[:1, -1] for k, v in batch.items() if k in model.config.input_features}
        for _ in range(3):
            assert torch.isfinite(model.select_action(snapshot)).all()
    assert torch.isfinite(loss) and torch.isfinite(prediction).all()


def identical(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            identical(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            identical(left, right)
    else:
        assert a == b


@pytest.mark.parametrize(
    "kind,processes", [("cdp", 1), ("dit_policy", 1), ("cdp", 2), ("dit_policy", 2)]
)
def test_detached_training_resume_and_ema(recording, tmp_path, kind, processes):
    from safetensors.torch import load_file

    from alohamini.learning.train_config import parse_training_args

    options = small_options(kind)
    options["dropout"] = 0.1
    cfg = dict(
        policy=kind,
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
                env={**os.environ, "NATIVE_DIT_TEST_STOP": str(int(stop))},
            )
        pid.write_text(str(proc.pid))
        try:
            assert proc.wait(timeout=180) == 0, log.read_text()
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
    resume, _ = parse_training_args(
        [
            "--config_path",
            str(tmp_path / "resumed/checkpoint/train_config.json"),
            "--resume=true",
            "--steps=2",
        ]
    )
    run(resume, "resume")
    paths = [tmp_path / name / "checkpoint" for name in ("full", "resumed")]
    identical(*(load_file(path / "model.safetensors") for path in paths))
    states = [
        torch.load(p.resolve().parent / "training_state/state.pt", weights_only=True) for p in paths
    ]
    identical(*states)
    assert states[0]["ema"]["num_updates"] == 2
    for path in paths:
        assert NativePolicy(path.parent / "checkpoint-ema").manifest["training"]["weights"] == "ema"
    for line in (tmp_path / "full/metrics.jsonl").read_text().splitlines():
        entry = json.loads(line)
        assert entry["metrics"]["loss_diffusion"] == pytest.approx(entry["loss"], rel=1e-5)


if __name__ == "__main__":
    import alohamini.learning.train as training

    torch.set_num_threads(2)
    if os.environ.get("NATIVE_DIT_TEST_STOP") == "1":
        save = training.save_training_checkpoint

        def save_and_stop(*args, **kwargs):
            save(*args, **kwargs)
            raise SystemExit(0)

        training.save_training_checkpoint = save_and_stop
    training.train(json.loads(Path(sys.argv[-1]).read_text()))

"""Weight-update precision and checkpoint restoration, without model downloads."""

import json
from types import SimpleNamespace

import pytest
import torch
from test_native_learning import recording as recording
from test_smolvla import options
from test_smolvla import tiny_assets as tiny_assets

from alohamini.learning.precision import apply_trainable_dtype


class MixedWeights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((1, 1), 0.01, dtype=torch.bfloat16))
        self.tied_weight = self.weight
        self.frozen = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)
        self.projection = torch.nn.Linear(1, 1)
        self.register_buffer("running_mean", torch.ones(1, dtype=torch.bfloat16))
        self.register_buffer("counter", torch.tensor(1))


@pytest.mark.parametrize("mode", ["preserve", "float32"])
def test_trainable_storage_preserves_frozen_weights_buffers_and_aliases(mode):
    model = MixedWeights()
    weight = model.weight
    frozen = model.frozen
    buffer = model.running_mean
    original = weight.detach().float().clone()
    assert apply_trainable_dtype(model, mode) is model
    assert model.weight is weight and model.tied_weight is weight
    assert model.weight.dtype == (torch.float32 if mode == "float32" else torch.bfloat16)
    torch.testing.assert_close(model.weight.float(), original, rtol=0, atol=0)
    assert model.projection.weight.dtype == torch.float32
    assert model.frozen is frozen and frozen.dtype == torch.bfloat16 and not frozen.requires_grad
    assert model.running_mean is buffer and buffer.dtype == torch.bfloat16
    assert model.counter.dtype == torch.int64


@pytest.mark.parametrize("mode", [None, "bfloat16", "float16", "auto", True])
def test_invalid_storage_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="trainable_dtype"):
        apply_trainable_dtype(MixedWeights(), mode)


def test_fp32_storage_keeps_updates_smaller_than_bf16_spacing_under_autocast():
    updated = {}
    for mode in ("preserve", "float32"):
        model = apply_trainable_dtype(MixedWeights(), mode)
        optimizer = torch.optim.AdamW(
            [model.weight], lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10
        )
        before = model.weight.detach().clone()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = torch.nn.functional.linear(torch.ones(1, 1), model.weight)
            loss = output.float().square().mean()
        assert output.dtype == torch.bfloat16
        loss.backward()
        optimizer.step()
        updated[mode] = not torch.equal(before, model.weight)
        for name in ("exp_avg", "exp_avg_sq"):
            assert optimizer.state[model.weight][name].dtype == model.weight.dtype
    assert updated == {"preserve": False, "float32": True}


def test_cli_storage_mode_and_json_validation(tmp_path):
    from alohamini.learning.train_config import parse_training_args

    args = ["--dataset.root=/tmp/data", "--output_dir=/tmp/run"]
    cfg, _ = parse_training_args(args)
    assert cfg["trainable_dtype"] == "preserve"
    cfg, _ = parse_training_args([*args, "--trainable_dtype=float32", "--mixed_precision=bfloat16"])
    assert cfg["trainable_dtype"] == "float32" and cfg["mixed_precision"] == "bfloat16"
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps({"trainable_dtype": "bfloat16"}))
    with pytest.raises(ValueError, match="trainable_dtype"):
        parse_training_args([*args, "--config", str(config)])
    with pytest.raises(SystemExit):
        parse_training_args([*args, "--trainable_dtype=bfloat16"])
    from alohamini.learning.train import launch_training

    with pytest.raises(ValueError, match="trainable_dtype"):
        launch_training({"trainable_dtype": "bfloat16"})


def test_bf16_smolvla_checkpoint_preserves_fp32_updates_and_finetuning(
    recording, tiny_assets, tmp_path
):
    from alohamini.learning.checkpoint import initialize_policy, resolve_pretrained
    from alohamini.learning.data import AlohaMiniDataset
    from alohamini.learning.policy import NativePolicy, make_policy, save_checkpoint
    from alohamini.policies.registry import algorithm

    assets = tmp_path / "config.json"
    backbone = json.loads(assets.read_text())
    backbone["text_config"]["dtype"] = "bfloat16"
    assets.write_text(json.dumps(backbone))
    samples = AlohaMiniDataset(recording, episodes=[0], chunk_size=3, image_size=(32, 32))
    opts = options(tiny_assets)
    opts.update(
        max_state_dim=32,
        max_action_dim=32,
        input_features=samples.input_features,
        output_features=samples.output_features,
    )
    native = make_policy("smolvla", opts)
    assert (
        native.model.vlm_with_expert.lm_expert.layers[0].self_attn.q_proj.weight.dtype
        == torch.bfloat16
    )
    model = make_policy("smolvla", opts, trainable_dtype="float32")
    assert {p.dtype for p in model.parameters() if p.requires_grad} == {torch.float32}
    weight = model.model.vlm_with_expert.lm_expert.layers[0].self_attn.q_proj.weight
    with torch.no_grad():
        weight[0, 0] = 0.010001
    assert weight[0, 0].item() != weight[0, 0].bfloat16().float().item()
    stats = algorithm("smolvla").statistics(samples)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model,
        stats,
        samples,
        training={"mixed_precision": "bfloat16"},
    )
    restored = NativePolicy(checkpoint)
    assert restored.manifest["training"]["trainable_dtype"] == "float32"
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[name], expected, rtol=0, atol=0)
    settings = resolve_pretrained({"pretrained_path": str(checkpoint)})
    assert settings["trainable_dtype"] == "float32"
    assert (
        resolve_pretrained({"pretrained_path": str(checkpoint), "trainable_dtype": "preserve"})[
            "trainable_dtype"
        ]
        == "preserve"
    )
    fresh = make_policy("smolvla", settings["model"], trainable_dtype=settings["trainable_dtype"])
    initialize_policy(algorithm("smolvla"), fresh, settings)
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[name], expected, rtol=0, atol=0)
    rejected = tmp_path / "wrong-storage"
    with pytest.raises(ValueError, match="trainable_dtype"):
        save_checkpoint(rejected, model, stats, samples, training={"trainable_dtype": "preserve"})
    assert not rejected.exists()


@pytest.mark.parametrize(
    "saved_mode,current_mode,rejected",
    [
        (None, "preserve", False),
        ("float32", "float32", False),
        (None, "float32", True),
        ("float32", "preserve", True),
    ],
)
def test_resume_storage_mode_and_legacy_default(tmp_path, saved_mode, current_mode, rejected):
    from alohamini.learning.training_state import load_training_checkpoint

    checkpoint = tmp_path / "checkpoints/000001"
    (checkpoint / "training_state").mkdir(parents=True)
    (checkpoint / "pretrained_model").mkdir()
    saved = dict(
        output_dir=str(tmp_path),
        steps=2,
        optimizer={},
        scheduler={"type": "none"},
        world_size=1,
        gradient_accumulation_steps=1,
        distributed_backend="ddp",
    )
    if saved_mode is not None:
        saved["trainable_dtype"] = saved_mode
    (checkpoint / "pretrained_model/train_config.json").write_text(json.dumps(saved))
    (checkpoint / "pretrained_model/policy.json").write_text(
        json.dumps(dict(sample_filter={}, table_sha256="same", training={}, config={}))
    )
    torch.save({"step": 1}, checkpoint / "training_state/state.pt")
    current = {**saved, "trainable_dtype": current_mode, "resume": True}
    samples = SimpleNamespace(sample_filter={}, table_sha256="same")
    if rejected:
        with pytest.raises(ValueError, match="trainable_dtype"):
            load_training_checkpoint(checkpoint, current, samples, None)
    else:
        assert load_training_checkpoint(checkpoint, current, samples, None)[2]["step"] == 1


@pytest.mark.parametrize("processes", [1, 2])
def test_detached_smolvla_fp32_storage_bf16_compute_and_exact_resume(
    recording, tiny_assets, tmp_path, monkeypatch, processes
):
    from pathlib import Path

    from test_smolvla import test_detached_native_smolvla_training_saves_resume_state

    from alohamini.learning import train as trainer

    config = Path(tiny_assets) / "config.json"
    backbone = json.loads(config.read_text())
    backbone["text_config"]["dtype"] = "bfloat16"
    config.write_text(json.dumps(backbone))
    launch = trainer.launch_training
    jobs = []

    def fp32_job(settings):
        if not settings.get("resume"):
            settings = {
                **settings,
                "trainable_dtype": "float32",
                "mixed_precision": "bfloat16",
                "num_processes": processes,
            }
        job = launch(settings)
        jobs.append(job)
        return job

    monkeypatch.setattr(trainer, "launch_training", fp32_job)
    # Reuse the native train/export/interrupt/resume comparison with a real BF16
    # expert configuration, FP32 trainable storage and BF16 autocast.
    test_detached_native_smolvla_training_saves_resume_state(
        recording, tiny_assets, tmp_path, monkeypatch
    )
    assert len(jobs) == 2
    checkpoint = Path(jobs[-1]["checkpoint"])
    manifest = json.loads((checkpoint / "policy.json").read_text())
    assert manifest["training"]["trainable_dtype"] == "float32"
    assert manifest["training"]["mixed_precision"] == "bfloat16"
    state = torch.load(checkpoint.resolve().parent / "training_state/state.pt", weights_only=True)
    assert all(
        slot["exp_avg"].dtype == slot["exp_avg_sq"].dtype == torch.float32
        for slot in state["optimizer"]["state"].values()
    )

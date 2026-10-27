"""Every registered algorithm shares original-v3 training and offline inference."""

import json

import numpy as np
import pytest
import torch
from test_diffusion_fastwam import diffusion_options, fake_frozen_assets, fastwam_options
from test_native_learning import model_options
from test_native_learning import recording as recording
from test_smolvla import options as smolvla_options
from test_smolvla import tiny_assets as tiny_assets

from alohamini.datasets.lerobotv3 import export_lerobot
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.evaluate import evaluate_checkpoint
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.learning.statistics import sample_arguments
from alohamini.policies.registry import ALGORITHMS, algorithm


def small_pi05(tmp_path, monkeypatch):
    from transformers import PaliGemmaConfig

    from alohamini.policies.pi05 import gemma_pytorch, modeling_pi05, tokenizer

    def vision_config():
        cfg = PaliGemmaConfig()
        cfg.vision_config.hidden_size = 32
        cfg.vision_config.num_hidden_layers = 1
        cfg.vision_config.num_attention_heads = 4
        cfg.vision_config.patch_size = 112
        return cfg

    monkeypatch.setattr(gemma_pytorch, "CONFIG_MAPPING", {"paligemma": vision_config})
    original = modeling_pi05.PI0Pytorch

    class SmallModel(original):
        def __init__(self, config):
            super().__init__(config)
            self.paligemma_with_expert.paligemma.model.multi_modal_projector.linear = (
                torch.nn.Linear(32, 64)
            )

    class Tokenizer:
        def __init__(self, *args):
            pass

        def tokenize(self, prompt, state):
            assert prompt and state.shape == (18,)
            return np.array([2, 3]), np.array([True, True])

    monkeypatch.setattr(modeling_pi05, "PI0Pytorch", SmallModel)
    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", Tokenizer)
    path = tmp_path / "test-tokenizer.model"
    path.write_bytes(b"test-only tokenizer; replaced in this test")
    return dict(
        network=dict(
            dtype="float32",
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_horizon=2,
        ),
        tokenizer_path=str(path),
        n_action_steps=1,
        num_inference_steps=2,
    )


@pytest.mark.parametrize("kind", list(ALGORITHMS))
def test_original_v3_forward_backward_checkpoint_and_offline_inference(
    recording, tmp_path, monkeypatch, request, kind
):
    root = tmp_path / "original-v3"
    export_lerobot(recording, root)
    (root / "meta/alohamini.json").unlink()
    for path in (root / "meta/safety").glob("*.jsonl"):
        path.unlink()
    recipe = algorithm(kind)
    if kind in ("act", "am_act"):
        options = model_options()
    elif kind in ("cdp", "dit_policy"):
        from test_cdp_dit_policy import small_options

        options = small_options(kind)
    elif kind == "multi_task_dit":
        from test_multi_task_dit import dit_options

        options = dit_options(tmp_path / "clip-assets")
    elif kind == "diffusion":
        options = diffusion_options()
    elif kind == "fastwam":
        fake_frozen_assets(monkeypatch)
        options = fastwam_options()
        options.update(
            {
                k: str(tmp_path)
                for k in ("vae_model_id", "text_encoder_model_id", "tokenizer_model_id")
            }
        )
    elif kind == "smolvla":
        options = smolvla_options(request.getfixturevalue("tiny_assets"))
        options.update(max_state_dim=32, max_action_dim=32)
    elif kind == "spirit":
        from test_spirit import tiny_spirit_options

        options = tiny_spirit_options(tmp_path / "spirit-assets")
    else:
        options = small_pi05(tmp_path, monkeypatch)
    cfg = dict(
        dataset=str(root),
        state="auto",
        allow_unannotated=True,
        image_size=[32, 32],
        train_episodes=[0],
        val_episodes=[1],
    )
    args, train, val = sample_arguments(cfg, recipe, options)
    assert args["state"] == "recorded" and train == [0] and val == [1]
    data = AlohaMiniDataset(**args, episodes=train)
    assert "robot_metadata" not in data.info
    assert data.records == [{}] * len(data.rows)
    options.update(input_features=data.input_features, output_features=data.output_features)
    stats = recipe.statistics(data, options)
    model = recipe.build(options, stats)
    processor = recipe.processor(model, stats, "cpu")
    batch = next(iter(torch.utils.data.DataLoader(data, batch_size=1)))
    prepared = processor(batch)
    loss, _ = model(prepared)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, stats, data, training={})
    saved = (checkpoint / "policy.json").read_bytes()
    loaded = NativePolicy(checkpoint, offline=True)
    predicted = loaded.predict(data[0])
    assert predicted.shape[-1] == 18 and torch.isfinite(predicted).all()
    with pytest.raises(ValueError, match="Offline policy"):
        loaded.select_action(None)
    with pytest.raises(ValueError, match="Offline policy"):
        loaded.bind_robot(None)
    result = evaluate_checkpoint(checkpoint, root, episodes=val, batch_size=2)
    assert len(result["mae_by_action"]) == 18
    assert result["valid_action_steps"] > 0
    assert saved == (checkpoint / "policy.json").read_bytes()
    assert not (root / "meta/alohamini.json").exists()


def test_recorded_state_rejects_bad_shape_and_unsupported_derived_units():
    from alohamini.learning.state import RecordedStateSelection

    info = {"features": {"observation.state": {"shape": [2], "names": ["a", "b"]}}}
    selection = RecordedStateSelection(info, "recorded")
    for value in ([1], [1, float("nan")]):
        with pytest.raises(ValueError, match="invalid shape|non-finite"):
            selection.frame({"observation.state": value})
    with pytest.raises(ValueError, match="derived velocity/current"):
        RecordedStateSelection(info, "joint_velocity")


def test_offline_cli_accepts_original_v3_flag(monkeypatch, capsys):
    from alohamini.learning import evaluate

    seen = []
    monkeypatch.setattr(evaluate, "evaluate_checkpoint", lambda *a, **kw: seen.append(kw) or {})
    assert (
        evaluate.main(
            [
                "--policy.path",
                "/checkpoint",
                "--dataset.root",
                "/data",
                "--dataset.episodes",
                "[0]",
                "--dataset.allow_unannotated",
                "true",
            ]
        )
        == 0
    )
    assert seen[0]["allow_unannotated"] is True
    assert json.loads(capsys.readouterr().out) == {}


def test_original_state_can_bind_to_declared_host_coordinates(recording, tmp_path):
    from copy import deepcopy

    from test_replay import replay_snapshot

    from alohamini.learning.train import offline_evaluate

    root = tmp_path / "v3"
    export_lerobot(recording, root)
    (root / "meta/alohamini.json").unlink()
    data = AlohaMiniDataset(
        root,
        episodes=[0],
        state="recorded",
        allow_unannotated=True,
        chunk_size=3,
        image_size=(32, 32),
    )
    recipe = algorithm("act")
    model = recipe.build(model_options())
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, recipe.statistics(data), data, training={})
    policy = NativePolicy(
        checkpoint, calibration_mode="legacy_normalized", robot_model="alohamini2pro"
    )
    snapshot = replay_snapshot()
    report = policy.bind_robot(snapshot)
    assert report["trained_metadata"] is None
    assert policy.selection.units == ["recorded (unit unverified)"] * 18
    assert torch.isfinite(policy.predict(data[0])).all()
    # Inference units bind explicitly; source metadata stays unknown.
    assert "robot_metadata" not in policy.source
    offline = NativePolicy(checkpoint, offline=True)
    data.info = deepcopy(data.info)
    data.info["features"]["action"]["names"].reverse()
    with pytest.raises(ValueError, match="contract"):
        offline_evaluate(offline, data)


def test_detached_state_training_and_resume_on_original_v3(recording, tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path

    from alohamini.learning.train import launch_training
    from alohamini.learning.train_config import parse_training_args

    root = tmp_path / "v3"
    export_lerobot(recording, root)
    (root / "meta/alohamini.json").unlink()
    for path in (root / "meta/safety").glob("*.jsonl"):
        path.unlink()
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    processes = []
    popen = subprocess.Popen

    def spawn(*args, **kwargs):
        assert kwargs["start_new_session"]
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", spawn)

    def run(cfg):
        job = launch_training(cfg)
        process = processes[-1]
        try:
            assert process.wait(timeout=120) == 0, Path(job["log"]).read_text()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        assert Path(job["pid_file"]).read_text().strip() == str(process.pid)
        return Path(job["checkpoint"])

    checkpoint = run(
        dict(
            dataset=str(root),
            allow_unannotated=True,
            policy="diffusion",
            state="recorded",
            model=diffusion_options(),
            image_size=[32, 32],
            device="cpu",
            batch_size=1,
            train_episodes=[0],
            val_episodes=[1],
            steps=1,
            eval_steps=1,
            save_freq=1,
            num_workers=0,
            scheduler={"type": "none"},
            output_dir=str(tmp_path / "training"),
        )
    )
    cfg, _ = parse_training_args(
        [
            "--config_path",
            str(checkpoint.resolve() / "train_config.json"),
            "--resume=true",
            "--steps=2",
        ]
    )
    checkpoint = run(cfg)
    manifest = json.loads((checkpoint / "policy.json").read_text())
    assert manifest["state"] == "recorded"
    assert manifest["source_info"]["training_only"]["calibration"] == "unavailable"
    assert evaluate_checkpoint(checkpoint, root, episodes=[1])["valid_action_steps"] > 0

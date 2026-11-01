"""Native detached training, checkpoint resume and distributed loss execution."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from test_gr00t_lewm import groot_assets as groot_assets
from test_gr00t_lewm import options
from test_native_learning import recording as recording

from alohamini.learning.train import launch_training
from alohamini.learning.train_config import parse_training_args


@pytest.mark.parametrize(
    "kind,processes", [("gr00t_n17", 1), ("lewm", 1), ("gr00t_n17", 2), ("lewm", 2)]
)
def test_detached_training_and_exact_resume(
    kind, processes, groot_assets, recording, tmp_path, monkeypatch
):
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    original = subprocess.Popen
    children = []

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"]
        process = original(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)

    def run(cfg):
        job = launch_training(cfg)
        child = children[-1]
        try:
            assert child.wait(timeout=150) == 0, Path(job["log"]).read_text()
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
        assert Path(job["pid_file"]).read_text().strip() == str(child.pid)
        return Path(job["checkpoint"])

    opts = options(kind, groot_assets, state=kind != "lewm")
    for key in (
        "input_features",
        "output_features",
        "state_names",
        "action_names",
        "state_slot_map",
        "action_slot_map",
    ):
        opts.pop(key, None)
    opts["gradient_checkpointing"] = True
    cfg = dict(
        policy=kind,
        dataset=str(recording),
        device="cpu",
        model=opts,
        state="none" if kind == "lewm" else "joint_position,base_velocity,lift_height",
        cameras=["forward"],
        image_size=[32, 32],
        mixed_precision="bfloat16",
        trainable_dtype="float32",
        num_workers=0,
        cpu_threads=1,
        batch_size=1,
        num_processes=processes,
        gradient_accumulation_steps=2,
        steps=2,
        save_freq=1,
        log_freq=1,
        eval_steps=1,
        train_episodes=[0],
        val_episodes=[1],
        optimizer={"lr": 1e-4},
        scheduler={"type": "none"},
        output_dir=str(tmp_path / "run"),
    )
    checkpoint = run(cfg)
    run_dir = tmp_path / "run"
    assert len((run_dir / "validation-metrics.jsonl").read_text().splitlines()) == 2
    assert (run_dir / "offline-evaluation.json").exists() == (kind == "gr00t_n17")
    final = checkpoint.resolve().parent
    reference = tmp_path / "reference"
    shutil.move(str(final), reference)
    previous = final.parent / "000001"
    resumed, _ = parse_training_args(
        ["--config_path", str(previous / "pretrained_model/train_config.json"), "--resume=true"]
    )
    restored = run(resumed)
    expected, actual = (
        load_file(reference / "pretrained_model/model.safetensors"),
        load_file(restored / "model.safetensors"),
    )
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)
    manifest = json.loads((restored / "policy.json").read_text())
    assert manifest["training"]["trainable_dtype"] == "float32"

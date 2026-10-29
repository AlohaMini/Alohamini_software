"""Detached native AC-DiT stage training and exact resume on RGB and XYZ inputs."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from test_acdit import siglip_assets as siglip_assets
from test_acdit import tiny_options
from test_native_learning import recording as recording

from alohamini.learning.train import launch_training
from alohamini.learning.train_config import parse_training_args


def pointcloud_dataset(recording, root):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from alohamini.datasets.lerobotv3 import export_lerobot

    export_lerobot(recording, root)
    (root / "meta/alohamini.json").unlink()
    for path in (root / "meta/safety").glob("*.jsonl"):
        path.unlink()
    key = "observation.point_cloud"
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"][key] = {
        "dtype": "float32",
        "shape": [32, 3],
        "coordinate_frame": "forward_camera",
        "units": "m",
    }
    info_path.write_text(json.dumps(info))
    for path in (root / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        cloud = np.random.default_rng(2).normal(size=(len(table), 32, 3)).astype("float32")
        table = table.append_column(
            key, pa.array(cloud.tolist(), type=pa.list_(pa.list_(pa.float32(), 3), 32))
        )
        pq.write_table(table, path)
    return root


@pytest.mark.parametrize("points,process_count", [(False, 1), (True, 1), (False, 2)])
def test_detached_two_stages_and_exact_resume(
    recording, siglip_assets, tmp_path, monkeypatch, points, process_count
):
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    original = subprocess.Popen
    processes = []

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"]
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)

    def run(cfg):
        job = launch_training(cfg)
        process = processes[-1]
        try:
            assert process.wait(timeout=60) == 0, Path(job["log"]).read_text()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        assert Path(job["pid_file"]).read_text().strip() == str(process.pid)
        return Path(job["checkpoint"])

    dataset = pointcloud_dataset(recording, tmp_path / "v3") if points else recording
    options = tiny_options(siglip_assets, points=points)
    for key in (
        "input_features",
        "output_features",
        "state_names",
        "action_names",
        "state_slot_map",
        "action_slot_map",
        "control_frequency",
    ):
        options.pop(key, None)
    cfg = dict(
        dataset=str(dataset),
        policy="acdit",
        device="cpu",
        model=options,
        allow_unannotated=points,
        steps=2,
        batch_size=1,
        num_workers=0,
        cpu_threads=1,
        mixed_precision="bfloat16",
        trainable_dtype="float32",
        image_size=[32, 32],
        save_freq=1,
        log_freq=1,
        eval_steps=1,
        train_episodes=[0],
        val_episodes=[1],
        output_dir=str(tmp_path / "mobility"),
        num_processes=process_count,
    )
    mobility = run(cfg)
    assert (tmp_path / "mobility/validation-metrics.jsonl").is_file()
    assert not (tmp_path / "mobility/offline-evaluation.json").exists()
    full, _ = parse_training_args(
        [
            f"--policy.path={mobility}",
            "--policy.stage=full",
            f"--dataset.root={dataset}",
            f"--output_dir={tmp_path / 'full'}",
            "--policy.device=cpu",
            "--steps=2",
            "--batch_size=1",
            "--num_workers=0",
            "--cpu_threads=1",
            "--save_freq=1",
            "--log_freq=1",
            "--eval_steps=1",
            "--dataset.episodes=[0]",
            "--dataset.eval_episodes=[1]",
            f"--num_processes={process_count}",
            f"--dataset.allow_unannotated={'true' if points else 'false'}",
        ]
    )
    checkpoint = run(full)
    final = checkpoint.resolve().parent
    reference = tmp_path / "reference"
    shutil.move(str(final), reference)
    previous = final.parent / "000001"
    resumed, _ = parse_training_args(
        ["--config_path", str(previous / "pretrained_model/train_config.json"), "--resume=true"]
    )
    assert run(resumed).resolve().parent == final
    expected, actual = (
        load_file(reference / "pretrained_model/model.safetensors"),
        load_file(final / "pretrained_model/model.safetensors"),
    )
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    manifest = json.loads((checkpoint / "policy.json").read_text())
    assert manifest["config"]["stage"] == "full"
    assert manifest["config"]["use_pointcloud"] is points
    assert manifest["image_resize_mode"] == "pad"
    assert manifest["training"]["trainable_dtype"] == "float32"
    assert (tmp_path / "full/offline-evaluation.json").is_file()
    state = torch.load(final / "training_state/state.pt", weights_only=True)
    assert state["optimizer"]["state"]
    assert all(s["exp_avg"].dtype == torch.float32 for s in state["optimizer"]["state"].values())

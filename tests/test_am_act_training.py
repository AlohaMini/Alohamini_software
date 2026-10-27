import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from test_am_act import conditional_options
from test_native_learning import recording as recording

from alohamini.learning.train import launch_training
from alohamini.learning.train_config import parse_training_args


@pytest.mark.parametrize("unannotated", [False, True])
def test_conditional_training_statistics_validation_and_exact_resume(
    recording, tmp_path, monkeypatch, unannotated
):
    if unannotated:
        from alohamini.datasets.lerobotv3 import export_lerobot

        original = tmp_path / "original-v3"
        export_lerobot(recording, original)
        (original / "meta/alohamini.json").unlink()
        for sidecar in (original / "meta/safety").glob("*.jsonl"):
            sidecar.unlink()
        recording = original
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    processes = []
    original = subprocess.Popen

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"]
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    config = dict(
        dataset=str(recording),
        allow_unannotated=unannotated,
        policy="am_act",
        state="none",
        device="cpu",
        train_episodes=[0],
        val_episodes=[1],
        batch_size=3,
        num_workers=0,
        save_freq=1,
        eval_steps=1,
        log_freq=1,
        image_size=[32, 32],
        seed=1000,
        cudnn_deterministic=True,
        deterministic_algorithms=True,
        model=conditional_options(
            kl_weight=1,
            latent_kl_warmup_steps=5000,
            discrete_action_mask_absent_classes=True,
            action_loss_groups={"arms": list(range(14)), "lift": [17]},
        ),
    )

    def run(settings):
        job = launch_training(settings)
        process = processes[-1]
        try:
            assert process.wait(timeout=120) == 0, Path(job["log"]).read_text()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        assert Path(job["pid_file"]).read_text().strip() == str(process.pid)
        return Path(job["checkpoint"])

    baseline = run({**config, "output_dir": str(tmp_path / "baseline"), "steps": 2})
    partial = run({**config, "output_dir": str(tmp_path / "resumed"), "steps": 1})
    cfg, _ = parse_training_args(
        [
            "--config_path",
            str(partial.resolve() / "train_config.json"),
            "--resume=true",
            "--steps=2",
        ]
    )
    resumed = run(cfg)
    expected, actual = (
        load_file(baseline / "model.safetensors"),
        load_file(resumed / "model.safetensors"),
    )
    assert actual["kl_updates"] == 2
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    manifest = json.loads((resumed / "policy.json").read_text())
    saved = manifest["config"]
    assert saved["discrete_action_dims"] == [14, 15, 16]
    assert saved["discrete_action_class_counts"] == [[0, 1, 3], [0, 1, 3], [0, 4, 0]]
    assert saved["discrete_action_class_weights"][2] == [5, 1, 5]
    assert saved["prior_type"] == "conditional_gaussian"
    assert bool(manifest["source_info"].get("training_only")) == unannotated
    assert (tmp_path / "resumed/offline-evaluation.json").is_file()
    if unannotated:
        from alohamini.learning.policy import NativePolicy

        with pytest.raises(ValueError, match="offline training only"):
            NativePolicy(resumed)
        # A new dataset can reactivate a previously unseen direction while the
        # three-output classifier tensors are loaded strictly from the checkpoint.
        import pyarrow as pa
        import pyarrow.parquet as pq

        extended = tmp_path / "with-backward"
        shutil.copytree(recording, extended)
        file = next((extended / "data").rglob("*.parquet"))
        table = pq.read_table(file)
        values = table.to_pylist()
        next(r for r in values if r["episode_index"] == 0 and r["frame_index"] == 1)["action"][
            14
        ] = -0.15
        pq.write_table(pa.Table.from_pylist(values, schema=table.schema), file)
        fine = run(
            {
                **config,
                "dataset": str(extended),
                "pretrained_path": str(resumed),
                "output_dir": str(tmp_path / "finetune"),
                "steps": 1,
            }
        )
        fine_manifest = json.loads((fine / "policy.json").read_text())
        assert saved["discrete_action_active_classes"][0] == [False, True, True]
        assert fine_manifest["config"]["discrete_action_active_classes"][0] == [True] * 3
        assert fine_manifest["config"]["discrete_action_class_counts"][0] == [1, 1, 2]
        assert fine_manifest["stats"] != manifest["stats"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "baseline/validation-metrics.jsonl").read_text().splitlines()
    ]
    assert "classification_dim_14_macro_recall" in rows[-1]["metrics"]

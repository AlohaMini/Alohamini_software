"""真实小模型训练、暂停及续训均通过现有断点接口完成。"""

import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from test_native_learning import model_options, recording

from alohamini.learning.desktop import TrainingInteraction, training_runs
from alohamini.learning.train import train
from alohamini.learning.train_config import parse_training_args


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_pause_saves_real_checkpoint_and_resume_lists_only_complete_runs(recording, tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("当前机器没有可用的 Apple MPS")
    result = subprocess.run([sys.executable, '-c',
        'from test_desktop_training import check_pause_resume;from pathlib import Path;import sys;'
        'check_pause_resume(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])',
        str(recording), str(tmp_path), device], capture_output=True, text=True, timeout=90,
        env={**os.environ, 'PYTHONPATH': os.pathsep.join([str(Path(__file__).parent), str(Path(__file__).parents[1] / 'src')])})
    assert result.returncode == 0, result.stdout + result.stderr


def check_pause_resume(recording, tmp_path, device):
    read_fd, write_fd = os.pipe()
    output = io.StringIO()
    with os.fdopen(read_fd) as source, os.fdopen(write_fd, "w"):
        interaction = TrainingInteraction(source, output)
        interaction.cancelled.set()
        settings = dict(dataset=str(recording), output_dir=str(tmp_path / "runs/pick"),
                        policy="act", device=device, num_workers=0, train_episodes=[0], val_episodes=[],
                        state="none", image_size=[32, 32], steps=2, batch_size=2, save_freq=50,
                        model=model_options(state=False))
        checkpoint = train(settings, interaction=interaction)
    assert interaction.progress["saved_step"] == 1
    assert json.loads((checkpoint.parent / "studio-progress.json").read_text())["status"] == "paused"
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert events[-1]["step"] == "paused"
    assert any(event["step"] == "saving_checkpoint" for event in events)
    runs = training_runs(tmp_path / "runs")
    assert len(runs) == 1 and runs[0]["resumable"] and runs[0]["saved_step"] == 1
    assert runs[0]["checkpoint"] == str(checkpoint.resolve())
    cfg, _ = parse_training_args(["--config_path=" + str(checkpoint / "train_config.json"),
                                 "--resume=true", "--policy.device=" + device, "--foreground"])
    resumed = io.StringIO()
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd) as source, os.fdopen(write_fd, "w"):
        interaction = TrainingInteraction(source, resumed)
        train(cfg, interaction=interaction)
    assert json.loads(resumed.getvalue().splitlines()[-1])["step"] == "succeeded"
    assert interaction.progress["saved_step"] == 2
    assert training_runs(tmp_path / "runs")[0]["resumable"] is False
    (checkpoint / "train_config.json").write_text("损坏的配置")
    assert training_runs(tmp_path / "runs") == []


def test_training_close_requests_checkpoint_and_failed_setup_has_no_resume(tmp_path):
    interaction = TrainingInteraction(io.StringIO(""), io.StringIO())
    assert interaction.cancelled.wait(1)
    interaction.complete("failed")
    assert training_runs(tmp_path) == []


def test_desktop_entry_uses_json_io_and_reports_failed_data_without_a_model(recording, tmp_path):
    settings = dict(dataset=str(recording), output_dir=str(tmp_path / "runs/desktop"),
                    policy="act", device="cpu", num_workers=0, train_episodes=[0], val_episodes=[],
                    state="none", image_size=[32, 32], steps=3, batch_size=2, save_freq=50,
                    model=model_options(state=False))
    config = tmp_path / "training.json"
    config.write_text(json.dumps(settings))
    command = [sys.executable, "-m", "alohamini.learning.desktop", "--config_path=" + str(config), "--foreground"]
    result = subprocess.run(command, input='{"action":"finish"}\n', capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["step"] == "paused" and report["training"]["saved_step"] == 1
    assert Path(report["training"]["checkpoint"], "train_config.json").is_file()
    config.write_text(json.dumps({**settings, "dataset": str(tmp_path / "missing"), "output_dir": str(tmp_path / "runs/failed")}))
    failed = subprocess.run(command, input="", capture_output=True, text=True, timeout=90)
    assert failed.returncode == 1
    assert json.loads(failed.stdout.splitlines()[-1])["step"] == "failed"
    assert not (tmp_path / "runs/failed/checkpoint").exists()

"""Real correction recording -> merge -> native checkpoint fine-tuning."""

import copy
import json
import subprocess
import time
from pathlib import Path
from unittest.mock import Mock

import torch
from safetensors.torch import load_file
from test_dagger import Leader
from test_dataset import frame, jpeg
from test_evaluation import EvaluationClient
from test_native_learning import model_options

from alohamini.apps.dagger import DAggerConfig, ManualHandover
from alohamini.apps.evaluation import run_evaluation
from alohamini.datasets.edit import edit_dataset, parse_args
from alohamini.datasets.record import LocalDataset, state_names
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import make_policy, save_checkpoint
from alohamini.learning.processor import act_statistics
from alohamini.learning.train import launch_training
from alohamini.learning.train_config import parse_training_args


def test_corrections_merge_and_detached_cpu_finetune(tmp_path, monkeypatch):
    class Clock:
        @property
        def now(self):
            return time.monotonic()

        sleep = staticmethod(time.sleep)

    client = EvaluationClient(Clock())
    client.state.payload["_robot_metadata"]["cameras"] = ["forward"]
    original = {"robot_metadata": copy.deepcopy(client.state.payload["_robot_metadata"])}
    recording = tmp_path / "demonstrations"
    with_dataset = LocalDataset(
        recording, fps=30, task="test", robot_metadata=original["robot_metadata"]
    )
    try:
        for episode in range(2):
            with_dataset.begin_episode()
            for index in range(4):
                value = frame(with_dataset)
                value["action"][:] = index
                stamp = 100 + episode + index / 30
                assert with_dataset.add_frame(
                    value,
                    {"forward": jpeg()},
                    {
                        "safety": copy.deepcopy(client.state.payload["_safety"]),
                        "requested_action": dict.fromkeys(with_dataset.names, float(index)),
                        "host_timing": {
                            "state_sample_monotonic_s": stamp,
                            "camera_capture_monotonic_s": {"forward": stamp},
                        },
                    },
                )
            with_dataset.save_episode()
    finally:
        with_dataset.close()
    client.state.images = {"forward": jpeg()}

    def cameras(_client):
        stamp = time.monotonic() + 100  # Host clock is independent of client clock.
        client.state.payload["_host_timing"] = {
            "state_sample_started_monotonic_s": stamp - 0.001,
            "state_sample_finished_monotonic_s": stamp,
            "camera_capture_monotonic_s": {"forward": stamp},
        }

    client.on_read = cameras
    policy = Mock()
    policy.robot_metadata = copy.deepcopy(original["robot_metadata"])
    policy.select_action.side_effect = lambda snapshot: {
        name: snapshot.payload[name] for name in state_names("alohamini2pro")
    }
    leader = Leader(client.state)
    corrections = tmp_path / "corrections"
    dataset = LocalDataset(corrections, fps=30, task="test", robot_metadata=policy.robot_metadata)

    class Script:
        stage = 0

        def events(self):
            if self.stage == 0:
                self.stage = 1
                return ["space"]
            if control.phase == "paused" and self.stage in (1, 4) and not control.save_pending:
                self.stage += 1
                return ["tab"]
            if control.phase == "ready" and self.stage in (2, 5):
                self.stage += 1
                return ["tab"]
            if control.phase == "human" and self.stage in (3, 6) and dataset.submitted >= 4:
                self.stage += 1
                return ["tab"]
            if control.phase == "paused" and self.stage == 7 and not control.save_pending:
                self.stage = 8
                return ["space"]
            if self.stage == 8 and policy.select_action.call_count:
                return ["quit"]
            return []

    control = ManualHandover(leader, Script(), DAggerConfig(settle_time_s=0.01))
    monkeypatch.setattr("alohamini.apps.evaluation.stop_owned_robot", Mock())
    try:
        run_evaluation(
            client, policy, "alohamini2pro", dataset=dataset, intervention=control, duration_s=15
        )
    finally:
        dataset.close()
    assert control.stopped and policy.select_action.call_count == 1
    info = json.loads((corrections / "meta/info.json").read_text())
    assert info["total_episodes"] == 2 and info["total_frames"] == 8
    for path in sorted((corrections / "meta/safety").glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        frames = [row for row in rows if row.get("frame_index") is not None]
        assert len(frames) == 4
        assert all(row["intervention"] and row["control_source"] == "human" for row in frames)
        assert all(row["action_semantics"] == "host_request_v1" for row in frames)
    merged = tmp_path / "merged"
    edit_dataset(
        parse_args(
            [
                "--operation.type=merge",
                "--operation.roots",
                json.dumps([str(recording), str(corrections)]),
                "--output",
                str(merged),
            ]
        )
    )
    samples = AlohaMiniDataset(
        merged, episodes=[0, 1, 2, 3], state="none", chunk_size=3, image_size=(32, 32)
    )
    assert len(samples) == 16
    assert [samples.segment_ends[i] for i in (0, 4, 8, 12)] == [4, 8, 12, 16]
    source_samples = AlohaMiniDataset(
        recording, episodes=[0], state="none", chunk_size=3, image_size=(32, 32)
    )
    source = make_policy("act", model_options(state=False))
    with torch.no_grad():
        source.model.action_head.bias.fill_(0.321)
    checkpoint = tmp_path / "base"
    save_checkpoint(
        checkpoint, source, act_statistics(source_samples), source_samples, training={"step": 23}
    )
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    cfg, _ = parse_training_args(
        [
            f"--policy.path={checkpoint}",
            f"--dataset.root={merged}",
            f"--output_dir={tmp_path / 'finetune'}",
            "--policy.device=cpu",
            "--num_workers=0",
            "--steps=1",
            "--batch_size=2",
            "--save_freq=1",
            "--log_freq=1",
            "--cpu_threads=1",
            "--optimizer.lr=0",
            "--policy.optimizer_lr_backbone=0",
        ]
    )
    processes = []
    popen = subprocess.Popen

    def detached(*args, **kwargs):
        assert kwargs["start_new_session"]
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", detached)
    job = launch_training(cfg)
    process = processes[-1]
    try:
        assert process.wait(timeout=60) == 0, Path(job["log"]).read_text()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
    assert Path(job["pid_file"]).read_text().strip() == str(process.pid)
    saved = load_file(Path(job["checkpoint"]).resolve() / "model.safetensors")
    for key, value in source.named_parameters():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)

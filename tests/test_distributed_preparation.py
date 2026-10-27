"""Real process groups: shared startup results, invalidation and failure delivery."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_native_learning import recording as recording

from alohamini.datasets.lerobotv3 import export_lerobot
from alohamini.learning.data import AlohaMiniDataset, DatasetInspection
from alohamini.learning.execution import Execution
from alohamini.learning.statistics import training_statistics, write_statistics
from alohamini.policies.registry import algorithm


@pytest.mark.parametrize("storage", ["native", "v3"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_distributed_dataset_preparation(recording, tmp_path, storage, device):
    if device == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("NCCL verification requires two visible GPUs")
    if storage == "v3":
        export_lerobot(recording, tmp_path / "v3")
        recording = tmp_path / "v3"
    log, pid = tmp_path / "preparation.log", tmp_path / "preparation.pid"
    with log.open("wb") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(Path(__file__).resolve()),
                str(recording),
                str(tmp_path),
                device,
            ],
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid.write_text(f"{process.pid}\n")
    try:
        assert process.wait(timeout=90) == 0, log.read_text()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
    assert json.loads((tmp_path / "counts.json").read_text()) == [
        {"inspection": 1, "statistics": 1},
        {"inspection": 0, "statistics": 0},
    ]


def distributed_check(root, output, device):
    import alohamini.learning.data as data_module

    runtime = Execution(device)
    counts = {"inspection": 0, "statistics": 0}
    check = data_module.check_dataset
    components = algorithm("act")
    fit = components.statistics

    def tracked_check(*args, **kwargs):
        counts["inspection"] += 1
        return check(*args, **kwargs)

    def tracked_fit(*args, **kwargs):
        counts["statistics"] += 1
        return fit(*args, **kwargs)

    data_module.check_dataset = tracked_check
    components.statistics = tracked_fit
    try:

        def inspect():
            inspection = DatasetInspection()
            inspection.check(root)
            return inspection

        inspection = runtime.main_process_call(inspect, label="Inspection", timeout_s=30)
        views = [
            AlohaMiniDataset(
                root,
                episodes=[episode],
                chunk_size=3,
                image_size=(32, 32),
                state="none",
                inspection=inspection,
            )
            for episode in (0, 1)
        ]
        options = dict(
            input_features=views[0].input_features, output_features=views[0].output_features
        )
        artifact = runtime.main_process_call(
            lambda: training_statistics(components, views[0], options, "act"),
            label="Statistics",
        )
        assert (
            training_statistics(components, views[0], options, "act", artifact=artifact) == artifact
        )
        assert artifact["contract"]["episodes"] == [0]
        with pytest.raises(ValueError, match="do not match.*episodes"):
            training_statistics(components, views[1], options, "act", artifact=artifact)
        with pytest.raises(ValueError, match="do not match.*source_sha256"):
            hashes = views[0].table_sha256
            views[0].table_sha256 = {**hashes, "changed": "changed"}
            try:
                training_statistics(components, views[0], options, "act", artifact=artifact)
            finally:
                views[0].table_sha256 = hashes

        def from_file():
            path = write_statistics(output / "statistics.json", artifact)
            return training_statistics(components, views[0], options, "act", path)

        assert runtime.main_process_call(from_file, label="Saved statistics") == artifact
        restored = runtime.main_process_call(
            lambda: training_statistics(components, views[0], options, "act", checkpoint=artifact),
            label="Checkpoint statistics",
        )
        assert restored["stats"] == artifact["stats"]
        totals = runtime.gather(counts)
        if runtime.main:
            (output / "counts.json").write_text(json.dumps(totals))

        def fail():
            raise ValueError("injected statistics failure")

        with pytest.raises((ValueError, RuntimeError), match="injected statistics failure"):
            runtime.main_process_call(fail, label="Statistics")
        assert runtime.main_process_call(lambda: "recovered", label="Next call") == "recovered"

        # The shared report cannot suppress detection of later corruption.
        runtime.main_process_call(
            lambda: (root / "meta/info.json").write_text("{}"), label="Mutate fixture"
        )
        with pytest.raises(ValueError, match="integrity check failed"):
            AlohaMiniDataset(root, episodes=[0], inspection=inspection)
        assert counts["inspection"] == (2 if runtime.main else 1)
        runtime.barrier()
    finally:
        data_module.check_dataset = check
        components.statistics = fit
        runtime.close()


if __name__ == "__main__":
    distributed_check(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])

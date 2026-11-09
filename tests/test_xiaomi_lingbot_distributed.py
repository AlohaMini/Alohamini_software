"""Real reduced XR1/LingBot dual-GPU update, export and optimizer restore."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch


def worker(root, kind):
    from test_xiaomi_lingbot import model_batch, options

    from alohamini.learning.execution import Execution
    from alohamini.policies.registry import algorithm

    torch.set_num_threads(2)
    execution = Execution("cuda", "bfloat16", "fsdp2")
    completed = False
    try:
        torch.manual_seed(42)
        component = algorithm(kind)
        cfg = options(kind, root / f"assets-{execution.rank}")
        cfg["gradient_checkpointing"] = True
        model = component.build(cfg)
        optimizer = torch.optim.AdamW(model.get_optim_params(), lr=1e-4)
        model, optimizer = execution.prepare(model, optimizer)
        raw = model_batch(kind)

        def prepare(batch):
            def move(value):
                if isinstance(value, dict):
                    return {k: move(v) for k, v in value.items()}
                return value.to(execution.device) if torch.is_tensor(value) else value

            result = move(batch)
            if kind == "xiaomi_robotics1":
                result.pop("action_is_pad")
            return result

        for _ in range(2):
            result = execution.update(
                model, [raw], optimizer, 1.0, prepare, reduction=component.loss_counts
            )
            assert result["optimizer_step"] and result["grad_norm"] > 0
        execution.distributed_checkpoint(root / "state", model, optimizer)
        saved = {n: p.to_local().clone() for n, p in model.named_parameters()}
        with torch.no_grad():
            next(model.parameters()).add_(1)
        execution.distributed_checkpoint(root / "state", model, optimizer, load=True)
        for n, p in model.named_parameters():
            torch.testing.assert_close(p.to_local(), saved[n], rtol=0, atol=0)
        model.eval()
        with torch.no_grad(), execution.autocast():
            action = model.predict_action_chunk(prepare(raw))
        assert action.isfinite().all()
        state = execution.accelerator.get_state_dict(model)
        if execution.main:
            portable = root / "portable"
            portable.mkdir()
            manifest = {"config": asdict(model.config)}
            component.save(portable, model, manifest, state)
            restored = component.build(
                component.checkpoint_options(
                    portable, manifest["config"], "cpu", None, "checkpoint"
                )
            )
            component.load(portable, restored)
            for n, v in restored.state_dict().items():
                torch.testing.assert_close(v, state[n], rtol=0, atol=0)
            (root / "result.json").write_text(json.dumps(result))
        execution.barrier()
        completed = True
    finally:
        if completed:
            execution.close()


@pytest.mark.parametrize("kind", ["xiaomi_robotics1", "lingbot_vla_v2"])
def test_two_gpu_update_export_reload(tmp_path, kind):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two idle CUDA GPUs")
    with (tmp_path / "worker.log").open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                str(Path(__file__).resolve()),
                str(tmp_path),
                kind,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "2"},
        )
        (tmp_path / "worker.pid").write_text(str(process.pid))
        try:
            result = process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
            raise
    assert result == 0, (tmp_path / "worker.log").read_text()


if __name__ == "__main__":
    worker(Path(sys.argv[1]), sys.argv[2])

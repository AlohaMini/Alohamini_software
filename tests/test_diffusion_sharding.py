"""Use the common FSDP2 update/export/resume checks for the diffusion branch."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import test_policy_sharding as shared
import torch


def build(kind, root, patch, **options):
    from test_native_learning import recording

    data_root = root / "data"
    data_root.mkdir()
    data = recording.__wrapped__(data_root)
    if kind == "multi_task_dit":
        from test_multi_task_dit import data_and_model, dit_options

        _, _, model, _, batch = data_and_model(data, dit_options(root))
    else:
        from test_cdp_dit_policy import data_and_model

        _, _, model, _, batch = data_and_model(data, kind)
    return model, batch


@pytest.mark.parametrize("kind", ["cdp", "dit_policy", "multi_task_dit"])
def test_cuda_diffusion_sharding(tmp_path, kind):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
            str(tmp_path),
            f"{kind}:flow:full:float32",
        ],
        env=dict(
            os.environ,
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            HF_HUB_OFFLINE="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout
    assert json.loads((tmp_path / "result.json").read_text())["passed"]


if __name__ == "__main__":
    shared._model = build
    shared._worker(Path(sys.argv[1]), sys.argv[2])

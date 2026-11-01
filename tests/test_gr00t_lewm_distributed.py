"""Opt-in CUDA FSDP2 checks, using real reduced models and local assets only."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from test_gr00t_lewm import groot_assets as groot_assets
from test_gr00t_lewm import options, raw_batch, stats


def worker(root, kind, assets, tune):
    from alohamini.learning.execution import Execution
    from alohamini.learning.policy import make_policy
    from alohamini.policies.registry import algorithm

    torch.set_num_threads(2)
    runtime = Execution("cuda", "bfloat16", "fsdp2")
    cfg = options(kind, assets, state=kind != "lewm")
    cfg["gradient_checkpointing"] = True
    if kind == "gr00t_n17":
        cfg.update(
            backbone_dtype="bfloat16", tune_llm=tune in ("llm", "full"), tune_visual=tune == "full"
        )
    model = make_policy(kind, cfg, trainable_dtype="float32")
    component = algorithm(kind)
    component.options({"model": cfg}, "cuda")
    processor = component.processor(model, stats(model), str(runtime.device))
    optimizer = torch.optim.AdamW(model.get_optim_params(), lr=1e-4)
    wrapped, optimizer = runtime.prepare(model, optimizer)
    batch = raw_batch(kind)
    for _ in range(2):
        result = runtime.update(
            wrapped, [batch], optimizer, 1.0, processor, reduction=component.loss_counts
        )
        assert result["optimizer_step"] and np.isfinite(result["loss"])
    runtime.distributed_checkpoint(root / "state", wrapped, optimizer)
    saved = {n: p.to_local().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        next(model.parameters()).add_(1)
    runtime.distributed_checkpoint(root / "state", wrapped, optimizer, load=True)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.to_local(), saved[name], rtol=0, atol=0)
    model.eval()
    if kind == "lewm":
        prepared = processor.prepare_goal({"forward": np.zeros((32, 32, 3), dtype=np.uint8)})
        model.set_goal({k: prepared[k] for k in model.config.image_features})
    with torch.no_grad(), runtime.autocast():
        action = wrapped.predict_action_chunk(processor(batch))
    assert action.shape == (2, model.config.chunk_size, 18) and action.isfinite().all()
    state = runtime.accelerator.get_state_dict(wrapped)
    if runtime.main:
        path = root / "portable"
        path.mkdir()
        manifest = {"config": asdict(model.config)}
        component.save(path, model, manifest, state)
        restored = make_policy(
            kind,
            component.checkpoint_options(path, manifest["config"], "cpu", None, "checkpoint"),
            trainable_dtype="float32",
        )
        component.load(path, restored)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        (root / "result.json").write_text(json.dumps(result))
    runtime.close()


@pytest.mark.parametrize(
    "kind,tune",
    [("gr00t_n17", "frozen"), ("gr00t_n17", "llm"), ("gr00t_n17", "full"), ("lewm", "full")],
)
def test_cuda_fsdp_update_export_and_reload(tmp_path, groot_assets, kind, tune):
    if os.environ.get("ALOHAMINI_TEST_GR00T_LEWM_FSDP") != "1" or not torch.cuda.is_available():
        pytest.skip("Opt in with ALOHAMINI_TEST_GR00T_LEWM_FSDP=1 on an idle CUDA device")
    count = int(os.environ.get("ALOHAMINI_TEST_FSDP_RANKS", "1"))
    if torch.cuda.device_count() < count:
        pytest.skip("Not enough visible CUDA devices")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={count}",
            str(Path(__file__).resolve()),
            str(tmp_path),
            kind,
            str(groot_assets),
            tune,
        ],
        env={
            **os.environ,
            "OMP_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=150,
    )
    assert result.returncode == 0, result.stdout
    assert (tmp_path / "result.json").is_file()


if __name__ == "__main__":
    worker(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4])

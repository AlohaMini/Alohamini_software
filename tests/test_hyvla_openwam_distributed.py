"""DDP/FSDP2 update, EMA, export and resume using reduced real policies."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from test_hyvla_openwam import raw_batch, tiny_options, tokenizer_assets


def worker(root, kind, backend):
    from alohamini.learning.ema import EMAManager
    from alohamini.learning.execution import Execution
    from alohamini.policies.registry import algorithm

    torch.set_num_threads(2)
    runtime = Execution("cuda" if backend == "fsdp2" else "cpu", "bfloat16", backend)
    local = root / f"rank-{runtime.rank}"
    local.mkdir(parents=True, exist_ok=True)
    tokenizer_assets(local)
    torch.manual_seed(100)
    component = algorithm(kind)
    model = component.build(tiny_options(kind, local))
    optimizer = torch.optim.AdamW(model.get_optim_params(), lr=1e-4)
    wrapped, optimizer = runtime.prepare(model, optimizer)
    ema = EMAManager(model, {"enable": True, "decay": 0.9})
    batch = {
        key: value.to(runtime.device) if isinstance(value, torch.Tensor) else value
        for key, value in raw_batch(kind, model.config).items()
    }
    for _ in range(2):
        metrics = runtime.update(
            wrapped, [batch], optimizer, 1, lambda x: x, reduction=component.loss_counts, ema=ema
        )
        assert torch.isfinite(torch.tensor(metrics["loss"]))
    if runtime.sharded:
        runtime.distributed_checkpoint(root / "state", wrapped, optimizer, ema=ema)
        saved = {name: p.to_local().clone() for name, p in model.named_parameters()}
        with torch.no_grad():
            next(model.parameters()).add_(1)
        runtime.distributed_checkpoint(root / "state", wrapped, optimizer, load=True, ema=ema)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.to_local(), saved[name], rtol=0, atol=0)
    inference = {key: value for key, value in batch.items() if not key.startswith("action")}
    if kind == "openwam":
        inference["observation.images.head"] = batch["observation.images.head"][:, 0]
    model.eval()
    with ema.average_parameters(model), torch.no_grad(), runtime.autocast():
        state = runtime.accelerator.get_state_dict(wrapped)
        predict = wrapped if runtime.sharded else model
        action = predict.predict_action_chunk(inference)
        assert action.isfinite().all()
        if runtime.main:
            target = root / "portable"
            target.mkdir()
            manifest = {"config": asdict(model.config)}
            component.save(target, model, manifest, state)
            restored = component.build(
                component.checkpoint_options(target, manifest["config"], "cpu", None, "checkpoint")
            )
            component.load(target, restored)
            for name, tensor in restored.state_dict().items():
                torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
    runtime.barrier()
    if runtime.main:
        (root / "result.json").write_text(
            json.dumps(
                {
                    "policy": kind,
                    "backend": backend,
                    "world_size": runtime.world_size,
                    "loss": metrics["loss"],
                    "passed": True,
                }
            )
        )
    runtime.close()


if __name__ == "__main__":
    worker(Path(sys.argv[1]), sys.argv[2], sys.argv[3])

"""FSDP2 on reduced real policies; GPU workers use local assets only."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from alohamini.learning.sharding import sharding_units


def test_sharding_units_preserve_functional_attention_and_shared_weights():
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = torch.nn.MultiheadAttention(8, 2, batch_first=True)
            self.embed = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16, bias=False)
            self.head.weight = self.embed.weight

    model = Network()
    units = sharding_units(model)
    assert units == [model.attention]
    assert model.attention.out_proj not in units


def _device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _device(v, device) for k, v in value.items()}
    if isinstance(value, SimpleNamespace):
        return SimpleNamespace(**_device(vars(value), device))
    return value


def _model(kind, root, patch, *, recipe="flow", mode="full", dtype="float32"):
    from alohamini.policies.registry import algorithm

    if kind == "pi05":
        from test_pi05_ki import batch, tiny_model

        factory = tiny_model.__wrapped__(patch)
        tokenizer = root / "tokenizer.model"
        tokenizer.write_text("unused test asset")
        fast = root / "fast"
        fast.mkdir(exist_ok=True)
        for name in ("tokenizer.json", "processor_config.json"):
            (fast / name).write_text("{}")
        model = factory(
            training_recipe=recipe,
            finetuning_mode=mode,
            num_inference_steps=2,
            tokenizer_path=str(tokenizer),
            fast_tokenizer_path=str(fast),
        )
        if dtype == "bfloat16":
            model.network.paligemma_with_expert.to_bfloat16_for_selected_params(dtype)
            # LoRA adapters and action projections retain their intended FP32 dtype.
            for name, param in model.named_parameters():
                if name.endswith((".lora_A", ".lora_B")):
                    param.data = param.data.float()
        return model, batch()
    if kind == "smolvla":
        from test_smolvla import batch, options, tiny_assets

        assets = root / "smol-assets"
        assets.mkdir(exist_ok=True)
        return algorithm(kind).build(options(tiny_assets.__wrapped__(assets))), batch()
    if kind in ("act", "am_act"):
        from test_native_learning import batch, model_options

        return algorithm(kind).build(model_options()), batch()
    from test_diffusion_fastwam import diffusion_options, fake_frozen_assets, fastwam_options

    if kind == "diffusion":
        return algorithm(kind).build(diffusion_options()), {
            "observation.images.forward": torch.rand(2, 2, 3, 32, 32),
            "observation.state": torch.rand(2, 2, 18),
            "action": torch.rand(2, 8, 18),
            "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
        }
    if kind == "fastwam":
        fake_frozen_assets(patch)
        return algorithm(kind).build(fastwam_options()), {
            "video": torch.rand(2, 3, 5, 32, 64),
            "context": torch.rand(2, 4, 16),
            "context_mask": torch.ones(2, 4, dtype=torch.bool),
            "observation.state": torch.rand(2, 18),
            "action": torch.rand(2, 4, 18),
            "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
        }
    raise AssertionError(kind)


def _worker(root, case):
    from alohamini.learning.ema import EMAManager
    from alohamini.learning.execution import Execution
    from alohamini.policies.registry import algorithm

    kind, recipe, mode, dtype = case.split(":")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    execution = Execution("cuda", backend="fsdp2")
    local = root / f"rank-{execution.rank}"
    local.mkdir(parents=True, exist_ok=True)
    succeeded = False

    def export_and_load(path, state, reference):
        path.mkdir()
        manifest = {
            "format": "alohamini-policy",
            "version": 1,
            "kind": kind,
            "config": asdict(model.config),
        }
        algorithm(kind).save(path, model, manifest, model_state=state)
        (path / "policy.json").write_text(json.dumps(manifest))
        algorithm(kind).load(path, reference)

    try:
        with pytest.MonkeyPatch.context() as patch:
            torch.manual_seed(104)
            model, batch = _model(kind, local, patch, recipe=recipe, mode=mode, dtype=dtype)
            reference = copy.deepcopy(model).to(execution.device)
            optimizer = torch.optim.AdamW(model.get_optim_params(), lr=1e-4)
            ref_optimizer = torch.optim.AdamW(reference.get_optim_params(), lr=1e-4)
            wrapped, optimizer = execution.prepare(model, optimizer)
            assert any(hasattr(p, "placements") for p in model.parameters())
            ema = EMAManager(model, {"enable": True, "decay": 0.9})
            ref_ema = EMAManager(reference, {"enable": True, "decay": 0.9})
            batch = _device(batch, execution.device)
            torch.manual_seed(208)
            ref_loss, _ = reference(batch)
            ref_loss.backward()
            ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1)
            ref_optimizer.step()
            ref_ema.update(reference)
            torch.manual_seed(208)
            metrics = execution.update(
                wrapped,
                [batch],
                optimizer,
                1,
                lambda x: x,
                reduction=algorithm(kind).loss_counts,
                ema=ema,
            )
            assert metrics["loss"] == pytest.approx(ref_loss.item(), rel=2e-3, abs=2e-4)
            assert metrics["grad_norm"] == pytest.approx(
                ref_norm.item(),
                rel=0.02 if dtype == "bfloat16" else 2e-3,
            )
            assert all(values["step"].item() == 1 for values in optimizer.state.values())
            tolerance = (
                {"rtol": 0.03, "atol": 3e-4}
                if dtype == "bfloat16"
                else {"rtol": 1e-4, "atol": 1e-6}
            )
            state = execution.accelerator.get_state_dict(wrapped)
            if execution.main:
                for name, expected in reference.state_dict().items():
                    torch.testing.assert_close(state[name], expected.cpu(), **tolerance)
                export_and_load(local / "export", state, reference)
                for name, loaded in reference.state_dict().items():
                    torch.testing.assert_close(loaded.cpu(), state[name], rtol=0, atol=0)
            execution.barrier()
            inference = dict(batch)
            if kind == "fastwam":
                inference["input_image"] = batch["video"][:, :, 0]
            # Exercise registered non-forward methods and invalidate gathered EMA
            # weights before restoring online shards (including the retained root).
            with ema.average_parameters(model), torch.no_grad():
                ema_state = execution.accelerator.get_state_dict(wrapped)
                if execution.main:
                    with ref_ema.average_parameters(reference):
                        for name, expected in reference.state_dict().items():
                            torch.testing.assert_close(
                                ema_state[name],
                                expected.cpu(),
                                rtol=0.03,
                                atol=3e-4,
                            )
                    export_and_load(local / "export-ema", ema_state, reference)
                torch.manual_seed(310)
                predicted = wrapped.predict_action_chunk(inference)
                assert predicted.isfinite().all()
                if execution.main:
                    reference.eval()
                    torch.manual_seed(310)
                    expected = reference.predict_action_chunk(inference)
                    torch.testing.assert_close(predicted, expected, rtol=0.03, atol=3e-4)
            restored = execution.accelerator.get_state_dict(wrapped)
            if execution.main:
                for name in state:
                    torch.testing.assert_close(restored[name], state[name], rtol=0, atol=0)
            execution.distributed_checkpoint(root / "distributed", wrapped, optimizer, ema=ema)
            before = {
                name: value.detach().to_local().clone() for name, value in model.named_parameters()
            }
            ema_before = {name: p.to_local().clone() for name, p in ema.parameters.items()}
            optimizer_before = copy.deepcopy(optimizer.state_dict())
            torch.manual_seed(412)
            execution.update(
                wrapped,
                [batch],
                optimizer,
                1,
                lambda x: x,
                reduction=algorithm(kind).loss_counts,
                ema=ema,
            )
            continued = execution.accelerator.get_state_dict(wrapped)
            ema_continued = {name: p.to_local().clone() for name, p in ema.parameters.items()}
            execution.distributed_checkpoint(
                root / "distributed",
                wrapped,
                optimizer,
                load=True,
                ema=ema,
            )
            assert ema.num_updates == 1
            for name, value in model.named_parameters():
                torch.testing.assert_close(
                    value.to_local(),
                    before[name],
                    rtol=0,
                    atol=0,
                    msg=name,
                )
            for name, expected in ema_before.items():
                torch.testing.assert_close(
                    ema.parameters[name].to_local(),
                    expected,
                    rtol=0,
                    atol=0,
                    msg="EMA " + name,
                )
            for key, values in optimizer.state_dict()["state"].items():
                for name, value in values.items():
                    expected = optimizer_before["state"][key][name]
                    if hasattr(value, "to_local"):
                        value, expected = value.to_local(), expected.to_local()
                    torch.testing.assert_close(
                        value,
                        expected,
                        rtol=0,
                        atol=0,
                        msg=lambda detail, key=key, name=name: f"optimizer {key}.{name}: {detail}",
                    )
            torch.manual_seed(412)
            execution.update(
                wrapped,
                [batch],
                optimizer,
                1,
                lambda x: x,
                reduction=algorithm(kind).loss_counts,
                ema=ema,
            )
            resumed = execution.accelerator.get_state_dict(wrapped)
            assert ema.num_updates == 2
            for name, expected in ema_continued.items():
                torch.testing.assert_close(
                    ema.parameters[name].to_local(),
                    expected,
                    rtol=0,
                    atol=0,
                )
            if execution.main:
                for name in continued:
                    torch.testing.assert_close(resumed[name], continued[name], rtol=0, atol=0)
            if kind == "pi05" and recipe == "ki" and mode == "full":

                def gradient(module):
                    values = [
                        p.grad.to_local().float().abs().sum()
                        for p in module.parameters()
                        if p.grad is not None
                    ]
                    return sum(values).item() if values else 0

                combined = model.network.paligemma_with_expert
                for flow_only in (True, False):
                    model.zero_grad(set_to_none=True)
                    model.config.flow_loss_weight = float(flow_only)
                    model.config.fast_loss_weight = float(not flow_only)
                    loss, _ = wrapped(batch)
                    execution.accelerator.backward(loss)
                    assert (gradient(combined.paligemma) == 0) == flow_only
                    assert (gradient(combined.gemma_expert) > 0) == flow_only
                model.config.flow_loss_weight = model.config.fast_loss_weight = 1.0
                model.zero_grad(set_to_none=True)
            if execution.main:
                (root / "result.json").write_text(
                    json.dumps({"case": case, "passed": True, **metrics})
                )
            execution.barrier()
            succeeded = True
    finally:
        # A failed rank must exit immediately so torchrun can stop its peers;
        # Accelerate.end_training() includes a collective barrier.
        if succeeded:
            execution.close()


@pytest.mark.parametrize(
    "case",
    [
        "act:flow:full:float32",
        "am_act:flow:full:float32",
        "diffusion:flow:full:float32",
        "fastwam:flow:full:float32",
        "smolvla:flow:full:float32",
        "pi05:flow:full:float32",
        "pi05:ki:a:float32",
        "pi05:ki:c:float32",
        "pi05:ki:d:float32",
        "pi05:flow:b:float32",
        "pi05:ki:a:bfloat16",
        "pi05:ki:full:bfloat16",
    ],
)
def test_cuda_fsdp_policy_update(tmp_path, case):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    env = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        HF_HUB_OFFLINE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
            str(tmp_path),
            case,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout
    assert json.loads((tmp_path / "result.json").read_text())["passed"]


if __name__ == "__main__":
    _worker(Path(sys.argv[1]), sys.argv[2])

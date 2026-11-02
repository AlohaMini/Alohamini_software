"""Exact validation coverage and reductions, including collective tail forwards."""

import math
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from alohamini.learning.loading import make_loader
from alohamini.learning.metrics import MetricAccumulator, MetricSpec, loss_count
from alohamini.learning.train import _offline_evaluate_chunks
from alohamini.learning.validation import EvaluationBatchSampler, evaluate_loss, loss_denominators


class Targets:
    delta_indices = {"action": [0, 1, 2]}

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            "action": torch.full((3, 1), float(index + 1)),
            "action_is_pad": torch.arange(3) >= 3 - index % 3,
        }

    action_metadata = __getitem__


def counts(batch):
    return {
        "_valid_weight": int((~batch["action_is_pad"]).sum()),
        "_sample_weight": len(batch["action"]),
    }


SPECS = (
    MetricSpec("reconstruction", "reconstruction", "valid_steps", loss_count("_valid_weight")),
    MetricSpec("sample_mean", "sample_mean", "samples", loss_count("_sample_weight")),
)


class Policy(torch.nn.Module):
    def __init__(self, *, collective=False):
        super().__init__()
        self.projection = torch.nn.Linear(1, 4, bias=False)
        torch.nn.init.ones_(self.projection.weight)
        self.collective = collective
        self.nonfinite = False
        self.calls = 0

    def forward(self, batch):
        if self.collective:
            # Exercise the equal-forward requirement without needing multiple GPUs.
            dist.all_reduce(torch.zeros((), device=batch["action"].device))
        self.calls += 1
        value = self.projection(batch["action"]).mean(-1)
        valid = ~batch["action_is_pad"]
        reconstruction = ((value - 1).square() * valid).sum() / valid.sum().clamp_min(1)
        sample_mean = value[:, 0].square().mean()
        loss = reconstruction * batch.get("_valid_weight", 1) + sample_mean * batch.get(
            "_sample_weight", 1
        )
        if self.nonfinite:
            loss = loss * float("nan")
        return loss, {
            "reconstruction": reconstruction,
            "sample_mean": sample_mean,
            "prediction": batch["action"] + value[..., None],
        }


@pytest.mark.parametrize(
    "size,batch_size,world_size", [(6601, 2, 4), (1, 2, 4), (7, 3, 2), (8, 2, 2)]
)
def test_disjoint_coverage_with_equal_forward_counts(size, batch_size, world_size):
    covered, lengths = [], []
    for rank in range(world_size):
        sampler = EvaluationBatchSampler(size, batch_size, rank=rank, world_size=world_size)
        batches = list(sampler)
        lengths.append(len(batches))
        real = [
            index
            for number, batch in enumerate(batches)
            if not sampler.is_padding_batch(number)
            for index in batch
        ]
        assert len(real) == sampler.real_samples
        assert all(batch for batch in batches)
        covered.extend(real)
    assert sorted(covered) == list(range(size))
    assert lengths == [math.ceil(size / (batch_size * world_size))] * world_size


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_collective_validation_and_mae_match_global_targets(tmp_path, device):
    if device == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("Real FSDP validation requires two visible CUDA GPUs")
    log = tmp_path / f"validation-{device}.log"
    with log.open("wb") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(Path(__file__).resolve()),
                device,
            ],
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (tmp_path / f"validation-{device}.pid").write_text(str(process.pid))
    try:
        assert process.wait(timeout=60) == 0, log.read_text()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


def distributed_check(device):
    import os

    torch.set_num_threads(1)
    if device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl" if device == "cuda" else "gloo", timeout=timedelta(seconds=30))
    rank, world_size = dist.get_rank(), dist.get_world_size()

    def reduce(value, *, reduction):
        assert reduction == "sum"
        result = value.clone()
        dist.all_reduce(result)
        return result

    def processor(batch):
        return {key: value.to(device) for key, value in batch.items()}

    try:
        # Includes a rank with no real samples, a partial tail, and exact divisibility.
        for size in (1, 5, 8):
            samples = Targets(size)
            sampler = EvaluationBatchSampler(size, 2, rank=rank, world_size=world_size)
            loader = make_loader(samples, batch_sampler=sampler)
            policy = Policy(collective=device == "cpu").eval()
            if device == "cuda":
                from alohamini.learning.sharding import prepare_sharded_model

                policy = prepare_sharded_model(
                    policy, torch.optim.SGD(policy.parameters(), lr=0.01)
                )
            metrics = MetricAccumulator(SPECS, device=device, phase="eval")
            with torch.no_grad():
                loss = evaluate_loss(
                    policy,
                    processor,
                    loader,
                    counts,
                    loss_denominators(samples, counts),
                    device=device,
                    metrics=metrics,
                    reduce=reduce,
                )
            result = metrics.result(reduce=reduce)
            valid = [3 - i % 3 for i in range(size)]
            expected_reconstruction = sum(i * i * n for i, n in enumerate(valid)) / sum(valid)
            expected_sample_mean = sum((i + 1) ** 2 for i in range(size)) / size
            assert loss == pytest.approx(expected_reconstruction + expected_sample_mean, rel=1e-6)
            assert result["metrics"] == pytest.approx(
                {"reconstruction": expected_reconstruction, "sample_mean": expected_sample_mean},
                rel=1e-6,
            )
            assert result["metric_totals"]["reconstruction"]["count"] == sum(valid)
            assert result["metric_totals"]["sample_mean"]["count"] == size
            assert policy.calls == math.ceil(size / 4)

            def predict(batch, policy=policy):
                with torch.no_grad():
                    return policy(processor(batch))[1]["prediction"].cpu()

            mae = _offline_evaluate_chunks(
                samples,
                predict,
                ["joint"],
                batch_size=2,
                loader=loader,
                device=device,
                reduce=reduce,
            )
            expected_mae = sum((i + 1) * n for i, n in enumerate(valid)) / sum(valid)
            assert mae["valid_action_steps"] == sum(valid)
            assert mae["mae_by_action"]["joint"] == pytest.approx(expected_mae, rel=1e-6)
            assert policy.calls == 2 * math.ceil(size / 4)
            dist.barrier()

        # A bad value on one rank must not strand another rank in the next forward.
        policy.nonfinite = rank == 1
        with torch.no_grad(), pytest.raises(RuntimeError, match="Non-finite validation loss"):
            evaluate_loss(
                policy,
                processor,
                loader,
                counts,
                loss_denominators(samples, counts),
                device=device,
                reduce=reduce,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    distributed_check(sys.argv[-1])

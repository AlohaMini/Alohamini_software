"""Ordered evaluation with explicit timing and algorithm-declared loss denominators."""

import logging
import math
import time

import torch
from torch.utils.data import default_collate

logger = logging.getLogger(__name__)


class EvaluationBatchSampler:
    """Partition ordered samples with equal forward counts for FSDP collectives.

    An empty tail rank executes a dummy batch, excluded from losses and metrics.
    """

    def __init__(self, size, batch_size, *, rank=0, world_size=1):
        if any(type(value) is not int or value < 1 for value in (size, batch_size, world_size)):
            raise ValueError("Evaluation size, batch_size and world_size must be positive integers")
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("Evaluation rank must be within world_size")
        self.size, self.batch_size = size, batch_size
        self.rank, self.world_size = rank, world_size

    def __len__(self):
        return math.ceil(self.size / (self.batch_size * self.world_size))

    def _start(self, number):
        return (number * self.world_size + self.rank) * self.batch_size

    def is_padding_batch(self, number):
        return self._start(number) >= self.size

    @property
    def real_samples(self):
        rounds, tail = divmod(self.size, self.batch_size * self.world_size)
        return rounds * self.batch_size + min(
            self.batch_size, max(0, tail - self.rank * self.batch_size)
        )

    def __iter__(self):
        for number in range(len(self)):
            start = self._start(number)
            yield list(range(start, min(start + self.batch_size, self.size))) or [0]


def is_padding_batch(loader, number):
    sampler = loader.batch_sampler
    return isinstance(sampler, EvaluationBatchSampler) and sampler.is_padding_batch(number)


def timed_batches(loader, *, device="cpu", label="Evaluation", log_freq=50, action_offsets=None):
    """Time exposed loader waits and computation, including transfers/processing.

    Synchronize CUDA at compute boundaries so kernel time is not mislabeled as
    loader waiting. Prefetch can run concurrently; data_s is the exposed wait.
    """
    if type(log_freq) is not int or log_freq < 1:
        raise ValueError("eval_log_freq must be a positive integer")
    device = torch.device(device)
    size = getattr(loader.batch_sampler, "real_samples", len(loader.dataset))
    logger.info("%s: starting %d batches, %d local samples", label, len(loader), size)
    started = last_log = time.perf_counter()
    iterator = iter(loader)
    data_s = compute_s = 0.0
    samples = valid = 0
    for number in range(1, len(loader) + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        before = time.perf_counter()
        batch = next(iterator)
        data_s += time.perf_counter() - before
        before = time.perf_counter()
        yield batch
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_s += time.perf_counter() - before
        padding = is_padding_batch(loader, number - 1)
        if not padding:
            samples += len(batch["action"])
        if not padding and "action_is_pad" in batch:
            mask = ~batch["action_is_pad"]
            if action_offsets is not None:
                mask = mask & (torch.tensor(action_offsets) >= 0)[None]
            valid += int(mask.sum())
        now = time.perf_counter()
        if number == 1 or number % log_freq == 0 or number == len(loader) or now - last_log >= 10:
            elapsed = now - started
            logger.info(
                "%s batches:%d/%d samples:%d valid_action_steps:%d "
                "data_s:%.3f compute_s:%.3f smp/s:%.1f eta_s:%.1f",
                label,
                number,
                len(loader),
                samples,
                valid,
                data_s / number,
                compute_s / number,
                samples / max(elapsed, 1e-9),
                elapsed / number * (len(loader) - number),
            )
            last_log = now


def loss_denominators(samples, reduction):
    """Count selected numeric targets without loading images or fitting statistics."""
    totals = {}
    for start in range(0, len(samples), 128):
        batch = default_collate(
            [samples.action_metadata(i) for i in range(start, min(start + 128, len(samples)))]
        )
        counts = reduction(batch)
        if totals and totals.keys() != counts.keys():
            raise ValueError("Evaluation loss terms changed between batches")
        for key, value in counts.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError("Evaluation loss denominators must be finite and nonnegative")
            totals[key] = totals.get(key, 0) + value
    if not totals or not any(value > 0 for value in totals.values()):
        raise ValueError("Evaluation requires at least one positive loss denominator")
    return totals


def evaluate_loss(
    model, processor, loader, reduction, totals, *, device, log_freq=50, metrics=None, reduce=None
):
    """Sum contributions using global denominators, including across FSDP ranks."""
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    batches = timed_batches(loader, device=device, label="Validation", log_freq=log_freq)
    for number, raw in enumerate(batches):
        padding = is_padding_batch(loader, number)
        counts = reduction(raw)
        batch = processor(raw)
        batch.update(
            {
                key: 0.0 if padding or total == 0 else counts[key] / total
                for key, total in totals.items()
            }
        )
        loss, outputs = model(batch)
        if metrics is not None and not padding:
            metrics.add(outputs, raw, counts)
        value = loss.detach().double()
        invalid = (~torch.isfinite(value)).int()
        if reduce is not None:
            # Fail together before another collective model forward.
            invalid = reduce(invalid, reduction="sum")
        if invalid.item():
            raise RuntimeError("Non-finite validation loss")
        if not padding:
            loss_sum += value
    if reduce is not None:
        loss_sum = reduce(loss_sum, reduction="sum")
    return loss_sum.item()

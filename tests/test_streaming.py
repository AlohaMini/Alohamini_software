"""Coverage, resume and bounded-working-set contracts for local sampling."""

import tracemalloc
from collections import Counter
from itertools import chain, islice

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from alohamini.learning.streaming import StreamingBatchSampler, _Permutation


class EpisodeSamples(Dataset):
    """Selected physical anchors, including gaps and two segments in one episode."""

    def __init__(self):
        self.sample_indices = [0, 2, 3, 5, 6, 7, 10, 11, 14, 17, 18, 21, 23]
        self.segment_starts = [0] * 5 + [5] * 4 + [9] * 6 + [15] * 10
        self.segment_ends = [5] * 5 + [9] * 4 + [15] * 6 + [25] * 10
        self.locations = (
            [(4, i) for i in range(5)]
            + [(4, i) for i in range(5, 9)]
            + [(8, i) for i in range(6)]
            + [(21, i) for i in range(10)]
        )

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, index):
        row = self.sample_indices[index]
        start, stop = self.segment_starts[row], self.segment_ends[row]
        return {
            "index": index,
            "physical_row": row,
            "episode": self.locations[row][0],
            "window": torch.tensor([max(start, row - 1), row, min(stop - 1, row + 1)]),
        }


class LinearSamples(Dataset):
    def __init__(self, lengths):
        self.sample_indices = range(sum(lengths))
        self.segment_starts, self.segment_ends, self.locations = [], [], []
        start = 0
        for episode, length in enumerate(lengths):
            self.segment_starts.extend([start] * length)
            self.segment_ends.extend([start + length] * length)
            self.locations.extend((episode, i) for i in range(length))
            start += length

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, index):
        return index


def flatten(batches):
    return list(chain.from_iterable(batches))


@pytest.mark.parametrize("size", [1, 2, 3, 7, 16, 19, 31, 65, 127, 257])
def test_constant_space_permutation_is_bijective_for_uneven_domains(size):
    for seed in (0, 781, 2**64 - 1):
        order = _Permutation(size, seed)
        assert sorted(order[i] for i in range(size)) == list(range(size))


def test_selected_anchors_are_covered_once_without_rewriting_temporal_windows():
    samples = EpisodeSamples()
    sampler = StreamingBatchSampler(samples, 4, 1000, pool_size=2)
    loader = DataLoader(samples, batch_sampler=sampler)
    actual = []
    for batch in loader:
        for number, index in enumerate(batch["index"].tolist()):
            expected = samples[index]
            assert batch["physical_row"][number].item() == expected["physical_row"]
            assert batch["episode"][number].item() == expected["episode"]
            assert torch.equal(batch["window"][number], expected["window"])
            actual.append(index)
    assert sorted(actual) == list(range(len(samples)))
    assert len(actual) == len(set(actual))
    assert len(sampler) == 4


@pytest.mark.parametrize("pool_size", [1, 2, 8])
def test_deterministic_epochs_and_pool_bounds(pool_size):
    samples = LinearSamples([37, 5, 19, 23, 2, 41, 7, 13, 17, 3])
    first = StreamingBatchSampler(samples, 9, 1000, pool_size=pool_size)
    same = StreamingBatchSampler(samples, 9, 1000, pool_size=pool_size)
    orders = []
    for _ in range(2):
        order = flatten(first)
        assert order == flatten(same)
        assert sorted(order) == list(range(len(samples)))
        counts = Counter(episode for episode, _ in samples.locations)
        active = set()
        for index in order:
            episode = samples.locations[index][0]
            active.add(episode)
            assert len(active) <= pool_size
            counts[episode] -= 1
            if counts[episode] == 0:
                active.remove(episode)
        assert not active
        orders.append(order)
    assert orders[0] != orders[1]
    assert flatten(StreamingBatchSampler(samples, 9, 1001, pool_size=pool_size)) != orders[0]


@pytest.mark.parametrize("size,world_size", [(35, 2), (35, 8), (2, 8), (1, 3)])
def test_distributed_ranks_cover_once_before_repeating_only_epoch_prefix(size, world_size):
    samples = LinearSamples([size])
    samplers = [
        StreamingBatchSampler(samples, 2, 12, rank, world_size) for rank in range(world_size)
    ]
    for _ in range(2):
        ranks = [list(sampler) for sampler in samplers]
        assert len({len(rank) for rank in ranks}) == 1
        assert all(len(batch) == 2 for rank in ranks for batch in rank)
        order = [item for step in zip(*ranks, strict=True) for batch in step for item in batch]
        assert sorted(order[:size]) == list(range(size))
        assert order[size:] == [order[i % size] for i in range(len(order) - size)]


def test_drop_last_keeps_unique_equal_rank_batches_without_padding():
    samples = LinearSamples([7, 3, 9])
    ranks = [list(StreamingBatchSampler(samples, 3, 5, rank, 2, True)) for rank in (0, 1)]
    order = [item for step in zip(*ranks, strict=True) for batch in step for item in batch]
    assert len(order) == 18
    assert len(set(order)) == 18
    assert all(len(batch) == 3 for rank in ranks for batch in rank)


@pytest.mark.parametrize("world_size,drop", [(1, False), (2, False), (2, True)])
def test_consumed_batch_resume_ignores_prefetch_and_advances_epoch_once(world_size, drop):
    samples = EpisodeSamples()
    for rank in range(world_size):
        original = StreamingBatchSampler(samples, 2, 772, rank, world_size, drop, pool_size=2)
        first, second, third = list(original), list(original), list(original)
        # Simulate prefetched indices from later batches; only one batch was consumed.
        prefetched = StreamingBatchSampler(samples, 2, 772, rank, world_size, drop, pool_size=2)
        list(prefetched)
        list(prefetched)
        resumed = StreamingBatchSampler(
            samples, 2, 772, rank, world_size, drop, consumed=len(first) + 1, pool_size=2
        )
        assert (resumed.epoch, resumed.offset) == (1, 1)
        iterator = iter(resumed)
        assert (resumed.epoch, resumed.offset) == (1, 1)
        batch = next(iterator)
        assert (resumed.epoch, resumed.offset) == (2, 0)
        assert [batch, *iterator] == second[1:]
        assert list(resumed) == third
        boundary = StreamingBatchSampler(
            samples, 2, 772, rank, world_size, drop, consumed=len(first), pool_size=2
        )
        assert list(boundary) == second


@pytest.mark.parametrize("workers", [0, 2])
def test_ordered_map_loader_matches_parent_sampler_across_persistent_epochs_and_resume(workers):
    samples = LinearSamples([17, 11, 9, 7, 5])
    options = (
        {"num_workers": workers, "persistent_workers": True, "multiprocessing_context": "spawn"}
        if workers
        else {}
    )
    expected_sampler = StreamingBatchSampler(samples, 4, 918, pool_size=2)
    sampler = StreamingBatchSampler(samples, 4, 918, pool_size=2)
    loader = DataLoader(samples, batch_sampler=sampler, **options)
    try:
        for _ in range(2):
            assert [batch.tolist() for batch in loader] == list(expected_sampler)
        resumed = StreamingBatchSampler(samples, 4, 918, consumed=2, pool_size=2)
        reference = StreamingBatchSampler(samples, 4, 918, pool_size=2)
        resumed_loader = DataLoader(samples, batch_sampler=resumed, **options)
        try:
            assert [batch.tolist() for batch in resumed_loader] == list(reference)[2:]
            assert [batch.tolist() for batch in resumed_loader] == list(reference)
        finally:
            if workers and resumed_loader._iterator is not None:
                resumed_loader._iterator._shutdown_workers()
    finally:
        if workers and loader._iterator is not None:
            loader._iterator._shutdown_workers()


class ConstantSequence:
    def __init__(self, size, value):
        self.size, self.value, self.reads = size, value, 0

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        self.reads += 1
        return self.value


class HugeSamples:
    def __init__(self):
        self.size = 1_000_000
        self.sample_indices = range(self.size)
        self.segment_starts = ConstantSequence(self.size, 0)
        self.segment_ends = ConstantSequence(self.size, self.size)
        self.locations = ConstantSequence(self.size, (0, 0))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        raise AssertionError("Constructing/resuming a sampler must not load samples")

    @property
    def rows(self):
        raise AssertionError("Sampler must not read training rows")

    @property
    def records(self):
        raise AssertionError("Sampler must not read sidecar records")


def test_huge_episode_needs_no_full_anchor_permutation_or_sample_materialization():
    samples = HugeSamples()
    tracemalloc.start()
    try:
        sampler = StreamingBatchSampler(samples, 16, 72, consumed=4, pool_size=2)
        batches = list(islice(iter(sampler), 3))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000
    assert len(set(flatten(batches))) == 48
    assert samples.segment_starts.reads <= 2
    assert samples.segment_ends.reads <= 2
    assert samples.locations.reads <= 3


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": True}, "batch_size"),
        ({"world_size": 0}, "world_size"),
        ({"rank": -1}, "rank"),
        ({"rank": 2, "world_size": 2}, "rank"),
        ({"pool_size": 0}, "pool_size"),
        ({"pool_size": 1.5}, "pool_size"),
        ({"consumed": -1}, "consumed"),
        ({"drop_last": 1}, "drop_last"),
        ({"seed": None}, "seed"),
        ({"batch_size": 99, "drop_last": True}, "discard every"),
    ],
)
def test_invalid_sampler_configuration_is_rejected(kwargs, match):
    options = dict(batch_size=2, seed=1000)
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        StreamingBatchSampler(EpisodeSamples(), **options)


def test_empty_dataset_is_rejected_and_episode_identity_cannot_recur_out_of_order():
    with pytest.raises(ValueError, match="dataset size"):
        StreamingBatchSampler(LinearSamples([]), 2, 1000)
    samples = LinearSamples([2, 2, 2])
    samples.locations[-2:] = [(0, 0), (0, 1)]
    with pytest.raises(ValueError, match="contiguous"):
        StreamingBatchSampler(samples, 2, 1000)
    samples = LinearSamples([1])
    samples.segment_starts = [1]
    with pytest.raises(ValueError, match="outside"):
        StreamingBatchSampler(samples, 2, 1000)


def test_unused_iterator_does_not_advance_epoch_or_lose_consumed_offset():
    samples = EpisodeSamples()
    reference = list(StreamingBatchSampler(samples, 2, 7))
    sampler = StreamingBatchSampler(samples, 2, 7, consumed=1)
    unused = iter(sampler)
    assert (sampler.epoch, sampler.offset) == (0, 1)
    assert list(sampler) == reference[1:]
    unused.close()

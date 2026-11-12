"""Bounded episode-pool sampling for local, disk-backed training datasets.

This is a map-style batch sampler: the parent process owns epoch advancement,
and ordinary DataLoader workers load only the requested samples. It retains
episode ranges and a bounded pool of integer cursors, never a full permutation
of the dataset or decoded samples. DataLoader prefetch and the dataset's own
disk caches have their separate, explicit memory bounds.
"""

import operator
import random
from bisect import bisect_left
from dataclasses import dataclass
from itertools import islice

_MASK64 = (1 << 64) - 1


def _mix(value):
    """SplitMix64 integer mixer; independent of Python's randomized hash."""
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def _key(seed, epoch, domain):
    return _mix(_mix(seed & _MASK64) ^ _mix(epoch) ^ _mix(domain))


class _Permutation:
    """Constant-space keyed bijection of range(size), via Feistel cycle walking."""

    def __init__(self, size, key):
        self.size = size
        self.half_bits = max(1, ((size - 1).bit_length() + 1) // 2)
        self.mask = (1 << self.half_bits) - 1
        self.keys = tuple(_mix(key + i) for i in range(6))

    def __getitem__(self, index):
        if self.size == 1:
            return 0
        value = index
        while True:
            left, right = value >> self.half_bits, value & self.mask
            for key in self.keys:
                left, right = right, left ^ (_mix(right ^ key) & self.mask)
            value = (left << self.half_bits) | right
            if value < self.size:
                return value


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _episode_ranges(samples):
    """Jump through sorted physical anchors using their episode/segment bounds.

    ``sample_indices`` is the dataset's ascending physical-row mapping. Binary
    search avoids reading every anchor merely to recover episode ranges; rows
    and sidecar records are never requested here. Adjacent segments belonging
    to one episode form a single range in the selected Dataset index space.
    """
    anchors = samples.sample_indices
    size = len(samples)
    if len(anchors) != size:
        raise ValueError("sample_indices length does not match the dataset")
    ranges = []
    position = 0
    previous_episode = None
    seen = set()
    while position < size:
        row = _integer(anchors[position], "physical row")
        start = _integer(samples.segment_starts[row], "segment start")
        end = _integer(samples.segment_ends[row], "segment end", minimum=1)
        if not start <= row < end:
            raise ValueError("Sample is outside its episode segment")
        stop = bisect_left(anchors, end, lo=position + 1)
        last = _integer(anchors[stop - 1], "physical row")
        episode = _integer(samples.locations[row][0], "episode")
        if not row <= last < end or samples.locations[last][0] != episode:
            raise ValueError("Sample anchors must ascend within episode segments")
        if ranges and previous_episode == episode:
            ranges[-1] = (ranges[-1][0], stop)
        else:
            if episode in seen:
                raise ValueError("Selected samples of each episode must be contiguous")
            seen.add(episode)
            ranges.append((position, stop))
        previous_episode = episode
        position = stop
    return tuple(ranges)


@dataclass
class _Cursor:
    start: int
    remaining: int
    order: _Permutation
    position: int = 0

    def take(self):
        result = self.start + self.order[self.position]
        self.position += 1
        self.remaining -= 1
        return result


class StreamingBatchSampler:
    """Exact-coverage local sampler with a bounded pool of active episodes.

    Episodes and their anchors receive deterministic constant-space integer
    permutations. At each step a pooled episode is selected proportionally to
    its remaining anchors, and an exhausted episode is replaced from the epoch
    order. Each selected sample occurs once before optional distributed tail
    padding. This changes mixing relative to a global random permutation.

    ``consumed`` counts completed rank batches, matching RankBatchSampler.
    Restore with the same seed, pool_size, batch_size and world_size. Resuming
    replays integer sampling up to the consumed offset, without loading those
    samples. Only the parent advances epoch/offset, on the iterator's first
    request. An unused iterator cannot skip an epoch during DataLoader worker
    initialization. Normal ordered dispatch also works with persistent workers.
    """

    def __init__(
        self,
        samples,
        batch_size,
        seed,
        rank=0,
        world_size=1,
        drop_last=False,
        consumed=0,
        pool_size=8,
    ):
        self.size = _integer(len(samples), "dataset size", minimum=1)
        self.batch_size = _integer(batch_size, "batch_size", minimum=1)
        self.world_size = _integer(world_size, "world_size", minimum=1)
        self.rank = _integer(rank, "rank")
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        if type(drop_last) is not bool:
            raise ValueError("drop_last must be a boolean")
        self.drop_last = drop_last
        self.pool_size = _integer(pool_size, "pool_size", minimum=1)
        consumed = _integer(consumed, "consumed")
        if isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        try:
            self.seed = operator.index(seed)
        except TypeError as error:
            raise ValueError("seed must be an integer") from error
        self._ranges = _episode_ranges(samples)
        width = self.batch_size * self.world_size
        self.batches = self.size // width if drop_last else (self.size + width - 1) // width
        if not self.batches:
            raise ValueError("drop_last would discard every global batch")
        self.epoch, self.offset = divmod(consumed, self.batches)

    def __len__(self):
        return self.batches

    def _indices(self, epoch):
        episode_order = _Permutation(len(self._ranges), _key(self.seed, epoch, 0))
        rng = random.Random(_key(self.seed, epoch, 1))
        pool = []
        next_episode = 0

        def admit():
            nonlocal next_episode
            number = episode_order[next_episode]
            start, stop = self._ranges[number]
            pool.append(
                _Cursor(
                    start,
                    stop - start,
                    _Permutation(stop - start, _key(self.seed, epoch, number + 2)),
                )
            )
            next_episode += 1

        for _ in range(min(self.pool_size, len(self._ranges))):
            admit()
        remaining = sum(cursor.remaining for cursor in pool)
        while pool:
            choice = rng.randrange(remaining)
            for _number, cursor in enumerate(pool):
                choice -= cursor.remaining
                if choice < 0:
                    break
            index = cursor.take()
            remaining -= 1
            if cursor.remaining == 0:
                del pool[_number]
                if next_episode < len(self._ranges):
                    admit()
                    remaining += pool[-1].remaining
            yield index

    def _padded_indices(self, epoch):
        prefix = []
        width = self.batch_size * self.world_size
        for index in self._indices(epoch):
            if self.world_size > 1 and len(prefix) < width:
                prefix.append(index)
            yield index
        if self.world_size > 1 and not self.drop_last:
            for i in range(self.batches * width - self.size):
                yield prefix[i % len(prefix)]

    def _iter_batches(self, epoch, offset):
        order = self._padded_indices(epoch)
        width = self.batch_size * self.world_size
        for number in range(self.batches):
            batch = list(islice(order, width))
            if number >= offset:
                yield batch[self.rank * self.batch_size : (self.rank + 1) * self.batch_size]

    def __iter__(self):
        # PyTorch's initial multiprocessing iterator can construct and discard
        # a sampler iterator before _reset(). Claim an epoch only on first next.
        epoch, offset = self.epoch, self.offset
        self.epoch += 1
        self.offset = 0
        yield from self._iter_batches(epoch, offset)

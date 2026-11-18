"""Compact episode layout views without a Python object for every frame."""

import operator
from bisect import bisect_right
from collections.abc import Sequence

from alohamini.learning.data_store import IndexSet


def ordered_indices(indices):
    """Keep legacy set order deterministic without copying file-backed bit masks."""
    return iter(indices) if isinstance(indices, IndexSet) else iter(sorted(indices))


class EpisodeLayout:
    def __init__(self, spans):
        self.spans = tuple(spans)
        self.ends = tuple(end for _, _, end in self.spans)
        previous = 0
        for _, start, end in self.spans:
            if start != previous or end < start:
                raise ValueError("Episode layout must have contiguous ordered spans")
            previous = end
        self.size = previous

    def view(self, field):
        if field not in ("location", "start", "end"):
            raise ValueError("Unknown episode layout field")
        return _EpisodeView(self, field)


class _EpisodeView(Sequence):
    def __init__(self, layout, field):
        self.layout, self.field = layout, field

    def __len__(self):
        return self.layout.size

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("episode layout index out of range")
        episode, start, end = self.layout.spans[bisect_right(self.layout.ends, index)]
        if self.field == "location":
            return episode, index - start
        return start if self.field == "start" else end

    def __iter__(self):
        for episode, start, end in self.layout.spans:
            for frame in range(end - start):
                yield (
                    (episode, frame)
                    if self.field == "location"
                    else (start if self.field == "start" else end)
                )

    def __eq__(self, other):
        if not isinstance(other, Sequence):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other, strict=True))

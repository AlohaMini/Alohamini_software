"""Exact numeric quantiles with a resumable sorted value distribution."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from alohamini.datasets.statistics import RunningQuantileStats


class ExactDistributionStats(RunningQuantileStats):
    """Keep FP64 moments and exact per-coordinate values with multiplicities.

    update() uses the unchanged parent moment calculation and retains owned new
    batches. Quantile or state export merges only these pending batches into the
    existing sorted distribution; historical values are never sorted again.
    """

    def __init__(self):
        super().__init__(num_quantile_bins=1)
        self._values = []
        self._value_counts = []
        self._pending = []

    def _adjust_histograms(self):
        pass

    def _update_histograms(self, batch):
        self._pending.append(batch.copy())

    def _merge_pending(self):
        if not self._pending:
            return
        pending = (
            self._pending[0] if len(self._pending) == 1 else np.concatenate(self._pending, axis=0)
        )
        values, counts = [], []
        for column in range(pending.shape[1]):
            new_values, new_counts = np.unique(pending[:, column], return_counts=True)
            new_counts = new_counts.astype(np.int64, copy=False)
            if not self._values:
                values.append(new_values)
                counts.append(new_counts)
                continue
            old_values = self._values[column]
            old_counts = self._value_counts[column].copy()
            positions = np.searchsorted(old_values, new_values)
            matched = positions < old_values.size
            matched[matched] = old_values[positions[matched]] == new_values[matched]
            old_counts[positions[matched]] += new_counts[matched]
            unmatched = ~matched
            if unmatched.any():
                values.append(np.insert(old_values, positions[unmatched], new_values[unmatched]))
                counts.append(np.insert(old_counts, positions[unmatched], new_counts[unmatched]))
            else:
                values.append(old_values)
                counts.append(old_counts)
        # Publish together so a failed allocation cannot partially merge a batch.
        self._values, self._value_counts = values, counts
        self._pending.clear()

    def _compute_quantiles(self):
        self._merge_pending()
        ranks = np.asarray(self._quantile_list, dtype=np.float64) * (self._count - 1)
        lower, upper = np.floor(ranks).astype(np.int64), np.ceil(ranks).astype(np.int64)
        fraction = ranks - lower
        result = np.empty((len(ranks), len(self._mean)), dtype=np.float64)
        for column, (values, counts) in enumerate(
            zip(self._values, self._value_counts, strict=True)
        ):
            cumulative = np.cumsum(counts, dtype=np.int64)
            left = values[np.searchsorted(cumulative, lower, side="right")]
            right = values[np.searchsorted(cumulative, upper, side="right")]
            difference = right - left
            # NumPy's linear quantile uses this branch to reduce cancellation
            # near the upper endpoint. Keep its operation ordering unchanged.
            result[:, column] = np.where(
                fraction < 0.5,
                left + difference * fraction,
                right - difference * (1 - fraction),
            )
        return result

    def export_state(self) -> dict:
        """Return independent numeric arrays suitable for allow_pickle=False NPZ.

        count is an int64 scalar; moment vectors and values_N are float64, and
        counts_N are int64. Coordinate count is the length of mean. Empty trackers
        have no distribution to persist and are rejected.
        """
        if self._count < 1:
            raise ValueError("Cannot export distribution statistics without vectors")
        self._merge_pending()
        state = {"count": np.array(self._count, dtype=np.int64)}
        state.update(
            {name: getattr(self, f"_{name}").copy() for name in ("mean", "m2", "min", "max")}
        )
        for column, (values, counts) in enumerate(
            zip(self._values, self._value_counts, strict=True)
        ):
            state[f"values_{column}"] = values.copy()
            state[f"counts_{column}"] = counts.copy()
        return state

    @classmethod
    def from_state(cls, state) -> ExactDistributionStats:
        """Validate an untrusted numeric state and restore owned accumulator arrays."""
        if not isinstance(state, Mapping):
            raise ValueError("Distribution state must be a mapping")
        required = {"count", "mean", "m2", "min", "max"}
        if not required <= state.keys():
            raise ValueError("Distribution state is missing moment fields")
        count = np.asarray(state["count"])
        if count.ndim != 0 or count.dtype.kind not in "iu":
            raise ValueError("Distribution count must be an integer scalar")
        count = int(count)
        if not 1 <= count <= np.iinfo(np.int64).max:
            raise ValueError("Distribution count must be a positive int64 value")

        def vector(name):
            value = np.asarray(state[name])
            if value.ndim != 1 or value.size == 0 or value.dtype.kind not in "fiu":
                raise ValueError(f"Distribution {name} must be a numeric vector")
            value = value.astype(np.float64, copy=True)
            if not np.isfinite(value).all():
                raise ValueError(f"Distribution {name} must contain finite values")
            return value

        moments = {name: vector(name) for name in ("mean", "m2", "min", "max")}
        width = moments["mean"].size
        if any(value.shape != (width,) for value in moments.values()):
            raise ValueError("Distribution moment shapes disagree")
        if (moments["m2"] < 0).any():
            raise ValueError("Distribution centered second moments must be nonnegative")
        required |= {
            f"{prefix}_{column}" for prefix in ("values", "counts") for column in range(width)
        }
        if set(state) != required:
            raise ValueError("Distribution coordinate fields disagree with moment dimensions")
        values, counts = [], []
        for column in range(width):
            coordinate = vector(f"values_{column}")
            multiplicities = np.asarray(state[f"counts_{column}"])
            if (
                multiplicities.shape != coordinate.shape
                or multiplicities.dtype.kind not in "iu"
                or not (multiplicities > 0).all()
                or multiplicities.size > count
                or (multiplicities > count).any()
            ):
                raise ValueError("Distribution counts must be positive integers summing to count")
            cumulative = np.cumsum(multiplicities, dtype=np.uint64)
            if (cumulative[1:] <= cumulative[:-1]).any() or cumulative[-1] != count:
                raise ValueError("Distribution counts must sum to count without overflow")
            if not (coordinate[1:] > coordinate[:-1]).all():
                raise ValueError("Distribution values must be strictly increasing")
            if coordinate[0] != moments["min"][column] or coordinate[-1] != moments["max"][column]:
                raise ValueError("Distribution endpoints disagree with moment min/max")
            values.append(coordinate)
            counts.append(multiplicities.astype(np.int64, copy=True))
        result = cls()
        result._count = count
        for name, value in moments.items():
            setattr(result, f"_{name}", value)
        result._values, result._value_counts = values, counts
        return result

"""Exact quantiles and unchanged FP64 moments across persisted recording state."""

import io

import numpy as np
import pytest

from alohamini.datasets.exact_distribution import ExactDistributionStats
from alohamini.datasets.statistics import DEFAULT_QUANTILES, ExactQuantileStats


def values():
    rng = np.random.default_rng(318)
    batch = rng.normal(size=(1027, 5))
    batch[:, 1] = rng.integers(-3, 4, len(batch))
    batch[:, 2] = 1e12 + rng.integers(-20, 21, len(batch)) / 1024
    batch[:, 3] = 0
    batch[:, 4] = 0.1
    return batch


def assert_statistics(actual, expected):
    assert set(actual) == set(expected)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.mark.parametrize("size", [1, 17, 512, 2048])
def test_quantiles_match_numpy_and_moments_match_old_tracker_for_each_batch_boundary(size):
    data = values()
    new, old = ExactDistributionStats(), ExactQuantileStats()
    try:
        for start in range(0, len(data), size):
            new.update(data[start : start + size])
            old.update(data[start : start + size])
        result = new.get_statistics()
        assert_statistics(result, old.get_statistics())
        expected = np.quantile(data, DEFAULT_QUANTILES, axis=0, method="linear")
        for index, q in enumerate(new._quantile_keys):
            np.testing.assert_array_equal(result[q], expected[index], err_msg=q)
    finally:
        old._values.close()


@pytest.mark.parametrize(
    "data",
    [
        np.array([[3.5, -7.25]]),
        np.zeros((23, 2)),
        np.full((1001, 2), 0.1),
        np.array([[0.1], [0.3]]),
        np.array([[0.1], [0.3], [4.1], [12.3]]),
        np.array([[0], [0], [0], [1], [1], [1], [2], [2], [2]], dtype=np.float32),
    ],
)
def test_constant_singleton_duplicates_and_linear_interpolation(data):
    tracker = ExactDistributionStats()
    tracker.update(data)
    stats = tracker.get_statistics()
    expected = np.quantile(data.astype(np.float64), DEFAULT_QUANTILES, axis=0, method="linear")
    for index, q in enumerate(tracker._quantile_keys):
        np.testing.assert_array_equal(stats[q], expected[index])
    restored = ExactDistributionStats.from_state(tracker.export_state())
    assert_statistics(restored.get_statistics(), stats)


def test_pending_batches_are_owned_and_only_new_values_are_sorted_on_export(monkeypatch):
    tracker = ExactDistributionStats()
    tracker.update(np.arange(1000, dtype=np.float64)[:, None])
    tracker.export_state()
    first = np.array([[-3.0], [-2.0], [0.0], [1.5], [1.6], [999.0], [1001.0]])
    second = first.copy()
    tracker.update(first)
    tracker.update(second)
    first[:] = second[:] = 123456
    unique = np.unique
    sorted_sizes = []

    def only_pending(data, *args, **kwargs):
        sorted_sizes.append(len(data))
        assert len(data) == 14, "old distributions must not be concatenated and sorted again"
        return unique(data, *args, **kwargs)

    monkeypatch.setattr(np, "unique", only_pending)
    state = tracker.export_state()
    assert sorted_sizes == [14]
    distribution = state["values_0"]
    counts = state["counts_0"]
    assert np.all(distribution[1:] > distribution[:-1])
    np.testing.assert_array_equal(distribution[:6], [-3, -2, 0, 1, 1.5, 1.6])
    assert counts[np.searchsorted(distribution, 0)] == 3
    assert counts[np.searchsorted(distribution, 999)] == 3
    assert counts[np.searchsorted(distribution, 1.5)] == 2
    assert distribution[-1] == 1001 and counts.sum() == 1014
    tracker.get_statistics()
    assert sorted_sizes == [14], "an unchanged state must not be rebuilt"


def test_npz_round_trip_resume_preserves_moments_and_quantiles_with_original_batch_boundaries():
    batches = np.array_split(values(), [3, 8, 519, 800])
    uninterrupted, persisted = ExactDistributionStats(), ExactDistributionStats()
    for batch in batches[:3]:
        uninterrupted.update(batch)
        persisted.update(batch)
    state = persisted.export_state()
    assert all(
        isinstance(value, np.ndarray) and value.dtype.kind != "O" for value in state.values()
    )
    stream = io.BytesIO()
    np.savez_compressed(stream, **state)
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as archive:
        restored = ExactDistributionStats.from_state(archive)
    for batch in batches[3:]:
        uninterrupted.update(batch)
        restored.update(batch)
    # Identical original batches keep the unchanged parent's moment path bit for bit.
    assert_statistics(restored.get_statistics(), uninterrupted.get_statistics())
    for key, value in restored.export_state().items():
        np.testing.assert_array_equal(value, uninterrupted.export_state()[key])


def test_state_export_and_restore_do_not_alias_caller_arrays():
    tracker = ExactDistributionStats()
    tracker.update(np.arange(12, dtype=np.float64).reshape(6, 2))
    baseline = tracker.get_statistics()
    state = tracker.export_state()
    restored = ExactDistributionStats.from_state(state)
    for value in state.values():
        value[...] = 0
    assert_statistics(tracker.get_statistics(), baseline)
    assert_statistics(restored.get_statistics(), baseline)


def state():
    tracker = ExactDistributionStats()
    tracker.update(np.array([[0, 1], [0, 2], [3, 2]], dtype=np.float64))
    return tracker.export_state()


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.pop("m2"),
        lambda s: s.update(extra=np.array(1)),
        lambda s: s.update(count=np.array(True)),
        lambda s: s.update(count=np.array(3.0)),
        lambda s: s.update(count=np.array(0)),
        lambda s: s.update(count=np.array([3])),
        lambda s: s.update(mean=np.array([0.0])),
        lambda s: s.update(mean=np.array([float("nan"), 1.0])),
        lambda s: s.update(m2=np.array([-1.0, 0.0])),
        lambda s: s.update(max=np.array([float("inf"), 2.0])),
        lambda s: s.update(values_0=np.array([0.0, 0.0])),
        lambda s: s.update(values_0=np.array([3.0, 0.0])),
        lambda s: s.update(values_0=np.array([0.0, float("nan")])),
        lambda s: s.update(values_0=np.array([0.0, 4.0])),
        lambda s: s.update(values_0=np.array([], dtype=np.float64)),
        lambda s: s.update(values_0=np.array([0.0, 3.0], dtype=object)),
        lambda s: s.update(counts_0=np.array([0, 3])),
        lambda s: s.update(counts_0=np.array([-1, 4])),
        lambda s: s.update(counts_0=np.array([2.0, 1.0])),
        lambda s: s.update(counts_0=np.array([True, True])),
        lambda s: s.update(counts_0=np.array([2, 2])),
        lambda s: s.update(counts_0=np.array([3])),
        lambda s: s.update(counts_0=np.array([2**63, 2**63 + 3], dtype=np.uint64)),
    ],
)
def test_invalid_persisted_state_is_rejected(change):
    current = state()
    change(current)
    with pytest.raises(ValueError):
        ExactDistributionStats.from_state(current)


def test_count_validation_rejects_uint64_wrap_even_if_the_wrapped_sum_matches():
    current = state()
    maximum = np.iinfo(np.int64).max
    current.update(
        count=np.array(maximum),
        values_0=np.arange(4, dtype=np.float64),
        counts_0=np.array([maximum, maximum, maximum, 2], dtype=np.uint64),
        counts_1=np.array([maximum - 1, 1], dtype=np.int64),
    )
    assert np.sum(current["counts_0"], dtype=np.uint64) == maximum
    with pytest.raises(ValueError, match="without overflow"):
        ExactDistributionStats.from_state(current)


def test_empty_and_nonfinite_updates_keep_existing_rejection_semantics():
    tracker = ExactDistributionStats()
    with pytest.raises(ValueError, match="without vectors"):
        tracker.get_statistics()
    with pytest.raises(ValueError, match="without vectors"):
        tracker.export_state()
    for batch in (np.empty((0, 2)), np.array([[float("nan"), 1]]), np.array([[1, float("inf")]])):
        with pytest.raises(ValueError, match="nonempty, finite"):
            tracker.update(batch)
    assert tracker._count == 0
    tracker.update(np.array([[1.0, 2.0]]))
    baseline = tracker.export_state()
    with pytest.raises(ValueError, match="vector length"):
        tracker.update(np.zeros((2, 3)))
    for key, value in baseline.items():
        np.testing.assert_array_equal(tracker.export_state()[key], value)

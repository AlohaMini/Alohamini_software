"""Field statistics retain their exact populations without whole-recording arrays."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_kinematic_fields import planar_spec

from alohamini.datasets.field_transforms import FieldTransforms
from alohamini.datasets.statistics import DEFAULT_QUANTILES, ExactQuantileStats
from alohamini.learning import data_fields, fixed
from alohamini.learning.data_fields import FieldDataset
from alohamini.learning.data_store import FileBackedIntSequence, IndexSet, RowStore
from alohamini.policies.configuration import PolicyFeature


class Selection:
    feature = dict(dtype="float32", names=["auxiliary", "elbow", "shoulder"], shape=[3])
    units = ["mm", "rad", "deg"]

    @staticmethod
    def frame(row):
        return np.asarray(row["observation.state"], dtype=np.float32)


def field_samples(state, action, rotation="rotvec", frame="base", *, selected_state=True):
    count, boundary = 73, 36
    rows = RowStore(cache_bytes=4096, block_rows=3)
    for i in range(count):
        rows.append(
            {
                "observation.state": [
                    250 + (i % 5) * 0.125,
                    -0.6 + (i % 17) * 0.003,
                    14 + i * 0.01,
                ],
                "action": [253 + (i % 5) * 0.25, -0.58 + (i % 13) * 0.004, 14.04 + i * 0.015],
            }
        )
    rows.finalize()

    def query(row):
        start, end = (0, boundary) if row < boundary else (boundary, count)
        return {"action": [max(start, min(end - 1, row + d)) for d in (-1, 0, 1, 3)]}, {}

    samples = FileBackedIntSequence()
    samples.extend(i for i in range(count) if i % 7 != 3)
    samples.finalize()
    used_actions, used_states = IndexSet(count), IndexSet(count)
    used_actions.update(i for row in samples for i in query(row)[0]["action"])
    used_states.update(range(0, count, 2))
    used_actions.finalize()
    used_states.finalize()
    info = {"features": {key: dict(Selection.feature) for key in ("action", "observation.state")}}
    selection = Selection() if selected_state else None
    source = SimpleNamespace(
        rows=rows,
        sample_indices=samples,
        _used_rows={"action": used_actions, "observation.state": used_states},
        _get_query_indices=query,
        selection=selection,
        info=info,
        input_features={"observation.state": PolicyFeature("STATE", (3,))} if selection else {},
    )
    fields = FieldTransforms(
        dict(
            state_representation=state,
            action_representation=action,
            rotation_representation=rotation,
            tcp_delta_frame=frame,
            kinematic_fields=planar_spec(),
            selected_state_feature=selection.feature if selection else None,
            selected_state_units=selection.units if selection else [],
            source_state_names=Selection.feature["names"],
        )
    )
    return FieldDataset(source, fields)


def numpy_populations(samples, action_windows):
    """Independent full-size reference for these small, synthetic test recordings."""
    fields = samples.field_transform
    if action_windows or fields.action == "tcp_delta":
        actions = []
        for row in samples.sample_indices:
            indices = samples._get_query_indices(row)[0]["action"]
            targets = np.asarray([samples.rows[i]["action"] for i in indices])
            if fields.action != "recorded":
                targets = fields.geometry.absolute(targets)
            if fields.action == "tcp_delta":
                anchor = fields.geometry.absolute(fields.context(samples.rows[row]))
                targets = fields.relative.encode(anchor, targets)
            actions.extend(targets)
        actions = np.asarray(actions)
    else:
        actions = np.asarray([samples.rows[i]["action"] for i in samples._used_rows["action"]])
        if fields.action != "recorded":
            actions = fields.geometry.absolute(actions)
    result = {"action": actions}
    if samples.selection:
        rows = [samples.rows[i] for i in samples._used_rows["observation.state"]]
        states = np.asarray([samples.selection.frame(row) for row in rows])
        contexts = np.asarray([fields.context(row) for row in rows])
        result["observation.state"] = fields.encode(states, anchor=contexts)[0]
    return result


def assert_statistics(actual, values):
    values = values.astype(np.float64)
    assert actual["count"] == [len(values)]
    np.testing.assert_allclose(actual["mean"], values.mean(0), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual["std"], values.std(0), rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(actual["min"], values.min(0))
    np.testing.assert_array_equal(actual["max"], values.max(0))
    quantiles = np.quantile(values, DEFAULT_QUANTILES, axis=0, method="linear")
    for q, expected in zip(DEFAULT_QUANTILES, quantiles, strict=True):
        np.testing.assert_array_equal(actual[f"q{int(q * 100):02d}"], expected)


@pytest.mark.parametrize("action_windows", [False, True])
@pytest.mark.parametrize(
    "state,action,rotation,frame",
    [
        ("tcp", "recorded", "rotvec", "base"),
        ("recorded", "tcp_absolute", "rotvec", "base"),
        ("tcp", "tcp_absolute", "rotation_6d", "base"),
        ("recorded", "tcp_delta", "rotvec", "base"),
        ("tcp", "tcp_delta", "rotation_6d", "tool"),
    ],
)
def test_numeric_statistics_are_batched_and_preserve_exact_population(
    monkeypatch, state, action, rotation, frame, action_windows
):
    samples = field_samples(state, action, rotation, frame)
    expected = numpy_populations(samples, action_windows)
    monkeypatch.setattr(fixed, "_BATCH_ROWS", 7)
    absolute = samples.field_transform.geometry.absolute
    batch_sizes = []

    def bounded_absolute(values):
        values = np.asarray(values)
        size = 1 if values.ndim == 1 else len(values)
        assert size <= 7, "whole-recording transform input"
        batch_sizes.append(size)
        return absolute(values)

    class BoundedTracker(ExactQuantileStats):
        def update(self, batch):
            assert len(batch) <= 7, "whole-recording numeric tracker input"
            return super().update(batch)

    monkeypatch.setattr(samples.field_transform.geometry, "absolute", bounded_absolute)
    monkeypatch.setattr(data_fields, "ExactQuantileStats", BoundedTracker)
    actual = samples.numeric_statistics(action_windows=action_windows)
    assert batch_sizes
    assert set(actual) == set(expected)
    for key, values in expected.items():
        assert_statistics(actual[key], values)
    count = actual["action"]["count"][0]
    assert count == (
        len(samples.sample_indices) * 4
        if action_windows or action == "tcp_delta"
        else len(samples._used_rows["action"])
    )
    assert samples.rows.currbytes <= samples.rows.cache_bytes


def test_relative_statistics_without_selected_state_keep_anchor_windows(monkeypatch):
    samples = field_samples("recorded", "tcp_delta", selected_state=False)
    expected = numpy_populations(samples, action_windows=False)
    monkeypatch.setattr(fixed, "_BATCH_ROWS", 5)
    actual = samples.numeric_statistics(action_windows=False)
    assert set(actual) == {"action"}
    assert_statistics(actual["action"], expected["action"])
    assert samples.recorded_state_statistics() is None


def _guard_array_batches(monkeypatch, limit):
    asarray = np.asarray
    sizes = []

    def bounded(values, *args, **kwargs):
        if isinstance(values, list) and values and isinstance(values[0], (list, np.ndarray)):
            sizes.append(len(values))
            assert len(values) <= limit, "whole-recording numeric matrix"
        return asarray(values, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", bounded)
    return sizes


def test_recorded_state_moments_are_bounded_selected_and_cached(monkeypatch):
    samples = field_samples("tcp", "tcp_delta")
    values = np.asarray(
        [samples.selection.frame(samples.rows[i]) for i in samples._used_rows["observation.state"]],
        dtype=np.float64,
    )
    monkeypatch.setattr(fixed, "_BATCH_ROWS", 7)
    sizes = _guard_array_batches(monkeypatch, 7)
    actual = samples.recorded_state_statistics()
    np.testing.assert_allclose(actual["mean"], values.mean(0), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual["std"], values.std(0), rtol=1e-12, atol=1e-14)
    assert max(sizes) == 7
    monkeypatch.setattr(samples.selection, "frame", lambda row: pytest.fail("read cached state"))
    assert samples.recorded_state_statistics() is actual


@pytest.mark.parametrize("count", [1, 37])
def test_physical_action_summary_is_bounded_centered_and_cached(monkeypatch, count):
    values = np.column_stack(
        [1e9 + np.arange(count) * 0.125, np.full(count, 0.1), (np.arange(count) % 3 - 1) * 0.001]
    )
    rows, selected = RowStore(block_rows=3, cache_bytes=4096), IndexSet(count)
    rows.extend({"action": value.tolist()} for value in values)
    selected.update(range(0, count, 2))
    rows.finalize()
    selected.finalize()
    values = values[::2]
    samples = SimpleNamespace(
        rows=rows,
        _used_rows={"action": selected},
        info={"features": {"action": {"names": ["offset", "constant", "small"]}}},
    )
    monkeypatch.setattr(fixed, "_BATCH_ROWS", 4)
    sizes = _guard_array_batches(monkeypatch, 4)
    actual = fixed.action_summary(samples)
    assert actual["count"] == len(values)
    assert actual["names"] == ["offset", "constant", "small"]
    assert actual["source"] == "unique_training_action_rows"
    np.testing.assert_allclose(actual["mean"], values.mean(0), rtol=0, atol=2e-7)
    np.testing.assert_allclose(actual["std"], values.std(0), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(actual["min"], values.min(0))
    np.testing.assert_array_equal(actual["max"], values.max(0))
    assert max(sizes) <= 4
    samples.rows = None
    assert fixed.action_summary(samples) is actual


@pytest.mark.parametrize("bad", [[1, 2], [1, np.nan, 3], [1, 2, np.inf]])
def test_action_summary_rejects_bad_selected_values_without_caching(bad):
    samples = SimpleNamespace(
        rows=[{"action": [1, 2, 3]}, {"action": bad}],
        _used_rows={"action": {0, 1}},
        info={"features": {"action": {"names": ["a", "b", "c"]}}},
    )
    with pytest.raises(ValueError):
        fixed.action_summary(samples)
    assert not hasattr(samples, "_physical_action_summary")


def test_action_summary_empty_or_unavailable_population_is_unchanged():
    assert fixed.action_summary(SimpleNamespace()) is None
    assert fixed.action_summary(SimpleNamespace(rows=[], _used_rows={"action": set()})) is None


def test_legacy_set_indices_retain_sorted_physical_traversal(monkeypatch):
    samples = field_samples("tcp", "tcp_absolute")
    original = samples.rows
    accessed = []

    class Rows:
        def __getitem__(self, index):
            accessed.append(index)
            return original[index]

    samples.source.rows = Rows()
    samples.source._used_rows = {"observation.state": {31, 2, 17}, "action": {42, 1, 9, 23}}
    monkeypatch.setattr(fixed, "_BATCH_ROWS", 2)
    samples.numeric_statistics(action_windows=False)
    assert accessed == [2, 17, 31, 1, 9, 23, 42]
    accessed.clear()
    fixed.action_summary(samples.source)
    assert accessed == [1, 9, 23, 42]

"""Policy statistics keep their original population without eager history copies."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_kinematic_fields import planar_spec

from alohamini.datasets.kinematic_fields import KinematicFields
from alohamini.datasets.statistics import ExactQuantileStats
from alohamini.learning.data_store import IndexSet
from alohamini.policies.am_act import classification
from alohamini.policies.gr00t_n17 import adapter as groot
from alohamini.policies.spirit import processor_spirit as spirit


class ObservedIndices(IndexSet):
    """Detect consuming a whole population before its first numeric row is read."""

    def __init__(self, size, root):
        super().__init__(size, root=root)
        self.emitted = 0

    def __iter__(self):
        self.emitted = 0
        for index in super().__iter__():
            self.emitted += 1
            yield index


class NumericRows:
    def __init__(self, getter):
        self.getter = getter

    def __getitem__(self, index):
        return self.getter(index)

    def __iter__(self):
        raise AssertionError("Policy statistics must not materialize complete training rows")


def no_image_decode(*args, **kwargs):
    raise AssertionError("Fitting numeric statistics must not decode images")


@pytest.mark.parametrize("cartesian", [False, True])
def test_spirit_statistics_match_full_population_oracle_using_only_each_current_window(
    monkeypatch, cartesian
):
    count = 17
    physical = np.arange(count, dtype=np.float64)
    states = np.column_stack([250 + physical * 0.17, -0.6 + physical * 0.019, 13 + physical])
    actions = states + np.array([0.05, 0.007, -0.3])
    anchors = [0, 1, 3, 5, 8, 11, 13, 16]
    config = SimpleNamespace(
        action_representation="eef_delta" if cartesian else "recorded",
        kinematic_fields=planar_spec(),
    )
    names = planar_spec()["source_names"]

    def window(anchor):
        stop = 8 if anchor < 8 else count
        return [min(stop - 1, anchor + offset) for offset in range(4)]

    fields = spirit.SpiritFields(config)
    full_states = fields.geometry.absolute(states) if cartesian else states
    full_actions = fields.geometry.absolute(actions) if cartesian else actions
    expected_actions = []
    for anchor in anchors:
        targets = full_actions[window(anchor)]
        expected_actions.append(
            fields.relative.encode(full_states[anchor], targets) if cartesian else targets
        )
    expected = {
        "observation.state": np.quantile(full_states[anchors], [0.01, 0.99], axis=0),
        # Duplicated padding/overlapping windows intentionally retain Spirit's
        # original per-anchor statistics population, unlike unique-frame stats.
        "action": np.quantile(np.concatenate(expected_actions), [0.01, 0.99], axis=0),
    }
    reads, updates = [], []

    def read(index):
        reads.append(index)
        return {"observation.state": states[index], "action": actions[index]}

    samples = SimpleNamespace(
        sample_indices=anchors,
        rows=NumericRows(read),
        selection=SimpleNamespace(
            feature={"names": names}, frame=lambda row: row["observation.state"]
        ),
        info={"features": {"action": {"names": names}}},
        _get_query_indices=lambda anchor: ({"action": window(anchor)}, {}),
        image=no_image_decode,
    )

    class BoundedTracker(ExactQuantileStats):
        def update(self, values):
            assert len(reads) <= (len(updates) // 2 + 1) * 5
            updates.append(len(values))
            return super().update(values)

    monkeypatch.setattr(spirit, "ExactQuantileStats", BoundedTracker)
    if cartesian:
        original_absolute = KinematicFields.absolute

        def bounded_geometry(self, values):
            assert values.ndim == 2
            assert len(values) <= 4
            return original_absolute(self, values)

        monkeypatch.setattr(KinematicFields, "absolute", bounded_geometry)
    result = spirit.fit_statistics(samples, config)
    for key, quantiles in expected.items():
        np.testing.assert_allclose(result[key]["min"], quantiles[0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(result[key]["max"], quantiles[1], rtol=0, atol=1e-12)
    assert reads == [index for anchor in anchors for index in [anchor, *window(anchor)]]
    assert updates == [1, 4] * len(anchors)


def test_spirit_field_dataset_keeps_its_existing_statistics_path(monkeypatch):
    samples = SimpleNamespace(
        field_transform=object(),
        numeric_statistics=lambda: {
            "observation.state": {"q01": [1.0], "q99": [3.0]},
            "action": {"q01": [-2.0], "q99": [4.0]},
        },
    )
    monkeypatch.setattr(spirit, "SpiritFields", no_image_decode)
    assert spirit.fit_statistics(samples, None) == {
        "observation.state": {"min": [1.0], "max": [3.0]},
        "action": {"min": [-2.0], "max": [4.0]},
    }


def test_groot_fallback_percentiles_match_old_fp64_torch_quantiles_in_bounded_blocks(
    tmp_path, monkeypatch
):
    selected = list(range(0, 4402, 2))
    indices = ObservedIndices(4402, tmp_path)
    indices.update(selected)
    indices.finalize()
    calls = {"observation.state": 0, "action": 0}

    def value(index, key):
        calls[key] += 1
        assert indices.emitted - calls[key] <= 1024
        if key.startswith("observation.images."):
            no_image_decode()
        return make_value(index, key)

    def make_value(index, key):
        offset = 0.123 if key == "action" else 0
        return torch.tensor(
            [np.sin(index) * 0.7 + offset, (index % 21 - 10) / 7 + offset, 7.5],
            dtype=torch.float64,
        )

    oracle = {
        key: torch.quantile(
            torch.stack([make_value(index, key) for index in selected]),
            torch.tensor([0.01, 0.99], dtype=torch.float64),
            dim=0,
        ).numpy()
        for key in calls
    }
    base = {key: {"mean": [0.0] * 3, "std": [1.0] * 3} for key in calls}
    monkeypatch.setattr(groot, "limits_statistics", lambda samples: deepcopy(base))
    samples = SimpleNamespace(_used_rows=dict.fromkeys(calls, indices), _value=value)
    original_stack = torch.stack
    sizes = []

    def bounded_stack(values, *args, **kwargs):
        assert len(values) <= 1024
        sizes.append(len(values))
        return original_stack(values, *args, **kwargs)

    monkeypatch.setattr(torch, "stack", bounded_stack)
    result = groot.GR00TN17Algorithm().statistics(samples)
    for key in calls:
        assert calls[key] == len(selected)
        assert result[key]["mean"] == base[key]["mean"]
        assert result[key]["std"] == base[key]["std"]
        for label, quantile in (("min", 0), ("q01", 0), ("max", 1), ("q99", 1)):
            np.testing.assert_allclose(
                result[key][label], oracle[key][quantile], rtol=0, atol=1e-12
            )
            assert np.asarray(result[key][label]).dtype == np.float64
    assert sizes == [1024, 1024, 153] * 2
    indices.close()


@pytest.mark.parametrize("use_percentiles", [False, True])
def test_groot_existing_percentiles_and_disabled_percentiles_do_not_scan_rows(
    monkeypatch, use_percentiles
):
    base = {
        key: {"min": [-100.0], "max": [100.0], "q01": [-2.0], "q99": [3.0]}
        for key in ("observation.state", "action")
    }
    monkeypatch.setattr(groot, "limits_statistics", lambda samples: deepcopy(base))
    samples = SimpleNamespace(_value=no_image_decode)
    result = groot.GR00TN17Algorithm().statistics(samples, {"use_percentiles": use_percentiles})
    for key in base:
        assert result[key]["min"] == ([-2.0] if use_percentiles else [-100.0])
        assert result[key]["max"] == ([3.0] if use_percentiles else [100.0])


def test_am_act_class_counts_match_unique_frame_oracle_without_collecting_all_indices(tmp_path):
    size = 20_006
    selected = list(range(0, size, 2))
    indices = ObservedIndices(size, tmp_path)
    indices.update(selected)
    indices.finalize()
    names = ["theta.vel", "arm", "y.vel", "x.vel"]

    def action(index):
        return [
            [-45.0, 0.0, 45.0, -22.5, 22.5][index % 5],
            index / 19,
            [-0.15, 0.0, 0.15][index % 3],
            [-0.15, 0.0, 0.15, -0.075, 0.075][index % 5],
        ]

    old_values = torch.tensor([action(index) for index in selected], dtype=torch.float32)
    expected = [
        torch.bincount(
            classification.nearest_class(old_values[:, names.index(name)], centers),
            minlength=len(centers),
        ).tolist()
        for name, centers in classification.BASE_CLASSES.items()
    ]
    read_count = 0

    def read(index):
        nonlocal read_count
        assert index % 2 == 0, "Unused physical frames must not enter class counts"
        assert indices.emitted - read_count <= 8192
        read_count += 1
        return {"action": action(index)}

    samples = SimpleNamespace(
        rows=NumericRows(read),
        _used_rows={"action": indices},
        info={"features": {"action": {"names": names}}},
        image=no_image_decode,
    )
    options = {"discrete_action_mask_absent_classes": True}
    classification.prepare_options(options, samples)
    assert read_count == len(selected)
    assert options["discrete_action_class_counts"] == expected
    assert options["discrete_action_active_classes"] == [
        [count > 0 for count in row] for row in expected
    ]
    assert options["discrete_action_class_weights"] == [
        classification.class_weights(counts, reference_index=1) for counts in expected
    ]
    assert options["discrete_action_weight_source"] == "training_frames"
    indices.close()

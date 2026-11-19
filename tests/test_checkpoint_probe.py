"""Cross-environment comparison must use the same weights, inputs and tolerance."""

from copy import deepcopy

import pytest
import torch
from test_native_learning import recording as recording

from env.checkpoint_probe import compare, digest


def snapshot():
    return {
        "versions": {"torch": "2.11.0"},
        "cases": {
            "act": {
                "case": {"name": "act", "atol": 1e-5},
                "checkpoint_sha256": "weights",
                "manifest_sha256": "config",
                "input_sha256": ["input"],
                "actions": [[[1.0, 2.0]]],
                "median_seconds": 0.1,
                "peak_cuda_bytes": 1024,
            }
        },
    }


def test_same_output_and_small_rounding_pass_but_large_drift_fails():
    old = snapshot()
    new = deepcopy(old)
    new["cases"]["act"]["actions"][0][0][0] += 1e-6
    assert compare(old, new)["passed"]
    new["cases"]["act"]["actions"][0][0][0] += 0.01
    result = compare(old, new)
    assert not result["passed"]
    assert result["cases"]["act"]["max_abs_difference"] > 0.009


@pytest.mark.parametrize("key,value", [("world_size", 2), ("distributed_backend", "fsdp2")])
def test_comparison_rejects_changed_execution_layout(key, value):
    old, new = snapshot(), snapshot()
    new[key] = value
    with pytest.raises(ValueError, match=key):
        compare(old, new)


def test_old_single_device_reports_remain_comparable():
    old, new = snapshot(), snapshot()
    new.update(world_size=1, distributed_backend="single")
    assert compare(old, new)["passed"]


def quality_snapshot():
    result = snapshot()
    row = result["cases"]["act"]
    row["case"]["acceptance"] = {"metric": "mae", "rtol": 0.02, "atol": [0.01, 0.01]}
    row.update(targets=[[[0.0, 0.0]]], valid_action_mask=[[True]], action_names=["a", "b"])
    return result


def test_mae_accepts_improved_predictions_without_hiding_numerical_drift():
    old, new = quality_snapshot(), quality_snapshot()
    new["cases"]["act"]["actions"] = [[[0.5, 1.0]]]
    result = compare(old, new)["cases"]["act"]
    assert result["passed"] and not result["numerical_match"]
    assert result["mae_by_action"]["a"]["candidate"] == 0.5


def test_mae_rejects_one_worsened_dimension_even_if_other_dimensions_improve():
    old, new = quality_snapshot(), quality_snapshot()
    new["cases"]["act"]["actions"] = [[[1.1, 0.0]]]
    result = compare(old, new)
    assert not result["passed"]
    assert not result["cases"]["act"]["mae_by_action"]["a"]["passed"]


def test_mae_weights_valid_steps_and_excludes_padding():
    old, new = quality_snapshot(), quality_snapshot()
    for snapshot_ in (old, new):
        snapshot_["cases"]["act"].update(
            actions=[[[1., 2.], [1., 2.]], [[4., 8.], [999., 999.]]],
            targets=[[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]],
            valid_action_mask=[[True, True], [True, False]],
        )
    new["cases"]["act"]["actions"][-1][-1] = [0., 0.]
    row = compare(old, new)["cases"]["act"]
    assert row["passed"] and row["valid_action_steps"] == 3
    assert row["mae_by_action"]["a"]["baseline"] == 2.0


@pytest.mark.parametrize("key,value", [
    ("targets", [[[1., 0.]]]), ("valid_action_mask", [[False]]),
    ("action_names", ["b", "a"]),
])
def test_mae_rejects_changed_ground_truth(key, value):
    old, new = quality_snapshot(), quality_snapshot()
    new["cases"]["act"][key] = value
    with pytest.raises(ValueError, match="requires identical"):
        compare(old, new)


@pytest.mark.parametrize("field,value", [
    ("atol", 0.1), ("atol", [0.1]), ("atol", [-0.1, 0.1]),
    ("rtol", float("inf")), ("rtol", -0.1),
])
def test_mae_requires_explicit_valid_per_dimension_budgets(field, value):
    old = quality_snapshot()
    old["cases"]["act"]["case"]["acceptance"][field] = value
    with pytest.raises(ValueError, match="per-action atol"):
        compare(old, deepcopy(old))


@pytest.mark.parametrize("key,value", [
    ("targets", [[[float("inf"), 0.]]]),
    ("valid_action_mask", [[False]]), ("valid_action_mask", [[1]]),
    ("valid_action_mask", [[True, True]]), ("action_names", ["a", "a"]),
])
def test_mae_rejects_invalid_or_empty_evaluation(key, value):
    old = quality_snapshot()
    old["cases"]["act"][key] = value
    with pytest.raises(ValueError, match="Invalid MAE"):
        compare(old, deepcopy(old))


@pytest.mark.parametrize("key", ["case", "checkpoint_sha256", "manifest_sha256", "input_sha256"])
def test_comparison_rejects_changed_inputs_weights_or_protocol(key):
    old, new = snapshot(), snapshot()
    new["cases"]["act"][key] = "changed"
    with pytest.raises(ValueError, match="changed between environments"):
        compare(old, new)


@pytest.mark.parametrize("actions", [[], [[[float("nan"), 2.0]]], [[[1.0]]]])
def test_comparison_rejects_missing_nonfinite_or_reshaped_actions(actions):
    old, new = snapshot(), snapshot()
    new["cases"]["act"]["actions"] = actions
    with pytest.raises(ValueError, match="invalid action"):
        compare(old, new)


def test_input_digest_tracks_tensor_contents_and_field_names():
    import torch

    sample = {"image": torch.zeros(2, 3), "task": "pick"}
    assert digest(sample) == digest({"task": "pick", "image": torch.zeros(2, 3)})
    assert digest(sample) != digest({"task": "pick", "image": torch.ones(2, 3)})
    assert digest(torch.tensor(1.0)) != digest(torch.tensor(2.0))


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_capture_restores_an_existing_checkpoint_with_reproducible_outputs(
    recording, tmp_path, device
):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Requires a Mac with MPS access")
    from test_native_learning import model_options

    from alohamini.learning.data import AlohaMiniDataset
    from alohamini.learning.policy import make_policy, save_checkpoint
    from alohamini.policies.registry import algorithm
    from env.checkpoint_probe import capture

    samples = AlohaMiniDataset(recording, episodes=[0], chunk_size=3, image_size=(32, 32))
    options = model_options()
    options.update(input_features=samples.input_features, output_features=samples.output_features)
    model = make_policy("act", options)
    stats = algorithm("act").statistics(samples)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, stats, samples, training={})
    cases = [
        {
            "name": "act",
            "checkpoint": str(checkpoint),
            "dataset": str(recording),
            "episodes": [0],
            "indices": [0, 1],
            "seed": 42,
        }
    ]
    old = capture(cases, device=device)
    new = capture(cases, device=device)
    assert compare(old, new)["passed"]
    row = old["cases"]["act"]
    assert row["sample_locations"] == [[0, 0], [0, 1]]
    assert len(row["targets"]) == len(row["valid_action_mask"]) == 2
    cases[0]["acceptance"] = {"metric": "mae", "atol": [0.] * len(row["action_names"])}
    assert compare(old, new)["cases"]["act"]["acceptance_metric"] == "mae"

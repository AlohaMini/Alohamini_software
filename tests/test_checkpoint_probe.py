"""Cross-environment comparison must use the same weights, inputs and tolerance."""

from copy import deepcopy

import pytest
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


def test_capture_restores_an_existing_checkpoint_with_reproducible_outputs(recording, tmp_path):
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
    old = capture(cases, device="cpu")
    new = capture(cases, device="cpu")
    assert compare(old, new)["passed"]

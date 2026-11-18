"""Shared normalization contracts for imported VLA and world-action models."""

import pytest
import torch

from alohamini.learning.processor import Processor


def test_quantiles_use_saved_bounds_without_implicit_clipping_and_preserve_gradients():
    stats = {"action": {"q01": [2.0, 10.0], "q99": [6.0, 18.0]}}
    processor = Processor(stats, modes={"action": "QUANTILES"})
    values = torch.tensor([[[0.0, 10.0], [4.0, 14.0], [8.0, 22.0]]], requires_grad=True)
    actual = processor({"action": values})["action"]
    torch.testing.assert_close(actual, torch.tensor([[[-2.0, -1.0], [0.0, 0.0], [2.0, 2.0]]]))
    torch.testing.assert_close(processor.action(actual), values)
    actual.sum().backward()
    torch.testing.assert_close(values.grad, torch.tensor([0.5, 0.25]).expand_as(values))


def test_stepwise_quantile_statistics_and_explicit_clipping_do_not_change_labels():
    stats = {"action": {"q01": [[0.0], [10.0]], "q99": [[2.0], [14.0]]}}
    processor = Processor(stats, modes={"action": "QUANTILES"}, clip={"action": [-1.0, 1.0]})
    labels = torch.tensor([[False, True]])
    result = processor({"action": torch.tensor([[[-2.0], [18.0]]]), "action_is_pad": labels})
    torch.testing.assert_close(result["action"], torch.tensor([[[-1.0], [1.0]]]))
    assert result["action_is_pad"] is labels
    torch.testing.assert_close(processor.action(result["action"]), torch.tensor([[[0.0], [14.0]]]))


@pytest.mark.parametrize(
    "bounds",
    [{}, {"q01": [2.0]}, {"q01": [2.0], "q99": [1.0]}, {"q01": [float("nan")], "q99": [2.0]}],
)
def test_quantiles_require_finite_ordered_statistics(bounds):
    with pytest.raises(ValueError, match="action:"):
        Processor({"action": bounds}, modes={"action": "QUANTILES"})


def test_degenerate_quantile_span_is_distinct_from_limits_constant_policy():
    quantile = Processor({"action": {"q01": [3.0], "q99": [3.0]}}, modes={"action": "QUANTILES"})
    value = quantile({"action": torch.tensor([[3.0]])})["action"]
    torch.testing.assert_close(value, torch.tensor([[-1.0]]))
    torch.testing.assert_close(quantile.action(value), torch.tensor([[3.0]]))
    limits = Processor({"action": {"min": [3.0], "max": [3.0]}}, modes={"action": "MIN_MAX"})
    torch.testing.assert_close(
        limits({"action": torch.tensor([[3.0]])})["action"], torch.tensor([[0.0]])
    )


def test_quantile_protocol_options_are_field_specific_and_do_not_change_other_fields():
    stats = {key: {"q01": [0.0, 2.0], "q99": [0.0, 6.0]} for key in ("action", "observation.state")}
    processor = Processor(
        stats,
        modes=dict.fromkeys(stats, "QUANTILES"),
        quantile_options={
            "action": dict(epsilon=1e-6, clip_to_bounds=True, zero_if_zero_bounds=True),
            "observation.state": dict(epsilon=1e-6),
        },
    )
    values = torch.tensor([[0.02, 8.0]], requires_grad=True)
    result = processor({"action": values, "observation.state": values})
    assert result["action"][0, 0] == 0 and result["observation.state"][0, 0] > 1
    assert result["action"][0, 1] < 1  # Clip raw bounds before the additive epsilon.
    torch.testing.assert_close(processor.action(result["action"]), torch.tensor([[0.0, 6.0]]))
    torch.testing.assert_close(
        processor.unnormalize("observation.state", result["observation.state"]), values
    )
    result["observation.state"].sum().backward()
    assert torch.isfinite(values.grad).all() and values.grad[0, 0] > 0


@pytest.mark.parametrize(
    "options",
    [
        {"epsilon": -1},
        {"epsilon": True},
        {"epsilon": float("nan")},
        {"clip_to_bounds": 1},
        {"typo": True},
    ],
)
def test_invalid_quantile_protocol_options_fail_early(options):
    with pytest.raises(ValueError, match="quantile"):
        Processor(
            {"action": {"q01": [0], "q99": [1]}},
            modes={"action": "QUANTILES"},
            quantile_options={"action": options},
        )


def test_normalizer_keeps_a_fixed_copy_of_its_protocol():
    options = {"action": dict(epsilon=1e-6, clip_to_bounds=False)}
    processor = Processor(
        {"action": {"q01": [0.0], "q99": [1.0]}},
        modes={"action": "QUANTILES"},
        quantile_options=options,
    )
    original = processor({"action": torch.tensor([[2.0]])})["action"]
    options["action"].update(epsilon=1.0, clip_to_bounds=True)
    torch.testing.assert_close(
        processor({"action": torch.tensor([[2.0]])})["action"], original, rtol=0, atol=0
    )

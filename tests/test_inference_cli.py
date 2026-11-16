"""Inference mode selection through local and remote public entry points."""

import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from alohamini.cli import main
from alohamini.learning.trajectory_execution import TrajectoryExecutor
from alohamini.remote.cli import main as remote_main
from alohamini.trajectory_execution import (
    ExecutionConfig,
    PlanCapabilities,
    add_inference_arguments,
    pop_inference_options,
)


def parse(arguments, *, remote=False):
    parser = argparse.ArgumentParser()
    add_inference_arguments(parser, default="async" if remote else "sync")
    return pop_inference_options(vars(parser.parse_args(arguments)), remote=remote)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--inference.prediction_lead_s=0.1"],
        ["--inference.type=sync", "--inference.continuity=none"],
        ["--inference.type=rtc", "--inference.continuity=none"],
        ["--inference.type=async", "--inference.replan_interval_s=0"],
        ["--inference.type=rtc", "--inference.max_prediction_age_s=nan"],
    ],
)
def test_conflicting_modes_and_invalid_budgets_fail_before_inference(arguments):
    with pytest.raises(ValueError):
        parse(arguments)


@pytest.mark.parametrize(
    "mode,capabilities,expected",
    [
        ("sync", PlanCapabilities(), None),
        ("async", PlanCapabilities(), "none"),
        ("rtc", PlanCapabilities(continuity_methods=("none", "rtc")), "rtc"),
        (
            "async",
            PlanCapabilities(
                kinds=("bspline",),
                continuity_methods=("none", "phase"),
                default_continuity="phase",
            ),
            "phase",
        ),
    ],
)
def test_local_entry_point_selects_executor_and_strategy_defaults(
    tmp_path, monkeypatch, mode, capabilities, expected
):
    checkpoint = tmp_path / "policy"
    checkpoint.mkdir()
    (checkpoint / "policy.json").write_text("{}")
    native = SimpleNamespace(
        names=["q"],
        fps=30 if mode == "sync" else 10,
        plan_capabilities=capabilities,
        reset=Mock(),
    )
    monkeypatch.setattr("alohamini.learning.policy.NativePolicy", Mock(return_value=native))
    selected = []

    def evaluate(**options):
        policy = options["policy_factory"]()
        if mode == "sync":
            assert policy is native
        else:
            assert isinstance(policy, TrajectoryExecutor)
            assert policy.fps == 30 and policy.policy.fps == 10
            assert policy.execution["continuity"] == expected
            policy.close()
        selected.append(policy)

    monkeypatch.setattr("alohamini.apps.evaluation.evaluate", evaluate)
    assert (
        main(
            [
                "evaluate",
                "--host=127.0.0.1",
                "--robot_model=alohamini2pro",
                f"--policy.path={checkpoint}",
                "--device=cpu",
                f"--inference.type={mode}",
            ]
        )
        == 0
    )
    assert len(selected) == 1


def test_local_sync_is_default_and_async_flags_are_fully_consumed():
    assert parse([]) == ExecutionConfig()
    options = dict(inference_type="async", inference_replan_interval_s=0.3, task="pick")
    config = pop_inference_options(options)
    assert config.enabled and config.replan_interval_s == 0.3
    assert options == {"task": "pick"}


def test_async_can_disable_bsp_phase_and_rtc_never_falls_back():
    capabilities = PlanCapabilities(
        kinds=("bspline",), continuity_methods=("none", "phase"), default_continuity="phase"
    )
    explicit = parse(["--inference.type=async", "--inference.continuity=none"])
    assert capabilities.resolve(explicit).continuity == "none"
    with pytest.raises(ValueError, match="does not support 'rtc'"):
        capabilities.resolve(parse(["--inference.type=rtc"]))


def test_remote_sync_rejected_before_loading_the_model(monkeypatch, capsys):
    policy = Mock(side_effect=AssertionError("must not load a model"))
    monkeypatch.setattr("alohamini.remote.server.NativePlanPolicy", policy)
    assert remote_main(["serve", "--policy.path=/model", "--inference.type=sync"]) == 1
    policy.assert_not_called()
    assert "requires --inference.type=async or rtc" in capsys.readouterr().err


@pytest.mark.parametrize("continuity", ["none", "phase"])
def test_remote_resolves_strategy_defaults_before_advertising_execution(monkeypatch, continuity):
    from alohamini.remote.server import NativePlanPolicy

    capabilities = PlanCapabilities(
        kinds=("bspline",),
        continuity_methods=("none", "phase"),
        default_continuity=continuity,
    )
    policy = SimpleNamespace(
        names=["q"],
        fps=30,
        manifest={"cameras": []},
        fixed_current=[],
        fixed_dimensions=None,
        fixed_deployment={},
        plan_capabilities=capabilities,
    )
    monkeypatch.setattr(
        "alohamini.learning.checkpoint.read_checkpoint", lambda _: (None, {"kind": "bspline"})
    )
    monkeypatch.setattr("alohamini.learning.policy.NativePolicy", Mock(return_value=policy))
    backend = NativePlanPolicy("/model", execution=parse([], remote=True))
    assert backend.execution["enabled"]
    assert backend.execution["continuity"] == continuity

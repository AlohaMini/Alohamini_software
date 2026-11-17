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


def test_remote_sync_selection_is_passed_to_the_checkpoint_capability_check(monkeypatch):
    backend = SimpleNamespace()
    policy, serve = Mock(return_value=backend), Mock()
    monkeypatch.setattr("alohamini.remote.server.NativePlanPolicy", policy)
    monkeypatch.setattr("alohamini.remote.server.serve", serve)
    assert remote_main(["serve", "--policy.path=/model", "--inference.type=sync"]) == 0
    assert policy.call_args.kwargs["execution"] == ExecutionConfig()
    assert serve.call_args.args[1] is backend


@pytest.mark.parametrize("continuity", ["none", "phase"])
def test_remote_resolves_strategy_defaults_before_advertising_execution(monkeypatch, continuity):
    from alohamini.policies.registry import algorithm
    from alohamini.remote.server import NativePlanPolicy

    algorithm("bspline")  # Resolve lazy adapter imports before replacing checkpoint reads.
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


@pytest.mark.parametrize("mode", [None, "sync", "async"])
def test_local_tcp_checkpoint_automatically_uses_protection_in_every_mode(
    tmp_path, monkeypatch, mode
):
    checkpoint = tmp_path / "policy"
    checkpoint.mkdir()
    (checkpoint / "policy.json").write_text("{}")
    native = SimpleNamespace(
        names=["q"],
        fps=10,
        task="pick",
        tcp_reference_actions=True,
        plan_capabilities=PlanCapabilities(kinds=("tcp",), default_replan_interval_s=0.25),
    )
    monkeypatch.setattr("alohamini.learning.policy.NativePolicy", Mock(return_value=native))
    selected = []

    def evaluate(**options):
        executor = options["policy_factory"]()
        assert isinstance(executor, TrajectoryExecutor)
        assert executor.fps == 30
        assert executor.settings.enabled == (mode == "async")
        assert executor.settings.tcp_tracking["max_tcp_linear_velocity"] == 0.2
        assert executor.settings.replan_interval_s == (0.25 if mode == "async" else None)
        executor.close()
        selected.append(executor)

    monkeypatch.setattr("alohamini.apps.evaluation.evaluate", evaluate)
    args = [
        "evaluate",
        "--host=127.0.0.1",
        "--robot_model=alohamini2pro",
        f"--policy.path={checkpoint}",
        "--device=cpu",
    ]
    if mode is not None:
        args.append(f"--inference.type={mode}")
    assert main(args) == 0
    assert len(selected) == 1


def test_sync_accepts_tcp_limits_but_rejects_them_for_a_joint_checkpoint():
    from alohamini.learning.trajectory_execution import prepare_execution

    config = parse(['--inference.tcp_tracking={"max_tcp_linear_velocity":0.05}'])
    assert not config.enabled
    assert config.tcp_tracking["max_tcp_linear_velocity"] == 0.05
    with pytest.raises(ValueError, match="require a TCP action checkpoint"):
        prepare_execution(SimpleNamespace(tcp_reference_actions=False, fps=30), config)


def test_tcp_input_alone_does_not_enable_tcp_execution():
    from test_kinematic_fields import descriptor

    from alohamini.datasets.field_transforms import FieldTransforms
    from alohamini.learning.data_fields import FieldProcessor
    from alohamini.learning.policy import NativePolicy
    from alohamini.learning.trajectory_execution import prepare_execution

    policy = NativePolicy.__new__(NativePolicy)
    fields = FieldTransforms(
        dict(
            version=1,
            state_representation="tcp",
            action_representation="recorded",
            kinematic_fields=descriptor(),
        )
    )
    policy.processor = FieldProcessor(SimpleNamespace(), fields, "cpu")
    policy.fps = 30
    assert not policy.tcp_reference_actions
    assert prepare_execution(policy, control_fps=30) is policy


@pytest.mark.parametrize("tcp", [False, True])
def test_remote_sync_validates_checkpoint_and_preserves_execution_window(monkeypatch, tcp):
    from test_tcp_execution import plan_for

    from alohamini.policies.registry import algorithm
    from alohamini.remote.server import NativePlanPolicy

    algorithm("act")  # Do not let lazy adapters retain the mocked checkpoint reader.
    plan = plan_for(count=10)
    policy = SimpleNamespace(
        names=plan.names,
        fps=30,
        manifest={"cameras": []},
        fixed_current=[],
        fixed_dimensions=None,
        fixed_deployment={},
        config=SimpleNamespace(n_action_steps=2, chunk_size=10),
        predict_plan=Mock(return_value=plan),
        plan_capabilities=PlanCapabilities(kinds=("tcp",) if tcp else ("dense",)),
    )
    monkeypatch.setattr(
        "alohamini.learning.checkpoint.read_checkpoint", lambda _: (None, {"kind": "act"})
    )
    monkeypatch.setattr("alohamini.learning.policy.NativePolicy", Mock(return_value=policy))
    if not tcp:
        with pytest.raises(ValueError, match="Remote sync.*TCP action checkpoint"):
            NativePlanPolicy("/model", execution=parse(["--inference.type=sync"], remote=True))
        return
    backend = NativePlanPolicy("/model", execution=parse(["--inference.type=sync"], remote=True))
    assert not backend.execution["enabled"]
    assert backend.execution["tcp_tracking"]["max_joint_velocity"] == 1.5
    backend._initialized = True
    result = backend.predict_plan(None)
    assert result.kind == "tcp" and len(result.actions) == 2


def test_evaluator_auto_protects_tcp_factories_and_closes_before_leaving_the_hardware_boundary(
    monkeypatch,
):
    from alohamini.apps.evaluation import evaluate
    from alohamini.learning.trajectory_execution import prepare_execution

    native = SimpleNamespace(
        tcp_reference_actions=True,
        fps=30,
        names=["q"],
        robot_metadata={},
        plan_capabilities=PlanCapabilities(kinds=("tcp",)),
    )
    selected = []

    def prepare(*args, **kwargs):
        value = prepare_execution(*args, **kwargs)
        selected.append(value)
        return value

    monkeypatch.setattr("alohamini.learning.trajectory_execution.prepare_execution", prepare)
    client = Mock(side_effect=RuntimeError("hardware boundary"))
    monkeypatch.setattr("alohamini.apps.evaluation.HostClient", client)
    with pytest.raises(RuntimeError, match="hardware boundary"):
        evaluate("pi", "alohamini2pro", policy_factory=lambda: native)
    assert len(selected) == 1 and isinstance(selected[0], TrajectoryExecutor)
    assert not selected[0].settings.enabled and selected[0].closed


def test_missing_tcp_dependencies_fail_before_opening_a_host_for_the_default_sync_mode(monkeypatch):
    from alohamini.apps.evaluation import evaluate

    native = SimpleNamespace(
        tcp_reference_actions=True,
        fps=30,
        names=["q"],
        robot_metadata={},
        plan_capabilities=PlanCapabilities(kinds=("tcp",)),
    )
    monkeypatch.setattr(
        "alohamini.tcp_execution.warmup_backend",
        Mock(side_effect=ImportError("TCP kinematics dependencies missing")),
    )
    client = Mock()
    monkeypatch.setattr("alohamini.apps.evaluation.HostClient", client)
    with pytest.raises(ImportError, match="TCP kinematics dependencies missing"):
        evaluate("pi", "alohamini2pro", policy_factory=lambda: native)
    client.assert_not_called()


def test_custom_policy_example_preserves_tcp_capabilities_and_current_robot_binding(monkeypatch):
    import importlib.util
    from pathlib import Path

    from alohamini.learning.trajectory_execution import prepare_execution

    path = Path(__file__).resolve().parents[1] / "examples/learning/custom_policy.py"
    spec = importlib.util.spec_from_file_location("custom_policy_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    native = SimpleNamespace(
        tcp_reference_actions=True,
        fps=30,
        names=["q"],
        robot_metadata={"binding": "old"},
        plan_capabilities=PlanCapabilities(kinds=("tcp",)),
    )
    monkeypatch.setattr(example, "NativePolicy", Mock(return_value=native))
    policy = example.RecedingHorizonPolicy("/model", device="cpu")
    executor = prepare_execution(policy, control_fps=30)
    try:
        assert isinstance(executor, TrajectoryExecutor) and not executor.settings.enabled
        native.robot_metadata = {"binding": "new"}
        assert executor.robot_metadata == {"binding": "new"}
    finally:
        executor.close()

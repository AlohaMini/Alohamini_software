"""Explicit old-data units bind to live calibration without rewriting provenance."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_calibration_transfer import JOINT, policy_for
from test_evaluation import EvaluationClient
from test_native_learning import model_options, samples
from test_native_learning import recording as recording
from test_replay import Clock, replay_snapshot

from alohamini.apps.evaluation import run_evaluation
from alohamini.datasets.record import dataset_features, state_names
from alohamini.learning.calibration import legacy_normalized_binding
from alohamini.learning.checkpoint import write_checkpoint_metadata
from alohamini.learning.policy import NativePolicy, make_policy, save_checkpoint


def source():
    return {
        "features": dataset_features("alohamini2pro"),
        "training_only": {"source": "original_lerobot_v3", "calibration": "unavailable"},
    }


def test_binding_preserves_unknown_source_and_validates_current_contract():
    recorded = source()
    before = deepcopy(recorded)
    snapshot = replay_snapshot()
    metadata, report = legacy_normalized_binding(recorded, "alohamini2pro", snapshot)
    assert recorded == before and "robot_metadata" not in recorded
    assert report["trained_metadata"] is None
    assert report["training_provenance"] == before["training_only"]
    assert report["host_metadata"] == metadata == snapshot.payload["_robot_metadata"]
    metadata["motors"][JOINT]["range_min"] += 1
    assert metadata != snapshot.payload["_robot_metadata"]


@pytest.mark.parametrize(
    "field,value",
    [("normalization", "degrees"), ("range_min", 4095), ("homing_offset", 3000), ("id", 99)],
)
def test_binding_rejects_incompatible_host(field, value):
    snapshot = replay_snapshot()
    snapshot.payload["_robot_metadata"]["motors"][JOINT][field] = value
    with pytest.raises(ValueError):
        legacy_normalized_binding(source(), "alohamini2pro", snapshot)


def test_binding_requires_gripper_units_and_exact_action_names():
    snapshot = replay_snapshot()
    snapshot.payload["_robot_metadata"]["motors"]["arm_left_gripper"]["normalization"] = (
        "range_m100_100"
    )
    with pytest.raises(ValueError, match="range_0_100"):
        legacy_normalized_binding(source(), "alohamini2pro", snapshot)
    recorded = source()
    recorded["features"]["action"]["names"].reverse()
    with pytest.raises(ValueError, match="names/order"):
        legacy_normalized_binding(recorded, "alohamini2pro", replay_snapshot())


def test_legacy_checkpoint_load_binding_fixed_lift_and_provenance(recording, tmp_path):
    data = samples(recording, episodes=[0])
    checkpoint = tmp_path / "legacy"
    model = make_policy("act", model_options(state=False))
    save_checkpoint(checkpoint, model, data.statistics(), data, training={})
    manifest = json.loads((checkpoint / "policy.json").read_text())
    manifest["source_info"].pop("robot_metadata")
    manifest["source_info"]["training_only"] = source()["training_only"]
    write_checkpoint_metadata(checkpoint, manifest)
    before = (checkpoint / "policy.json").read_bytes()
    for mode in ("strict", "normalized"):
        with pytest.raises(ValueError, match="offline training only"):
            NativePolicy(checkpoint, calibration_mode=mode)
    with pytest.raises(ValueError, match="explicit robot_model"):
        NativePolicy(checkpoint, calibration_mode="legacy_normalized")
    policy = NativePolicy(
        checkpoint,
        calibration_mode="legacy_normalized",
        robot_model="alohamini2pro",
        fixed_current="lift_axis",
        n_action_steps=1,
        temporal_ensemble_coeff=0.01,
    )
    snapshot = replay_snapshot()
    with pytest.raises(ValueError, match="must bind"):
        policy._check_calibration(snapshot)
    report = policy.bind_robot(snapshot)
    assert report["mode"] == "legacy_normalized"
    policy.bind_fixed_current(snapshot, "test-only")
    assert policy.fixed_deployment["current_capture"]
    assert policy.bind_robot(snapshot) is None
    assert policy.source == manifest["source_info"]
    assert before == (checkpoint / "policy.json").read_bytes()
    snapshot.payload["_robot_metadata"]["motors"][JOINT]["range_max"] += 1
    with pytest.raises(ValueError, match="range_max"):
        policy.bind_robot(snapshot)


def test_evaluator_binds_legacy_before_sending_actions(monkeypatch, tmp_path):
    clock = Clock()
    client = EvaluationClient(clock)
    policy = policy_for(client.state.payload["_robot_metadata"], "legacy_normalized")
    policy.source = source()
    policy.manifest = {"state": "none", "cameras": []}
    policy.selection = None
    policy.robot_metadata = {"robot_model": "alohamini2pro", "binding_pending": True}
    policy.reset = Mock()

    def select(snapshot):
        policy._check_calibration(snapshot)
        return {n: snapshot.payload[n] for n in state_names("alohamini2pro")}

    policy.select_action = select
    monkeypatch.setattr("time.monotonic", lambda: clock.now)
    monkeypatch.setattr("time.sleep", clock.sleep)
    monkeypatch.setattr("alohamini.apps.evaluation.stop_owned_robot", Mock())
    monkeypatch.setattr(
        "alohamini.apps.evaluation.WorkspacePaths", lambda: SimpleNamespace(logs=tmp_path)
    )
    assert run_evaluation(client, policy, "alohamini2pro", duration_s=0.1) >= 2
    reports = list(tmp_path.rglob("calibration-transfer-*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["mode"] == "legacy_normalized"


def test_cli_routes_legacy_units_and_explicit_model(tmp_path, monkeypatch):
    from alohamini.cli import main

    (tmp_path / "policy.json").write_text("{}")
    constructor = Mock(return_value=SimpleNamespace(fps=30))
    monkeypatch.setattr("alohamini.learning.policy.NativePolicy", constructor)
    monkeypatch.setattr("alohamini.apps.evaluation.evaluate", lambda **kw: kw["policy_factory"]())
    assert (
        main(
            [
                "evaluate",
                "--host",
                "test-only",
                "--robot_model",
                "alohamini2pro",
                "--policy.path",
                str(tmp_path),
                "--calibration-mode",
                "legacy_normalized",
                "--fixed-current",
                "lift_axis",
            ]
        )
        == 0
    )
    assert constructor.call_args.kwargs["calibration_mode"] == "legacy_normalized"
    assert constructor.call_args.kwargs["robot_model"] == "alohamini2pro"
    assert constructor.call_args.kwargs["fixed_current"] == ["lift_axis"]

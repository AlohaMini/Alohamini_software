"""New model coordinates are portable while device offsets stay pinned at runtime."""

from copy import deepcopy
from dataclasses import asdict
from unittest.mock import Mock

import pytest
from test_calibration_transfer import policy_for
from test_replay import replay_snapshot

from alohamini.calibration.reference import PROTOCOL, side_calibration
from alohamini.learning.calibration import reference_binding
from alohamini.model.joint_spec import ModelJointSpecification


def standard_robot(raw, *, version=None):
    snapshot = replay_snapshot()
    metadata = snapshot.payload["_robot_metadata"]
    spec = ModelJointSpecification.from_model(version=version)
    arms, references, captures = {}, {}, {}
    for side in ("left", "right"):
        motors = tuple(
            m for m in spec.model.actuators if m.bus == side and m.name.startswith("arm_")
        )
        values, ticks, mapping = side_calibration(
            spec, side, motors, dict.fromkeys((m.name for m in motors), raw), leader=False
        )
        for name, value in values.items():
            metadata["motors"][name].update(asdict(value))
        arms[side] = mapping
        references.update(ticks)
        captures.update(
            {
                f"arm_{side}_{name}": entry["capture_tick"]
                for name, entry in mapping["joints"].items()
            }
        )
    metadata["reference_calibration"] = dict(
        schema_version=1,
        protocol=PROTOCOL,
        robot_model=spec.model.model_id,
        role="robot",
        specification_sha256=spec.signature,
        reference_pose=spec.descriptor["reference_pose"],
        capture_pose=spec.descriptor["calibration_pose"]["name"],
        boundary_policy=spec.descriptor["boundary_policy"],
        arms=arms,
        reference_ticks=references,
        capture_ticks=captures,
    )
    return snapshot


def test_old_checkpoint_remains_readable_but_is_not_bound_to_expanded_wrist_ranges():
    trained = standard_robot(1300, version=1).payload["_robot_metadata"]
    original = deepcopy(trained)
    reference_binding(trained, standard_robot(4095, version=1))
    with pytest.raises(ValueError, match="different model coordinate specifications"):
        reference_binding(trained, standard_robot(4095))
    assert trained == original


def test_default_binding_accepts_device_offsets_and_the_shifted_encoder_branch():
    trained = standard_robot(1300).payload["_robot_metadata"]
    snapshot = standard_robot(4095)
    policy = policy_for(trained, "strict")
    before = deepcopy(trained)
    columns = deepcopy(policy.selection.columns)
    report = policy.bind_robot(snapshot)
    assert report["mode"] == "model_coordinates"
    assert report["differences"]
    assert policy.robot_metadata == snapshot.payload["_robot_metadata"]
    assert policy.source["robot_metadata"] == before
    assert trained == before
    assert policy.selection.columns == columns
    assert policy.bind_robot(deepcopy(snapshot)) is None
    with pytest.raises(ValueError, match="homing_offset|range_min"):
        policy.bind_robot(standard_robot(900))


@pytest.mark.parametrize("change", ["signature", "direction", "range", "capture", "lift"])
def test_invalid_profiles_or_changed_non_arm_calibration_are_rejected(change):
    trained = standard_robot(1300).payload["_robot_metadata"]
    snapshot = standard_robot(4095)
    metadata = snapshot.payload["_robot_metadata"]
    profile = metadata["reference_calibration"]
    if change == "signature":
        profile["specification_sha256"] = "0" * 64
    elif change == "direction":
        profile["arms"]["left"]["joints"]["shoulder_pan"]["sign"] *= -1
    elif change == "range":
        metadata["motors"]["arm_left_shoulder_pan"]["range_max"] -= 1
    elif change == "capture":
        profile["capture_ticks"]["arm_left_shoulder_pan"] += 1
    else:
        metadata["motors"]["lift_axis"]["homing_offset"] += 1
    policy = policy_for(trained, "strict")
    with pytest.raises(ValueError):
        policy.bind_robot(snapshot)
    assert not policy._robot_bound
    assert policy.robot_metadata == trained


def test_raw_readings_without_reference_profiles_are_not_treated_as_standard_data():
    trained = standard_robot(1300).payload["_robot_metadata"]
    trained.pop("reference_calibration")
    with pytest.raises(ValueError, match="reference profiles"):
        reference_binding(trained, standard_robot(4095))


def test_direct_inference_check_binds_new_model_coordinates_once():
    trained = standard_robot(1300).payload["_robot_metadata"]
    snapshot = standard_robot(4095)
    policy = policy_for(trained, "strict")
    policy._check_calibration(snapshot)
    assert policy._robot_bound
    policy._check_calibration(deepcopy(snapshot))


def test_new_checkpoint_requires_a_live_reference_even_if_raw_calibration_matches():
    trained = standard_robot(1300).payload["_robot_metadata"]
    snapshot = standard_robot(1300)
    snapshot.payload["_robot_metadata"].pop("reference_calibration")
    policy = policy_for(trained, "strict")
    with pytest.raises(ValueError, match="reference profiles"):
        policy._check_calibration(snapshot)
    assert not policy._robot_bound


def test_bound_reference_change_is_rejected_without_raw_motor_changes():
    policy = policy_for(standard_robot(1300).payload["_robot_metadata"], "strict")
    snapshot = standard_robot(4095)
    policy.bind_robot(snapshot)
    snapshot.payload["_robot_metadata"]["reference_calibration"]["capture_pose"] = "unknown"
    with pytest.raises(ValueError, match="reference calibration changed"):
        policy._check_calibration(snapshot)


def test_recorded_replay_accepts_equivalent_new_coordinates_without_changing_actions():
    from pathlib import Path

    import numpy as np

    from alohamini.apps.replay import ReplayEpisode, check_coordinates
    from alohamini.datasets.record import state_names

    trained = standard_robot(1300).payload["_robot_metadata"]
    actions = np.zeros((2, 18))
    episode = ReplayEpisode(
        Path("/test-only"), 0, 30, trained, tuple(state_names("alohamini2pro")), actions
    )
    check_coordinates(episode, standard_robot(4095))
    assert episode.metadata == trained
    np.testing.assert_array_equal(episode.actions, np.zeros((2, 18)))


def test_tcp_field_transforms_and_spirit_keep_the_frozen_training_coordinates():
    from types import SimpleNamespace

    import numpy as np

    from alohamini.datasets.field_transforms import FieldTransforms
    from alohamini.datasets.kinematic_fields import KinematicFields, descriptor_from_reference_pose
    from alohamini.policies.registry import algorithm

    policy = policy_for(standard_robot(1300).payload["_robot_metadata"], "strict")
    snapshot = standard_robot(4095)
    descriptor = descriptor_from_reference_pose(policy.source, "upright_forward")
    policy.field_transform = FieldTransforms(
        dict(
            state_representation="tcp",
            action_representation="tcp_delta",
            kinematic_fields=descriptor,
        )
    )
    policy.algorithm = algorithm("spirit")
    policy.config = SimpleNamespace(action_representation="eef_delta", kinematic_fields=descriptor)
    policy.manifest["state_feature"] = policy.selection.feature
    original = deepcopy(policy.field_transform.configuration)
    values = np.linspace(-30, 30, len(descriptor["source_names"]))
    expected = policy.field_transform.geometry.absolute(values)
    assert policy.bind_robot(snapshot)["mode"] == "model_coordinates"
    target = descriptor_from_reference_pose(
        {**policy.source, "robot_metadata": policy.robot_metadata}, "upright_forward"
    )
    np.testing.assert_allclose(KinematicFields(target).absolute(values), expected, atol=1e-12)
    assert policy.field_transform.configuration == original


def test_evaluator_binds_model_coordinates_without_requesting_a_transfer_mode(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from test_evaluation import EvaluationClient
    from test_replay import Clock

    from alohamini.apps.evaluation import run_evaluation
    from alohamini.datasets.record import state_names

    clock = Clock()
    client = EvaluationClient(clock)
    client.state.payload["_robot_metadata"] = standard_robot(4095).payload["_robot_metadata"]
    policy = policy_for(standard_robot(1300).payload["_robot_metadata"], "strict")
    policy.manifest["cameras"] = []
    policy.reset = Mock()
    seen = []

    def select(snapshot):
        policy._check_calibration(snapshot)
        seen.append(snapshot.payload["arm_left_shoulder_pan.pos"])
        return {name: snapshot.payload[name] for name in state_names("alohamini2pro")}

    policy.select_action = select
    monkeypatch.setattr("time.monotonic", lambda: clock.now)
    monkeypatch.setattr("time.sleep", clock.sleep)
    monkeypatch.setattr("alohamini.apps.evaluation.stop_owned_robot", Mock())
    monkeypatch.setattr(
        "alohamini.apps.evaluation.WorkspacePaths", lambda: SimpleNamespace(logs=tmp_path)
    )
    assert run_evaluation(client, policy, "alohamini2pro", duration_s=0.1) >= 2
    assert seen == [0.0] * len(seen)

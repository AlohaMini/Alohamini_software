import copy
from dataclasses import replace
from unittest.mock import Mock

import pytest
from test_evaluation import EvaluationClient
from test_replay import Clock, replay_snapshot

from alohamini.apps.dagger import DAggerConfig, DAggerKeyboard, ManualHandover
from alohamini.apps.evaluation import run_evaluation
from alohamini.datasets.record import state_names


class Keys:
    def __init__(self):
        self.pending = []

    def events(self):
        events, self.pending = self.pending, []
        return events


class Leader:
    def __init__(self, snapshot):
        self.pose = {
            key: value
            for key, value in snapshot.payload.items()
            if key.startswith("arm_") and key.endswith(".pos")
        }
        self.held = []
        self.releases = 0

    def read_handover(self, units):
        assert set(units) == set(self.pose)
        return dict(self.pose)

    def hold_joint(self, key, *, output_limit):
        self.held.append((key, output_limit))

    def release_holds(self):
        self.releases += 1


@pytest.fixture
def handover():
    snapshot = replay_snapshot()
    leader, keys, policy, dataset = Leader(snapshot), Keys(), Mock(), Mock()
    control = ManualHandover(leader, keys, DAggerConfig(settle_time_s=0.01))
    yield snapshot, leader, keys, policy, dataset, control
    control.close()


def press(handover, key, now):
    snapshot, _, keys, policy, dataset, control = handover
    keys.pending = [key]
    control.handle_events(snapshot, policy, dataset, now=now)


def align(handover):
    snapshot, leader, _, _, _, control = handover
    press(handover, "space", 0)
    press(handover, "tab", 0.1)
    for i in range(len(leader.pose) + 3):
        action = control.action(snapshot, now=0.2 + i * 0.1)
        assert all(action[key] == snapshot.payload[key] for key in leader.pose)
        assert action["x.vel"] == action["y.vel"] == action["theta.vel"] == 0
    assert control.phase == "ready"


def test_manual_alignment_release_record_save_and_fresh_policy(handover):
    snapshot, leader, _, policy, dataset, control = handover
    align(handover)
    assert len(leader.held) == len(leader.pose)
    assert len({key for key, _ in leader.held}) == len(leader.pose)
    dataset.begin_episode.assert_not_called()
    before_release = leader.releases
    press(handover, "tab", 2)
    assert leader.releases == before_release + 1
    assert control.recording and control.phase == "human"
    first = control.action(snapshot, now=2)
    assert first == control.hold_targets(snapshot)
    dataset.begin_episode.assert_called_once()
    joint = "arm_left_shoulder_pan.pos"
    leader.pose[joint] += 15
    action = control.action(snapshot, now=2.05)
    assert 0 < action[joint] - first[joint] <= 60 * 0.05 / control._scale(joint, snapshot)
    press(handover, "tab", 2.1)
    assert control.phase == "paused" and not control.recording and control.save_pending
    control.finish_save(dataset)
    control._save_future.result(timeout=2)
    control.finish_save(dataset)
    dataset.save_episode.assert_called_once()
    press(handover, "space", 2.2)
    assert control.phase == "policy"
    assert policy.reset.call_count == 3  # Pause, correction end, policy resume.


def test_moving_or_unmatched_joint_is_not_enabled(handover):
    snapshot, leader, _, _, _, control = handover
    press(handover, "space", 0)
    press(handover, "tab", 0.1)
    joint = "arm_left_shoulder_pan.pos"
    leader.pose[joint] += 30
    for i in range(20):
        control.action(snapshot, now=0.2 + i * 0.05)
    assert joint not in dict(leader.held)
    assert control.phase == "aligning"
    assert len(leader.held) == len(leader.pose) - 1
    # A second Tab during incomplete alignment cancels rather than bypasses it.
    press(handover, "tab", 2)
    assert control.phase == "paused"


def test_release_shift_and_hold_slip_do_not_reach_follower(handover):
    snapshot, leader, _, _, _, control = handover
    align(handover)
    press(handover, "tab", 2)
    leader.pose["arm_left_shoulder_pan.pos"] += 30
    with pytest.raises(RuntimeError, match="moved during release"):
        control.action(snapshot, now=2.1)


def test_feedback_recovery_requires_explicit_resume_and_relatched_hold(handover):
    snapshot, leader, _, policy, dataset, control = handover
    align(handover)
    control.feedback_lost(snapshot, policy, dataset)
    assert control.phase == "paused"
    snapshot.payload["arm_left_shoulder_pan.pos"] += 10
    assert (
        control.action(snapshot, now=3)["arm_left_shoulder_pan.pos"]
        == (snapshot.payload["arm_left_shoulder_pan.pos"])
    )
    assert leader.releases > 0


def test_alignment_times_out_without_auto_starting_correction(handover):
    snapshot, _, _, _, dataset, control = handover
    press(handover, "space", 0)
    press(handover, "tab", 0.1)
    with pytest.raises(TimeoutError, match="alignment timed out"):
        control.action(snapshot, now=61)
    dataset.begin_episode.assert_not_called()


def test_disconnect_discards_takeover_keys_but_accepts_exit(handover):
    _, _, keys, _, _, control = handover
    keys.pending = ["tab"]
    control.poll_disconnected()
    assert control.phase == "policy"
    keys.pending = ["quit"]
    control.poll_disconnected()
    assert control.stopped


def test_confirmation_rechecks_alignment_before_releasing(handover):
    _, leader, _, _, dataset, control = handover
    align(handover)
    leader.pose["arm_left_shoulder_pan.pos"] += 30
    press(handover, "tab", 2)
    assert control.phase == "paused"
    dataset.begin_episode.assert_not_called()


def test_pause_resume_discards_an_inflight_async_plan(handover):
    from types import SimpleNamespace

    from test_trajectory import Worker

    from alohamini.learning.trajectory_execution import TrajectoryExecutor
    from alohamini.trajectory import DenseActionPlan
    from alohamini.trajectory_execution import ExecutionConfig, PlanCapabilities

    snapshot, leader, keys, _, dataset, _ = handover
    clock = SimpleNamespace(now=snapshot.request_started_s)
    worker = Worker()
    policy = SimpleNamespace(
        fps=30,
        names=["q"],
        task="pick",
        reset=Mock(),
        plan_capabilities=PlanCapabilities(),
        predict_plan=Mock(return_value=DenseActionPlan(("q",), [[99]], 1 / 30)),
    )
    executor = TrajectoryExecutor(
        policy, ExecutionConfig(enabled=True), clock=lambda: clock.now, executor=worker
    )
    control = ManualHandover(leader, keys)
    try:
        assert executor.select_action(snapshot) is None
        generation = executor.generation
        keys.pending = ["space"]
        control.handle_events(snapshot, executor, dataset, now=0)
        assert executor.generation > generation
        worker.complete()
        keys.pending = ["space"]
        control.handle_events(snapshot, executor, dataset, now=0.1)
        assert control.phase == "policy"
        assert executor.select_action(snapshot) is None
        assert executor.plan is None
    finally:
        control.close()
        executor.close()


def test_keyboard_repeats_and_queued_keys_cannot_skip_alignment():
    keys = DAggerKeyboard()
    keys.read = lambda: set()
    keys.update("space", True)
    keys.update("space", True)
    keys.update("tab", True)
    assert keys.events() == ["space"]
    keys.update("space", True)
    assert keys.events() == []
    keys.update("space", False)
    keys.update("tab", False)
    keys.update("tab", True)
    assert keys.events() == ["tab"]


def test_pause_during_sync_inference_discards_its_action(monkeypatch):
    clock = Clock()
    client = EvaluationClient(clock)
    leader, keys, policy = Leader(client.state), Keys(), Mock()
    policy.robot_metadata = copy.deepcopy(client.state.payload["_robot_metadata"])
    control = ManualHandover(leader, keys)

    def select(snapshot):
        clock.sleep(0.15)
        keys.pending = ["space"]
        client.state.payload["arm_left_shoulder_pan.pos"] = 12
        return {key: 50 for key in state_names("alohamini2pro")}

    policy.select_action.side_effect = select
    monkeypatch.setattr("alohamini.apps.evaluation.time.monotonic", lambda: clock.now)
    monkeypatch.setattr("alohamini.apps.evaluation.time.sleep", clock.sleep)
    monkeypatch.setattr("alohamini.apps.evaluation.stop_owned_robot", Mock())
    count = run_evaluation(client, policy, "alohamini2pro", duration_s=0.3, intervention=control)
    assert count > 0
    assert all(action["arm_left_shoulder_pan.pos"] == 12 for _, action, _ in client.sent)
    assert control.phase == "paused"
    policy.select_action.assert_called_once()


@pytest.mark.parametrize(
    "key,value",
    [("hold_output_limit", 1.1), ("settle_time_s", 0), ("alignment_tolerance_deg", float("nan"))],
)
def test_invalid_configuration_rejected(key, value):
    with pytest.raises(ValueError):
        replace(DAggerConfig(), **{key: value})


def test_cli_dagger_is_opt_in_and_validated_before_policy_loading(monkeypatch):
    from alohamini.cli import main

    run = Mock()
    monkeypatch.setattr("alohamini.apps.evaluation.evaluate", run)
    assert (
        main(
            [
                "evaluate",
                "--host",
                "pi",
                "--robot_model",
                "alohamini2pro",
                "--policy",
                "examples:policy",
                "--dagger",
                "--dagger.hold_output_limit",
                "0.1",
                "--left_port",
                "/dev/left",
            ]
        )
        == 0
    )
    options = run.call_args.kwargs["dagger"]
    assert options.hold_output_limit == 0.1 and options.left_port == "/dev/left"
    run.reset_mock()
    assert (
        main(
            [
                "evaluate",
                "--host",
                "pi",
                "--robot_model",
                "alohamini2pro",
                "--policy",
                "examples:policy",
                "--dagger.hold_output_limit",
                "0.1",
            ]
        )
        == 1
    )
    run.assert_not_called()

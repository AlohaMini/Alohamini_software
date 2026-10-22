import copy
import unittest
from unittest.mock import patch

from alohamini.errors import ProtocolError
from alohamini.protocol import HostSnapshot, command_target_keys
from alohamini.remote.agent import ActionBuffer, run_session
from alohamini.schema import CommandIdentity

MODEL = "alohamini2pro"
JPEG = b"\xff\xd8encoded image\xff\xd9"


class ActionBufferTests(unittest.TestCase):
    def setUp(self):
        self.buffer = ActionBuffer("session", {"epoch": 1}, ["joint.pos"], 10, 1.0, 8)

    def actions(self, request_id=1, tick=0, values=None):
        return dict(
            id=request_id, session="session", context={"epoch": 1}, tick=tick,
            fps=10, names=["joint.pos"], actions=values or [[i] for i in range(6)],
            inference_s=0.1,
        )

    def test_late_chunk_skips_elapsed_prefix_and_never_replays_targets(self):
        self.buffer.remember(1, 0, 100.0)
        count = self.buffer.accept(self.actions(), now=100.31, tick=3)
        self.assertEqual(count, 3)
        self.assertEqual(self.buffer.skipped, 3)
        self.assertAlmostEqual(self.buffer.last_round_trip_s, 0.31)
        self.assertIsNone(self.buffer.pop(2, 100.31))
        self.assertEqual(self.buffer.pop(3, 100.31), {"joint.pos": 3.0})
        self.assertIsNone(self.buffer.pop(3, 100.32))
        self.assertEqual(self.buffer.pop(5, 100.51), {"joint.pos": 5.0})
        self.assertIsNone(self.buffer.pop(4, 100.52))

    def test_short_new_plan_supersedes_the_entire_old_tail(self):
        self.buffer.remember(1, 0, 100.0)
        self.buffer.accept(self.actions(), now=100.0, tick=0)
        self.buffer.remember(2, 1, 100.1)
        self.buffer.accept(self.actions(2, 1, [[10], [11]]), now=100.2, tick=2)
        self.assertEqual(self.buffer.pop(2, 100.2), {"joint.pos": 11.0})
        self.assertIsNone(self.buffer.pop(3, 100.3))
        self.assertIsNone(self.buffer.pop(5, 100.5))

    def test_age_boundary_and_missed_horizon_cannot_supply_actions(self):
        for now, tick in ((101.0, 0), (100.2, 6), (99.9, 0)):
            with self.subTest(now=now, tick=tick):
                self.setUp()
                self.buffer.remember(1, 0, 100.0)
                self.assertEqual(self.buffer.accept(self.actions(), now=now, tick=tick), 0)
                self.assertIsNone(self.buffer.pop(tick, now))
                self.assertEqual(self.buffer.expired, 1)

    def test_buffered_target_expires_at_observation_age_limit(self):
        self.buffer.remember(1, 0, 100.0)
        self.buffer.accept(self.actions(), now=100.1, tick=1)
        self.assertIsNone(self.buffer.pop(2, 101.0))
        self.assertEqual(self.buffer.targets, {})

    def test_fully_missed_new_short_chunk_removes_an_old_long_tail(self):
        self.buffer.remember(1, 0, 100.0)
        self.buffer.accept(self.actions(), now=100.0, tick=0)
        self.buffer.remember(2, 1, 100.1)
        count = self.buffer.accept(self.actions(2, 1, [[10]]), now=100.3, tick=3)
        self.assertEqual(count, 0)
        self.assertIsNone(self.buffer.pop(3, 100.3))
        self.assertEqual(self.buffer.targets, {})

    def test_rejects_wrong_request_and_changed_contract_without_accepting_it(self):
        changes = (
            {"id": True}, {"id": 99}, {"session": "other"}, {"context": {"epoch": 2}},
            {"tick": 1}, {"tick": False}, {"fps": 30}, {"fps": True},
            {"names": ["other.pos"]},
        )
        for change in changes:
            with self.subTest(change=change):
                self.setUp()
                self.buffer.remember(1, 0, 100.0)
                with self.assertRaises(ProtocolError):
                    self.buffer.accept({**self.actions(), **change}, now=100.1, tick=1)
                self.assertEqual(self.buffer.last_response_id, 0)
                self.assertEqual(self.buffer.targets, {})
        self.buffer.accept(self.actions(), now=100.1, tick=1)
        with self.assertRaises(ProtocolError):
            self.buffer.accept(self.actions(), now=100.2, tick=2)

    def test_rejects_invalid_shape_and_nonfinite_values_before_replacing_plan(self):
        for change in (
            {"actions": []}, {"actions": [[0]] * 9}, {"actions": [[0, 1]]},
            {"actions": [[float("nan")]]}, {"actions": [[True]]},
            {"inference_s": float("inf")}, {"inference_s": -1},
        ):
            with self.subTest(change=change):
                self.setUp()
                self.buffer.remember(1, 0, 100.0)
                with self.assertRaises((ProtocolError, ValueError)):
                    self.buffer.accept({**self.actions(), **change}, now=100.1, tick=1)
                self.assertEqual(self.buffer.targets, {})

    def test_retained_request_history_is_bounded(self):
        for request_id in range(1, 301):
            self.buffer.remember(request_id, request_id, 100 + request_id / 10)
        self.assertEqual(len(self.buffer.requests), 256)
        with self.assertRaisesRegex(ProtocolError, "retained observation"):
            self.buffer.accept(self.actions(), now=131.0, tick=310)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeHost:
    client_id = "pi-client"

    def __init__(self, clock, events):
        self.clock, self.events = clock, events
        self.sent = []
        self.acknowledge = True
        self.context_change_after = None
        self.missing_camera_after = None
        self.refreshes = 0
        self.refresh_delays = {}
        self.last_sequence = 0

    def connect_control(self):
        self.events.append("host-connect")

    def refresh(self, *, include_images):
        assert include_images
        self.refreshes += 1
        started = self.clock()
        self.clock.sleep(0.001 + self.refresh_delays.get(self.refreshes, 0.0))
        names = command_target_keys(MODEL) - {"lift_axis.stop"}
        payload = {name: 10.0 if name.endswith(".pos") else 0.0 for name in names}
        payload["lift_axis.reference_sequence"] = 1
        payload["_robot_metadata"] = dict(
            robot_model=MODEL,
            motors={
                name.removesuffix(".pos"): dict(
                    normalization="range_0_100" if "gripper" in name else "range_m100_100"
                )
                for name in names if name.endswith(".pos")
            },
        )
        epoch = int(
            self.context_change_after is not None and self.refreshes >= self.context_change_after
        )
        payload["_safety"] = dict(
            version=1, host_session_id="host", control_epoch=epoch, control_owner=None,
            feedback_valid=True, lift_reference_valid=True, phase="ready",
            command=dict(client_id=self.client_id, sequence=self.last_sequence),
        )
        missing = self.missing_camera_after is not None
        missing = missing and self.refreshes >= self.missing_camera_after
        return HostSnapshot(payload, {} if missing else {"front": JPEG}, started, self.clock())

    def send_command(self, targets, *, based_on):
        self.sent.append((self.clock(), dict(targets), based_on))
        self.events.append("host-command")
        sequence = len(self.sent)
        if self.acknowledge:
            self.last_sequence = sequence
        return CommandIdentity(self.client_id, sequence, "host", 0)


class FakeLink:
    def __init__(self, clock, events):
        self.clock, self.events = clock, events
        self.error = None
        self.pending = []
        self.observations = []
        self.chunk_size = 8
        self.response_limit = None
        self.ready_enabled = True
        self.actions_enabled = True
        self.fail_after = None
        self.closed = False
        self.names = sorted(command_target_keys(MODEL) - {"lift_axis.stop"})

    def start(self):
        self.events.append("link-start")
        return self

    def submit(self, kind, body, images=None):
        if kind == "hello" and self.ready_enabled:
            fixed = body["fixed"]
            self.pending.append(("ready", dict(
                session=body["session"], id=body["id"], context=body["context"], fps=body["fps"],
                names=self.names, cameras=["front"], chunk_size=self.chunk_size,
                fixed_current=list(fixed["targets"]) if fixed else [],
            ), {}))
        elif kind == "observation":
            self.observations.append(copy.deepcopy(body))
            if self.fail_after == len(self.observations):
                self.error = RuntimeError("cloud disconnected")
            if not self.actions_enabled:
                return
            if self.response_limit is not None and len(self.observations) > self.response_limit:
                return
            rows = [[(body["tick"] + i) / 100] * len(self.names) for i in range(self.chunk_size)]
            self.pending.append(("actions", dict(
                session=body["session"], id=body["id"], context=body["context"],
                tick=body["tick"], fps=body["fps"], names=self.names,
                actions=rows, inference_s=0.01,
            ), {}))
        elif kind == "stop":
            self.events.append("cloud-stop")
            self.pending.append(("ack", {}, {}))

    def poll(self):
        return self.pending.pop(0) if self.pending else None

    def close(self):
        self.events.append("link-close")
        self.closed = True


class RemoteSessionTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.events = []
        self.client = FakeHost(self.clock, self.events)
        self.link = FakeLink(self.clock, self.events)
        self.stops = []

    def run_session(self, **options):
        def stop(client, model, identity):
            self.events.append("host-stop")
            self.stops.append((client, model, identity))

        with (
            patch("alohamini.remote.agent.stop_owned_robot", side_effect=stop),
            patch("alohamini.client.time.monotonic", self.clock),
        ):
            return run_session(
                self.client, self.link, MODEL, fps=10, duration_s=options.pop("duration_s", 0.35),
                rpc_timeout_s=0.25, clock=self.clock, sleep=self.clock.sleep, **options,
            )

    def assert_stopped_and_closed(self):
        self.assertTrue(self.link.closed)
        self.assertEqual(len(self.stops), 1)
        self.assertLess(self.events.index("host-stop"), self.events.index("cloud-stop"))
        self.assertLess(self.events.index("cloud-stop"), self.events.index("link-close"))

    def test_executes_current_ticks_with_fresh_feedback_and_observed_acknowledgements(self):
        stats = self.run_session()
        self.assertEqual(stats["sent"], 3)
        self.assertEqual(stats["acknowledged_sequence"], 2)
        first = self.client.sent[0]
        self.assertEqual(first[1]["arm_left_shoulder_pan.pos"], 0.01)
        self.assertGreater(first[2].request_started_s, self.link.observations[0]["snapshot"][
            "request_started_s"
        ])
        self.assertEqual(self.stops[0][2].sequence, 3)
        self.assert_stopped_and_closed()

    def test_policy_startup_timeout_sends_no_motion_and_still_closes_link(self):
        self.link.ready_enabled = False
        with self.assertRaisesRegex(RuntimeError, "startup timed out"):
            self.run_session()
        self.assertEqual(self.client.sent, [])
        self.assertIsNone(self.stops[0][2])
        self.assert_stopped_and_closed()

    def test_incompatible_readiness_contract_never_sends_motion(self):
        self.link.names[-1] = self.link.names[0]
        with self.assertRaisesRegex(ProtocolError, "incompatible policy contract"):
            self.run_session()
        self.assertEqual(self.client.sent, [])
        self.assert_stopped_and_closed()

    def test_missing_first_action_times_out_without_motion(self):
        self.link.actions_enabled = False
        with self.assertRaisesRegex(RuntimeError, "No unexpired remote action"):
            self.run_session(duration_s=0.8)
        self.assertEqual(self.client.sent, [])
        self.assert_stopped_and_closed()

    def test_exhausted_action_horizon_stops_without_repeating_last_target(self):
        self.link.chunk_size = 3
        self.link.response_limit = 1
        with self.assertRaisesRegex(RuntimeError, "No unexpired remote action"):
            self.run_session(duration_s=0.8)
        self.assertEqual(len(self.client.sent), 2)
        self.assertEqual([row[1][self.link.names[0]] for row in self.client.sent], [0.01, 0.02])
        self.assert_stopped_and_closed()

    def test_host_context_change_stops_before_the_next_command(self):
        self.client.context_change_after = 4
        with self.assertRaisesRegex(RuntimeError, "Host session, control epoch"):
            self.run_session()
        self.assertEqual(len(self.client.sent), 1)
        self.assert_stopped_and_closed()

    def test_feedback_stall_skips_elapsed_ticks_without_catchup_commands(self):
        self.client.refresh_delays[4] = 0.25
        self.run_session(duration_s=0.55)
        self.assertEqual(
            [row[1]["arm_left_shoulder_pan.pos"] for row in self.client.sent],
            [0.01, 0.04, 0.05],
        )
        self.assert_stopped_and_closed()

    def test_stale_host_feedback_stops_before_a_cloud_target_is_sent(self):
        self.client.refresh_delays[3] = 1.1
        with self.assertRaisesRegex(RuntimeError, "Fresh, controllable Host feedback"):
            self.run_session(duration_s=2.0)
        self.assertEqual(self.client.sent, [])
        self.assert_stopped_and_closed()

    def test_absent_host_acknowledgement_stops_even_with_continuing_cloud_actions(self):
        self.client.acknowledge = False
        with self.assertRaisesRegex(RuntimeError, "stopped acknowledging"):
            self.run_session(duration_s=0.8, ack_timeout_s=0.15)
        self.assertEqual(len(self.client.sent), 2)
        self.assert_stopped_and_closed()

    def test_cloud_failure_does_not_leave_the_local_lease_running(self):
        self.link.fail_after = 2
        with self.assertRaisesRegex(RuntimeError, "cloud disconnected"):
            self.run_session()
        self.assertEqual(len(self.client.sent), 1)
        self.assert_stopped_and_closed()

    def test_lost_policy_camera_stops_before_sending_a_target(self):
        self.client.missing_camera_after = 3
        with self.assertRaisesRegex(RuntimeError, "missing a policy camera"):
            self.run_session()
        self.assertEqual(self.client.sent, [])
        self.assert_stopped_and_closed()

    def test_fixed_current_target_overrides_a_conflicting_cloud_prediction(self):
        self.run_session(fixed_current=["arm_left_shoulder_pan.pos"])
        self.assertTrue(self.client.sent)
        for _, targets, _ in self.client.sent:
            self.assertEqual(targets["arm_left_shoulder_pan.pos"], 10.0)
            self.assertNotEqual(targets["arm_right_shoulder_pan.pos"], 10.0)
        self.assert_stopped_and_closed()


if __name__ == "__main__":
    unittest.main()

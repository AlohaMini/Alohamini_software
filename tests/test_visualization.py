import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec("numpy"), "PC image dependencies unavailable")
class VisualizationTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        from alohamini.apps import visualization

        self.np, self.viz = np, visualization
        self.rr = ModuleType("rerun")
        for name in (
            "log",
            "Scalars",
            "Image",
            "DepthImage",
            "init",
            "spawn",
            "connect_grpc",
            "send_blueprint",
            "rerun_shutdown",
        ):
            setattr(self.rr, name, Mock())
        blueprint = ModuleType("rerun.blueprint")
        for name in ("Spatial2DView", "TimeSeriesView", "Grid", "Blueprint"):
            setattr(blueprint, name, Mock())
        self.blueprint = blueprint
        # Restore only mocked entries; unloading newly imported native extensions
        # (e.g. cv2) through a whole sys.modules snapshot breaks later imports.
        for name, module in {"rerun": self.rr, "rerun.blueprint": blueprint}.items():
            if name in sys.modules:
                self.addCleanup(sys.modules.__setitem__, name, sys.modules[name])
            else:
                self.addCleanup(sys.modules.pop, name, None)
            sys.modules[name] = module
        start = patch.object(
            self.viz, "_start_local_viewer", return_value="rerun+http://127.0.0.1:43210/proxy"
        )
        self.start_viewer = start.start()
        self.addCleanup(start.stop)
        self.viz.init_rerun()

    def test_images_are_temporal_not_static_entities(self):
        self.viz.log_rerun_data({"forward": self.np.zeros((8, 16, 3), dtype=self.np.uint8)})
        self.assertFalse(self.rr.log.call_args.kwargs.get("static", False))

    def test_camera_arriving_after_first_state_gets_a_view(self):
        self.viz.log_rerun_data({"arm_left_gripper.pos": 10.0}, {"arm_left_gripper.pos": 20.0})
        self.viz.log_rerun_data({"forward": self.np.zeros((8, 16, 3), dtype=self.np.uint8)})
        self.blueprint.Spatial2DView.assert_called_with(
            origin="observation.forward", name="observation.forward"
        )
        self.assertEqual(self.rr.send_blueprint.call_count, 2)
        self.viz.log_rerun_data({"arm_left_gripper.pos": 11.0})
        self.assertEqual(self.rr.send_blueprint.call_count, 2)

    def test_scalar_and_image_names_keep_original_namespaces(self):
        self.viz.log_rerun_data({"joint.pos": 1.0, "observation.other": 2.0}, {"joint.pos": 3.0})
        self.assertEqual(
            [c.args[0] for c in self.rr.log.call_args_list],
            ["observation.joint.pos", "observation.other", "action.joint.pos"],
        )

    def test_init_and_shutdown_use_existing_viewer_operations(self):
        self.rr.init.assert_called_once_with("alohamini_teleop")
        self.start_viewer.assert_called_once_with("10%")
        self.rr.connect_grpc.assert_called_once_with(url="rerun+http://127.0.0.1:43210/proxy")
        self.rr.spawn.assert_not_called()
        self.viz.shutdown_rerun()
        self.rr.rerun_shutdown.assert_called_once()

    def test_snapshot_preserves_rgb_and_does_not_create_missing_camera_frames(self):
        import cv2

        from alohamini.protocol import HostSnapshot

        rgb = self.np.zeros((8, 16, 3), dtype=self.np.uint8)
        rgb[:, :, 0] = 255
        ok, jpeg = cv2.imencode(".jpg", rgb)
        self.assertTrue(ok)
        snapshot = HostSnapshot(
            {"joint.pos": 1.0, "_safety": {}, "_robot_metadata": {}},
            {"forward": jpeg.tobytes()},
            1.0,
            1.01,
        )
        with patch.object(self.viz, "log_rerun_data") as log:
            self.viz.log_snapshot(snapshot, {"joint.pos": 2.0})
        observation, action = log.call_args.args
        self.assertEqual(set(observation), {"joint.pos", "forward"})
        self.assertGreater(int(observation["forward"][0, 0, 0]), 240)
        self.assertLess(int(observation["forward"][0, 0, 2]), 10)
        self.assertEqual(action, {"joint.pos": 2.0})

    def test_bad_jpeg_does_not_replace_real_state_or_fabricate_black_frame(self):
        from alohamini.protocol import HostSnapshot

        snapshot = HostSnapshot({"joint.pos": 1.0}, {"forward": b"bad-jpeg"}, 1.0, 1.01)
        with patch.object(self.viz, "log_rerun_data") as log:
            self.viz.log_snapshot(snapshot, {})
        self.assertEqual(log.call_args.args, ({"joint.pos": 1.0}, {}))

    def test_slow_preview_is_latest_only_and_does_not_block_control(self):
        from test_teleoperation import snapshot

        from alohamini.apps.teleoperation import run_loop

        entered, release, latest_seen = threading.Event(), threading.Event(), threading.Event()
        rendered = []

        def render(_snapshot, action):
            rendered.append(action["joint.pos"])
            if len(rendered) == 1:
                entered.set()
                release.wait(2)
            else:
                latest_seen.set()

        stop = threading.Event()
        client, leader = Mock(client_id="client"), Mock()
        client.read.side_effect = snapshot
        leader.read.return_value = {"joint.pos": 1}
        count = 0

        def send(*_args, **_kwargs):
            nonlocal count
            count += 1
            if count == 4:
                stop.set()
            return Mock()

        client.send_command.side_effect = send
        with patch.object(self.viz, "log_snapshot", side_effect=render):
            with self.viz.TeleopPreview(None, "alohamini2pro", fps=30) as worker:
                try:
                    worker.submit(snapshot(), {"joint.pos": 0})
                    self.assertTrue(entered.wait(1))
                    with patch("alohamini.apps.teleoperation.stop_owned_robot"):
                        run_loop(
                            client,
                            "alohamini2pro",
                            leader,
                            None,
                            on_frame=worker.submit,
                            stop_event=stop,
                        )
                    self.assertEqual(count, 4)
                    for index in range(2, 100):
                        worker.submit(snapshot(), {"joint.pos": index})
                    self.assertEqual(rendered, [0])
                    release.set()
                    self.assertTrue(latest_seen.wait(1))
                finally:
                    release.set()
            self.assertFalse(worker._thread.is_alive())
        self.assertEqual(rendered, [0, 99])
        self.assertTrue(
            all(not call.kwargs.get("include_images") for call in client.read.call_args_list)
        )

    def test_camera_subscription_is_bounded_session_bound_and_standard_rgb(self):
        import cv2
        import zmq
        from test_teleoperation import snapshot

        sample = snapshot()
        sample.payload["_robot_metadata"]["cameras"] = ["forward"]
        sample.payload["_host_timing"] = {"state_sample_monotonic_s": 2.0}
        worker = self.viz.TeleopPreview("test-only", "alohamini2pro", fps=30)

        def packet(seq, *, session="session", name="forward", stamp=2.0, red=True):
            bgr = self.np.zeros((8, 16, 3), dtype=self.np.uint8)
            bgr[:, :, 2 if red else 0] = 255
            ok, jpeg = cv2.imencode(".jpg", bgr)
            self.assertTrue(ok)
            metadata = dict(
                schema_version=1,
                camera_name=name,
                sequence=seq,
                width=16,
                height=8,
                encoding="jpeg",
                capture_monotonic_s=stamp,
                capture_unix_ns=2_000_000_000,
                host_session_id=session,
            )
            return [f"camera/{name}".encode(), json.dumps(metadata).encode(), jpeg.tobytes()]

        context, socket = Mock(), Mock()
        context.socket.return_value = socket
        socket.recv_multipart.side_effect = [
            packet(1, red=False),
            packet(2),
            packet(1, red=False),
            packet(3, session="old-session", red=False),
            packet(3, stamp=1.0, red=False),
            packet(3, name="disabled"),
            [b"bad"],
            zmq.Again(),
        ]
        worker.submit(sample, {})
        with (
            patch("zmq.Context", return_value=context),
            patch.object(self.viz, "log_snapshot"),
            patch.object(
                self.viz, "log_rerun_data", side_effect=lambda *_: worker._stop.set()
            ) as log,
        ):
            worker._run()
        context.socket.assert_called_once_with(zmq.SUB)
        socket.connect.assert_called_once_with("tcp://test-only:5557")
        socket.send.assert_not_called()
        image = log.call_args.args[0]["forward"]
        self.assertGreater(int(image[0, 0, 0]), 240)
        self.assertLess(int(image[0, 0, 2]), 10)
        self.assertEqual(set(log.call_args.args[0]), {"forward"})
        socket.close.assert_called_once_with(linger=0)
        context.term.assert_called_once()


@unittest.skipUnless(importlib.util.find_spec("numpy"), "PC image dependencies unavailable")
class ViewerLifecycleTests(unittest.TestCase):
    def setUp(self):
        from alohamini.apps import visualization

        self.viz = visualization
        self.original_executable = visualization._viewer_executable
        self.original_wait = visualization._wait_for_viewer
        self.actual_popen = subprocess.Popen
        self.rr = ModuleType("rerun")
        for name in ("init", "connect_grpc", "rerun_shutdown"):
            setattr(self.rr, name, Mock())
        if "rerun" in sys.modules:
            self.addCleanup(sys.modules.__setitem__, "rerun", sys.modules["rerun"])
        else:
            self.addCleanup(sys.modules.pop, "rerun", None)
        sys.modules["rerun"] = self.rr
        self.process = Mock()
        self.process.poll.return_value = None
        self.process.wait.return_value = 0
        self.popen = Mock(return_value=self.process)
        patches = (
            patch.object(self.viz, "_viewer_process", None),
            patch.object(self.viz, "_viewer_executable", return_value="/opt/rerun-native"),
            patch.object(self.viz, "_unused_local_port", return_value=43210),
            patch.object(self.viz, "_wait_for_viewer"),
            patch.object(self.viz.subprocess, "Popen", self.popen),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_local_viewer_is_native_owned_and_reaped_once(self):
        with patch.dict(os.environ, {"ALOHAMINI_RERUN_MEMORY_LIMIT": "12%"}):
            with patch.dict(os.environ):
                os.environ.pop("RUST_LOG", None)
                self.viz.init_rerun()
        command = self.popen.call_args.args[0]
        self.assertEqual(command, [
            "/opt/rerun-native", "--bind=127.0.0.1", "--port=43210", "--memory-limit=12%",
            "--server-memory-limit=0B", "--expect-data-soon",
        ])
        options = self.popen.call_args.kwargs
        self.assertEqual(options["stdin"], subprocess.DEVNULL)
        self.assertTrue(options["start_new_session"])
        self.assertNotIn("stdout", options)
        self.assertNotIn("stderr", options)
        self.assertEqual(options["env"]["RERUN_APP_ONLY"], "true")
        self.assertEqual(options["env"]["RUST_LOG"], "warn")
        self.rr.connect_grpc.assert_called_once_with(url="rerun+http://127.0.0.1:43210/proxy")
        self.viz.shutdown_rerun()
        self.viz.shutdown_rerun()
        self.process.terminate.assert_called_once()
        self.process.wait.assert_called_once_with(timeout=1.0)
        self.process.kill.assert_not_called()
        self.assertIsNone(self.viz._viewer_process)

    def test_child_respects_explicit_log_filter(self):
        with patch.dict(os.environ, {"RUST_LOG": "error"}):
            self.viz.init_rerun()
        self.assertEqual(self.popen.call_args.kwargs["env"]["RUST_LOG"], "error")
        self.viz.shutdown_rerun()

    def test_external_viewer_is_not_spawned_or_owned(self):
        self.viz.init_rerun(ip="192.0.2.10", port=9876)
        self.rr.connect_grpc.assert_called_once_with(url="rerun+http://192.0.2.10:9876/proxy")
        self.viz.shutdown_rerun()
        self.popen.assert_not_called()
        self.process.terminate.assert_not_called()
        self.process.kill.assert_not_called()

    def test_reinitialization_reaps_previous_viewer_before_spawning_replacement(self):
        second = Mock()
        second.poll.return_value = None
        second.wait.return_value = 0
        self.popen.side_effect = [self.process, second]
        self.viz.init_rerun("first")
        self.viz.init_rerun("second")
        self.process.terminate.assert_called_once()
        self.process.wait.assert_called_once()
        second.terminate.assert_not_called()
        self.viz.shutdown_rerun()
        self.process.terminate.assert_called_once()
        second.terminate.assert_called_once()

    def test_sdk_shutdown_failure_still_reaps_owned_viewer(self):
        self.viz.init_rerun()
        self.rr.rerun_shutdown.side_effect = RuntimeError("SDK shutdown failed")
        with self.assertRaisesRegex(RuntimeError, "SDK shutdown failed"):
            self.viz.shutdown_rerun()
        self.process.terminate.assert_called_once()
        self.process.wait.assert_called_once()
        self.assertIsNone(self.viz._viewer_process)

    def test_stubborn_owned_viewer_is_killed_with_bounded_waits(self):
        self.process.wait.side_effect = [subprocess.TimeoutExpired("rerun", 1.0), 0]
        self.viz.init_rerun()
        self.viz.shutdown_rerun()
        self.process.terminate.assert_called_once()
        self.process.kill.assert_called_once()
        self.assertEqual([call.kwargs for call in self.process.wait.call_args_list], [
            {"timeout": 1.0}, {"timeout": 1.0},
        ])

    def test_already_exited_owned_viewer_is_reaped_without_signals(self):
        self.viz.init_rerun()
        self.process.poll.return_value = 0
        self.viz.shutdown_rerun()
        self.process.wait.assert_called_once_with(timeout=1.0)
        self.process.terminate.assert_not_called()
        self.process.kill.assert_not_called()

    def test_startup_and_connection_failures_reap_child_preserving_original_error(self):
        for phase, error in (("startup", TimeoutError("viewer startup timeout")),
                             ("connect", KeyboardInterrupt())):
            with self.subTest(phase=phase):
                self.process.reset_mock()
                self.rr.rerun_shutdown.reset_mock()
                self.viz._wait_for_viewer.side_effect = error if phase == "startup" else None
                self.rr.connect_grpc.side_effect = error if phase == "connect" else None
                with self.assertRaises(type(error)) as raised:
                    self.viz.init_rerun()
                self.assertIs(raised.exception, error)
                self.process.terminate.assert_called_once()
                self.process.wait.assert_called_once()
                self.assertIsNone(self.viz._viewer_process)

    def test_spawn_failure_cleans_up_sdk_without_signalling_any_child(self):
        failure = FileNotFoundError("native viewer missing")
        self.popen.side_effect = failure
        with self.assertRaises(FileNotFoundError) as raised:
            self.viz.init_rerun()
        self.assertIs(raised.exception, failure)
        self.rr.rerun_shutdown.assert_called_once()
        self.process.terminate.assert_not_called()
        self.assertIsNone(self.viz._viewer_process)

    def test_child_cleanup_failure_is_raised_without_primary_and_can_be_retried(self):
        self.viz.init_rerun()
        self.process.terminate.side_effect = OSError("viewer termination failed")
        with self.assertRaisesRegex(OSError, "viewer termination failed"):
            self.viz.shutdown_rerun()
        self.assertIs(self.viz._viewer_process, self.process)
        self.process.terminate.side_effect = None
        self.viz.shutdown_rerun()
        self.assertIsNone(self.viz._viewer_process)

    def test_cleanup_failure_preserves_active_exception_and_reports_each_failure(self):
        self.viz.init_rerun()
        self.rr.rerun_shutdown.side_effect = RuntimeError("SDK shutdown failed")
        self.process.terminate.side_effect = OSError("viewer termination failed")
        for original in (RuntimeError("leader read failed"), KeyboardInterrupt()):
            with (
                self.subTest(original=type(original).__name__),
                self.assertLogs(level="WARNING") as log,
            ):
                self.assertFalse(self.viz.shutdown_rerun(type(original), original, None))
            self.assertTrue(any("SDK shutdown failed" in message for message in log.output))
            self.assertTrue(any("viewer termination failed" in message for message in log.output))

    def test_init_failure_reports_cleanup_error_and_preserves_init_exception(self):
        failure = RuntimeError("viewer connection failed")
        self.rr.connect_grpc.side_effect = failure
        self.process.terminate.side_effect = OSError("viewer termination failed")
        with self.assertLogs(level="WARNING") as log, self.assertRaises(RuntimeError) as raised:
            self.viz.init_rerun()
        self.assertIs(raised.exception, failure)
        self.assertTrue(any("viewer termination failed" in message for message in log.output))

    def test_native_path_respects_cli_override_without_starting_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / ("viewer.exe" if os.name == "nt" else "viewer")
            executable.touch()
            with patch.dict(os.environ, {"RERUN_CLI_PATH": str(executable)}):
                self.assertEqual(self.original_executable(), str(executable))

    def test_readiness_checks_only_selected_local_port_and_child_liveness(self):
        with (
            patch.object(self.viz.socket, "create_connection") as connect,
            patch.object(self.viz.time, "monotonic", side_effect=[0.0, 0.0, 0.02]),
            patch.object(self.viz.time, "sleep") as pause,
        ):
            connect.side_effect = [ConnectionRefusedError(), connect.return_value]
            self.original_wait(self.process, 43210)
        self.assertEqual([call.args for call in connect.call_args_list], [
            (("127.0.0.1", 43210),), (("127.0.0.1", 43210),),
        ])
        self.assertTrue(all(call.kwargs["timeout"] <= 0.1 for call in connect.call_args_list))
        pause.assert_called_once_with(0.02)
        self.assertEqual(self.process.poll.call_count, 3)

    def test_readiness_rejects_dead_child_even_if_port_accepts_connection(self):
        self.process.poll.side_effect = [None, 2]
        with patch.object(self.viz.socket, "create_connection"):
            with self.assertRaisesRegex(RuntimeError, "退出码：2"):
                self.original_wait(self.process, 43210)

    def test_readiness_timeout_is_bounded_without_contacting_any_other_port(self):
        with (
            patch.object(self.viz.time, "monotonic", side_effect=[0.0, 6.0]),
            patch.object(self.viz.socket, "create_connection") as connect,
        ):
            with self.assertRaisesRegex(TimeoutError, "127.0.0.1:43210"):
                self.original_wait(self.process, 43210)
        connect.assert_not_called()

    def test_shutdown_reaps_real_owned_python_child_without_viewer_or_hardware(self):
        child = self.actual_popen(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            self.viz._viewer_process = child
            self.viz.shutdown_rerun()
            self.assertIsNotNone(child.poll())
            self.assertIsNone(self.viz._viewer_process)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=1.0)


@unittest.skipUnless(importlib.util.find_spec("rerun"), "PC viewer dependency unavailable")
class RerunMemoryTests(unittest.TestCase):
    def test_preview_worker_writes_to_real_sdk_without_viewer(self):
        import rerun as rr
        from test_teleoperation import snapshot

        from alohamini.apps import visualization

        rr.init("alohamini-worker-test", spawn=False, strict=True)
        complete = threading.Event()
        original = visualization.log_snapshot

        def render(*args):
            original(*args)
            complete.set()

        try:
            memory = rr.memory_recording()
            visualization.log_rerun_data.paths = (set(), set(), set())
            with patch.object(visualization, "log_snapshot", side_effect=render):
                with visualization.TeleopPreview(None, "alohamini2pro", fps=30) as worker:
                    worker.submit(snapshot(), {"joint.pos": 2.0})
                    self.assertTrue(complete.wait(1))
            self.assertGreater(memory.num_msgs(), 0)
            self.assertTrue(memory.drain_as_bytes())
        finally:
            rr.rerun_shutdown()

    def test_real_sdk_accepts_native_state_action_and_images_without_viewer(self):
        import numpy as np
        import rerun as rr

        from alohamini.apps import visualization

        rr.init("alohamini-offline-test", spawn=False, strict=True)
        try:
            memory = rr.memory_recording()
            visualization.log_rerun_data.paths = (set(), set(), set())
            visualization.log_rerun_data({"joint.pos": 1.0}, {"joint.pos": 2.0})
            visualization.log_rerun_data({"forward": np.zeros((8, 16, 3), dtype=np.uint8)})
            self.assertGreater(memory.num_msgs(), 0)
            self.assertTrue(memory.drain_as_bytes())
        finally:
            rr.rerun_shutdown()

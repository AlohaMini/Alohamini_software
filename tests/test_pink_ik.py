import json
import subprocess
import sys
import time

import numpy as np
import pytest

from alohamini.kinematics.arm import SerialArmKinematics
from alohamini.kinematics.pink import IKError
from alohamini.kinematics.worker import PinkWorkerClient


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("rebased", [False, True])
def test_pinocchio_matches_frozen_geometry_and_jacobians(side, rebased):
    geometry = SerialArmKinematics.from_model("alohamini2pro", side)
    if rebased:
        geometry = geometry.with_joint_zero([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
    # Reconstruct exactly as a checkpoint does, without model files or ROS.
    geometry = SerialArmKinematics.from_parameters(json.loads(json.dumps(geometry.parameters())))
    solver = geometry.pink_solver()
    lower, upper = np.asarray(geometry.joint_limits).T
    for q in np.random.default_rng(12).uniform(lower, upper, (16, len(lower))):
        solver.configuration.update(q)
        transform = solver.configuration.get_transform_frame_to_world("tcp")
        np.testing.assert_allclose(transform.homogeneous, geometry.forward(q), atol=1e-12)
        # Pink returns an analytic body Jacobian; the shared DH helper uses
        # forward finite differences (1e-6 rad) in the arm base frame.
        rotate = np.zeros((6, 6))
        rotate[:3, :3] = rotate[3:, 3:] = transform.rotation
        np.testing.assert_allclose(
            rotate @ solver.configuration.get_frame_jacobian("tcp"),
            geometry.jacobian(q),
            atol=5e-7,
            rtol=1e-6,
        )


def test_pose_solve_reports_failures_and_preserves_the_seed():
    geometry = SerialArmKinematics.from_model("alohamini2pro", "left")
    q = np.array([-0.4, -1, 1, -0.3, 0.2, -0.1])
    before = q.copy()
    target = geometry.forward(q + [0.02, -0.01, 0.01, 0, 0, 0])
    solved = geometry.inverse(target, q)
    np.testing.assert_allclose(geometry.forward(solved), target, atol=1e-4)
    np.testing.assert_array_equal(q, before)
    target[0, 3] += 5
    with pytest.raises(IKError) as error:
        geometry.inverse(target, q, max_iterations=2)
    assert error.value.metrics["iterations"] <= 2
    assert error.value.metrics["position_error_m"] > 1
    np.testing.assert_equal(error.value.seed, q)
    with pytest.raises(IKError, match="time budget"):
        geometry.inverse(target, q, timeout_s=1e-10)
    assert geometry.pink_solver() is geometry.pink_solver()


def test_tracking_obeys_velocity_acceleration_and_margin_through_target_reversal():
    geometry = SerialArmKinematics.from_model("alohamini2pro", "right")
    solver = geometry.pink_solver()
    q = np.array([-0.4, -1, 1, -0.3, 0.2, -0.1])
    initial = q.copy()
    previous_velocity = np.zeros(6)
    dt, acceleration = 0.02, 3.0
    targets = [geometry.forward(initial + [sign * 0.5, 0, 0, 0, 0, 0]) for sign in (1, -1)]
    for tick in range(100):
        candidate, metrics = solver.step(
            q,
            targets[tick // 50],
            dt,
            max_joint_velocity=0.4,
            max_joint_acceleration=acceleration,
        )
        velocity = (candidate - q) / dt
        assert np.max(np.abs(velocity)) <= 0.4 + 1e-8
        assert np.max(np.abs(velocity - previous_velocity)) <= acceleration * dt + 1e-8
        limits = np.asarray(geometry.joint_limits)
        assert np.all(candidate >= limits[:, 0] + 0.03 - 1e-8)
        assert np.all(candidate <= limits[:, 1] - 0.03 + 1e-8)
        assert metrics["backend"] == "pink"
        previous_velocity, q = velocity, candidate
    solver.reset()
    np.testing.assert_equal(solver.previous_velocity, 0)


@pytest.fixture(scope="module")
def worker(tmp_path_factory):
    endpoint = "ipc://" + str(tmp_path_factory.mktemp("pink") / "ik.sock")
    process = subprocess.Popen(
        [sys.executable, "-m", "alohamini.kinematics.worker", "--endpoint", endpoint],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    client = PinkWorkerClient(endpoint)
    try:
        assert client.request("health", timeout_ms=10000)["pink"] == "4.4.0"
        yield client
    finally:
        client.close()
        process.terminate()
        process.communicate(timeout=5)


def test_worker_matches_local_tracking_and_resets_velocity_history(worker):
    geometry = SerialArmKinematics.from_model("alohamini2pro", "left")
    q = np.array([-0.4, -1, 1, -0.3, 0.2, -0.1])
    target = geometry.forward(q + [0.02, 0, 0, 0, 0, 0])
    expected, _ = geometry.pink_solver().step(q, target, 0.02)
    actual, metrics = worker.step(geometry, q, target, 0.02)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert metrics["rpc_elapsed_s"] >= metrics["elapsed_s"]
    worker.reset(geometry)
    actual, _ = worker.step(geometry, q, target, 0.02)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_worker_deadline_is_bounded_without_silently_switching_solver(tmp_path):
    client = PinkWorkerClient("ipc://" + str(tmp_path / "absent.sock"), timeout_s=0.01)
    started = time.monotonic()
    try:
        session = client.session
        with pytest.raises(TimeoutError, match="deadline"):
            client.request("health")
        assert time.monotonic() - started < 0.5
        assert client.session != session
    finally:
        client.close()
    with pytest.raises(ValueError, match="local"):
        PinkWorkerClient("tcp://0.0.0.0:7777")

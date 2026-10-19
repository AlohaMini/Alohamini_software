import numpy as np
import pytest
from alohamini_mujoco.control import DynamicFollower, JointTrajectory, solve_ik


def test_following_uses_dynamics_and_tracks_both_arms(robot):
    trajectory = JointTrajectory(
        robot, ["left_shoulder_pan", "right_shoulder_pan"], [0, 2, 4], [[0, 0], [0.1, -0.1], [0, 0]]
    )
    follower = DynamicFollower(robot)
    observations = []
    report = follower.follow(trajectory, observer=observations.append)
    assert report["passed"], report
    assert report["duration_s"] == pytest.approx(5, abs=1e-10)
    assert max(report["maximum_tracking_error"]) < 0.002
    assert max(report["final_error"]) < 1e-5
    assert report["final_contacts"] == []
    assert max(report["drive_saturated_steps"]) == 0
    assert observations[0]["time_s"] == 0
    assert np.max(np.abs(observations[1]["reference"] - observations[0]["reference"])) > 0
    for observation in observations:
        assert not np.shares_memory(observation["position"], follower.data.qpos)


def test_gravity_feedforward_stays_inside_motor_effort_limits(robot):
    follower = DynamicFollower(robot)
    index = follower.names.index("right_elbow_flex")
    follower.target[index] = robot.description.joints["right_elbow_flex"].upper
    saturated = follower.step()
    assert saturated[index]
    assert np.all(np.abs(follower.data.actuator_force) <= follower.effort_limits + 1e-12)
    assert np.all(np.abs(follower.data.ctrl) <= follower.effort_limits + 1e-12)
    assert np.linalg.norm(follower.data.qvel) > 0


def test_trajectory_checks_names_start_limits_and_peak_interpolated_speed(robot):
    for names, times, positions in [
        (["right_shoulder_pan"], [0, 0.01], [[0], [1]]),
        (["right_shoulder_pan"], [0, 2], [[0], [4]]),
        (["root_x_axis_joint"], [0, 2], [[0], [0.1]]),
        (["right_shoulder_pan"], [1, 2], [[0], [0.1]]),
        (["right_shoulder_pan"], [0, 2], [[0], [float("nan")]]),
    ]:
        with pytest.raises(ValueError):
            JointTrajectory(robot, names, times, positions)
    trajectory = JointTrajectory(robot, ["right_shoulder_pan"], [0, 2], [[0.1], [0.2]])
    with pytest.raises(ValueError, match="measured joint positions"):
        DynamicFollower(robot).follow(trajectory)
    np.testing.assert_allclose(trajectory.reference(1)[0], [0.15])
    np.testing.assert_allclose(trajectory.reference(1)[1], [0.075])
    np.testing.assert_array_equal(trajectory.reference(2)[1], [0])


def test_ik_solves_full_pose_without_changing_live_data(robot):
    live = robot.data()
    before = live.qpos.copy()
    positions = {"right_shoulder_pan": 0.04, "right_elbow_flex": 0.03}
    target = robot.description.forward(
        {**robot.initial_positions, **positions}, robot.profile.T_world_root
    )["right_tcp"]
    result = solve_ik(robot, "right", target)
    solved = robot.data(result["joint_positions"])
    np.testing.assert_allclose(solved.body("right_tcp").xpos, target[:3, 3], atol=1e-5)
    np.testing.assert_allclose(
        solved.body("right_tcp").xmat.reshape(3, 3), target[:3, :3], atol=1e-3
    )
    np.testing.assert_array_equal(live.qpos, before)
    assert result["rotation_error_rad"] < 1e-3


def test_unreachable_full_pose_is_rejected(robot):
    target = np.eye(4)
    target[:3, 3] = [10, 10, 10]
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_ik(robot, "right", target, max_iterations=40)


def test_static_gravity_hold_is_not_disabled(robot):
    follower = DynamicFollower(robot)
    initial = follower.data.qpos.copy()
    trajectory = JointTrajectory(robot, ["vertical_move"], [0, 1], [[0.2], [0.2]])
    report = follower.follow(trajectory)
    assert report["passed"]
    actuator_id = robot.model.actuator("vertical_move").id
    # The lifted subtree is 5.1 kg; its weight must be carried by the motor.
    assert follower.data.actuator_force[actuator_id] == pytest.approx(5.1 * 9.80665, rel=1e-4)
    assert robot.model.opt.gravity[2] == -9.80665
    np.testing.assert_allclose(follower.data.qpos, initial, atol=1e-4)

"""CLI routing without loading models or connecting to hardware."""

from unittest.mock import Mock, patch

from alohamini.cli import main
from alohamini.remote.cli import main as remote_main


def test_top_level_routes_inference_before_regular_parser():
    with patch("alohamini.remote.cli.main", return_value=7) as remote:
        assert main(["inference", "run", "--help"]) == 7
    remote.assert_called_once_with(["run", "--help"])


def test_pi_command_forwards_bounded_runtime_options(capsys):
    with patch("alohamini.remote.agent.run") as run:
        run.return_value = dict(sent=4, acknowledged_sequence=3)
        assert (
            remote_main(
                [
                    "run",
                    "--server",
                    "tcp://127.0.0.1:5560",
                    "--robot_model",
                    "alohamini2pro",
                    "--episode_time",
                    "2",
                    "--rpc-timeout-s",
                    "3",
                    "--max-prediction-age-s",
                    "0.7",
                    "--fixed-current",
                    "arm_right",
                    "lift_axis",
                ]
            )
            == 0
        )
    run.assert_called_once_with(
        "tcp://127.0.0.1:5560",
        host="127.0.0.1",
        robot_model="alohamini2pro",
        fps=30,
        duration_s=2.0,
        rpc_timeout_s=3.0,
        max_prediction_age_s=0.7,
        ack_timeout_s=0.5,
        fixed_current=["arm_right", "lift_axis"],
    )
    assert "last Host acknowledgement=3" in capsys.readouterr().out


def test_cloud_command_keeps_model_options_on_cloud():
    backend = Mock()
    with (
        patch("alohamini.remote.server.NativeChunkPolicy", return_value=backend) as policy,
        patch("alohamini.remote.server.serve") as serve,
    ):
        assert (
            remote_main(
                [
                    "serve",
                    "--policy.path",
                    "/model",
                    "--task",
                    "place block",
                    "--chunk-size",
                    "32",
                    "--calibration-mode",
                    "normalized",
                    "--robot_model",
                    "alohamini2pro",
                    "--fixed-current",
                    "arm_right",
                    "lift_axis",
                ]
            )
            == 0
        )
    policy.assert_called_once_with(
        "/model",
        device="cuda",
        chunk_size=32,
        overrides=dict(
            calibration_mode="normalized", task="place block", robot_model="alohamini2pro"
        ),
        fixed_current=("arm_right", "lift_axis"),
    )
    assert serve.call_args.args[:2] == ("tcp://127.0.0.1:5560", backend)
    assert serve.call_args.kwargs == dict(request_timeout_s=15.0)


def test_runtime_failure_is_reported_without_traceback(capsys):
    with patch("alohamini.remote.agent.run", side_effect=RuntimeError("expired action")):
        assert (
            remote_main(
                [
                    "run",
                    "--server",
                    "tcp://127.0.0.1:5560",
                    "--robot_model",
                    "alohamini2pro",
                ]
            )
            == 1
        )
    assert capsys.readouterr().err == "alohamini: expired action\n"


def test_default_cloud_command_preserves_absent_fixed_selection():
    with (
        patch("alohamini.remote.server.NativeChunkPolicy") as policy,
        patch("alohamini.remote.server.serve"),
    ):
        assert remote_main(["serve", "--policy.path", "/model"]) == 0
    assert policy.call_args.kwargs["fixed_current"] is None

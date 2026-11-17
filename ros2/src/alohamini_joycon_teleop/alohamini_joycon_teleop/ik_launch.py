"""Start the CPU IK worker outside the ROS Python/NumPy environment."""

import os
import uuid

from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import EnvironmentVariable, LaunchConfiguration


def local_endpoint():
    return f"ipc:///tmp/alohamini-ik-{os.getuid()}-{uuid.uuid4().hex[:12]}.sock"


def start_worker(context, endpoint):
    python = LaunchConfiguration("ik_python").perform(context)
    command = (
        [python]
        if python
        else [
            EnvironmentVariable("CONDA_EXE", default_value="conda").perform(context),
            "run",
            "--no-capture-output",
            "-n",
            "alohamini",
            "python",
        ]
    )
    process = ExecuteProcess(
        cmd=[*command, "-m", "alohamini.kinematics.worker", "--endpoint", endpoint],
        additional_env={"PYTHONPATH": "", "PYTHONHOME": "", "LD_LIBRARY_PATH": ""},
        output="screen",
    )
    return [
        RegisterEventHandler(
            OnProcessExit(
                target_action=process,
                on_exit=[EmitEvent(event=Shutdown(reason="Pink IK worker exited"))],
            )
        ),
        process,
    ]

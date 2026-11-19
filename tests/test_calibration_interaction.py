"""真实 CLI 管道交互，不连接任何电机。"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("response, expected, code", [
    ('{"action":"cancel"}\n', "cancelled", 0),
    ("", "cancelled", 0),
    ('{"action":"confirm","prompt_id":999}\n', "failed", 1),
    ('{"action":"unknown"}\n', "failed", 1),
])
def test_gui_cli_accepts_buttons_without_tty_and_reports_cancellation_or_errors(
    tmp_path, response, expected, code
):
    with subprocess.Popen(
        [sys.executable, "-m", "alohamini", "calibrate", "leader", "--gui",
         "--robot_model=alohamini2pro"],
        env={**os.environ, "ALOHAMINI_WORKSPACE": str(tmp_path)},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
    ) as child:
        first = json.loads(child.stdout.readline())
        assert first == dict(step="disconnect_usb", prompt_id=1)
        if response:
            child.stdin.write(response)
            child.stdin.flush()
            final = child.stdout.readline()
            output, error = child.communicate(timeout=5)
            output = final + output
        if not response:
            output, error = child.communicate(response, timeout=5)
    assert child.returncode == code, error
    assert json.loads(output.splitlines()[-1])["step"] == expected
    assert not list(tmp_path.rglob("*.json"))


def test_gui_cli_success_exits_cleanly_while_desktop_keeps_input_open(tmp_path):
    script = """
import sys
from pathlib import Path
from unittest.mock import patch
from test_calibration_procedure import CalibrationProcedureTests
from alohamini.cli import main
case = CalibrationProcedureTests()
case.setUp()
case.configure()
case.root = Path(sys.argv[1])
with patch('sys.stdin.isatty', return_value=False):
    code = main(['calibrate','leader','--gui','--robot_model=alohamini2pro',
                 '--left_port=/dev/test-left','--right_port=/dev/test-right',
                 '--calibration_dir=' + str(case.root)])
case.assert_passive_and_closed()
case.doCleanups()
sys.exit(code)
"""
    with subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join([
            str(Path("src").resolve()), str(Path("tests").resolve())
        ]), "ALOHAMINI_WORKSPACE": str(tmp_path)},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
    ) as child:
        assert json.loads(child.stdout.readline())["step"] == "connecting"
        home = json.loads(child.stdout.readline())
        assert home["step"] == "home"
        child.stdin.write(json.dumps(dict(action="confirm", prompt_id=home["prompt_id"])) + "\n")
        child.stdin.flush()
        child.wait(timeout=5)
        output, error = child.communicate(timeout=5)
    assert child.returncode == 0, error
    assert json.loads(output.splitlines()[-1])["step"] == "succeeded"
    assert (tmp_path / "am_leader_bi_left.reference.json").is_file()

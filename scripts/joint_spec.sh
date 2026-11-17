#!/usr/bin/env bash
set -euo pipefail

source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_python="$(command -v python)"
dependency_path="$("$runtime_python" -S - <<'PY'
import os
import sysconfig
from pathlib import Path
site = Path(sysconfig.get_path('purelib'))
paths = [site, site / 'cmeel.prefix' / 'lib' / site.parent.name / 'site-packages']
print(os.pathsep.join(str(path) for path in paths if path.is_dir()))
PY
)"
export PYTHONPATH="$source_root/src:$source_root/simulation/mujoco:$source_root/ros2/src/alohamini_bridge:$dependency_path"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd "$source_root"

"$runtime_python" -S - <<'PY'
from pathlib import Path
import alohamini
from alohamini.model import get_robot_model
root = Path.cwd().resolve()
assert Path(alohamini.__file__).resolve().is_relative_to(root)
assert get_robot_model('alohamini2pro').directory == root / 'models/alohamini2pro'
print('Isolated source and model assets:', root)
PY
if [[ "${1:-}" == "calibrate" ]]; then
    exec "$runtime_python" -S -m alohamini "$@"
fi
"$runtime_python" -S scripts/check_joint_spec.py "$@"
"$runtime_python" -S -m pytest -q \
    tests/test_joint_spec.py \
    tests/test_reference_workflow.py \
    tests/test_usb_setup.py \
    tests/test_standard_joint_space.py \
    tests/test_calibration.py \
    tests/test_calibration_procedure.py \
    tests/test_cli.py \
    tests/test_arm_control.py::ArmControllerTests::test_inward_buffer_keeps_requested_target_feedback_and_calibration_distinct \
    tests/test_arm_control.py::ArmControllerTests::test_invalid_inward_buffer_is_rejected_without_device_writes

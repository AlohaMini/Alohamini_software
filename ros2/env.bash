#!/usr/bin/env bash
# Source this file to activate the ROS Conda environment and workspace overlay.
_alohamini_ros_environment() {
    local workspace distro conda_executable profile ID VERSION_ID
    workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || return
    distro="${ALOHAMINI_ROS_DISTRO:-${ROS_DISTRO:-}}"
    if [[ -z "$distro" ]]; then
        source /etc/os-release
        case "$ID:$VERSION_ID" in
            ubuntu:22.04) distro=humble ;;
            ubuntu:24.04) distro=jazzy ;;
            *) echo "Select ALOHAMINI_ROS_DISTRO=humble or jazzy for this system." >&2; return 1 ;;
        esac
    fi
    case "$distro" in
        humble|jazzy) ;;
        *) echo "Supported ROS distributions: humble, jazzy." >&2; return 1 ;;
    esac
    if [[ -n "${ROS_DISTRO:-}" && "$ROS_DISTRO" != "$distro" ]]; then
        echo "Open a fresh terminal before changing ROS distribution." >&2
        return 1
    fi
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        echo "Deactivate the Python venv before loading the ROS Conda environment." >&2
        return 1
    fi
    case "${AMENT_PREFIX_PATH:-}:${CMAKE_PREFIX_PATH:-}:${PYTHONPATH:-}:${LD_LIBRARY_PATH:-}" in
        */opt/ros/*)
            echo "Open a fresh terminal without sourcing /opt/ros before loading RoboStack." >&2
            return 1 ;;
    esac
    conda_executable="${CONDA_EXE:-$(command -v conda)}"
    if [[ ! -x "$conda_executable" ]]; then
        echo "Install Conda before loading the ROS environment." >&2
        return 1
    fi
    profile="$(dirname -- "$(dirname -- "$conda_executable")")/etc/profile.d/conda.sh"
    source "$profile" || return
    conda activate "${ALOHAMINI_ROS_ENV:-alohamini_ros}" || {
        echo "Create the ROS environment: conda create -n alohamini_ros --file $workspace/locks/$distro-linux-64.txt" >&2
        return 1
    }
    if [[ "${ROS_DISTRO:-}" != "$distro" ]]; then
        echo "The selected Conda environment must contain ROS $distro." >&2
        return 1
    fi
    export PYTHONNOUSERSITE=1
    if ! "$CONDA_PREFIX/bin/python" -c '
import sys
from pathlib import Path
import rclpy
assert Path(rclpy.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
'; then
        echo "ROS Python packages must come from the selected Conda environment." >&2
        return 1
    fi
    export ALOHAMINI_ROS_DISTRO="$distro"
    export ALOHAMINI_ROS_PYTHON="$CONDA_PREFIX/bin/python"
    if [[ "${1:-}" != --base && -f "$workspace/install/$distro/local_setup.bash" ]]; then
        source "$workspace/install/$distro/local_setup.bash"
    fi
}

_alohamini_ros_environment "$@"
_alohamini_ros_result=$?
unset -f _alohamini_ros_environment
return "$_alohamini_ros_result" 2>/dev/null || exit "$_alohamini_ros_result"

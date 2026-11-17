#!/usr/bin/env bash
# Source this file in a ROS terminal. Each distribution has its own overlay.
_alohamini_ros_environment() {
    local workspace distro ID VERSION_ID
    workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return
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
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        echo "Deactivate Conda before sourcing the ROS environment." >&2
        return 1
    fi
    if [[ ! -f "/opt/ros/$distro/setup.bash" ]]; then
        echo "ROS $distro is not installed at /opt/ros/$distro." >&2
        return 1
    fi
    source "/opt/ros/$distro/setup.bash" || return
    export ALOHAMINI_ROS_DISTRO="$distro"
    if [[ "${1:-}" != --base && -f "$workspace/install/$distro/local_setup.bash" ]]; then
        source "$workspace/install/$distro/local_setup.bash"
    fi
}

_alohamini_ros_environment "$@"
_alohamini_ros_result=$?
unset -f _alohamini_ros_environment
return "$_alohamini_ros_result" 2>/dev/null || exit "$_alohamini_ros_result"

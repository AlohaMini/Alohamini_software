#!/usr/bin/env bash
set -e
workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$workspace/env.bash" --base
cd "$workspace"
exec "$ALOHAMINI_ROS_PYTHON" -m colcon --log-base "log/$ALOHAMINI_ROS_DISTRO" build \
    --symlink-install \
    --build-base "build/$ALOHAMINI_ROS_DISTRO" \
    --install-base "install/$ALOHAMINI_ROS_DISTRO" \
    --cmake-args "-DPython3_EXECUTABLE=$ALOHAMINI_ROS_PYTHON" "$@"

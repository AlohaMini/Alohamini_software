#!/usr/bin/env bash
set -eo pipefail
EDU_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/humble/setup.bash
cd "$EDU_ROOT/ros2"
/usr/bin/python3 -m colcon build --base-paths src ../simulation/navigation \
  --packages-up-to alohamini_nav_sim --parallel-workers 2 \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

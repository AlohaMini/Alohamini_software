#!/usr/bin/env bash
set -eo pipefail
EDU_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Humble C++ dependencies must come from the system, not an active Conda prefix.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /opt/ros/humble/setup.bash
cd "$EDU_ROOT/ros2"
/usr/bin/python3 -m colcon build --base-paths src ../simulation/navigation \
  --packages-up-to alohamini_nav_sim --parallel-workers 2 \
  --allow-overriding nav2_rviz_plugins \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

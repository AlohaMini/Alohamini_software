#!/usr/bin/env bash
set -eo pipefail
EDU_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
exec bash "$EDU_ROOT/ros2/build.bash" --base-paths src ../simulation/navigation \
  --packages-up-to alohamini_nav_sim --parallel-workers 2 \
  "$@"

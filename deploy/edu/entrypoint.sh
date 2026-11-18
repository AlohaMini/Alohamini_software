#!/usr/bin/env bash
set -e
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
source /opt/ros/${ROS_DISTRO}/setup.bash
source /opt/alohamini/ros2/install/local_setup.bash
exec "$@"

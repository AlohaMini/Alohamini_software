ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-colcon-common-extensions python3-numpy python3-yaml \
    python3-serial python3-zmq ros-${ROS_DISTRO}-ament-cmake-python ros-${ROS_DISTRO}-tf2-ros \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp ros-${ROS_DISTRO}-visualization-msgs && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/alohamini
COPY src/alohamini src/alohamini
COPY models models
COPY LICENSE NOTICE ./
COPY ros2/src/alohamini_core ros2/src/alohamini_core
COPY ros2/src/alohamini_imu ros2/src/alohamini_imu
COPY ros2/src/alohamini_rplidar_c1 ros2/src/alohamini_rplidar_c1
COPY ros2/src/alohamini_sensors ros2/src/alohamini_sensors
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && cd ros2 && \
    MAKEFLAGS=-j2 colcon build --packages-up-to alohamini_sensors --parallel-workers 2 \
      --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 && \
    rm -rf build log
COPY deploy/edu/entrypoint.sh /edu-entrypoint.sh
ENTRYPOINT ["/bin/bash", "/edu-entrypoint.sh"]
CMD ["ros2", "launch", "alohamini_sensors", "sensors.launch.py"]

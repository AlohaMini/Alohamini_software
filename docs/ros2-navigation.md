# Alohamini2 EDU: Sensors, Mapping, and Navigation

**English** | [简体中文](zh-CN/ros2-navigation.md)

EDU adds a C1 lidar and Hiwonder USB IMU to the 2 Pro mechanical, motor, and kinematic configuration. Its model ID is `alohamini2edu`. Device calibration is saved separately; offsets and travel limits are not reused automatically from other robots. Run either native teleoperation/policies or ROS motion control at a time.

Starting a native Host with only `--robot_model alohamini2edu` does not enable lidar, SLAM, recording, or obstacle avoidance automatically.

| Entry point | Automatically started features |
| --- | --- |
| `sensors.launch.py` | Lidar, IMU, mounting TF, scan filtering, and coverage display; no Host connection |
| `mapping.launch.py` | Bridge, EKF, collision monitoring, SLAM, and map saving; external sensors by default |
| `navigation.launch.py` | Bridge, EKF, collision monitoring, AMCL, and Nav2; requires a map |
| `edu_session` | The mapping/navigation workflow above, keyboard control, dedicated RViz, and Host camera subscription; recording requires `--record` |

ROS control is disabled by default and must be explicitly enabled; navigation also needs an initial pose and goal. Native teleoperation commands do not pass through ROS collision monitoring.

Connect lidar and IMU to Raspberry Pi USB. Run the Host and sensor drivers on the Pi, and the Bridge, EKF, and SLAM/Nav2 on the PC. Use the same `ROS_DOMAIN_ID` on both machines and ensure DDS connectivity. Motor serial ports belong exclusively to the Host.

Use ROS domain IDs 0–101 to avoid conflicts with Linux ephemeral ports; see the [ROS 2 domain ID documentation](https://github.com/ros2/ros2_documentation/blob/humble/source/Concepts/Intermediate/About-Domain-ID.rst).

## Installation and devices

The PC uses the `alohamini_ros` Conda environment from [ROS2 installation](install.md#5-ros2-environment): Humble on Ubuntu 22.04, Jazzy on Ubuntu 24.04. Prepare the environment from the current branch's lockfile, then build:

```bash
cd ~/Alohamini
source ros2/env.bash --base
bash deploy/edu/build-pc.sh
source ros2/env.bash
export ROS_DOMAIN_ID=5 ROS_LOCALHOST_ONLY=0
```

The Raspberry Pi runs the Host and sensor drivers. On Debian, use the sensor container described here; on Ubuntu, build `--packages-up-to alohamini_sensors` with the matching ROS distribution. The container's ROS distribution must match the PC.

The Host uses the existing `alohamini_host` environment with `--robot_model alohamini2edu`; complete Home calibration for that robot following the user guide.

Run `alohamini calibrate robot --robot_model alohamini2edu` on the Raspberry Pi. Without explicit serial ports, follow the first-run prompts to plug in the left and right follower-arm USB devices in turn. Board serial numbers determine `/dev/am_arm_follower_left` and `/dev/am_arm_follower_right`; bindings are stored in the workspace at `calibration/devices/follower.json`. Later runs reuse these bindings, so changing port numbers does not swap arms. Home calibration begins only after USB binding. Custom connections require both `--left_port` and `--right_port`.

If cameras are missing or lack persistent device names, follow [Camera troubleshooting and device naming](alohamini.md#missing-cameras-or-persistent-device-names).

The EDU Host enables `forward_left forward_right wrist_right` by default; override with `--cameras`. ROS camera nodes set to `[auto]` follow the Host image streams, publishing front-left and front-right views to `/alohamini/cameras/forward_left/image_raw/compressed` and `/alohamini/cameras/forward_right/image_raw/compressed`. Select these topics directly in RViz Image displays. Without intrinsic calibration, only images are published; no CameraInfo is generated.

Default devices are `/dev/lidar_c1` (460800) and `/dev/imu_usb` (115200); launch arguments can select `/dev/serial/by-id/...`. The serial-port user must belong to `dialout`. The C1 udev template is in `alohamini_rplidar_c1/udev/`; fill in the actual USB serial number before installation. Identical VID/PID values alone cannot distinguish multiple adapters. An IMU without a unique serial number can use `/dev/serial/by-path/...` with a fixed USB port.

If these persistent names are missing, stop the sensor drivers, identify each serial device by unplugging/reconnecting it individually, then inspect its identity:

```bash
ls -l /dev/serial/by-id/ /dev/serial/by-path/
udevadm info --query=property --name=/dev/ttyUSBX
sudoedit /etc/udev/rules.d/70-alohamini-sensors.rules
```

Bind devices with a unique `ID_SERIAL_SHORT` by serial number; bind devices without one to a fixed USB port using `ID_PATH`. Below are templates for both rules. Fill in the measured VID, PID, and identity; use `lidar_c1` or `imu_usb` as the device name:

```text
SUBSYSTEM=="tty", ATTRS{idVendor}=="<VID>", ATTRS{idProduct}=="<PID>", ATTRS{serial}=="<SERIAL_NUMBER>", SYMLINK+="<DEVICE_NAME>", GROUP="dialout", MODE="0660", ENV{ID_MM_DEVICE_IGNORE}="1"
SUBSYSTEM=="tty", ATTRS{idVendor}=="<VID>", ATTRS{idProduct}=="<PID>", ENV{ID_PATH}=="<USB_PATH>", SYMLINK+="<DEVICE_NAME>", GROUP="dialout", MODE="0660", ENV{ID_MM_DEVICE_IGNORE}="1"
```

Choose one rule per device. `ID_SERIAL` may not include a unique serial number: for example, `/dev/serial/by-id/` names cannot distinguish multiple identical CH340 devices without serial numbers. Re-register path bindings after changing USB ports or hubs.

```bash
sudo udevadm control --reload-rules
# Run after reconnecting the corresponding USB device:
ls -l /dev/lidar_c1 /dev/imu_usb
```

Sensor mounts are configured in `models/alohamini2edu/config/sensors.yaml`, using REP-103 `base_link`: X forward, Y left, Z up. The default lidar position is nominal. IMU translation has not been measured; localization uses only Z-axis angular velocity. Store measured extrinsics in a configuration copy under the workspace's `calibration/sensors/` and select it with `sensor_profile:=/absolute/path/sensors.yaml`.

## Start sensors independently

Run on the machine connected to the USB devices:

```bash
ros2 launch alohamini_sensors sensors.launch.py
# Optional: lidar_port:=/dev/serial/by-id/... imu_port:=/dev/serial/by-path/...
```

This entry point publishes only sensor topics and mounting TF and can record alongside native teleoperation. Use `enable_imu:=false` for lidar only, or `publish_mounts:=false` if another sole publisher already provides mounting TF. Do not run duplicate drivers for a device.

`/scan_filtered` preserves the original `/scan` timestamps; stale, future, and out-of-order data are dropped. C1 timestamps come from the driver host; IMU timestamps mark USB reception. Neither provides hardware synchronization. Synchronize the PC and Raspberry Pi clocks; mapping publishes wheel velocities with `state_timestamp_mode:=host_wall` by default.

## Mapping and base keyboard control

Start the Host and Raspberry Pi sensors first, then run on the PC:

```bash
ros2 launch alohamini_navigation mapping.launch.py host:=192.168.44.4
```

If sensors connect to the PC, add `start_sensors:=true` and stop drivers elsewhere. The default `base_only:=true` needs no arm/lift URDF mapping, but still requires a ready Host and control ownership. Use `base_only:=false` to show the actual full robot.

By default, the EKF fuses wheel vx/vy and IMU gz, excluding gravity-containing acceleration and uncalibrated absolute heading. Without an IMU, explicitly add `use_imu:=false` to use wheel-derived yaw rate; this is not an automatic runtime fallback.

Check state and motion directions before enabling ROS control:

```bash
ros2 service call /alohamini_lerobot_bridge/command_enable std_srvs/srv/SetBool '{data: true}'
ros2 run alohamini_navigation base_teleop
```

`w/s`: forward/backward; `a/d`: rotation; `z/x`: lateral motion; Space: stop; `q`: exit. Velocity returns to zero 0.35 s after the last repeated key event; the Bridge separately enforces a 0.5 s `/cmd_vel` timeout. The keyboard does not automatically enable control or depend on leader-arm teleoperation.

SLAM defaults to 5 cm resolution, 8 m maximum range, and loop closure. Its `/scan_slam` input updates after 5 cm translation or 0.05 rad rotation. Collision monitoring uses full-rate `/scan_filtered`. Configuration is in the model's `config/navigation/slam.yaml`; retain `/scan_slam` input and a zero internal translation threshold in custom configurations. Set the RViz fixed frame to `map`.

## Save, resume, and use maps

Save on the PC running SLAM/map_saver:

```bash
ros2 run alohamini_navigation map_session save office
```

Disable Bridge commands and wait for fresh stationary feedback before saving to `~/Alohamini_workspace/maps/office/<UTC_REVISION>/`:

- `map.yaml`, `map.pgm`: Nav2 localization and navigation.
- `slam.posegraph`, `slam.data`: resume mapping with slam_toolbox.
- `manifest.json`, `model_config/`: model, Host session, file checksums, and nominal model configuration. Nominal configuration does not replace runtime parameter overrides or hardware calibration.

Use `--revision v1` to select a new revision; overwriting is prohibited. `--grid-only` exports only the occupancy grid. Failed exports remain in `.incomplete-*` and cannot be loaded as complete maps. Commands stay disabled after saving; re-enable them before driving.

Before exiting or switching modes, disable control and lifecycle nodes, then press Ctrl+C in the launch terminal:

```bash
ros2 run alohamini_navigation map_session stop
```

On Humble, run the stop command above before closing launch; simultaneous TF removal and cleanup may crash collision monitoring.

After stopping the original mapping process, resume from the pose graph with the robot at its original mapping start position:

```bash
ros2 launch alohamini_navigation mapping.launch.py host:=192.168.44.4 \
  pose_graph:=/absolute/path/maps/office/REVISION/slam
```

For occupancy-grid navigation, stop SLAM and start AMCL + Nav2:

```bash
ros2 launch alohamini_navigation navigation.launch.py host:=192.168.44.4 \
  map:=/absolute/path/maps/office/REVISION/map.yaml
```

Set the initial pose in RViz, then enable the Bridge and send a navigation goal. Do not run `mapping.launch.py`, `navigation.launch.py`, or another full-robot bringup simultaneously.

Navigation velocity commands pass through collision monitoring. The forward 180° scan preset permits only forward automatic translation and disables reverse recovery. Observe the surroundings manually during reverse/lateral keyboard motion. The footprint in `config/navigation/nav2.yaml` covers only the nominal folded base, excluding extended arms. After expanding lidar coverage and verifying the hardware footprint, enable corresponding omnidirectional velocities in custom `nav2_params:=/path/to/nav2.yaml` and configure matching collision-monitor coverage with `collision_params:=/path/to/collision.yaml`.

## ROS recording and dual arms

```bash
ros2 run alohamini_navigation map_session record office_episode_001
```

Recording writes to the workspace's `runs/office_episode_001/bag/`, including scans, IMU, wheel velocity, odometry, TF, joints, camera JPEG/intrinsics, sampling-clock correlations, maps, ROS commands, and diagnostics. Ctrl+C flushes files normally; the final manifest lists message counts and missing topics. For dual-front-camera EDU, select cameras with `--cameras forward_left forward_right wrist_right`; `--duration 30` stops after a fixed duration, and `--require-topics /scan /imu/data_raw` checks required topics. Recording enables no motion. Absent topics produce no data.

Use `alohamini_bringup hardware.launch.py robot_model:=alohamini2edu` for full ROS hardware bringup; MoveIt/Joy-Con previews support the same argument. Joint mappings automatically read the EDU Host's Home calibration. Device offsets and travel limits remain in each robot's calibration files.

## Raspberry Pi containers and device checks

The PC uses the Conda ROS environment. If the Raspberry Pi already has matching ROS, build the sensor packages directly; on Debian, use the sensor container. Place this repository at `~/Alohamini` on the Raspberry Pi and run there:

```bash
cd ~/Alohamini
python3 deploy/edu/sensors-container build --ros-distro humble
python3 deploy/edu/sensors-container run --ros-distro humble --domain 5 \
  --lidar /dev/serial/by-id/ACTUAL_LIDAR_DEVICE \
  --imu /dev/serial/by-path/ACTUAL_IMU_DEVICE
```

If the PC uses Jazzy, replace `humble` with `jazzy` in both commands above. Both machines use Cyclone DDS by default.

The container runs sensors only: it installs no SLAM/Nav2, starts no Host, and maps no motor serial ports. It uses host networking and IPC. The native Host continues in its existing environment, enabling both front cameras and the right wrist by default on EDU; use `--cameras` to select other configured cameras. Preview the container command with `--dry-run`; stop it from another terminal with `python3 deploy/edu/sensors-container stop`.

Build the image on the machine running the USB drivers; PC x86-64 binaries cannot run directly on Raspberry Pi ARM64. Only selected devices are mapped. If reconnection changes a device node, restart the container to refresh mappings. Mount measured installation settings with `--profile /absolute/path/sensors.yaml`.

Check clock synchronization with `timedatectl status` on both hosts. After startup, run this read-only check on the PC:

```bash
ros2 run alohamini_navigation edu_check
# Add wheel-velocity checks when mapping/navigation is running:
ros2 run alohamini_navigation edu_check --require-wheel
```

Checks inspect message freshness, valid scans, and IMU values without enabling motors. At rest, IMU acceleration magnitude should be about 9.81 m/s² and angular velocity near zero. Mounting axes, offsets, wheel-velocity signs, odometry scale, and stopping distances still require hardware validation.

## Unified session, cameras, and navigation RViz

After starting the Host and Raspberry Pi sensors, run on the PC:

```bash
ros2 run alohamini_navigation edu_session --host 192.168.44.4 \
  --map-name office --record office_run_001
```

Mapping, dedicated RViz, and Host camera subscriptions start by default. Use `--no-camera` to disable image subscriptions, `--no-rviz` without a desktop, or `--start-sensors` for sensors connected to the PC. Runtime logs are in the workspace's `logs/navigation/`.

| Key | Function |
| --- | --- |
| `e` | Explicitly enable the current ROS control session |
| `w/s`, `a/d`, `z/x` | Forward/backward, rotation, lateral motion |
| `r/f` | Increase/decrease linear speed within 0.03–0.20 m/s |
| Space, `0` | Stop; velocity also returns to zero 0.35 s after the last repeated key event |
| `Shift+S` | Disable control, confirm stationary feedback, and save; press `e` again to resume motion |
| `Shift+X` | Stop and exit after a successful save |
| `q`, Ctrl+C | Stop and exit without saving the map |

On exit, motion is disabled and RViz closes before navigation lifecycle nodes and launch stop; recording then ends and flushes. Navigation mode:

```bash
ros2 run alohamini_navigation edu_session --mode navigation \
  --host 192.168.44.4 --map /absolute/path/maps/office/REVISION/map.yaml
```

RViz includes map, scan, coverage, global/local paths, global/local costmaps, TF, robot, and front camera displays. Set the initial pose with **2D Pose Estimate**, check map/scan alignment, press `e`, then send a goal with **Nav2 Goal**. Navigation topics not yet produced during mapping show no data.

In navigation mode, Nav2 publishes velocity and directional keyboard input is disabled. Space/`0` disables motion and cancels the navigation goal. Manual driving and save shortcuts apply to mapping mode to avoid competing velocity publishers.

Use `--no-rviz` for headless or unattended tasks. Check the corresponding `manifest.json` `status` to determine recording completeness.

Standalone mapping/navigation launches also support `enable_cameras:=true use_rviz:=true`. Cameras use the Host image stream rather than opening USB cameras separately. Override configuration with `ekf_params` and `collision_params`; use `slam_params` for mapping or `nav2_params` for navigation.

## Align with native collection episodes

Alongside native teleoperation recording, start only sensors and camera subscriptions, without mapping/navigation:

```bash
ros2 launch alohamini_camera camera.launch.py host:=192.168.44.4 \
  timestamp_mode:=host_wall publish_raw:=false
```

Once the native episode and rosbag are saved, align recordings from the same Host session:

```bash
ros2 run alohamini_navigation map_session attach /absolute/path/recording/bag \
  /absolute/path/native_dataset --episode-index 0
```

Alternatively, pass `--dataset /path/to/dataset --episode-index 0` to `record` for automatic alignment after recording ends; the native episode must already be fully saved. Platform v3 `meta/safety/` and local `episodes/` layouts are supported.

Results are stored in the dataset's `ros_sensors/episode_000000/`, keyed by native episode/frame index and retaining sample times, frames, units, and validity. Scans use samples at most 150 ms old; IMU uses samples at most 50 ms old. Future samples are excluded, missing data is marked invalid, and invalid scan distances remain null. Alignment is rejected for different Host sessions, missing/discontinuous clock correlations, or incomplete native episodes. With cameras disabled and no recorded Bridge state, no clock correlation is available for automatic alignment.

```python
from alohamini.datasets.ros_sensors import iter_sensor_frames
for frame in iter_sensor_frames(dataset_path, 0):
    scan = frame["sensors"]["scan"]  # valid, sample, age_ns
    imu = frame["sensors"]["imu"]
```

This extension preserves native action/image formats and does not automatically change policy input dimensions. Policies must explicitly read sensor fields and validity. Acceleration includes gravity and must not be used directly as translational acceleration.

## Simulation and replay

Use a dedicated local ROS domain in a separate terminal to avoid mixing hardware topics:

```bash
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1
ros2 run alohamini_navigation edu_session --simulation \
  --map-name simulated_room --record simulated_run_001
```

`simulation/navigation/` uses the `worlds/room.yaml` scene and simulation clock `/clock`, supporting EKF, SLAM, collision monitoring, and Nav2 without connecting to a Host, USB devices, or hardware cameras. Simulation covers planar motion and circular base collisions, excluding tire contact, slip, IMU noise, and extended arms.

```bash
ros2 launch alohamini_nav_sim simulation.launch.py mode:=navigation \
  map:=/absolute/path/maps/simulated_room/REVISION/map.yaml use_rviz:=true
ros2 run alohamini_navigation map_session stop --runtime-mode simulation
```

Inspect a recording or remap its raw data:

```bash
ros2 launch alohamini_navigation replay.launch.py bag:=/absolute/path/recording/bag
ros2 launch alohamini_navigation replay.launch.py bag:=/absolute/path/recording/bag mode:=mapping
ros2 run alohamini_navigation map_session save replayed --runtime-mode replay
ros2 run alohamini_navigation map_session stop --runtime-mode replay
```

Replay consistently uses `/clock` and never republishes motion commands. Remapping consumes raw scans, IMU, wheel velocities, and static TF, excluding old dynamic TF, odometry, and maps. Sensor-only recordings missing `/wheel/twist` or mounting TF support inspection but cannot reconstruct a motion trajectory.

Offline integration check:

```bash
ROS_DOMAIN_ID=43 ROS_LOCALHOST_ONLY=1 /usr/bin/python3 \
  simulation/navigation/test/stack_check.py \
  --output ~/Alohamini_workspace/logs/edu-navigation-check-NEW_DIRECTORY
```

Use `simulation/navigation/test/session_check.py` to check terminal shortcuts and shutdown, providing a new `--output` and the `--navigation-map /path/to/workspace/maps/room/v1/map.yaml` produced by the previous check. Add `--screenshot` with an X display to inspect RViz. Camera frames are labeled offline test images.

Standalone RViz shutdown checks require `xvfb` and `python3-xlib`. They cover SIGINT, SIGTERM, and window close, repeating each three times:

```bash
ROS_DOMAIN_ID=52 ROS_LOCALHOST_ONLY=1 xvfb-run -a /usr/bin/python3 \
  simulation/navigation/test/rviz_shutdown_check.py \
  --output ~/Alohamini_workspace/logs/rviz-shutdown-check
```

Hardware validation must still verify extrinsics, base geometry, stopping distances, fault recovery, and clocks on both machines. Simulation does not replace these measurements.

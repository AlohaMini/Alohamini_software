# ROS2

**English** | [简体中文](zh-CN/ros2.md)

Start the Raspberry Pi Host according to the user guide. PC ROS2, MoveIt, Gazebo, and ROS nodes use the `alohamini_ros` Conda environment; launch entry points invoke Pink IK and the Joy-Con reader from the existing `alohamini` environment.

| PC system | ROS2 | Gazebo |
|---|---|---|
| Ubuntu 22.04 | Humble | Fortress |
| Ubuntu 24.04 | Jazzy | Harmonic |

Source, models, and configuration are shared; build and install directories are separated by ROS distribution. Do not reuse compiled artifacts from another distribution.

## Load the environment

Complete [ROS2 environment installation](install.md#5-ros2-environment), then load it in each ROS terminal:

```bash
source ~/Alohamini/ros2/env.bash
```

The script activates `alohamini_ros`, loads the current distribution's workspace, and uses Cyclone DDS by default. Use `RMW_IMPLEMENTATION` and `CYCLONEDDS_URI` for custom communication settings. Do not also source system `/opt/ros` or a Python venv. See [Installation](install.md#5-ros2-environment) for Python and ROS dependency checks.

## Build

These paths assume the default clone directory; substitute your actual location if different:

```bash
cd ~/Alohamini
bash ros2/build.bash
source ros2/env.bash
```

Dependency installation, building, and node startup are separate checks. Run repository tests in a terminal with the workspace built and loaded:

```bash
cd ~/Alohamini/ros2
test_paths=(src)
if [[ "$ROS_DISTRO" == humble ]]; then test_paths+=(vendor/humble); fi
"$ALOHAMINI_ROS_PYTHON" -m colcon --log-base "log/$ROS_DISTRO" test --base-paths "${test_paths[@]}" \
  --build-base "build/$ROS_DISTRO" --install-base "install/$ROS_DISTRO" \
  --executor sequential --event-handlers console_direct+
"$ALOHAMINI_ROS_PYTHON" -m colcon test-result --test-result-base "build/$ROS_DISTRO" --verbose
```

Bridge, trajectory, and MoveIt integration tests use a local simulated Host; they do not replace validation with real robots, cameras, or Joy-Cons.

## Full-robot state and TF

Supported models: `alohamini2`, `alohamini2pro`, `alohamini2edu`. Complete follower Home calibration using [Leader and follower calibration](alohamini.md#2-leader-and-follower-calibration) before starting the Host. ROS reads Host calibration automatically, with no mapping-file copying required. When changing robots of the same model, use the new robot's Host address.

```bash
ros2 launch alohamini_bringup hardware.launch.py host:=<PI_IP> robot_model:=<MODEL>
```

Cameras enabled on the Host are subscribed by default; add `enable_cameras:=false` for state-only inspection. The command channel starts disabled and sends no motion commands.

- `/joint_states`, `/tf`, `/tf_static`: model joints and frame tree; joint positions use rad or m.
- `/alohamini_lerobot_bridge/measured_joint_states`: dual-arm positions and lift position derived from its encoder. No height sensor or joint torque is provided.
- `/alohamini_lerobot_bridge/derived_wheel_states`: wheel angles integrated from velocity; these are not localization or measured odometry.
- `/alohamini/base_velocity`: base velocity in m/s and rad/s; `/diagnostics`: connection and state validity.

State updates stop when disconnected or feedback is invalid. PC receive time is used by default; `state_timestamp_mode:=host_wall` uses Host time and requires synchronized clocks.

Start the Bridge alone with a parameter file:

```bash
ros2 launch alohamini_bridge bridge.launch.py host:=<PI_IP> params_file:=/absolute/path/bridge.yaml
```

See `ros2/src/alohamini_bridge/config/bridge.yaml` for request-window, timeout, trajectory-tolerance, and base-frame-transform settings. Launch arguments override matching parameter-file entries. `alohamini_bridge runtime.launch.py` starts the model, Bridge, and cameras, without MoveIt or Joy-Con.

## Enable/disable commands and control the base

Ensure the robot's surroundings are safe, the Host is ready, and no teleoperation or other client owns control, then enable:

```bash
ros2 service call /alohamini_lerobot_bridge/command_enable std_srvs/srv/SetBool '{data: true}'
```

Then publish `geometry_msgs/msg/Twist` to `/cmd_vel`: `linear.x/y` use m/s in the base frame; `angular.z` uses rad/s. Default limits are 0.25, 0.25, and 1.0, respectively. Inputs received before enabling are not executed; sending begins only on fresh input. Without new input for 0.5 s, measured from callback reception, the base stops without interrupting arms or lift. Fresh velocity input can resume base control.

Stop control:

```bash
ros2 service call /alohamini_lerobot_bridge/command_enable std_srvs/srv/SetBool '{data: false}'
```

Inspect `stop_pending` and `command_status` in `/diagnostics` for stop-request status, and still confirm the physical robot has stopped. After protection or disconnection, inspect the robot and re-enable. Before switching clients, wait for the Host to release ownership.

## Trajectories and jog

Once the shared command channel is enabled, both arms, grippers, lift, and base share a single Host control ownership:

| Interface | Message type | Position units |
| --- | --- | --- |
| `/left_arm_controller/follow_joint_trajectory`, `/right_arm_controller/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` | rad |
| `/lift_controller/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` | m (URDF `vertical_move`, not height above the ground) |
| `/left_gripper_controller/gripper_cmd`, `/right_gripper_controller/gripper_cmd` | `control_msgs/action/GripperCommand` | rad |

Trajectories use timed position points with linear interpolation and a zero start timestamp. Position-error tolerances and cancellation are supported; a new goal for the same control group preempts the old one. Velocity/acceleration fields are not feedforward inputs; nonzero velocity/acceleration tolerances and effort feedforward are rejected. Gripper `max_effort` must be zero. The Host enforces current protection; returned `effort=NaN` means no force estimate is available. Contact preventing target completion returns `stalled=true`, without claiming a successful grasp.

`/left_arm_controller/joint_jog` and `/right_arm_controller/joint_jog` accept `control_msgs/msg/JointJog`, using the standard six-joint order and rad increments. `/lift_controller/joint_jog` uses the velocity field of `vertical_move` (m/s); a nonzero value selects direction for a default 50 mm lookahead target, without guaranteeing constant motion at that speed. Supply a current ROS timestamp and send continuously. Expired inputs are discarded; arms hold the latest feedback position when input stops. For lift stopping, the Host first zeros velocity, then locks subsequent local height feedback. The lift needs a valid height reference.

## MoveIt

Connect to the Host, publish real state, and start MoveIt:

```bash
ros2 launch alohamini_bringup hardware.launch.py host:=<PI_IP> robot_model:=<MODEL> \
  enable_moveit:=true use_rviz:=true
```

Check that the model pose matches the hardware, then enable execution through `command_enable`. MoveIt uses the arm, gripper, and lift actions above; starting MoveIt alone does not enable motion.

For offline planning without a robot connection:

```bash
ros2 launch alohamini_moveit_config plan_only.launch.py
```

Offline state, TF, and planning services use `/alohamini_plan_only` and cannot execute on hardware. Add `use_rviz:=false` without a graphical interface.

Model and offline validation:

```bash
ros2 run alohamini_validation validate_assets
# After starting plan_only above, run in another terminal:
ros2 run alohamini_validation validate_moveit
ros2 run alohamini_validation validate_tf
```

Check model/FK/collision geometry, MoveIt, and TF in sequence. The last two default to `/alohamini_plan_only`; add `--ros-args -r __ns:=/` to inspect a hardware ROS graph. The tools send no actions; inspect the hardware motion area separately.

Both entry points check validity using the same reference, folded, closed-gripper, and known-collision poses. These cases do not replace physical clearance checks.

## Joy-Con teleoperation

Supported models: `alohamini2`, `alohamini2pro`, `alohamini2edu`. Pair Joy-Cons with the PC, install the [`joycon` extension](install.md#3-optional-extensions), and build the ROS packages as described at the start of this guide:

```bash
source ~/Alohamini/ros2/env.bash
ros2 launch alohamini_joycon_teleop preview.launch.py
```

Keep controllers still for about two seconds after startup for IMU calibration. The reader runs separately in `alohamini`; ROS nodes use `alohamini_ros`. Preview needs no Host and also starts planning-only MoveIt; state, TF, and FK/IK services use `/alohamini_plan_only`. Do not start `plan_only.launch.py` separately. Add `start_native_reader:=false` without controllers or `use_rviz:=false` without a desktop.

The launch entry point starts a Pink IK worker using `conda run -n alohamini python` by default. Select an interpreter with the `kinematics` extension installed using `ik_python:=/absolute/path/python`. The worker computes joint targets only; the ROS Bridge still sends Host commands. It exits with launch, and an unexpected worker exit terminates that launch.

Full hardware startup (do not start duplicate bringup or reader processes):

```bash
ros2 launch alohamini_bringup hardware.launch.py host:=<PI_IP> robot_model:=<MODEL> \
  enable_joycon:=true use_rviz:=true
```

Add `enable_moveit:=true` if MoveIt services are needed; only one Joy-Con RViz instance is started.
The reader defaults to `alohamini`. Select an interpreter with `native_python:=/absolute/path/python`, or connect an existing reader with `start_native_reader:=false`.

After checking the pose, manually enable through `command_enable` as above. Release all buttons and center the sticks before operating. After protection or disconnection recovery, repeat inspection, enabling, and centering. Do not run preview and hardware readers competing for the same controllers.

- Each controller operates its corresponding arm. Hold `SL/SR` and use the stick to translate the TCP; rotate the controller to change relative orientation. Arm-base axes are `+X` toward the robot's left, `-Y` forward, and `+Z` up.
- During arm control, the shoulder button moves up and pressing the stick moves down. `ZL/ZR` toggles the gripper; `Capture/Home` relocks the current pose.
- Without `SL/SR`, sticks control the base. Right shoulder plus horizontal stick controls turning; left shoulder plus vertical stick controls the lift.

Default differential IK does not plan around collisions; provide safe clearance. Host current protection and ROS joint limits remain active and do not replace obstacle avoidance.
Differential tracking uses Pink/Pinocchio/DAQP, limiting joint position, velocity, and acceleration according to the actual control period. `ik_max_joint_acceleration_rad_s2` defaults to 8; `dls_*` gain, velocity, displacement, and limit-margin parameters remain effective. The RPC timeout defaults to 0.05 s; failure stops new arm motion and latches a fault. Parameters are in `ros2/src/alohamini_joycon_teleop/config/joycon.yaml`.

Record raw controller input and replay offline (relative output paths are placed in the workspace's `logs/`):

```bash
ros2 run alohamini_joycon_teleop joycon_input_log record joycon.ndjson
ros2 run alohamini_joycon_teleop joycon_input_log replay joycon.ndjson
```

Recording defaults to hardware input port `5567`; add `--endpoint tcp://127.0.0.1:5568` for preview input. Replay is only for preview without a running reader; hardware mode rejects replay-tagged inputs. Reader diagnostics are in the workspace's `logs/joycon_sticks.log`.

## Gazebo simulation

`simulation/gazebo/` selects Gazebo by ROS distribution: Fortress for Humble, Harmonic for Jazzy. Gazebo dependencies are installed with `alohamini_ros`. Build and launch using distribution-specific directories:

```bash
cd ~/Alohamini
bash ros2/build.bash --base-paths src ../simulation/gazebo \
  --packages-up-to alohamini_gazebo
source ros2/env.bash
ros2 launch alohamini_gazebo simulation.launch.py
```

After building, run the offline check to validate the simulation clock, joints, controllers, and lift trajectory automatically, shutting down simulation on completion:

```bash
python simulation/gazebo/test/stack_check.py
```

Add `headless:=true` without a graphical interface. Use `lift_pick_place_demo.launch.py` for the automatic lift pick-and-place demo; it exits after success or failure.

Simulation commands, state, TF, and clock use `/alohamini_sim`, without a hardware bridge. Geometry is shared with platform models. The base uses planar motion control, arms use ideal position actuation, and grasping uses fixed attachments. This does not validate real wheel-ground friction, gripping force, or gravity compensation.

## Subscribe to cameras independently

```bash
ros2 launch alohamini_camera camera.launch.py host:=<PI_IP>
```

By default, subscriptions follow cameras actually enabled on the Host. Add `cameras:="[forward, wrist_right]"` to select a subset. The node only receives images, sends no motion commands, and opens no local cameras. It recognizes a new session after Host restart.

Topics for each camera:

- `/alohamini/cameras/<name>/image_raw/compressed`: standard JPEG.
- `/alohamini/cameras/<name>/image_raw`: RGB, decoded only when subscribers exist.
- `/alohamini/cameras/<name>/camera_info`: published when usable intrinsics are available.

PC receive time is used by default. For Host capture time, add `timestamp_mode:=host_wall` and synchronize PC/Raspberry Pi clocks first. Moving-camera calibration must not treat receive time as exposure time.

## Calibration files

The default directory is `~/Alohamini_workspace/calibration/cameras/`, respecting `ALOHAMINI_WORKSPACE`. Override with `calibration_dir:=/absolute/path`.

- `intrinsics/<CAMERA_NAME>.yaml`: ROS CameraInfo format; name, dimensions, and optical frame must match. Candidate or unaccepted files are not used to publish CameraInfo. Images remain available without intrinsics.
- `extrinsics/<FILE_NAME>.yaml`: extrinsics from `mount_link` to `optical_frame`; TF publication is disabled by default.

Explicitly publish accepted extrinsics:

```bash
ros2 launch alohamini_camera camera.launch.py host:=<PI_IP> \
  enable_extrinsics:=true extrinsics:=forward.yaml,wrist_right.yaml
```

Extrinsics need a status starting with `accepted_`, `T_mount_link_from_camera_optical.xyz_m`, and explicit `quaternion_xyzw` or `quaternion_wxyz`. Candidates are rejected by default; frames in calibration files are not overridden.

## Lift measurement and asset import

Arms use Host Home calibration. The lift uses the Host height reference in millimeters, automatically converted to metric model joint coordinates. Restart the Host and ROS bringup after recalibration.

Import existing camera, lift, and other assets without overwriting files or writing servos:

```bash
ros2 run alohamini_calibration import_calibration \
  --source ~/alohamini_ros2/src/alohamini_calibration/config \
  --output ~/Alohamini_workspace/calibration/imports/previous_robot
```

Import does not apply results automatically. `import_manifest.json` records sources and checksums. Verify the current devices first; validate candidate intrinsics/extrinsics separately.

Use this read-only tool to check lift lead and direction on `alohamini2pro`; it does not enable, home, or write EEPROM.

Lift: confirm the Host has a valid height reference and the mechanism is physically at Home, then start:

```bash
ros2 run alohamini_calibration calibrate_lift_axis --host <PI_IP>
```

Follow prompts to move the lift using existing controls. Once stopped, measure its height relative to Home and enter millimeters; collect at least two different heights. Submit an empty line to finish and generate a fit report and candidate mapping. The matching `.samples.yaml` stores completed samples. Results are not automatically applied and do not change Host lead, direction, or soft limits.

## Camera calibration

Tools capture images/TF and compute candidate parameters without controlling the robot. Captures and results default to `~/Alohamini_workspace/calibration/cameras/`, respecting `ALOHAMINI_WORKSPACE`; the terminal shows actual paths. `--output` selects a new capture directory or result file without overwriting existing content. Samples already listed in the manifest survive capture interruption.

Prepare a ChArUco board:

```bash
AM_BOARD="$(ros2 pkg prefix --share alohamini_calibration)/config/cameras/boards/charuco_9x7_26mm_18p7_ids300_330.yaml"
ros2 run alohamini_calibration generate_charuco_board --board "$AM_BOARD"
```

Print the PDF under the workspace's `calibration/cameras/boards/` at 100% actual size, verifying a 26 mm square side.

Capture and solve intrinsics (right wrist camera):

```bash
ros2 run alohamini_calibration capture_camera_calibration \
  --host <PI_IP> --camera wrist_right --count 40
ros2 run alohamini_calibration calibrate_camera_intrinsics \
  --capture-dir <INTRINSICS_CAPTURE_DIR> --board "$AM_BOARD" \
  --camera wrist_right --frame-id right_camera_optical
```

Enable the corresponding camera on the Host. Vary the board's distance, angle, and image position. Preview is enabled by default; add `--no-preview` without a desktop. Capture at the resolution used in operation.

Before hand-eye calibration, verify arm URDF mappings and synchronize PC/Raspberry Pi clocks. Use Host timestamps for both full-robot state and cameras:

```bash
ros2 launch alohamini_bringup hardware.launch.py host:=<PI_IP> robot_model:=<MODEL> \
  state_timestamp_mode:=host_wall camera_timestamp_mode:=host_wall
```

For a right wrist camera fixed to the arm and a board fixed in the environment:

```bash
ros2 run alohamini_calibration capture_hand_eye_samples \
  --preset wrist_right
ros2 run alohamini_calibration calibrate_hand_eye \
  --capture-dir <HAND_EYE_CAPTURE_DIR> --intrinsics <RIGHT_WRIST_INTRINSICS_FILE>
```

`--preset` accepts `forward`, `backward`, `chest`, `wrist_left`, `wrist_right`, or a YAML path. Presets supply calibration type, topics, frames, board definition, and sampling thresholds; explicit arguments take precedence. Capture directories save the effective configuration and board file for automatic solver loading. Legacy capture directories without board metadata still require `--board` and `--optical-frame`.

Change arm poses using existing controls; sampling occurs automatically after each pose settles. Include multiple rotation axes. For fixed cameras, use `eye_to_hand`: fix the board to the arm, keep the camera, base, and lift stationary during capture, and specify the corresponding topics and frames. The tool looks up TF at image timestamps; it does not automatically verify clock synchronization or timestamp configuration.

Candidates are saved under `intrinsics/` and `extrinsics/`. Check reprojection error, alignment at independent poses, and physical dimensions before changing status to an `accepted_` prefix and installing as described above. Name intrinsics `<CAMERA_NAME>.yaml`. Do not enable extrinsics solely because the solver succeeded.

## EDU navigation

See [EDU navigation](ros2-navigation.md) for lidar, IMU, mapping, localization, and autonomous navigation setup and usage.

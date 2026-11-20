# AlohaMini User Guide

**English** | [简体中文](zh-CN/alohamini.md)

Complete [installation](install.md) first and connect the PC and Raspberry Pi to the same LAN. Replace `<MODEL>` with the actual model and `<PI_IP>` with the Raspberry Pi address below.

## 1. Models and wiring

| Model | Each follower arm | Leader calibration ID |
| --- | --- | --- |
| `alohamini1` | SO-ARM, 5 DOF + gripper | `so101_leader_bi` |
| `alohamini2` | AM-ARM, 6 DOF + gripper | `am_leader_bi` |
| `alohamini2pro` | AM-ARM Pro, 6 DOF + gripper | `am_leader_bi` |
| `alohamini2edu` | AM-ARM Pro, 6 DOF + gripper | `am_leader_bi` |

EDU uses the Pro leader arms, follower arms, and motors, adding a C1 lidar and Hiwonder USB IMU. The dual-front-camera version uses separate left and right forward cameras.

AM-ARM leaders use 5 V; followers use 12 V.

| Device | Connection | Default device name |
| --- | --- | --- |
| Left and right leaders | PC USB | `/dev/am_arm_leader_left`, `/dev/am_arm_leader_right` |
| Left follower, base, and lift | Raspberry Pi USB | `/dev/am_arm_follower_left` |
| Right follower | Raspberry Pi USB | `/dev/am_arm_follower_right` |
| Forward and right wrist cameras | Raspberry Pi USB | `/dev/am_camera_forward`, `/dev/am_camera_wrist_right` |
| Dual forward cameras (EDU) | Raspberry Pi USB | `/dev/am_camera_forward_left`, `/dev/am_camera_forward_right` |
| C1 lidar (EDU) | Raspberry Pi USB | `/dev/lidar_c1` |
| USB IMU (EDU) | Raspberry Pi USB | `/dev/imu_usb` |

For EDU sensor naming, driver startup, mapping, and navigation, see [EDU sensors, mapping, and navigation](ros2-navigation.md).

## 2. Leader and follower calibration

Use the Home procedure below for AM-ARM models (`alohamini2`, `alohamini2pro`, and `alohamini2edu`). Stop the Host, teleoperation, and other serial programs first. Secure both arms and support the lift.

Press Enter to reuse an existing calibration, or enter `c` to redo it. Calibration survives power cycles; normal startup does not require resetting the zero position.

On first use, configure device access groups and log in again:

```bash
sudo usermod -aG dialout,video "$USER"
```

If the current terminal still reports insufficient serial permissions, run `newgrp dialout` and reactivate the Conda environment.

Home pose:

| Servo | Pose |
| --- | --- |
| 1: shoulder pan, 5: wrist yaw, 6: wrist roll | Align to the center position |
| 2: shoulder lift, 3: elbow flex, 4: wrist flex | Fold inward until the corresponding links or housing surfaces just touch |
| Gripper | Closed |

### Leader calibration and serial-number registration (PC)

```bash
conda activate alohamini
alohamini calibrate leader --robot_model <MODEL> --teleop.id am_leader_bi
```

On first use, follow the prompts to register both leader USB controller serial numbers and bind device names: unplug both leader USB connections and press Enter; connect the left arm first, leave it connected after identification, then connect the right arm. Enter the system `sudo` password when prompted.

One PC can store multiple leader sets. Register and calibrate a new set with the command above; previously registered leaders automatically use their saved records when reconnected. Connect only one set at a time.

For first-time or repeated calibration, place both leaders at Home together, hold them still, and press Enter once. Changing the paired robot does not require recalibrating the leaders.

### Follower calibration and serial-number registration (Raspberry Pi)

```bash
conda activate alohamini_host
alohamini calibrate robot --robot_model <MODEL>
```

On first use, follow the prompts to register both follower USB controller serial numbers and bind device names: unplug both follower USB connections and press Enter; connect the left arm first, leave it connected after identification, then connect the right arm. Enter the system `sudo` password when prompted to create `/dev/am_arm_follower_left` and `/dev/am_arm_follower_right`.

Bindings are stored in `calibration/devices/follower.json` in the Raspberry Pi workspace; udev rules are installed in `/etc/udev/rules.d/`. Serial numbers identify the same arms after port changes or restarts, so left/right confirmation is not repeated. Register again when replacing a controller.

Home calibration starts only after USB binding completes. For first-time or repeated calibration, place both followers at Home together, hold them still, and press Enter once.

Use `alohamini find-port` to inspect serial ports. Custom ports require both `--left_port` and `--right_port`.

## 3. Cameras

On the Raspberry Pi, stop the Host using the cameras before discovery:

```bash
conda activate alohamini_host
alohamini find-cameras opencv
```

Add `--record-time-s 0` to list devices without saving a preview.

### Missing cameras or persistent device names

Stop the Host and other camera applications first, then run on the Raspberry Pi:

```bash
ls -l /dev/video* /dev/am_camera*
alohamini find-cameras opencv --record-time-s 0
```

- No matching `/dev/video*`: check power, USB cables, and hubs; reconnect and inspect again.
- Device exists but cannot be opened: use `fuser -v /dev/videoX` to check ownership and verify that the user belongs to the `video` group.
- Camera is found but `/dev/am_camera_<NAME>` is missing: add a device naming rule. `videoX` numbers may change after startup or reconnection.

Reconnect cameras one at a time or inspect previews to identify each mounting position and working `/dev/videoX`, then query its identity:

```bash
udevadm info --query=property --name=/dev/videoX
cat /sys/class/video4linux/videoX/index
sudoedit /etc/udev/rules.d/70-alohamini-cameras.rules
```

Create one rule per camera. Replace `<VID>`, `<PID>`, and `<SERIAL>` with the reported `ID_VENDOR_ID`, `ID_MODEL_ID`, and `ID_SERIAL_SHORT`; use the reported `<INDEX>` and a Host camera `<NAME>` listed below:

```text
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="<VID>", ATTRS{idProduct}=="<PID>", ATTRS{serial}=="<SERIAL>", ATTR{index}=="<INDEX>", SYMLINK+="am_camera_<NAME>", GROUP="video", MODE="0660"
```

Without a unique serial number, replace `ATTRS{serial}=="<SERIAL>"` with `ENV{ID_PATH}=="<REPORTED_ID_PATH>"` and keep the camera on the same USB port. Do not distinguish identical cameras by VID/PID alone, or assign the same name to different cameras through multiple rules.

After saving, reload the rules, reconnect the affected camera, and verify:

```bash
sudo udevadm control --reload-rules
# Run after reconnecting the camera USB cable:
ls -l /dev/am_camera*
alohamini find-cameras opencv
```

Confirm that each preview matches its persistent name and physical mounting position before starting the Host. Persistent names identify devices; camera intrinsics and extrinsics are calibrated separately.

## 4. Start the Raspberry Pi Host

> Startup enables servo torque and homes the lift against its lower stop.

```bash
conda activate alohamini_host
alohamini host --robot_model <MODEL>
```
The Host control rate is 50 Hz. Cameras are requested at 640×480 and 30 Hz.

Supported camera names are `forward`, `forward_left`, `forward_right`, `backward`, `chest`, `wrist_left`, and `wrist_right`, mapped to `/dev/am_camera_<NAME>`. EDU defaults to dual forward cameras and the right wrist (`forward_left forward_right wrist_right`); other models default to one forward camera and the right wrist. Override model defaults with `--cameras`, separating names with spaces.

ROS 2 and other clients can subscribe to enabled cameras; see [ROS 2](ros2.md).

| Setting | Append to the startup command |
| --- | --- |
| Forward camera only | `--cameras forward` |
| EDU dual forward cameras only | `--cameras forward_left forward_right` |
| No cameras | `--cameras` without names |
| Actual rates and timings | `--profile_timing` |
| Serial ports | `--left_port /dev/ttyACM0 --right_port /dev/ttyACM1` |
| Robot calibration file | `--calibration /absolute/path/AlohaMiniRobot.json` |

## 5. Inspect state from the PC

Keep the Host running and run on the PC:

```bash
conda activate alohamini
alohamini inspect --host <PI_IP> --model <MODEL>
```

Reads one state snapshot without sending motion commands. On connection failure, check the Host, IP address, and model on both sides. See the [client API](host-protocol.md) for fields, units, and Python usage.

## 6. PC teleoperation

Keep the Host running and run on the PC:

```bash
conda activate alohamini
alohamini teleoperate \
  --host <PI_IP> \
  --robot_model <MODEL> \
  --teleop.id am_leader_bi
```

Use the same `--teleop.id` as calibration. Defaults are 50 Hz control, 30 Hz image requests, and Rerun preview enabled.

| Key | Action |
| --- | --- |
| `w` / `s` | Base forward / backward |
| `z` / `x` | Base left / right |
| `a` / `d` | Base rotate left / right |
| `t` / `g` | Increase / decrease speed level |
| `u` / `j` | Lift up / down |
| `q`, Esc, Ctrl+C | Exit teleoperation |

**Keyboard control requires an X11 desktop.**

Add `--no_keyboard` for leaders only, `--no_leader` for keyboard only, or `--no_preview` to disable preview.

The PC terminal shows teleoperation state and tracking; the Raspberry Pi terminal reports Host rates, timings, and hardware faults. Run only one motion-control client at a time. **Exit teleoperation before starting data collection.**

## 7. Local data collection on the PC

Select the cameras for this session and start the Host on the Raspberry Pi. Keep it running throughout collection:

```bash
conda activate alohamini_host
alohamini host --robot_model <MODEL> --cameras <CAMERA_LIST>
```

Separate `<CAMERA_LIST>` entries with spaces, for example `forward_left wrist_right`. Available names are listed under [Host configuration](#4-start-the-raspberry-pi-host). Collection records all cameras enabled by the Host, using separate data fields and video directories for each name; left and right views are not merged.

Then start collection on the PC:

```bash
conda activate alohamini
alohamini record \
  --host <PI_IP> \
  --robot_model <MODEL> \
  --teleop.id am_leader_bi \
  --dataset pickup_01 \
  --task "pick up the object" \
  --num_episodes 1 \
  --episode_time 8 \
  --reset_time 3
```

Collection defaults to 30 Hz. Add `--display_data` to enable preview.
`--profile_timing` enables capture, save-stage, and per-camera video-processing timings. It is off by default, and detailed video timings are not collected when disabled.

| Option | Purpose |
| --- | --- |
| `--video-encoder nvenc` | NVIDIA GPU H.264 encoding; requires a compatible driver and PyAV NVENC support. Default: `cpu` |
| `--no-background-video-encoding` | Disable background encoding, which is enabled by default |
| `--video-encoding-workers 2` | Limit concurrently encoded cameras; defaults to the number of enabled cameras |
| `--video-encoding-memory-mib 2048` | Encoder child-process RAM limit (MiB); excludes the full recording application and GPU memory |

NVENC and CPU encoding can produce different pixels and file sizes.

| Key | Action |
| --- | --- |
| Right arrow | End the current phase early |
| `R` / Left arrow | Delete temporary data for the current episode and rerecord |
| Esc, `q`, Ctrl+C | End collection |

Teleoperation remains available during reset, but data is not recorded. On exit, the application attempts to save valid frames already collected.

Data defaults to `~/Alohamini_workspace/datasets/pickup_01/`. Use `--root` for another location or `--resume` to append to an existing dataset; model, calibration, cameras, task, and frame rate must match.

Use `dataset check --decode-videos` for a full content check.

To fix selected joints for a task, first place the robot by teleoperation, exit teleoperation, and append the following to the collection command:

```bash
--fixed-dimensions arm_right lift_axis
```

Supported groups are `arm_left`, `arm_right` (including the gripper), `lift_axis`, and `base`. Individual action fields such as
`arm_right_elbow_flex.pos` and `lift_axis.height_mm` are also supported. Fixing an entire arm removes the need to connect its leader. Cameras on a fixed arm are still recorded normally.
Fixing the base commands zero velocity; it does not hold a world-frame position.
To keep a wrist camera fixed in space, also select any parent axes that move it, such as the lift, as required by the task. Fixing an arm does not automatically lock the lift.

When continuing with the same collection command and `--resume`, `--fixed-dimensions` can be omitted; saved targets are restored automatically.

Recordings use the LeRobot v3 dataset format.

| File | Contents |
| --- | --- |
| `meta/info.json`, `meta/alohamini.json` | Field definitions, model, calibration, and units |
| `data/chunk-*/file-*.parquet` | State, actions, indices, servo feedback, and validity masks |
| `videos/observation.images.*/chunk-*/file-*.mp4` | Playable video for each camera |
| `meta/episodes/`, `meta/tasks.parquet`, `meta/stats.json` | Episode indices, tasks, and statistics |
| `meta/safety/episode_*.jsonl` | Protection events, command identities, and actual sampling times |

### Validation and recovery

```bash
alohamini dataset check ~/Alohamini_workspace/datasets/pickup_01
```

Add `--decode-videos` to validate video contents, or `--output-json <NEW_FILE>` to save a report. `VALID` confirms structural checks; inspect warnings about missing frames and protection events before training.

If saving is interrupted, retain the original directory and recover into a new directory:

```bash
alohamini dataset recover ~/Alohamini_workspace/datasets/pickup_01 \
  --output ~/Alohamini_workspace/datasets/pickup_01_recovered
```

Only complete frames are recovered; original data is unchanged. Keep any `*.pending/` directories left by interruption for recovery; do not delete them manually.
Pressing `R` or Left arrow to rerecord deletes temporary images and logs for the current episode.

Repair broken dataset indices or media references into a new directory:

```bash
alohamini dataset repair ~/Alohamini_workspace/datasets/task_demo \
  --output ~/Alohamini_workspace/datasets/task_demo_repaired
```

The original data is retained while unambiguous indices and media references are rebuilt. Repair refuses missing images or ambiguous pairings. Repaired datasets cannot be used for appended recordings, and repair does not remove actual sampling gaps or timing errors.

For Rerun evaluation viewing, comparison with training demonstrations, and report export, see [Comparing evaluations with training demonstrations](training.md#8-compare-evaluation-runs-and-training-demonstrations).

### Export and field selection

To save a separate copy or select state fields, run in the `alohamini` environment:

```bash
alohamini dataset export ~/Alohamini_workspace/datasets/pickup_01 \
  --output ~/Alohamini_workspace/datasets/pickup_01_copy \
  --format lerobot-v3
```

Default state includes both arm joint positions, base velocity, and lift height (18 dimensions for AM-ARM); actions and feedback are retained. `--state` selects input fields. For vision-only copies and training, see [Training and deployment](training.md#export-and-vision-only-copies).

Videos in current-format datasets are copied unchanged during export, without re-encoding. Recovery and export require new directories; `.pending-*` marks an incomplete export.

### Action replay

Replay moves the robot; it does not just play video. Exit teleoperation or recording, place the robot at the demonstration starting pose, and clear the motion area.

```bash
alohamini replay --dataset pickup_01 \
  --host <PI_IP> --robot_model <MODEL> --episode 0
```

Use `--root` for a custom directory. Replay executes actions; model, action units, and calibration must match the Host.

Replay defaults to the dataset frame rate. Override it with `--fps`, or adjust speed with `--speed`. Low rates or skipped frames may miss critical actions, and exact base-path reproduction is not guaranteed.

## 8. Working files

The default workspace is `~/Alohamini_workspace/`, separate from source code:

| Subdirectory | Contents |
| --- | --- |
| `calibration/robots/`, `calibration/teleoperators/` | Robot and leader calibrations |
| `calibration/devices/` | Leader/follower USB serial bindings and rule copies |
| `datasets/` | Local datasets |
| `runs/` | Training and evaluation artifacts |
| `incoming/` | Returned data awaiting inspection |
| `logs/` | Runtime logs and debug output |

Run `alohamini paths` to inspect paths. Set `ALOHAMINI_WORKSPACE` to an absolute path to change the root. Existing files are not moved automatically, and PC and Raspberry Pi files are not synchronized automatically.

## 9. Debugging tools

Inspect servo state:

```bash
python examples/debug/motors.py get_motors_states --port /dev/ttyACM0
```

Queries IDs 1–22 without changing torque. `POS` and `OFF` are raw ticks; `CURR(MA)` is current in mA. Missing values appear as `-`; `servo_error` indicates servo fault bits.

| Command | Purpose |
| --- | --- |
| `python examples/debug/test_cv.py --camera /dev/am_camera_forward` | Check one camera |
| `python examples/debug/test_cuda.py` | Check PyTorch and the GPU |
| `python examples/debug/test_input.py` | Check keyboard input |
| `python examples/debug/test_dataset.py /path/to/frames.parquet` | Inspect dataset columns and values |
| `python examples/debug/test_network.py --help` | Check connectivity to a specified HTTP(S) address |
| `python examples/debug/test_mic.py` | Record audio and check levels; requires the `audio-debug` extra and system PortAudio |

Camera previews, audio recordings, and other debug outputs default to `logs/debug/` in the workspace. See [optional extensions](install.md#3-optional-extensions) for microphone dependencies.

<details>
<summary>Servo maintenance: torque release, ID changes, and phase changes</summary>

```bash
python examples/debug/motors.py reset_motors_torque --port /dev/ttyACM0 --id 1
python examples/debug/motors.py configure_motor_id --help
python examples/debug/motors.py configure_motor_phase --help
```

Without `--id`, the torque-release command handles all identified servos with IDs 1–22 on the serial port. Connect only the servo being configured when changing its ID, and check device configuration and calibration afterward. Phase values must match the specific model. After maintenance errors, verify device state before starting motion.

</details>

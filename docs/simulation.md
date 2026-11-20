# Simulation

**English** | [简体中文](zh-CN/simulation.md)

Complete [simulation environment installation](install.md#4-simulation) first. Run these commands from the repository root in the corresponding environment.

## MuJoCo

Open the full robot with EGL device 0 for rendering:

```bash
conda activate alohamini
alohamini-mujoco view --window-egl-device 0
```

Omit `--window-egl-device 0` to use the desktop's default OpenGL device. Robot assets are installed with the platform; the base is fixed by default.

Set the initial lift position and start paused:

```bash
alohamini-mujoco view --window-egl-device 0 \
  --initial-positions '{"vertical_move":0.2}' --paused
```

`--initial-positions` takes URDF joint names. Revolute joints use rad; the lift uses m in `[-0.3, 0.3]`.

Open the black-table-and-cube scene; edit the JSON to adjust object and robot initial states:

```bash
alohamini-mujoco view --window-egl-device 0 \
  --scene-config examples/simulation/research.json --paused
```

## Window controls

Expand `Control` on the right and drag sliders to set joint targets. Targets are stored while paused; press Space to play and let the model follow. `Joint` shows current joint positions; dragging it also sets the corresponding target. Motion respects joint, velocity, and motor-force limits.

| Input | Effect |
| --- | --- |
| Space | Start or pause simulation |
| Right arrow | Advance one physics step while paused |
| F9 | Reset and pause |
| F8 | Toggle visual models and collision geometry |
| T | Toggle transparency |
| F6 | Toggle frame display |
| Tab / Shift + Tab | Show or hide the left/right panel |
| Left drag | Rotate the view |
| Right drag / Shift + right drag | Pan in the vertical/horizontal plane |
| Scroll wheel | Zoom |
| Right double-click | Move the view center to the clicked point |

In standard `view`, left double-click selects a link; Ctrl + right drag applies translational force, Ctrl + left drag applies torque, and Ctrl + Shift + right drag applies force in the horizontal plane.

The `alohamini-mujoco` window saves `Print data`, model exports, and screenshots under `~/Alohamini_workspace/simulation/viewer/` by default, respecting `ALOHAMINI_WORKSPACE`. Relative command paths resolve from the launch directory.

## ManiSkill and RoboTwin

View the same scene in the ManiSkill environment:

```bash
conda activate alohamini
alohamini-maniskill view --scene-config examples/simulation/research.json
```

The window targets 60 FPS; adjust with `--render-fps 30`. Scene `control_frequency_hz` sets control frequency, and `timestep_s` sets the physics step.

Linux X11 systems with hybrid graphics default to an isolated window, preserving the native viewer. Use `--window-mode native` for a direct window or `--window-mode isolated` to force isolation.

View in the RoboTwin environment, with `--source` pointing to its source directory:

```bash
conda activate alohamini_robotwin
alohamini-robotwin view --source /path/to/RoboTwin \
  --scene-config examples/simulation/research.json
```

Add `--headless --steps 1000` to disable rendering and advance 1000 physics steps; the default step is 1 ms. Both currently support one CPU physics environment.

## LIBERO scenes and replay

After extracting the asset archive according to [Installation](install.md#libero-scenes-and-reference-data), set its path in the current terminal:

```bash
export SIM_ASSETS="$HOME/Alohamini_workspace/simulation/alohamini-mujoco-assets-v1"
export LIBERO_BUNDLE="$SIM_ASSETS/libero_spatial"
export REFERENCE_DATASET="$SIM_ASSETS/reference/task8"
export COLLISION_BUNDLE="$SIM_ASSETS/robot/collision"
export INERTIAL_BUNDLE="$SIM_ASSETS/robot/inertia.json"
```

Open a LIBERO Spatial task with `--task-index` 0–9 and `--reset-index` 0–49:

```bash
alohamini-mujoco view --window-egl-device 0 \
  --libero-bundle "$LIBERO_BUNDLE" \
  --scene-config examples/simulation/libero_spatial_near.json \
  --task-index 8 --reset-index 0 --paused
```

Replay a reference episode, or replace the path with your own episode directory:

```bash
alohamini-mujoco replay "$REFERENCE_DATASET" \
  --view --window-egl-device 0 --paused
```

Replay uses the episode's saved model and initial state and pauses automatically when actions end. Omit `--view` to validate replay only.

## Collect data

Collect one episode using a reference trajectory and matching robot assets:

```bash
DATASET="$HOME/Alohamini_workspace/simulation/collections/task8_demo_01"
MUJOCO_GL=egl alohamini-mujoco collect "$REFERENCE_DATASET/trajectory.json" \
  --libero-bundle "$LIBERO_BUNDLE" \
  --scene-config examples/simulation/libero_spatial_near.json \
  --collision-bundle "$COLLISION_BUNDLE" --inertial-bundle "$INERTIAL_BUNDLE" \
  --task-index 8 --reset-index 0 --output "$DATASET"
```

The output directory must be new. `--cameras` selects cameras; `--image-size 480 640` specifies image height and width. Defaults are three 128×128 streams and 20 Hz control. Changed layouts or initial states require matching trajectories.

Collect a batch from reference episodes:

```bash
DATASET="$HOME/Alohamini_workspace/simulation/collections/task8_batch_01"
MUJOCO_GL=egl alohamini-mujoco collect \
  --reference "$REFERENCE_DATASET" \
  --recipe examples/simulation/libero_motion_forward.json \
  --reset-indices {0..39} --task-index 8 \
  --libero-bundle "$LIBERO_BUNDLE" \
  --scene-config examples/simulation/libero_spatial_near.json \
  --collision-bundle "$COLLISION_BUNDLE" --inertial-bundle "$INERTIAL_BUNDLE" \
  --output "$DATASET"
```

After interruption, rerun the same command with `--resume`. Initial states 0–39 are for training and 40–49 for evaluation; changing reference episodes requires a matching recipe. Batch results go to `index.json`; training reads only indexed training-group episodes. See the [training guide](training.md) for parameters.

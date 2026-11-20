# Installation

**English** | [简体中文](zh-CN/install.md)

Use the `alohamini` environment on the PC for teleoperation, data collection, and development. Use the lightweight `alohamini_host` environment on the Raspberry Pi for hardware access.

## 0. Prepare the repository and environments

If Git is not installed, run in Bash:

```bash
sudo apt update
sudo apt install -y git curl ca-certificates
```

If Conda is not installed, run in Bash (automatically selects PC x86_64 or Raspberry Pi aarch64):

```bash
curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh" -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
source "$HOME/.bashrc"
```

Clone the repository and enter its root directory:

```bash
git clone https://github.com/AlohaMini/Alohamini_software.git Alohamini
cd Alohamini
```

If you already have a clone, enter its root directory without cloning again.

On a fresh Ubuntu / Debian installation, install the system command alias if `python` is not found:

```bash
sudo apt install -y python3 python-is-python3
```

The project uses Python from the Conda environments below; activate the appropriate environment first. Run the following commands from the repository root. Do not recreate an existing environment.

## 1. PC

Supported on Linux x86_64 (glibc 2.28+). GPU training and inference require an NVIDIA driver compatible with CUDA 13.

On a fresh Ubuntu / Debian installation, first install the build tools required by keyboard input:

```bash
sudo apt install build-essential linux-libc-dev
```

Create and install the environment:

```bash
conda create -n alohamini --file env/explicit-pc-linux-64.txt
conda activate alohamini
python -m pip install --require-hashes --no-build-isolation -r env/pc-linux-64.lock
python -m pip install --no-deps --no-build-isolation -e '.[pc]'
```

The PC environment includes dependencies for teleoperation, data collection, notebooks, and policy training.
Prepare base weights and tokenizers as needed; see [Training and deployment](training.md).

**Keyboard teleoperation requires an X11 desktop.**

## 2. Raspberry Pi

Supported on 64-bit ARM Linux (aarch64).

```bash
conda create -n alohamini_host --file env/explicit-linux-aarch64.txt
conda activate alohamini_host
python -m pip install --require-hashes --no-build-isolation -r env/host-linux-aarch64.lock
python -m pip install --no-deps --no-build-isolation -e '.[host]'
```

The Host environment includes servo, camera, and communication dependencies.

## 3. Optional extensions

Install extensions from the repository root with the PC or cloud environment activated:

```bash
EXTRA=kinematics
python -m pip install --no-build-isolation -e ".[${EXTRA}]"
```

Replace `EXTRA` with a name from the table below.

| Extension | Purpose |
| --- | --- |
| `learning` | Common training dependencies |
| `kinematics` | TCP representations and forward/inverse kinematics |
| `analysis` | Standalone data analysis and evaluation inspection |
| `audio-debug` | Microphone diagnostics; requires system PortAudio |
| `joycon` | Joy-Con driver |
| `pi05`, `smolvla` | π0.5, SmolVLA |
| `acdit` | AC-DiT |
| `gr00t_n17` | GR00T N1.7 |
| `lewm` | LeWorldModel |
| `hyvla05` | HyVLA-0.5 |
| `openwam` | OpenWAM-α |
| `xiaomi_robotics1` | Xiaomi-Robotics-1 |
| `lingbot_vla_v2` | LingBot-VLA v2 |
| `spirit` | Spirit v1.5 |

For cloud inference, install both policy and communication extensions, for example:

```bash
python -m pip install -e '.[pi05,zmq]'
```

SmolVLA and Spirit use `.[smolvla,zmq]` and `.[spirit,zmq]`, respectively; the Raspberry Pi reuses the Host environment above. See [Training and deployment](training.md) for base weights and tokenizers.

Register the notebook kernel once in the PC environment:

```bash
conda activate alohamini
python -m ipykernel install --user --name alohamini --display-name 'AlohaMini'
```

## 4. Simulation

### MuJoCo

Install in the PC environment:

```bash
conda activate alohamini
python -m pip install -e './simulation/mujoco[collection]'
```

Robot meshes, physical properties, and joint limits are installed with the platform model assets.

Linux x86_64 desktops using `--window-egl-device` also require [VirtualGL 3.1.5](https://github.com/VirtualGL/virtualgl/releases/tag/3.1.5):

```bash
curl -fL https://github.com/VirtualGL/virtualgl/releases/download/3.1.5/virtualgl_3.1.5_amd64.deb \
  -o /tmp/virtualgl_3.1.5_amd64.deb
sudo apt install /tmp/virtualgl_3.1.5_amd64.deb
```

### ManiSkill and RoboTwin

Install either backend as needed. ManiSkill uses the PC environment:

```bash
conda activate alohamini
python -m pip install -e ./simulation/maniskill
python -m pip check
```

Prepare the environment and source for RoboTwin using the [official instructions](https://robotwin-platform.github.io/doc/usage/robotwin-install.html), naming the environment `alohamini_robotwin`. Its planning interface uses MPLib 0.2.1 and NumPy 1.26:

```bash
conda activate alohamini_robotwin
python -m pip install -e . -e ./simulation/robotwin
python -m pip check
```

The ManiSkill adapter uses SAPIEN 3.0.3; the RoboTwin adapter uses SAPIEN 3.0.0b1. Both share the platform robot assets.

Use Xephyr for windows on Linux X11 systems with hybrid graphics:

```bash
sudo apt install xserver-xephyr xauth
```

### LIBERO scenes and reference data

If you have the scene archive `alohamini-mujoco-assets-v1.tar.gz`, extract it into the workspace:

```bash
mkdir -p "$HOME/Alohamini_workspace/simulation"
tar -xzf /path/to/alohamini-mujoco-assets-v1.tar.gz \
  -C "$HOME/Alohamini_workspace/simulation"
```

Replace `/path/to/` with the archive directory. See [Simulation](simulation.md) for model viewing, scenes, replay, and collection commands.

## 5. ROS2 environment

PC ROS2 applications use a separate `alohamini_ros` Conda environment containing RoboStack ROS2, MoveIt, Gazebo, colcon, and Python dependencies. Training, Pink IK, and the Joy-Con driver continue to use `alohamini`.

From the repository root, install the locked environment for your operating system (Linux x86-64):

| PC system | ROS2 | Lockfile |
| --- | --- | --- |
| Ubuntu 22.04 | Humble | `ros2/locks/humble-linux-64.txt` |
| Ubuntu 24.04 | Jazzy | `ros2/locks/jazzy-linux-64.txt` |

Ubuntu 24.04:

```bash
cd ~/Alohamini
conda create -n alohamini_ros --file ros2/locks/jazzy-linux-64.txt
```

Ubuntu 22.04:

```bash
cd ~/Alohamini
conda create -n alohamini_ros --file ros2/locks/humble-linux-64.txt
```

After creation, check:

```bash
source ros2/env.bash --base
python -c 'import sys, rclpy; print(sys.executable); print(rclpy.__file__)'
python -m pip check
ros2 pkg prefix moveit_ros_move_group
ros2 pkg prefix gz_ros2_control
```

Both Python and ROS package paths should be inside `alohamini_ros`. See [ROS2](ros2.md) for build, model, and planning validation.

### Joy-Con and Pink IK

For Joy-Con teleoperation, install the reader and Pink IK in `alohamini`:

```bash
cd ~/Alohamini
conda activate alohamini
python -m pip install -e '.[joycon,kinematics]'
source ros2/env.bash
```

The final command switches back to `alohamini_ros`. Launch entry points automatically invoke worker processes in `alohamini`.

## 6. Check the installation

Run in `alohamini` on the PC or `alohamini_host` on the Raspberry Pi:

```bash
python -m pip check
alohamini --help
```

Then follow the [AlohaMini User Guide](alohamini.md) to configure serial permissions, device names, and calibration before starting the Host and teleoperation.

For daily use, activate the appropriate environment without reinstalling. If you move the source directory, rerun the corresponding `pip install ... -e` command from its new location.

See [ROS2 cameras](ros2.md) for building and launching the ROS2 camera client; the Raspberry Pi environment needs no changes.

## 7. Synchronize environments after updates

After updating the code, run in `alohamini` on the PC or `alohamini_host` on the Raspberry Pi:

```bash
alohamini env check
alohamini env sync
```

Startup commands check relevant dependencies and report version differences without installing anything. `sync` synchronizes against the current code's lockfile and supports a `--dry-run` preview; use `--role host` to select the Host environment explicitly. If the command is not installed yet, use `python env/manage.py check` or `python env/manage.py sync`.

Stop training, inference, Host, and notebook processes in the environment before synchronization. Conflicts with additional packages cause an exit before installation. Successful runs leave no record; failed or interrupted installations save diagnostics in the workspace's `logs/environment/`. Recreate the environment as described above if the Python major or minor version changes.

The ROS environment is managed by lockfiles in `ros2/locks/` and does not use `alohamini env sync`.

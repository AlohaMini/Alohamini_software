# 安装

PC 使用 `alohamini` 环境进行遥操、数采和开发；树莓派使用轻量的 `alohamini_host` 环境连接硬件。

## 0. 仓库与环境准备

未安装 Git 时，在 Bash 中执行：

```bash
sudo apt update
sudo apt install -y git curl ca-certificates
```

未安装 Conda 时，在 Bash 中执行（自动匹配 PC x86_64 / 树莓派 aarch64）：

```bash
curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh" -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
source "$HOME/.bashrc"
```

克隆仓库，再进入仓库根目录：

```bash
git clone https://github.com/AlohaMini/Alohamini_software.git
cd Alohamini_software
```

已有仓库时，直接进入其根目录，无需重复克隆。

新装 Ubuntu / Debian 若提示找不到 `python`，可安装系统命令别名：

```bash
sudo apt install -y python3 python-is-python3
```

项目仍使用下方 Conda 环境中的 Python，请先激活对应环境。以下命令在仓库根目录执行；已有对应环境时无需重复创建。

## 1. PC

适用于 Linux x86_64（glibc 2.28+）。GPU 训练与推理需要兼容 CUDA 13 的 NVIDIA 驱动。

Ubuntu / Debian 首次安装时，先准备键盘依赖的编译工具：

```bash
sudo apt install build-essential linux-libc-dev
```

创建并安装环境：

```bash
conda create -n alohamini --file env/explicit-pc-linux-64.txt
conda activate alohamini
python -m pip install --require-hashes --no-build-isolation -r env/pc-linux-64.lock
python -m pip install --no-deps --no-build-isolation -e '.[pc]'
```

PC 环境包含遥操、数采、Notebook 和策略训练依赖。
基座权重和 tokenizer 按需准备，见[训练与部署](training.md)。

**键盘遥操需要 X11 桌面。**

## 2. 树莓派

适用于 64 位 ARM Linux（aarch64）。

```bash
conda create -n alohamini_host --file env/explicit-linux-aarch64.txt
conda activate alohamini_host
python -m pip install --require-hashes --no-build-isolation -r env/host-linux-aarch64.lock
python -m pip install --no-deps --no-build-isolation -e '.[host]'
```

Host 环境包含舵机、相机和通信依赖。

## 3. 可选扩展

在已激活的 PC 或云端环境、仓库根目录安装所需扩展：

```bash
EXTRA=kinematics
python -m pip install --no-build-isolation -e ".[${EXTRA}]"
```

将 `EXTRA` 换成下表中的名称。

| 扩展名 | 用途 |
| --- | --- |
| `learning` | 通用训练依赖 |
| `kinematics` | TCP 表示、正逆运动学 |
| `analysis` | 独立数据分析与评估查看 |
| `audio-debug` | 麦克风调试，系统需安装 PortAudio |
| `joycon` | Joy-Con 驱动 |
| `pi05`、`smolvla` | π0.5、SmolVLA |
| `acdit` | AC-DiT |
| `gr00t_n17` | GR00T N1.7 |
| `lewm` | LeWorldModel |
| `hyvla05` | HyVLA-0.5 |
| `openwam` | OpenWAM-α |
| `xiaomi_robotics1` | Xiaomi-Robotics-1 |
| `lingbot_vla_v2` | LingBot-VLA v2 |
| `spirit` | Spirit v1.5 |

云端推理服务同时安装策略与通信扩展，例如：

```bash
python -m pip install -e '.[pi05,zmq]'
```

SmolVLA／Spirit 分别使用 `.[smolvla,zmq]`、`.[spirit,zmq]`；树莓派复用前述 Host 环境。基座权重与 tokenizer 的准备见[训练与部署](training.md)。

Notebook 内核在 PC 环境注册一次：

```bash
conda activate alohamini
python -m ipykernel install --user --name alohamini --display-name 'AlohaMini'
```

## 4. 仿真

### MuJoCo

在 PC 环境安装：

```bash
conda activate alohamini
python -m pip install -e './simulation/mujoco[collection]'
```

机器人网格、物性与限位随平台模型资产安装。

使用 `--window-egl-device` 的 Linux x86_64 桌面还需安装 [VirtualGL 3.1.5](https://github.com/VirtualGL/virtualgl/releases/tag/3.1.5)：

```bash
curl -fL https://github.com/VirtualGL/virtualgl/releases/download/3.1.5/virtualgl_3.1.5_amd64.deb \
  -o /tmp/virtualgl_3.1.5_amd64.deb
sudo apt install /tmp/virtualgl_3.1.5_amd64.deb
```

### ManiSkill 与 RoboTwin

两个后端按需安装。ManiSkill 使用 PC 环境：

```bash
conda activate alohamini
python -m pip install -e ./simulation/maniskill
python -m pip check
```

RoboTwin 按[官方步骤](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)准备环境和源码，环境命名为 `alohamini_robotwin`。其规划接口使用 MPLib 0.2.1、NumPy 1.26：

```bash
conda activate alohamini_robotwin
python -m pip install -e . -e ./simulation/robotwin
python -m pip check
```

ManiSkill 适配使用 SAPIEN 3.0.3，RoboTwin 适配使用 SAPIEN 3.0.0b1。两者共用平台机器人资产。

Linux X11 混合显卡窗口使用 Xephyr：

```bash
sudo apt install xserver-xephyr xauth
```

### LIBERO 场景与参考数据

已有配套场景包 `alohamini-mujoco-assets-v1.tar.gz` 时，解压到工作目录：

```bash
mkdir -p "$HOME/Alohamini_workspace/simulation"
tar -xzf /path/to/alohamini-mujoco-assets-v1.tar.gz \
  -C "$HOME/Alohamini_workspace/simulation"
```

将 `/path/to/` 替换为资产包所在目录。模型查看、场景、回放与采集命令见[仿真](simulation.md)。

## 5. ROS2 依赖

Joy-Con 差分 IK 使用 `alohamini` 环境中的 Pink 工作进程。在该环境安装：

```bash
conda activate alohamini
python -m pip install -e '.[kinematics]'
conda deactivate
```

Ubuntu 22.04 使用 ROS2 Humble，Ubuntu 24.04 使用 ROS2 Jazzy；分别安装对应发行版的 colcon 和 rosdep。在系统 Python 终端中，从仓库根目录执行：

```bash
source ros2/env.bash --base
cd ros2
rosdep install --from-paths src --ignore-src -r -y
```

Gazebo Fortress 扩展的依赖：

```bash
rosdep install --from-paths src/alohamini_core ../simulation/gazebo --ignore-src -r -y
```

构建与启动见 [ROS2](ros2.md)。

## 6. 检查安装

在各自环境中执行：

```bash
python -m pip check
alohamini --help
```

随后按 [AlohaMini 使用手册](alohamini.md) 配置串口权限、设备名称和标定，再启动 Host 与遥操。

日常使用只需激活对应环境，不必重复安装。若移动了源码目录，需在新位置重新执行对应的 `pip install ... -e` 命令。

ROS2 相机客户端的构建与启动见 [ROS2 相机](ros2.md)；树莓派环境无需变动。

## 7. 更新后同步环境

更新代码后，在对应环境中执行：

```bash
alohamini env check
alohamini env sync
```

启动命令会检查相关依赖并提示版本差异，不自动安装。`sync` 按当前代码的锁文件同步，支持 `--dry-run` 预检；Host 环境可显式指定 `--role host`。尚未安装命令时，使用 `python env/manage.py check` 或 `python env/manage.py sync`。

同步前先结束该环境中的训练、推理、Host 和 Notebook。额外包存在版本冲突时会在安装前退出；成功后不保留记录，安装失败或中断时将诊断信息保存在工作目录的 `logs/environment/`。Python 大小版本变化时按前述步骤重新创建环境。

# macOS 使用手册

Mac 使用 `alohamini` 环境进行遥操、数采、训练和评估，机器人 Host 使用 Ubuntu。以下以 `alohamini2pro` 为例；`alohamini2` 使用相同流程，将命令中的型号替换即可。

| 芯片 | 系统与 Python | 使用范围 |
| --- | --- | --- |
| Apple Silicon | macOS 15+，原生 arm64 Python 3.12 | 本手册的完整 PC 流程，训练可选 MPS 或 CPU |
| Intel | Python 3.12 | 主臂串口标定，见[串口工具安装](#仅安装串口工具) |

## 1. 安装

### Git 与 Conda

在终端执行 `git --version`。首次使用时，按系统提示安装 Xcode Command Line Tools，也可执行：

```bash
xcode-select --install
```

Apple Silicon 安装原生 arm64 Conda。已有 Conda 时，直接进入下一步；首次安装 Miniconda 可执行：

```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init zsh
source "$HOME/.zshrc"
```

使用 Bash 终端时，将最后两行的 `zsh`、`.zshrc` 换为 `bash`、`.bashrc`。

### 仓库与 PC 环境

```bash
cd ~
git clone https://github.com/AlohaMini/Alohamini_software.git Alohamini
cd ~/Alohamini
conda env create -f environment.yml
conda activate alohamini
python -m pip install --require-hashes --no-build-isolation -r env/pc-osx-arm64.lock
python -m pip install --no-deps --no-build-isolation -e '.[pc]'
python -m pip check
alohamini env check --role pc
```

已有仓库时进入其根目录，已有环境时直接激活。检查 Python 架构：

```bash
python -c 'import platform; print(platform.system(), platform.machine())'
```

Apple Silicon 应显示 `Darwin arm64`。PC 环境包含遥操、数采、Notebook 和通用训练依赖；策略权重按各策略 README 准备。

更新代码后，同步当前环境：

```bash
alohamini env sync --role pc
```

同步前关闭该环境中的训练、Notebook 和其他 Python 任务。

Notebook 内核注册一次：

```bash
python -m ipykernel install --user --name alohamini --display-name AlohaMini
```

### 仅安装串口工具

Apple Silicon 和 Intel Mac 均可在 Python 3.12 环境中单独使用主臂标定：

```bash
cd ~/Alohamini
conda create -n alohamini python=3.12 pip setuptools wheel
conda activate alohamini
python -m pip install --require-hashes -r env/hardware.lock
python -m pip install --no-deps --no-build-isolation -e .
```

已有完整 PC 环境时直接使用其中的串口工具。

## 2. 接线与权限

| 设备 | 连接位置 |
| --- | --- |
| 左、右主臂 USB 控制板 | Mac USB |
| 左、右从臂、底盘与升降 | Ubuntu Host |
| 前视、腕部相机 | Ubuntu Host |

AM-ARM 主臂使用 5 V，从臂使用 12 V。Mac 与 Ubuntu Host 连接到同一局域网。

在「系统设置 → 隐私与安全性」中，为所用终端或应用开启「输入监控」和「辅助功能」权限。授权后重新启动终端或应用，再激活 `alohamini`。

## 3. 主臂标定

先结束占用主臂串口的程序，将主臂稳固放置。执行：

```bash
conda activate alohamini
alohamini calibrate leader --robot_model alohamini2pro --teleop.id am_leader_bi
```

首次按提示登记左右 USB 控制板：拔下两根主臂 USB 线并按 Enter；先插左臂，识别后保持连接，再插右臂。

将双主臂摆到 Home，保持静止：

| 部件 | Home 摆位 |
| --- | --- |
| 舵机 1、5、6 | 对齐中位 |
| 舵机 2、3、4 | 蜷缩，轻靠对应连杆或外壳接触面 |
| 夹爪 | 合拢 |

轻靠接触面，保持双臂稳定，按一次 Enter 完成标定。已有标定时按 Enter 复用，输入 `c` 重做。

Mac 根据控制板序列号匹配当前 `/dev/cu.*` 串口，并加载这对主臂的标定。换 USB 接口后继续使用原标定；同一时间连接一套主臂。

查询串口：

```bash
alohamini find-port
```

按提示逐个拔插控制板。手动指定端口时，在标定或遥操命令后同时添加 `--left_port /dev/cu.usbmodem左端口 --right_port /dev/cu.usbmodem右端口`，替换为实际查询结果。

## 4. 启动机器人 Host

Ubuntu Host 的安装、从臂标定与相机设备配置见[安装](install.md#2-树莓派)和[使用手册](alohamini.md#2-主从臂标定)。完成后，在机器人上执行：

> 启动会使能舵机并执行升降触底回零。先支撑双臂，清空升降下降路径。

```bash
conda activate alohamini_host
alohamini host --robot_model alohamini2pro --cameras forward wrist_right
```

保持该终端运行。相机名称和配置见 [Host 配置](alohamini.md#4-启动树莓派-host)。

在 Mac 检查连接，将 `<机器人_IP>` 换成 Ubuntu Host 的实际地址：

```bash
conda activate alohamini
alohamini inspect --host <机器人_IP> --model alohamini2pro
```

连接异常时先检查 Host 是否运行、两端 IP 与型号，以及局域网和防火墙设置。

## 5. 遥操

在 Mac 执行：

```bash
alohamini teleoperate \
  --host <机器人_IP> \
  --robot_model alohamini2pro \
  --teleop.id am_leader_bi
```

程序自动匹配已登记的主臂，默认 50 Hz 控制、30 Hz 图像请求，并打开 Rerun 预览。

| 按键 | 操作 |
| --- | --- |
| `w` / `s` | 底盘前进 / 后退 |
| `z` / `x` | 底盘左移 / 右移 |
| `a` / `d` | 底盘左转 / 右转 |
| `t` / `g` | 提高 / 降低速度档位 |
| `u` / `j` | 升降上升 / 下降 |
| `q`、Esc、Ctrl+C | 退出 |

仅使用主臂时加 `--no_keyboard`，仅使用键盘时加 `--no_leader`，关闭预览时加 `--no_preview`。退出遥操后再启动数采。

## 6. 数采

保持 Ubuntu Host 运行，其启用的全部相机将参与录制。在 Mac 执行：

```bash
alohamini record \
  --host <机器人_IP> \
  --robot_model alohamini2pro \
  --teleop.id am_leader_bi \
  --dataset pickup_01 \
  --task "pick up the object" \
  --num_episodes 1 \
  --episode_time 8 \
  --reset_time 3
```

默认 30 Hz 数采，使用 CPU 编码 H.264；加 `--display_data` 开启预览。`--video-encoding-workers 2` 可限制同时编码的相机数量。

Apple Silicon 可加 `--video-encoder videotoolbox` 使用硬件编码，每路目标码率为 8 Mbit/s。编码器不可用时回退 CPU，保留帧数与时间顺序。硬件编码与 CPU 编码的像素和文件大小可能不同。

| 按键 | 操作 |
| --- | --- |
| 右方向键 | 提前结束当前阶段 |
| `R` / 左方向键 | 删除当前回合临时数据并重录 |
| Esc、`q`、Ctrl+C | 结束数采 |

复位阶段可继续遥操，下一回合开始时继续录制。数据保存在 `~/Alohamini_workspace/datasets/pickup_01/`。

继续录制时使用相同命令加 `--resume`，保持型号、标定、相机、任务和帧率一致。固定关节等选项见[数采参数](alohamini.md#7-pc-本地数采)。

## 7. 检查数据与回放

检查结构和视频：

```bash
alohamini dataset check ~/Alohamini_workspace/datasets/pickup_01 --decode-videos
```

异常中断后保留原目录，恢复到新目录：

```bash
alohamini dataset recover ~/Alohamini_workspace/datasets/pickup_01 \
  --output ~/Alohamini_workspace/datasets/pickup_01_recovered
```

动作回放会驱动整机。先退出遥操和数采，将机器人摆到示教起始姿态，并清空运动区域：

```bash
alohamini replay --dataset pickup_01 \
  --host <机器人_IP> --robot_model alohamini2pro --episode 0
```

Host 的型号、动作单位和标定应与数据一致。数据编辑、导出与训练准备见[训练与部署](training.md#2-检查与编辑数据)。

## 8. 训练与评估

按[训练与部署](training.md)和对应策略 README 准备数据与权重，Apple Silicon 的设备参数见 [MPS 配置](training.md#apple-siliconmps)。

## 9. 工作文件与调试

```bash
alohamini paths
```

默认工作目录为 `~/Alohamini_workspace/`：

| 目录 | 内容 |
| --- | --- |
| `calibration/` | 主臂标定和 USB 登记 |
| `datasets/` | 本地数据集 |
| `runs/` | 训练与评估产物 |
| `logs/` | 运行日志 |

自定义工作目录时，将 `ALOHAMINI_WORKSPACE` 设为绝对路径。迁移电脑时一并保留标定、USB 登记及数据。

调试直接接在 Mac 上的相机：

```bash
alohamini find-cameras opencv --record-time-s 0
python examples/debug/test_cv.py --camera 0
```

首次访问时允许终端或应用使用摄像头。Mac 使用数字索引，插拔或重启后重新核对索引；预览图片保存到工作目录的 `logs/debug/`。机器人相机通过 Ubuntu Host 读取。


## 10. 安装与流程验证

在已安装的 PC 环境中运行：

```bash
python env/verify_pc.py --device mps
python env/macos_validation.py
```

完整验证覆盖通信、缓存、数据编辑与恢复、视频编码和训练续训，报告保存在工作目录的 `logs/macos-validation/`。每次运行使用独立子目录，包含 `report.json`、分项日志和 JUnit XML；硬件不可用的项目记录为跳过。

需要同时检查 Ubuntu Host 的状态与图像连接时：

```bash
python env/macos_validation.py --host <机器人_IP> --model alohamini2pro
```

该连接检查只读取状态和图像。仅验证 Intel Mac 串口环境时，先安装 `pytest==8.4.2`，再运行 `python env/macos_validation.py --sections serial`。

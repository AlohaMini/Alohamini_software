# 仿真

[English](../simulation.md) | **简体中文**

先完成[仿真环境安装](install.md#4-仿真)。以下命令在仓库根目录、对应环境中执行。

## MuJoCo

打开整机并使用 EGL 设备 0 渲染：

```bash
conda activate alohamini
alohamini-mujoco view --window-egl-device 0
```

省略 `--window-egl-device 0` 使用桌面默认 OpenGL 设备。机器人资产随平台安装，默认固定底盘。

指定升降初值并暂停启动：

```bash
alohamini-mujoco view --window-egl-device 0 \
  --initial-positions '{"vertical_move":0.2}' --paused
```

`--initial-positions` 使用 URDF 关节名；旋转关节单位 rad，升降单位 m，范围为 `[-0.3, 0.3]`。

打开黑色桌面与方块场景；修改 JSON 可调整物体和机器人初态：

```bash
alohamini-mujoco view --window-egl-device 0 \
  --scene-config examples/simulation/research.json --paused
```

## 窗口操作

展开右侧 `Control`，拖动滑块设置关节目标。暂停时保存目标，按空格播放后模型开始跟随。`Joint` 显示当前关节位置；拖动它也会设置对应目标。运动受关节限位、速度和电机力限制。

| 操作 | 效果 |
| --- | --- |
| 空格 | 开始或暂停仿真 |
| 右方向键 | 暂停时推进一个物理步 |
| F9 | 复位并暂停 |
| F8 | 切换视觉模型与碰撞几何 |
| T | 切换半透明显示 |
| F6 | 切换坐标系显示 |
| Tab／Shift + Tab | 显示或隐藏左／右面板 |
| 左键拖动 | 旋转视角 |
| 右键拖动／Shift + 右键拖动 | 在竖直面／水平面平移视角 |
| 滚轮 | 缩放视角 |
| 右键双击 | 将观察中心移到点击位置 |

普通 `view` 中，左键双击选中连杆，Ctrl + 右键拖动施加平移外力，Ctrl + 左键拖动施加转矩；Ctrl + Shift + 右键拖动在水平面施力。

`alohamini-mujoco` 窗口的 `Print data`、模型导出和截图默认保存到 `~/Alohamini_workspace/simulation/viewer/`；自定义工作区使用 `ALOHAMINI_WORKSPACE`。命令中的相对路径按启动目录解析。

## ManiSkill 与 RoboTwin

在 ManiSkill 环境查看同一场景：

```bash
conda activate alohamini
alohamini-maniskill view --scene-config examples/simulation/research.json
```

窗口目标刷新率为 60 FPS，使用 `--render-fps 30` 可调整。控制频率由场景的 `control_frequency_hz` 设置，物理步长由 `timestep_s` 设置。

Linux X11 混合显卡默认使用隔离窗口，保留原生查看器。`--window-mode native` 使用直接窗口，`--window-mode isolated` 强制隔离窗口。

在 RoboTwin 环境查看，`--source` 指向其源码目录：

```bash
conda activate alohamini_robotwin
alohamini-robotwin view --source /path/to/RoboTwin \
  --scene-config examples/simulation/research.json
```

添加 `--headless --steps 1000` 可关闭渲染并推进 1000 个物理步；默认步长为 1 ms。两者当前支持单个 CPU 物理环境。

## LIBERO 场景与回放

按[安装文档](install.md#libero-场景与参考数据)解压配套资产包后，在当前终端设置路径：

```bash
export SIM_ASSETS="$HOME/Alohamini_workspace/simulation/alohamini-mujoco-assets-v1"
export LIBERO_BUNDLE="$SIM_ASSETS/libero_spatial"
export REFERENCE_DATASET="$SIM_ASSETS/reference/task8"
export COLLISION_BUNDLE="$SIM_ASSETS/robot/collision"
export INERTIAL_BUNDLE="$SIM_ASSETS/robot/inertia.json"
```

打开 LIBERO Spatial 任务，`--task-index` 为 0–9，`--reset-index` 为 0–49：

```bash
alohamini-mujoco view --window-egl-device 0 \
  --libero-bundle "$LIBERO_BUNDLE" \
  --scene-config examples/simulation/libero_spatial_near.json \
  --task-index 8 --reset-index 0 --paused
```

回放参考回合，或将路径替换为自己的回合目录：

```bash
alohamini-mujoco replay "$REFERENCE_DATASET" \
  --view --window-egl-device 0 --paused
```

回放使用该回合保存的模型与初态，动作结束自动暂停；省略 `--view` 只验证回放。

## 采集数据

使用参考轨迹与配套机器人资产采集一个回合：

```bash
DATASET="$HOME/Alohamini_workspace/simulation/collections/task8_demo_01"
MUJOCO_GL=egl alohamini-mujoco collect "$REFERENCE_DATASET/trajectory.json" \
  --libero-bundle "$LIBERO_BUNDLE" \
  --scene-config examples/simulation/libero_spatial_near.json \
  --collision-bundle "$COLLISION_BUNDLE" --inertial-bundle "$INERTIAL_BUNDLE" \
  --task-index 8 --reset-index 0 --output "$DATASET"
```

输出目录须为新目录。`--cameras` 选择相机，`--image-size 480 640` 设置图像高和宽；默认使用三路 128×128 图像、20 Hz 控制。更换布局或初态后需生成匹配轨迹。

从参考回合批量采集：

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

中断后用原命令加 `--resume`。初态 0–39 用于训练，40–49 用于评估；更换参考回合须使用匹配配方。批次结果写入 `index.json`，训练只读取已收录的训练组回合。训练参数见[训练手册](training.md)。

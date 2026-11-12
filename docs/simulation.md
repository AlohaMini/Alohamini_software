# MuJoCo 仿真

查看 AlohaMini2Pro 和 LIBERO 桌面场景，采集任务数据，回放已保存的动作。

## 安装与资产路径

在仓库根目录执行：

```bash
cd ~/Alohamini
conda activate alohamini
python -m pip install -e './simulation/mujoco[collection]'
```

Linux 窗口命令需要安装 [VirtualGL 3.1.5](https://github.com/VirtualGL/virtualgl/releases/tag/3.1.5)，并在桌面终端运行。

本机已准备好以下资产，无需重新导入。先在当前终端设置路径：

```bash
export SIM_WORKSPACE="$HOME/Alohamini_workspace"
export COLLISION_BUNDLE="$SIM_WORKSPACE/simulation/collision/structural_contact_robot_20261026"
export INERTIAL_BUNDLE="$SIM_WORKSPACE/simulation/cad/alohamini2pro_active_inertias_20261102/inertia.json"
export LIBERO_BUNDLE="$SIM_WORKSPACE/simulation/libero_spatial_8f1084e_v3"
```

`COLLISION_BUNDLE` 是含 `collision.json` 和网格文件的完整目录，`INERTIAL_BUNDLE` 是物性 JSON，`LIBERO_BUNDLE` 是含 `bundle.json`、任务和网格文件的完整目录。换机器时复制完整资产并修改上述路径。环境变量仅对当前终端及其子进程生效；新开终端须重新设置。

16 GB 内存的主机串行运行完整场景的可视化、数采与评测；批量数采前关闭仿真窗口。

## 打开窗口

| 要查看的内容 | 使用方式 | 播放后的行为 |
| --- | --- | --- |
| 机器人与桌面任务 | `view` 加场景参数 | 保持初始关节目标，可交互施力 |
| 单独的机器人 | `view` 不加场景参数 | 保持初始关节目标，可交互施力 |
| 已采集的抓放动作 | `replay 数据目录 --view` | 按保存的命令重新运行物理仿真 |

`--paused` 表示启动后暂停。普通 `view` 用于查看和操控，自动执行动作需提供轨迹或打开数据回放。

### 查看桌面任务

```bash
MUJOCO_GL=glfw alohamini-mujoco view --window-egl-device 0 \
  --libero-bundle "$LIBERO_BUNDLE" \
  --scene-config examples/simulation/libero_spatial_near.json \
  --task-index 8 --reset-index 0 \
  --collision-bundle "$COLLISION_BUNDLE" \
  --inertial-bundle "$INERTIAL_BUNDLE" --paused
```

近距配置中，活动物品沿世界 X 负方向平移 0.08 m。底盘初始位置为世界坐标 `(-0.089, 0, 0)` m，双肩 pan 均为 0，升降轴为 +0.3 m；地面为 0 m，桌面为 0.9 m。底盘和轮轴在运行中固定。

`--task-index` 为 0–9，`--reset-index` 为 0–49。物体先经过 0.5 s 落稳检查再打开窗口；F9 返回落稳后的初态。

### 只看机器人

```bash
MUJOCO_GL=glfw alohamini-mujoco view --window-egl-device 0 \
  --collision-bundle "$COLLISION_BUNDLE" \
  --inertial-bundle "$INERTIAL_BUNDLE" \
  --initial-positions '{"vertical_move":0.2}' --paused
```

此模式不加载桌子和任务物体。

### 回放抓放动作

本机已有的任务 8 参考回合：

```bash
REFERENCE_DATASET="$SIM_WORKSPACE/simulation/collections/libero_spatial_task8_near_fixed_lift_reference_20261105"
MUJOCO_GL=glfw alohamini-mujoco replay "$REFERENCE_DATASET" \
  --view --paused --window-egl-device 0
```

将数据路径替换为自己的回合目录即可。回放恢复该回合保存的模型和初态，不读取当前场景示例；回放时不接收鼠标外力，动作结束自动暂停。不加 `--view` 时只验证回放，不打开窗口。

## 窗口操作

| 操作 | 效果 |
| --- | --- |
| 左键拖动 | 旋转视角 |
| 右键拖动 | 在竖直面平移视角 |
| Shift + 右键拖动 | 在水平面平移视角 |
| 滚轮 | 缩放视角 |
| 右键双击 | 将观察中心移到点击位置 |
| 空格 | 开始或暂停仿真 |
| F8 | 切换视觉模型与碰撞几何 |
| F9 | 复位并暂停 |
| 右方向键 | 暂停时推进一个物理步 |
| F6 | 切换坐标系显示 |

普通 `view` 运行时，左键双击选中连杆，Ctrl + 右键拖动施加平移外力，Ctrl + 左键拖动施加转矩；Ctrl + Shift + 右键拖动在水平面施力。松手后控制器尝试恢复保持姿态，关节限位、碰撞与电机力限制继续生效。

使用 `view --trajectory 轨迹.json` 可预览轨迹，此模式不接收鼠标外力。检查已采集动作应使用 `replay --view`。

## 采集数据

轨迹、场景、碰撞资产和物性须相互匹配。以下使用已有参考轨迹及其配套场景重新采集一个回合：

```bash
REFERENCE_DATASET="$SIM_WORKSPACE/simulation/collections/libero_spatial_task8_near_fixed_lift_reference_20261105"
SCENE_CONFIG="examples/simulation/libero_spatial_near.json"
DATASET="$SIM_WORKSPACE/simulation/collections/task8_demo_01"
MUJOCO_GL=egl alohamini-mujoco collect "$REFERENCE_DATASET/trajectory.json" \
  --libero-bundle "$LIBERO_BUNDLE" --scene-config "$SCENE_CONFIG" \
  --task-index 8 --reset-index 0 \
  --collision-bundle "$COLLISION_BUNDLE" --inertial-bundle "$INERTIAL_BUNDLE" \
  --output "$DATASET"
```

输出目录必须不存在；再次采集时修改 `DATASET`。更换布局或初态后需重新生成匹配轨迹。

采集使用 20 Hz 绝对关节位置命令，每条命令保持 50 个 1 ms 物理步，参考速度为零。默认保存三路 128×128 无损 RGB；`--cameras` 选择相机，`--image-size 256` 设置正方形分辨率，`--image-size 480 640` 设置高和宽。图像不含 TCP、任务标记或坐标轴。

每帧保存控制前的关节位置、速度、图像及相机内外参；`actions` 为该周期的 rad/m 命令，`states` 保存 N+1 个完整积分状态。`actions_standard` 和 `obs/joint_positions_standard` 保存固定标准范围的动作和反馈，其关节名称、转换定义及校验值随回合保存。

`camera/<相机名>/` 保存 `K_opencv`、畸变系数 `D` 和逐帧 `T_world_camera_optical`：光学坐标 X 向右、Y 向下、Z 向前，像素中心为整数坐标。`K` 和 `T_world_camera` 保留原有半整数像素中心及 OpenGL 坐标约定。

收录要求：原始任务成功、双指接触至少 0.1 s、夹持时抬升至少 2 cm、释放后稳定至少 0.5 s、穿透不超过 0.5 mm，并通过关节速度、限位和完整状态回放检查。失败退出码为 1。

通过的回合位于 `episodes/`，未通过的完整录制位于 `rejected/`。输出还包含 `dataset.json`、`trajectory.json` 和可搬移的 `model/`。回放不需要原始 CAD 或 LIBERO 环境，MuJoCo 版本须与采集记录一致。

### 从参考回合批量采集

```bash
REFERENCE_DATASET="$SIM_WORKSPACE/simulation/collections/libero_spatial_task8_near_fixed_lift_reference_20261105"
DATASET="$SIM_WORKSPACE/simulation/collections/task8_batch_01"
MUJOCO_GL=egl alohamini-mujoco collect \
  --reference "$REFERENCE_DATASET" \
  --recipe examples/simulation/libero_motion_forward.json \
  --reset-indices {0..39} \
  --libero-bundle "$LIBERO_BUNDLE" \
  --scene-config examples/simulation/libero_spatial_near.json --task-index 8 \
  --collision-bundle "$COLLISION_BUNDLE" --inertial-bundle "$INERTIAL_BUNDLE" \
  --output "$DATASET"
```

参考回合须已收录且通过回放验证。配方以 20 Hz 命令帧号标注接近、抓持、搬运和释放阶段；更换参考回合须重新标注。每个初态独立规划、执行和验收，拒收原因保留在批次中。`allow_lift_motion: false` 表示升降命令全程保持初始高度；无法完成规划的初态记为失败。

`index.json` 记录全部尝试，`reset_NNN/` 保存回合；训练只读取收录的训练组数据。中断后用原命令加 `--resume`，已完成的成功及失败尝试均不重跑，参考回合、配方、模型和配置须保持一致。训练与闭环评测见 [训练手册](training.md#mujoco-仿真数据)。

## 自定义场景与轨迹

场景使用米和弧度，世界坐标 X 向前、Y 向左、Z 向上。

| 配置字段 | 含义 |
| --- | --- |
| `T_world_root` | 机器人根坐标到世界坐标的刚体变换 |
| `T_world_libero` | 场景坐标到世界坐标的刚体变换，保持 Z 轴向上 |
| `initial_object_translation_world_m` | 活动物体初始位置的世界坐标平移，单位 m，Z 须为 0 |
| `initial_positions` | 以 URDF 关节名指定初态，覆盖 SRDF Home；命令行同名值优先 |
| `cameras` | 相机名称、父刚体、局部位姿与投影参数 |

修改布局或初态后须重新验证可达性和碰撞间隙。

标定相机使用 `name`、`body`、`T_body_camera_optical` 和 `intrinsics`。`intrinsics` 包含 `resolution: [宽, 高]`、3×3 的 `K`、`D` 及 `distortion_model`（`plumb_bob` 对应 5 项，`equidistant` 对应 4 项）。图像按标定畸变生成，完整参数、标定状态和来源写入回合及导出的模型。父刚体上的安装变换固定，世界位姿随机器人运动更新。

虚拟相机继续使用 `T_body_camera`（OpenGL 坐标）和 `fovy_deg`。

全局相机添加 `role: "workspace"` 后，采集会逐帧检查目标物体完整落在画幅内，并保存投影边界；裁切物体的回合按拒收保存。腕部相机允许任务物体随手臂运动进出画面。

轨迹 JSON 示例：

```json
{
  "schema_version": 1,
  "joint_names": ["left_shoulder_pan"],
  "times_s": [0.0, 2.0],
  "positions": [[0.0], [0.1]]
}
```

时间从零开始并严格递增；旋转关节使用 rad，升降轴使用 m。首个位置须匹配初态，未列出的关节保持初始位置。位置和三次插值峰值速度须符合关节限制。

可选 `controller` 支持 `bandwidth_hz`、`damping_ratio` 和按关节名指定的 `joint_gains`（`kp`、`kd`）；旋转关节增益单位为 N·m/rad、N·m·s/rad，直线关节为 N/m、N·s/m。

## 支持范围

- 支持 LIBERO Spatial 的 10 个任务场景及 `On` 目标，使用 AlohaMini 轨迹；参考扩增限单个 `On` 目标，不保证每个初态可达。
- 初态 0–39 为训练组，40–49 为评估组，同组不能跨训练与验证。AlohaMini 适配结果不等同于原 Panda LIBERO 得分。
- 活动机构须有完整物性；解除基座或轮轴锁定后不能继续使用固定底盘的物性文件。
- 相机与控制器尚未完成真机标定；rad/m 仿真动作暂不支持直接与 Host 归一化动作混训或下发真机。

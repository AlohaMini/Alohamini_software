# ROS2

树莓派按使用手册启动 Host；PC 的 ROS 节点使用系统 Python，不在 Conda 环境中运行。

| PC 系统 | ROS2 | 系统 Python |
|---|---|---|
| Ubuntu 22.04 | Humble | 3.10 |
| Ubuntu 24.04 | Jazzy | 3.12 |

两套组合共用源码、模型与配置；构建及安装目录按 ROS 发行版隔离，不能复用另一发行版的编译产物。
Pink／Pinocchio 安装在 `alohamini` 环境；ROS 遥操通过本机 IPC 调用常驻 IK 工作进程，不在 ROS Python 中安装其二进制依赖。

## 构建

完成 [ROS2 依赖安装](install.md#5-ros2-依赖)，在系统 Python 终端构建：

```bash
cd ~/Alohamini
bash ros2/build.bash
source ros2/env.bash
```

环境脚本按 Ubuntu 版本选择 Humble 或 Jazzy，也可先设置 `export ALOHAMINI_ROS_DISTRO=humble` 或 `jazzy`。切换发行版须使用新终端；依赖分别安装。构建目录为 `ros2/build/<发行版>`，安装目录为 `ros2/install/<发行版>`。

`alohamini_core` 从平台根目录安装同一份 Python 源码和模型资产，无需另装 LeRobot、PyTorch 或新建 Conda 环境。

## 整机状态与 TF

适用 `alohamini2pro`。将左右臂 URDF 映射 `hardware_joint_map_left.yaml`、`hardware_joint_map_right.yaml` 及 `lift_axis.yaml` 放在 `~/Alohamini_workspace/calibration/hardware/`。这些文件与舵机标定 JSON 分开。

真机与离线规划共享同一份关节、碰撞几何、惯量和 SRDF。真机以 `base_link` 为根；离线 `plan_only` 自动附加 X／Y／偏航虚拟关节，以 `root` 为根。

```bash
export ROS_LOG_DIR="${ALOHAMINI_WORKSPACE:-$HOME/Alohamini_workspace}/logs/ros2"
ros2 launch alohamini_bringup hardware.launch.py host:=<PI_IP>
```

默认同时订阅 Host 已开启的相机；只查看状态时追加 `enable_cameras:=false`。自定义关节标定目录用 `arm_mapping_dir:=/绝对路径`，也支持 `ALOHAMINI_WORKSPACE`。启动时命令通道关闭，不发送运动命令。

- `/joint_states`、`/tf`、`/tf_static`：模型关节与坐标树，关节位置使用 rad 或 m。
- `/alohamini_lerobot_bridge/measured_joint_states`：双臂位置与升降编码器推算位置。没有高度传感器，不提供关节力矩。
- `/alohamini_lerobot_bridge/derived_wheel_states`：速度积分的车轮角度，不代表定位或实测里程计。
- `/alohamini/base_velocity`：底盘速度，m/s 和 rad/s；`/diagnostics`：连接与状态有效性。

失联或反馈无效时停止刷新状态。默认使用 PC 接收时间；`state_timestamp_mode:=host_wall` 使用 Host 时间，须先同步两端时钟。

单独启动 Bridge 并指定参数文件：

```bash
ros2 launch alohamini_bridge bridge.launch.py host:=<PI_IP> params_file:=/绝对路径/bridge.yaml
```

参数模板见 `ros2/src/alohamini_bridge/config/bridge.yaml`，可配置请求窗口、超时、轨迹容差和底盘坐标变换；启动参数覆盖参数文件中的同名项。`alohamini_bridge runtime.launch.py` 同时启动模型、Bridge 和相机，不启动 MoveIt 或 Joy-Con。

## 命令启停与底盘控制

确认机器人周围安全、Host 就绪且未被遥操或其他客户端占用后启用：

```bash
ros2 service call /alohamini_lerobot_bridge/command_enable std_srvs/srv/SetBool '{data: true}'
```

然后向 `/cmd_vel` 发布 `geometry_msgs/msg/Twist`：`linear.x/y` 为底盘坐标系的 m/s，`angular.z` 为 rad/s。默认上限分别为 0.25、0.25、1.0。启用前的输入不执行；收到新输入才开始发送。按回调接收时间计，0.5 秒没有新输入即停止底盘，不中断双臂或升降；新的速度输入可继续控制底盘。

停止控制：

```bash
ros2 service call /alohamini_lerobot_bridge/command_enable std_srvs/srv/SetBool '{data: false}'
```

停止请求状态见 `/diagnostics` 的 `stop_pending`、`command_status`，仍须确认实物停止。保护或失联后检查机器人并重新启用；更换客户端前等待 Host 释放控制权。

## 轨迹与 Jog

启用同一命令通道后，双臂、夹爪、升降与底盘共用一个 Host 控制权：

| 接口 | 消息类型 | 位置单位 |
| --- | --- | --- |
| `/left_arm_controller/follow_joint_trajectory`、`/right_arm_controller/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` | rad |
| `/lift_controller/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` | m（URDF `vertical_move`，不是离地高度） |
| `/left_gripper_controller/gripper_cmd`、`/right_gripper_controller/gripper_cmd` | `control_msgs/action/GripperCommand` | rad |

轨迹使用带时间的位置点，点间线性插值；起始时间戳为零。支持位置误差容限与取消，同一控制组的新目标抢占旧目标。速度／加速度字段不作为前馈，非零速度／加速度容限和力前馈会被拒绝。夹爪 `max_effort` 须为零；电流保护由 Host 执行，返回的 `effort=NaN` 表示没有力估计。接触导致未达到目标时返回 `stalled=true`，不声称抓取成功。

`/left_arm_controller/joint_jog`、`/right_arm_controller/joint_jog` 接收 `control_msgs/msg/JointJog`，使用对应六关节的标准顺序和 rad 增量；`/lift_controller/joint_jog` 使用 `vertical_move` 的速度字段（m/s），非零值按方向生成默认 50 mm 前视目标，不保证按该数值匀速运动。须填写当前 ROS 时间戳并持续发送。过期输入被丢弃；手臂停止输入后保持最新反馈位置。升降停止由 Host 先归零速度，再锁定后续本机高度反馈；升降须已有有效高度参考。

## MoveIt

连接 Host、发布真实状态并启动 MoveIt：

```bash
ros2 launch alohamini_bringup hardware.launch.py host:=<PI_IP> \
  enable_moveit:=true use_rviz:=true
```

检查模型姿态与实物一致，再通过 `command_enable` 服务启用执行。MoveIt 使用双臂、夹爪和升降的上述 action；启动 MoveIt 本身不会启用运动。

旧入口 `alohamini_moveit_config hardware_execution.launch.py` 仅启动 MoveIt；`alohamini_joycon_teleop hardware.launch.py` 仅启动 Joy-Con 组件。两者都不启动 Bridge；整机使用上面的 Bringup 命令。

不连接机器人，仅离线规划：

```bash
ros2 launch alohamini_moveit_config plan_only.launch.py
```

离线状态、TF 和规划服务位于 `/alohamini_plan_only`，不允许硬件执行。无图形界面时追加 `use_rviz:=false`。

模型与离线验收：

```bash
ros2 run alohamini_validation validate_assets
# 启动上述 plan_only 后，在另一终端执行：
ros2 run alohamini_validation validate_moveit
ros2 run alohamini_validation validate_tf
```

依次检查模型/FK/碰撞几何、MoveIt 和 TF。后两项默认访问 `/alohamini_plan_only`；检查真机 ROS 图时加 `--ros-args -r __ns:=/`。工具不发送动作，实机仍须检查运动区域。

两种入口使用相同的参考姿态、折叠、闭爪及已知碰撞样例检查实际有效性；这些样例不替代实机间隙验证。

## Joy-Con 遥操

适用 `alohamini2pro`。先将 Joy-Con 与 PC 配对，安装 [`joycon` 扩展](install.md#3-可选扩展)，再按本文开头构建 ROS 包：

```bash
conda deactivate
source ~/Alohamini/ros2/env.bash
ros2 launch alohamini_joycon_teleop preview.launch.py
```

启动后保持手柄静止约两秒完成 IMU 校准。读取器独立运行于 `alohamini` 环境；ROS 节点使用系统 Python。预览无需 Host，同时启动只规划的 MoveIt，状态、TF 和 FK/IK 服务位于 `/alohamini_plan_only`。不要再单独启动 `plan_only.launch.py`。不接手柄时可加 `start_native_reader:=false`，无桌面时加 `use_rviz:=false`。

启动入口自动启动 Pink IK 工作进程，默认使用 `conda run -n alohamini python`；可通过 `ik_python:=/绝对路径/python` 指定已安装 `kinematics` 扩展的解释器。工作进程只计算关节目标；Host 命令仍由 ROS Bridge 发送。停止 launch 时一并退出，工作进程异常退出会结束该 launch。

真机整机启动（不要重复启动 Bringup 或读取器）：

```bash
ros2 launch alohamini_bringup hardware.launch.py host:=<PI_IP> \
  enable_joycon:=true use_rviz:=true
```

需要 MoveIt 服务时追加 `enable_moveit:=true`，仍只开启一个 Joy-Con RViz。
读取器默认使用 `alohamini` 环境；可用 `native_python:=/绝对路径/python` 指定解释器，或用 `start_native_reader:=false` 接入已运行的读取器。

检查姿态后通过前述 `command_enable` 服务手动启用。松开所有按钮、摇杆回中，再开始操作；保护或失联恢复后同样需要重新检查、启用和回中。不要同时运行预览和真机读取器争用同一手柄。

- 左右手柄分别控制对应手臂；按住 `SL/SR` 后用摇杆平移 TCP、转动手柄改变相对姿态。臂基坐标为 `+X` 向机器人左侧、`-Y` 向前、`+Z` 向上。
- 手臂控制时，肩键上移、摇杆按下下移；`ZL/ZR` 切换夹爪，`Capture/Home` 重新锁定当前姿态。
- 未按 `SL/SR` 时，摇杆控制底盘；右肩键加横向摇杆转向，左肩键加纵向摇杆控制升降。

默认差分 IK 不做碰撞规划，须留出安全空间。Host 电流保护与 ROS 关节限位仍有效，不能替代避障。
差分跟踪使用 Pink／Pinocchio／DAQP，按实际控制周期限制关节位置、速度及加速度。`ik_max_joint_acceleration_rad_s2` 默认 8；`dls_*` 增益、速度、位移及限位余量参数继续有效。RPC 超时默认 0.05 s，失败时停止双臂的新动作并锁定故障。参数文件为 `ros2/src/alohamini_joycon_teleop/config/joycon.yaml`。

记录原始手柄输入并离线回放（相对路径保存到工作区 `logs/`）：

```bash
ros2 run alohamini_joycon_teleop joycon_input_log record joycon.ndjson
ros2 run alohamini_joycon_teleop joycon_input_log replay joycon.ndjson
```

默认记录真机输入端口 `5567`；记录预览时追加 `--endpoint tcp://127.0.0.1:5568`。回放只供不启动读取器的预览使用，真机模式拒绝带回放标记的输入。读取器诊断日志为工作区 `logs/joycon_sticks.log`。

## Gazebo 仿真

Gazebo Fortress 扩展在 `simulation/gazebo/`。安装 [Gazebo 依赖](install.md#5-ros2-依赖) 后，在 ROS2 终端构建并启动：

```bash
cd ~/Alohamini/ros2
/usr/bin/python3 -m colcon build --base-paths src ../simulation/gazebo \
  --packages-up-to alohamini_gazebo
source install/local_setup.bash
export ROS_LOG_DIR="${ALOHAMINI_WORKSPACE:-$HOME/Alohamini_workspace}/logs/ros2"
ros2 launch alohamini_gazebo simulation.launch.py
```

无图形界面时追加 `headless:=true`。运行自动升降抓放演示时，改用 `lift_pick_place_demo.launch.py`；成功或失败后自动退出。

仿真命令、状态、TF 和时钟位于 `/alohamini_sim`，不启动真机桥。使用平台同一份模型几何；底盘采用平面运动控制、手臂采用理想位置执行、抓取采用固定连接，不用于验证真实轮地摩擦、夹持力或重力补偿。

## 单独订阅相机

```bash
export ROS_LOG_DIR="${ALOHAMINI_WORKSPACE:-$HOME/Alohamini_workspace}/logs/ros2"
ros2 launch alohamini_camera camera.launch.py host:=<PI_IP>
```

默认跟随 Host 实际开启的相机；只订阅部分相机时追加 `cameras:="[forward, wrist_right]"`。该节点只接收图像，不发送运动命令，不打开本地相机。Host 重启后自动识别新会话。

每路相机的话题：

- `/alohamini/cameras/<name>/image_raw/compressed`：标准 JPEG。
- `/alohamini/cameras/<name>/image_raw`：RGB，仅在有订阅者时解码。
- `/alohamini/cameras/<name>/camera_info`：有可用内参时发布。

默认使用 PC 接收时间。需要 Host 采集时间时追加 `timestamp_mode:=host_wall`，并先同步 PC 与树莓派系统时钟；移动相机标定不能把接收时间当作曝光时间。

## 标定文件

默认读取 `~/Alohamini_workspace/calibration/cameras/`，支持 `ALOHAMINI_WORKSPACE`；自定义目录用 `calibration_dir:=/绝对路径`。

- `intrinsics/<相机名>.yaml`：ROS CameraInfo 格式，名称、尺寸和光学坐标系须匹配。带候选或未验收状态的文件不用于发布 CameraInfo；没有内参仍可接收图像。
- `extrinsics/<文件名>.yaml`：`mount_link` → `optical_frame` 的外参，默认不发布 TF。

显式发布已验收外参：

```bash
ros2 launch alohamini_camera camera.launch.py host:=<PI_IP> \
  enable_extrinsics:=true extrinsics:=forward.yaml,wrist_right.yaml
```

外参须有 `accepted_` 开头的状态、`T_mount_link_from_camera_optical.xyz_m`，以及明确的 `quaternion_xyzw` 或 `quaternion_wxyz`。默认拒绝候选结果，不覆盖标定文件中的坐标系。

## 关节与升降映射

双臂先按 [单姿态标定](alohamini.md#3-单姿态标定) 执行 `alohamini calibrate robot --robot_model alohamini2pro`，自动生成 `hardware_joint_map_left.yaml`、`hardware_joint_map_right.yaml`，无需再次摆位采样。

ROS 运行在 PC 时，将树莓派生成的映射复制到 PC 的工作区；重做从臂标定后同步更新：

```bash
mkdir -p ~/Alohamini_workspace/calibration/hardware
scp '<用户>@<PI_IP>:Alohamini_workspace/calibration/hardware/hardware_joint_map_*.yaml' \
  ~/Alohamini_workspace/calibration/hardware/
```

两端使用相同型号规格。桥接会核对映射与 Host 的舵机参数；自定义路径用 `arm_mapping_dir:=/绝对路径`。

导入已有相机、升降等资产（不覆盖现有文件、不写舵机）：

```bash
ros2 run alohamini_calibration import_calibration \
  --source ~/alohamini_ros2/src/alohamini_calibration/config \
  --output ~/Alohamini_workspace/calibration/imports/previous_robot
```

输出保留测量与候选状态；舵机 JSON 放在 `robots/`，ROS 映射在 `hardware/`，相机结果在 `cameras/`。`import_manifest.json` 记录来源、校验值和读取问题。导入不自动启用，须核对当前设备；候选内外参仍须单独验证。

升降机构需要实测映射时使用以下只读工具，适用 `alohamini2pro`，不使能、回零或写 EEPROM。

升降：确认 Host 已建立有效高度参考、机构处于真实 Home，再启动：

```bash
ros2 run alohamini_calibration calibrate_lift_axis --host <PI_IP>
```

按提示用已有控制方式移动升降，停止后实测相对 Home 的高度并输入毫米值；至少采集两个不同高度。无需撞击上限取点。空行结束，输出拟合报告和候选映射；同名 `.samples.yaml` 保存已完成采样，中断不会丢失这些点。工具不更改 Host 导程、方向或软限位，候选映射的上界仅取实际采样的最大高度。

核对方向、RViz 姿态和已测升降行程后，将升降候选文件安装为 `lift_axis.yaml`，保留原文件备份。双臂映射由单姿态标定生成，不需要此采样过程。

## 相机标定

工具只采集图像与 TF、计算候选参数，不控制机器人。采集目录和结果默认保存在 `~/Alohamini_workspace/calibration/cameras/`，支持 `ALOHAMINI_WORKSPACE`；终端显示实际路径。`--output` 可指定新的采集目录或结果文件，不覆盖已有内容。采集中断时已写入清单的样本保留。

准备 ChArUco 板：

```bash
AM_BOARD="$(ros2 pkg prefix --share alohamini_calibration)/config/cameras/boards/charuco_9x7_26mm_18p7_ids300_330.yaml"
ros2 run alohamini_calibration generate_charuco_board --board "$AM_BOARD"
```

打印工作区 `calibration/cameras/boards/` 下的 PDF，选择 100% 实际尺寸，确认方格边长为 26 mm。

内参采集与求解（右腕相机）：

```bash
ros2 run alohamini_calibration capture_camera_calibration \
  --host <PI_IP> --camera wrist_right --count 40
ros2 run alohamini_calibration calibrate_camera_intrinsics \
  --capture-dir <内参采集目录> --board "$AM_BOARD" \
  --camera wrist_right --frame-id right_camera_optical
```

Host 须开启对应相机；改变标定板距离、角度和画面位置。默认预览，无桌面时加 `--no-preview`。采集分辨率须与使用时一致。

手眼标定前，确认双臂 URDF 映射正确，并同步 PC 与树莓派系统时钟。整机状态和相机均使用 Host 时间：

```bash
ros2 launch alohamini_bringup hardware.launch.py host:=<PI_IP> \
  state_timestamp_mode:=host_wall camera_timestamp_mode:=host_wall
```

右腕相机固定在手臂上，标定板固定在环境中：

```bash
ros2 run alohamini_calibration capture_hand_eye_samples \
  --preset wrist_right
ros2 run alohamini_calibration calibrate_hand_eye \
  --capture-dir <手眼采集目录> --intrinsics <右腕内参文件>
```

`--preset` 支持 `forward`、`backward`、`chest`、`wrist_left`、`wrist_right` 或 YAML 路径。预设提供标定类型、话题、坐标系、板定义和采样阈值，显式参数优先；采集目录保存实际配置与板文件，求解自动读取。旧采集目录没有板信息时，仍须提供 `--board` 和 `--optical-frame`。

通过已有控制方式改变手臂姿态，每个姿态稳定后自动采样；须包含多个旋转轴。固定相机使用 `eye_to_hand`：标定板固定在手臂上，采集期间保持相机、底盘与升降不动，并填写对应话题及坐标系。工具按图像时间查询 TF，不自动确认时钟同步或时间戳配置。

候选结果保存在 `intrinsics/`、`extrinsics/`。检查重投影误差、独立姿态下的对齐和实际尺寸后，再将状态标记为 `accepted_` 开头并按上一节安装；内参命名为 `<相机名>.yaml`。不要仅凭求解成功就启用外参。

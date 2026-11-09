# Alohamini2 EDU：传感器、建图与导航

EDU 按 2 Pro 的机械、电机与运动学配置增加 C1 雷达和幻尔 USB IMU，机型 ID 为 `alohamini2edu`。设备标定独立保存，不自动复用其他机器的零偏和行程。原生遥操／策略与 ROS 运动控制择一运行。

仅指定 `--robot_model alohamini2edu` 启动原生 Host，不会自动开启雷达、SLAM、录包或避障。

| 入口 | 自动启动的功能 |
| --- | --- |
| `sensors.launch.py` | 雷达、IMU、安装 TF、扫描过滤及覆盖范围显示；不连接 Host |
| `mapping.launch.py` | Bridge、EKF、碰撞监测、SLAM、地图保存；默认使用外部传感器 |
| `navigation.launch.py` | Bridge、EKF、碰撞监测、AMCL、Nav2；需要地图 |
| `edu_session` | 上述建图／导航流程、键盘、专用 RViz、Host 相机订阅；`--record` 才开启录制 |

ROS 控制默认禁用，须显式使能；导航还需要初始位姿和目标。原生遥操命令不经过 ROS 碰撞监测。

| 层级 | 位置与职责 |
| --- | --- |
| 机型资源 | `models/alohamini2edu/`：电机清单、URDF／网格、双臂运动学、传感器安装与导航预设 |
| 硬件控制 | 原生 `NativeHost` 独占电机串口；ROS 不另启 `ros2_control` 电机驱动 |
| ROS 接入 | `alohamini_bridge` 内的一个 HostClient：命令、状态、独立底盘超时与控制权校验 |
| 传感器 | `alohamini_rplidar_c1`、`alohamini_imu`：USB 驱动；`alohamini_sensors`：安装 TF、扫描过滤 |
| 定位／导航 | `alohamini_navigation`：EKF、slam_toolbox、Nav2、地图版本与 rosbag 录制 |

```text
ROS /cmd_vel → collision_monitor → /cmd_vel_safe → Bridge → ZMQ → Host → 电机总线
Host 实测轮速 → Bridge /wheel/twist ─┐
USB IMU → /imu/data_raw ───────────┴→ EKF → /odom、odom→base_link
USB C1 → /scan → /scan_filtered ─────→ AMCL → map→odom
                  └→ scan_gate → /scan_slam → SLAM → map→odom
```

雷达／IMU 独立接树莓派 USB，不接入 Feetech 电机总线，也不放入 Host 电机状态 JSON。两端使用同一 ROS_DOMAIN_ID 并保证 DDS 可达；Host 仍用 ZMQ。PC 运行 Bridge、EKF、SLAM／Nav2，树莓派只需运行 Host 和所需传感器驱动。

ROS 域建议使用 0–101（本流程实机用 5、离线用 42–44），避免 Linux 临时端口范围的冲突，参见 [ROS 2 域编号说明](https://github.com/ros2/ros2_documentation/blob/humble/source/Concepts/Intermediate/About-Domain-ID.rst)。

## 安装与设备

ROS 节点使用 Humble 的系统 Python。已有 Debian 树莓派可使用 Humble 对应容器，不直接套用 Ubuntu apt 源。Ubuntu 22.04／Humble 环境中的 PC 构建：

PC 使用 `tf2_ros >= 0.25.24` 及配套 RViz 包，包含 [TF 等待死锁修复](https://github.com/ros2/geometry2/pull/990)。已装旧版 Humble 的 PC 先更新：

```bash
sudo apt-get update
sudo apt-get install ros-humble-tf2-ros ros-humble-rviz2 \
  ros-humble-rviz-common ros-humble-rviz-default-plugins
```

```bash
source /opt/ros/humble/setup.bash
cd ~/Alohamini/ros2
rosdep install --from-paths src ../simulation/navigation --ignore-src -r -y
bash ../deploy/edu/build-pc.sh
source install/local_setup.bash
export ROS_DOMAIN_ID=5 ROS_LOCALHOST_ONLY=0
```

树莓派只构建 `--packages-up-to alohamini_sensors`，不必为传感器安装 SLAM／Nav2。Host 使用原有 `alohamini_host` 环境和启动参数，机型改为 `--robot_model alohamini2edu`。仅注册型号不会改变已有舵机 EEPROM 标定。

PC 构建包含 `third_party/nav2_rviz_plugins`（固定为 Nav2 1.1.20，修复面板线程清理）和 `alohamini_rviz`（修复 Humble 相机显示重复取消订阅）。`build-pc.sh` 显式覆盖系统同名 Nav2 插件；运行前须 source 本工作区的 `install/local_setup.bash`。这些界面包不进入树莓派传感器容器。升级 ROS／Nav2 时须检查对应源码兼容性并重跑退出检查。

默认设备为 `/dev/lidar_c1`（460800）和 `/dev/imu_usb`（115200）；可用启动参数指定 `/dev/serial/by-id/...`。串口用户须有 `dialout` 权限。C1 的 udev 模板位于 `alohamini_rplidar_c1/udev/`，填写实际 USB 序列号后安装；不能仅凭相同 VID/PID 区分多个适配器。没有唯一序列号的 IMU 可使用固定 USB 插口的 `/dev/serial/by-path/...`。

传感器安装配置为 `models/alohamini2edu/config/sensors.yaml`。以 Pro 的 REP-103 `base_link` 为基准：X 前、Y 左、Z 上。雷达的旧 CAD 安装变换已换算到此坐标系，不再额外旋转底盘。默认雷达位置是名义值；IMU 平移尚未实测，只使用不依赖平移的 Z 轴角速度。实际外参放入工作区 `calibration/sensors/` 的配置副本，通过 `sensor_profile:=/绝对路径/sensors.yaml` 指定。

## 独立启动传感器

在连接 USB 的主机上执行：

```bash
ros2 launch alohamini_sensors sensors.launch.py
# 可选：lidar_port:=/dev/serial/by-id/... imu_port:=/dev/serial/by-path/...
```

此入口只发布传感器话题及安装 TF，可与原生遥操并行记录。`enable_imu:=false` 可只启动雷达；`publish_mounts:=false` 用于已有其他唯一安装 TF 发布者的场景。不要重复运行同一设备驱动。

`/scan_filtered` 保留原始 `/scan` 时间戳；陈旧、未来和乱序数据被丢弃。C1 时间是驱动主机时间，IMU 为 USB 接收时间，均不声称硬件同步。PC／树莓派须同步系统时钟；建图默认以 `state_timestamp_mode:=host_wall` 发布轮速。

## 建图与底盘键盘控制

先启动 Host 和树莓派传感器，再在 PC 上执行：

```bash
ros2 launch alohamini_navigation mapping.launch.py host:=192.168.44.4
```

传感器也连接 PC 时追加 `start_sensors:=true`，并停止另一处驱动。默认 `base_only:=true`，无需双臂／升降 URDF 映射；仍要求 Host 原有就绪状态与控制权条件。需要显示真实整机时使用 `base_only:=false arm_mapping_dir:=/实际EDU映射目录`。导航描述移除了规划用虚拟根关节，避免与 EKF 的 `odom→base_link` 冲突。

默认 EKF 融合轮速的 vx／vy 和 IMU 的 gz，不融合含重力的加速度或未标定绝对航向。没有 IMU 时显式追加 `use_imu:=false`，改用轮速推算的 yaw rate；这不是运行中自动降级。

确认状态与方向后启用 ROS 控制：

```bash
ros2 service call /alohamini_lerobot_bridge/command_enable std_srvs/srv/SetBool '{data: true}'
ros2 run alohamini_navigation base_teleop
```

`w/s` 前后、`a/d` 旋转、`z/x` 横移，空格停、`q` 退出。最后一次按键重复后 0.35 s 归零，Bridge 的 `/cmd_vel` 另有 0.5 s 超时。键盘不自动使能控制，也不依赖主臂遥操。

建图使用 slam_toolbox 的异步 2D 扫描匹配，默认 5 cm 分辨率、8 m 最大范围并启用回环。`slam_scan_gate` 根据 `/odom` 筛选 `/scan_filtered`，移动达到 5 cm 或转动达到 0.05 rad 时发布 `/scan_slam`，支持原地旋转建图；碰撞监测继续使用完整频率的 `/scan_filtered`。SLAM 参数位于模型的 `config/navigation/slam.yaml`，应保留 `/scan_slam` 输入和为零的内部平移门限。RViz 固定坐标系选 `map`。

## 保存、继续建图与使用地图

在运行 SLAM／map_saver 的同一台 PC 上保存：

```bash
ros2 run alohamini_navigation map_session save office
```

先禁用 Bridge 命令并等待新的静止反馈，再保存到 `~/Alohamini_workspace/maps/office/<UTC版本>/`：

- `map.yaml`、`map.pgm`：Nav2 定位导航。
- `slam.posegraph`、`slam.data`：slam_toolbox 继续建图。
- `manifest.json`、`model_config/`：机型、Host 会话、文件校验值和名义模型配置。运行时参数覆盖和实机标定不由名义配置代替。

可用 `--revision v1` 指定新版本，禁止覆盖；`--grid-only` 仅导出栅格。失败数据保存在 `.incomplete-*`，不能作为完整地图加载。保存后命令保持禁用，继续行驶需重新启用。

退出或切换模式时，先停用控制与生命周期节点，再在 launch 终端按 Ctrl+C：

```bash
ros2 run alohamini_navigation map_session stop
```

此顺序让 Nav2／碰撞监测在 TF 提供者退出前清理。Humble 碰撞监测在同时撤销 TF 与清理时可能崩溃，应先执行上述停止命令再关闭 launch。

结束原建图进程后，从位姿图继续（机器人置于原首次建图位置）：

```bash
ros2 launch alohamini_navigation mapping.launch.py host:=192.168.44.4 \
  pose_graph:=/绝对路径/maps/office/版本/slam
```

使用栅格导航时结束 SLAM，改启动 AMCL＋Nav2：

```bash
ros2 launch alohamini_navigation navigation.launch.py host:=192.168.44.4 \
  map:=/绝对路径/maps/office/版本/map.yaml
```

在 RViz 设置初始位姿，再启用 Bridge 并发送导航目标。不要同时启动 `mapping.launch.py`、`navigation.launch.py` 或另一个整机 Bringup。

默认 Nav2 使用 OmniMotionModel、NavFn、DWB 和速度平滑，经过碰撞监测后才到 Bridge。当前前方 180° 扫描预设限制自动平移为前进，恢复行为不倒车；键盘反向／横移需人工观察。`config/navigation/nav2.yaml` 的轮廓仅描述名义收拢底盘，不包含伸展双臂。扩大雷达覆盖和核对实机轮廓后，才在自定义 `nav2_params:=/路径/nav2.yaml` 中启用相应全向速度，并以 `collision_params:=/路径/collision.yaml` 配置对应的碰撞监测范围。

## ROS 数据录制与双臂

```bash
ros2 run alohamini_navigation map_session record office_episode_001
```

写入工作区 `runs/office_episode_001/bag/`，记录扫描、IMU、轮速、里程计、TF、关节、相机 JPEG／内参、采样时钟关联、地图、ROS 命令和诊断。Ctrl+C 正常刷新文件；结束清单列出各话题消息数和缺失项。可用 `--cameras forward wrist_left wrist_right` 选择相机，`--duration 30` 定时结束，`--require-topics /scan /imu/data_raw` 检查必需话题。录制器不创建 HostClient、不启用运动，不存在的话题不会产生数据。

双臂 FK／IK 使用平台的同一套算法，EDU 通过自己的模型资源加载。完整 ROS 整机入口用 `alohamini_bringup hardware.launch.py robot_model:=alohamini2edu`；MoveIt／Joy-Con 预览同样支持该参数。关节映射须属于 EDU，`sync_arm_mapping` 根据实际 Host 型号生成模板，不能直接更改另一台机器标定文件中的型号来复用。

## 树莓派容器与设备检查

PC 已有 Humble 时无需 Docker。树莓派已有 Humble 时直接构建传感器包；Debian 主机可使用传感器容器。将本仓库源码放到树莓派 `~/Alohamini`，在树莓派执行：

```bash
cd ~/Alohamini
python3 deploy/edu/sensors-container build
python3 deploy/edu/sensors-container run --domain 5 \
  --lidar /dev/serial/by-id/实际雷达设备 \
  --imu /dev/serial/by-path/实际IMU设备
```

容器只运行传感器，不安装 SLAM／Nav2，不启动 Host，也不映射电机串口；使用 host 网络和 IPC。原生 Host 继续在现有环境中运行，相机须在 Host 启动时显式开启，例如原有 `--cameras forward` 参数。`--dry-run` 查看容器命令；另一个终端用 `python3 deploy/edu/sensors-container stop` 停止。

镜像在运行 USB 驱动的机器上构建，PC 的 x86-64 程序不能直接作为树莓派 ARM64 程序使用。容器只映射选定设备，拔插后若设备节点改变，须重启容器刷新映射。实际安装配置用 `--profile /绝对路径/sensors.yaml` 挂载。

两端宿主机用 `timedatectl status` 检查时钟同步。启动后在 PC 执行只读检查：

```bash
ros2 run alohamini_navigation edu_check
# mapping/navigation 已启动时增加轮速检查：
ros2 run alohamini_navigation edu_check --require-wheel
```

检查关注消息新鲜度、有效扫描和 IMU 数值，不使能电机。静止时 IMU 加速度模长应约为 9.81 m/s²、角速度接近零；安装轴向、零偏、轮速符号、里程计尺度和停止距离仍需实机验收。

## 统一会话、相机与导航 RViz

Host 和树莓派传感器启动后，在 PC 执行：

```bash
ros2 run alohamini_navigation edu_session --host 192.168.44.4 \
  --map-name office --record office_run_001
```

默认启动建图、专用 RViz 和 Host 相机订阅。`--no-camera` 关闭图像订阅，`--no-rviz` 用于无桌面环境；传感器接在 PC 时用 `--start-sensors`。运行日志位于工作区 `logs/navigation/`。

| 按键 | 功能 |
| --- | --- |
| `e` | 显式使能当前 ROS 控制 |
| `w/s`、`a/d`、`z/x` | 前后、旋转、横移 |
| `r/f` | 提高／降低线速度，范围 0.03–0.20 m/s |
| 空格、`0` | 停止；最后按键重复后 0.35 s 也会归零 |
| `Shift+S` | 停用控制、确认静止、保存；继续运动需再按 `e` |
| `Shift+X` | 保存成功后停止并退出 |
| `q`、Ctrl+C | 停止并退出，不保存地图 |

退出时先禁用运动、关闭 RViz，再停止导航生命周期节点和 launch，并结束、刷新录包。导航模式：

```bash
ros2 run alohamini_navigation edu_session --mode navigation \
  --host 192.168.44.4 --map /绝对路径/maps/office/版本/map.yaml
```

RViz 预置地图、扫描、覆盖范围、全局／局部路径、全局／局部代价地图、TF、机器人和前置相机。用 **2D Pose Estimate** 设置初始位姿，检查地图与扫描重合后按 `e`，再用 **Nav2 Goal** 发送目标。建图期间尚未产生的导航话题不会显示数据。

导航模式由 Nav2 发布速度，不接受键盘方向键；空格／`0` 会禁用运动并取消导航目标。手动驾驶与保存快捷键用于建图模式，避免两个速度发布者互相覆盖。

无桌面或无人值守任务用 `--no-rviz`；录制是否完整以对应 `manifest.json` 的 `status` 为准。


独立 mapping/navigation launch 同样支持 `enable_cameras:=true use_rviz:=true`；相机使用 Host 图像流，不另开 USB 摄像头。可用 `ekf_params`、`collision_params` 覆盖配置；建图用 `slam_params`，导航用 `nav2_params`。

## 对齐原生数采回合

与原生遥操数采并行时，只启动传感器和相机订阅，不启动 mapping/navigation：

```bash
ros2 launch alohamini_camera camera.launch.py host:=192.168.44.4 \
  timestamp_mode:=host_wall publish_raw:=false
```

相机节点发布 Host 会话及单调时钟／Unix 时钟关联，不额外创建 HostClient。原生回合和 rosbag 保存结束后，用同一 Host 会话的记录对齐：

```bash
ros2 run alohamini_navigation map_session attach /绝对路径/记录/bag \
  /绝对路径/原生数据集 --episode-index 0
```

也可在 `record` 时指定 `--dataset /路径 --episode-index 0`，录包结束后自动对齐；此时原生回合必须已保存完成。支持平台 v3 的 `meta/safety/` 和本地 `episodes/` 布局。

结果位于数据集 `ros_sensors/episode_000000/`，按原生回合／帧索引关联，保留采样时间、坐标系、单位及有效性。扫描最多取前 150 ms、IMU 最多取前 50 ms 的样本，不取未来数据；缺失标为无效，扫描无效距离保留为空值。Host 会话不同、时钟关联缺失／跳变或原生回合不完整时拒绝对齐。相机关闭且未记录 Bridge 状态时，没有时钟关联，不能自动对齐。

```python
from alohamini.datasets.ros_sensors import iter_sensor_frames
for frame in iter_sensor_frames(dataset_path, 0):
    scan = frame["sensors"]["scan"]  # valid、sample、age_ns
    imu = frame["sensors"]["imu"]
```

该扩展保持原生动作、图像数据格式，不自动改变策略输入维度；策略须显式读取传感器字段及有效性。加速度含重力，不直接当作平移加速度。

## 仿真与回放

在单独终端使用专用本机 ROS 域，避免与实机话题混用：

```bash
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1
ros2 run alohamini_navigation edu_session --simulation \
  --map-name simulated_room --record simulated_run_001
```

`simulation/navigation/` 提供几何环境、雷达射线、IMU、模拟实测速度和 `/clock`，运行真实 EKF、SLAM、碰撞监测与 Nav2；没有 HostClient、USB 驱动或相机网络连接。使用平面运动学及圆形轮廓碰撞约束，不模拟轮胎接触、打滑、IMU 噪声和伸展手臂。默认环境在 `worlds/room.yaml`。

```bash
ros2 launch alohamini_nav_sim simulation.launch.py mode:=navigation \
  map:=/绝对路径/maps/simulated_room/版本/map.yaml use_rviz:=true
ros2 run alohamini_navigation map_session stop --runtime-mode simulation
```

观察录包或用原始数据重新建图：

```bash
ros2 launch alohamini_navigation replay.launch.py bag:=/绝对路径/记录/bag
ros2 launch alohamini_navigation replay.launch.py bag:=/绝对路径/记录/bag mode:=mapping
ros2 run alohamini_navigation map_session save replayed --runtime-mode replay
ros2 run alohamini_navigation map_session stop --runtime-mode replay
```

回放统一使用 `/clock`，不会重发运动命令。重新建图输入原始扫描、IMU、轮速和静态 TF，不重播旧动态 TF、里程计及地图。缺少 `/wheel/twist` 或安装 TF 的纯传感器记录只能观察，不能恢复移动轨迹。

离线集成检查：

```bash
ROS_DOMAIN_ID=43 ROS_LOCALHOST_ONLY=1 /usr/bin/python3 \
  simulation/navigation/test/stack_check.py \
  --output ~/Alohamini_workspace/logs/edu-navigation-check-新目录
```

终端快捷键与退出检查使用 `simulation/navigation/test/session_check.py`，指定新的 `--output` 和上一检查生成的 `--navigation-map /路径/workspace/maps/room/v1/map.yaml`；有 X 显示环境时增加 `--screenshot` 检查 RViz。相机画面为带标识的离线测试图。

RViz 独立退出检查需要 `xvfb`、`python3-xlib`，覆盖 SIGINT、SIGTERM 和窗口关闭，每种重复三次：

```bash
ROS_DOMAIN_ID=52 ROS_LOCALHOST_ONLY=1 xvfb-run -a /usr/bin/python3 \
  simulation/navigation/test/rviz_shutdown_check.py \
  --output ~/Alohamini_workspace/logs/rviz-shutdown-check
```

实机验收仍须核对外参、底盘几何、停止距离、异常恢复和双机时钟；仿真结果不替代这些测量。

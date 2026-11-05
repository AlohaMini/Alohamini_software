# Alohamini2 EDU：传感器、建图与导航

EDU 按 2 Pro 的机械、电机与运动学配置增加 C1 雷达和幻尔 USB IMU，机型 ID 为 `alohamini2edu`。设备标定独立保存，不自动复用其他机器的零偏和行程。原生遥操／策略与 ROS 运动控制择一运行。

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
USB C1 → /scan → /scan_filtered ─────→ SLAM 或 AMCL → map→odom
```

雷达／IMU 独立接树莓派 USB，不接入 Feetech 电机总线，也不放入 Host 电机状态 JSON。两端使用同一 ROS_DOMAIN_ID 并保证 DDS 可达；Host 仍用 ZMQ。PC 运行 Bridge、EKF、SLAM／Nav2，树莓派只需运行 Host 和所需传感器驱动。

## 安装与设备

ROS 节点使用 Humble 的系统 Python。已有 Debian 树莓派可使用 Humble 对应容器，不直接套用 Ubuntu apt 源。Ubuntu 22.04／Humble 环境中的 PC 构建：

```bash
source /opt/ros/humble/setup.bash
cd ~/Alohamini/ros2
rosdep install --from-paths src --ignore-src -r -y
/usr/bin/python3 -m colcon build --packages-up-to alohamini_navigation \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/local_setup.bash
```

树莓派只构建 `--packages-up-to alohamini_sensors`，不必为传感器安装 SLAM／Nav2。Host 使用原有 `alohamini_host` 环境和启动参数，机型改为 `--robot_model alohamini2edu`。仅注册型号不会改变已有舵机 EEPROM 标定。

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

建图使用 slam_toolbox 的异步 2D 扫描匹配，默认 5 cm 分辨率、8 m 最大范围并启用回环。参数位于模型的 `config/navigation/slam.yaml`。RViz 可显示 `/map`、`/scan_filtered`、TF，固定坐标系选 `map`。

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

写入工作区 `runs/office_episode_001/`，记录扫描、IMU、轮速、里程计、TF、ROS 命令、Host 状态和诊断；Ctrl+C 结束 rosbag。录制器不创建 HostClient、不启用运动。仅启动传感器时也可录制，但不存在的话题不会产生数据。这是 ROS 原始记录，尚未转换为平台训练数据集或自动与原生数采回合对齐。

双臂 FK／IK 使用平台的同一套算法，EDU 通过自己的模型资源加载。完整 ROS 整机入口用 `alohamini_bringup hardware.launch.py robot_model:=alohamini2edu`；MoveIt／Joy-Con 预览同样支持该参数。关节映射须属于 EDU，`sync_arm_mapping` 根据实际 Host 型号生成模板，不能直接更改另一台机器标定文件中的型号来复用。

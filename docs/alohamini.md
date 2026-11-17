# AlohaMini 使用手册

先完成[安装](install.md)，将 PC 与树莓派连接到同一局域网。下文将 `<型号>` 替换为实际型号，`<树莓派_IP>` 替换为树莓派地址。

## 1. 型号与接线

| 型号 | 单侧从臂 | 主臂标定 ID |
| --- | --- | --- |
| `alohamini1` | SO-ARM，5 自由度 + 夹爪 | `so101_leader_bi` |
| `alohamini2` | AM-ARM，6 自由度 + 夹爪 | `am_leader_bi` |
| `alohamini2pro` | AM-ARM Pro，6 自由度 + 夹爪 | `am_leader_bi` |

AM-ARM 主臂使用 5 V、从臂使用 12 V。

更新包内型号定义或模型资源后，请重启 Host 和 PC 控制程序，使新资源生效。

| 设备 | 连接位置 | 默认设备名 |
| --- | --- | --- |
| 左、右主臂 | PC USB | `/dev/am_arm_leader_left`、`/dev/am_arm_leader_right` |
| 左从臂、底盘、升降 | 树莓派 USB | `/dev/am_arm_follower_left` |
| 右从臂 | 树莓派 USB | `/dev/am_arm_follower_right` |
| 前视、右腕相机 | 树莓派 USB | `/dev/am_camera_forward`、`/dev/am_camera_wrist_right` |

## 2. 主从臂标定

AM-ARM（`alohamini2`、`alohamini2pro`）按以下 Home 流程标定。每套双主臂、双从臂各标定一次；更换配对机器人无需重标主臂。先停止 Host、遥操和其他串口程序，确认双臂稳固、升降已支撑。

首次使用需设置设备权限，重新登录后生效：

```bash
sudo usermod -aG dialout,video "$USER"
```

PC 标定双主臂：

```bash
conda activate alohamini
alohamini calibrate leader --robot_model <型号> --teleop.id am_leader_bi
```

树莓派标定双从臂：

```bash
conda activate alohamini_host
alohamini calibrate robot --robot_model <型号>
```

首次运行按提示绑定 USB：拔下该组左右 USB，按 Enter；先插左臂，识别后保持连接，再插右臂。按提示输入系统 `sudo` 密码后继续标定。已有正确设备名时自动复用。

将双臂一起摆到 Home：

| 舵机 | 摆位 |
| --- | --- |
| 1：肩部转向、5：腕部偏转、6：腕部旋转 | 对齐中位 |
| 2：肩部抬升、3：肘部弯曲、4：腕部弯曲 | 蜷缩，轻靠对应的连杆或外壳接触面 |
| 夹爪 | 合拢 |

不要挤压打印件。双臂稳固、保持静止后按一次 Enter；程序写入并核对两侧零偏与编码器限位，再保存文件。任一侧失败时恢复原标定。过程不使能扭矩、不自动运动、不执行整机升降回零。

已有标定时按 Enter 复用，输入 `c` 重做。标定掉电保留，正常开机无需重新置零。

识别失败用 `alohamini find-port` 排查；自定义串口须同时传入 `--left_port`、`--right_port`。更换控制板时，备份并移走 `calibration/devices/` 下对应的 `leader.json` 或 `follower.json`，再运行标定；序列号缺失或重复的控制板无法自动绑定。

## 3. 相机

在树莓派查找相机，先停止占用相机的 Host：

```bash
conda activate alohamini_host
alohamini find-cameras opencv
```

相机名称可选 `forward`、`backward`、`chest`、`wrist_left`、`wrist_right`，对应 `/dev/am_camera_<名称>`。默认开启前视和右腕；仅选择已连接、已配置设备名的相机。用 `--record-time-s 0` 只列出设备，不保存预览。

## 4. 启动树莓派 Host

> 启动会使能舵机并执行升降触底回零。

```bash
conda activate alohamini_host
alohamini host --robot_model <型号>
```
Host 控制频率为 50 Hz，相机请求配置为 640×480、30 Hz。

ROS2 等客户端可订阅已开启的相机，见 [ROS2](ros2.md)。

| 需要调整的内容 | 启动命令后追加 |
| --- | --- |
| 仅开启前视相机 | `--cameras forward` |
| 不开启相机 | `--cameras`，后面不填名称 |
| 查看实际 Hz 与耗时 | `--profile_timing` |
| 指定串口 | `--left_port /dev/ttyACM0 --right_port /dev/ttyACM1` |
| 指定整机标定文件 | `--calibration /绝对路径/AlohaMiniRobot.json` |

## 5. PC 查询状态

保持 Host 运行，在 PC 执行：

```bash
conda activate alohamini
alohamini inspect --host <树莓派_IP> --model <型号>
```

只读取一次状态，不发送运动命令。连接失败时先确认 Host、IP、两端型号；字段单位与 Python 调用见[客户端接口](host-protocol.md)。

## 6. PC 遥操

保持 Host 运行，在 PC 执行：

```bash
conda activate alohamini
alohamini teleoperate \
  --host <树莓派_IP> \
  --robot_model <型号> \
  --teleop.id am_leader_bi
```

`--teleop.id` 与标定时一致。默认 50 Hz 控制、30 Hz 图像请求，开启 Rerun 预览。

| 按键 | 操作 |
| --- | --- |
| `w` / `s` | 底盘前进 / 后退 |
| `z` / `x` | 底盘左移 / 右移 |
| `a` / `d` | 底盘左转 / 右转 |
| `t` / `g` | 提高 / 降低速度档位 |
| `u` / `j` | 升降上升 / 下降 |
| `q`、Esc、Ctrl+C | 退出遥操 |

**键盘需要 X11 桌面。**

仅用主臂加 `--no_keyboard`；仅用键盘加 `--no_leader`；关闭预览加 `--no_preview`。

PC 终端显示遥操状态和跟踪信息，树莓派终端显示 Host 的 Hz、耗时与硬件故障。同一时间只运行一个运动控制客户端；**退出遥操后再启动数采**。

## 7. PC 本地数采

```bash
conda activate alohamini
alohamini record \
  --host <树莓派_IP> \
  --robot_model <型号> \
  --teleop.id am_leader_bi \
  --dataset pickup_01 \
  --task "pick up the object" \
  --num_episodes 1 \
  --episode_time 8 \
  --reset_time 3
```

默认 30 Hz 数采，记录 Host 开启的全部相机。`--display_data` 开启预览。
`--profile_timing` 开启采集、保存分段和各相机视频处理计时；默认关闭，不采集详细视频计时。

| 参数 | 用途 |
| --- | --- |
| `--video-encoder nvenc` | 使用 NVIDIA GPU 编码 H.264，需驱动和 PyAV NVENC 支持；默认 `cpu` |
| `--no-background-video-encoding` | 关闭后台编码，默认开启 |
| `--video-encoding-workers 2` | 限制同时编码的相机数量，默认按启用相机数 |
| `--video-encoding-memory-mib 2048` | 编码子进程的内存限额（MiB）；不包含整个录制程序或 GPU 显存 |

NVENC 与 CPU 编码的图像像素和文件大小可能不同。

| 按键 | 操作 |
| --- | --- |
| 右方向键 | 提前结束当前阶段 |
| `R` / 左方向键 | 删除当前 episode 的临时数据并重录 |
| Esc、`q`、Ctrl+C | 结束数采 |

复位阶段可继续遥操，但不记录数据。退出时尝试保存已采到的有效帧。

数据默认保存到 `~/Alohamini_workspace/datasets/pickup_01/`。自定义位置用 `--root`；继续已有数据集用 `--resume`，型号、标定、相机、任务和帧率须一致。

完整内容检查使用 `dataset check --decode-videos`。

任务需要固定部分关节时，先遥操摆好位置，退出遥操，再在上述采集命令后追加：

```bash
--fixed-dimensions arm_right lift_axis
```

支持 `arm_left`、`arm_right`（含夹爪）、`lift_axis`、`base`，也可逐个指定
`arm_right_elbow_flex.pos`、`lift_axis.height_mm` 等动作字段。固定整个手臂时无需连接该侧主臂。固定臂上的相机继续正常记录。
底盘固定表示零速度，不是世界坐标位置保持。
若要保持腕相机的空间位置，还需按任务选择会带动它的升降等父轴；固定手臂不会自动锁住升降。

后续使用相同采集命令加 `--resume`，可省略 `--fixed-dimensions`；程序自动恢复保存的目标。

| 文件 | 内容 |
| --- | --- |
| `meta/info.json`、`meta/alohamini.json` | 字段定义、型号、标定和单位 |
| `data/chunk-*/file-*.parquet` | state、action、索引、舵机反馈及有效掩码 |
| `videos/observation.images.*/chunk-*/file-*.mp4` | 各相机的视频，可直接播放 |
| `meta/episodes/`、`meta/tasks.parquet`、`meta/stats.json` | episode 索引、任务和统计 |
| `meta/safety/episode_*.jsonl` | 保护记录、命令标识与实际采样时间 |

### 检查与恢复

```bash
alohamini dataset check ~/Alohamini_workspace/datasets/pickup_01
```

自动识别 AlohaMini 和 LeRobot v3 数据；两个解码选项检查完整图像和视频。`--output-json <新文件>` 保存报告。`VALID` 表示结构检查通过；训练前仍须检查缺帧和保护事件等警告。

保存中断时保留原目录，恢复到新目录：

```bash
alohamini dataset recover ~/Alohamini_workspace/datasets/pickup_01 \
  --output ~/Alohamini_workspace/datasets/pickup_01_recovered
```

仅恢复完整帧，不修改原数据。异常中断留下的 `*.pending/` 用于恢复，不要手动删除。
按 `R` 或左方向键重录会删除当前回合的临时图片和日志。

AlohaMini 或 LeRobot v3 数据的索引、媒体引用异常时，修复到新目录：

```bash
alohamini dataset repair ~/Alohamini_workspace/datasets/task_demo \
  --output ~/Alohamini_workspace/datasets/task_demo_repaired
```

保留原数据，重建可确定的索引与媒体引用；缺失图像或配对不明时拒绝修复。修复结果不用于追加录制，也不消除实际漏采与时间偏差。

评估记录的 Rerun 查看、训练示范对比和报告导出，见 [训练指南：对比评估与训练示范](training.md#8-对比评估与训练示范)。

### 导出与选择字段

需要另存或选择 state 字段时，在 `alohamini` 环境中执行：

```bash
alohamini dataset export ~/Alohamini_workspace/datasets/pickup_01 \
  --output ~/Alohamini_workspace/datasets/pickup_01_lerobot \
  --format lerobot-v3
```

默认 state 为双臂关节位置、底盘速度、升降高度（alohamini2pro 共 18 维），action 不变，反馈保留。`--state` 选择输入字段；纯视觉副本及训练见 [训练与部署](training.md#导出与纯视觉副本)。

当前数据的视频在导出时原样复制，不重新编码。恢复与导出须使用新目录；`.pending-*` 表示导出未完成。

### 动作回放

回放会驱动整机，不是播放视频。先退出遥操或数采，将机器人置于示教起始姿态，并清空运动区域。

```bash
alohamini replay --dataset pickup_01 \
  --host <PI_IP> --robot_model <型号> --episode 0
```

支持 AlohaMini 数据集及其 LeRobot v3 导出；自定义目录用 `--root`。按 action 执行，型号、动作单位和标定须与 Host 一致。

默认按数据集帧率回放；`--fps` 覆盖帧率，`--speed` 调整倍率。低频或跳帧可能漏掉关键动作，无法保证底盘路径精确复现。

## 8. 工作文件

默认工作目录为 `~/Alohamini_workspace/`，与源码分开：

| 子目录 | 内容 |
| --- | --- |
| `calibration/robots/`、`calibration/teleoperators/` | 整机与主臂标定 |
| `datasets/` | 本地数据集 |
| `runs/` | 训练与评估产物 |
| `incoming/` | 待检查的回流数据 |
| `logs/` | 运行日志与调试输出 |

执行 `alohamini paths` 查看路径。更换根目录时设置 `ALOHAMINI_WORKSPACE` 为绝对路径；已有文件不会自动移动，PC 与树莓派之间也不会自动同步。

## 9. 调试工具

查看舵机状态：

```bash
python examples/debug/motors.py get_motors_states --port /dev/ttyACM0
```

只读查询 ID 1–22，不改变力矩。`POS`、`OFF` 是原始刻度，`CURR(MA)` 是电流 mA；缺失项显示 `-`，`servo_error` 表示舵机故障位。

| 命令 | 用途 |
| --- | --- |
| `python examples/debug/test_cv.py --camera /dev/am_camera_forward` | 检查单路相机 |
| `python examples/debug/test_cuda.py` | 检查 PyTorch 与 GPU |
| `python examples/debug/test_input.py` | 检查键盘输入 |
| `python examples/debug/test_dataset.py /路径/frames.parquet` | 查看数据列与内容 |
| `python examples/debug/test_network.py --help` | 指定 HTTP(S) 地址进行连接检查 |
| `python examples/debug/test_mic.py` | 录音与音量检查，需额外安装 `audio-debug` 和系统 PortAudio |

相机预览、录音等调试输出默认位于工作目录的 `logs/debug/`。麦克风依赖见[可选扩展](install.md#3-可选扩展)。

<details>
<summary>舵机维护：卸力、改 ID、改相位</summary>

```bash
python examples/debug/motors.py reset_motors_torque --port /dev/ttyACM0 --id 1
python examples/debug/motors.py configure_motor_id --help
python examples/debug/motors.py configure_motor_phase --help
```

卸力命令省略 `--id` 时会处理该串口上 ID 1–22 内所有已识别舵机。改 ID 只连接待配置的一颗舵机，并同步检查设备配置与标定；相位值须对应具体型号。维护报错后先确认设备状态，不要直接启动运动。

</details>

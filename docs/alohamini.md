# AlohaMini 使用手册

先完成[安装](install.md)，将 PC 与树莓派连接到同一局域网。下文以 `alohamini2pro`、树莓派 IP `<树莓派_IP>` 为例。

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

## 2. 串口与相机

Ubuntu / Raspberry Pi OS 的设备权限设置，重新登录后生效：

```bash
sudo usermod -aG dialout,video "$USER"
```
已有正确的默认设备名时无需重新配置。

### PC：检查主臂 Leader

```bash
conda activate alohamini
ls /dev/ttyACM*
alohamini find-port
```

按提示只拔下左主臂的 USB，按 Enter，记录输出的端口并接回 USB。再次运行 `alohamini find-port`，用同样方法确认右主臂。一次只拔一侧；没有端口变化或同时消失多个端口时，重新检查。

接回后确认端口号。以下假设左臂为 `ttyACM0`、右臂为 `ttyACM1`，若不同则替换命令中的端口：

```bash
udevadm info --attribute-walk --name=/dev/ttyACM0 | awk -F'"' '/ATTRS\{serial\}/{print $2; exit}'
udevadm info --attribute-walk --name=/dev/ttyACM1 | awk -F'"' '/ATTRS\{serial\}/{print $2; exit}'
```

确认左右序列号非空且不同，创建 PC 端规则：

```bash
sudo nano /etc/udev/rules.d/90-alohamini-leader.rules
```

写入实际序列号：

```udev
SUBSYSTEM=="tty", ATTRS{serial}=="<左Leader序列号>", SYMLINK+="am_arm_leader_left"
SUBSYSTEM=="tty", ATTRS{serial}=="<右Leader序列号>", SYMLINK+="am_arm_leader_right"
```

**应用**并检查两个别名是否指向对应主臂：

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty
sudo udevadm settle
ls -l /dev/am_arm_leader_left /dev/am_arm_leader_right
```

<details>
<summary>树莓派：设置从臂 Follower 别名</summary>

在 `alohamini_host` 环境中用 `alohamini find-port` 分别确认两块控制板，再查询各自的序列号：

```bash
udevadm info --attribute-walk --name=/dev/ttyACM0
```

将命令中的端口换成实际设备，在 `/etc/udev/rules.d/90-alohamini-follower.rules` 写入实际序列号：

```udev
SUBSYSTEM=="tty", ATTRS{serial}=="<左从臂控制板序列号>", SYMLINK+="am_arm_follower_left"
SUBSYSTEM=="tty", ATTRS{serial}=="<右从臂控制板序列号>", SYMLINK+="am_arm_follower_right"
```

重新加载规则并重新插入控制板：

```bash
sudo udevadm control --reload-rules
```

确认 `/dev/am_arm_follower_left`、`/dev/am_arm_follower_right` 对应正确后，再启动 Host。

</details>

### 相机

在树莓派查找相机，先停止占用相机的 Host：

```bash
conda activate alohamini_host
alohamini find-cameras opencv
```

相机名称可选 `forward`、`backward`、`chest`、`wrist_left`、`wrist_right`，对应 `/dev/am_camera_<名称>`。默认开启前视和右腕；仅选择已连接、已配置设备名的相机。用 `--record-time-s 0` 只列出设备，不保存预览。

## 3. 标定

PC 标定左右主臂：

```bash
conda activate alohamini
alohamini calibrate leader --robot_model alohamini2pro --teleop.id am_leader_bi
```

树莓派标定从臂：

```bash
conda activate alohamini_host
alohamini calibrate robot --robot_model alohamini2pro
```

按终端提示将关节放到中位，再手动移动至完整活动范围。依次完成左右臂；`wrist_roll`、底盘和升降无需手动采集范围。已有标定时按 Enter 复用，输入 `c` 重做。

标定保存在 `~/Alohamini_workspace/calibration/`。可复用同一台机器的完整标定，不能套用另一台机器的数据。自定义标定目录用 `--calibration_dir`；Host 的整机 JSON 用 `--calibration` 指定。重做会保留 `*.backup.json` 备份；标定失败或恢复失败时先排查，不要直接启动运动。

## 4. 启动树莓派 Host

> 启动会使能舵机并执行升降触底回零。

```bash
conda activate alohamini_host
alohamini host --robot_model alohamini2pro
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
alohamini inspect --host <树莓派_IP> --model alohamini2pro
```

只读取一次状态，不发送运动命令。连接失败时先确认 Host、IP、两端型号；字段单位与 Python 调用见[客户端接口](host-protocol.md)。

## 6. PC 遥操

保持 Host 运行，在 PC 执行：

```bash
conda activate alohamini
alohamini teleoperate \
  --host <树莓派_IP> \
  --robot_model alohamini2pro \
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
  --robot_model alohamini2pro \
  --teleop.id am_leader_bi \
  --dataset pickup_01 \
  --task "pick up the object" \
  --num_episodes 1 \
  --episode_time 8 \
  --reset_time 3
```

默认 30 Hz 数采，记录 Host 开启的全部相机。`--display_data` 开启预览，`--profile_timing` 显示采集耗时和保存分段计时。

| 按键 | 操作 |
| --- | --- |
| 右方向键 | 提前结束当前阶段 |
| `R` / 左方向键 | 重录当前 episode |
| Esc、`q`、Ctrl+C | 结束数采 |

复位阶段可继续遥操，但不记录数据。退出时尝试保存已采到的有效帧。

数据默认保存到 `~/Alohamini_workspace/datasets/pickup_01/`。自定义位置用 `--root`；继续已有数据集用 `--resume`，型号、标定、相机、任务和帧率须一致。

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

仅恢复完整帧，不修改原数据。不要手动删除 `*.pending/`；重录前的数据保留在 `discarded/`。

AlohaMini 或 LeRobot v3 数据的索引、媒体引用异常时，修复到新目录：

```bash
alohamini dataset repair ~/Alohamini_workspace/datasets/task_demo \
  --output ~/Alohamini_workspace/datasets/task_demo_repaired
```

保留原数据，重建可确定的索引与媒体引用；缺失图像或配对不明时拒绝修复。修复结果不用于追加录制，也不消除实际漏采与时间偏差。

### 对比评估与训练示范

```bash
alohamini dataset analyze ~/Alohamini_workspace/datasets/eval_01 --episode 0
```

新录制的评估自动保存本次 checkpoint 身份、训练集与 episode 划分、实际推理参数。
分析时自动关联模型对应的数据，在 Rerun 中并排查看评估与示范画面、动作曲线。
普通浏览不加载模型；命令、Host 反馈、电流和耗时默认隐藏，可加 `--control-diagnostics` 查看。

默认参考 checkpoint 训练划分中的首条示范；这不是自动检索的最相似场景。
用 `--reference-episode 12` 选择同一训练划分中的其他示范，或用 `--reference-split val`
选择保存的验证划分。两次运行进度不同时，可指定同一任务阶段的两个时间点：

```bash
alohamini dataset analyze ~/Alohamini_workspace/datasets/eval_01 \
  --episode 0 --reference-episode 12 \
  --time-range 20 35 --reference-time-range 8 23 --align 20 8
```

`--align 20 8` 将评估第 20 秒和示范第 8 秒对齐到公共时间轴的 0 秒；原时间戳保持不变。
不指定时仅按 episode 起点并排。不同运行的场景与轨迹不同，不计算它们的逐帧误差。
只有字段和单位明确一致时才叠加曲线；无法确认的单位分开展示。

旧评估没有保存模型关联，需要加 `--checkpoint <当时使用的模型目录>`。
模型搬迁时也使用这个选项重定位，并核对已记录的身份。训练数据搬迁使用
`--training-root <原训练数据集的本地副本>`；按 checkpoint 的元数据及所选数值文件指纹核对，
不把同名目录或其他相似数据集当成原训练集。缺少数据时仍能浏览评估，但会提示示范对比不可用。
旧记录的实际 NAS、时序融合等参数未保存时显示未知，checkpoint 默认参数单独展示。

ACT／AM-ACT 可以显式加载 checkpoint，在所选示范的观测上重新预测动作：

```bash
alohamini dataset analyze ~/Alohamini_workspace/datasets/eval_01 \
  --reference-split val --probe-samples 8 --probe-device cuda \
  --viewer none --export-html
```

这会均匀抽取最多 8 个观测，按 checkpoint 的图像预处理预测独立 Chunk，
逐维计算首步及有效未来动作的 MAE，跳过 episode 尾部没有标签的部分和固定输出维度。
未来示范动作只用于计算误差，不输入模型。每个观测重置队列和潜变量，`--probe-seed` 控制随机种子。
这是重新预测的离线检查，不是录制时的原始 Chunk，也不代表闭环任务成功率。
训练划分上的误差用于检查拟合；评估泛化应优先使用验证划分。
预测、标签和有效掩码保存在 `demonstration_predictions.npz`。不加 `--probe-samples` 不运行模型。

| 选项 | 作用 |
| --- | --- |
| `--viewer none` | 不启动 Rerun；保存 `summary.json` 和 `events.csv` |
| `--export-html` | 增加浏览器可打开的 `report.html` 和 `plots/` |
| `--keyframes 21 25` | 按评估秒数导出画面及锚点对齐后的示范画面，同时启用 HTML |
| `--cameras wrist_left wrist_right` | 指定显示或导出的相机及顺序 |
| `--segment 0` / `--reference-segment 0` | 选择时钟重置后的评估／示范时间段 |
| `--save-rrd` | 保存 `recording.rrd`，用打印的 `rerun` 命令打开；包含压缩媒体 |
| `--media-budget-mb 256` | 两组画面的压缩媒体累计预算；超出部分跳过并提示 |
| `--output <新目录>` | 指定数据集外的报告目录 |

结果默认保存在 `~/Alohamini_workspace/reports/<数据集>/episode_<编号>/<时间编号>/`。
MP4 直接复用，不逐帧转码；关键帧按需读取并缓存。共享 MP4 需要整文件载入，
缩短时间范围不一定减少该视频资源大小。缓存位于工作区 `cache/evaluation-keyframes/`，
最多保留约 1 GiB。旧原生 Host JPEG 的颜色通道校正也会缓存。
已有 PC 环境包含查看依赖；独立分析环境可运行 `pip install -e '.[analysis]'`，
模型检查另需安装对应策略依赖。

需要检查旧 ACT 记录的 Chunk 交接时，可提供
`--assume-n-action-steps 100 --first-command-sequence 33 --assume-no-resets`。
这要求确认关闭时序融合、每个 tick 一个命令、全程无队列重置，并知道首次预测的真实命令序号。
结果始终标为推断的边界后首个录制帧；不要用首个录制帧的序号替代首次预测的命令序号。

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
  --host <PI_IP> --robot_model alohamini2pro --episode 0
```

支持 AlohaMini 数据集及其 LeRobot v3 导出；自定义目录用 `--root`。按 action 执行，型号、动作单位和标定须与 Host 一致。

回放评估数据时应保留完整元数据与 safety 文件。记录的提交动作不等于机器人实际执行轨迹。

默认按数据集帧率回放；`--fps` 覆盖帧率，`--speed` 调整倍率。它们不缩放底盘速度值，包含底盘运动时应保持原帧率。Ctrl+C、关节保护、反馈中断或控制权变化会终止回放，并尝试保持当前位置、停止底盘。

回放按绝对时间推进并跳过过期动作。短暂超时会重试，持续失联达到看门狗时限时停止；结束显示跳过行数和超时次数。低频或跳帧可能漏掉关键动作，无法保证底盘路径精确复现。

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

相机预览、录音等调试输出默认位于工作目录的 `logs/debug/`。麦克风扩展安装命令为 `python -m pip install -e '.[audio-debug]'`。

<details>
<summary>舵机维护：卸力、改 ID、改相位</summary>

```bash
python examples/debug/motors.py reset_motors_torque --port /dev/ttyACM0 --id 1
python examples/debug/motors.py configure_motor_id --help
python examples/debug/motors.py configure_motor_phase --help
```

卸力命令省略 `--id` 时会处理该串口上 ID 1–22 内所有已识别舵机。改 ID 只连接待配置的一颗舵机，并同步检查设备配置与标定；相位值须对应具体型号。维护报错后先确认设备状态，不要直接启动运动。

</details>

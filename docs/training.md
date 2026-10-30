# 数据整理、训练与部署

## 1. 准备

完成 [PC 安装](install.md)，在仓库根目录执行：

```bash
cd ~/Alohamini
conda activate alohamini
```

默认工作目录为 `~/Alohamini_workspace/`：`datasets/` 保存数据，`runs/` 保存模型，`logs/training/` 保存日志与 PID。

## 2. 检查与编辑数据

```bash
alohamini dataset edit --dataset task_demo \
  --operation.type info --operation.show_features true

alohamini dataset check ~/Alohamini_workspace/datasets/task_demo \
  --decode-images --decode-videos

alohamini dataset edit --dataset task_demo \
  --output ~/Alohamini_workspace/datasets/task_demo_ready \
  --operation.type delete_episodes --operation.episode_indices '[3, 8]'
```

episode 从 0 编号。`--dataset` 指工作区中的名称，也可换成 `--root /完整路径`。
输出目录须不存在；保留回合按原顺序重新编号，图像、state/action 和时间记录保持配对。

其他编辑操作共用 `--root`、`--output`：

| `--operation.type` | 参数示例 |
| --- | --- |
| `split` | `--operation.splits '{"train": 0.8, "val": 0.2}'` 或 `'{"train": [0, 2], "val": [1]}'` |
| `merge` | `--operation.roots '["/数据集A", "/数据集B"]'`，无需 `--root` |
| `remove_feature` | `--operation.feature_names '["observation.images.wrist_right"]'` |
| `modify_tasks` | `--operation.new_task "拿起积木"`；可加 `--operation.episode_tasks '{"2": "放下积木"}'` |
| `recompute_stats` | 默认统计数值；`--operation.skip_image_video false` 同时统计图像 |
| `convert_image_to_video` | JPEG/PNG 转 MP4；`--operation.episode_indices '[0, 2]'` 可选回合 |
| `reencode_videos` | MP4 重编码，如 `--operation.rgb_encoder.crf 23` |
| `info` | `--operation.show_features true`；无需 `--output` |

拆分结果位于 `<output>/train` 等子目录。合并要求 FPS、字段、相机、标定和动作坐标一致。
训练与验证按完整回合隔离，同一示教的不同处理副本不得分置两边。

视频默认 H.264、CRF 18、GOP 2、`yuv420p`，每回合每相机一个 MP4。
可用 `--operation.rgb_encoder.vcodec`、`.crf`、`.g`、`.preset`、`.pix_fmt` 调整。
转换保持帧数和时间；`yuv420p` 要求偶数图像尺寸。

`recompute_stats` 排除无效反馈，统计诊断保存在 `meta/stats_info.json`。
`--operation.relative_action true --operation.chunk_size 50` 统计动作块的关节增量分布，不改写 action。

### 修复

```bash
alohamini dataset repair /path/to/dataset --output /path/to/dataset_repaired
alohamini dataset check /path/to/dataset_repaired --decode-images --decode-videos
```

支持 AlohaMini 和 LeRobot 数据的可确定索引、媒体引用问题；缺失图像或配对不明时拒绝修复。
`*.pending-*` 是未完成输出。修复不会消除实际漏采或时间偏差。

## 3. 选择数据与输入

数据集采用 LeRobot v3 格式：

| 目录 | 内容 |
| --- | --- |
| `data/` | Parquet：state、action、索引、舵机反馈及有效掩码 |
| `videos/` | 各相机的 MP4；读取器按 episode 元数据和时间索引取帧 |
| `meta/` | 字段定义、任务、统计、episode 索引，以及标定和实际采样时间等平台扩展信息 |

缺少平台标定和安全记录的外部 v3 数据，训练时加 `--dataset.allow_unannotated=true`。
有 state 时可用 `--state=recorded` 读取原向量，保留字段顺序和数值；纯视觉任务使用 `--state=none`。

### 导出与纯视觉副本

需要另存或移除数值观测时：

```bash
alohamini dataset export ~/Alohamini_workspace/datasets/task_demo_ready \
  --output ~/Alohamini_workspace/datasets/task_demo_vision \
  --format lerobot-v3 --vision-only
```

以 `alohamini2pro` 为例，默认 state 为双臂位置 14 维、底盘速度 3 维、升降高度 1 维。
纯视觉副本保留视频和 action，去掉数值观测输入；视频直接复制，不改变配对、帧数或时间戳。早期导出的图片内嵌 Parquet 数据仍可读取，但不是当前采集格式。

### 输入参数

| 参数 | 含义 |
| --- | --- |
| 省略 `--state` | 按算法预设选择；自动模式在外部 v3 中读取原 state，无 state 时使用纯视觉 |
| `--state=none` | 纯视觉，无需另存数据副本 |
| `--state=recorded` | 外部 v3 的原始 state 向量，不做单位换算 |
| `--state=joint_position,base_velocity,lift_height` | 18 维位置、底盘速度与高度 |
| `--state=joint_velocity,joint_current` | 双臂 28 维速度/电流，含夹爪 |
| `--cameras='["forward","wrist_right"]'` | 使用指定相机；省略时使用全部记录相机 |

平台字段按参数顺序组装。关节位置使用标定坐标，关节速度为对应坐标单位/秒，电流为 A；
底盘速度为 m/s、deg/s，升降高度为 mm。action 为双臂绝对位置目标、底盘速度及升降高度目标。

### 关节、TCP 与相对目标

`observation.state` 描述当前观测，`action` 是对应记录中的控制目标。两者即使名称和维度相同，
也不能互相替代。Host 的关节位置可以是归一化舵机坐标，不能直接作为 URDF／DH 的关节角。

`--state` 选择送入模型的源字段；以下参数在同一数据处理层选择这些字段和 action 的表示，
转换后才计算训练统计和归一化，不改写原数据集。目前 ACT、Spirit 接入此接口：

| 参数 | 选项与用途 |
| --- | --- |
| `--data.state_representation` | `recorded` 保留所选字段；`tcp` 将完整臂关节位置组替换为 TCP 位姿，保留其他所选字段 |
| `--data.action_representation` | `recorded` 原目标；`tcp_absolute` 绝对 TCP 目标；`tcp_delta` 相对当前观测的 TCP 目标 |
| `--data.kinematics.reference_pose` | 按型号登记的标定姿态解析几何与编码器坐标，如 `upright_forward` |
| `--data.kinematics.spec` | 自定义字段、坐标和几何的 JSON 文件 |
| `--data.kinematics.path` | 已解析、包含完整几何快照的转换描述 JSON |

三种 `kinematics` 来源选一。ACT 默认保留记录表示；Spirit 新训练默认使用 TCP state 和 TCP 增量动作，
仍须显式提供一种几何来源。原始字段、模型字段、转换配置和归一化统计随 checkpoint 保存，续训无需原配置文件。

state 与 action 表示可以独立选择。例如保留关节 state、只转换动作时，使用
`--data.state_representation=recorded --data.action_representation=tcp_delta`。
ACT 使用 `--state=none` 时仍可预测 TCP 动作：数据须保留原关节观测作为转换及 IK 的锚点，
这些关节值不送入模型；已经移除数值观测的纯视觉副本无法提供此锚点。
TCP 动作执行按窗口解码，不支持 ACT 时间融合；其他策略须完成对应适配后才能启用此接口。

关节到 TCP 的转换依次使用以下参数：

| 转换 | 所需参数 |
| --- | --- |
| Host 记录值 → 归零后编码器位置 | 数据集中的归一化方式、范围和方向 |
| 归零后编码器位置 → 模型关节角（rad） | 对应设备的参考编码器值、参考关节角、正方向和传动比 |
| 模型关节角 → TCP 位姿 | 机器人型号的 DH 几何、基坐标与工具变换 |

`q_URDF` 表示 URDF 模型定义的关节角，零值对应模型的 CAD 零姿态，单位为 rad。
采集关节角也可使用其他明确的零姿态；切换零姿态时，将对应模型角加到 DH 的 `theta_offset`，
即可保持同一套物理 FK，不要求把模型 CAD 零位当成电机标定零位。

**标准中位标定数据**：`alohamini2pro` 的 `upright_forward` 预设采用大臂向上平行 z 轴、
小臂及夹爪沿机器人 x 轴前伸、夹爪开合平面水平，肩部转向与腕部偏转处于中位。
`upright` 指大臂竖直，`forward` 指小臂及夹爪前伸；这是标定坐标预设的名称。
仅在采集时按此姿态执行半圈归零的情况下，训练时选择：

```bash
pip install -e '.[kinematics]'
alohamini train \
  --policy.type=act \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --state=joint_position \
  --data.state_representation=tcp \
  --data.action_representation=tcp_delta \
  --data.kinematics.reference_pose=upright_forward \
  --output_dir="$HOME/Alohamini_workspace/runs/act_tcp_01" --background
```

此预设将上述姿态定义为六个关节角均为零，将与 CAD 零姿态的差值吸收到 DH 角偏移。
该姿态对应原模型关节角 `[0°, -90°, 90°, 0°, 0°, 0°]`；方向按连杆主轴确定，
不使用偏置安装的舵机轴心连线代替连杆方向。
STS 半圈归零后的参考 tick 为 `2047`；当前数据的归一化、方向、范围及归零参数从记录元数据读取，
不读取旧设备的关节映射文件。IK 关节范围由该记录的编码器范围换算，预设及解析后的参数随 checkpoint 保存。
选择此预设是声明采集所用的标定协议，不会移动机器人或重新归零，也不从文件名或向量维度推断姿态。

已经使用模型关节坐标的弧度或角度数据可直接进行 FK。编码器数据还需定义它与模型关节坐标的关系；
电机范围和 `homing_offset` 本身不能唯一确定模型中的关节零位。计算不需要启动 ROS 或读取在线 TF。
编码器值和参考编码器值均使用归零后坐标，不重复加减 `homing_offset`。
每关节的模型角按 `q = q_ref + sign × wrap(tick - tick_ref) × 2π/N × ratio` 计算，
其中 `N` 为一圈编码器计数，`wrap` 取单圈有符号差；限位用于选择明确的角度分支。

需要单独检查或复用解析结果时，可选导出转换描述：

```bash
mkdir -p "$HOME/Alohamini_workspace/configs"
python -m alohamini.datasets.kinematic_fields \
  --dataset "$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --reference-pose upright_forward \
  --output "$HOME/Alohamini_workspace/configs/task_demo_kinematics.json"
```

此命令只将当前数据的字段、编码器坐标和型号几何解析为 JSON，不生成 TCP 数据列，也不训练。
训练可通过 `--data.kinematics.path` 复用该文件；直接选择预设时无需先执行导出。

自定义坐标使用 `--data.kinematics.spec=/完整路径/坐标配置.json`；导出命令也支持用 `--spec` 替代
`--reference-pose`。JSON 不要求固定型号、左右臂名称或 ROS 标定文件，配置包括：

| 字段 | 配置要求 |
| --- | --- |
| `version` | `1` |
| `source_names` | 原 action 顺序；省略时读取数据集；原关节观测须包含这些命名坐标，所选模型 state 可以只取其中一部分 |
| `arms` | 一个或多个命名运动链；关节数由几何确定，夹爪可省略 |
| `arms[].model` | `{"id":"alohamini2pro","chain":"left"}`；自定义型号用 `directory` 替代 `id`；可用 `joint_zero` 声明新零姿态对应的模型关节角 |
| `arms[].geometry` | 与 `model` 二选一；直接提供标准 DH 的 `rows`、`base_transform`、`tool_transform` 和 `joint_limits`，同时声明 `frame` |
| `arms[].coordinates` | 按模型关节顺序列出 `field` 与 `unit`；支持模型 `rad`／`deg`、归零后 `encoder` tick、Host `host` 坐标 |
| `arms[].gripper` | 声明源字段、源开闭端点、输出开闭端点及单位；不从数值范围推断实际开口 |
| `passthrough`／`passthrough_units` | 按顺序透传的字段及单位；与运动链合计覆盖每个源字段一次 |
| `source_calibration` | Host 数据所用电机的采集期参数，按电机名保存归一化、范围、方向、`homing_offset`，以及可用的 ID／型号 |

`encoder`／`host` 坐标另需 `encoder` 映射：`ticks_per_revolution`、`reference_tick`、
`reference_position_rad`、`direction`、`joint_per_encoder_ratio` 及弧度限位
`position_min_rad`／`position_max_rad`。`host` 还需 `host_units` 中的
`normalization`、`range_min`、`range_max`、`drive_mode`。这些参数必须对应采集期的坐标。

关节已经是模型弧度时，可复制[单臂配置示例](../examples/datasets/kinematics-radians.json)，
按数据修改字段名、运动链和夹爪端点后传给 `--data.kinematics.spec`。Host 的归一化值或中心角度不能直接标成模型 `rad`／`deg`。
型号的 `model.json` 通过 `kinematic_chains` 登记几何文件；当前内置 `alohamini2pro` 左右臂，
未登记的型号不会回退到其他机器人的几何。标准 DH 后端支持转动关节链，不支持移动关节。

已有左右臂 Host YAML 时，可使用导入入口：

```bash
python -m alohamini.datasets.kinematic_fields \
  --dataset "$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --left-map "$HOME/Alohamini_workspace/calibration/hardware_joint_map_left.yaml" \
  --right-map "$HOME/Alohamini_workspace/calibration/hardware_joint_map_right.yaml" \
  --output "$HOME/Alohamini_workspace/configs/task_demo_kinematics.json"
```

此导入器处理双六关节臂格式，映射中每个关节须包含采集期的 `motor_calibration`：
`homing_offset`、`range_min`、`range_max`、`drive_mode`。缺失或与数据不符时拒绝生成，不能用当前配置补填旧标定来源。
生成的描述保存字段、单位、编码器映射、DH／TCP 几何、参考坐标、限位和夹爪映射；输出文件须尚不存在。
训练统计及 checkpoint 加载均校验来源坐标，后续计算直接读取快照，不再访问原型号文件和标定路径。

训练读取样本后调用共享转换，保留原始记录、时间配对和动作请求语义，不将动作请求替换为未来实测反馈。
Spirit 的训练资源与命令见 [Spirit v1.5](#spirit-v15)。

每条运动链生成 `[x, y, z, rx, ry, rz]`，配置夹爪时再追加 `gripper`：位置为 m，旋转向量为 rad。
Host YAML 导入器将夹爪映射到 `canonical_0_0.1`，该数值不表示实际开口米数。
`alohamini2pro` 两臂分别使用 `left_Base`、`right_Base`，不隐含世界坐标或底盘里程计。

相对目标以当前观测为同一个锚点，对每个未来目标分别计算：

```text
T_anchor    = FK(current_model_joints)
T_target[k] = FK(recorded_target_joints[k])
delta_p[k]  = p_target[k] - p_anchor
delta_r[k]  = rotvec(R_target[k] @ R_anchor.T)
```

这些值是各臂基坐标系下相对当前观测的目标，不是相邻两帧的差，也不逐步累积。
旋转通过矩阵组合计算，不能直接相减旋转向量。
夹爪保持绝对开合目标并按配置换算单位；底盘速度和升降高度保留记录值及声明单位。
单纯计算关节坐标差不需要 FK；计算 TCP 差需要正确的模型关节角和几何。

推理时还原 `p_target = p_anchor + delta_p`、`R_target = exp(delta_r) @ R_anchor`，
再以当前观测关节为初值执行 IK，后续目标沿用上一步解。
本地逐步评估先解码整个执行窗口，再逐步发送，沿用同一锚点和连续 IK 解；新窗口重新取观测。
不可达、超出约束或未收敛的目标会报错。转换描述与处理顺序随 checkpoint 保存。
部署时继续使用匹配的编码器坐标；普通 `normalized` 关节比例迁移不能替代 TCP 映射，来源坐标变化会拒绝绑定。

## 4. 训练

以下命令留出 episode 1 做验证，其余回合训练。数据只有一个回合时，省略 `--dataset.eval_episodes` 和 `--eval_steps`。

### 统计准备（可选）

查看 AlohaMini 数据集全部数值字段的统计，不改写数据：

```bash
alohamini dataset stats ~/Alohamini_workspace/datasets/task_demo_ready \
  --output ~/Alohamini_workspace/logs/task_demo_stats.json
```

按训练 JSON 中的策略、输入字段、时间窗口和训练回合生成归一化统计（支持 AlohaMini 和 LeRobot v3）：

```bash
alohamini dataset stats ~/Alohamini_workspace/datasets/task_demo_ready \
  --config /path/to/train.json \
  --output ~/Alohamini_workspace/logs/task_demo_training_stats.json

python -m alohamini.learning.train --config /path/to/train.json \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --stats "$HOME/Alohamini_workspace/logs/task_demo_training_stats.json" --background
```

`--stats` 只接受带 `--config` 生成的统计；数据或样本配置不匹配时须重新生成。
省略 `--stats` 时训练自动计算。新训练均保存 `runs/<名称>/statistics.json`，checkpoint 保留实际使用的统计；恢复训练沿用 checkpoint。

### ACT

```bash
python -m alohamini.learning.train \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' \
  --policy.type=act --policy.device=cuda --policy.chunk_size=100 \
  --output_dir="$HOME/Alohamini_workspace/runs/act_01" \
  --steps=100000 --batch_size=2 --save_freq=10000 --log_freq=100 \
  --eval_steps=1000 --background
```

`--state=none` 使用纯视觉；`--image_size='[480,640]'` 设置高、宽。
ResNet18 默认加载 ImageNet 权重，首次下载缓存到工作区 `pretrained/`。
离线使用须提前准备；`--policy.pretrained_backbone_weights=null` 改为随机初始化。

ACT 和 AM-ACT 新训练默认启用相机身份编码，将前视、左腕、右腕等图像与其来源对应。
训练和推理使用同一组编码，相机名称及顺序随 checkpoint 保存；采集和部署时须保持相机命名一致。
`--policy.use_camera_embeddings=false` 可关闭，保留原 ACT 的视觉融合方式。
旧 checkpoint 的推理、续训及默认微调保留原配置；为旧模型补上编码时，在
`--policy.path=<旧 checkpoint>` 微调命令中增加 `--policy.use_camera_embeddings=true`，
使用新的输出目录，不使用 `--resume=true`。新增编码从零开始学习，需要微调后才具有区分能力。

### AM-ACT

```bash
python -m alohamini.learning.train \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_vision" \
  --dataset.eval_episodes='[1]' \
  --policy.type=am_act --policy.device=cuda --policy.chunk_size=100 \
  --output_dir="$HOME/Alohamini_workspace/runs/am_act_vision_01" \
  --steps=100000 --batch_size=2 --save_freq=10000 --log_freq=100 \
  --eval_steps=1000 --background
```

也支持含 state 的数据。速度/电流实验改用保留反馈的数据目录，并加 `--state=joint_velocity,joint_current`。
新 AM-ACT 训练默认把 `action.names` 中的 `x.vel`、`y.vel`、`theta.vel` 分别设为三分类：

| 字段 | 类别顺序：负向、停止、正向 | 单位 |
| --- | --- | --- |
| `x.vel` | `[-0.15, 0, 0.15]` | m/s |
| `y.vel` | `[-0.15, 0, 0.15]` | m/s |
| `theta.vel` | `[-45, 0, 45]` | deg/s |

可通过 `--policy.discrete_action_weighting=none|inverse_frequency|sqrt_inverse_frequency`
选择权重策略，`--policy.discrete_action_max_weight_ratio` 调整上限。

常用选项：

| 参数 | 用途 |
| --- | --- |
| `--policy.base_classification=false` | 新训练关闭默认底盘分类，改为连续回归 |
| `--policy.discrete_action_mask_absent_classes=true` | 屏蔽训练集中完全没有出现的类别；默认关闭 |
| `--policy.discrete_action_values='[[-0.15,0,0.15]]'` | 自定义分类值，须配合 `discrete_action_dims` 设置 |

自定义维度须按数据的 `action.names` 核对。屏蔽缺类后仍保留三输出头，
两类有数据时等效二分类；把缺类权重设为零不会禁止模型输出该类别。
补录新方向后，用 `--policy.path=<旧 checkpoint>` 微调，重新统计并启用新增类别。
数据变化不能使用 `--resume=true`。启用 `action_loss_groups` 时须覆盖全部需要回归的连续维度。

#### 条件 CVAE

默认 AM-ACT 使用动作及可选 state 后验、标准高斯先验，推理时 `z=0`。
启用图像条件后验和先验可使用通用示例：

```bash
python -m alohamini.learning.train \
  --config=examples/learning/am_act_cvae.json \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_vision" \
  --dataset.eval_episodes='[1]' --eval_steps=1000 --background
```

该示例是机器人适配配置：ResNet-18、30 步 Chunk、执行 10 步、纯视觉、
4 Query Attention Pooling、Image+Action 后验、图像条件高斯先验、KL 权重 1、
5000 个优化步 warmup。动作主干 decoder 为 7 层，主干 encoder 和后验 encoder 各 4 层。
图像使用 RGB/ImageNet 归一化，动作使用训练统计和原始目标单位，不使用语言。

| 配置项 | 可选值 |
| --- | --- |
| `posterior_condition` | `image_action` / `action` |
| `prior_type` | `conditional_gaussian` / `standard_normal` |
| `latent_visual_pooling` | `multi_query_attention` / `gap` |
| `latent_inference_mode` | `sample` / `mean` / `zero` |
| `latent_refresh_mode` | `phase` / `chunk` |

示例按阶段缓存 z：首次根据图像采样，后续预测使用新图像并复用 z。
任务执行器调用 `policy.refresh_latent()` 切换阶段；新 episode 的 `reset()` 也会清空缓存。
模型不会自动识别阶段。刷新 z 同时清空动作队列和时序融合，避免混合不同潜变量的预测。

### Diffusion Policy

```bash
python -m alohamini.learning.train \
  --policy.type=diffusion --policy.device=cuda \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' \
  --cameras='["forward","wrist_right"]' \
  --output_dir="$HOME/Alohamini_workspace/runs/diffusion_01" \
  --batch_size=2 --save_freq=10000 --log_freq=100 --background
```

默认 2 帧观测、64 步预测、执行 32 步，DDPM 100 步；视觉配置参考
[LeRobot PR #3202](https://github.com/huggingface/lerobot/pull/3202)。
RGB 使用 ImageNet `MEAN_STD`，state/action 使用训练集 `MIN_MAX`。
`--policy.n_action_steps` 调整执行步数，`--policy.num_inference_steps` 调整去噪步数。

尾部排除数默认按 `horizon - n_action_steps - n_obs_steps + 1` 计算，为 **31**；
`--policy.drop_n_last_frames` 可显式覆盖，设为 `0` 保留所有采样起点。
EMA 默认关闭，使用 `--ema.enable=true` 开启。

### CDP

```bash
python -m alohamini.learning.train \
  --policy.type=cdp --policy.device=cuda \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --batch_size=2 \
  --output_dir="$HOME/Alohamini_workspace/runs/cdp_01" --background
```

采用 [CDP2 RGB 配置](https://github.com/GaavaMa/Causal-Diffusion-Policy/tree/58b23af1e9ac246a11001d7d5388a1de4f53daa5)：
32 帧历史、总窗口 40 步、每次执行 8 步；DDPM 100 步，预测干净动作。
在线保留动作缓冲区和 KV 缓存，episode 开始须调用 `reset()`。
历史动作只取当前帧之前的数据；图像和 state 历史包含当前帧。
RGB 缩放至 84×84、裁剪 76×76，ImageNet `MEAN_STD`；state/action 使用训练集 `MIN_MAX`。
ResNet18 使用 GroupNorm，无预训练权重；训练器 EMA 默认开启，尾部排除数为 0。
默认训练 100000 步、预热 1000 步，可通过 `--steps` 和 `--scheduler.warmup_steps` 调整。
仅支持 RGB CDP2；不含点云 CDP3。动作块大小属于模型结构，加载 checkpoint 时不可更改；
去噪步数必须等于训练扩散步数。离线评估使用示范历史动作，结果不代表闭环成功率。

### dit-policy（DiT-Block）

```bash
python -m alohamini.learning.train \
  --policy.type=dit_policy --policy.device=cuda \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --batch_size=2 \
  --output_dir="$HOME/Alohamini_workspace/runs/dit_policy_01" --background
```

采用 [SudeepDasari/dit-policy](https://github.com/SudeepDasari/dit-policy/tree/ba22ee3ff455f0423fc9cfa5658a1a1cdddc6e73)
的独立 DiT 编码器／解码器：默认 1 帧观测、100 步动作、6 层、隐藏维度 512；
训练扩散 100 步，DDIM 推理 8 步。`--policy.n_action_steps` 可缩短执行窗口。
RGB 缩放至 256×256，训练随机缩放裁剪至 224×224 并模糊，推理缩放至 224×224；
采用 ImageNet `MEAN_STD`、ResNet18 GroupNorm 和空间 token。
`--policy.pooled_vision=true` 改用每帧单 token。
state 保留原值；本地 action 使用训练集 `MIN_MAX`，上游数据需预先归一化。
默认使用 torchvision ImageNet V1 权重初始化，不含作者另行分发的 GN 权重。
默认训练 150000 步、预热 2000 步；EMA 关闭，尾部保留并屏蔽填充动作。不使用语言输入。

### Multi-Task DiT

```bash
conda activate alohamini
python -m alohamini.learning.train \
  --policy.type=multi_task_dit --policy.device=cuda \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --eval_steps=1000 \
  --cameras='["forward","wrist_right"]' \
  --output_dir="$HOME/Alohamini_workspace/runs/multi_task_dit_01" \
  --batch_size=2 --gradient_accumulation_steps=8 --steps=100000 \
  --ema.enable=true --save_freq=10000 --log_freq=100 --background
```

| 项目 | 默认配置 |
| --- | --- |
| 观测／预测／执行 | 2 帧／32 步／24 步；排除最后 7 个采样起点 |
| DiT | 隐藏维度 512、6 层、8 heads、AdaLN-Zero、RoPE |
| 视觉 | CLIP ViT-B/16，共享相机编码器；缩放 256×256，训练随机／推理中心裁剪至 224×224 |
| 语言 | 必需的非空 `task`，最长 77 tokens；文本编码器冻结，投影层训练 |
| 归一化 | RGB：CLIP `MEAN_STD`；state/action：训练集 `MIN_MAX`，保留原字段与单位 |
| 优化 | Adam `2e-5`，视觉学习率 ×0.1；cosine，无 warmup |
| 生成目标 | diffusion，DDPM 100 步 |

state 必须存在。CLIP 权重与 tokenizer 默认读取工作区 `pretrained/clip-vit-base-patch16/`
（默认 `~/Alohamini_workspace/`，支持 `ALOHAMINI_WORKSPACE`），须提前下载 `openai/clip-vit-base-patch16`。
`--policy.vision_encoder_name`、`--policy.text_encoder_name` 可覆盖为其他本地目录或 Hugging Face 模型 ID。
尾部排除数按同一时序公式计算；`--policy.do_mask_loss_for_padding=true` 可屏蔽补齐动作的损失。

```bash
# DDIM
--policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10

# Flow matching；integration_method 也支持 rk4
--policy.objective=flow_matching --policy.num_integration_steps=10 \
  --policy.integration_method=euler --policy.timestep_sampling_strategy=beta
```

支持单设备、DDP、FSDP2 和训练器 EMA。
checkpoint 打包权重、CLIP 配置和 tokenizer；EMA 部署使用 `checkpoint-ema`，续训使用在线 checkpoint。

### FastWAM

准备本地基座、Wan VAE／UMT5 和 tokenizer；下载只需执行一次：

```bash
hf download lerobot/fastwam_base \
  --revision 0a868ec1dcf6ff00bcdfa9b7196d6e211ed7e616 \
  --local-dir "$HOME/Alohamini_workspace/pretrained/fastwam_base"
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --include 'vae/*' 'text_encoder/*' \
  --local-dir "$HOME/Alohamini_workspace/pretrained/wan22_assets"
hf download google/umt5-xxl --include '*.json' '*.model' \
  --local-dir "$HOME/Alohamini_workspace/pretrained/umt5_tokenizer"

python -m alohamini.learning.train \
  --policy.type=fastwam --policy.device=cuda \
  --policy.path="$HOME/Alohamini_workspace/pretrained/fastwam_base" \
  --policy.vae_model_id="$HOME/Alohamini_workspace/pretrained/wan22_assets" \
  --policy.text_encoder_model_id="$HOME/Alohamini_workspace/pretrained/wan22_assets" \
  --policy.tokenizer_model_id="$HOME/Alohamini_workspace/pretrained/umt5_tokenizer" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --cameras='["forward","wrist_right"]' \
  --output_dir="$HOME/Alohamini_workspace/runs/fastwam_01" \
  --batch_size=1 --policy.use_gradient_checkpointing=true \
  --save_freq=1000 --log_freq=100 --background
```

训练使用任务文本、当前 state、32 步动作和间隔 4 行的 9 帧视频；两路图像按名称排序，
各缩放至 224×224 后横向拼接。推理只用当前图像，不需要未来视频。
state/action 使用训练集 min/max，动作仍是记录的关节目标、底盘速度和升降高度，
不套用 LIBERO 的末端／夹爪变换。缺少未来帧时使用边界 padding 及对应损失掩码。

此入口读取转换后的 FastWAM `model.safetensors`，不直接读取作者的 `.pt`；
采用该发布基座的 flow shift=5。基座中的机器人输入／输出层缺失或维度不匹配时重新初始化。
checkpoint 不重复打包冻结的 Wan／UMT5 文件，迁移机器时须保留配置中的资产路径；
微调时也可用上述三个资源参数指定新路径。从平台 checkpoint 微调会恢复全部机器人输入／输出层。
完整基座需要较大显存，示例 batch 不保证适用于所有 GPU。

### AC-DiT

策略名为 `acdit`，支持底盘预训练和全身策略训练。只有 RGB 时显式设置
`--policy.use_pointcloud=false`。论文／开源脚本预设、输入坐标、两阶段命令及资源配置见
[AC-DiT](acdit.md)。

### SmolVLA

准备基座及视觉语言模型的配置、tokenizer 文件：

```bash
hf download lerobot/smolvla_base --local-dir "$HOME/Alohamini_workspace/pretrained/smolvla_base"
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --include '*.json' '*.jinja' '*.model' '*.txt' \
  --local-dir "$HOME/Alohamini_workspace/pretrained/smolvlm_assets"

python -m alohamini.learning.train \
  --policy.type=smolvla --policy.device=cuda \
  --policy.path="$HOME/Alohamini_workspace/pretrained/smolvla_base" \
  --policy.vlm_model_name="$HOME/Alohamini_workspace/pretrained/smolvlm_assets" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' \
  --output_dir="$HOME/Alohamini_workspace/runs/smolvla_01" \
  --batch_size=2 --policy.compile_model=false \
  --save_freq=10000 --log_freq=100 --background
```

自动预设 `smolvla-realworld-alohamini-v1`：位置 state、任务文本、512×512 等比例补边、
50 步动作块、10 步流匹配采样、冻结 VLM、BF16、AdamW 与 warmup/cosine。
训练步数默认 200000；来源及覆盖参数保存在 `paper_preset` 和训练配置中。
checkpoint 自带 tokenizer 和骨干配置。不支持纯视觉、PEFT 或 ACT 时间融合。

### π0.5

准备转换后的 OpenPI PyTorch `model.safetensors` 和 PaliGemma SentencePiece tokenizer。
不支持直接读取 Orbax/JAX checkpoint。

```bash
python -m alohamini.learning.train \
  --policy.type=pi05 --policy.device=cuda \
  --policy.path=/path/to/pi05_pytorch \
  --policy.tokenizer_path=/path/to/paligemma_tokenizer.model \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' \
  --cameras='["forward","wrist_right"]' \
  --output_dir="$HOME/Alohamini_workspace/runs/pi05_01" \
  --batch_size=1 --save_freq=1000 --background
```

自动预设 `openpi-pi05-alohamini-v1`：位置 state、任务文本、224×224 等比例补边、
50 步动作块、q01/q99 归一化、AdamW 与 warmup/cosine。
相机自动采用数据集声明的视角，可用 `--cameras` 选择子集；至少选择一路，名称沿用数据集。
相机顺序自动保存到 checkpoint；推理须提供训练时选定的全部视角。
state 保留完整 18 维原始位置、底盘速度和升降高度，包括数据集中声明的固定维度。
双臂各 6 个关节预测相对当前 state 的增量，执行前还原为绝对目标；夹爪、底盘速度、升降高度保持原义。
网络参数用 `--policy.network='{"action_horizon":30}'` 覆盖；checkpoint 自带 tokenizer。

先选择训练方案，再选择参数更新方式：

- `--policy.training_recipe=flow`（默认）：OpenPI 公开的流匹配微调，保留动作损失到 VLM 的梯度。
- `--policy.training_recipe=ki`：FAST 交叉熵与流匹配联合训练；逐层隔离动作专家到 VLM 的 K/V 梯度，两种动作分支互不可见。推理仍使用连续动作专家。

| `--policy.finetuning_mode` | 视觉编码器及投影 | LLM | 动作专家 Transformer |
| --- | --- | --- | --- |
| `a` | 冻结；`--policy.a_vision_mode=full` 可全量训练 | LoRA | LoRA |
| `b` | 冻结 | 冻结 | 全量 |
| `c` | 全量 | LoRA | 全量 |
| `d` | LoRA | LoRA | 全量 |
| `full`（默认） | 全量 | 全量 | 全量 |

四份配置见 [`examples/learning/pi05/`](../examples/learning/pi05/)。`ki + b` 不支持：冻结整个 VLM 不属于双目标 KI。
LoRA 作用于注意力和 MLP 投影；D 还包含视觉投影。动作输入/输出及时间 MLP 全量训练。
LLM/视觉默认 rank=16、alpha=16；动作专家 rank=32、alpha=32，分别用 `lora_rank/lora_alpha`、`expert_lora_rank/expert_lora_alpha` 覆盖。
这些微调组合是平台配置，不是官方发布的四组实验。

下载小型 tokenizer 资源（不包含完整基座）：

```bash
python -m alohamini.policies.pi05.assets
python -m alohamini.learning.train \
  --config=examples/learning/pi05/a.json \
  --policy.training_recipe=ki --policy.a_vision_mode=frozen \
  --policy.path=/path/to/pi05_pytorch \
  --dataset.root=/path/to/dataset --dataset.eval_episodes='[1]' \
  --run_name=pi05_ki_a --background
```

资源默认位于工作区 `pretrained/pi05_tokenizers/`；KI 的 FAST 默认读取其中的 `fast/`，可用 `--policy.fast_tokenizer_path` 覆盖。
FAST 编码补零前的 18 维归一化动作，沿用 π0.5 的 task/state/Action 提示词；目标超出 `--policy.fast_max_tokens=256` 时须增大上限，不截断动作。
KI checkpoint 包含完整网络、LoRA、两种 tokenizer 和训练方案；推理不需要原始基座目录。续训不允许更换方案或微调方式。

默认采用公开 OpenPI 的 BF16 主干并保留部分 FP32 模块（平台 `mixed_precision=none`）、完整梯度检查点、30000 步、AdamW `2.5e-5`、betas `(0.9,0.95)`、weight decay `1e-10`、梯度裁剪 `1`、1000 步预热及余弦衰减至 `2.5e-6`；每 100 步日志、1000 步保存。EMA 默认关闭。
示例使用 batch=1、累积 32 次，单卡有效 batch=32。可关闭 `--policy.gradient_checkpointing=false`；`network.pytorch_compile_mode` 默认 `max-autotune`，只编译采样，首次推理较慢，设为 `null` 可关闭。

KI 可通过 `--policy.vlm_data_path=/path/to/train.jsonl` 混入图文或规划数据，每个机器人样本配一个独立抽样的 VLM 样本：

```json
{"images":{"forward":"images/001.jpg"},"prompt":"下一步做什么？","answer":"抓起红色方块。"}
```

图像路径相对 JSONL，相机集合从清单自动读取，缺少的视角用 mask 屏蔽；只对 answer 计算语言损失，不计算动作损失。
默认三项损失权重均为 1；`flow_loss_weight/fast_loss_weight/vlm_loss_weight` 可覆盖。token 损失先按每条样本有效 token 平均，再按样本平均。
联合数据仅用于训练，保存清单 SHA256 并在续训时核验；无该数据时运行机器人数据 KI。官方通用预训练数据混合未复现。
方法见 [KI 论文 §5](https://www.pi.website/download/pi05_KI.pdf)；训练默认值见 [OpenPI](https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac)。

SmolVLA／π0.5 的完整基座须按可用显存调整 batch 和精度。

### HyVLA-0.5 与 OpenWAM-α

策略名为 `hyvla05`、`openwam`。本地基座、历史图像／未来视频窗口与动作表示见
[HyVLA-0.5](hyvla05.md) 和 [OpenWAM-α](openwam.md)。两者共用本章的训练、验证、多卡和 checkpoint 配置。

### Spirit v1.5

使用本地 Spirit 基座及 Qwen 配置、tokenizer 和图像处理资源：

```bash
pip install -e '.[spirit]'
hf download Spirit-AI-robotics/Spirit-v1.5 \
  --revision 9886fa7552de70b097c4214b7e40e66ff937a6fd \
  --local-dir "$HOME/Alohamini_workspace/pretrained/spirit-v1.5"
python -m alohamini.policies.spirit.assets \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --output "$HOME/Alohamini_workspace/pretrained/spirit-qwen-assets"
```

Qwen 权重已包含在 Spirit 基座中，此资源命令只下载配置和处理器文件，记录版本及文件摘要；
输出目录须为空。训练和 checkpoint 加载只读取本地资源。

按[关节、TCP 与相对目标](#关节tcp-与相对目标)选择数据表示。以下命令适用于标准中位标定数据：

```bash
alohamini train \
  --policy.type=spirit \
  --policy.path="$HOME/Alohamini_workspace/pretrained/spirit-v1.5" \
  --policy.backbone_path="$HOME/Alohamini_workspace/pretrained/spirit-qwen-assets" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --state=joint_position,base_velocity,lift_height \
  --data.state_representation=tcp \
  --data.action_representation=tcp_delta \
  --data.kinematics.reference_pose=upright_forward \
  --dataset.eval_episodes='[1]' \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --distributed_backend=fsdp2 --num_processes=4 \
  --mixed_precision=bfloat16 --batch_size=1 --gradient_accumulation_steps=8 \
  --steps=40000 --save_freq=5000 --log_freq=100 \
  --output_dir="$HOME/Alohamini_workspace/runs/spirit_01" --background
```

`num_processes`、batch 和累积次数按可用显卡与显存调整。
默认采用 16 层、1536 宽、32 头 DiT，交替自注意力与跨注意力；生成 60 步动作，推理执行 10 步 Euler 采样。
训练采用 Flow Matching、全参 AdamW `2.5e-5`、betas `(0.9,0.95)`、weight decay `1e-10`、
梯度裁剪 `1`、1000 步预热，调度器默认到 50000 步衰减至 `2.5e-6`。
训练预设为 40000 步、每卡 batch=32、BF16；上例显式使用较小 batch 和梯度累积。
梯度检查点默认开启，可用 `--policy.gradient_checkpointing=false` 关闭。

| 数据表示 | 输入与执行 |
| --- | --- |
| state=`tcp`、action=`tcp_delta`（默认） | 共享数据层转换关节 state/action；预测相对当前 TCP 的位姿目标，执行前用 IK 恢复 Host 目标 |
| state=`recorded`、action=`recorded` | 保留所选 state 和原 action；无需运动学配置 |

保留记录表示时，将上例两个表示参数均改为 `recorded` 并省略 `--data.kinematics.reference_pose`。
Spirit 需要数值 state 和非空任务文本，不支持 `--state=none`。
原 action 的名称与顺序须匹配转换描述；所选 state 单独记录，FK 所需的完整关节观测由读取器提供。
底盘和升降动作保持完整，不自动固定这些自由度。
转换后的 state/action 宽度最多各 32 维，原始关节数与派生 TCP 维度分别记录。
填充通道和 episode 尾部补齐目标不计入动作损失。
归一化分别在转换后的训练 state/action 上计算 p1/p99 并映射到 `[-1,1]`，不复用基座机器人的统计。

最多选择三路 RGB，相机映射须覆盖全部选中视角：

| 记录相机 | Spirit 槽位 |
| --- | --- |
| `forward` | `cam_high` |
| `wrist_left` | `cam_left_wrist` |
| `wrist_right` | `cam_right_wrist` |

其他名称通过 `--policy.camera_mapping='{"cam_high":"top","cam_left_wrist":"left","cam_right_wrist":"right"}'`
显式映射，并用 `--cameras` 选择对应视角。未使用的槽位填黑图。
图像先读取为 RGB 240×320，再由 Qwen processor 处理；训练使用 ColorJitter，验证和推理关闭增强。
每行 `task` 用作任务文本，机器人提示类型默认为 `aloha`，用 `--policy.robot_type` 覆盖。

预设参考实现固定到 [6d67377](https://github.com/Spirit-AI-Team/spirit-v1.5/tree/6d67377f16f67d6575447a1f29fdd83a2e0099d9)，
基座固定到 [9886fa7](https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5/tree/9886fa7552de70b097c4214b7e40e66ff937a6fd)。
参考 Franka 数据使用前一帧末端状态及未来实测位姿；平台保留记录的观测／动作配对和数据 FPS，
不重采样时间戳，使用共享训练调度器和完整训练窗口统计。这些协议差异、显式覆盖值和几何快照保存在 checkpoint 中。
官方说明见 [Finetune Guide](https://github.com/Spirit-AI-Team/spirit-v1.5#finetune-guide)。

原生 checkpoint 包含完整模型及 Qwen 配置、tokenizer、图像处理资源，可脱离原资源目录加载。
`preprocessing.json` 列出原始执行字段、派生统计字段的名称／单位和转换顺序。
支持单设备、DDP、FSDP2、离线评估和云端动作块推理；Spirit 不支持 RTC 前缀引导。

### 日志与配置

```bash
tail -f ~/Alohamini_workspace/logs/training/act_01.log
cat ~/Alohamini_workspace/logs/training/act_01.pid
```

命令行训练默认在后台运行，输出独立的日志、PID、配置路径和可复制的 `tail -f` 命令；输出目录须未使用。
需要直接在当前终端运行时使用 `--foreground`。
关闭终端或按 Ctrl+C 退出 `tail -f` 只停止日志查看，不会停止后台训练。
直接由 `torchrun` 启动时，每个 rank 在当前进程运行，不再启动新的后台任务。
可完全使用命令行参数，也可把 `examples/learning/act.json` 复制到工作区的 `configs/`，
修改数据路径、运行名称和回合编号后用 `--config /路径/train.json` 加载。
命令行参数优先，例如 `--steps=150000` 覆盖配置中的总步数。
个人实验配置留在工作区，不放入仓库的 `examples/`。

每一步的原始指标仍写入运行目录的 `metrics.jsonl`，续训写入 `metrics-from-*.jsonl`：

```bash
tail -f ~/Alohamini_workspace/runs/act_01/metrics.jsonl
```

### 验证与视频加载

`--eval_steps` 设置验证间隔，`--eval_batch_size` 设置验证 batch；
`--eval_num_workers`、`--eval_prefetch_factor` 可独立配置验证加载。
未指定时沿用对应训练设置，worker 为 0 时关闭多进程预取。
ACT／AM-ACT 的验证动作损失不含 KL，不能直接与训练总损失比较。

未指定视频后端时，训练器检测到已安装 TorchCodec 就使用它，否则使用 PyAV。
命令行或配置中显式指定的后端（包括 checkpoint 保存的设置）优先于自动选择；
可通过 `--video_backend=pyav` 切换到 PyAV。后端加载或解码失败时直接报错，不自动切换。
视频解码器使用每个 worker 独立的有界 LRU 缓存，淘汰时关闭解码器及文件；
`--video_cache_size=0` 表示逐次打开、读取和关闭。缓存上限不保证底层库内存始终有界。
需要调整读取性能时，可设置 `--video_backend=pyav|torchcodec`、`--camera_workers`、
`--video_cache_size`；增大 worker、预取与缓存会增加主机内存占用。
图像按各策略的尺寸、裁剪和归一化规则处理，配置随 checkpoint 保存。

### 优化器、精度与多卡

在训练命令后追加所需参数：

```text
--optimizer.type=adamw --optimizer.lr=0.0001
--optimizer.weight_decay=0.0001 --optimizer.betas='[0.9,0.999]'
--optimizer.eps=1e-8 --optimizer.grad_clip_norm=10
--gradient_accumulation_steps=4 --mixed_precision=bfloat16
```

优化器支持 `adamw`、`adam`、`sgd`，默认沿用策略配置。
ACT／AM-ACT 默认无调度，SmolVLA 使用 `cosine`，π0.5／Spirit 使用 `warmup_cosine`，
Diffusion／FastWAM 使用 `diffusers_cosine`；
`--scheduler.type=none` 禁用调度。可覆盖 `warmup_steps`、`decay_steps`、`decay_lr`。

`--mixed_precision=none|bfloat16|float16` 控制 autocast 计算精度；
`--trainable_dtype=preserve|float32` 独立控制可训练参数的存储类型。
默认 `preserve` 沿用模型类型；`float32` 在加载权重、创建优化器前将可训练参数设为 FP32，
冻结参数与 buffer 保留原类型。Adam／AdamW 的动量随参数采用 FP32。
需要保留低于 BF16 间距的小更新时，追加：

```text
--trainable_dtype=float32 --mixed_precision=bfloat16
```

该组合采用 FP32 权重更新与 BF16 混合精度计算，实现
[FP32 主权重方案](https://arxiv.org/html/2610.31859#S6)，适用于各原生策略及单设备、DDP、FSDP2。
相较 BF16 参数，FP32 参数与优化器动量占用更多内存；已有配置的学习率可能需要重新调整。
选项随 checkpoint 保存并在加载时恢复，从原生 checkpoint 微调时默认继承，显式参数可覆盖。
精确续训须保持原值；旧 checkpoint 按 `preserve` 处理。改变存储类型时，使用 `--policy.path` 开始新训练。

| 配置 | 追加参数 |
| --- | --- |
| 单卡 | 默认 |
| DDP，每卡完整模型 | `--num_processes=2 --background` |
| FSDP2 分片 | `--distributed_backend=fsdp2 --num_processes=2 --background` |

指定显卡：`CUDA_VISIBLE_DEVICES=2,5 alohamini train --config train.json --policy.device=cuda --num_processes=2 --background`。
所选显卡在进程内重新编号为 `cuda:0`、`cuda:1`。

有效 batch 通常为 `batch_size × 进程数 × 累积次数`；`steps` 按成功的优化器更新计数。
多进程尾批默认重复样本补齐，`--drop_last=true` 改为丢弃。
`num_workers`、`prefetch_factor` 和视频缓存按进程生效，多卡时需合计 CPU 内存与解码负载。
DataLoader 默认使用 `spawn`；可用 `--dataloader_multiprocessing_context=forkserver` 切换。
多卡启动时，rank 0 检查数据并计算或加载归一化统计，其他 rank 校验后复用；文件变化时重新检查。
`--stats` 可跨训练复用统计；启动结果同步默认最多等待 3600 秒，可用 `--dataset_preparation_timeout_s` 调整。
多节点使用 `torchrun ... -m alohamini.learning.train --config train.json`，各节点须共享输出目录并使用相同数据和资产路径。

FSDP2 由通用训练器管理参数、梯度和优化器分片，保留所选参数存储精度与冻结设置。
每个进程需要足够 CPU 内存构建模型；保存时同时导出分片训练状态和完整推理权重，主进程需留出完整权重的 CPU 内存。
FSDP2 不分摊输入激活；单卡仍需容纳当前计算模块及其激活。不支持 ZeRO 或张量并行。
FastWAM 的外置冻结 VAE／文本编码器仍在每张卡保留一份。

### 训练器 EMA

EMA 默认关闭，使用 `--ema.enable=true` 开启。
默认衰减参数为 `inv_gamma=1.0`、`power=0.75`、`min_decay=0`、`max_decay=0.9999`、`update_after_step=0`。
固定衰减可设 `--ema.decay=0.99`，与自定义 `min_decay/max_decay` 互斥。

EMA 以 FP32 追踪可训练参数，每次成功的优化器更新后推进；冻结参数不复制，BatchNorm buffer 直接复制。
验证默认使用 EMA 权重，`--ema.use_for_eval=false` 改用在线权重。
支持单设备、DDP 和 FSDP2；FSDP2 的 EMA 参数按相同布局分片。评估／导出需要临时内存。

## 5. 保存与续训

### 保存模型

`--save_freq=5000` 表示每完成 5000 个训练步保存一次，训练结束时也会保存。
最近保存的模型位于 `runs/<运行名称>/checkpoint`，可用于微调或推理。
若需要中断续训，保留整个运行目录；仅复制模型权重不足以恢复训练。

启用训练器 EMA 后，每个完整 checkpoint 同时保存：

| 路径 | 用途 |
| --- | --- |
| `pretrained_model/` | 在线模型，与优化器状态对应，用于恢复训练 |
| `ema_pretrained_model/` | 可直接部署的 EMA 模型，包含同一套预处理信息 |
| `training_state/state.pt` 中的 `ema` | 单设备／DDP 的完整 EMA 状态；FSDP2 的更新计数 |
| `distributed/` | FSDP2 模型、优化器及完整 EMA 分片状态 |

运行目录中的 `checkpoint` 指向最新在线模型，`checkpoint-ema` 指向最新 EMA 模型。
评估或部署 EMA 时指定 `--policy.path=<运行目录>/checkpoint-ema`。
续训仍使用 `checkpoint/train_config.json`，并保持 EMA 配置不变。
缺失或不匹配的 EMA 状态会拒绝续训。

### 中断后恢复原训练

数据和训练设置不变，只是进程中断时，使用保存的训练配置恢复：

```bash
python -m alohamini.learning.train \
  --config_path="$HOME/Alohamini_workspace/runs/act_01/checkpoints/last/pretrained_model/train_config.json" \
  --resume=true --background
```

| 选项 | 作用 |
| --- | --- |
| `--config_path` | 读取该次训练保存的配置，无需重新填写数据路径和训练参数 |
| `--resume=true` | 从最近保存的训练步继续，恢复权重、优化器和学习率进度 |
| `--background` | 在后台运行；启动后打印日志与 PID 路径 |

恢复时保持训练/验证数据、模型、学习率、调度器、batch、精度及多卡设置一致。
若原训练使用 `--scheduler.type=none`，可追加 `--steps=150000` 延长训练。
这里的步数是累计总目标：已训练 100000 步时，设为 150000 表示再训练 50000 步。
启用学习率调度的原运行须保持原定总步数。

### 补充数据或调整学习率后微调

新增示范、改用合并数据集、更换验证回合，或希望调整学习率时，使用 `--policy.path`
加载已有模型，并指定一个新的输出目录。此时不加 `--resume=true`。
各原生策略均支持从平台 checkpoint 微调；下面以 ACT／AM-ACT 为例：

```bash
python -m alohamini.learning.train \
  --policy.path="$HOME/Alohamini_workspace/runs/act_01/checkpoint" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/merged_task" \
  --dataset.eval_episodes='[1]' \
  --output_dir="$HOME/Alohamini_workspace/runs/act_finetune_01" \
  --policy.device=cuda \
  --optimizer.lr=5e-6 \
  --policy.optimizer_lr_backbone=5e-6 \
  --scheduler.type=none \
  --steps=50000 --batch_size=2 \
  --save_freq=5000 --eval_steps=1000 --log_freq=100 \
  --background
```

替换数据集路径和验证回合编号；合并数据后按合并后的编号选择验证回合。
验证示范不要与训练示范重复。模型结构、相机、图像尺寸和 state 选择默认沿用旧模型，
新数据的动作字段、单位和输入含义须与它匹配。

| 选项 | 作用 |
| --- | --- |
| `--policy.path` | 要继续学习的已有模型，策略类型自动读取 |
| `--dataset.root` | 本次微调使用的数据集目录 |
| `--dataset.eval_episodes='[1]'` | 留出回合 1 做验证；未指定训练回合时，其余回合用于训练 |
| `--output_dir` | 新模型的保存目录，须尚不存在；原模型保留 |
| `--policy.device=cuda` | 使用 GPU 训练；`cpu` 使用 CPU |
| `--steps=50000` | 本次微调训练 50000 步，不包含旧模型已经训练的步数 |
| `--batch_size=2` | 每个进程每批读取 2 个样本；增大会增加显存需求 |
| `--save_freq=5000` | 每 5000 步保存模型 |
| `--eval_steps=1000` | 每 1000 步在留出的回合上验证；须同时指定验证回合 |
| `--log_freq=100` | 每 100 步打印一次训练指标 |

微调保留已有权重，但重新开始优化器和学习率计划。上例中的步数、batch 和学习率均可按任务调整。

### 调整学习率

学习率控制每次训练更新参数的幅度。较小的值通常使已有模型改变得更缓慢，
但不保证更高成功率，应结合验证指标选择。`5e-6` 就是 `0.000005`。

| 选项 | 作用 |
| --- | --- |
| `--optimizer.lr=5e-6` | 设置主学习率；在 ACT／AM-ACT 中用于视觉骨干以外的网络 |
| `--policy.optimizer_lr_backbone=5e-6` | 单独设置 ACT／AM-ACT 的视觉骨干学习率，控制图像特征提取网络的更新幅度 |
| `--scheduler.type=none` | 不随训练步数改变学习率，各部分始终使用设定值；不是停止学习 |

**只改 `--optimizer.lr` 不会覆盖 ACT／AM-ACT 的视觉骨干学习率。**
想让两部分使用同一学习率，就像上例一样同时设置；想让视觉特征变化更小，
可分别设为 `--optimizer.lr=5e-6 --policy.optimizer_lr_backbone=1e-6`。
其他算法使用各自的参数分组，不要直接套用 ACT 的骨干选项。

若希望先逐渐提高学习率、再逐渐降低，可将微调命令中的 `--scheduler.type=none` 替换为：

```text
--scheduler.type=warmup_cosine \
--scheduler.warmup_steps=1000 \
--scheduler.decay_steps=50000 \
--scheduler.decay_lr=5e-7
```

| 选项 | 作用 |
| --- | --- |
| `--scheduler.type=warmup_cosine` | 先预热，再按余弦曲线降低学习率 |
| `--scheduler.warmup_steps=1000` | 前 1000 步从较小学习率逐渐提高到设定值 |
| `--scheduler.decay_steps=50000` | 到本次训练第 50000 步时降到目标值，包含预热阶段 |
| `--scheduler.decay_lr=5e-7` | 主学习率的最终目标；其他参数组按相同比例降低 |

调度器不为 `none` 时，`--optimizer.lr` 指设定的峰值，不代表每一步的实际学习率。
使用 `none` 时不填写上述预热和衰减参数。

### 选择归一化统计与查看日志

| 微调选项 | 作用 |
| --- | --- |
| 省略或 `--normalization=dataset` | 默认按本次训练数据重算统计，验证回合不参与 |
| `--normalization=checkpoint` | 沿用旧模型的统计，不根据新增数据重算 |
| `--stats=/路径/statistics.json` | 使用预先生成、与本次数据及训练配置匹配的统计文件，不能与上一项同时使用 |

后台微调的日志和 PID 位于 `~/Alohamini_workspace/logs/training/`。查看上例日志：

```bash
tail -f ~/Alohamini_workspace/logs/training/act_finetune_01.log
```

退出 `tail -f` 不会停止训练。恢复原运行时日志名带有续训标识，使用启动时打印的实际日志路径。

## 6. 离线评估

```bash
python -m alohamini.learning.evaluate \
  --policy.path ~/Alohamini_workspace/runs/act_01/checkpoint \
  --dataset.root ~/Alohamini_workspace/datasets/task_demo_ready \
  --dataset.episodes '[1]' --device cuda \
  --output ~/Alohamini_workspace/runs/act_01/offline-review.json
```

使用 checkpoint 的相机、字段、图像尺寸及归一化统计，输出各动作维度的原单位 MAE 和数据检查报告。
数据须保持相同标定、动作含义和 FPS；验证回合不得与训练示教重复。离线误差不等于真机任务成功率。

## 7. 真机评估

PC 加载模型，树莓派运行 Host。停止其他运动控制程序，确认标定、相机和机器人姿态，并清空运动区域。

树莓派：

```bash
conda activate alohamini_host
alohamini host --robot_model alohamini2pro --cameras forward wrist_right
```

PC（下列命令会驱动机器人）：

```bash
conda activate alohamini
alohamini evaluate --host <PI_IP> --robot_model alohamini2pro \
  --policy.path ~/Alohamini_workspace/runs/act_01/checkpoint \
  --device cuda --fps 30 --episode_time 10 --num_episodes 1 \
  --policy.n_action_steps 10 --policy.temporal_ensemble_coeff none
```

`--fps` 与训练数据一致。每个动作块执行前 `n_action_steps` 步，再重新预测。
ACT／AM-ACT 每步预测并融合时，使用 `--policy.n_action_steps 1 --policy.temporal_ensemble_coeff 0.01`；
`none` 关闭融合，`0` 为等权融合。更换执行参数无需重新训练。

SmolVLA／π0.5／FastWAM／Spirit 评估须增加 `--task "拿起物体"`。Diffusion／FastWAM／Spirit 不使用 ACT 时间融合。
`--dataset eval_01 --task "拿起物体"` 保存评估回合。Ctrl+C 结束。持续反馈中断或关节保护时暂停，恢复后清空旧策略缓存并继续；Host 重启、标定变化或其他客户端接管时结束评估。
推理耗时限制实际执行频率。

跨机器使用时复制完整 checkpoint 目录，勿只复制符号链接。
自定义策略接口见 [客户端接口](host-protocol.md#python-策略评估)，示例为 `examples/learning/custom_policy.py`。
Notebook 入口见 [Notebook](notebook.md)，LeRobot 权重使用见 [LeRobot](lerobot.md)。

### RTC 异步推理

π0.5（flow／KI）和 SmolVLA 支持推理期引导 RTC，无需重新训练；默认关闭。
在评估命令中增加：

```bash
--rtc --rtc.replan_steps 10 --rtc.inference_delay_steps 3
```

RTC 使用完整预测块，重规划间隔由 `rtc.replan_steps` 控制，覆盖 `n_action_steps` 的执行节奏。
旧块在后台生成新块时继续执行；π0.5 的旧绝对目标会按新 state 重编码。
动作按观测时间对齐，跳过过期前缀；队列耗尽或预测过期时停止，复位清空旧计划。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `rtc.replan_steps` | 10 | 两次预测起点的最小间隔，单位为控制步 |
| `rtc.inference_delay_steps` | 3 | 延迟估计下限；运行中结合实测延迟加一控制步更新 |
| `rtc.max_guidance_weight` | 10 | 引导强度上限 |
| `rtc.prefix_attention_schedule` | `linear` | 重叠区权重，可选 `linear/exp/ones/zeros` |
| `rtc.max_prediction_age_s` | 1.0 | 动作源观测的最大允许年龄，单位秒 |

预测块长度须大于重规划间隔加延迟，实测延迟也须小于剩余重叠区。
RTC 需要对动作输入求梯度，使用 eager 采样，会增加计算和显存开销；不与 ACT 时间融合组合。
评估数据的 `meta/evaluation.json` 保存 RTC 参数，帧元数据保存预测来源时间。

云端推理时，在 GPU 服务端的 `alohamini inference serve` 命令后追加 `--rtc`；
Pi 根据握手配置执行，连接步骤见[云端推理](#云端推理)。
RTC 模式的 `--chunk-size` 默认使用完整预测块；有效预测年龄上限取服务端 RTC 设置与 Pi 的
`--max-prediction-age-s`（默认 1 秒）的较小值。两端须使用支持 RTC 协议的版本。
算法参考 [Physical Intelligence RTC](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/tree/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b)，仅提供推理期引导模式。

### 固定维度的训练与推理

数据没有声明固定维度时，可在评估命令中显式补选，例如：

```bash
alohamini evaluate --host <PI_IP> --robot_model alohamini2pro \
  --policy.path ~/Alohamini_workspace/runs/pretrained_model \
  --device cuda --fps 30 --episode_time 10 \
  --fixed-dimensions arm_right \
  --fixed-dataset ~/Alohamini_workspace/datasets/task_demo_ready
```

支持左右臂、升降、底盘组和具名关节，字段与采集选项相同。已有固定配置继续继承，补选只增加固定维度。
新增位置目标使用训练 **action 的物理坐标均值**；旧模型相应 state 输入采用 checkpoint 的
**state 均值**，避免微小测量差异被归一化放大。真实反馈仍用于安全检查和记录。
固定底盘仍为零速度。把训练中活动的维度改为固定，会改变部署条件，不等同于恢复原任务效果。
保持遥操摆好的当前位置，`--fixed-current` 不需要 `--fixed-dataset`，也不读取训练数据来决定位置。支持 `arm_left`、`arm_right`、
`lift_axis`、`base` 和单个具名动作字段。模型加载后，Host 握手的首次新鲜、可控制反馈决定目标，
随后确认稳定再开始推理；本次运行的后续回合沿用同一目标，不重新采样。底盘仍固定为零速度。
选中字段会覆盖 checkpoint 中继承的固定目标，其余继承目标保持不变。

两种来源可以分开指定，例如 `--fixed-dimensions arm_right --fixed-current lift_axis`，
表示右臂用原有固定目标或训练动作均值、升降保持启动时高度。同一字段不能同时出现在两组选项中。
仅省略 `--fixed-dataset` 不会改变 `--fixed-dimensions` 原有的均值语义。

### 标定兼容

| `--calibration-mode` | 用途 |
| --- | --- |
| `strict`（默认） | 要求训练标定与 Host 一致 |
| `normalized` | 同型号机器间按归一化关节比例迁移，须有训练标定 |
| `legacy_normalized` | 原始 LeRobot v3 模型缺少标定时，显式声明动作单位并绑定当前 Host |

`legacy_normalized` 仅适用于普通关节 `[-100,100]`、夹爪 `[0,100]`、
底盘平移 m/s、旋转 deg/s、升降 mm 的数据，动作名称与顺序须匹配型号。
使用 state 时，原 state 也须为相同名称与顺序的 Host 位置、底盘速度和升降高度。
迁移不会恢复旧机器的物理姿态或升降零点；升降零点不明确时，
可加 `--fixed-current lift_axis` 保持启动高度。

使用 `--fixed-current` 前，先遥操摆好位置，再退出遥操并保持同一 Host 进程。
绑定标定与固定目标报告保存在工作区 `logs/evaluation/`；运行中标定变化会结束评估。

### 云端推理

树莓派直接发送图像和 state 到云端 GPU，接收动作块并在本地执行；PC 用于登录管理。
当前支持平台原生 PI05、SmolVLA 和 Spirit checkpoint。
云端与树莓派使用相同版本的 Alohamini；树莓派先启动本地 Host，完成标定和升降归零，退出其他控制客户端。
Host 地址保持 `127.0.0.1`。ZeroMQ 推理端口 `5560` 通过 SSH 隧道或 WireGuard 连接，不直接开放到公网。

在各自已激活的环境、项目根目录安装。云端 PI05 环境执行：

```bash
pip install -e '.[pi05,zmq]'
```

SmolVLA／Spirit 云端环境分别使用 `.[smolvla,zmq]`、`.[spirit,zmq]`。
树莓派复用运行 Host 的环境：

```bash
pip install -e '.[host]'
```

以下服务端命令以 `pi05_01/checkpoint` 为例，替换为实际模型目录和任务文本；
SSH 命令中的 `PORT`、`USER`、`CLOUD` 分别替换为云端 SSH 端口、用户名和地址。
更换租用实例时更新 SSH 地址和端口；WireGuard 还需更新云端公钥及树莓派的 `Endpoint`。

#### SSH 隧道

云端启动推理服务：

```bash
alohamini inference serve --bind tcp://127.0.0.1:5560 \
  --policy.path "$HOME/Alohamini_workspace/runs/pi05_01/checkpoint" \
  --device cuda --task "拿起物体"
```

等待显示 `Remote policy server listening on tcp://127.0.0.1:5560` 后，树莓派建立隧道并保持终端运行：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:5560:127.0.0.1:5560 -p PORT USER@CLOUD
```

首次连接核对云端 SSH 主机指纹。SSH 服务须允许本地端口转发，且它所在的网络环境能访问
云端 `127.0.0.1:5560` 上的推理服务。[OpenSSH 转发说明](https://man.openbsd.org/ssh.1)

树莓派另开终端执行：

```bash
alohamini inference run --server tcp://127.0.0.1:5560 \
  --host 127.0.0.1 --robot_model alohamini2pro --fps 30
```

`ExitOnForwardFailure` 只检查转发监听，SSH 保活不替代动作有效期和控制 watchdog。
[OpenSSH 配置说明](https://man.openbsd.org/ssh_config.5)
停止时先按 Ctrl+C 退出树莓派推理客户端，再关闭隧道和云端服务。
若树莓派本地 `5560` 已占用，同时修改 `-L` 的第一个端口和 `--server` 端口，云端目标端口保持不变。

#### WireGuard

两端需能创建 WireGuard 接口，云端需提供可达 UDP 端口。租用容器可能缺少网络管理权限、
WireGuard 内核支持或用户态实现所需的 `/dev/net/tun`；环境不满足时使用 SSH 隧道。
按系统安装 [WireGuard 工具](https://www.wireguard.com/install/)。

两端各生成一次密钥，以下命令拒绝覆盖已有文件：

```bash
(
    umask 077
    set -eC
    mkdir -p "$HOME/.config/alohamini/wireguard"
    chmod 700 "$HOME/.config/alohamini/wireguard"
    wg genkey > "$HOME/.config/alohamini/wireguard/private.key"
    wg pubkey < "$HOME/.config/alohamini/wireguard/private.key" > "$HOME/.config/alohamini/wireguard/public.key"
)
```

只交换公钥。私钥和填好私钥的配置存放在各自机器的仓库之外，权限为 `600`。
[密钥生成说明](https://www.wireguard.com/quickstart/)
树莓派密钥可在更换云端实例后继续使用，新云端配置填入该树莓派公钥。

云端复制[云端模板](../deploy/remote-inference/wireguard-cloud.conf.example)：

```bash
cp -n deploy/remote-inference/wireguard-cloud.conf.example "$HOME/.config/alohamini/wireguard/wg-infer.conf"
chmod 600 "$HOME/.config/alohamini/wireguard/wg-infer.conf"
```

树莓派复制[树莓派模板](../deploy/remote-inference/wireguard-pi.conf.example)：

```bash
cp -n deploy/remote-inference/wireguard-pi.conf.example "$HOME/.config/alohamini/wireguard/wg-infer.conf"
chmod 600 "$HOME/.config/alohamini/wireguard/wg-infer.conf"
```

在各自本机编辑 `wg-infer.conf`，填入本机私钥及对端公钥；树莓派将 `CLOUD_PUBLIC_HOST` 改为云端公网地址。
模板使用云端 `10.88.0.1/32`、树莓派 `10.88.0.2/32`，仅路由对端 `/32`，不设置 NAT、转发或互联网默认路由。
若地址冲突，先更换两端地址并同步修改配置及命令。
[路由说明](https://git.zx2c4.com/wireguard-tools/about/src/man/wg-quick.8)

云端安全组和防火墙允许 UDP `51820`，推理 TCP `5560` 仅允许 WireGuard 接口上的树莓派访问。
若平台映射了其他外部 UDP 端口，修改树莓派 `Endpoint`，云端 `ListenPort` 保持内部端口。
`PersistentKeepalive = 25` 用于维持 NAT 映射。
[WireGuard 配置说明](https://git.zx2c4.com/wireguard-tools/about/src/man/wg.8)

两端分别启动：

```bash
sudo wg-quick up "$HOME/.config/alohamini/wireguard/wg-infer.conf"
sudo wg show wg-infer
```

树莓派可用 `ping -c 3 10.88.0.1` 检查目标连通性。ICMP 被禁用时仍需检查握手、路由和服务监听。
云端启动：

```bash
alohamini inference serve --bind tcp://10.88.0.1:5560 \
  --policy.path "$HOME/Alohamini_workspace/runs/pi05_01/checkpoint" \
  --device cuda --task "拿起物体"
```

等待服务就绪后，树莓派执行：

```bash
alohamini inference run --server tcp://10.88.0.1:5560 \
  --host 127.0.0.1 --robot_model alohamini2pro --fps 30
```

停止推理客户端和服务后，两端按需关闭隧道：

```bash
sudo wg-quick down "$HOME/.config/alohamini/wireguard/wg-infer.conf"
```

#### 连接与执行参数

SSH 隧道连接失败时，检查云端推理进程及 SSH 转发目标是否处于同一网络环境。
WireGuard 没有握手时检查公钥、UDP 入口和 `Endpoint`；已有握手时检查 `/32` 路由、绑定地址和接口防火墙。
云端查看推理监听地址：

```bash
ss -ltn '( sport = :5560 )'
```

`--fps` 必须与 checkpoint 数据频率一致。服务端 `--chunk-size` 省略时使用 checkpoint 的可执行窗口，
可用 `--chunk-size 32` 请求更短动作块，但不能超过该窗口；RTC 使用完整预测块并单独校验重叠区。
这些参数不保证云端持续按执行频率返回结果。迟到结果跳过过期前缀，动作耗尽或 Host 会话、控制参考变化时停止，需显式重新运行。

| 树莓派 `run` 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--episode_time` | 90 秒 | 每回合运行时长 |
| `--rpc-timeout-s` | 10 秒 | 云端请求超时 |
| `--max-prediction-age-s` | 1 秒 | 预测源观测的最大年龄 |
| `--ack-timeout-s` | 0.5 秒 | Host 确认超时 |

固定当前关节时，在云端 `serve` 和树莓派 `run` 两端传入相同的 `--fixed-current FIELD ...`。
支持 `base`、`lift_axis`、`arm_left`、`arm_right` 或具名 Host 字段；树莓派从新鲜反馈捕获固定目标。
PI05／SmolVLA 可在服务端开启[RTC](#rtc-异步推理)，仅提供推理期引导，不包含 RTC 训练。
Spirit 使用普通动作块推理。实际链路时延与真机成功率须分别评估，不通过放宽控制 watchdog 补偿网络延迟。

## 8. 对比评估与训练示范

`alohamini dataset analyze` 在本地查看已保存的评估记录，并与评估所用 checkpoint 的训练或验证示范对比。无需连接机器人。

| 功能 | 支持算法 |
| --- | --- |
| Rerun 画面与动作曲线、示范对比、报告导出 | 平台各算法，包括 ACT、AM-ACT、Diffusion Policy、FastWAM、SmolVLA、π0.5 |
| 加 `--probe-samples`，重新预测示范动作并计算误差 | 平台策略，按各自声明的推理输入和动作窗口检查 |

以下以评估数据集 `eval_01` 为例。请替换数据集路径、episode 编号和时间点；episode 编号从 0 开始。

### 打开评估记录

```bash
alohamini dataset analyze ~/Alohamini_workspace/datasets/eval_01 \
  --episode 0
```

默认打开 Rerun，显示评估第 0 条回合的相机画面和动作曲线。
有模型关联记录时，自动找到对应 checkpoint 和原训练数据集，并显示保存的训练划分中的首条示范。
普通查看不运行模型，也不需要 GPU。命令、Host 反馈、电流和耗时默认隐藏。

如果评估记录没有保存模型关联，使用 `--checkpoint` 指定**当时评估使用的模型**。
模型或原训练数据搬到其他目录时，可分别补充这两个路径：

```bash
alohamini dataset analyze ~/Alohamini_workspace/datasets/eval_01 \
  --episode 0 \
  --checkpoint ~/Alohamini_workspace/runs/pickup_01/checkpoint \
  --training-root ~/Alohamini_workspace/datasets/pickup_train
```

`--training-root` 应指向原训练数据集的本地副本；它不用于替换成另一份训练数据。

### 选择示范并对齐抓取阶段

假设评估回合 0 在第 20 秒开始抓取，训练示范 12 在第 8 秒开始抓取，可以对比各自接下来的 15 秒：

```bash
alohamini dataset analyze ~/Alohamini_workspace/datasets/eval_01 \
  --episode 0 --reference-split train --reference-episode 12 \
  --time-range 20 35 --reference-time-range 8 23 \
  --align 20 8
```

| 选项 | 本例含义 |
| --- | --- |
| `--episode 0` | 查看评估数据集中的回合 0 |
| `--reference-split train` | 从 checkpoint 保存的训练划分中选择示范；`val` 选择验证划分 |
| `--reference-episode 12` | 对照原训练数据集中的回合 12，须属于所选划分 |
| `--time-range 20 35` | 截取评估第 20～35 秒 |
| `--reference-time-range 8 23` | 截取示范第 8～23 秒 |
| `--align 20 8` | 将评估第 20 秒与示范第 8 秒对齐到公共时间轴的 0 秒 |

省略 `--reference-episode` 时选择该划分的首条示范。省略 `--align` 时只按回合起点并排，抓取阶段可能并不同步。
对齐不会改变播放速度。画面和曲线用于观察动作顺序、选臂及轨迹差异；两次运行不计算逐帧动作误差。

### 导出图表与画面报告

沿用上面的片段，导出评估第 21、25、30 秒的画面及对齐后的示范画面：

```bash
alohamini dataset analyze ~/Alohamini_workspace/datasets/eval_01 \
  --episode 0 --reference-split train --reference-episode 12 \
  --time-range 20 35 --reference-time-range 8 23 --align 20 8 \
  --viewer none --export-html --keyframes 21 25 30 \
  --output ~/Alohamini_workspace/reports/eval_01_compare
```

`--viewer none` 不打开 Rerun；用浏览器打开输出目录中的 `report.html` 查看图表和画面。
`--output` 须为数据集目录之外的新目录。省略时，结果保存在
`~/Alohamini_workspace/reports/<数据集>/episode_<编号>/<时间编号>/`。
每次分析都会保存 `summary.json` 和 `events.csv`，HTML 报告的图表另存于 `plots/`。

### 检查模型在示范上的预测

以下命令加载评估对应的 checkpoint，从其验证划分的首条示范中均匀抽取最多 8 个观测，重新预测动作：

```bash
alohamini dataset analyze ~/Alohamini_workspace/datasets/eval_01 \
  --episode 0 --reference-split val \
  --probe-samples 8 --probe-device cuda \
  --viewer none --export-html
```

需要 checkpoint 保存过非空的验证划分；可加 `--reference-episode` 选择其中某条示范。
如果训练时全部示范都用于训练、没有留出验证回合，改用 `--reference-split train` 检查训练拟合。
`--training-root` 只指定数据位置，不会创建验证划分。
`--probe-device cuda` 使用 GPU，改为 `cpu` 可在 CPU 上运行。

报告逐个动作维度显示首步预测和有效动作段的平均绝对误差（MAE），单位与示范动作一致。
首步对应当前动作（行偏移 0）；动作段只统计当前及未来标签，排除历史前缀和回合末尾 padding。
预测使用示范中的观测，对照的是**同一条示范的后续真实动作**，不是评估回合的动作。
输入包含策略所需的任务文本和历史观测；CDP 还使用示范中严格过去的动作，不读取当前或未来动作标签。
每个查询独立重置模型，不复现在线动作队列、时序融合或 RTC。FastWAM 只读取当前图像，不输入训练用未来图像。
保持基线重复当前同名、同单位的观测值；无法确认坐标单位时该维基线不可用。模型与基线分别报告有效步数。
`demonstration_probe.json` 保存指标、模型指纹和采样配置；`demonstration_predictions.npz`
保存预测、标签、字段单位、行偏移和有效掩码。这项误差用于检查模型预测，不代表真机成功率。
改用 `--reference-split train` 可检查训练拟合；检查泛化优先使用验证示范。
不加 `--probe-samples` 就不会加载模型。

### 其他查看选项

| 选项 | 作用 |
| --- | --- |
| `--cameras wrist_left wrist_right` | 只显示或导出指定相机，并按给定顺序排列 |
| `--save-rrd` | 保存 `recording.rrd`，随后用终端打印的 `rerun` 命令打开 |
| `--control-diagnostics` | 额外显示命令、Host 反馈、电流和耗时 |
| `--probe-seed 0` | 设置重新预测时的随机种子，便于重复比较 |
| `--segment 0` / `--reference-segment 0` | 数据有时钟重置时，选择评估／示范的时间段 |

已有 PC 环境包含查看依赖；独立分析环境可在仓库目录运行 `pip install -e '.[analysis]'`。
重新预测还需要对应策略的运行依赖。

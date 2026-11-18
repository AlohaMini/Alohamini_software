# 数据整理、训练与部署

## 1. 准备

完成 [Linux 安装](install.md)或 [macOS 安装](macos.md#1-安装)，按需安装[策略与功能扩展](install.md#3-可选扩展)，在仓库根目录执行：

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

`--state` 选择输入字段，以下参数选择输入和动作的表示；原数据不变。

| 参数 | 选项与用途 |
| --- | --- |
| `--data.state_representation` | `recorded`：原始输入；`tcp`：将臂关节位置改为末端位姿 |
| `--data.action_representation` | `recorded`：原始控制目标；`joint_delta`：相对关节目标；`tcp_absolute`：绝对末端目标；`tcp_delta`：相对末端目标 |
| `--data.joint_delta_fields` | 外部数据的动作字段到观测字段 JSON 映射；平台关节自动匹配，双方坐标与单位须一致 |
| `--data.rotation_representation` | `rotvec`（默认，3 维）或 `rotation_6d`（6 维） |
| `--data.tcp_delta_frame` | TCP 增量坐标系：`base`（默认，各臂基座）或 `tool`（当前末端） |
| `--data.kinematics.reference_pose` | 型号标定预设，如 `upright_forward` |
| `--data.kinematics.spec` / `--data.kinematics.path` | 自定义映射 JSON / 映射文件路径，见[示例](../examples/datasets/kinematics-radians.json) |

输入和动作可独立选择；未指定时沿用[策略预设](#策略选择)。相对动作均以当前观测为基准，整段共用同一基准；夹爪等其他字段仍为绝对目标。
TCP 位置单位为 m，旋转向量单位为 rad；左右臂分别使用 `left_Base`、`right_Base`。

TCP 训练示例：

```bash
alohamini train --policy.type=act \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --state=joint_position \
  --data.state_representation=tcp \
  --data.action_representation=tcp_delta \
  --run_name=act_tcp_01 --background
```

标定默认读取数据集 `meta/kinematics.json`，缺省使用型号预设。主从臂标定见 [使用手册](alohamini.md#2-主从臂标定)；非标准标定或外部数据须指定实际映射。
映射随 checkpoint 保存，部署时须与机器人标定一致。需要导出映射时：

```bash
mkdir -p "$HOME/Alohamini_workspace/configs"
python -m alohamini.datasets.kinematic_fields \
  --dataset "$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --output "$HOME/Alohamini_workspace/configs/task_demo_kinematics.json"
```

导出文件须尚不存在；自定义映射加 `--spec`，使用 Host 标定文件加 `--left-map /path/left.yaml --right-map /path/right.yaml`。
训练时通过 `--data.kinematics.path` 加载；`reference_pose`、`spec`、`path` 三选一。
TCP 数据须保留关节观测，即使模型使用 `--state=none`。TCP 推理设置见[推理模式](#推理模式)。

## 4. 训练

### 策略选择

各策略的训练命令、权重准备和专用参数见对应目录：

| 策略 | `policy.type` |
| --- | --- |
| <a id="act"></a>[ACT](../src/alohamini/policies/act/README.md) | `act` |
| <a id="am-act"></a><a id="条件-cvae"></a>[AM-ACT](../src/alohamini/policies/am_act/README.md)（[条件 CVAE](../src/alohamini/policies/am_act/README.md#条件-cvae)） | `am_act` |
| <a id="diffusion-policy"></a>[Diffusion Policy](../src/alohamini/policies/diffusion/README.md) | `diffusion` |
| <a id="b-spline-policy"></a>[B-spline Policy](../src/alohamini/policies/bspline/README.md) | `bspline` |
| <a id="cdp"></a>[CDP](../src/alohamini/policies/cdp/README.md) | `cdp` |
| <a id="dit-policydit-block"></a>[dit-policy（DiT-Block）](../src/alohamini/policies/dit_policy/README.md) | `dit_policy` |
| <a id="multi-task-dit"></a>[Multi-Task DiT](../src/alohamini/policies/multi_task_dit/README.md) | `multi_task_dit` |
| <a id="fastwam"></a>[FastWAM](../src/alohamini/policies/fastwam/README.md) | `fastwam` |
| [Long-WAM](../src/alohamini/policies/longwam/README.md) | `longwam` |
| [VQ-BeT](../src/alohamini/policies/vqbet/README.md) | `vqbet` |
| <a id="ac-dit"></a>[AC-DiT](../src/alohamini/policies/acdit/README.md) | `acdit` |
| <a id="gr00t-n17"></a>[GR00T N1.7](../src/alohamini/policies/gr00t_n17/README.md) | `gr00t_n17` |
| <a id="leworldmodel"></a>[LeWorldModel](../src/alohamini/policies/lewm/README.md) | `lewm` |
| <a id="smolvla"></a>[SmolVLA](../src/alohamini/policies/smolvla/README.md) | `smolvla` |
| <a id="π05"></a>[π0.5](../src/alohamini/policies/pi05/README.md) | `pi05` |
| [π0](../src/alohamini/policies/pi0/README.md) | `pi0` |
| [π0-FAST](../src/alohamini/policies/pi0_fast/README.md) | `pi0_fast` |
| [EO-1](../src/alohamini/policies/eo1/README.md) | `eo1` |
| [EVO1](../src/alohamini/policies/evo1/README.md) | `evo1` |
| [X-VLA](../src/alohamini/policies/xvla/README.md) | `xvla` |
| [DM0.5](../src/alohamini/policies/dm05/README.md) | `dm05` |
| [FineART-VLA](../src/alohamini/policies/fineart_vla/README.md) | `fineart_vla` |
| [WALL-OSS-0.5](../src/alohamini/policies/wall_oss_05/README.md) | `wall_oss_05` |
| [VLA-JEPA](../src/alohamini/policies/vla_jepa/README.md) | `vla_jepa` |
| [LingBot-VA](../src/alohamini/policies/lingbot_va/README.md) | `lingbot_va` |
| [Cosmos3 Nano](../src/alohamini/policies/cosmos3/README.md) | `cosmos3_nano` |
| [Cosmos3 Edge](../src/alohamini/policies/cosmos3/README.md) | `cosmos3_edge` |
| [G0.5](../src/alohamini/policies/g05/README.md) | `g05` |
| <a id="hyvla-05"></a>[HyVLA-0.5](../src/alohamini/policies/hyvla05/README.md) | `hyvla05` |
| <a id="openwam-α"></a>[OpenWAM-α](../src/alohamini/policies/openwam/README.md) | `openwam` |
| <a id="xiaomi-robotics-1"></a>[Xiaomi-Robotics-1](../src/alohamini/policies/xiaomi_robotics1/README.md) | `xiaomi_robotics1` |
| <a id="lingbot-vla-20"></a>[LingBot-VLA 2.0](../src/alohamini/policies/lingbot_vla_v2/README.md) | `lingbot_vla_v2` |
| <a id="spirit-v15"></a>[Spirit v1.5](../src/alohamini/policies/spirit/README.md) | `spirit` |

各策略示例保留 episode 1 做验证，其余回合训练。数据只有一个回合时，省略 `--dataset.eval_episodes` 和 `--eval_steps`。

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

本地数据的数值与诊断记录使用磁盘缓存，默认每个数据集、每个进程缓存 64 MiB。
`--dataset.cache_size_mb=64` 设置缓存预算；`--dataset.cache_dir=/路径/cache` 指定源数据目录之外的缓存位置，默认位于工作区 `cache/datasets/`。总内存还包括图像解码、DataLoader 预取和 batch 张量。

`--dataset.streaming=true` 启用有限回合池混合采样，适合长回合数据；`--dataset.streaming_episode_pool_size=8` 设置回合池大小。此方式改变样本混合顺序，验证集仍按序读取；续训沿用原采样方式与回合池大小。

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

### 优化器与调度

在训练命令后追加所需参数：

```text
--optimizer.type=adamw --optimizer.lr=0.0001
--optimizer.weight_decay=0.0001 --optimizer.betas='[0.9,0.999]'
--optimizer.eps=1e-8 --optimizer.grad_clip_norm=10
```

优化器支持 `adamw`、`adam`、`sgd`，默认沿用策略配置。
ACT／AM-ACT 默认无调度，SmolVLA 使用 `cosine`，π0.5／Spirit 使用 `warmup_cosine`，
Diffusion／FastWAM 使用 `diffusers_cosine`；
`--scheduler.type=none` 禁用调度。可覆盖 `warmup_steps`、`decay_steps`、`decay_lr`。
`warmup_linear` 在预热后线性衰减；它与 `staged_cosine`、`constant_with_warmup`
均支持 `--scheduler.start_step`；
`--scheduler.before_start` 接收“参数组名到学习率倍率”的 JSON 对象，设置阶段开始前各组的倍率。
组名由策略声明；不在对象中的组默认倍率为 1，未知组名会报错。阶段按成功优化器更新计数。

### 数值精度

不指定时沿用策略预设；三项设置分别控制计算、参数存储和 FP32 加速：

| 设置 | 参数 | 默认 |
| --- | --- | --- |
| 自动混合精度计算 | `--mixed_precision=none\|bfloat16\|float16` | 策略预设，未配置时为 `none` |
| 可训练参数的存储与更新 | `--trainable_dtype=preserve\|float32` | 策略预设，未配置时为 `preserve` |
| 允许 FP32 矩阵乘法、卷积使用 TF32 | `--tf32=true\|false` | `true` |

常用组合，追加到训练命令：

```text
# FP32 可训练参数，BF16 混合精度计算
--trainable_dtype=float32 --mixed_precision=bfloat16

# 关闭自动混合精度和 TF32
--mixed_precision=none --tf32=false
```

`none` 关闭自动混合精度，不会把模型已有的 BF16 权重改成 FP32。
`trainable_dtype=float32` 只转换可训练参数，冻结参数和 buffer 保持原类型；显存占用会增加。
TF32 不改变参数存储类型，只影响 FP32 运算，可与 BF16/FP16 混合精度并用。
`--cudnn_deterministic=true` 时 TF32 默认关闭；显式 `--tf32` 优先。

设置随 checkpoint 保存，推理沿用已保存的设置。
`--policy.path` 开始新微调时，只默认继承 `trainable_dtype`；混合精度、TF32 使用本次配置或策略默认值。
显式参数优先；`--resume=true` 续训须保持原设置。模型内部必须使用 FP32 的算子保留其数值保护。

### Apple Silicon（MPS）

将策略示例的 `--policy.device=cuda` 改为 `--policy.device=mps`，首次运行使用：

```text
--policy.device=mps --num_processes=1 --mixed_precision=none
```

CPU 训练使用 `--policy.device=cpu`。MPS 使用单进程，DataLoader 沿用默认 `spawn`；
内存紧张时减小 `--batch_size`、`--num_workers`，并按需增加 `--gradient_accumulation_steps`。
`--mixed_precision=bfloat16` / `float16` 按策略验证；FP16 使用 MPS GradScaler。
可训练权重精度仍由 `--trainable_dtype` 控制，`tf32` 仅影响 CUDA。

梯度累积、归一化、优化器、EMA 和调度器沿用原配置。续训保持原设备和精度；
跨设备微调使用 `--policy.path` 加载权重。推理、评估命令使用 `--device=mps`。
含 CUDA 专用算子的策略按对应 README 的设备要求运行。

### 多卡与梯度累积

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

EMA 使用策略预设，未配置时关闭；使用 `--ema.enable=true` 开启。
默认衰减参数为 `inv_gamma=1.0`、`power=0.75`、`min_decay=0`、`max_decay=0.9999`、`update_after_step=0`。
固定衰减可设 `--ema.decay=0.99`，与自定义 `min_decay/max_decay` 互斥。
`--ema.schedule=power_law` 使用 `(1-1/n)^power`，`n` 为预热后的成功更新次数；
默认 `inverse_power` 使用 `1-(1+(n-1)/inv_gamma)^(-power)`。策略可提供各自的指数与上限。

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

### 推理模式

在本地评估命令中增加：

```bash
--inference.type=rtc
```

| `--inference.type` | 行为 |
|---|---|
| `sync`（本地默认） | 执行当前动作块后再预测；TCP 块间制动保持 |
| `async` | 后台预测；ACT／DP 默认按时间切换，BSP 默认相位对齐 |
| `rtc` | 后台预测并进行去噪前缀引导；目前支持 π0.5／SmolVLA |

`async` 可用 `--inference.continuity=none` 关闭策略默认衔接；
`--inference.continuity=phase` 仅适用于 BSP。`rtc` 无需另选连续性方法。
ACT 异步执行时须关闭时间融合：`--policy.temporal_ensemble_coeff none`。

支持语言规划的策略可在本地 `evaluate` 或云端 `inference serve` 中追加
`--language-replan-interval-s=2 --task='任务指令'`。默认关闭；按观测时间更新子任务，切换时清空旧动作队列，每个回合重置。

异步模式默认根据剩余轨迹时间与预测耗时自动重规划。必要时可调整：

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--inference.replan_interval_s` | 自动（TCP 为 `0.25`） | 重规划时间预算，单位秒 |
| `--inference.prediction_lead_s` | `0.1` | 提前预测预算下限，单位秒 |
| `--inference.max_prediction_age_s` | `5` | 源观测最大年龄，单位秒 |
| `--inference.startup_timeout_s` | `15` | 首次及 TCP 同步块间预测等待上限，单位秒 |

TCP 执行及异步执行的 `--fps` 是命令查询频率，可与 checkpoint 数据 FPS 不同。

TCP 动作 checkpoint 加载后自动启用 TCP 跟踪，`sync`／`async` 共用限速和恢复；执行机须安装 `kinematics` 扩展。跟踪滞后时自动放慢，预测过期后制动保持。可选参考限制如下（默认值；速度、加速度分别为 m/s、m/s² 或 rad/s、rad/s²）：

```bash
--inference.tcp_tracking='{
  "max_tcp_linear_velocity":0.2, "max_tcp_linear_acceleration":0.5,
  "max_tcp_angular_velocity":1.0, "max_tcp_angular_acceleration":2.0
}'
```

关节位置、速度及加速度限制仍有效。IK 失败时制动保持并重新预测，超过 1 s 或连续 3 次新预测失败时停止；正常同步块间等待使用 `startup_timeout_s`。
TCP 暂不支持 `rtc`、ACT 时间融合、单个手臂关节固定覆盖及 Host 关节动作缩放。参考跟踪允许路径偏差，不含碰撞避障或 jerk 约束，尚未完成实机验证。

预测过期、有效轨迹耗尽或 Host 状态变化时停止使用旧计划；TCP 异步执行只在上述恢复预算内保持等待。
评估记录保存执行参数及源观测时间。云端服务默认 `async`，支持 `rtc`，TCP 动作 checkpoint 也可选择 `sync`；
使用相同 `--inference.*` 参数，Pi 从服务端握手读取配置。
两端必须使用相同平台版本，旧动作块协议不兼容。

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

### 云端推理

树莓派发送图像和 state 到云端 GPU，接收轨迹计划并在本地执行；PC 用于登录管理。
支持声明远程轨迹能力的原生策略，包括 ACT、Diffusion、BSP、PI05、SmolVLA 和 Spirit。
云端与树莓派使用相同版本的 Alohamini；树莓派先启动本地 Host，完成标定和升降归零，退出其他控制客户端。
Host 地址保持 `127.0.0.1`。ZeroMQ 推理端口 `5560` 通过 SSH 隧道或 WireGuard 连接，不直接开放到公网。

云端安装[策略与通信扩展](install.md#3-可选扩展)，树莓派复用运行 Host 的环境。

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

`--fps` 设置 Pi 的命令查询频率，checkpoint 数据频率用于解释轨迹时间。
密集动作可用服务端 `--chunk-size 32` 缩短可执行窗口，不能超过模型窗口；BSP 不接受该参数。
TCP 同样支持 `--chunk-size`，执行端须安装 `kinematics` 扩展。
这些参数不保证云端持续按执行频率返回结果。迟到结果跳过过期前缀，动作耗尽或 Host 会话、控制参考变化时停止，需显式重新运行。

| 树莓派 `run` 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--episode_time` | 90 秒 | 每回合运行时长 |
| `--rpc-timeout-s` | 10 秒 | 云端请求超时 |
| `--max-prediction-age-s` | 1 秒 | 预测源观测的最大年龄 |
| `--ack-timeout-s` | 0.5 秒 | Host 确认超时 |

固定当前关节时，在云端 `serve` 和树莓派 `run` 两端传入相同的 `--fixed-current FIELD ...`。
支持 `base`、`lift_axis`、`arm_left`、`arm_right` 或具名 Host 字段；树莓派从新鲜反馈捕获固定目标。
PI05／SmolVLA 可在服务端开启[RTC 引导](#推理模式)，仅提供推理期引导，不包含 RTC 训练。
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

已有 PC 环境包含查看依赖；独立分析环境安装 [`analysis` 扩展](install.md#3-可选扩展)。
重新预测还需要对应策略的运行依赖。

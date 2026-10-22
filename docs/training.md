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
训练归一化统计只使用训练回合；可自动计算或复用下方提前生成的统计文件。

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
未提供的标定与单位会记为未知，不要求生成平台 sidecar。
离线推理使用 checkpoint 保存的字段、预处理与缺失元信息声明；真机部署另按动作单位和标定检查。

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
底盘分类头可在配置的 `model` 中设置：

```json
{
  "discrete_action_dims": [14],
  "discrete_action_values": [[-0.1, 0.0, 0.1]],
  "discrete_action_class_weights": [[1.0, 1.0, 1.0]]
}
```

第 14 维为底盘 x 速度；类别值按实际示教速度填写。分类输出会恢复为物理速度。

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

支持单设备、DDP 和训练器 EMA，暂不支持 FSDP2。
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
checkpoint 自带 tokenizer 和骨干配置。此入口不支持纯视觉、RTC、PEFT 或 ACT 时间融合。

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

训练器优先使用已安装的 TorchCodec，否则使用 PyAV。
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
ACT／AM-ACT 默认无调度，SmolVLA 使用 `cosine`，π0.5 使用 `warmup_cosine`，
Diffusion／FastWAM 使用 `diffusers_cosine`；
`--scheduler.type=none` 禁用调度。可覆盖 `warmup_steps`、`decay_steps`、`decay_lr`。

| 配置 | 追加参数 |
| --- | --- |
| 单卡 | 默认 |
| DDP，每卡完整模型 | `--num_processes=2 --background` |
| ACT／AM-ACT FSDP2 分片 | `--distributed_backend=fsdp2 --num_processes=2 --background` |

有效 batch 通常为 `batch_size × 进程数 × 累积次数`；`steps` 按成功的优化器更新计数。
多进程尾批默认重复样本补齐，`--drop_last=true` 改为丢弃。
多节点使用 `torchrun ... -m alohamini.learning.train --config train.json`，各节点须共享输出目录并使用相同数据和资产路径。

FSDP2 需要每个进程有足够 CPU 内存构建模型，保存时同时导出分片训练状态和完整推理权重。
当前仅 ACT／AM-ACT 支持 FSDP2；不支持 ZeRO 或张量并行。

### 训练器 EMA

EMA 默认关闭，使用 `--ema.enable=true` 开启。
默认衰减参数为 `inv_gamma=1.0`、`power=0.75`、`min_decay=0`、`max_decay=0.9999`、`update_after_step=0`。
固定衰减可设 `--ema.decay=0.99`，与自定义 `min_decay/max_decay` 互斥。

EMA 以 FP32 追踪可训练参数，每次成功的优化器更新后推进；冻结参数不复制，BatchNorm buffer 直接复制。
验证默认使用 EMA 权重，`--ema.use_for_eval=false` 改用在线权重。
支持单设备与 DDP，不支持 FSDP2；需额外的参数存储及评估／导出临时内存。

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
| `training_state/state.pt` 中的 `ema` | FP32 影子参数、复制的 buffer、衰减配置和更新计数 |

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
六种算法均支持从平台 checkpoint 微调；下面以 ACT／AM-ACT 为例：

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

SmolVLA／π0.5／FastWAM 评估须增加 `--task "拿起物体"`。Diffusion／FastWAM 不使用 ACT 时间融合。
`--dataset eval_01 --task "拿起物体"` 保存评估回合。Ctrl+C 结束。持续反馈中断或关节保护时暂停，恢复后清空旧策略缓存并继续；Host 重启、标定变化或其他客户端接管时结束评估。
推理耗时限制实际执行频率。

跨机器使用时复制完整 checkpoint 目录，勿只复制符号链接。
自定义策略接口见 [客户端接口](host-protocol.md#python-策略评估)，示例为 `examples/learning/custom_policy.py`。
Notebook 入口见 [Notebook](notebook.md)，LeRobot 权重使用见 [LeRobot](lerobot.md)。

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

## 8. 对比评估与训练示范

`alohamini dataset analyze` 在本地查看已保存的评估记录，并与评估所用 checkpoint 的训练或验证示范对比。无需连接机器人。

| 功能 | 支持算法 |
| --- | --- |
| Rerun 画面与动作曲线、示范对比、报告导出 | 平台各算法，包括 ACT、AM-ACT、Diffusion Policy、FastWAM、SmolVLA、π0.5 |
| 加 `--probe-samples`，重新预测示范动作并计算误差 | 目前仅 ACT／AM-ACT |

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

### 检查模型在示范上的预测（仅 ACT／AM-ACT）

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

报告逐个动作维度显示首步预测和整个动作块的平均绝对误差（MAE），单位与示范动作一致。
预测使用示范中的观测，对照的是**同一条示范的后续真实动作**，不是评估回合的动作。
预测数据保存为 `demonstration_predictions.npz`。这项误差用于检查模型预测，不代表真机成功率。
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

# π0.5

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

命令在仓库根目录执行。示例中的 episode 1 用于验证；数据只有一个回合时，省略 `--dataset.eval_episodes` 和 `--eval_steps`。

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

归一化统计只使用训练集，先计算相对动作，再统计 q01/q99（1%/99% 分位数）。
范围外的数值不裁剪，q01/q99 相同的维度不置零；稀少但有效的动作仍参与训练。
分母默认采用 `q99-q01+1e-6`。窄范围会放大数值；可用 `--policy.span_floors` 设置跨度下限。
参数为 JSON 对象，`state`、`actions` 分别接收按保存字段顺序排列的 18 个非负物理跨度。
默认不设跨度下限；设置随 checkpoint 保存，归一化与还原使用同一跨度。

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

四份配置见 [`examples/learning/pi05/`](../../../../examples/learning/pi05)。`ki + b` 不支持：冻结整个 VLM 不属于双目标 KI。
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

默认模型类型为 BF16，部分模块保留 FP32；平台不额外开启 autocast（`mixed_precision=none`）。
需要 FP32 可训练参数与 BF16 计算时，追加 `--trainable_dtype=float32 --mixed_precision=bfloat16`；
其余公共选项见[数值精度](../../../../docs/training.md#数值精度)。
默认完整梯度检查点、30000 步、AdamW `2.5e-5`、betas `(0.9,0.95)`、weight decay `1e-10`、梯度裁剪 `1`、1000 步预热及余弦衰减至 `2.5e-6`；每 100 步日志、1000 步保存。EMA 默认关闭。
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

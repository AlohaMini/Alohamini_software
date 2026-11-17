# DM0.5

[策略索引](../../../../docs/training.md#策略选择)

准备原始 OpenDM 基座，包含权重、`config.json`、Gemma3 图像处理器、tokenizer 与 chat template：

```bash
hf download Dexmal/DM05 --revision=5cd18734814abb075a9ccfd9ad6d16777b5cf10e \
  --local-dir="$HOME/Alohamini_workspace/pretrained/dm05"

alohamini train --policy.type=dm05 \
  --policy.path="$HOME/Alohamini_workspace/pretrained/dm05" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/dm05_01" --background
```

默认使用具名的记录坐标；action 补齐至基座的 32 维，填充维度和回合尾部不计入损失。
state 在补零前离散为语言 token。图像先居中补为正方形，再由基座处理器缩放。
发布基座为 34 层 Gemma3、34 层动作专家，两者层数必须匹配。

`--policy.normalization_recipe=recorded`（默认）使用训练集 q01/q99 与 `1e-6`，不裁剪、不置零退化维度。
`opendm` 使用原文裁剪及零分位数维度归零规则，会改变范围外及稀少动作的监督值。
归一化方案、字段与相机顺序随 checkpoint 保存。

默认 BF16 模型、平台 `mixed_precision=none`，开启 VLM 与动作专家重计算；
需要 FP32 可训练参数与 BF16 计算时追加 `--trainable_dtype=float32 --mixed_precision=bfloat16`。
动作块默认 50 步，推理积分 10 步，可用 `n_action_steps`、`diffusion_steps` 调整。
当前为动作 flow 微调，不包含语言交叉熵、MEM 历史模型、LoRA 或 RTC。

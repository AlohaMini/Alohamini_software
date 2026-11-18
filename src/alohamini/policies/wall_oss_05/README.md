# WALL-OSS-0.5

[策略索引](../../../../docs/training.md#策略选择)

准备包含转换权重、Qwen tokenizer、图像处理器和 `config.json` 的本地基座：

```bash
hf download lerobot/wall-oss-0.5 --revision=03ab102b8c258a5333a8313f38b17c52e49eb467 \
  --local-dir="$HOME/Alohamini_workspace/pretrained/wall_oss_05"

alohamini train --policy.type=wall_oss_05 \
  --policy.path="$HOME/Alohamini_workspace/pretrained/wall_oss_05" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/wall_oss_05_01" --background
```

默认用数据集具名 state、action，分别归一化后放入 26 维槽位，填充维度与回合尾部不计入损失。
字段顺序、单位、槽位和相机标签随 checkpoint 保存；相机按选择顺序全部送入模型。
默认重置动作输入、输出头以适配新机器人，其余权重继承基座。
复用已有机器人头时，设置 `--policy.reinitialize_robot_heads=false`，并显式提供
`--policy.state_slots='{"字段名":槽位}'`、`--policy.action_slots='{"字段名":槽位}'`。
映射只决定位置；数据坐标、单位和控制语义须与保留的机器人头一致。

`--policy.normalization_recipe=recorded`（默认）使用 q01/q99 加 `1e-6`，保留范围外动作；
`wall` 使用原文分位数边界裁剪。图像长边默认 448，保持比例，再由本地 Qwen 处理器生成视觉 token。
state 离散为 512 档，仅影响输入语言 token。

基座的 VLM 和动作专家各 36 层，视觉编码器 32 层。默认 BF16 参数、FP32 动作头和 RMSNorm，
开启视觉及联合 Transformer 重计算。FP32 可训练参数与 BF16 计算可使用
`--trainable_dtype=float32 --mixed_precision=bfloat16`。
默认动作块 32 步、积分 10 步，支持 `n_action_steps` 和 `num_inference_timesteps`。
当前提供机器人 flow 微调和动作推理，不包含语言监督、DMuon、专用 FlashMask 内核或 RTC。

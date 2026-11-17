# EVO1

[策略索引](../../../../docs/training.md#策略选择)

基座目录需包含转换后的 `config.json`、`model.safetensors`（或分片与索引），
以及 `assets/` 下的 InternVL3 配置和 tokenizer。

```bash
alohamini train --policy.type=evo1 \
  --policy.path="$HOME/Alohamini_workspace/pretrained/evo1" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --policy.training_stage=stage1 --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/evo1_01" --background
```

`stage1` 冻结 VLM，训练动作网络；`stage2` 默认同时训练 VLM 和动作网络。
以 stage1 checkpoint 开始 stage2 时，改用 `--policy.path=/path/to/checkpoint --policy.training_stage=stage2`；
继续相同阶段使用公共 `--resume` 流程。显式指定的冻结开关优先于阶段默认值。
默认 14 层语言网络、8 层动作 Transformer，预测 50 步、积分 32 步。
state/action 使用训练集 MIN_MAX，补齐至 24 维；不自动转换夹爪符号或二值化。
使用平台混合精度和 SDPA；暂不支持 RTC。

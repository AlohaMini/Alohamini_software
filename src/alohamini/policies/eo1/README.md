# EO-1

[策略索引](../../../../docs/training.md#策略选择)

基座目录需包含转换后的 `config.json`、`model.safetensors`（或分片与索引），
以及 `assets/` 下的 Qwen2.5-VL processor/tokenizer。配置须保存 `vlm_config`。

```bash
alohamini train --policy.type=eo1 \
  --policy.path="$HOME/Alohamini_workspace/pretrained/eo1" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/eo1_01" --background
```

默认使用机器人数值监督的 flow 微调，不包含 CoT 或文本答案训练。
相机按所选顺序送入 Qwen2.5-VL 图像处理器；像素预算和 backbone 层数随配置保存。
state/action 使用训练集均值标准差，补齐至 32 维；默认预测 16 步动作。
可训练参数默认 FP32 存储、BF16 混合精度计算。暂不支持 RTC。

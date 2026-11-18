# X-VLA

[策略索引](../../../../docs/training.md#策略选择)

基座目录需包含转换后的 `config.json`、`model.safetensors`（或分片与索引），
以及 `assets/` 下的 Florence 图像处理器和 tokenizer；配置须保存 `florence_config`。

```bash
alohamini train --policy.type=xvla \
  --policy.path="$HOME/Alohamini_workspace/pretrained/xvla" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/xvla_01" --background
```

当前支持记录坐标的 `action_mode=auto`：state/action 使用训练集均值标准差，补齐至 20 维，
动作使用 MSE。不按维度猜测 EE6D、夹爪或机器人类型；暂不提供论文 EE6D 协议或 RTC。
相机顺序、数值字段与处理器随 checkpoint 保存。
默认 24 层动作 Transformer、32 个 soft prompts、30 步动作块；Florence 层数来自基座配置。
`--policy.default_domain_id` 显式选择 soft-prompt 域，默认 0。

默认前 1000 次优化器更新只让 prompts、动作投影使用非零学习率，之后所有可训练组使用恒定学习率。
`--scheduler.start_step` 可调整阶段边界；
`--scheduler.type=staged_cosine --scheduler.warmup_steps=2000` 启用后续预热和余弦调度。
`--policy.learning_coef` 同时缩放 VLM 与 prompts 的学习率，默认 1。
`freeze_vision_encoder`、`freeze_language_encoder`、`train_policy_transformer`、`train_soft_prompts` 可显式控制参数范围。

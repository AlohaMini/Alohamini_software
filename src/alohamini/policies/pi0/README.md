# π0

[策略索引](../../../../docs/training.md#策略选择)

准备转换后的 OpenPI PyTorch `model.safetensors` 和本地 PaliGemma SentencePiece 文件。

```bash
alohamini train --policy.type=pi0 \
  --policy.path="$HOME/Alohamini_workspace/pretrained/pi0" \
  --policy.tokenizer_path="$HOME/Alohamini_workspace/pretrained/pi05_tokenizers/paligemma_tokenizer.model" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --policy.finetuning_mode=b --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/pi0_01" --background
```

要求具名的 18 维原始 state/action，使用均值标准差归一化；12 个机械臂关节相对当前 state，
夹爪、底盘和升降保持绝对命令。连续 state 送入动作专家；语言不含离散 state。
归一化采用 `(x-mean)/(std+1e-6)`，先计算相对动作，再拟合训练集统计；不裁剪、不按分位数过滤样本。
π0 不使用 π0.5 的 q01/q99 或 `span_floors`。
默认预测、执行 50 步，flow 推理 10 步；执行步数可用 `--policy.n_action_steps` 调小。
微调范围通过 `--policy.finetuning_mode=full/a/b/c/d` 选择，含义见 [π0.5](../pi05/README.md)。
支持平台 RTC，不包含 KI 或 FAST token 损失。
默认模型类型为 BF16，部分模块保留 FP32，平台 `mixed_precision=none`；
公共配置见[数值精度](../../../../docs/training.md#数值精度)。

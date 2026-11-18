# π0-FAST

[策略索引](../../../../docs/training.md#策略选择)

准备本地 `lerobot/pi0fast-base` PyTorch 权重、PaliGemma SentencePiece 文件和
FAST 的 `tokenizer.json`、`processor_config.json`。不执行 tokenizer 仓库的远程 Python 代码。

```bash
alohamini train --policy.type=pi0_fast \
  --policy.path="$HOME/Alohamini_workspace/pretrained/pi0fast-base" \
  --policy.tokenizer_path="$HOME/Alohamini_workspace/pretrained/pi05_tokenizers/paligemma_tokenizer.model" \
  --policy.fast_tokenizer_path="$HOME/Alohamini_workspace/pretrained/pi05_tokenizers/fast" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/pi0_fast_01" --background
```

默认基座键前缀为 `model.paligemma_with_expert.`，可用 `--policy.base_weights_prefix` 显式覆盖。
原生平台 checkpoint 的加载与续训不需要重设前缀。
要求具名的 18 维原始 state/action；使用分位数归一化，机械臂关节相对当前 state。
动作经过 DCT/BPE 转为 token，以交叉熵训练；模型不包含 flow 动作专家。
默认预测、执行 50 步。任务/state 超过 `--policy.max_token_len` 或动作编码超过
`--policy.fast_max_tokens` 时会报错，不能截断动作 token。
推理使用贪心 KV-cache 解码；不完整或非法动作会报错。暂不支持 RTC。
训练默认验证 token 交叉熵；需要训练结束后自动计算动作 MAE 时，
设置 `--policy.evaluate_action_predictions=true`，模型须能生成完整、维度正确的动作。

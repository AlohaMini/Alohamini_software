# π0-FAST

[Policy index](../../../../docs/training.md#policy-selection)

Prepare local `lerobot/pi0fast-base` PyTorch weights, a PaliGemma SentencePiece file, and
FAST `tokenizer.json` / `processor_config.json`. Remote Python code from tokenizer repositories is not executed.

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

The base-weight key prefix defaults to `model.paligemma_with_expert.`; explicitly override with `--policy.base_weights_prefix`.
Native platform checkpoint loading/resume requires no prefix reset.
Requires named original 18-dimensional state/action, using quantile normalization with arm joints relative to current state.
Actions become DCT/BPE tokens for cross-entropy training; the model has no flow action expert.
Defaults predict/execute 50 steps. Task/state exceeding `--policy.max_token_len` or encoded actions exceeding
`--policy.fast_max_tokens` cause an error; action tokens cannot be truncated.
Inference uses greedy KV-cache decoding; incomplete or invalid actions cause errors. RTC is currently unsupported.
Validation defaults to token cross-entropy. For automatic post-training action MAE,
set `--policy.evaluate_action_predictions=true`; the model must generate complete, dimensionally valid actions.

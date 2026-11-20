# π0

[Policy index](../../../../docs/training.md#policy-selection)

Prepare converted OpenPI PyTorch `model.safetensors` and a local PaliGemma SentencePiece file.

```bash
alohamini train --policy.type=pi0 \
  --policy.path="$HOME/Alohamini_workspace/pretrained/pi0" \
  --policy.tokenizer_path="$HOME/Alohamini_workspace/pretrained/pi05_tokenizers/paligemma_tokenizer.model" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --policy.finetuning_mode=b --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/pi0_01" --background
```

Requires named, original 18-dimensional state/action with mean/std normalization. The 12 arm joints are relative to current state;
grippers, base, and lift retain absolute commands. Continuous state feeds the action expert; language contains no discretized state.
Normalization uses `(x-mean)/(std+1e-6)`, computing relative actions before fitting training statistics, without clipping or quantile-based sample filtering.
π0 uses neither π0.5 q01/q99 nor `span_floors`.
Defaults predict/execute 50 steps with 10 flow-inference steps; reduce execution using `--policy.n_action_steps`.
Select fine-tuning scope with `--policy.finetuning_mode=full/a/b/c/d`; see [π0.5](../pi05/README.md) for meanings.
Supports platform RTC, excluding KI and FAST-token loss.
The model defaults to BF16 with some modules retained in FP32, using platform `mixed_precision=none`;
see [Numeric precision](../../../../docs/training.md#numeric-precision) for common settings.

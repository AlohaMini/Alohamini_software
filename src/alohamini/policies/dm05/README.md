# DM0.5

[Policy index](../../../../docs/training.md#policy-selection)

Prepare the original OpenDM base, including weights, `config.json`, Gemma3 image processor, tokenizer, and chat template:

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

Defaults use named recorded coordinates. Actions pad to the base's 32 dimensions; padding dimensions and episode tails are excluded from loss.
State is discretized into language tokens before zero-padding. Images are center-padded to square before resizing by the base processor.
The released base has 34 Gemma3 layers and 34 action-expert layers; layer counts must match.

`--policy.normalization_recipe=recorded` (default) uses training-set q01/q99 with `1e-6`, without clipping or zeroing degenerate dimensions.
`opendm` follows the original clipping and zero-quantile-dimension rules, changing supervision for out-of-range and rare actions.
Normalization recipe, fields, and camera order are saved with checkpoints.

Defaults use a BF16 model and platform `mixed_precision=none`, with activation checkpointing for the VLM and action expert;
for FP32 trainable parameters and BF16 computation, add `--trainable_dtype=float32 --mixed_precision=bfloat16`.
Defaults are 50 action steps and 10 integration steps; adjust with `n_action_steps` and `diffusion_steps`.
This is action-flow fine-tuning, without language cross-entropy, MEM history models, LoRA, or RTC.

# EVO1

[Policy index](../../../../docs/training.md#policy-selection)

The base directory needs converted `config.json`, `model.safetensors` (or shards and index),
and InternVL3 configuration/tokenizer under `assets/`.

```bash
alohamini train --policy.type=evo1 \
  --policy.path="$HOME/Alohamini_workspace/pretrained/evo1" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --policy.training_stage=stage1 --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/evo1_01" --background
```

`stage1` freezes the VLM and trains the action network; `stage2` defaults to training both.
To start stage2 from a stage1 checkpoint, use `--policy.path=/path/to/checkpoint --policy.training_stage=stage2`;
resume the same stage with the common `--resume` workflow. Explicit freezing flags override stage defaults.
Defaults: 14 language layers, 8 action-Transformer layers, 50 predicted steps, and 32 integration steps.
State/action uses training-set MIN_MAX and pads to 24 dimensions; gripper signs and binarization are not converted automatically.
Uses platform mixed precision and SDPA; RTC is currently unsupported.

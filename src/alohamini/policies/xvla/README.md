# X-VLA

[Policy index](../../../../docs/training.md#policy-selection)

The base directory needs converted `config.json`, `model.safetensors` (or shards and index),
plus Florence image processor/tokenizer under `assets/`; configuration must retain `florence_config`.

```bash
alohamini train --policy.type=xvla \
  --policy.path="$HOME/Alohamini_workspace/pretrained/xvla" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/xvla_01" --background
```

Currently supports `action_mode=auto` in recorded coordinates: state/action uses training-set mean/std and pads to 20 dimensions,
with MSE action loss. EE6D, gripper, and robot types are not inferred from dimensions; the paper's EE6D protocol and RTC are currently unavailable.
Camera order, numeric fields, and processor are saved with checkpoints.
Defaults: 24 action-Transformer layers, 32 soft prompts, and 30-step action chunks; Florence layer count follows base configuration.
`--policy.default_domain_id` explicitly selects the soft-prompt domain, defaulting to 0.

By default, only prompts/action projections have nonzero learning rates for the first 1000 optimizer updates; all trainable groups then use constant learning rates.
Adjust the stage boundary with `--scheduler.start_step`;
`--scheduler.type=staged_cosine --scheduler.warmup_steps=2000` enables subsequent warmup and cosine scheduling.
`--policy.learning_coef` scales both VLM and prompt learning rates, defaulting to 1.
`freeze_vision_encoder`, `freeze_language_encoder`, `train_policy_transformer`, and `train_soft_prompts` explicitly control parameter scope.

# WALL-OSS-0.5

[Policy index](../../../../docs/training.md#policy-selection)

Prepare a local base with converted weights, Qwen tokenizer, image processor, and `config.json`:

```bash
hf download lerobot/wall-oss-0.5 --revision=03ab102b8c258a5333a8313f38b17c52e49eb467 \
  --local-dir="$HOME/Alohamini_workspace/pretrained/wall_oss_05"

alohamini train --policy.type=wall_oss_05 \
  --policy.path="$HOME/Alohamini_workspace/pretrained/wall_oss_05" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/wall_oss_05_01" --background
```

Defaults use named dataset state/action, separately normalized into 26-dimensional slots. Padding dimensions and episode tails are excluded from loss.
Field order, units, slots, and camera labels are saved with checkpoints. All cameras feed the model in selection order.
Action input/output heads reset for new robots by default; other weights inherit the base.
To reuse existing robot heads, set `--policy.reinitialize_robot_heads=false` and explicitly provide
`--policy.state_slots='{"FIELD_NAME":SLOT}'` and `--policy.action_slots='{"FIELD_NAME":SLOT}'`.
Mappings determine positions only; data coordinates, units, and control semantics must match retained robot heads.

`--policy.normalization_recipe=recorded` (default) uses q01/q99 plus `1e-6`, retaining out-of-range actions;
`wall` uses the original quantile-boundary clipping. Images default to long side 448 with aspect ratio preserved, then become visual tokens through the local Qwen processor.
State discretizes into 512 bins, affecting input language tokens only.

The base VLM and action expert have 36 layers each; the vision encoder has 32. Defaults use BF16 parameters, FP32 action heads and RMSNorm,
with vision/joint-Transformer activation checkpointing. For FP32 trainable parameters and BF16 computation, use
`--trainable_dtype=float32 --mixed_precision=bfloat16`.
Defaults: 32-step action chunks and 10 integration steps; supports `n_action_steps` and `num_inference_timesteps`.
Currently provides robot flow fine-tuning and action inference, excluding language supervision, DMuon, dedicated FlashMask kernels, and RTC.

# LingBot-VA

[Policy index](../../../../docs/training.md#policy-selection)

Download local base assets; the directory must include `transformer/`, `vae/`, `text_encoder/`, and `tokenizer/`:

```bash
BASE="$HOME/Alohamini_workspace/pretrained/lingbot_va"
hf download robbyant/lingbot-va-base --revision=68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c \
  --local-dir="$BASE"

alohamini train --policy.type=lingbot_va --policy.path="$BASE" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/lingbot_va_01" --background
```

The shared Transformer defaults to 30 layers; actual architecture loads from base `transformer/config.json`.
Defaults: FP32 trainable parameters, BF16 mixed precision, frozen encoders, activation checkpointing, and GPU Flex Attention;
inference automatically uses SDPA without editing base configuration.
For FP32 debugging, set `--policy.dtype=float32 --mixed_precision=none`.
For multiple GPUs and parameter sharding, use [common training settings](../../../../docs/training.md#multiple-gpus-and-gradient-accumulation).

By default, VAE latents from each 128×128 view are concatenated horizontally in selected-camera order.
For a three-view T layout, explicitly set `--policy.camera_layout=robotwin_tshape` and
declare head, left-wrist, and right-wrist order with `--policy.obs_cam_keys='["forward","wrist_left","wrist_right"]'`.
State is not a model input but remains available for common field transforms. Actions train with recorded names/units,
up to 30 coordinates; robot action input/output heads reset by default.
Preserving the same robot head requires `reinitialize_robot_heads=false` and a complete named `action_slots` mapping.

Actions use training-set q01/q99 with `1e-6` added to the denominator; out-of-range actions remain by default.
`--policy.normalization_recipe=lingbot` limits normalized training targets to `[-1.5,1.5]`.
Defaults are `frame_chunk_size=4` and `action_per_frame=4`, producing 16 future actions per chunk;
training needs complete 17-frame windows and excludes the last 16 starts per episode. The initial conditioning frame is separate; the zero-action prefix is conditioning only.
Video/action defaults are 20/50 denoising steps, adjustable with `num_inference_steps` and `action_num_inference_steps`.

At first training, the main process encodes fixed task text on CPU and caches it in the workspace's `pretrained/frozen_features/`;
the cache is saved with checkpoints, so ranks need not reload UMT5 individually.
New instructions still need the local text encoder; VAE and tokenizer must also remain at `wan_pretrained_path`.
`--policy.precompute_text_features=false` disables precomputation.

Supports standalone offline action chunks, single-environment synchronous inference, and per-frame observation feedback. Remote action chunks, asynchronous execution, RTC, and LoRA are currently unsupported.

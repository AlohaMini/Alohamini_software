# Cosmos3 Nano / Edge

[Policy index](../../../../docs/training.md#policy-selection)

Nano uses Qwen3-VL MoT with 36 layers by default; Edge uses Nemotron MoT with 28.
Network and VAE configurations load from local base assets without automatic layer truncation.

```bash
BASE="$HOME/Alohamini_workspace/pretrained/cosmos3_nano"
hf download lerobot/Cosmos3-Nano --revision=104d12ccc544e14984443b812233969f389ee8dc --local-dir="$BASE"
hf download nvidia/Cosmos3-Nano --include 'tokenizer*' '*vocab*' '*merges*' '*special_tokens*' '*.jinja' \
  --local-dir="$BASE/tokenizer"

alohamini train --policy.type=cosmos3_nano --policy.path="$BASE" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/cosmos3_01" --background
```

For Edge, change the policy to `cosmos3_edge`, the base to `lerobot/Cosmos3-Edge`
(revision `35d07f7f5ebedc020125dd06e87eee47109276ed`), and the tokenizer to `nvidia/Cosmos3-Edge`.
Runtime reads only local weights/tokenizers; tokenizers are saved with checkpoints.

Defaults use FP32 trainable parameters and AdamW states, BF16 mixed precision, and frozen modules; time embeddings and RoPE remain FP32.
Only the generation branch and video/action projections are trained; the text-understanding branch and vision tower are frozen, with activation checkpointing enabled.
The action parameter group uses 5 times the common learning rate; adjust with `optimizer_action_lr_multiplier`.
Defaults use linear warmup/decay and power-law EMA; disable EMA with `--ema.enable=false`.
For sharded training, use the [common FSDP2 configuration](../../../../docs/training.md#multiple-gpus-and-gradient-accumulation).

Supports 1–3 cameras in selection order, using a single view, side-by-side pair, or one upper/two lower views.
Explicitly order with `camera_keys`, declare views with `camera_roles`, or supply `view_description`; wrist-camera roles are not inferred.
Each view defaults to 256×256, then follows native canvas assembly/padding rules before Wan VAE encoding.
Each chunk predicts and executes 16 steps by default. Training uses complete current-plus-16-frame windows, excluding the final 16 starts per episode.
Inference uses only the current frame, with 30 denoising steps by default; adjust using `--policy.num_inference_steps`.

Actions use recorded names/units and q01/q99 normalization, without default clipping outside bounds; up to 64 coordinates.
Robot action heads reset by default. To retain the same robot head, set `reinitialize_robot_heads=false`,
provide complete `pretrained_action_names`, and explicitly select the corresponding `domain_id`.
State is disabled by default. With `use_state=true`, state must match action coordinates, order, and units;
explicitly declare `state_action_names` and set STATE normalization to the ACTION mode. Both share action statistics.
LIBERO relative-end-effector/rot6d and gripper conversions are not automatic; use common `data` settings for field representations.

The idle-frame count is currently fixed at 0. The original `quantile_rot` protocol, inference text-KV reuse, LoRA, and RTC are not included.
`compile_model` defaults to disabled; distributed compilation and full-base training have not been validated.

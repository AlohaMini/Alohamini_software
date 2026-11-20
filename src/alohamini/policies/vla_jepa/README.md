# VLA-JEPA

[Policy index](../../../../docs/training.md#policy-selection)

Prepare policy weights and local processors:

```bash
BASE="$HOME/Alohamini_workspace/pretrained/vla_jepa"
hf download lerobot/VLA-JEPA-Pretrain --revision=e946c3e5b538d760f4b4ff239d1b1c12090c041d --local-dir="$BASE"
hf download Qwen/Qwen3-VL-2B-Instruct --revision=89644892e4d85e24eaac8bacfd4f463576704203 \
  --include '*.json' '*.txt' '*.jinja' --local-dir="$BASE/qwen"
hf download facebook/vjepa2-vitl-fpc64-256 --revision=b3c1679b7c34d3255ef3547f27c7b226aefab26f \
  --include '*.json' --local-dir="$BASE/vjepa2"

alohamini train --policy.type=vla_jepa --policy.path="$BASE" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --policy.world_model_cameras='["forward","wrist_left"]' \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/vla_jepa_01" --background
```

Current frames from all selected cameras feed Qwen; `world_model_cameras` explicitly selects views for world-model supervision.
The released base's world model uses two views. Changing the count requires `--policy.reinitialize_world_heads=true`
to reset world-model input/output projections; cameras are not duplicated or discarded automatically.
Defaults: 7-step action chunks, 8-frame video windows, and 4 integration steps. Online action inference needs only current images.

State uses training-set MEAN_STD; actions use MIN_MAX. Fields train with recorded names, units, and representations;
inference outputs default to normalized `[-1,1]` clipping; disable with `--policy.clip_normalized_actions=false`.
Action input/output and state heads reset by default, without automatic end-effector, relative-action, or binary-gripper conversion.
Set `--policy.reinitialize_robot_heads=false` to retain heads for the same robot.
Field transforms use common `data` configuration, saved with checkpoints.

`--policy.world_model_context=causal` (default) encodes teacher context by prefix to avoid reading later frames;
`shared` uses the original full-video shared encoding. The frozen V-JEPA2 teacher stays in eval mode; episode-tail padding is excluded from world-model loss.
Disable the world model and train only the action head:

```bash
--policy.enable_world_model=false --policy.freeze_qwen=true
```

Defaults use BF16 Qwen, FP32 action DiT, world-model predictor, and frozen teacher. Common mixed precision is BF16, while the action branch computes in FP32.
Activation checkpointing defaults to enabled for Qwen, action DiT, and world-model predictor. Actual Qwen layers follow saved `qwen_config`;
action DiT defaults to 16 layers and predictor to 12; adjust with `action_num_layers` and `predictor_depth`.
Learning rates scale common `--optimizer.lr` by `1/3`, `10/3`, and `50/3` for Qwen, action head, and predictor, respectively;
override with `--policy.lr_scales`.

To train from separate backbone bases, download full Qwen and V-JEPA2 weights into the same subdirectories and set
`--policy.base_weights_mode=backbones`. Language cross-entropy, LoRA, RTC, and dedicated FlashAttention kernels are not currently included.

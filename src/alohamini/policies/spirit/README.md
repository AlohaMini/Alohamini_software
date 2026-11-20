# Spirit v1.5

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Example episode 1 is reserved for validation; for a single episode, omit `--dataset.eval_episodes` and `--eval_steps`.

Use local Spirit base weights and Qwen configuration, tokenizer, and image-processing assets:

```bash
hf download Spirit-AI-robotics/Spirit-v1.5 \
  --revision 9886fa7552de70b097c4214b7e40e66ff937a6fd \
  --local-dir "$HOME/Alohamini_workspace/pretrained/spirit-v1.5"
python -m alohamini.policies.spirit.assets \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --output "$HOME/Alohamini_workspace/pretrained/spirit-qwen-assets"
```

Spirit base weights already include Qwen. This resource command downloads only configuration/processor files and records revisions/checksums;
the output directory must be empty. Training and checkpoint loading read only local assets.

Select data representations using [Joint, TCP, and relative targets](../../../../docs/training.md#joint-tcp-and-relative-targets). These commands apply to standard neutral-pose calibration data:

```bash
alohamini train \
  --policy.type=spirit \
  --policy.path="$HOME/Alohamini_workspace/pretrained/spirit-v1.5" \
  --policy.backbone_path="$HOME/Alohamini_workspace/pretrained/spirit-qwen-assets" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --state=joint_position,base_velocity,lift_height \
  --data.state_representation=tcp \
  --data.action_representation=tcp_delta \
  --dataset.eval_episodes='[1]' \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --distributed_backend=fsdp2 --num_processes=4 \
  --mixed_precision=bfloat16 --batch_size=1 --gradient_accumulation_steps=8 \
  --steps=40000 --save_freq=5000 --log_freq=100 \
  --output_dir="$HOME/Alohamini_workspace/runs/spirit_01" --background
```

Adjust `num_processes`, batch size, and accumulation to available GPUs and memory.
Defaults use a 16-layer, width-1536, 32-head DiT alternating self/cross-attention, generating 60 action steps with 10 Euler sampling steps at inference.
Training uses flow matching, full-parameter AdamW `2.5e-5`, betas `(0.9,0.95)`, weight decay `1e-10`,
gradient clipping `1`, 1000-step warmup, and a scheduler defaulting to decay to `2.5e-6` by step 50000.
The training preset uses 40000 steps, batch=32 per GPU, and BF16; the example explicitly uses a smaller batch with accumulation.
Activation checkpointing defaults to enabled; disable with `--policy.gradient_checkpointing=false`.

| Data representation | Input and execution |
| --- | --- |
| state=`tcp`, action=`tcp_delta` (default) | Shared data layer converts joint state/action; predicts poses relative to current TCP, restoring Host targets through IK before execution |
| state=`recorded`, action=`recorded` | Preserve selected state and original action; no kinematics configuration needed |

To retain recorded representations, change both representation arguments above to `recorded`.
Spirit requires numeric state and nonempty task text; `--state=none` is unsupported.
Original action names/order must match the transform description. Selected state is recorded separately; the reader supplies complete joint observations required for FK.
Base/lift actions remain complete; these DOFs are not fixed automatically.
Transformed state/action each supports up to 32 dimensions; original joint counts and derived TCP dimensions are recorded separately.
Padding channels and episode-end padded targets are excluded from action loss.
Normalization computes p1/p99 separately on transformed training state/action and maps to `[-1,1]`, without reusing the base robot's statistics.

Select up to three RGB views; mappings must cover all selected cameras:

| Recorded camera | Spirit slot |
| --- | --- |
| `forward` | `cam_high` |
| `wrist_left` | `cam_left_wrist` |
| `wrist_right` | `cam_right_wrist` |

Explicitly map other names with `--policy.camera_mapping='{"cam_high":"top","cam_left_wrist":"left","cam_right_wrist":"right"}'`
and select matching views with `--cameras`. Unused slots receive black images.
For dual-front EDU, select one front view plus right wrist, such as `--cameras='["forward_left","wrist_right"]'`
with `--policy.camera_mapping='{"cam_high":"forward_left","cam_right_wrist":"wrist_right"}'`.
Using both front cameras requires separate view-slot adaptation; do not label the second front camera as left wrist.
Images first load as RGB 240×320, then pass through the Qwen processor. Training uses ColorJitter; validation/inference disables augmentation.
Each row's `task` supplies task text. The robot prompt type defaults to `aloha`; override with `--policy.robot_type`.

The preset pins the reference implementation to [6d67377](https://github.com/Spirit-AI-Team/spirit-v1.5/tree/6d67377f16f67d6575447a1f29fdd83a2e0099d9),
and base weights to [9886fa7](https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5/tree/9886fa7552de70b097c4214b7e40e66ff937a6fd).
Reference Franka data uses previous-frame end-effector state and future measured poses. The platform preserves recorded observation/action pairing and data FPS,
without resampling timestamps, using the shared training scheduler and full-window training statistics. These protocol differences, explicit overrides, and geometry snapshots are saved in checkpoints.
See the official [Finetune Guide](https://github.com/Spirit-AI-Team/spirit-v1.5#finetune-guide).

Native checkpoints include the complete model plus Qwen configuration, tokenizer, and image-processing assets, loading independently of the original resource directory.
`preprocessing.json` lists names/units of original execution fields and derived statistical fields, plus transform order.
Supports single-device, DDP, FSDP2, offline evaluation, and cloud action-chunk inference. Spirit does not support RTC prefix guidance.

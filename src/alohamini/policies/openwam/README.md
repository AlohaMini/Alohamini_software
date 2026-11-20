# OpenWAM-α

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Example episode 1 is reserved for validation; for a single episode, omit `--dataset.eval_episodes` and `--eval_steps`.

```bash
hf download OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model \
  --revision 52df4e66c82c5c8b480adcc8d01f4db7415dfb56 \
  --local-dir "$HOME/Alohamini_workspace/pretrained/openwam"

alohamini train --policy.type=openwam \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --policy.path="$HOME/Alohamini_workspace/pretrained/openwam" \
  --run_name=openwam_task --batch_size=1 --steps=30000 \
  --keep_last_checkpoints=1 --background
```

The base needs `checkpoint_step_*.safetensors` and `tokenizer/google/umt5-xxl/`; no separate Wan download is needed.
Up to three cameras form a 384×320 canvas in order: main view above, lower-left and lower-right views, with missing positions black;
explicitly set `--policy.camera_layout='["observation.images.head","observation.images.left",null]'`, covering all selected cameras.

Default action chunks have 32 rows; video samples rows 0, 4, …, 32. `chunk_size` must be an integer multiple of `4 × video_stride`.
The default `--policy.action_representation=recorded` preserves recorded state/action fields and units, using training-set min/max;
importing official base weights reinitializes action input, action output, and state projections.
Video/action uses joint flow matching with frozen VAE/umT5. Future images supervise training only; inference uses current images with 10 synchronous denoising steps and shift 5 by default.
Defaults: BF16 and activation checkpointing. Use common FSDP2 when GPU memory is insufficient. Supports trainer EMA and ordinary remote action chunks, without RTC.

Runtime reads local assets only; checkpoints include frozen encoders/tokenizer and migrate independently.
Reserve space for two full checkpoints when saving; old files are removed only after new files are written successfully.
See [SOURCE.json](SOURCE.json) for provenance and presets.

To retain official 80-dimensional end-effector semantics, use optional `alpha80` mode:

```bash
alohamini train --policy.type=openwam \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --policy.path="$HOME/Alohamini_workspace/pretrained/openwam" \
  --policy.action_representation=alpha80 \
  --state=joint_position,base_velocity,lift_height \
  --run_name=openwam_tcp80 --batch_size=1 --steps=30000 \
  --keep_last_checkpoints=1 --background
```

This mode automatically sets TCP state, absolute TCP action, and `rotation_6d_columns`, preserving the three official 80-dimensional projection layers.
FK uses dataset kinematics. Standard AlohaMini data resolves these from recorded calibration and the model reference pose.
For nonstandard calibration, declare `--data.kinematics.spec=/path/to/coordinates.json`;
see [Joint, TCP, and relative targets](../../../../docs/training.md#joint-tcp-and-relative-targets). Missing calibration or mismatched fields prevent training.
Arms must be explicitly named `left` and `right`; selected state must include every arm joint position and gripper.

| Contents | 80-dimensional slots (zero-based, inclusive) | Encoding / unit |
| --- | --- | --- |
| Left arm position, rotation, gripper | 0–2, 3–8, 9 | m, first two rotation-matrix columns, closed −1 / open +1 |
| Right arm position, rotation, gripper | 34–36, 37–42, 43 | Same as above |
| Base x, y, yaw velocities | 68, 69, 70 | m/s, m/s, deg/s; platform reserved-slot extension |
| Lift height | 71 | mm; platform reserved-slot extension |
| Unused hand and reserved channels | 10–33, 44–67, 72–79 | Excluded from action loss |

Each end effector still has 6 DOF, represented by 3 position + 6 rotation channels, or 9 total; adding a gripper gives 10 per arm.
The 18-dimensional joint record becomes **24 active channels** through FK, placed by name into the **80-dimensional interface** before network input.
Grippers keep official dedicated slots. Base/lift slots 68–71 are platform conventions; their meanings are unspecified by the official model.
Poses follow frames saved in calibration, such as `left_Base` and `right_Base`, without automatic world-frame conversion.

Rotation 6D values remain unchanged. Grippers map through calibrated closed/open endpoints; other channels use training-set min/max, with inputs clipped to [−1,1].
For min/max-normalized action channels with identical bounds, decoding restores the training constant;
for example, all-zero base labels prevent denoising residuals from becoming base velocity.
Checkpoints save slots, statistics, calibration, and FK/IK geometry. Deployment denormalizes automatically and solves IK continuously from current joint feedback,
restoring Host action order/units, with base/lift restored directly. Degenerate rotations, unreachable poses, or joint-limit violations reject the chunk.
An 18-dimensional `recorded` checkpoint cannot become `alpha80` by toggling a flag; fine-tune again from the official base.

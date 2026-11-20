# AC-DiT

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Examples reserve episode 1 for validation; for single-episode data, omit `--dataset.eval_episodes` and `--eval_steps`.

Download the frozen SigLIP image/language encoder:

```bash
hf download google/siglip-so400m-patch14-384 \
  --local-dir "$HOME/Alohamini_workspace/pretrained/siglip-so400m-patch14-384"

alohamini train --policy.type=acdit --policy.stage=mobility \
  --policy.use_pointcloud=false \
  --policy.assets_path="$HOME/Alohamini_workspace/pretrained/siglip-so400m-patch14-384" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --run_name=acdit_mobility --background
```

After the mobility stage completes, start a new whole-body training run:

```bash
alohamini train \
  --policy.path="$HOME/Alohamini_workspace/runs/acdit_mobility/checkpoint" \
  --policy.stage=full \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --run_name=acdit_full --background
```

The mobility stage defaults to 20000 steps at `1e-4`; the whole-body stage uses 30000 steps at `1e-5`, with the mobility branch frozen.
Per-process batch 16, AdamW, constant learning rate, BF16, FP32 trainable parameters, EMA disabled.
If GPU memory is insufficient, use common FSDP2 support, reduce batch size, and increase gradient accumulation.

Requires nonempty task text and named state/action fields. Recorded fields/units are retained, with training-set mean/std normalization.
Images are aspect-preserving padded to 384×384 and mapped to `[-1,1]`. Inputs use previous/current images and current state, with a 2-row action chunk.
Action heads initialize randomly by default. External RDT initialization uses `mobility_pretrained_path` / `body_pretrained_path` and requires explicit named-slot mappings.

Without point-cloud data, use the RGB mode above. Point-cloud mode requires `--policy.use_pointcloud=true` in stage one,
with `observation.point_cloud` (N×3 or N×6), an actual coordinate frame, and metric-unit declarations; do not generate placeholder point clouds.
Stage two inherits stage-one cameras, input mode, and statistics. Mobility checkpoints cannot directly deploy whole-body control; remote action-chunk serving is not currently integrated.
See [SOURCE.json](SOURCE.json) for provenance and presets.

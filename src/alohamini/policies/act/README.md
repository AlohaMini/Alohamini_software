# ACT

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Examples reserve episode 1 for validation; for single-episode data, omit `--dataset.eval_episodes` and `--eval_steps`.

```bash
python -m alohamini.learning.train \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' \
  --policy.type=act --policy.device=cuda --policy.chunk_size=100 \
  --output_dir="$HOME/Alohamini_workspace/runs/act_01" \
  --steps=100000 --batch_size=2 --save_freq=10000 --log_freq=100 \
  --eval_steps=1000 --background
```

`--state=none` uses vision only; `--image_size='[480,640]'` specifies height and width.
ResNet18 loads ImageNet weights by default, caching the first download under the workspace's `pretrained/`.
Prepare weights beforehand for offline use; `--policy.pretrained_backbone_weights=null` selects random initialization.

## Camera identity embeddings

New ACT and AM-ACT training enables camera identity embeddings by default to associate front, left-wrist, right-wrist, and other images with their sources.
Training and inference share embeddings. Camera names/order are saved with checkpoints; keep naming consistent during collection and deployment.
`--policy.use_camera_embeddings=false` disables them, retaining original ACT visual fusion.
Existing checkpoints preserve their settings for inference, resume, and default fine-tuning. To add embeddings to an older model,
append `--policy.use_camera_embeddings=true` to fine-tuning with `--policy.path=<OLD_CHECKPOINT>`,
using a new output directory without `--resume=true`. New embeddings learn from scratch and require fine-tuning before distinguishing cameras.

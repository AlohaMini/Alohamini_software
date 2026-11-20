# VQ-BeT

[Policy index](../../../../docs/training.md#policy-selection)

```bash
alohamini train --policy.type=vqbet \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --steps=100000 --batch_size=64 \
  --output_dir="$HOME/Alohamini_workspace/runs/vqbet_01" --background
```

First training requires ResNet18 ImageNet-1K V1 weights; prepare them in the platform vision-weight cache.
For random vision initialization, explicitly set `--policy.pretrained_backbone_weights=null`.
Defaults: 5 historical frames and 3 prediction positions, each predicting 5 action steps. Incomplete windows at both episode ends are excluded.
The quantizer trains for the first 20000 successful optimizer updates, followed by the action-prediction network. Stage/counters are saved with checkpoints.
Total training steps must exceed `--policy.n_vqvae_training_steps`.
Synchronous action queues are supported; remote action chunks, asynchronous execution, and RTC are currently unsupported.

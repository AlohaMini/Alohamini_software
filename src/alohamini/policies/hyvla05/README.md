# HyVLA-0.5

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Example episode 1 is reserved for validation; for a single episode, omit `--dataset.eval_episodes` and `--eval_steps`.

```bash
hf download tencent/Hy-Embodied-0.5-VLA-RoboTwin \
  --revision bd7bba6f5934ad62293a2a34f74760c6a3ef2ff8 \
  --local-dir "$HOME/Alohamini_workspace/pretrained/hyvla05"

alohamini train --policy.type=hyvla05 \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --policy.path="$HOME/Alohamini_workspace/pretrained/hyvla05" \
  --run_name=hyvla05_task --batch_size=1 --steps=60000 --background
```

Requires nonempty task text and state. Cameras follow `--cameras` or dataset order; images are aspect-preserving padded to 224×224.
Defaults: 6 history frames spaced 15 rows apart, 20 action rows, 10 flow-matching sampling steps, BF16; state/action uses training-set mean/std.
Actions are 20 consecutive rows of original recorded targets, without RoboTwin relative/absolute pose concatenation or automatic FK/IK.
Online history reads past observations by client timestamps; `reset()` clears history and action queues.

`--policy.network='{"freeze_vision_encoder":true}'` freezes vision while retaining image features in training.
Runtime reads local assets only; checkpoints include tokenizer/weights. Supports DDP/FSDP2, trainer EMA, and ordinary remote action chunks.
RTC, FlowPRO, and Bézier transitions are not integrated. See [SOURCE.json](SOURCE.json) for provenance and presets.

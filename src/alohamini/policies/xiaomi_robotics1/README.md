# Xiaomi-Robotics-1

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Example episode 1 is reserved for validation; for a single episode, omit `--dataset.eval_episodes` and `--eval_steps`.

Use `xiaomi_robotics1` with official `model_states.pt` in the local base directory; `assets_path` points to
Qwen3-VL-4B-Instruct configuration, tokenizer, and image-processing files.

```bash
alohamini train --policy.type=xiaomi_robotics1 \
  --policy.path="$HOME/Alohamini_workspace/pretrained/xiaomi-robotics1" \
  --policy.assets_path="$HOME/Alohamini_workspace/pretrained/qwen3-vl-4b-assets" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --distributed_backend=fsdp2 --num_processes=4 \
  --output_dir="$HOME/Alohamini_workspace/runs/xiaomi_01" --background
```

Defaults read `forward,wrist_left,wrist_right` RGB and task text, using 30-step action chunks, BF16, and synchronous execution.
Current multiview prompts assume front, left wrist, and right wrist. Dual-front plus right-wrist input does not match and cannot be used by overriding `--cameras` alone. Adapt view text/configuration first, or collect front plus both wrist views.
Joint observations are retained. Saved calibration transforms actions to tool-frame end-effector deltas, with rotation vectors and gripper-opening deltas.
Statistics fit training data; action mean/std is computed per time step, while state uses q01/q99.
If kinematics are missing, provide them according to [Joint, TCP, and relative targets](../../../../docs/training.md#joint-tcp-and-relative-targets).
The lift holds current height; the base uses dataset velocity labels. Official asynchronous action-prefix execution is not yet supported.

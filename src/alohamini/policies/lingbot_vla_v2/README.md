# LingBot-VLA 2.0

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Example episode 1 is reserved for validation; for a single episode, omit `--dataset.eval_episodes` and `--eval_steps`.

Use `lingbot_vla_v2` with a local base containing official LeRobot-format `config.json` and `model.safetensors`,
plus Qwen configuration, tokenizer, and image-processing assets. The teacher directory contains `moge/model.pt`, `depth/model.pt`,
`dino_video/teacher_step_10000.pth`, and `dino_video/config.yaml`.

```bash
alohamini train --policy.type=lingbot_vla_v2 \
  --policy.path="$HOME/Alohamini_workspace/pretrained/lingbot_vla_v2_base" \
  --policy.teacher_root="$HOME/Alohamini_workspace/pretrained/lingbot_vla_v2_teachers" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --distributed_backend=fsdp2 --num_processes=4 \
  --output_dir="$HOME/Alohamini_workspace/runs/lingbot_01" --background
```

Defaults read `forward,wrist_left,wrist_right` RGB, the full 18-dimensional state, and task text, mapped to 55 semantic slots.
Dual-front plus right-wrist cameras do not match this default view protocol; do not substitute them based only on having three images. Other views require explicit input-order/teacher-input checks, with adaptation settings saved in checkpoints.
Joint actions are relative to the current observation; grippers and base velocities remain absolute. Training mean/std is computed after action conversion.
The lift holds current height by default; explicit `--policy.lift_mode=reserved` uses platform extension slots for lift-action training.
Training uses future frames for teacher losses; inference reads only current frames and needs no teacher weights.
The optimizer uses platform AdamW with MoE expert learning-rate multipliers. Muon and RTC execution are not yet supported.
Adjust GPU count and batch size for model memory requirements.

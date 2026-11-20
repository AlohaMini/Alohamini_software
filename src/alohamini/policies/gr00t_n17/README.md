# GR00T N1.7

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Example episode 1 is reserved for validation; for a single episode, omit `--dataset.eval_episodes` and `--eval_steps`.

Download local base and Cosmos processor assets. Obtain Cosmos repository access and log in to Hugging Face first:

```bash
hf download nvidia/GR00T-N1.7-3B \
  --include 'config.json' 'model*.safetensors' 'model*.index.json' 'embodiment_id.json' \
  --local-dir "$HOME/Alohamini_workspace/pretrained/gr00t_n17"
hf download nvidia/Cosmos-Reason2-2B \
  --include '*.json' '*.txt' '*.model' '*.jinja' \
  --local-dir "$HOME/Alohamini_workspace/pretrained/gr00t_n17/backbone"

CUDA_VISIBLE_DEVICES=0,1 alohamini train --policy.type=gr00t_n17 \
  --policy.path="$HOME/Alohamini_workspace/pretrained/gr00t_n17" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --eval_steps=1000 \
  --distributed_backend=fsdp2 --num_processes=2 --batch_size=2 \
  --gradient_accumulation_steps=4 --run_name=gr00t_task --background
```

The LLM and vision encoder are frozen by default; action heads, projections, and VL adapters are trained.
`--policy.tune_llm=true` and `--policy.tune_visual=true` unfreeze them individually; `--policy.tune_top_llm_layers=4` unfreezes only the final four LLM layers.
Defaults: 10000 steps, AdamW `1e-4`, 5% warmup with cosine decay, BF16 computation, FP32 trainable parameters, activation checkpointing enabled, EMA disabled.
The example has effective batch 16; the resource preset is 8 per process. Adjust for GPU memory.

Uses selected cameras' current frames, nonempty task text, named state, and 40 action rows; state retains recorded coordinates/units.
Default actions use relative arm joint positions and absolute gripper targets; base/lift targets remain absolute. The entire joint-action chunk shares the current observation anchor and is decoded before execution.
Platform fields bind automatically by name. External data requires `--data.joint_delta_fields='{"action_joint":"state_joint"}'` to declare all relative joints, with matching coordinates and units.
`--data.action_representation=recorded` retains original absolute targets; existing checkpoints retain their saved action representation.
For relative end-effector actions, add `--data.state_representation=tcp --data.action_representation=tcp_delta --data.rotation_representation=rotation_6d --data.tcp_delta_frame=tool`. Standard neutral-pose data resolves kinematics automatically.
Images are cropped, resized to short side 256, and passed to the Qwen3-VL processor. State/action uses training-set q01/q99 mapped to `[-1,1]` with clipping.
Defaults to new embodiment 10 (the official `NEW_EMBODIMENT` ID), reinitializing its state/action projections.
Existing checkpoints load/resume with saved `embodiment_id`. Reusing an embodiment requires `state_slot_map`, `action_slot_map`, and `reinitialize_embodiment` disabled.
Runtime reads only local assets; checkpoints migrate independently. Common inference and remote action chunks are supported, without RTC integration.
See [SOURCE.json](SOURCE.json) for provenance and presets.

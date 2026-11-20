# G0.5

[Policy index](../../../../docs/training.md#policy-selection)

Prepare a locally converted base containing `config.json`, `model.safetensors`, and `hf_processor/`:

```bash
BASE="$HOME/Alohamini_workspace/pretrained/g05"
hf download lerobot/g05_libero --revision=9b7ee7d7286b2744e7cfae855dd71aa3cf4c075a \
  --local-dir="$BASE"

alohamini train --policy.type=g05 --policy.path="$BASE" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --output_dir="$HOME/Alohamini_workspace/runs/g05_01" --background
```

Default flow fine-tuning uses FP32 parameters, BF16 mixed precision, batch 8, 100k steps, learning rate `1e-5`,
1000 warmup steps, and 8 sampled flow times per sample. Action loss updates the VLM by default.
`--policy.gradient_checkpointing=true` enables activation checkpointing; use [common settings](../../../../docs/training.md#multiple-gpus-and-gradient-accumulation) for distributed training.
Architecture, observation history, and action-chunk length load from the local base, retaining all Transformer layers.

All selected cameras participate. Defaults are 256×256 per view and RGB `[-1,1]`; order is saved with checkpoints.
Multiframe bases require matching temporal vision architecture. Synchronous and remote inference use common observation-history caching.
State/action uses recorded or explicitly transformed fields/units mapped to base slots; robot input/output heads reset by default.
To preserve heads, set `--policy.reinitialize_robot_heads=false` and provide complete named `state_slots` and `action_slots` mappings.

Defaults use training-set q01/q99, computing actions separately per future step and retaining out-of-range actions. Inference restores the complete action chunk before stepwise execution.
`--policy.data_protocol=g05` enables the original implementation's post-normalization `[-5,5]` clipping.
Platform robot adaptation does not automatically convert relative poses, gripper direction, or joint frames; it is not a direct reproduction of the original benchmark data protocol.

ActionCodec training also needs `action_tokenizer.safetensors`, compatible original-coordinate slots, and normalization:
`train_discrete_actions=true text_loss_weight=1 reinitialize_robot_heads=false`.
`action_head=flow|actioncodec` selects the inference head, defaulting to flow. Flow inference does not generate action tokens.
CoT text training requires `predict_cot=true text_loss_weight=1 use_language_recipe=true` and actual language annotations,
with `Subtask: ${subtask}` declaring the target in the recipe; `runtime_system=system2` enables inference CoT.
LoRA, RTC, compilation, and distributed execution of this new policy remain unvalidated.

Without `flash-linear-attention` / `causal-conv1d`, linear attention uses a slower PyTorch path;
current validation covers that path, while accelerated kernels remain unvalidated.
Code and weights are subject to noncommercial and other conditions in the [G0.5 Community License](LICENSE-G0.5).
This project does not imply endorsement by Galaxea.

# LeWorldModel

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Example episode 1 is reserved for validation; for a single episode, omit `--dataset.eval_episodes` and `--eval_steps`.

Train from scratch without downloading a base. Requires at least one RGB view and named action fields, with no state or language input:

```bash
alohamini train --policy.type=lewm \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --eval_steps=1000 \
  --batch_size=32 --run_name=lewm_task --background
```

Images use 224×224 and ImageNet mean/std; cameras share an encoder and average features.
Actions retain recorded coordinates/units with training-set z-score normalization. Supports `recorded` and `tcp_absolute`, excluding `tcp_delta`.
Defaults use 3 historical observations and 5-row action chunks. Prediction loss is feature MSE plus `0.09 × SIGReg`; validation does not automatically produce action MAE.
Defaults: 30000 steps, AdamW `5e-5`, constant learning rate, BF16, EMA disabled; DDP/FSDP2 supported.
Distributed SIGReg aggregates the global batch within one forward pass; gradient accumulation does not merge statistics across passes.

Inference requires explicit goal images for all cameras; CEM defaults to planning 25 action rows:

```bash
alohamini evaluate --host 192.168.77.2 --robot_model alohamini2pro \
  --policy.path "$HOME/Alohamini_workspace/runs/lewm_task/checkpoint" \
  --device cuda --fps 30 --goal-images '{"forward":"/path/to/goal.png"}'
```

For multiple cameras, add entries to the same JSON. Offline evaluation also uses `--goal-images`;
the Python interface is `NativePolicy.set_goal({"forward": "/path/to/goal.png"})`. Goals are not saved in weights; `reset()` retains them.
Import official `weights.pt` through `--policy.path` with matching architecture and action-chunk dimensions. PushT weights cannot directly serve different action dimensions.
Remote goal transfer and RTC are not currently integrated. See [SOURCE.json](SOURCE.json) for provenance and presets.

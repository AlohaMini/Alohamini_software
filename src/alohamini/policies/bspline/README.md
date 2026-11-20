# B-spline Policy

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Examples reserve episode 1 for validation; for single-episode data, omit `--dataset.eval_episodes` and `--eval_steps`.

```bash
python -m alohamini.learning.train \
  --policy.type=bspline --policy.device=cuda \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --batch_size=2 \
  --policy.fit_tolerance=0.01 \
  --output_dir="$HOME/Alohamini_workspace/runs/bspline_01" --background
```

Supports absolute commands with declared coordinates/units and joint prediction of both arms and grippers; joint/end-effector coordinates are not converted automatically.
Defaults: 2 observation frames, 16 spline-parameter rows, cubic splines, 16 DDIM steps, EMA enabled.
`policy.horizon` counts parameter rows; knots determine trajectory duration.
`policy.fit_tolerance` uses recorded action units; knot and control-point statistics are fitted on training data.
Episodes require at least `degree+1` frames; separate training and validation by episode.

Parameter caches default to the system temporary directory; use `--policy.parameter_cache_dir` for persistent storage.
Caches store numeric labels only, without modifying source data. Changes to fit accuracy, parameter rows, or data create separate caches automatically.

Evaluate with `--inference.type=async`, which defaults to phase alignment; add
`--inference.continuity=none` to disable it.
`--policy.time_scale 1.0` is the default time scale; values greater than 1 shorten trajectories.
BSP uses neither ACT temporal ensembling nor dense-action RTC guidance. Keep the time scale at 1 for initial use.

# Diffusion Policy

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Examples reserve episode 1 for validation; for single-episode data, omit `--dataset.eval_episodes` and `--eval_steps`.

```bash
python -m alohamini.learning.train \
  --policy.type=diffusion --policy.device=cuda \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' \
  --cameras='["forward","wrist_right"]' \
  --output_dir="$HOME/Alohamini_workspace/runs/diffusion_01" \
  --batch_size=2 --save_freq=10000 --log_freq=100 --background
```

Defaults: 2 observation frames, 64 predicted steps, 32 executed steps, and 100 DDPM steps. The vision configuration follows
[LeRobot PR #3202](https://github.com/huggingface/lerobot/pull/3202).
RGB uses ImageNet `MEAN_STD`; state/action uses training-set `MIN_MAX`.
`--policy.n_action_steps` changes executed steps; `--policy.num_inference_steps` changes denoising steps.

Trailing-start exclusion defaults to `horizon - n_action_steps - n_obs_steps + 1`, or **31**;
override with `--policy.drop_n_last_frames`, setting `0` to retain all sampling starts.
EMA is disabled by default; enable with `--ema.enable=true`.

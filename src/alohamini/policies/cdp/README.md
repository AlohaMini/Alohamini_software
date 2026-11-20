# CDP

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Examples reserve episode 1 for validation; for single-episode data, omit `--dataset.eval_episodes` and `--eval_steps`.

```bash
python -m alohamini.learning.train \
  --policy.type=cdp --policy.device=cuda \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --batch_size=2 \
  --output_dir="$HOME/Alohamini_workspace/runs/cdp_01" --background
```

Uses the [CDP2 RGB configuration](https://github.com/GaavaMa/Causal-Diffusion-Policy/tree/58b23af1e9ac246a11001d7d5388a1de4f53daa5):
32 historical frames, a 40-step total window, and 8 executed steps; 100 DDPM steps predicting clean actions.
Online execution retains an action buffer and KV cache; call `reset()` at episode start.
Historical actions strictly precede the current frame; image/state histories include the current frame.
RGB is resized to 84×84, cropped to 76×76, and normalized with ImageNet `MEAN_STD`; state/action uses training-set `MIN_MAX`.
ResNet18 uses GroupNorm without pretrained weights. Trainer EMA defaults to enabled; zero trailing starts are excluded.
Training defaults to 100000 steps with 1000 warmup steps; adjust with `--steps` and `--scheduler.warmup_steps`.
Only RGB CDP2 is supported, excluding point-cloud CDP3. Action-chunk size is part of model architecture and cannot change when loading checkpoints;
denoising steps must equal training diffusion steps. Offline evaluation uses demonstration action history and does not represent closed-loop success.

# dit-policy (DiT-Block)

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Examples reserve episode 1 for validation; for single-episode data, omit `--dataset.eval_episodes` and `--eval_steps`.

```bash
python -m alohamini.learning.train \
  --policy.type=dit_policy --policy.device=cuda \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --batch_size=2 \
  --output_dir="$HOME/Alohamini_workspace/runs/dit_policy_01" --background
```

Uses the independent DiT encoder/decoder from [SudeepDasari/dit-policy](https://github.com/SudeepDasari/dit-policy/tree/ba22ee3ff455f0423fc9cfa5658a1a1cdddc6e73)
with defaults of 1 observation frame, 100 action steps, 6 layers, and hidden dimension 512;
100 training diffusion steps and 8 DDIM inference steps. `--policy.n_action_steps` can shorten execution.
RGB is resized to 256×256, randomly resized/cropped to 224×224 and blurred during training, or resized to 224×224 during inference;
uses ImageNet `MEAN_STD`, ResNet18 GroupNorm, and spatial tokens.
`--policy.pooled_vision=true` uses one token per frame instead.
State retains original values. Local actions use training-set `MIN_MAX`; upstream data must be normalized beforehand.
Initialization defaults to torchvision ImageNet V1 weights, excluding the author's separately distributed GN weights.
Defaults: 150000 training steps, 2000 warmup steps, EMA disabled. Trailing samples are retained with padded actions masked. No language input.

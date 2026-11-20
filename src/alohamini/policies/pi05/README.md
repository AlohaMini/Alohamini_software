# π0.5

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Example episode 1 is reserved for validation; for a single episode, omit `--dataset.eval_episodes` and `--eval_steps`.

Prepare converted OpenPI PyTorch `model.safetensors` and a PaliGemma SentencePiece tokenizer.
Direct Orbax/JAX checkpoint loading is unsupported.

```bash
python -m alohamini.learning.train \
  --policy.type=pi05 --policy.device=cuda \
  --policy.path=/path/to/pi05_pytorch \
  --policy.tokenizer_path=/path/to/paligemma_tokenizer.model \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' \
  --cameras='["forward","wrist_right"]' \
  --output_dir="$HOME/Alohamini_workspace/runs/pi05_01" \
  --batch_size=1 --save_freq=1000 --background
```

Automatic preset `openpi-pi05-alohamini-v1`: position state, task text, aspect-preserving padding to 224×224,
50-step action chunks, q01/q99 normalization, AdamW, and warmup/cosine scheduling.
Cameras follow dataset-declared views automatically; use `--cameras` for a subset. Select at least one and retain dataset names.
Camera order is saved automatically; inference requires all training-selected views.
State retains all 18 original position, base-velocity, and lift-height dimensions, including declared fixed dimensions.
Each arm's 6 joints predict deltas from current state, restored to absolute targets before execution. Grippers, base velocity, and lift height retain their original meaning.
Override network arguments with `--policy.network='{"action_horizon":30}'`; checkpoints include the tokenizer.

Normalization statistics use training data only, computing relative actions first, then q01/q99 (1st/99th percentiles).
Out-of-range values are not clipped; dimensions with equal q01/q99 are not zeroed. Rare valid actions still contribute to training.
The denominator defaults to `q99-q01+1e-6`. Narrow ranges amplify values; set minimum spans with `--policy.span_floors`.
This accepts a JSON object whose `state` and `actions` each specify 18 nonnegative physical spans in saved-field order.
No minimum spans apply by default. Settings are saved with checkpoints; normalization and reconstruction share the same spans.

Choose the training recipe, then the parameter-update method:

- `--policy.training_recipe=flow` (default): OpenPI's public flow-matching fine-tuning, preserving action-loss gradients into the VLM.
- `--policy.training_recipe=ki`: joint FAST cross-entropy and flow matching. Layerwise isolation blocks action-expert K/V gradients into the VLM; the two action branches cannot see each other. Inference still uses the continuous action expert.

| `--policy.finetuning_mode` | Vision encoder and projection | LLM | Action-expert Transformer |
| --- | --- | --- | --- |
| `a` | Frozen; `--policy.a_vision_mode=full` enables full training | LoRA | LoRA |
| `b` | Frozen | Frozen | Full |
| `c` | Full | LoRA | Full |
| `d` | LoRA | LoRA | Full |
| `full` (default) | Full | Full | Full |

Four configurations are in [`examples/learning/pi05/`](../../../../examples/learning/pi05). `ki + b` is unsupported: freezing the whole VLM is not dual-objective KI.
LoRA applies to attention and MLP projections; mode D also covers visual projection. Action input/output and time MLPs train fully.
LLM/vision defaults are rank=16, alpha=16; action-expert defaults are rank=32, alpha=32. Override with `lora_rank/lora_alpha` and `expert_lora_rank/expert_lora_alpha`, respectively.
These fine-tuning combinations are platform configurations, not four officially published experiments.

Download lightweight tokenizer assets, excluding the full base:

```bash
python -m alohamini.policies.pi05.assets
python -m alohamini.learning.train \
  --config=examples/learning/pi05/a.json \
  --policy.training_recipe=ki --policy.a_vision_mode=frozen \
  --policy.path=/path/to/pi05_pytorch \
  --dataset.root=/path/to/dataset --dataset.eval_episodes='[1]' \
  --run_name=pi05_ki_a --background
```

Assets default to workspace `pretrained/pi05_tokenizers/`. KI FAST uses its `fast/` directory by default; override with `--policy.fast_tokenizer_path`.
FAST encodes normalized 18-dimensional actions before zero-padding, retaining π0.5 task/state/Action prompts. If targets exceed `--policy.fast_max_tokens=256`, increase the limit; never truncate actions.
KI checkpoints include the complete network, LoRA, both tokenizers, and training recipe. Inference needs no original base directory. Resume cannot change the recipe or fine-tuning mode.

The model defaults to BF16 with some FP32 modules; platform autocast is disabled (`mixed_precision=none`).
For FP32 trainable parameters and BF16 computation, append `--trainable_dtype=float32 --mixed_precision=bfloat16`;
see [Numeric precision](../../../../docs/training.md#numeric-precision) for other common options.
Defaults: full activation checkpointing, 30000 steps, AdamW `2.5e-5`, betas `(0.9,0.95)`, weight decay `1e-10`, gradient clipping `1`, 1000-step warmup, and cosine decay to `2.5e-6`; logs every 100 steps, saves every 1000. EMA is disabled.
The example uses batch=1 and 32 accumulation steps for single-GPU effective batch=32. Disable activation checkpointing with `--policy.gradient_checkpointing=false`. `network.pytorch_compile_mode` defaults to `max-autotune`, compiling sampling only with slower first inference; set `null` to disable.

KI can mix image/text or planning data with `--policy.vlm_data_path=/path/to/train.jsonl`, pairing each robot sample with an independently sampled VLM example:

```json
{"images":{"forward":"images/001.jpg"},"prompt":"What should happen next?","answer":"Pick up the red block."}
```

Image paths are relative to JSONL; camera sets load from the manifest, with missing views masked. Only answers contribute language loss; no action loss is computed.
All three loss weights default to 1; override with `flow_loss_weight/fast_loss_weight/vlm_loss_weight`. Token loss averages valid tokens per sample, then averages samples.
Joint data is training-only, with manifest SHA256 saved and checked on resume. Without it, KI uses robot data alone. The official general-pretraining data mixture is not reproduced.
See [KI paper §5](https://www.pi.website/download/pi05_KI.pdf) for the method and [OpenPI](https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac) for training defaults.

For full SmolVLA/π0.5 bases, adjust batch size and precision to available GPU memory.

# FineART-VLA

[Policy index](../../../../docs/training.md#policy-selection)

Prepare released base weights and a local PaliGemma tokenizer:

```bash
hf download lerobot/FineART-VLA-subtask-midtrained \
  --revision=b6cf4faf2630f971dd390bc32365e9d9c330da8e \
  --local-dir="$HOME/Alohamini_workspace/pretrained/fineart"
hf download google/paligemma-3b-pt-224 \
  --revision=35e4f46485b4d07967e7e9935bc3786aad50687c \
  --include 'tokenizer*' 'special_tokens_map.json' \
  --local-dir="$HOME/Alohamini_workspace/pretrained/paligemma-tokenizer"

alohamini train --policy.type=fineart_vla \
  --policy.path="$HOME/Alohamini_workspace/pretrained/fineart" \
  --policy.assets_path="$HOME/Alohamini_workspace/pretrained/paligemma-tokenizer" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_with_subtasks" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/fineart_01" --background
```

Data must include timestamped subtask annotations in `language_persistent`, with task text from the original data.
By default, 30% of samples supervise task→subtask text and 70% supervise subtask→action. Missing valid annotations cause an error.
Training labels are neither generated nor guessed.

State/action uses named recorded coordinates and training-set q01/q99 without clipping. Actions pad to 32 dimensions with padded dimensions excluded from loss; repeated episode-end targets are retained.
Cameras follow saved field order, resized with aspect ratio preserved and padded to 224. Defaults: 18 layers each for the language Transformer and action expert, 27 for the SigLIP vision tower.

Base loading inherits model configuration, with explicit arguments overriding it. The released base uses `flow_num_repeats=1` and `fast_skip_tokens=128`;
new configurations default to 5 and 1152, respectively.

By default, the main rank fits FAST on training windows and checks reconstruction error/length on independently sampled training windows, excluding validation episodes and never truncating action tokens.
The cache is under `Alohamini_workspace/pretrained/action_tokenizers/fast/`. Codec, configuration, and fit report are saved with checkpoints and reused directly on resume.
`auto_fit_fast_tokenizer=false` uses local `fast_tokenizer_path`; `fast_action_loss_weight=0` disables FAST loss.

KI and activation checkpointing default to enabled; flow/text/FAST loss weights are 10/1/1. AMP, TF32, and distributed settings use common training configuration.
Joint attention defaults to FP32 math SDPA; other modules retain configured model precision.
Validation separately aggregates action-sample means and text-token means, without interpreting mixed-task validation as action MAE.

To enable spatial VQA, add:

```bash
--policy.vqa_cameras='["forward"]' --policy.fast_skip_tokens=1152
```

`language_events` must contain paired VQA questions/answers bound to that camera. Points use `point_format=xy`; boxes use `bbox_format=xyxy`, in original-image `[0,1]` coordinates.
Position supervision uses PaliGemma `<loc>` tokens; FAST must avoid the location-token range.

Add `--language-replan-interval-s=2 --task='task instruction'` to local `evaluate` or cloud `inference serve` for periodic subtask generation.
Disabled by default. Updates use observation time and discard old action queues on subtask changes. Episode termination follows evaluation duration or manual stop.
`NativePolicy.query_language(sample, kind="vqa", camera="forward", question="Where is the cup?")` returns text and points/boxes;
`sample` can be a platform sample or Host snapshot; querying does not advance the controller. Tool calls are currently unsupported.

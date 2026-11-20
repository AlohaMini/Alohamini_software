# AM-ACT

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Examples reserve episode 1 for validation; for single-episode data, omit `--dataset.eval_episodes` and `--eval_steps`.

See [ACT: Camera identity embeddings](../act/README.md#camera-identity-embeddings) for shared settings and existing-model migration.

```bash
python -m alohamini.learning.train \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_vision" \
  --dataset.eval_episodes='[1]' \
  --policy.type=am_act --policy.device=cuda --policy.chunk_size=100 \
  --output_dir="$HOME/Alohamini_workspace/runs/am_act_vision_01" \
  --steps=100000 --batch_size=2 --save_freq=10000 --log_freq=100 \
  --eval_steps=1000 --background
```

State-containing data is also supported. For velocity/current experiments, use data retaining feedback and add `--state=joint_velocity,joint_current`.
New AM-ACT training defaults to three-class classification for each of `x.vel`, `y.vel`, and `theta.vel` in `action.names`:

| Field | Class order: negative, stop, positive | Unit |
| --- | --- | --- |
| `x.vel` | `[-0.15, 0, 0.15]` | m/s |
| `y.vel` | `[-0.15, 0, 0.15]` | m/s |
| `theta.vel` | `[-45, 0, 45]` | deg/s |

Select weighting with `--policy.discrete_action_weighting=none|inverse_frequency|sqrt_inverse_frequency`
and adjust its cap with `--policy.discrete_action_max_weight_ratio`.

Common options:

| Option | Purpose |
| --- | --- |
| `--policy.base_classification=false` | Disable default base classification for new training and use continuous regression |
| `--policy.discrete_action_mask_absent_classes=true` | Mask classes absent from training data; disabled by default |
| `--policy.discrete_action_values='[[-0.15,0,0.15]]'` | Custom class values; also requires `discrete_action_dims` |

Check custom dimensions against `action.names`. Masking absent classes retains three output heads;
with only two observed classes it effectively becomes binary classification. Zero class weight does not prohibit predicting that class.
After recording a new direction, fine-tune with `--policy.path=<OLD_CHECKPOINT>` to recompute statistics and enable new classes.
Changed data cannot use `--resume=true`. With `action_loss_groups`, include every continuous dimension requiring regression.

## Conditional CVAE

Default AM-ACT uses an action/optional-state posterior, standard Gaussian prior, and `z=0` at inference.
Use the common example to enable image-conditioned posterior and prior:

```bash
python -m alohamini.learning.train \
  --config=examples/learning/am_act_cvae.json \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_vision" \
  --dataset.eval_episodes='[1]' --eval_steps=1000 --background
```

This example is a robot-adaptation configuration: ResNet-18, a 30-step chunk, 10 executed steps, vision only,
4-query attention pooling, Image+Action posterior, image-conditioned Gaussian prior, KL weight 1,
and 5000 optimizer-step warmup. The action-backbone decoder has 7 layers; backbone and posterior encoders have 4 each.
Images use RGB/ImageNet normalization; actions use training statistics and original target units. No language input is used.

| Configuration | Values |
| --- | --- |
| `posterior_condition` | `image_action` / `action` |
| `prior_type` | `conditional_gaussian` / `standard_normal` |
| `latent_visual_pooling` | `multi_query_attention` / `gap` |
| `latent_inference_mode` | `sample` / `mean` / `zero` |
| `latent_refresh_mode` | `phase` / `chunk` |

The example caches z by phase: first sample from images, then reuse z with fresh images for later predictions.
The task executor calls `policy.refresh_latent()` to switch phases; `reset()` at a new episode also clears the cache.
The model does not detect phases automatically. Refreshing z also clears action queues and temporal ensembling to avoid mixing predictions from different latents.

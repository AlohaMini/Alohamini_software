# Data Preparation, Training, and Deployment

**English** | [简体中文](zh-CN/training.md)

## 1. Preparation

Complete [PC installation](install.md), install [policy and feature extensions](install.md#3-optional-extensions) as needed, and run from the repository root:

```bash
cd ~/Alohamini
conda activate alohamini
```

The default workspace is `~/Alohamini_workspace/`: `datasets/` stores data, `runs/` stores models, and `logs/training/` stores logs and PID files.

## 2. Inspect and edit data

```bash
alohamini dataset edit --dataset task_demo \
  --operation.type info --operation.show_features true

alohamini dataset check ~/Alohamini_workspace/datasets/task_demo \
  --decode-images --decode-videos

alohamini dataset edit --dataset task_demo \
  --output ~/Alohamini_workspace/datasets/task_demo_ready \
  --operation.type delete_episodes --operation.episode_indices '[3, 8]'
```

Episodes are indexed from 0. `--dataset` selects a workspace dataset name; alternatively, use `--root /absolute/path`.
The output directory must not exist. Retained episodes are renumbered in their original order, preserving image, state/action, and timing alignment.

Other editing operations share `--root` and `--output`:

| `--operation.type` | Example arguments |
| --- | --- |
| `split` | `--operation.splits '{"train": 0.8, "val": 0.2}'` or `'{"train": [0, 2], "val": [1]}'` |
| `merge` | `--operation.roots '["/dataset_A", "/dataset_B"]'`; no `--root` needed |
| `remove_feature` | `--operation.feature_names '["observation.images.wrist_right"]'` |
| `modify_tasks` | `--operation.new_task "pick up the block"`; optionally `--operation.episode_tasks '{"2": "put down the block"}'` |
| `recompute_stats` | Numeric statistics by default; `--operation.skip_image_video false` includes images |
| `convert_image_to_video` | JPEG/PNG to MP4; `--operation.episode_indices '[0, 2]'` selects episodes |
| `reencode_videos` | Reencode MP4, e.g. `--operation.rgb_encoder.crf 23` |
| `info` | `--operation.show_features true`; no `--output` needed |

Split results go into subdirectories such as `<output>/train`. Merging requires matching FPS, fields, cameras, calibration, and action coordinates.
Separate training and validation by complete episodes. Never place differently processed copies of the same demonstration in both splits.

Videos default to H.264, CRF 18, GOP 2, `yuv420p`, with one MP4 per episode per camera.
Adjust with `--operation.rgb_encoder.vcodec`, `.crf`, `.g`, `.preset`, and `.pix_fmt`.
Conversion preserves frame counts and timing; `yuv420p` requires even image dimensions.

`recompute_stats` excludes invalid feedback and writes statistical diagnostics to `meta/stats_info.json`.
`--operation.relative_action true --operation.chunk_size 50` computes joint-delta distributions for action chunks without rewriting action data.

### Repair

```bash
alohamini dataset repair /path/to/dataset --output /path/to/dataset_repaired
alohamini dataset check /path/to/dataset_repaired --decode-images --decode-videos
```

Repair supports unambiguous index and media-reference problems in AlohaMini and LeRobot data. It refuses repairs when images are missing or pairing is ambiguous.
`*.pending-*` indicates incomplete output. Repair cannot eliminate actual missing samples or timing errors.

## 3. Select data and inputs

Datasets use the LeRobot v3 format:

| Directory | Contents |
| --- | --- |
| `data/` | Parquet: state, action, indices, servo feedback, and validity masks |
| `videos/` | MP4 files per camera; readers locate frames using episode metadata and time indices |
| `meta/` | Field definitions, tasks, statistics, episode indices, and platform extensions such as calibration and actual sample times |

For external v3 data lacking platform calibration and safety records, train with `--dataset.allow_unannotated=true`.
Use `--state=recorded` to read the original state vector with field order and values preserved, or `--state=none` for vision-only tasks.

### Export and vision-only copies

To save a separate copy or remove numeric observations:

```bash
alohamini dataset export ~/Alohamini_workspace/datasets/task_demo_ready \
  --output ~/Alohamini_workspace/datasets/task_demo_vision \
  --format lerobot-v3 --vision-only
```

For example, `alohamini2pro` defaults to 14 arm-position dimensions, 3 base-velocity dimensions, and 1 lift-height dimension.
Vision-only copies retain videos and actions while removing numeric observation inputs. Videos are copied directly without changing pairing, frame counts, or timestamps.

### Input arguments

| Argument | Meaning |
| --- | --- |
| Omit `--state` | Follow the algorithm preset; automatic mode reads original state from external v3 data, or uses vision only if no state exists |
| `--state=none` | Vision only, without creating a separate dataset copy |
| `--state=recorded` | Original external v3 state vector without unit conversion |
| `--state=joint_position,base_velocity,lift_height` | 18 dimensions of positions, base velocity, and height |
| `--state=joint_velocity,joint_current` | 28 arm velocity/current dimensions, including grippers |
| `--cameras='["forward","wrist_right"]'` | Select cameras in the specified order; when omitted, follow the algorithm preset or use all recorded cameras if none is defined |

Platform fields are assembled in argument order. Joint positions use calibrated coordinates, joint velocities use the corresponding coordinate units per second, and current uses A;
base velocities use m/s and deg/s, and lift height uses mm. Actions contain absolute arm-position targets, base velocities, and lift-height targets.

### Joint, TCP, and relative targets

`--state` selects input fields; the following arguments select input and action representations without changing source data.

| Argument | Options and purpose |
| --- | --- |
| `--data.state_representation` | `recorded`: original input; `tcp`: replace arm joint positions with end-effector poses |
| `--data.action_representation` | `recorded`: original control targets; `joint_delta`: relative joint targets; `tcp_absolute`: absolute end-effector targets; `tcp_delta`: relative end-effector targets |
| `--data.joint_delta_fields` | JSON mapping from action fields to observation fields for external data; platform joints match automatically; coordinates and units must agree |
| `--data.rotation_representation` | `rotvec` (default, 3 dimensions) or `rotation_6d` (6 dimensions) |
| `--data.tcp_delta_frame` | TCP delta frame: `base` (default, each arm's base) or `tool` (current end effector) |
| `--data.kinematics.reference_pose` | Model calibration preset, such as `upright_forward` |
| `--data.kinematics.spec` / `--data.kinematics.path` | Custom mapping JSON / mapping file path; see the [example](../examples/datasets/kinematics-radians.json) |

Inputs and actions can be selected independently; unspecified settings follow the [policy preset](#policy-selection). Relative actions use the current observation as a shared reference for the entire chunk. Other fields, such as grippers, remain absolute targets.
TCP positions use m and rotation vectors use rad. Left and right arms use `left_Base` and `right_Base`, respectively.

TCP training example:

```bash
alohamini train --policy.type=act \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --state=joint_position \
  --data.state_representation=tcp \
  --data.action_representation=tcp_delta \
  --run_name=act_tcp_01 --background
```

Calibration defaults to dataset `meta/kinematics.json`, falling back to the model preset. See [the user guide](alohamini.md#2-leader-and-follower-calibration) for leader/follower calibration; nonstandard calibration or external data requires the actual mapping.
Mappings are saved with checkpoints and must match robot calibration at deployment. To export a mapping:

```bash
mkdir -p "$HOME/Alohamini_workspace/configs"
python -m alohamini.datasets.kinematic_fields \
  --dataset "$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --output "$HOME/Alohamini_workspace/configs/task_demo_kinematics.json"
```

The export file must not exist. Add `--spec` for custom mappings, or `--left-map /path/left.yaml --right-map /path/right.yaml` for Host calibration files.
Load it during training with `--data.kinematics.path`. Choose only one of `reference_pose`, `spec`, or `path`.
TCP data must retain joint observations even with `--state=none`. See [Inference modes](#inference-modes) for TCP inference settings.

## 4. Training

### Policy selection

See each policy directory for training commands, weight preparation, and policy-specific arguments:

| Policy | `policy.type` |
| --- | --- |
| <a id="act"></a>[ACT](../src/alohamini/policies/act/README.md) | `act` |
| <a id="am-act"></a><a id="conditional-cvae"></a>[AM-ACT](../src/alohamini/policies/am_act/README.md) ([Conditional CVAE](../src/alohamini/policies/am_act/README.md#conditional-cvae)) | `am_act` |
| <a id="diffusion-policy"></a>[Diffusion Policy](../src/alohamini/policies/diffusion/README.md) | `diffusion` |
| <a id="b-spline-policy"></a>[B-spline Policy](../src/alohamini/policies/bspline/README.md) | `bspline` |
| <a id="cdp"></a>[CDP](../src/alohamini/policies/cdp/README.md) | `cdp` |
| <a id="dit-policydit-block"></a>[dit-policy (DiT-Block)](../src/alohamini/policies/dit_policy/README.md) | `dit_policy` |
| <a id="multi-task-dit"></a>[Multi-Task DiT](../src/alohamini/policies/multi_task_dit/README.md) | `multi_task_dit` |
| <a id="fastwam"></a>[FastWAM](../src/alohamini/policies/fastwam/README.md) | `fastwam` |
| [Long-WAM](../src/alohamini/policies/longwam/README.md) | `longwam` |
| [VQ-BeT](../src/alohamini/policies/vqbet/README.md) | `vqbet` |
| <a id="ac-dit"></a>[AC-DiT](../src/alohamini/policies/acdit/README.md) | `acdit` |
| <a id="gr00t-n17"></a>[GR00T N1.7](../src/alohamini/policies/gr00t_n17/README.md) | `gr00t_n17` |
| <a id="leworldmodel"></a>[LeWorldModel](../src/alohamini/policies/lewm/README.md) | `lewm` |
| <a id="smolvla"></a>[SmolVLA](../src/alohamini/policies/smolvla/README.md) | `smolvla` |
| <a id="π05"></a>[π0.5](../src/alohamini/policies/pi05/README.md) | `pi05` |
| [π0](../src/alohamini/policies/pi0/README.md) | `pi0` |
| [π0-FAST](../src/alohamini/policies/pi0_fast/README.md) | `pi0_fast` |
| [EO-1](../src/alohamini/policies/eo1/README.md) | `eo1` |
| [EVO1](../src/alohamini/policies/evo1/README.md) | `evo1` |
| [X-VLA](../src/alohamini/policies/xvla/README.md) | `xvla` |
| [DM0.5](../src/alohamini/policies/dm05/README.md) | `dm05` |
| [FineART-VLA](../src/alohamini/policies/fineart_vla/README.md) | `fineart_vla` |
| [WALL-OSS-0.5](../src/alohamini/policies/wall_oss_05/README.md) | `wall_oss_05` |
| [VLA-JEPA](../src/alohamini/policies/vla_jepa/README.md) | `vla_jepa` |
| [LingBot-VA](../src/alohamini/policies/lingbot_va/README.md) | `lingbot_va` |
| [Cosmos3 Nano](../src/alohamini/policies/cosmos3/README.md) | `cosmos3_nano` |
| [Cosmos3 Edge](../src/alohamini/policies/cosmos3/README.md) | `cosmos3_edge` |
| [G0.5](../src/alohamini/policies/g05/README.md) | `g05` |
| <a id="hyvla-05"></a>[HyVLA-0.5](../src/alohamini/policies/hyvla05/README.md) | `hyvla05` |
| <a id="openwam-α"></a>[OpenWAM-α](../src/alohamini/policies/openwam/README.md) | `openwam` |
| <a id="xiaomi-robotics-1"></a>[Xiaomi-Robotics-1](../src/alohamini/policies/xiaomi_robotics1/README.md) | `xiaomi_robotics1` |
| <a id="lingbot-vla-20"></a>[LingBot-VLA 2.0](../src/alohamini/policies/lingbot_vla_v2/README.md) | `lingbot_vla_v2` |
| <a id="spirit-v15"></a>[Spirit v1.5](../src/alohamini/policies/spirit/README.md) | `spirit` |

Policy examples reserve episode 1 for validation and train on the rest. For a single-episode dataset, omit `--dataset.eval_episodes` and `--eval_steps`.

### Prepare statistics (optional)

Inspect statistics for all numeric AlohaMini dataset fields without modifying data:

```bash
alohamini dataset stats ~/Alohamini_workspace/datasets/task_demo_ready \
  --output ~/Alohamini_workspace/logs/task_demo_stats.json
```

Generate normalization statistics for the policy, input fields, time windows, and training episodes in a training JSON (AlohaMini and LeRobot v3 supported):

```bash
alohamini dataset stats ~/Alohamini_workspace/datasets/task_demo_ready \
  --config /path/to/train.json \
  --output ~/Alohamini_workspace/logs/task_demo_training_stats.json

python -m alohamini.learning.train --config /path/to/train.json \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --stats "$HOME/Alohamini_workspace/logs/task_demo_training_stats.json" --background
```

`--stats` accepts only statistics generated with `--config`; regenerate them if the data or sample configuration changes.
Without `--stats`, training computes statistics automatically. New runs save `runs/<NAME>/statistics.json`; checkpoints retain the statistics actually used, and resumed training reuses checkpoint statistics.

### Logs and configuration

```bash
tail -f ~/Alohamini_workspace/logs/training/act_01.log
cat ~/Alohamini_workspace/logs/training/act_01.pid
```

CLI training runs in the background by default, printing separate log, PID, and configuration paths plus a copyable `tail -f` command. The output directory must be unused.
Use `--foreground` to run directly in the current terminal.
Closing the terminal or pressing Ctrl+C to leave `tail -f` stops log viewing only; background training continues.
When launched directly by `torchrun`, each rank runs in the current process without starting another background task.
Use CLI arguments alone, or copy `examples/learning/act.json` into the workspace's `configs/`,
edit the data path, run name, and episode indices, and load it with `--config /path/to/train.json`.
CLI arguments take precedence; for example, `--steps=150000` overrides total steps in the configuration.
Keep personal experiment configurations in the workspace, outside the repository's `examples/`.

Raw per-step metrics remain in the run directory's `metrics.jsonl`; resumed runs use `metrics-from-*.jsonl`:

```bash
tail -f ~/Alohamini_workspace/runs/act_01/metrics.jsonl
```

### Validation and video loading

Local numeric data and diagnostic records use a disk cache, with a default 64 MiB budget per dataset per process.
`--dataset.cache_size_mb=64` sets the cache budget; `--dataset.cache_dir=/path/to/cache` selects a cache location outside the source dataset, defaulting to the workspace's `cache/datasets/`. Total memory also includes image decoding, DataLoader prefetching, and batch tensors.

`--dataset.streaming=true` enables mixed sampling from a bounded episode pool for long episodes; `--dataset.streaming_episode_pool_size=8` sets pool size. This changes sample mixing order; validation remains sequential. Resumed training preserves the original sampling mode and pool size.

`--eval_steps` sets the validation interval; `--eval_batch_size` sets its batch size;
`--eval_num_workers` and `--eval_prefetch_factor` configure validation loading independently.
Unspecified values inherit training settings; zero workers disables multiprocessing prefetch.
ACT/AM-ACT validation action loss excludes KL and cannot be directly compared with total training loss.

When no video backend is specified, the trainer uses TorchCodec if installed, otherwise PyAV.
An explicit backend in CLI arguments or configuration (including checkpoint settings) takes precedence over automatic selection;
switch to PyAV with `--video_backend=pyav`. Backend load or decode errors are reported without automatic fallback.
`--video_cache_size` sets the decoder-cache count per worker; `0` disables decoder reuse. Underlying libraries may retain additional memory.
To tune loading performance, set `--video_backend=pyav|torchcodec`, `--camera_workers`,
and `--video_cache_size`. More workers, prefetching, and caching increase host memory use.
Images follow each policy's size, cropping, and normalization rules, saved with the checkpoint.

### Optimizers and schedules

Append the required arguments to the training command:

```text
--optimizer.type=adamw --optimizer.lr=0.0001
--optimizer.weight_decay=0.0001 --optimizer.betas='[0.9,0.999]'
--optimizer.eps=1e-8 --optimizer.grad_clip_norm=10
```

Supported optimizers are `adamw`, `adam`, and `sgd`; defaults follow the policy configuration.
ACT/AM-ACT defaults to no scheduler; SmolVLA uses `cosine`; π0.5/Spirit uses `warmup_cosine`;
Diffusion/FastWAM uses `diffusers_cosine`.
`--scheduler.type=none` disables scheduling. Override `warmup_steps`, `decay_steps`, and `decay_lr` as needed.
`warmup_linear` decays linearly after warmup; it, `staged_cosine`, and `constant_with_warmup`
all support `--scheduler.start_step`.
`--scheduler.before_start` accepts a JSON object mapping parameter-group names to learning-rate multipliers before the stage starts.
Policies declare group names. Unlisted groups default to multiplier 1; unknown names cause an error. Stages count successful optimizer updates.

### Numeric precision

Defaults follow policy presets. Three independent settings control computation, parameter storage, and FP32 acceleration:

| Setting | Argument | Default |
| --- | --- | --- |
| Automatic mixed-precision computation | `--mixed_precision=none\|bfloat16\|float16` | Policy preset, or `none` if unspecified |
| Trainable parameter storage and updates | `--trainable_dtype=preserve\|float32` | Policy preset, or `preserve` if unspecified |
| Allow TF32 for FP32 matrix multiplication and convolution | `--tf32=true\|false` | `true` |

Common combinations to append to training commands:

```text
# FP32 trainable parameters with BF16 mixed-precision computation
--trainable_dtype=float32 --mixed_precision=bfloat16

# Disable automatic mixed precision and TF32
--mixed_precision=none --tf32=false
```

`none` disables automatic mixed precision without converting existing BF16 model weights to FP32.
`trainable_dtype=float32` converts only trainable parameters; frozen parameters and buffers keep their original types. GPU memory use increases.
TF32 changes only FP32 computation, not parameter storage, and can be used alongside BF16/FP16 mixed precision.
With `--cudnn_deterministic=true`, TF32 defaults to disabled; explicit `--tf32` takes precedence.

Settings are saved with checkpoints and reused for inference.
Starting new fine-tuning with `--policy.path` inherits only `trainable_dtype` by default; mixed precision and TF32 follow the new configuration or policy defaults.
Explicit arguments take precedence; `--resume=true` requires the original settings.

### Multiple GPUs and gradient accumulation

| Configuration | Additional arguments |
| --- | --- |
| Single GPU | Default |
| DDP, full model per GPU | `--num_processes=2 --background` |
| FSDP2 sharding | `--distributed_backend=fsdp2 --num_processes=2 --background` |

Select GPUs: `CUDA_VISIBLE_DEVICES=2,5 alohamini train --config train.json --policy.device=cuda --num_processes=2 --background`.
Selected GPUs are renumbered within the process as `cuda:0` and `cuda:1`.

Effective batch size is generally `batch_size × process count × accumulation steps`; `steps` counts successful optimizer updates.
Multiprocess final batches repeat samples for padding by default; `--drop_last=true` drops them instead.
`num_workers`, `prefetch_factor`, and video caches apply per process. Account for combined CPU memory and decoding load across GPUs.
DataLoader uses `spawn` by default; switch with `--dataloader_multiprocessing_context=forkserver`.
`--stats` reuses statistics across runs. Startup synchronization waits up to 3600 seconds by default; adjust with `--dataset_preparation_timeout_s`.
For multiple nodes, use `torchrun ... -m alohamini.learning.train --config train.json`. Nodes must share the output directory and use identical data and asset paths.

Each process needs enough CPU memory to load the model; the main process also needs room for full inference weights when saving.
FSDP2 does not shard input activations; each GPU must still fit the current compute module and its activations. ZeRO and tensor parallelism are unsupported.
FastWAM's external frozen VAE/text encoder still has one copy on each GPU.

### Trainer EMA

EMA follows policy presets and is disabled if unspecified; enable with `--ema.enable=true`.
Default decay settings are `inv_gamma=1.0`, `power=0.75`, `min_decay=0`, `max_decay=0.9999`, and `update_after_step=0`.
For fixed decay, use `--ema.decay=0.99`; this is mutually exclusive with custom `min_decay/max_decay`.
`--ema.schedule=power_law` uses `(1-1/n)^power`, where `n` counts successful post-warmup updates;
the default `inverse_power` uses `1-(1+(n-1)/inv_gamma)^(-power)`. Policies can supply their own exponent and upper bound.

EMA tracks trainable parameters in FP32 and counts successful optimizer updates.
Validation uses EMA weights by default; `--ema.use_for_eval=false` selects online weights.
Single-device, DDP, and FSDP2 are supported; evaluation/export needs temporary memory.

## 5. Saving and resuming

### Save models

`--save_freq=5000` saves every 5000 completed training steps and again at training completion.
The latest model is at `runs/<RUN_NAME>/checkpoint`, ready for fine-tuning or inference.
Keep the entire run directory to resume interrupted training; copying only model weights is insufficient.

With trainer EMA enabled, every complete checkpoint also saves:

| Path | Purpose |
| --- | --- |
| `pretrained_model/` | Online model matching optimizer state, used for resuming training |
| `ema_pretrained_model/` | Deployable EMA model with the same preprocessing metadata |
| `ema` in `training_state/state.pt` | Full EMA state for single-device/DDP; update counter for FSDP2 |
| `distributed/` | FSDP2 model, optimizer, and complete EMA shard state |

In the run directory, `checkpoint` points to the latest online model and `checkpoint-ema` to the latest EMA model.
Use `--policy.path=<RUN_DIR>/checkpoint-ema` to evaluate or deploy EMA.
Resume using `checkpoint/train_config.json` with the EMA configuration unchanged.
Missing or mismatched EMA state prevents resuming.

### Resume interrupted training

If only the process was interrupted and data/settings are unchanged, resume from the saved training configuration:

```bash
python -m alohamini.learning.train \
  --config_path="$HOME/Alohamini_workspace/runs/act_01/checkpoints/last/pretrained_model/train_config.json" \
  --resume=true --background
```

| Option | Effect |
| --- | --- |
| `--config_path` | Read the saved run configuration without reentering data paths or training settings |
| `--resume=true` | Continue from the latest saved step, restoring weights, optimizer, and learning-rate progress |
| `--background` | Run in the background and print log/PID paths after startup |

Keep training/validation data, model, learning rate, scheduler, batch size, precision, and distributed settings unchanged when resuming.
If the original run used `--scheduler.type=none`, append `--steps=150000` to extend training.
This is the cumulative target: after 100000 completed steps, 150000 requests another 50000 steps.
Runs with learning-rate scheduling must retain the original total step count.

### Fine-tune with more data or a new learning rate

When adding demonstrations, using merged data, changing validation episodes, or adjusting the learning rate, use `--policy.path`
to load an existing model and select a new output directory. Do not add `--resume=true`.
All native policies support fine-tuning from platform checkpoints; ACT/AM-ACT is used below:

```bash
python -m alohamini.learning.train \
  --policy.path="$HOME/Alohamini_workspace/runs/act_01/checkpoint" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/merged_task" \
  --dataset.eval_episodes='[1]' \
  --output_dir="$HOME/Alohamini_workspace/runs/act_finetune_01" \
  --policy.device=cuda \
  --optimizer.lr=5e-6 \
  --policy.optimizer_lr_backbone=5e-6 \
  --scheduler.type=none \
  --steps=50000 --batch_size=2 \
  --save_freq=5000 --eval_steps=1000 --log_freq=100 \
  --background
```

Replace the dataset path and validation episode indices. After merging, select validation episodes by their new indices.
Validation demonstrations must not duplicate training demonstrations. Model architecture, cameras, image size, and state selection default to the existing model;
new data must match its action fields, units, and input semantics.

| Option | Effect |
| --- | --- |
| `--policy.path` | Existing model to continue learning from; policy type is detected automatically |
| `--dataset.root` | Dataset directory for this fine-tuning run |
| `--dataset.eval_episodes='[1]'` | Reserve episode 1 for validation; use the rest for training if training episodes are unspecified |
| `--output_dir` | New model directory; must not exist; the original model is retained |
| `--policy.device=cuda` | Train on GPU; `cpu` selects CPU |
| `--steps=50000` | Train 50000 steps in this fine-tuning run, excluding earlier model training |
| `--batch_size=2` | Read 2 samples per process per batch; larger values require more GPU memory |
| `--save_freq=5000` | Save a model every 5000 steps |
| `--eval_steps=1000` | Validate on held-out episodes every 1000 steps; requires validation episodes |
| `--log_freq=100` | Print training metrics every 100 steps |

Fine-tuning retains weights but restarts the optimizer and learning-rate schedule. Adjust steps, batch size, and learning rates for the task.

### Adjust learning rates

The learning rate controls the size of each parameter update. Smaller values generally change an existing model more slowly,
but do not guarantee higher success; select them using validation metrics. `5e-6` means `0.000005`.

| Option | Effect |
| --- | --- |
| `--optimizer.lr=5e-6` | Set the main learning rate; for ACT/AM-ACT, this applies outside the visual backbone |
| `--policy.optimizer_lr_backbone=5e-6` | Set the ACT/AM-ACT visual-backbone learning rate independently, controlling image-feature extractor updates |
| `--scheduler.type=none` | Keep learning rates constant at their configured values throughout training; learning continues |

**Changing only `--optimizer.lr` does not override the ACT/AM-ACT visual-backbone learning rate.**
Set both as above to use the same rate, or update visual features more slowly
with `--optimizer.lr=5e-6 --policy.optimizer_lr_backbone=1e-6`.
Other algorithms use their own parameter groups; do not reuse ACT backbone options indiscriminately.

To increase the learning rate gradually and then reduce it, replace `--scheduler.type=none` in the fine-tuning command with:

```text
--scheduler.type=warmup_cosine \
--scheduler.warmup_steps=1000 \
--scheduler.decay_steps=50000 \
--scheduler.decay_lr=5e-7
```

| Option | Effect |
| --- | --- |
| `--scheduler.type=warmup_cosine` | Warm up, then decrease the learning rate along a cosine curve |
| `--scheduler.warmup_steps=1000` | Increase from a small learning rate to the configured value over the first 1000 steps |
| `--scheduler.decay_steps=50000` | Reach the target at step 50000 of this run, including warmup |
| `--scheduler.decay_lr=5e-7` | Final main learning rate; other parameter groups decrease proportionally |

With a scheduler other than `none`, `--optimizer.lr` is the configured peak, not the actual rate at every step.
With `none`, omit these warmup and decay arguments.

### Choose normalization statistics and view logs

| Fine-tuning option | Effect |
| --- | --- |
| Omitted or `--normalization=dataset` | Recompute statistics from this run's training data by default, excluding validation episodes |
| `--normalization=checkpoint` | Reuse existing model statistics without recomputing from new data |
| `--stats=/path/to/statistics.json` | Use pregenerated statistics matching this data and training configuration; mutually exclusive with the previous option |

Background fine-tuning logs and PID files are in `~/Alohamini_workspace/logs/training/`. View the example log:

```bash
tail -f ~/Alohamini_workspace/logs/training/act_finetune_01.log
```

Leaving `tail -f` does not stop training. Resumed runs add a resume identifier to log names; use the actual path printed at startup.

## 6. Offline evaluation

```bash
python -m alohamini.learning.evaluate \
  --policy.path ~/Alohamini_workspace/runs/act_01/checkpoint \
  --dataset.root ~/Alohamini_workspace/datasets/task_demo_ready \
  --dataset.episodes '[1]' --device cuda \
  --output ~/Alohamini_workspace/runs/act_01/offline-review.json
```

Evaluation uses checkpoint cameras, fields, image sizes, and normalization statistics, reporting per-action-dimension MAE in original units and a data-validation report.
Data must retain matching calibration, action semantics, and FPS; validation episodes must not duplicate training demonstrations. Offline error is not hardware task success rate.

## 7. Hardware evaluation

The PC loads the model and the Raspberry Pi runs the Host. Stop other motion controllers, verify calibration, cameras, and robot pose, and clear the motion area.

Raspberry Pi:

```bash
conda activate alohamini_host
alohamini host --robot_model alohamini2pro --cameras forward wrist_right
```

PC (these commands move the robot):

```bash
conda activate alohamini
alohamini evaluate --host <PI_IP> --robot_model alohamini2pro \
  --policy.path ~/Alohamini_workspace/runs/act_01/checkpoint \
  --device cuda --fps 30 --episode_time 10 --num_episodes 1 \
  --policy.n_action_steps 10 --policy.temporal_ensemble_coeff none
```

Match `--fps` to training data. Each action chunk executes its first `n_action_steps` before predicting again.
For ACT/AM-ACT prediction and temporal ensembling at every step, use `--policy.n_action_steps 1 --policy.temporal_ensemble_coeff 0.01`;
`none` disables ensembling; `0` uses equal weights. Execution settings can change without retraining.

SmolVLA/π0.5/FastWAM/Spirit evaluation also requires `--task "pick up the object"`. Diffusion/FastWAM/Spirit does not use ACT temporal ensembling.
`--dataset eval_01 --task "pick up the object"` saves evaluation episodes. Ctrl+C ends evaluation. Prolonged feedback interruption or joint protection pauses execution; recovery clears old policy caches before resuming. Host restart, calibration changes, or control takeover by another client ends evaluation.
Inference latency limits the actual execution rate.

When moving between machines, copy the complete checkpoint directory, not only symbolic links.
See [Client API](host-protocol.md#python-policy-evaluation) for custom policies and `examples/learning/custom_policy.py` for an example.
See [Notebooks](notebook.md) for notebooks and [LeRobot](lerobot.md) for LeRobot weights.

### DAgger corrections and retraining

Add `--dagger --dataset task_corrections` to evaluation and connect calibrated leader arms; keyboard input requires X11.
Space pauses and holds the follower arms; Tab begins manual alignment. Enable hold on each leader joint after it settles into position;
once all joints align, support the leader arms and press Tab again to release hold and start corrections. Tab ends and saves the segment; Space returns control to the policy; Esc exits.
Handing control back clears old actions and interpolation caches, restarting inference from the current observation.

Only human corrections are recorded, each as a separate episode; pauses and alignment are excluded from training sequences.
Align gripper openings too. During correction, the base stops and lift holds. The default leader hold output limit is `0.2`,
adjustable with `--dagger.hold_output_limit`; joint alignment tolerance defaults to 3 degrees (`--dagger.alignment_tolerance_deg`).
Holding depends on actual power and manual support. Faults, feedback interruption, or alignment timeout release leader hold.
Slow synchronous inference responds to pause after the current computation returns; the asynchronous mode below is recommended. Fixed arm dimensions are unsupported.

Merge demonstrations and corrections, then begin a new fine-tuning run:

```bash
alohamini dataset edit --operation.type merge \
  --operation.roots '["~/Alohamini_workspace/datasets/task_demo", "~/Alohamini_workspace/datasets/task_corrections"]' \
  --output ~/Alohamini_workspace/datasets/task_merged

python -m alohamini.learning.train \
  --policy.path ~/Alohamini_workspace/runs/act_01/checkpoint \
  --dataset.root ~/Alohamini_workspace/datasets/task_merged \
  --output_dir ~/Alohamini_workspace/runs/act_dagger_01 \
  --policy.device cuda --steps 10000 --batch_size 8 --background
```

Both datasets must share fields, cameras, calibration, and FPS. Select an independent validation set using merged episode indices.
Do not add `--resume=true` for new fine-tuning. Logs are at `~/Alohamini_workspace/logs/training/act_dagger_01.log`;
after training, point evaluation's `--policy.path` to the new checkpoint and continue collecting.

### Inference modes

Add to the local evaluation command:

```bash
--inference.type=rtc
```

| `--inference.type` | Behavior |
|---|---|
| `sync` (local default) | Predict after executing the current action chunk; brake and hold between TCP chunks |
| `async` | Background prediction; ACT/DP defaults to time-based switching, BSP to phase alignment |
| `rtc` | Background prediction with denoising-prefix guidance; currently supports π0.5/SmolVLA |

For `async`, `--inference.continuity=none` disables the policy's default transition method;
`--inference.continuity=phase` is BSP-only. `rtc` needs no separate continuity selection.
Disable temporal ensembling for asynchronous ACT: `--policy.temporal_ensemble_coeff none`.

Policies supporting language planning accept the following in local `evaluate` or cloud `inference serve`:
`--language-replan-interval-s=2 --task='task instruction'`. Disabled by default; subtasks update by observation time, clearing the old action queue on changes and resetting each episode.

Asynchronous mode replans automatically using remaining trajectory time and prediction latency. Adjust if needed:

| Argument | Default | Purpose |
|---|---|---|
| `--inference.replan_interval_s` | Automatic (`0.25` for TCP) | Replanning time budget in seconds |
| `--inference.prediction_lead_s` | `0.1` | Minimum advance-prediction budget in seconds |
| `--inference.max_prediction_age_s` | `5` | Maximum source-observation age in seconds |
| `--inference.startup_timeout_s` | `15` | Maximum wait for first prediction and between synchronous TCP chunks, in seconds |

For TCP and asynchronous execution, `--fps` is the command-query rate and may differ from checkpoint data FPS.

Loading a TCP-action checkpoint automatically enables TCP tracking. `sync` and `async` share rate limits and recovery; install the `kinematics` extension on the execution machine. Tracking slows automatically when lagging and brakes to hold when predictions expire. Optional reference limits follow (defaults; velocity/acceleration use m/s, m/s² or rad/s, rad/s²):

```bash
--inference.tcp_tracking='{
  "max_tcp_linear_velocity":0.2, "max_tcp_linear_acceleration":0.5,
  "max_tcp_angular_velocity":1.0, "max_tcp_angular_acceleration":2.0
}'
```

Joint position, velocity, and acceleration limits still apply. IK failure brakes to hold and requests a new prediction; execution stops after 1 s or three consecutive failed new predictions. Normal synchronous inter-chunk waits use `startup_timeout_s`.
TCP currently does not support `rtc`, ACT temporal ensembling, individual arm-joint fixed overrides, or Host joint-action scaling. Reference tracking permits path deviation, includes no collision avoidance or jerk constraints, and has not completed hardware validation.

Old plans are discarded when predictions expire, valid trajectories run out, or Host state changes. Asynchronous TCP waits only within the recovery budget above.
Evaluation records execution parameters and source-observation times. Cloud serving defaults to `async`, supports `rtc`, and also allows `sync` for TCP-action checkpoints;
it uses the same `--inference.*` arguments, with the Pi reading configuration from the server handshake.
Both machines must use the same platform version.

### Fixed dimensions in training and inference

If data declares no fixed dimensions, explicitly add them to evaluation, for example:

```bash
alohamini evaluate --host <PI_IP> --robot_model alohamini2pro \
  --policy.path ~/Alohamini_workspace/runs/pretrained_model \
  --device cuda --fps 30 --episode_time 10 \
  --fixed-dimensions arm_right \
  --fixed-dataset ~/Alohamini_workspace/datasets/task_demo_ready
```

Left/right arm, lift, base groups, and named joints are supported, using the same fields as collection options. Existing fixed configuration is inherited; additional selection only adds fixed dimensions.
New position targets use the training **action mean in physical coordinates**. Corresponding state inputs for older models use the checkpoint's
**state mean**. Real feedback remains available for safety checks and recording.
A fixed base still receives zero velocity. Fixing dimensions active during training changes deployment conditions and does not reproduce original task performance.
To hold positions arranged through teleoperation, `--fixed-current` requires no `--fixed-dataset` and reads no training data to select positions. It supports `arm_left`, `arm_right`,
`lift_axis`, `base`, and individual named action fields. After model loading, the first fresh, controllable feedback in the Host handshake determines targets;
execution waits for stability before inference. Later episodes in the same run reuse these targets without resampling. The base remains fixed at zero velocity.
Selected fields override inherited checkpoint fixed targets; all other inherited targets remain unchanged.

Specify the two sources separately, for example `--fixed-dimensions arm_right --fixed-current lift_axis`,
to use the right arm's existing fixed target or training action mean while holding the lift at startup height. A field cannot appear in both selections.
Omitting `--fixed-dataset` alone does not change the original mean-based meaning of `--fixed-dimensions`.

### Cloud inference

The Raspberry Pi sends images/state to a cloud GPU, receives trajectory plans, and executes them locally. The PC handles remote administration.
Supported native policies declare remote-trajectory capability, including ACT, Diffusion, BSP, PI05, SmolVLA, and Spirit.
Use the same Alohamini version on the cloud and Raspberry Pi. Start a local Host on the Pi, complete calibration and lift homing, and stop other control clients.
Keep the Host address at `127.0.0.1`. Connect ZeroMQ inference port `5560` through an SSH tunnel or WireGuard; do not expose it directly to the public internet.

Install [policy and communication extensions](install.md#3-optional-extensions) on the cloud; the Pi reuses its Host environment.

Server examples below use `pi05_01/checkpoint`; substitute your model directory and task text.
In SSH commands, replace `PORT`, `USER`, and `CLOUD` with the cloud SSH port, username, and address.
After changing rented instances, update the SSH address/port. For WireGuard, also update the cloud public key and the Pi's `Endpoint`.

#### SSH tunnel

Start the inference service on the cloud:

```bash
alohamini inference serve --bind tcp://127.0.0.1:5560 \
  --policy.path "$HOME/Alohamini_workspace/runs/pi05_01/checkpoint" \
  --device cuda --task "pick up the object"
```

Wait for `Remote policy server listening on tcp://127.0.0.1:5560`, then establish the tunnel on the Pi and leave the terminal running:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:5560:127.0.0.1:5560 -p PORT USER@CLOUD
```

Verify the cloud SSH host fingerprint on first connection. SSH must permit local forwarding, and its network environment must reach
the inference service at cloud `127.0.0.1:5560`. [OpenSSH forwarding documentation](https://man.openbsd.org/ssh.1)

Run in another Raspberry Pi terminal:

```bash
alohamini inference run --server tcp://127.0.0.1:5560 \
  --host 127.0.0.1 --robot_model alohamini2pro --fps 30
```

`ExitOnForwardFailure` checks only the forwarding listener; SSH keepalive does not replace action expiration or the control watchdog.
[OpenSSH configuration documentation](https://man.openbsd.org/ssh_config.5)
To stop, first press Ctrl+C in the Pi inference client, then close the tunnel and cloud service.
If local Pi port `5560` is occupied, change both the first port in `-L` and the `--server` port, keeping the cloud target port unchanged.

#### WireGuard

Both machines must be able to create WireGuard interfaces, and the cloud needs a reachable UDP port. Rented containers may lack network-administration permissions,
WireGuard kernel support, or `/dev/net/tun` for a userspace implementation. Use an SSH tunnel if these requirements are unmet.
Install [WireGuard tools](https://www.wireguard.com/install/) for your system.

Generate keys once on each machine; these commands refuse to overwrite existing files:

```bash
(
    umask 077
    set -eC
    mkdir -p "$HOME/.config/alohamini/wireguard"
    chmod 700 "$HOME/.config/alohamini/wireguard"
    wg genkey > "$HOME/.config/alohamini/wireguard/private.key"
    wg pubkey < "$HOME/.config/alohamini/wireguard/private.key" > "$HOME/.config/alohamini/wireguard/public.key"
)
```

Exchange public keys only. Keep private keys and completed configurations outside the repository on their respective machines, with mode `600`.
[Key generation documentation](https://www.wireguard.com/quickstart/)
The Pi key can be reused after changing cloud instances; put its public key in the new cloud configuration.

On the cloud, copy the [cloud template](../deploy/remote-inference/wireguard-cloud.conf.example):

```bash
cp -n deploy/remote-inference/wireguard-cloud.conf.example "$HOME/.config/alohamini/wireguard/wg-infer.conf"
chmod 600 "$HOME/.config/alohamini/wireguard/wg-infer.conf"
```

On the Pi, copy the [Raspberry Pi template](../deploy/remote-inference/wireguard-pi.conf.example):

```bash
cp -n deploy/remote-inference/wireguard-pi.conf.example "$HOME/.config/alohamini/wireguard/wg-infer.conf"
chmod 600 "$HOME/.config/alohamini/wireguard/wg-infer.conf"
```

Edit `wg-infer.conf` locally on each machine, entering its private key and its peer's public key. On the Pi, replace `CLOUD_PUBLIC_HOST` with the cloud public address.
Templates use cloud `10.88.0.1/32` and Pi `10.88.0.2/32`, routing only the peer `/32`, without NAT, forwarding, or an internet default route.
For address conflicts, choose new addresses on both ends and update configuration and commands consistently.
[Routing documentation](https://git.zx2c4.com/wireguard-tools/about/src/man/wg-quick.8)

Allow UDP `51820` through the cloud security group and firewall. Allow inference TCP `5560` only from the Pi on the WireGuard interface.
If the provider maps another external UDP port, change the Pi's `Endpoint`; keep the cloud `ListenPort` at its internal port.
`PersistentKeepalive = 25` maintains NAT mappings.
[WireGuard configuration documentation](https://git.zx2c4.com/wireguard-tools/about/src/man/wg.8)

Start on both machines:

```bash
sudo wg-quick up "$HOME/.config/alohamini/wireguard/wg-infer.conf"
sudo wg show wg-infer
```

On the Pi, test target connectivity with `ping -c 3 10.88.0.1`. If ICMP is blocked, still check the handshake, routes, and service listener.
Start on the cloud:

```bash
alohamini inference serve --bind tcp://10.88.0.1:5560 \
  --policy.path "$HOME/Alohamini_workspace/runs/pi05_01/checkpoint" \
  --device cuda --task "pick up the object"
```

After the service is ready, run on the Pi:

```bash
alohamini inference run --server tcp://10.88.0.1:5560 \
  --host 127.0.0.1 --robot_model alohamini2pro --fps 30
```

After stopping the inference client and server, close the tunnel on both machines as needed:

```bash
sudo wg-quick down "$HOME/.config/alohamini/wireguard/wg-infer.conf"
```

#### Connection and execution settings

If the SSH tunnel fails, check that the cloud inference process and SSH forwarding target share the same network environment.
Without a WireGuard handshake, check public keys, UDP ingress, and `Endpoint`. With a handshake, check `/32` routes, binding address, and interface firewall.
Inspect the inference listener on the cloud:

```bash
ss -ltn '( sport = :5560 )'
```

`--fps` sets the Pi's command-query rate; checkpoint data frequency determines trajectory timing.
For dense actions, server `--chunk-size 32` shortens the executable window without exceeding the model window; BSP does not accept it.
TCP also supports `--chunk-size`; the execution machine needs the `kinematics` extension.
These settings do not guarantee cloud results at the execution rate. Late results skip expired prefixes. Exhausted actions or changes to the Host session/control reference stop execution and require an explicit restart.

| Raspberry Pi `run` argument | Default | Purpose |
| --- | --- | --- |
| `--episode_time` | 90 s | Episode duration |
| `--rpc-timeout-s` | 10 s | Cloud request timeout |
| `--max-prediction-age-s` | 1 s | Maximum prediction-source observation age |
| `--ack-timeout-s` | 0.5 s | Host acknowledgement timeout |

To fix current joints, pass identical `--fixed-current FIELD ...` selections to cloud `serve` and Pi `run`.
Supported fields are `base`, `lift_axis`, `arm_left`, `arm_right`, and named Host fields. The Pi captures fixed targets from fresh feedback.
PI05/SmolVLA can enable server-side [RTC guidance](#inference-modes).
Spirit uses ordinary action-chunk inference. Evaluate actual network latency and hardware success separately; do not compensate for latency by relaxing the control watchdog.

## 8. Compare evaluation runs and training demonstrations

`alohamini dataset analyze` locally inspects saved evaluation records and compares them with training or validation demonstrations from the checkpoint used for evaluation. No robot connection is needed.

| Feature | Supported algorithms |
| --- | --- |
| Rerun images/action curves, demonstration comparisons, report export | Platform algorithms, including ACT, AM-ACT, Diffusion Policy, FastWAM, SmolVLA, and π0.5 |
| Add `--probe-samples` to predict demonstration actions again and compute errors | Platform policies, checked against their declared inference inputs and action windows |

The examples use evaluation dataset `eval_01`. Replace paths, episode indices, and time points; episode indices start at 0.

### Open evaluation records

```bash
alohamini dataset analyze ~/Alohamini_workspace/datasets/eval_01 \
  --episode 0
```

By default, Rerun opens camera images and action curves for evaluation episode 0.
With model-association metadata, it automatically locates the checkpoint and original training dataset, showing the first demonstration from the saved training split.
Ordinary inspection runs no model and requires no GPU. Commands, Host feedback, current, and timings are hidden by default.

If evaluation records lack model-association metadata, use `--checkpoint` to specify **the model used for that evaluation**.
If the model or original training dataset has moved, provide the corresponding paths:

```bash
alohamini dataset analyze ~/Alohamini_workspace/datasets/eval_01 \
  --episode 0 \
  --checkpoint ~/Alohamini_workspace/runs/pickup_01/checkpoint \
  --training-root ~/Alohamini_workspace/datasets/pickup_train
```

`--training-root` must point to a local copy of the original training dataset; it does not substitute another dataset.

### Select demonstrations and align grasp phases

If evaluation episode 0 begins grasping at 20 s and training demonstration 12 at 8 s, compare the next 15 s from each:

```bash
alohamini dataset analyze ~/Alohamini_workspace/datasets/eval_01 \
  --episode 0 --reference-split train --reference-episode 12 \
  --time-range 20 35 --reference-time-range 8 23 \
  --align 20 8
```

| Option | Meaning in this example |
| --- | --- |
| `--episode 0` | Inspect episode 0 in the evaluation dataset |
| `--reference-split train` | Select from the checkpoint's saved training split; `val` selects validation |
| `--reference-episode 12` | Compare original training-dataset episode 12, which must belong to the selected split |
| `--time-range 20 35` | Select evaluation seconds 20–35 |
| `--reference-time-range 8 23` | Select demonstration seconds 8–23 |
| `--align 20 8` | Align evaluation second 20 and demonstration second 8 at second 0 on a shared timeline |

Omitting `--reference-episode` selects the split's first demonstration. Without `--align`, episodes are placed side by side from their starts; grasp phases may not align.
Alignment does not change playback speed. Images and curves reveal action order, arm selection, and trajectory differences; no framewise action error is computed between the runs.

### Export charts and image reports

Using the same segments, export evaluation images at seconds 21, 25, and 30 with aligned demonstration images:

```bash
alohamini dataset analyze ~/Alohamini_workspace/datasets/eval_01 \
  --episode 0 --reference-split train --reference-episode 12 \
  --time-range 20 35 --reference-time-range 8 23 --align 20 8 \
  --viewer none --export-html --keyframes 21 25 30 \
  --output ~/Alohamini_workspace/reports/eval_01_compare
```

`--viewer none` skips Rerun; open `report.html` in the output directory with a browser to view charts and images.
`--output` must be a new directory outside the dataset. When omitted, results go to
`~/Alohamini_workspace/reports/<DATASET>/episode_<INDEX>/<TIMESTAMP>/`.
Each analysis saves `summary.json` and `events.csv`; HTML report charts are also saved under `plots/`.

### Check model predictions on demonstrations

This command loads the checkpoint associated with evaluation, uniformly samples up to 8 observations from its first validation demonstration, and predicts actions again:

```bash
alohamini dataset analyze ~/Alohamini_workspace/datasets/eval_01 \
  --episode 0 --reference-split val \
  --probe-samples 8 --probe-device cuda \
  --viewer none --export-html
```

The checkpoint must contain a nonempty validation split. Add `--reference-episode` to choose a particular demonstration.
If all demonstrations were used for training with no validation holdout, use `--reference-split train` to inspect training fit.
`--training-root` only selects the data location; it creates no validation split.
`--probe-device cuda` uses GPU; change to `cpu` for CPU execution.

The report shows first-step and valid-action-segment mean absolute error (MAE) per action dimension, in demonstration action units.
The first step is the current action (row offset 0); segment statistics include only current/future labels, excluding historical prefixes and episode-end padding.
Predictions use demonstration observations and compare with **subsequent ground-truth actions from that same demonstration**, not actions from the evaluation episode.
Inputs include required task text and observation history. CDP also uses strictly past demonstration actions, never current/future action labels.
The model resets independently for each query, without reproducing online action queues, temporal ensembling, or RTC. FastWAM reads only current images, excluding future images used in training.
The hold baseline repeats current observations with matching names and units; dimensions with unverifiable coordinate units have no baseline. Valid step counts are reported separately for model and baseline.
`demonstration_probe.json` stores metrics, model fingerprints, and sampling configuration; `demonstration_predictions.npz`
stores predictions, labels, field units, row offsets, and validity masks. This error checks predictions and does not represent hardware success.
Use `--reference-split train` to inspect training fit; prefer validation demonstrations for generalization checks.
Without `--probe-samples`, the model is not loaded.

### Other viewing options

| Option | Effect |
| --- | --- |
| `--cameras wrist_left wrist_right` | Show/export only these cameras in the given order |
| `--save-rrd` | Save `recording.rrd`, then open it with the printed `rerun` command |
| `--control-diagnostics` | Also show commands, Host feedback, current, and timings |
| `--probe-seed 0` | Set the prediction random seed for repeatable comparisons |
| `--segment 0` / `--reference-segment 0` | Select evaluation/demonstration time segments after clock resets |

The PC environment includes viewing dependencies. For standalone analysis, install the [`analysis` extension](install.md#3-optional-extensions).
Prediction probes also require the relevant policy's runtime dependencies.

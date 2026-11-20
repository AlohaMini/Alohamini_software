# Multi-Task DiT

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Examples reserve episode 1 for validation; for single-episode data, omit `--dataset.eval_episodes` and `--eval_steps`.

```bash
conda activate alohamini
python -m alohamini.learning.train \
  --policy.type=multi_task_dit --policy.device=cuda \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --eval_steps=1000 \
  --cameras='["forward","wrist_right"]' \
  --output_dir="$HOME/Alohamini_workspace/runs/multi_task_dit_01" \
  --batch_size=2 --gradient_accumulation_steps=8 --steps=100000 \
  --ema.enable=true --save_freq=10000 --log_freq=100 --background
```

| Item | Default configuration |
| --- | --- |
| Observation / prediction / execution | 2 frames / 32 steps / 24 steps; exclude final 7 sampling starts |
| DiT | Hidden dimension 512, 6 layers, 8 heads, AdaLN-Zero, RoPE |
| Vision | CLIP ViT-B/16, shared camera encoder; resize to 256×256, random training / center inference crop to 224×224 |
| Language | Required nonempty `task`, up to 77 tokens; frozen text encoder, trainable projection |
| Normalization | RGB: CLIP `MEAN_STD`; state/action: training-set `MIN_MAX`, original fields/units retained |
| Optimization | Adam `2e-5`, visual learning rate ×0.1; cosine, no warmup |
| Generation objective | Diffusion, 100 DDPM steps |

State is required. CLIP weights/tokenizer default to workspace `pretrained/clip-vit-base-patch16/`
(workspace defaults to `~/Alohamini_workspace/`, respecting `ALOHAMINI_WORKSPACE`). Download `openai/clip-vit-base-patch16` beforehand.
`--policy.vision_encoder_name` and `--policy.text_encoder_name` accept local directories or online model IDs. Missing local files cause an error. Set `HF_HUB_OFFLINE=1` to prevent online model-ID loading.
Trailing-start exclusion uses the same temporal formula; `--policy.do_mask_loss_for_padding=true` masks padded-action loss.

```bash
# DDIM
--policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10

# Flow matching; integration_method also supports rk4
--policy.objective=flow_matching --policy.num_integration_steps=10 \
  --policy.integration_method=euler --policy.timestep_sampling_strategy=beta
```

Supports single-device, DDP, FSDP2, and trainer EMA.
Checkpoints include weights, CLIP configuration, and tokenizer. Use `checkpoint-ema` for EMA deployment and the online checkpoint for resume.

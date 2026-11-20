# Long-WAM

[Policy index](../../../../docs/training.md#policy-selection) · [Common training arguments](../../../../docs/training.md)

The local base directory needs converted `config.json`, `model.safetensors` (or shards and index),
and a UMT5 tokenizer under `tokenizer/`. Model configuration must include video, VAE, and text-encoder configurations.

```bash
alohamini train --policy.type=longwam \
  --policy.path="$HOME/Alohamini_workspace/pretrained/longwam" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --policy.reinitialize_embodiment_heads=true \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --distributed_backend=fsdp2 --num_processes=4 \
  --output_dir="$HOME/Alohamini_workspace/runs/longwam_01" --background
```

Defaults: BF16, 48 history frames, 32 future supervision frames, images sampled every 4 frames; inference replans after 10 executed steps.
Video/action expert layer counts both follow `video_config.num_layers`, originally 30.
Camera order, field names, units, and training statistics are saved with checkpoints.
Actions use coordinates declared in the data. Different robot dimensions require explicit permission to reinitialize input/output projections.
Synchronous inference is supported; remote action chunks, asynchronous execution, and RTC are currently unsupported. Inference uses no future images.

# EO-1

[Policy index](../../../../docs/training.md#policy-selection)

The base directory needs converted `config.json`, `model.safetensors` (or shards and index),
and Qwen2.5-VL processor/tokenizer under `assets/`. Configuration must retain `vlm_config`.

```bash
alohamini train --policy.type=eo1 \
  --policy.path="$HOME/Alohamini_workspace/pretrained/eo1" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/eo1_01" --background
```

Defaults use flow fine-tuning with robot numeric supervision, excluding CoT/text-answer training.
Selected cameras feed the Qwen2.5-VL image processor in order; pixel budget and backbone layer count are saved in configuration.
State/action uses training-set mean/std and pads to 32 dimensions; 16 action steps are predicted by default.
Trainable parameters default to FP32 storage with BF16 mixed-precision computation. RTC is currently unsupported.

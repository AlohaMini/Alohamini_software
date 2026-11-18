# Long-WAM

[策略索引](../../../../docs/training.md#策略选择) · [公共训练参数](../../../../docs/training.md)

本地基座目录需包含转换后的 `config.json`、`model.safetensors`（或分片与索引），
以及 `tokenizer/` 下的 UMT5 tokenizer。模型配置须包含 video、VAE 和 text encoder 配置。

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

默认使用 BF16、48 帧历史、32 帧未来监督、每 4 帧采样图像；推理执行 10 步后重规划。
视频与动作专家的层数统一读取 `video_config.num_layers`，原版为 30 层。
相机顺序、字段名、单位和训练统计量随 checkpoint 保存。
动作采用数据中已声明的坐标；不同机器人维度需要显式允许重新初始化输入、输出投影。
支持同步推理；暂不支持远程动作块、异步执行或 RTC。推理不使用未来图像。

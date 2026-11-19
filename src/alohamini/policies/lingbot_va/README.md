# LingBot-VA

[策略索引](../../../../docs/training.md#策略选择)

GPU 训练与推理使用 CUDA。

下载本地基座，目录须包含 `transformer/`、`vae/`、`text_encoder/` 和 `tokenizer/`：

```bash
BASE="$HOME/Alohamini_workspace/pretrained/lingbot_va"
hf download robbyant/lingbot-va-base --revision=68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c \
  --local-dir="$BASE"

alohamini train --policy.type=lingbot_va --policy.path="$BASE" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/lingbot_va_01" --background
```

默认共享 Transformer 为 30 层，实际结构读取基座 `transformer/config.json`。
默认 FP32 可训练参数、BF16 混合精度与冻结编码器，开启重计算和 GPU Flex Attention；
推理自动使用 SDPA，无需修改基座配置。
FP32 调试可设 `--policy.dtype=float32 --mixed_precision=none`。
多卡与参数分片使用[公共训练配置](../../../../docs/training.md#多卡与梯度累积)。

默认按所选相机顺序，将各路 128×128 图像的 VAE latent 横向拼接。
三路 T 布局需显式设置 `--policy.camera_layout=robotwin_tshape`，并用
`--policy.obs_cam_keys='["forward","wrist_left","wrist_right"]'` 声明头、左腕、右腕顺序。
state 不参与模型，仍可供公共字段变换使用。action 按记录的字段名和单位训练，
最多 30 个坐标，默认重置机器人动作输入、输出头。
保留同一机器人头需设置 `reinitialize_robot_heads=false` 并提供完整的 `action_slots` 名称映射。

动作使用训练集 q01/q99，分母加 `1e-6`；默认保留边界外动作。
`--policy.normalization_recipe=lingbot` 将归一化训练目标限制到 `[-1.5,1.5]`。
默认 `frame_chunk_size=4`、`action_per_frame=4`，每块输出 16 个未来动作；
训练需 17 帧完整窗口，回合尾裁除 16 个起点。初始条件帧另计，零动作前缀只用于条件。
视频与动作默认分别去噪 20、50 步，使用 `num_inference_steps`、`action_num_inference_steps` 调整。

首次训练由主进程在 CPU 编码固定任务文本，缓存到工作区 `pretrained/frozen_features/`；
缓存会随 checkpoint 保存，各 rank 无需重复加载 UMT5。
新指令仍需本地文本编码器；VAE 和 tokenizer 也须保留在 `wan_pretrained_path`。
`--policy.precompute_text_features=false` 可关闭预计算。

支持独立离线动作块、单环境同步推理及逐帧观测反馈；当前不支持远程动作块、异步执行、RTC 或 LoRA。

# Multi-Task DiT

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

命令在仓库根目录执行。示例保留 episode 1 用于验证；单回合数据省略 `--dataset.eval_episodes` 和 `--eval_steps`。

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

| 项目 | 默认配置 |
| --- | --- |
| 观测／预测／执行 | 2 帧／32 步／24 步；排除最后 7 个采样起点 |
| DiT | 隐藏维度 512、6 层、8 heads、AdaLN-Zero、RoPE |
| 视觉 | CLIP ViT-B/16，共享相机编码器；缩放 256×256，训练随机／推理中心裁剪至 224×224 |
| 语言 | 必需的非空 `task`，最长 77 tokens；文本编码器冻结，投影层训练 |
| 归一化 | RGB：CLIP `MEAN_STD`；state/action：训练集 `MIN_MAX`，保留原字段与单位 |
| 优化 | Adam `2e-5`，视觉学习率 ×0.1；cosine，无 warmup |
| 生成目标 | diffusion，DDPM 100 步 |

state 必须存在。CLIP 权重与 tokenizer 默认读取工作区 `pretrained/clip-vit-base-patch16/`
（默认 `~/Alohamini_workspace/`，支持 `ALOHAMINI_WORKSPACE`），须提前下载 `openai/clip-vit-base-patch16`。
`--policy.vision_encoder_name`、`--policy.text_encoder_name` 可指定本地目录或在线模型 ID；本地目录缺文件时报错。设置 `HF_HUB_OFFLINE=1` 可禁止模型 ID 联网加载。
尾部排除数按同一时序公式计算；`--policy.do_mask_loss_for_padding=true` 可屏蔽补齐动作的损失。

```bash
# DDIM
--policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10

# Flow matching；integration_method 也支持 rk4
--policy.objective=flow_matching --policy.num_integration_steps=10 \
  --policy.integration_method=euler --policy.timestep_sampling_strategy=beta
```

支持单设备、DDP、FSDP2 和训练器 EMA。
checkpoint 打包权重、CLIP 配置和 tokenizer；EMA 部署使用 `checkpoint-ema`，续训使用在线 checkpoint。

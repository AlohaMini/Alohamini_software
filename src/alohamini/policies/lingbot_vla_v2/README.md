# LingBot-VLA 2.0

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

命令在仓库根目录执行。示例中的 episode 1 用于验证；数据只有一个回合时，省略 `--dataset.eval_episodes` 和 `--eval_steps`。

使用 `lingbot_vla_v2`，本地基座包含官方 LeRobot 格式 `config.json`、`model.safetensors`，
以及 Qwen 配置、tokenizer、图像处理资源。教师目录包含 `moge/model.pt`、`depth/model.pt`、
`dino_video/teacher_step_10000.pth` 和 `dino_video/config.yaml`。

```bash
alohamini train --policy.type=lingbot_vla_v2 \
  --policy.path="$HOME/Alohamini_workspace/pretrained/lingbot_vla_v2_base" \
  --policy.teacher_root="$HOME/Alohamini_workspace/pretrained/lingbot_vla_v2_teachers" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --distributed_backend=fsdp2 --num_processes=4 \
  --output_dir="$HOME/Alohamini_workspace/runs/lingbot_01" --background
```

默认读取 `forward,wrist_left,wrist_right` 三路 RGB、完整 18 维状态与任务文本，映射到 55 维语义槽。
关节动作相对当前观测，夹爪和底盘速度保留绝对值；动作转换后计算训练集 mean/std。
升降轴默认保持当前高度；显式设置 `--policy.lift_mode=reserved` 时使用平台扩展槽训练升降动作。
训练使用未来帧计算教师损失，推理只读取当前帧，部署无需教师权重。
优化器采用平台 AdamW 并保留 MoE 专家学习率倍率；尚不支持 Muon 和 RTC 执行。
显卡数和 batch 按模型显存需求调整。

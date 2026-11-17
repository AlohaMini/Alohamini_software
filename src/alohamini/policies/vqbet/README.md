# VQ-BeT

[策略索引](../../../../docs/training.md#策略选择)

```bash
alohamini train --policy.type=vqbet \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --steps=100000 --batch_size=64 \
  --output_dir="$HOME/Alohamini_workspace/runs/vqbet_01" --background
```

首次训练需要 ResNet18 ImageNet-1K V1 权重；可预先放入平台视觉权重缓存。
从随机视觉权重训练可显式设置 `--policy.pretrained_backbone_weights=null`。
默认使用 5 帧历史、3 个预测位置，每个位置预测 5 步动作；裁除 episode 两端不完整窗口。
前 20000 次成功优化器更新训练量化器，此后训练动作预测网络；阶段和计数随 checkpoint 保存。
训练步数须大于 `--policy.n_vqvae_training_steps`。
支持同步动作队列；暂不支持远程动作块、异步执行或 RTC。

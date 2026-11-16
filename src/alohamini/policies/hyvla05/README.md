# HyVLA-0.5

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

命令在仓库根目录执行。示例中的 episode 1 用于验证；数据只有一个回合时，省略 `--dataset.eval_episodes` 和 `--eval_steps`。

```bash
hf download tencent/Hy-Embodied-0.5-VLA-RoboTwin \
  --revision bd7bba6f5934ad62293a2a34f74760c6a3ef2ff8 \
  --local-dir "$HOME/Alohamini_workspace/pretrained/hyvla05"

alohamini train --policy.type=hyvla05 \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --policy.path="$HOME/Alohamini_workspace/pretrained/hyvla05" \
  --run_name=hyvla05_task --batch_size=1 --steps=60000 --background
```

需要非空任务文本与 state，相机按 `--cameras` 或数据集顺序输入；图像等比例补边至 224×224。
默认 6 帧历史、间隔 15 行，动作块 20 行、流匹配采样 10 步、BF16；state/action 使用训练集均值／标准差。
动作是连续 20 行原始记录目标，不套用 RoboTwin 的相对／绝对姿态拼接，不自动执行 FK/IK。
在线历史按客户端时间戳读取过去观测，`reset()` 清空历史和动作队列。

`--policy.network='{"freeze_vision_encoder":true}'` 可冻结视觉，图像特征仍参与训练。
运行时只读本地资源，checkpoint 自带 tokenizer 与权重；支持 DDP/FSDP2、训练器 EMA、普通远程动作块。
未接入 RTC、FlowPRO 或 Bézier 拼接。来源与预设见 [SOURCE.json](SOURCE.json)。

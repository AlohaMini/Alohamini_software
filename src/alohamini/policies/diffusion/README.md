# Diffusion Policy

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

命令在仓库根目录执行。示例保留 episode 1 用于验证；单回合数据省略 `--dataset.eval_episodes` 和 `--eval_steps`。

```bash
python -m alohamini.learning.train \
  --policy.type=diffusion --policy.device=cuda \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' \
  --cameras='["forward","wrist_right"]' \
  --output_dir="$HOME/Alohamini_workspace/runs/diffusion_01" \
  --batch_size=2 --save_freq=10000 --log_freq=100 --background
```

默认 2 帧观测、64 步预测、执行 32 步，DDPM 100 步；视觉配置参考
[LeRobot PR #3202](https://github.com/huggingface/lerobot/pull/3202)。
RGB 使用 ImageNet `MEAN_STD`，state/action 使用训练集 `MIN_MAX`。
`--policy.n_action_steps` 调整执行步数，`--policy.num_inference_steps` 调整去噪步数。

尾部排除数默认按 `horizon - n_action_steps - n_obs_steps + 1` 计算，为 **31**；
`--policy.drop_n_last_frames` 可显式覆盖，设为 `0` 保留所有采样起点。
EMA 默认关闭，使用 `--ema.enable=true` 开启。

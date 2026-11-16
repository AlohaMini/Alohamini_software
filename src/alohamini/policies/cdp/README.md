# CDP

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

命令在仓库根目录执行。示例保留 episode 1 用于验证；单回合数据省略 `--dataset.eval_episodes` 和 `--eval_steps`。

```bash
python -m alohamini.learning.train \
  --policy.type=cdp --policy.device=cuda \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --batch_size=2 \
  --output_dir="$HOME/Alohamini_workspace/runs/cdp_01" --background
```

采用 [CDP2 RGB 配置](https://github.com/GaavaMa/Causal-Diffusion-Policy/tree/58b23af1e9ac246a11001d7d5388a1de4f53daa5)：
32 帧历史、总窗口 40 步、每次执行 8 步；DDPM 100 步，预测干净动作。
在线保留动作缓冲区和 KV 缓存，episode 开始须调用 `reset()`。
历史动作只取当前帧之前的数据；图像和 state 历史包含当前帧。
RGB 缩放至 84×84、裁剪 76×76，ImageNet `MEAN_STD`；state/action 使用训练集 `MIN_MAX`。
ResNet18 使用 GroupNorm，无预训练权重；训练器 EMA 默认开启，尾部排除数为 0。
默认训练 100000 步、预热 1000 步，可通过 `--steps` 和 `--scheduler.warmup_steps` 调整。
仅支持 RGB CDP2；不含点云 CDP3。动作块大小属于模型结构，加载 checkpoint 时不可更改；
去噪步数必须等于训练扩散步数。离线评估使用示范历史动作，结果不代表闭环成功率。

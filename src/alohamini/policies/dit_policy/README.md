# dit-policy（DiT-Block）

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

命令在仓库根目录执行。示例保留 episode 1 用于验证；单回合数据省略 `--dataset.eval_episodes` 和 `--eval_steps`。

```bash
python -m alohamini.learning.train \
  --policy.type=dit_policy --policy.device=cuda \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --batch_size=2 \
  --output_dir="$HOME/Alohamini_workspace/runs/dit_policy_01" --background
```

采用 [SudeepDasari/dit-policy](https://github.com/SudeepDasari/dit-policy/tree/ba22ee3ff455f0423fc9cfa5658a1a1cdddc6e73)
的独立 DiT 编码器／解码器：默认 1 帧观测、100 步动作、6 层、隐藏维度 512；
训练扩散 100 步，DDIM 推理 8 步。`--policy.n_action_steps` 可缩短执行窗口。
RGB 缩放至 256×256，训练随机缩放裁剪至 224×224 并模糊，推理缩放至 224×224；
采用 ImageNet `MEAN_STD`、ResNet18 GroupNorm 和空间 token。
`--policy.pooled_vision=true` 改用每帧单 token。
state 保留原值；本地 action 使用训练集 `MIN_MAX`，上游数据需预先归一化。
默认使用 torchvision ImageNet V1 权重初始化，不含作者另行分发的 GN 权重。
默认训练 150000 步、预热 2000 步；EMA 关闭，尾部保留并屏蔽填充动作。不使用语言输入。

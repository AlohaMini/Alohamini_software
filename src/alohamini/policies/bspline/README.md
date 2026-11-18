# B-spline Policy

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

命令在仓库根目录执行。示例保留 episode 1 用于验证；单回合数据省略 `--dataset.eval_episodes` 和 `--eval_steps`。

```bash
python -m alohamini.learning.train \
  --policy.type=bspline --policy.device=cuda \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --batch_size=2 \
  --policy.fit_tolerance=0.01 \
  --output_dir="$HOME/Alohamini_workspace/runs/bspline_01" --background
```

支持已声明坐标及单位的绝对命令，可联合预测两臂与两夹爪；不自动转换关节／末端坐标。
默认 2 帧观测、16 行样条参数、三次样条、DDIM 16 步，开启 EMA。
`policy.horizon` 表示参数行数，轨迹持续时间由节点决定。
`policy.fit_tolerance` 使用录制动作单位，按训练集拟合节点与控制点统计。
回合至少包含 `degree+1` 帧；训练与验证保持回合隔离。

参数缓存默认位于系统临时目录，可用 `--policy.parameter_cache_dir` 指定持久目录。
缓存仅保存数值标签，不改写原始数据。拟合精度、参数行数或数据变化后自动分开缓存。

评估使用 `--inference.type=async`，默认进行相位对齐；可加
`--inference.continuity=none` 关闭。
`--policy.time_scale 1.0` 是默认时间倍率，大于 1 会缩短轨迹时间。
BSP 不使用 ACT 时间融合，不接受密集动作的 RTC 引导；首次使用保持时间倍率为 1。

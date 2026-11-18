# LeWorldModel

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

命令在仓库根目录执行。示例中的 episode 1 用于验证；数据只有一个回合时，省略 `--dataset.eval_episodes` 和 `--eval_steps`。

从零训练，无需下载基座；需要至少一路 RGB 与命名动作字段，不输入 state 或语言：

```bash
alohamini train --policy.type=lewm \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --eval_steps=1000 \
  --batch_size=32 --run_name=lewm_task --background
```

图像为 224×224、ImageNet mean/std；多相机共享编码器并平均特征。
动作保留记录坐标与单位，使用训练集 z-score；支持 `recorded`、`tcp_absolute`，不支持 `tcp_delta`。
默认 3 个历史观测、每 5 行一个动作块；预测损失为特征 MSE 加 `0.09 × SIGReg`，验证不自动生成动作 MAE。
默认 30000 步、AdamW `5e-5`、恒定学习率、BF16、EMA 关闭；支持 DDP/FSDP2。
多卡 SIGReg 汇集单次 forward 的全局 batch，梯度累积不合并多次 forward 的统计。

推理须显式提供全部相机的目标图片，CEM 默认规划 25 行动作：

```bash
alohamini evaluate --host 192.168.77.2 --robot_model alohamini2pro \
  --policy.path "$HOME/Alohamini_workspace/runs/lewm_task/checkpoint" \
  --device cuda --fps 30 --goal-images '{"forward":"/path/to/goal.png"}'
```

多相机目标使用同一 JSON 增加相机项。离线评估同样使用 `--goal-images`；
Python 接口为 `NativePolicy.set_goal({"forward": "/path/to/goal.png"})`，目标不写入权重，`reset()` 保留目标。
官方 `weights.pt` 可通过 `--policy.path` 导入，须匹配结构和动作块维度；PushT 权重不能直接用于不同动作维度。
当前未接入远程目标传递与 RTC。来源与预设见 [SOURCE.json](SOURCE.json)。

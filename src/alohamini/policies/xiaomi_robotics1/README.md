# Xiaomi-Robotics-1

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

命令在仓库根目录执行。示例中的 episode 1 用于验证；数据只有一个回合时，省略 `--dataset.eval_episodes` 和 `--eval_steps`。

使用 `xiaomi_robotics1`，本地基座目录包含官方 `model_states.pt`；`assets_path` 指向
Qwen3-VL-4B-Instruct 的配置、tokenizer 和图像处理文件。

```bash
alohamini train --policy.type=xiaomi_robotics1 \
  --policy.path="$HOME/Alohamini_workspace/pretrained/xiaomi-robotics1" \
  --policy.assets_path="$HOME/Alohamini_workspace/pretrained/qwen3-vl-4b-assets" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --distributed_backend=fsdp2 --num_processes=4 \
  --output_dir="$HOME/Alohamini_workspace/runs/xiaomi_01" --background
```

默认读取 `forward,wrist_left,wrist_right` 三路 RGB 和任务文本；使用 30 步动作块、BF16、同步执行。
关节观测保留，动作经保存的标定转换为工具坐标系末端增量；旋转采用旋转向量，夹爪采用开度增量。
统计量按训练集拟合，动作逐时间步计算 mean/std，状态使用 q01/q99。
缺少运动学描述时，按[关节、TCP 与相对目标](../../../../docs/training.md#关节tcp-与相对目标)补充来源。
升降轴保持当前高度，底盘使用数据中的速度标签。尚不支持官方异步动作前缀执行。

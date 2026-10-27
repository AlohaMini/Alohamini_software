# Spirit v1.5

策略类型为 `spirit`，通过原生训练、统计、checkpoint、离线评估和远端推理入口使用。
参考代码固定到 [6d67377](https://github.com/Spirit-AI-Team/spirit-v1.5/tree/6d67377f16f67d6575447a1f29fdd83a2e0099d9)，
发布模型固定到 [9886fa7](https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5/tree/9886fa7552de70b097c4214b7e40e66ff937a6fd)。
官方微调说明见 [Finetune Guide](https://github.com/Spirit-AI-Team/spirit-v1.5#finetune-guide)。

## 资源与训练

在已有 GPU 环境安装 `pip install -e '.[spirit]'`，保留与设备匹配的 PyTorch。
训练与 checkpoint 加载只读取本地资源。下载操作需显式执行：

```bash
hf download Spirit-AI-robotics/Spirit-v1.5 \
  --revision 9886fa7552de70b097c4214b7e40e66ff937a6fd \
  --local-dir /path/to/spirit-base
python -m alohamini.policies.spirit.assets \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --output /path/to/qwen-assets
```

Qwen 资源固定到 `ebb281e`，下载结果保存该版本及文件摘要；可显式覆盖为其他完整提交 SHA。
Qwen 只需配置、tokenizer 和图像处理资源；其权重已包含在 Spirit checkpoint 中。

准备[运动学转换描述](kinematic-fields.md)后训练：

```bash
alohamini train \
  --policy.type=spirit --policy.path=/path/to/spirit-base \
  --policy.backbone_path=/path/to/qwen-assets \
  --policy.action_representation=eef_delta \
  --policy.kinematics_path=/path/to/kinematic_fields.json \
  --dataset.root=/path/to/dataset --dataset.eval_episodes='[1]' \
  --distributed_backend=fsdp2 --num_processes=8 \
  --mixed_precision=bfloat16 --batch_size=32 \
  --output_dir=/path/to/runs/spirit --background
```

启动器返回独立 `.log`、`.pid` 文件和 `tail -f` 命令。也可由 `torchrun` 启动共享训练入口。
显式设置的参数、数据预设及解析后的几何快照随 checkpoint 保存。
`preprocessing.json` 同时列出原始执行字段、派生统计字段的名称/单位和转换顺序。

## 字段与动作

- `eef_delta`：完整位置 state 与命名 action 经过 FK，再编码为每臂
  `[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]`；所有步相对同一当前观测。
  底盘速度和升降高度保留原单位的绝对值，不固定这些自由度。
- `recorded`：显式使用记录的 state/action 字段，不推断关节、TCP、旋转或单位。
  可通过 `--policy.action_representation=recorded` 进行该表示下的算法实验。

`eef_delta` 的源 state/action 名称与顺序必须匹配转换描述；原始字段不被覆盖。
推理结果经相同相对编码的逆变换与 IK 恢复成完整 Host 关节和底盘/升降目标。
基础模型的归一化统计不用于新机器人；在转换后的训练窗口上计算独立 state/action p1/p99。

默认相机映射为 `forward → cam_high`、`wrist_left → cam_left_wrist`、
`wrist_right → cam_right_wrist`。其他名称使用 `--policy.camera_mapping` 显式配置。
最多使用三路相机，通过 `--cameras` 选择；映射须覆盖所有选中相机。
未使用的槽位填黑图。输入为 RGB 240×320，再由 Qwen processor 处理；训练使用官方
ColorJitter，验证和推理关闭增强。任务文本沿用每行 `task`；机器人提示类型默认 `aloha`，
可用 `--policy.robot_type` 显式覆盖。

## 参考协议与平台适配

发布网络为 16 层、1536 宽、32 头的 DiT，交替自注意力与跨注意力；默认 60 步动作和
10 步 Euler 采样。平台预设保留 Flow Matching、Qwen 提示结构、分位数归一化和全参 AdamW。

官方 Franka 示例使用前一帧末端状态及未来实测末端姿态。
平台使用现有图像/state/action 配对，保留已记录的动作请求语义，并扩展到底盘与升降字段；
不会将请求隐式替换为未来实测反馈。平台统计遍历完整训练窗口，使用共享训练调度器；
尾部重复目标和补齐动作维度不计入损失。上述差异记录在 `paper_preset` 中。

原生 checkpoint 打包 Qwen 配置、tokenizer、图像处理资源与完整模型权重，可脱离原资源目录加载。
支持单设备、DDP 和 FSDP2 训练及共享远端动作块推理；尚未提供 Spirit 的 RTC 前缀引导。
离线误差与训练损失不代表真机任务成功率。

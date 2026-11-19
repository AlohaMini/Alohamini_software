# Spirit v1.5

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

命令在仓库根目录执行。示例中的 episode 1 用于验证；数据只有一个回合时，省略 `--dataset.eval_episodes` 和 `--eval_steps`。

使用本地 Spirit 基座及 Qwen 配置、tokenizer 和图像处理资源：

```bash
hf download Spirit-AI-robotics/Spirit-v1.5 \
  --revision 9886fa7552de70b097c4214b7e40e66ff937a6fd \
  --local-dir "$HOME/Alohamini_workspace/pretrained/spirit-v1.5"
python -m alohamini.policies.spirit.assets \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --output "$HOME/Alohamini_workspace/pretrained/spirit-qwen-assets"
```

Qwen 权重已包含在 Spirit 基座中，此资源命令只下载配置和处理器文件，记录版本及文件摘要；
输出目录须为空。训练和 checkpoint 加载只读取本地资源。

按[关节、TCP 与相对目标](../../../../docs/training.md#关节tcp-与相对目标)选择数据表示。以下命令适用于标准中位标定数据：

```bash
alohamini train \
  --policy.type=spirit \
  --policy.path="$HOME/Alohamini_workspace/pretrained/spirit-v1.5" \
  --policy.backbone_path="$HOME/Alohamini_workspace/pretrained/spirit-qwen-assets" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --state=joint_position,base_velocity,lift_height \
  --data.state_representation=tcp \
  --data.action_representation=tcp_delta \
  --dataset.eval_episodes='[1]' \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --distributed_backend=fsdp2 --num_processes=4 \
  --mixed_precision=bfloat16 --batch_size=1 --gradient_accumulation_steps=8 \
  --steps=40000 --save_freq=5000 --log_freq=100 \
  --output_dir="$HOME/Alohamini_workspace/runs/spirit_01" --background
```

`num_processes`、batch 和累积次数按可用显卡与显存调整。
默认采用 16 层、1536 宽、32 头 DiT，交替自注意力与跨注意力；生成 60 步动作，推理执行 10 步 Euler 采样。
训练采用 Flow Matching、全参 AdamW `2.5e-5`、betas `(0.9,0.95)`、weight decay `1e-10`、
梯度裁剪 `1`、1000 步预热，调度器默认到 50000 步衰减至 `2.5e-6`。
训练预设为 40000 步、每卡 batch=32、BF16；上例显式使用较小 batch 和梯度累积。
梯度检查点默认开启，可用 `--policy.gradient_checkpointing=false` 关闭。

| 数据表示 | 输入与执行 |
| --- | --- |
| state=`tcp`、action=`tcp_delta`（默认） | 共享数据层转换关节 state/action；预测相对当前 TCP 的位姿目标，执行前用 IK 恢复 Host 目标 |
| state=`recorded`、action=`recorded` | 保留所选 state 和原 action；无需运动学配置 |

保留记录表示时，将上例两个表示参数均改为 `recorded`。
Spirit 需要数值 state 和非空任务文本，不支持 `--state=none`。
原 action 的名称与顺序须匹配转换描述；所选 state 单独记录，FK 所需的完整关节观测由读取器提供。
底盘和升降动作保持完整，不自动固定这些自由度。
转换后的 state/action 宽度最多各 32 维，原始关节数与派生 TCP 维度分别记录。
填充通道和 episode 尾部补齐目标不计入动作损失。
归一化分别在转换后的训练 state/action 上计算 p1/p99 并映射到 `[-1,1]`，不复用基座机器人的统计。

最多选择三路 RGB，相机映射须覆盖全部选中视角：

| 记录相机 | Spirit 槽位 |
| --- | --- |
| `forward` | `cam_high` |
| `wrist_left` | `cam_left_wrist` |
| `wrist_right` | `cam_right_wrist` |

其他名称通过 `--policy.camera_mapping='{"cam_high":"top","cam_left_wrist":"left","cam_right_wrist":"right"}'`
显式映射，并用 `--cameras` 选择对应视角。未使用的槽位填黑图。
双前视 EDU 可选一颗前视和右腕，例如 `--cameras='["forward_left","wrist_right"]'`
配合 `--policy.camera_mapping='{"cam_high":"forward_left","cam_right_wrist":"wrist_right"}'`。
同时使用双前视需要另行适配视角槽，不能把第二颗前视标作左腕。
图像先读取为 RGB 240×320，再由 Qwen processor 处理；训练使用 ColorJitter，验证和推理关闭增强。
每行 `task` 用作任务文本，机器人提示类型默认为 `aloha`，用 `--policy.robot_type` 覆盖。

预设参考实现固定到 [6d67377](https://github.com/Spirit-AI-Team/spirit-v1.5/tree/6d67377f16f67d6575447a1f29fdd83a2e0099d9)，
基座固定到 [9886fa7](https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5/tree/9886fa7552de70b097c4214b7e40e66ff937a6fd)。
参考 Franka 数据使用前一帧末端状态及未来实测位姿；平台保留记录的观测／动作配对和数据 FPS，
不重采样时间戳，使用共享训练调度器和完整训练窗口统计。这些协议差异、显式覆盖值和几何快照保存在 checkpoint 中。
官方说明见 [Finetune Guide](https://github.com/Spirit-AI-Team/spirit-v1.5#finetune-guide)。

原生 checkpoint 包含完整模型及 Qwen 配置、tokenizer、图像处理资源，可脱离原资源目录加载。
`preprocessing.json` 列出原始执行字段、派生统计字段的名称／单位和转换顺序。
支持单设备、DDP、FSDP2、离线评估和云端动作块推理；Spirit 不支持 RTC 前缀引导。

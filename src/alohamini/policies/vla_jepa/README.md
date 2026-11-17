# VLA-JEPA

[策略索引](../../../../docs/training.md#策略选择)

准备策略权重及本地处理器：

```bash
BASE="$HOME/Alohamini_workspace/pretrained/vla_jepa"
hf download lerobot/VLA-JEPA-Pretrain --revision=e946c3e5b538d760f4b4ff239d1b1c12090c041d --local-dir="$BASE"
hf download Qwen/Qwen3-VL-2B-Instruct --revision=89644892e4d85e24eaac8bacfd4f463576704203 \
  --include '*.json' '*.txt' '*.jinja' --local-dir="$BASE/qwen"
hf download facebook/vjepa2-vitl-fpc64-256 --revision=b3c1679b7c34d3255ef3547f27c7b226aefab26f \
  --include '*.json' --local-dir="$BASE/vjepa2"

alohamini train --policy.type=vla_jepa --policy.path="$BASE" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --policy.world_model_cameras='["forward","wrist_left"]' \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/vla_jepa_01" --background
```

所有选中相机的当前帧送入 Qwen；`world_model_cameras` 明确选择世界模型监督使用的相机。
发布基座的世界模型使用两路。改变路数需加 `--policy.reinitialize_world_heads=true`，
重置世界模型的输入、输出投影；不会自动复制或丢弃相机。
默认动作块 7 步、视频窗口 8 帧、积分 4 步；在线动作推理只需要当前画面。

state 使用训练集 MEAN_STD，action 使用 MIN_MAX。字段按记录的名称、单位和表示训练；
推理输出默认限制在归一化的 `[-1,1]`，可用 `--policy.clip_normalized_actions=false` 关闭。
默认重置动作输入、输出及 state 头，不自动执行末端坐标、相对动作或夹爪二值转换。
保留同一机器人头时可设 `--policy.reinitialize_robot_heads=false`。
字段变换使用公共 `data` 配置，随 checkpoint 保存。

`--policy.world_model_context=causal`（默认）按前缀编码教师上下文，避免读到后续帧；
`shared` 使用原文的完整视频共享编码。冻结的 V-JEPA2 教师保持 eval，回合尾部填充不计入世界模型损失。
关闭世界模型并仅训练动作头：

```bash
--policy.enable_world_model=false --policy.freeze_qwen=true
```

默认 BF16 Qwen，FP32 动作 DiT、世界模型预测器与冻结教师；公共混合精度为 BF16，动作分支保持 FP32 计算。
默认对 Qwen、动作 DiT 和世界模型预测器开启重计算。实际 Qwen 层数来自保存的 `qwen_config`；
动作 DiT 默认 16 层，预测器 12 层；可分别用 `action_num_layers`、`predictor_depth` 调整。
学习率以公共 `--optimizer.lr` 为基准，Qwen、动作头、预测器的默认比例为 `1/3`、`10/3`、`50/3`，
可通过 `--policy.lr_scales` 覆盖。

从独立基座训练时，完整下载 Qwen 与 V-JEPA2 权重到相同子目录，再设置
`--policy.base_weights_mode=backbones`。当前不包含语言交叉熵、LoRA、RTC 或专用 FlashAttention 内核。

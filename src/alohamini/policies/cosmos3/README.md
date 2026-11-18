# Cosmos3 Nano／Edge

[策略索引](../../../../docs/training.md#策略选择)

Nano 使用 Qwen3-VL MoT，默认 36 层；Edge 使用 Nemotron MoT，默认 28 层。
网络与 VAE 配置读取本地基座，不自动截断层数。

```bash
BASE="$HOME/Alohamini_workspace/pretrained/cosmos3_nano"
hf download lerobot/Cosmos3-Nano --revision=104d12ccc544e14984443b812233969f389ee8dc --local-dir="$BASE"
hf download nvidia/Cosmos3-Nano --include 'tokenizer*' '*vocab*' '*merges*' '*special_tokens*' '*.jinja' \
  --local-dir="$BASE/tokenizer"

alohamini train --policy.type=cosmos3_nano --policy.path="$BASE" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/cosmos3_01" --background
```

Edge 将策略改为 `cosmos3_edge`，基座改为 `lerobot/Cosmos3-Edge`
（revision `35d07f7f5ebedc020125dd06e87eee47109276ed`），tokenizer 改为 `nvidia/Cosmos3-Edge`。
运行只读取本地权重和 tokenizer，后者随 checkpoint 保存。

默认 FP32 可训练参数与 AdamW 状态、BF16 混合精度和冻结模块；时间嵌入、RoPE 保持 FP32。
仅训练生成分支与视频／动作投影，冻结文本理解分支和视觉塔，开启重计算。
动作参数组学习率为公共学习率的 5 倍，使用 `optimizer_action_lr_multiplier` 调整。
默认线性预热／衰减及 Power-law EMA；使用 `--ema.enable=false` 关闭 EMA。
分片训练使用[公共 FSDP2 配置](../../../../docs/training.md#多卡与梯度累积)。

支持 1～3 路相机，按选择顺序使用单视图、左右拼接或上方一路＋下方两路布局。
可用 `camera_keys` 显式排序、`camera_roles` 声明视角，或提供 `view_description`；不会猜测腕部相机。
每路默认缩放为 256×256，再按原生画布规则拼接、补边并送入 Wan VAE。
默认每块预测并执行 16 步，训练使用当前＋16 帧完整窗口，回合尾裁除 16 个起点。
推理只使用当前帧；默认去噪 30 步，可用 `--policy.num_inference_steps` 调整。

action 按记录的字段名和单位使用 q01/q99，不默认裁剪边界外动作；最多 64 个坐标。
默认重置机器人动作头。保留同一机器人头需设置 `reinitialize_robot_heads=false`、
完整的 `pretrained_action_names`，并明确选择对应 `domain_id`。
默认不使用 state；开启 `use_state=true` 时，state 必须与动作具有相同坐标、顺序和单位，
显式声明 `state_action_names`，并把 STATE 归一化模式设为 ACTION 的模式；两者共用动作统计量。
不自动执行 LIBERO 相对末端／rot6d 转换或夹爪转换；字段表示使用公共 `data` 配置。

空闲帧计数当前固定为 0。暂不包含原文 `quantile_rot` 协议、推理文本 KV 复用、LoRA 或 RTC。
`compile_model` 默认关闭；多卡编译与完整基座训练尚未验证。

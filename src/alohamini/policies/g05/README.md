# G0.5

[策略索引](../../../../docs/training.md#策略选择)

准备本地转换版基座，需包含 `config.json`、`model.safetensors` 和 `hf_processor/`：

```bash
BASE="$HOME/Alohamini_workspace/pretrained/g05"
hf download lerobot/g05_libero --revision=9b7ee7d7286b2744e7cfae855dd71aa3cf4c075a \
  --local-dir="$BASE"

alohamini train --policy.type=g05 --policy.path="$BASE" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --output_dir="$HOME/Alohamini_workspace/runs/g05_01" --background
```

默认使用 flow 微调：FP32 参数、BF16 混合精度，batch 8、100k 步、学习率 `1e-5`、
预热 1000 步，每个样本抽取 8 个 flow 时刻。默认允许动作损失更新 VLM。
`--policy.gradient_checkpointing=true` 开启重计算；多卡使用[公共配置](../../../../docs/training.md#多卡与梯度累积)。
结构、观测历史和动作块长度读取本地基座，保留完整 Transformer 层数。

所有选定相机均参与模型；默认各路 256×256、RGB `[-1,1]`，顺序随 checkpoint 保存。
多帧基座需要对应的时序视觉结构，同步与远程推理使用公共观测历史缓存。
state/action 使用记录或显式变换后的字段和单位，映射到基座槽位；默认重置机器人输入、输出头。
保留原头须设置 `--policy.reinitialize_robot_heads=false` 并提供完整的 `state_slots`、`action_slots` 名称映射。

默认使用训练集 q01/q99，动作逐未来步统计，保留边界外动作；推理先还原整块动作再逐步执行。
`--policy.data_protocol=g05` 开启原实现的归一化后 `[-5,5]` 裁剪。
平台机器人适配不自动转换相对位姿、夹爪方向或关节坐标系；不是原 benchmark 数据协议的直接复现。

ActionCodec 训练另需 `action_tokenizer.safetensors`、兼容的原始坐标槽位及归一化：
`train_discrete_actions=true text_loss_weight=1 reinitialize_robot_heads=false`。
`action_head=flow|actioncodec` 选择推理头，默认 flow；flow 推理不运行动作 token 生成。
CoT 文本训练需 `predict_cot=true text_loss_weight=1 use_language_recipe=true` 和真实语言标注，
recipe 中用 `Subtask: ${subtask}` 声明目标；`runtime_system=system2` 开启推理时 CoT。
LoRA、RTC、编译和新增策略的分布式执行尚未验证。

线性注意力缺少 `flash-linear-attention` / `causal-conv1d` 时会使用较慢的 PyTorch 路径；
当前验证覆盖该路径，加速内核尚未验证。
代码和权重受 [G0.5 Community License](LICENSE-G0.5) 的非商业及其他使用条件约束。
本项目不代表 Galaxea 背书。

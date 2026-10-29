# HyVLA-0.5

策略名：`hyvla05`。[论文](https://arxiv.org/abs/2606.14409v2)；代码和权重版本见 [SOURCE.json](../src/alohamini/policies/hyvla05/SOURCE.json)。

```bash
conda activate alohamini
python -m pip install -e '.[hyvla05]'
hf download tencent/Hy-Embodied-0.5-VLA-RoboTwin \
  --revision bd7bba6f5934ad62293a2a34f74760c6a3ef2ff8 \
  --local-dir ~/Alohamini_workspace/pretrained/hyvla05

alohamini train --policy.type hyvla05 \
  --dataset.root ~/Alohamini_workspace/datasets/task \
  --policy.path ~/Alohamini_workspace/pretrained/hyvla05 \
  --run_name hyvla05_task --batch_size 1 --steps 60000
```

使用本地 tokenizer、嵌入式 VLM 配置和 `model.safetensors`，运行时不自动下载。平台 checkpoint 包含 tokenizer、统计量与网络权重，迁移时无需原基座目录。

| 配置 | 默认值 |
|---|---|
| `chunk_size` / `n_action_steps` | 20 / 20 |
| `history_frames` / `history_stride` | 6 / 15 个数据行 |
| `network.num_steps` | 10 |
| `network.resize_imgs_with_padding` | 224×224，保持宽高比并补边 |
| state / action 归一化 | 训练集均值、标准差 |
| 精度 | BF16 |

相机顺序取 `--cameras` 或数据集顺序；需要非空任务文本及 state。`--policy.network` 接受 JSON，例如 `'{"freeze_vision_encoder":true}'`。冻结视觉仍使用图像特征。

默认微调 **连续 20 步原始记录动作**，使用数据集字段名、单位和训练集统计量。官方 RoboTwin 的 40 个 token 是同一段未来的 20 个相对姿态加 20 个绝对姿态，平台不把它们解释为连续 40 步。不读取官方 `norm_stats.pkl`，不自动推断末端坐标或执行 FK/IK。

训练与离线推理从同一 episode 读取历史，开头重复首帧。在线历史按来源客户端的时间戳采样，只取已收到的观测；远程请求较稀疏时保持最近的过去观测。`reset()` 清空历史和动作队列。

支持平台 DDP、FSDP2、训练器 EMA 和普通远程动作块推理；RTC 引导、FlowPRO 与论文的 Bézier 拼接不包含在此适配中。论文预设、机器人适配与覆盖参数随 checkpoint 保存。

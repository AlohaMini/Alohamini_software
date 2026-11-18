# FineART-VLA

[策略索引](../../../../docs/training.md#策略选择)

准备发布基座和本地 PaliGemma tokenizer：

```bash
hf download lerobot/FineART-VLA-subtask-midtrained \
  --revision=b6cf4faf2630f971dd390bc32365e9d9c330da8e \
  --local-dir="$HOME/Alohamini_workspace/pretrained/fineart"
hf download google/paligemma-3b-pt-224 \
  --revision=35e4f46485b4d07967e7e9935bc3786aad50687c \
  --include 'tokenizer*' 'special_tokens_map.json' \
  --local-dir="$HOME/Alohamini_workspace/pretrained/paligemma-tokenizer"

alohamini train --policy.type=fineart_vla \
  --policy.path="$HOME/Alohamini_workspace/pretrained/fineart" \
  --policy.assets_path="$HOME/Alohamini_workspace/pretrained/paligemma-tokenizer" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_with_subtasks" \
  --cameras='["forward","wrist_left","wrist_right"]' \
  --batch_size=1 --gradient_accumulation_steps=8 \
  --output_dir="$HOME/Alohamini_workspace/runs/fineart_01" --background
```

数据须含 `language_persistent` 的时间戳子任务标注，任务文本来自原数据。
默认 30% 样本监督“任务→子任务”文本，70% 监督“子任务→动作”；缺少有效标注时报错。
不生成或猜测训练标签。

state/action 使用具名记录坐标和训练集 q01/q99，不裁剪；动作补至 32 维，填充维度不计损失，回合尾重复目标保留。
相机按保存的字段顺序输入，等比缩放并补边至 224。默认语言 Transformer、动作专家各 18 层，SigLIP 视觉塔 27 层。

加载基座时继承其模型配置，显式参数覆盖。发布基座为 `flow_num_repeats=1`、`fast_skip_tokens=128`；
新配置默认分别为 5、1152。

默认由主 rank 使用训练窗口拟合 FAST，在独立采样的训练窗口上检查重构误差和长度；不使用验证回合，不截断动作 token。
缓存位于 `Alohamini_workspace/pretrained/action_tokenizers/fast/`，codec、配置和拟合报告随 checkpoint 保存，续训直接复用。
`auto_fit_fast_tokenizer=false` 使用本地 `fast_tokenizer_path`；`fast_action_loss_weight=0` 关闭 FAST 损失。

默认开启 KI 和重计算，flow/text/FAST 权重为 10/1/1。AMP、TF32 与多卡参数使用公共训练配置。
联合注意力默认使用 FP32 math SDPA，其余模块保留配置的模型精度。
验证分别统计动作样本均值与文本 token 均值，不将混合任务验证换算为动作 MAE。

启用空间 VQA 时追加：

```bash
--policy.vqa_cameras='["forward"]' --policy.fast_skip_tokens=1152
```

`language_events` 须包含绑定该相机的成对 VQA 问答。点使用 `point_format=xy`，框使用 `bbox_format=xyxy`，坐标为原图 `[0,1]` 比例。
位置监督采用 PaliGemma `<loc>` token；FAST 须避开位置 token 区间。

本地 `evaluate` 或云端 `inference serve` 追加 `--language-replan-interval-s=2 --task='任务指令'`，启用周期性子任务生成。
默认关闭；按观测时间检查更新，切换子任务时丢弃旧动作队列。回合结束沿用评估时长或人工停止。
`NativePolicy.query_language(sample, kind="vqa", camera="forward", question="Where is the cup?")` 返回文本和点/框；
`sample` 可为平台样本或 Host 快照，查询不会推进控制器。工具调用暂不支持。

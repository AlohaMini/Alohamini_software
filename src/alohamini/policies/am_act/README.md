# AM-ACT

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

命令在仓库根目录执行。示例保留 episode 1 用于验证；单回合数据省略 `--dataset.eval_episodes` 和 `--eval_steps`。

Apple Silicon 将 `--policy.device=cuda` 改为 `--policy.device=mps`，公共参数见 [MPS 配置](../../../../docs/training.md#apple-siliconmps)。

共有相机身份编码和旧模型迁移设置见 [ACT：相机身份编码](../act/README.md#相机身份编码)。

```bash
python -m alohamini.learning.train \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_vision" \
  --dataset.eval_episodes='[1]' \
  --policy.type=am_act --policy.device=cuda --policy.chunk_size=100 \
  --output_dir="$HOME/Alohamini_workspace/runs/am_act_vision_01" \
  --steps=100000 --batch_size=2 --save_freq=10000 --log_freq=100 \
  --eval_steps=1000 --background
```

也支持含 state 的数据。速度/电流实验改用保留反馈的数据目录，并加 `--state=joint_velocity,joint_current`。
新 AM-ACT 训练默认把 `action.names` 中的 `x.vel`、`y.vel`、`theta.vel` 分别设为三分类：

| 字段 | 类别顺序：负向、停止、正向 | 单位 |
| --- | --- | --- |
| `x.vel` | `[-0.15, 0, 0.15]` | m/s |
| `y.vel` | `[-0.15, 0, 0.15]` | m/s |
| `theta.vel` | `[-45, 0, 45]` | deg/s |

可通过 `--policy.discrete_action_weighting=none|inverse_frequency|sqrt_inverse_frequency`
选择权重策略，`--policy.discrete_action_max_weight_ratio` 调整上限。

常用选项：

| 参数 | 用途 |
| --- | --- |
| `--policy.base_classification=false` | 新训练关闭默认底盘分类，改为连续回归 |
| `--policy.discrete_action_mask_absent_classes=true` | 屏蔽训练集中完全没有出现的类别；默认关闭 |
| `--policy.discrete_action_values='[[-0.15,0,0.15]]'` | 自定义分类值，须配合 `discrete_action_dims` 设置 |

自定义维度须按数据的 `action.names` 核对。屏蔽缺类后仍保留三输出头，
两类有数据时等效二分类；把缺类权重设为零不会禁止模型输出该类别。
补录新方向后，用 `--policy.path=<旧 checkpoint>` 微调，重新统计并启用新增类别。
数据变化不能使用 `--resume=true`。启用 `action_loss_groups` 时须覆盖全部需要回归的连续维度。

## 条件 CVAE

默认 AM-ACT 使用动作及可选 state 后验、标准高斯先验，推理时 `z=0`。
启用图像条件后验和先验可使用通用示例：

```bash
python -m alohamini.learning.train \
  --config=examples/learning/am_act_cvae.json \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_vision" \
  --dataset.eval_episodes='[1]' --eval_steps=1000 --background
```

该示例是机器人适配配置：ResNet-18、30 步 Chunk、执行 10 步、纯视觉、
4 Query Attention Pooling、Image+Action 后验、图像条件高斯先验、KL 权重 1、
5000 个优化步 warmup。动作主干 decoder 为 7 层，主干 encoder 和后验 encoder 各 4 层。
图像使用 RGB/ImageNet 归一化，动作使用训练统计和原始目标单位，不使用语言。

| 配置项 | 可选值 |
| --- | --- |
| `posterior_condition` | `image_action` / `action` |
| `prior_type` | `conditional_gaussian` / `standard_normal` |
| `latent_visual_pooling` | `multi_query_attention` / `gap` |
| `latent_inference_mode` | `sample` / `mean` / `zero` |
| `latent_refresh_mode` | `phase` / `chunk` |

示例按阶段缓存 z：首次根据图像采样，后续预测使用新图像并复用 z。
任务执行器调用 `policy.refresh_latent()` 切换阶段；新 episode 的 `reset()` 也会清空缓存。
模型不会自动识别阶段。刷新 z 同时清空动作队列和时序融合，避免混合不同潜变量的预测。

# ACT

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

命令在仓库根目录执行。示例保留 episode 1 用于验证；单回合数据省略 `--dataset.eval_episodes` 和 `--eval_steps`。

Apple Silicon 将 `--policy.device=cuda` 改为 `--policy.device=mps`，公共参数见 [MPS 配置](../../../../docs/training.md#apple-siliconmps)。

```bash
python -m alohamini.learning.train \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' \
  --policy.type=act --policy.device=cuda --policy.chunk_size=100 \
  --output_dir="$HOME/Alohamini_workspace/runs/act_01" \
  --steps=100000 --batch_size=2 --save_freq=10000 --log_freq=100 \
  --eval_steps=1000 --background
```

`--state=none` 使用纯视觉；`--image_size='[480,640]'` 设置高、宽。
ResNet18 默认加载 ImageNet 权重，首次下载缓存到工作区 `pretrained/`。
离线使用须提前准备；`--policy.pretrained_backbone_weights=null` 改为随机初始化。

## 相机身份编码

ACT 和 AM-ACT 新训练默认启用相机身份编码，将前视、左腕、右腕等图像与其来源对应。
训练和推理使用同一组编码，相机名称及顺序随 checkpoint 保存；采集和部署时须保持相机命名一致。
`--policy.use_camera_embeddings=false` 可关闭，保留原 ACT 的视觉融合方式。
旧 checkpoint 的推理、续训及默认微调保留原配置；为旧模型补上编码时，在
`--policy.path=<旧 checkpoint>` 微调命令中增加 `--policy.use_camera_embeddings=true`，
使用新的输出目录，不使用 `--resume=true`。新增编码从零开始学习，需要微调后才具有区分能力。

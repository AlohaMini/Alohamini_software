# AC-DiT

原生策略名为 `acdit`，支持两阶段训练、验证损失、全身策略离线评估、checkpoint 微调和精确续训。
训练器共用优化器、累积、多卡、EMA 和参数存储精度配置；当前未开放本策略的远程动作块服务。

来源：[NeurIPS 2025 正式论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5c6c63b7ec141825ff1327563e699f76-Abstract-Conference.html)，
[官方代码](https://github.com/PKU-HMI-Lab/AC-DiT/tree/90ad00a926f34da04816ed9c3312aaf3bc845b7f)。
代码固定在 `90ad00a926f34da04816ed9c3312aaf3bc845b7f`，完整来源见策略目录的 `SOURCE.json`。

## 安装与权重

```bash
conda activate alohamini
python -m pip install -e '.[acdit]'
hf download google/siglip-so400m-patch14-384 \
  --local-dir "$HOME/Alohamini_workspace/pretrained/siglip-so400m-patch14-384"
```

模型只读取本地资源，不在训练和部署期间下载权重。
SigLIP 图像／语言编码器冻结；点云模式另有 Lift3D 几何 tokenizer 和 q/k/v LoRA。
不需要安装 ManiSkill、PyTorch3D 或 PEFT。checkpoint 包含编码器配置、tokenizer 和完整权重，
加载原生 checkpoint 不依赖原 SigLIP 下载目录或第一阶段目录。

## 输入与动作

| 输入 | 要求 |
| --- | --- |
| RGB | 至少一路记录相机；使用记录顺序，可通过 `--cameras` 选择 |
| 语言 | 每个样本有非空 `task`；冻结 SigLIP 文本编码器，右侧补齐，默认最多 64 tokens |
| state | 明确命名的数值状态；默认选择记录的关节位置、底盘速度和升降高度 |
| action | 保留记录字段顺序和单位；默认底盘字段为 `x.vel,y.vel,theta.vel` |
| 点云 | 完整模式需要记录的 `observation.point_cloud`，形状 `N×3` 或 `N×6`；前三列是 XYZ，米制，并声明坐标系 |
| 可选上下文 | `--policy.context_fields` 显式列出记录中的数值观测；未配置时不生成目标、物体或抓取状态 |

没有点云时，在两个阶段均使用 `--policy.use_pointcloud=false`。该模式保留底盘到全身条件机制，
并按语言与图像特征的相关性调整各相机权重；关闭点云编码与融合，属于 RGB 适配版本。
它不能直接作为完整论文配置或沿用论文效果结论。

共享数据层采用 `--image_resize_mode=pad --image_pad_value=0.5` 等比例居中补边，
默认输出 384×384；进入 SigLIP 前将 RGB `[0,1]` 映射为 `[-1,1]`。
默认图像／点云窗口为当前及前一帧，数值 state 使用当前帧，动作窗口为当前行及后一行。
保留数据集 FPS，episode 边界重复补齐，不按时间戳重采样。

默认对 state/action 使用训练集 `MEAN_STD`，验证、推理和第二阶段复用保存的统计。
这与开源 MSHab 数据直接使用原始数值的做法不同。若数据已采用论文坐标和单位，可在 JSON 的
`model.normalization_mapping` 中将 `STATE`、`ACTION` 设为 `IDENTITY`。
MSHab 的关节增量动作不能直接当成 AlohaMini 绝对关节位置；坐标转换使用平台通用数据层，
不由 AC-DiT 根据维度推断。

`action_slot_map`／`state_slot_map` 把明确命名的字段放入默认 128 维向量。
原生训练默认按声明顺序建立并保存映射；这不是 Fetch/RDT 的语义槽位映射。
外部 RDT 权重初始化须显式提供对应的命名映射。外部数据的底盘控制名不同时，设置
`--policy.base_action_names='["base_linear_velocity","base_angular_velocity"]'`。
移动协调训练需要含底盘运动的示范。

## 两阶段 RGB 训练

第一阶段只监督明确命名的底盘动作：

```bash
python -m alohamini.learning.train \
  --policy.type=acdit --policy.stage=mobility --policy.device=cuda \
  --policy.use_pointcloud=false \
  --policy.assets_path="$HOME/Alohamini_workspace/pretrained/siglip-so400m-patch14-384" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' \
  --output_dir="$HOME/Alohamini_workspace/runs/acdit_mobility" \
  --background
```

第一阶段 checkpoint 用于初始化第二阶段，不是可部署的全身控制策略。
验证仍计算底盘扩散损失，训练结束时不生成全身动作 MAE 报告。

第二阶段通过通用 `--policy.path` 加载第一阶段的完整编码器、LoRA、底盘 head 和投影器，
冻结整个底盘分支，训练全身 head 与模态适配模块：

```bash
python -m alohamini.learning.train \
  --policy.path="$HOME/Alohamini_workspace/runs/acdit_mobility/checkpoint" \
  --policy.stage=full --policy.device=cuda \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' \
  --output_dir="$HOME/Alohamini_workspace/runs/acdit_full" \
  --background
```

第二阶段继承相机、输入模式、槽位映射和图像处理，自动选择 `normalization=checkpoint`。
改变冻结底盘分支的输入、坐标、采样步数或归一化时会拒绝初始化。
切换阶段创建新训练目录；精确续训使用保存的 `train_config.json` 和 `--resume=true`，不能切换阶段。

## 论文预设与资源配置

| 项目 | 默认 `recipe=paper` | `recipe=released` |
| --- | --- | --- |
| 底盘阶段训练步数 | 20000，正式论文 B.3 | 30000，开源启动脚本 |
| 全身阶段训练步数 | 30000，正式论文 B.3 | 20000，开源启动脚本 |
| 每进程 batch | 16 | 底盘 24，全身 16 |
| 底盘／全身学习率 | `1e-4 / 1e-5`，取自开源脚本 | 相同 |
| 优化器 | AdamW，betas `(0.9,0.999)`，weight decay `0.01`，裁剪 `1` | 相同 |
| 调度／EMA | 恒定学习率；EMA 关闭 | 相同 |
| 底盘 head | width 1024、14 层、32 heads | 相同 |
| 全身 head | width 2048、28 层、32 heads | 相同 |
| 扩散 | DDPM 训练 1000 步，cosine betas，预测干净动作；DPM-Solver++ 采样 5 步 | 相同 |
| 数值精度 | FP32 可训练参数与优化器状态；BF16 autocast | 原生适配仍采用该组合 |

论文 B.3 的 batch 16 未明确区分全局与每卡；平台 `batch_size` 按每进程定义，
有效 batch 为 `batch_size × 进程数 × 累积次数`。

开源脚本加载 RDT-170M／RDT-1B 初始化 head；原生默认随机初始化 head。
可使用 `--policy.mobility_pretrained_path`／`--policy.body_pretrained_path` 指定本地、尺寸匹配的
RDT `model.safetensors`，同时提供显式槽位映射。只导入兼容张量，所有 transformer block 必须匹配；
新条件投影器保留初始化值。平台不直接导入上游训练框架的完整 AC-DiT checkpoint。

正式论文在八张 A100 80GB 上训练，并报告 batch 16 时约 60GB 训练显存。
小显存机器应显式缩小 head 作流程验证；减小 batch 不会消除权重和优化器状态占用。
例如在两个阶段的第一份配置中设置：

```text
--batch_size=1 --gradient_accumulation_steps=16
--policy.mobility_hidden_dim=256 --policy.mobility_num_layers=4 --policy.mobility_num_heads=8
--policy.hidden_dim=512 --policy.num_layers=8 --policy.num_heads=8
--policy.gradient_checkpointing=true
```

这些是缩小结构的机器人适配配置，不是论文规模。`paper_preset` 和 checkpoint 保存来源、
论文／脚本差异、所用输入、训练预算及显式覆盖项。

## 加入点云

在第一阶段启用 `use_pointcloud=true`，并使数据集包含所选点云字段。
字段元数据需声明 `coordinate_frame` 和 `units: "m"`；也可用
`--policy.pointcloud_frame=forward_camera --policy.pointcloud_units=m` 显式声明。
这些值描述真实记录，不进行隐式单位换算或坐标变换。

默认将点数统一为 1024，按点云中心和最大半径归一化；RGB 列不参与几何编码。
Lift3D 默认三层局部几何聚合、kNN 64，输出 128 个 token。
`pointcloud_sampling=random` 对应开源代码实际使用的随机中心；`fps` 是显式适配选项。
点数变化时采用确定性重采样，缺失点云或坐标声明时报错，不生成占位几何。

支持 AlohaMini／LeRobot v3 已记录的数值点云字段；外部 v3 数据使用
`--dataset.allow_unannotated=true`，按平台规则仅用于离线训练与评估。
本地实时完整模式需要观测源提供同名、同坐标系的点云字段；本接入不新增深度相机驱动。

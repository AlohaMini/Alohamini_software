# AC-DiT

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

命令在仓库根目录执行。示例保留 episode 1 用于验证；单回合数据省略 `--dataset.eval_episodes` 和 `--eval_steps`。

下载冻结的 SigLIP 图像／语言编码器：

```bash
hf download google/siglip-so400m-patch14-384 \
  --local-dir "$HOME/Alohamini_workspace/pretrained/siglip-so400m-patch14-384"

alohamini train --policy.type=acdit --policy.stage=mobility \
  --policy.use_pointcloud=false \
  --policy.assets_path="$HOME/Alohamini_workspace/pretrained/siglip-so400m-patch14-384" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --run_name=acdit_mobility --background
```

底盘阶段完成后，新建全身训练任务：

```bash
alohamini train \
  --policy.path="$HOME/Alohamini_workspace/runs/acdit_mobility/checkpoint" \
  --policy.stage=full \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --run_name=acdit_full --background
```

默认底盘阶段 20000 步、学习率 `1e-4`；全身阶段 30000 步、`1e-5`，冻结底盘分支。
每进程 batch 16，AdamW、恒定学习率、BF16、FP32 可训练参数、EMA 关闭。
显存不足时使用通用 FSDP2、减小 batch 并增加梯度累积。

需要非空任务文本与命名的 state/action；保留记录字段及单位，训练集均值／标准差归一化。
图像等比例补边至 384×384，再映射至 `[-1,1]`；图像取前一帧和当前帧，state 取当前帧，动作块为 2 行。
默认随机初始化动作头；外部 RDT 初始化使用 `mobility_pretrained_path`／`body_pretrained_path`，须显式提供命名槽位映射。

无点云数据时使用上例的 RGB 模式；点云模式须在第一阶段启用 `--policy.use_pointcloud=true`，
提供 `observation.point_cloud`（N×3 或 N×6）及真实坐标系、米制单位声明，不能生成占位点云。
第二阶段继承第一阶段相机、输入模式及统计量。底盘 checkpoint 不可直接部署全身控制；当前未接入远程动作块服务。
来源与预设见 [SOURCE.json](SOURCE.json)。

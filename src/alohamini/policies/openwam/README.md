# OpenWAM-α

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

GPU 训练与推理使用 CUDA。

命令在仓库根目录执行。示例中的 episode 1 用于验证；数据只有一个回合时，省略 `--dataset.eval_episodes` 和 `--eval_steps`。

```bash
hf download OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model \
  --revision 52df4e66c82c5c8b480adcc8d01f4db7415dfb56 \
  --local-dir "$HOME/Alohamini_workspace/pretrained/openwam"

alohamini train --policy.type=openwam \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --policy.path="$HOME/Alohamini_workspace/pretrained/openwam" \
  --run_name=openwam_task --batch_size=1 --steps=30000 \
  --keep_last_checkpoints=1 --background
```

基座须含 `checkpoint_step_*.safetensors` 和 `tokenizer/google/umt5-xxl/`，无需另下载 Wan 基座。
最多三路相机，按顺序组成 384×320 画布：上方主视角、左下与右下，缺失位置填黑；
`--policy.camera_layout='["observation.images.head","observation.images.left",null]'` 可显式指定，须覆盖全部所选相机。

默认动作块 32 行、视频取 0、4、…、32 行；`chunk_size` 须为 `4 × video_stride` 的整数倍。
默认 `--policy.action_representation=recorded`，保留记录 state/action 字段和单位，使用训练集 min/max；
从官方基座导入时重新初始化动作输入、动作输出和状态投影。
视频与动作联合流匹配，VAE、umT5 冻结；未来图像仅作训练监督，推理使用当前图像，默认同步去噪 10 步、shift 5。
默认 BF16、重计算开启；显存不足时使用通用 FSDP2。支持训练器 EMA 和普通远程动作块，未接入 RTC。

运行时只读本地资源，checkpoint 含冻结编码器及 tokenizer，可独立迁移。
保存时须预留两份完整 checkpoint 的空间；新文件写入成功后才清理旧文件。
来源与预设见 [SOURCE.json](SOURCE.json)。

要保留官方 80 维末端语义，使用可选的 `alpha80` 模式：

```bash
alohamini train --policy.type=openwam \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --policy.path="$HOME/Alohamini_workspace/pretrained/openwam" \
  --policy.action_representation=alpha80 \
  --state=joint_position,base_velocity,lift_height \
  --run_name=openwam_tcp80 --batch_size=1 --steps=30000 \
  --keep_last_checkpoints=1 --background
```

该模式自动设置 TCP state、绝对 TCP action 和 `rotation_6d_columns`，保留官方三个 80 维投影层。
FK 使用数据集的运动学描述；标准 AlohaMini 数据可由记录的标定信息和机型参考姿态解析。
非标准标定须用 `--data.kinematics.spec=/path/to/coordinates.json` 声明，
参见[关节、TCP 与相对目标](../../../../docs/training.md#关节tcp-与相对目标)。缺少标定或字段不匹配时拒绝训练。
双臂必须明确命名为 `left`、`right`，且所选 state 包含每臂完整关节位置与夹爪。

| 内容 | 80 维槽位（从 0 开始，含两端） | 编码／单位 |
| --- | --- | --- |
| 左臂位置、旋转、夹爪 | 0–2、3–8、9 | m、旋转矩阵前两列、闭合 −1／张开 +1 |
| 右臂位置、旋转、夹爪 | 34–36、37–42、43 | 同上 |
| 底盘 x、y、yaw 速度 | 68、69、70 | m/s、m/s、deg/s；平台预留槽扩展 |
| 升降高度 | 71 | mm；平台预留槽扩展 |
| 未使用的手部及预留通道 | 10–33、44–67、72–79 | 不计入动作损失 |

每臂末端仍为 6 自由度，但数值表示为位置 3＋旋转 6，共 9 个通道；加夹爪后每臂 10 个。
18 维关节记录经 FK 得到 **24 个有效通道**，进入网络前按名称填入 **80 维接口**。
夹爪保留官方专用槽，底盘和升降占用的 68–71 是本平台约定，官方未规定这些预留槽的含义。
位姿遵循标定描述保存的参考系，例如 `left_Base`、`right_Base`，不会自动改成世界坐标。

旋转 6D 保持原值；夹爪按标定的闭合／张开端点映射；其余通道使用训练集 min/max，输入截断到 [−1,1]。
使用训练集 min/max 的动作通道若上下界相同，解码时还原为训练常数；
例如底盘标签全零时，去噪残差不会转换成底盘速度。
checkpoint 保存槽位、统计量、标定和 FK/IK 几何；部署时自动反归一化，并以当前关节反馈为起点连续求解 IK，
恢复原 Host 动作顺序和单位，底盘／升降直接还原。退化旋转、不可达位姿或超出关节限位时拒绝该动作块。
原 18 维 `recorded` checkpoint 不能通过改开关变成 `alpha80`，须从官方基座重新微调。

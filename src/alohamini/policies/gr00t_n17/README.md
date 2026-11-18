# GR00T N1.7

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

命令在仓库根目录执行。示例中的 episode 1 用于验证；数据只有一个回合时，省略 `--dataset.eval_episodes` 和 `--eval_steps`。

下载本地基座与 Cosmos 处理器资源；Cosmos 仓库须先取得访问权限并登录 Hugging Face：

```bash
hf download nvidia/GR00T-N1.7-3B \
  --include 'config.json' 'model*.safetensors' 'model*.index.json' 'embodiment_id.json' \
  --local-dir "$HOME/Alohamini_workspace/pretrained/gr00t_n17"
hf download nvidia/Cosmos-Reason2-2B \
  --include '*.json' '*.txt' '*.model' '*.jinja' \
  --local-dir "$HOME/Alohamini_workspace/pretrained/gr00t_n17/backbone"

CUDA_VISIBLE_DEVICES=0,1 alohamini train --policy.type=gr00t_n17 \
  --policy.path="$HOME/Alohamini_workspace/pretrained/gr00t_n17" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --eval_steps=1000 \
  --distributed_backend=fsdp2 --num_processes=2 --batch_size=2 \
  --gradient_accumulation_steps=4 --run_name=gr00t_task --background
```

默认冻结 LLM、视觉编码器，训练动作头、投影与 VL 适配层；
`--policy.tune_llm=true`、`--policy.tune_visual=true` 分别解冻，`--policy.tune_top_llm_layers=4` 仅解冻最后四层 LLM。
默认 10000 步、AdamW `1e-4`、5% 预热与余弦下降，BF16 计算、FP32 可训练参数、重计算开启、EMA 关闭。
上例有效 batch 为 16；资源预设为每进程 8，按显存调整。

使用选中相机当前帧、非空任务文本、命名 state 与 40 行动作；state 保留记录坐标和单位。
默认动作是相对手臂关节位置与绝对夹爪目标；底盘、升降保持绝对目标。整段关节动作共用当前观测锚点，解码后再执行。
平台字段自动按名称绑定；外部数据须用 `--data.joint_delta_fields='{"action_joint":"state_joint"}'` 指明全部相对关节，保证对应字段使用相同坐标与单位。
`--data.action_representation=recorded` 保留原始绝对目标；已有 checkpoint 沿用保存的动作表示。
选择相对末端动作时，增加 `--data.state_representation=tcp --data.action_representation=tcp_delta --data.rotation_representation=rotation_6d --data.tcp_delta_frame=tool`；标准中位数据自动解析运动学。
图像先裁剪再缩放短边至 256，交给 Qwen3-VL 处理器；state/action 使用训练集 q01/q99 映射至 `[-1,1]` 并裁剪。
默认新 embodiment 10（官方 `NEW_EMBODIMENT` 编号），重新初始化其状态／动作投影。
已有 checkpoint 按保存的 `embodiment_id` 加载与续训；复用已有 embodiment 须指定 `state_slot_map`、`action_slot_map` 并关闭 `reinitialize_embodiment`。
运行时只读本地资源，checkpoint 可独立迁移；支持通用推理和远程动作块，未接入 RTC。
来源与预设见 [SOURCE.json](SOURCE.json)。

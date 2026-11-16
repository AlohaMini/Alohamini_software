# FastWAM

[策略索引](../../../../docs/training.md#策略选择) · [安装](../../../../docs/install.md#3-可选扩展) · [数据与输入](../../../../docs/training.md#3-选择数据与输入) · [保存与续训](../../../../docs/training.md#5-保存与续训)

命令在仓库根目录执行。示例保留 episode 1 用于验证；单回合数据省略 `--dataset.eval_episodes` 和 `--eval_steps`。

准备本地基座、Wan VAE／UMT5 和 tokenizer；下载只需执行一次：

```bash
hf download lerobot/fastwam_base \
  --revision 0a868ec1dcf6ff00bcdfa9b7196d6e211ed7e616 \
  --local-dir "$HOME/Alohamini_workspace/pretrained/fastwam_base"
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --include 'vae/*' 'text_encoder/*' \
  --local-dir "$HOME/Alohamini_workspace/pretrained/wan22_assets"
hf download google/umt5-xxl --include '*.json' '*.model' \
  --local-dir "$HOME/Alohamini_workspace/pretrained/umt5_tokenizer"

python -m alohamini.learning.train \
  --policy.type=fastwam --policy.device=cuda \
  --policy.path="$HOME/Alohamini_workspace/pretrained/fastwam_base" \
  --policy.vae_model_id="$HOME/Alohamini_workspace/pretrained/wan22_assets" \
  --policy.text_encoder_model_id="$HOME/Alohamini_workspace/pretrained/wan22_assets" \
  --policy.tokenizer_model_id="$HOME/Alohamini_workspace/pretrained/umt5_tokenizer" \
  --dataset.root="$HOME/Alohamini_workspace/datasets/task_demo_ready" \
  --dataset.eval_episodes='[1]' --cameras='["forward","wrist_right"]' \
  --output_dir="$HOME/Alohamini_workspace/runs/fastwam_01" \
  --batch_size=1 --policy.use_gradient_checkpointing=true \
  --save_freq=1000 --log_freq=100 --background
```

训练使用任务文本、当前 state、32 步动作和间隔 4 行的 9 帧视频；两路图像按名称排序，
各缩放至 224×224 后横向拼接。推理只用当前图像，不需要未来视频。
state/action 使用训练集 min/max，动作仍是记录的关节目标、底盘速度和升降高度，
不套用 LIBERO 的末端／夹爪变换。缺少未来帧时使用边界 padding 及对应损失掩码。

此入口读取转换后的 FastWAM `model.safetensors`，不直接读取作者的 `.pt`；
采用该发布基座的 flow shift=5。基座中的机器人输入／输出层缺失或维度不匹配时重新初始化。
checkpoint 不重复打包冻结的 Wan／UMT5 文件，迁移机器时须保留配置中的资产路径；
微调时也可用上述三个资源参数指定新路径。从平台 checkpoint 微调会恢复全部机器人输入／输出层。
完整基座需要较大显存，示例 batch 不保证适用于所有 GPU。

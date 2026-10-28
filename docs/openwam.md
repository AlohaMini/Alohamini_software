# OpenWAM-α

策略名：`openwam`。[论文](https://arxiv.org/abs/2609.07398v1)；代码和权重版本见 [SOURCE.json](../src/alohamini/policies/openwam/SOURCE.json)。

```bash
conda activate alohamini
python -m pip install -e '.[openwam]'
hf download OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model \
  --revision 52df4e66c82c5c8b480adcc8d01f4db7415dfb56 \
  --local-dir ~/Alohamini_workspace/pretrained/openwam

alohamini train --policy.type openwam \
  --dataset.root ~/Alohamini_workspace/datasets/task \
  --policy.path ~/Alohamini_workspace/pretrained/openwam \
  --run_name openwam_task --batch_size 1 --steps 30000
```

基座目录须包含一个 `checkpoint_step_*.safetensors` 和 `tokenizer/google/umt5-xxl/`。发布 checkpoint 已含 Wan2.2 视频 DiT、ActionDiT、VAE 和 umT5 权重，不另行下载 Wan 基座。运行时只读本地资产。

| 配置 | 默认值 |
|---|---|
| `chunk_size` / `n_action_steps` | 32 / 32 |
| `video_stride` | 4；视频取 0、4、…、32 行 |
| `canvas_size` | 384×320 |
| `num_steps` / `shift` | 10 / 5 |
| `lambda_video` / `lambda_action` | 1 / 1 |
| state / action 归一化 | 训练集 min/max |
| `gradient_checkpointing` / 精度 | true / BF16 |

采用 Wan2.2-TI2V-5B 与独立 ActionDiT，30 层联合 attention，当前帧保持干净；VAE、umT5 冻结。训练分别采样视频与动作噪声，推理同步去噪。**未来图像仅作训练监督**，验证预测和在线推理只使用当前图像。

最多三路相机，画布依次为上方主视角、左下和右下。默认使用所选相机顺序，缺失位置填黑；可指定 `--policy.camera_layout='["observation.images.head","observation.images.left",null]'`，须覆盖全部所选相机。`chunk_size` 必须是 `4 × video_stride` 的整数倍。

默认使用原始记录的 state/action 字段及单位，导入官方 80 维基座时重新初始化动作输入、动作输出和 state 投影；不凭维度推断官方末端、夹爪、灵巧手槽位。论文协议与平台适配随 checkpoint 保存。

使用通用训练器的 DDP、FSDP2、EMA 和 checkpoint；支持普通远程动作块推理，未接入 RTC 引导。平台 checkpoint 包含冻结编码器权重及 tokenizer，可独立迁移。显存不足时使用通用多卡配置，见[训练说明](training.md)。

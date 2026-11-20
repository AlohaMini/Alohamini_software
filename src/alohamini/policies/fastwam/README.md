# FastWAM

[Policy index](../../../../docs/training.md#policy-selection) · [Installation](../../../../docs/install.md#3-optional-extensions) · [Data and inputs](../../../../docs/training.md#3-select-data-and-inputs) · [Saving and resuming](../../../../docs/training.md#5-saving-and-resuming)

Run commands from the repository root. Examples reserve episode 1 for validation; for single-episode data, omit `--dataset.eval_episodes` and `--eval_steps`.

Prepare local base weights, Wan VAE/UMT5, and tokenizers; download once:

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

Training uses task text, current state, 32 action steps, and 9 video frames spaced 4 rows apart. Two views are sorted by name,
resized to 224×224 each, and concatenated horizontally. Inference uses only current images without future video.
State/action uses training-set min/max. Actions remain recorded joint targets, base velocities, and lift heights,
without LIBERO end-effector/gripper transforms. Missing future frames use boundary padding and corresponding loss masks.

This entry point reads converted FastWAM `model.safetensors`, not the author's `.pt` directly;
it uses flow shift=5 from that release. Missing or dimension-mismatched robot input/output layers are reinitialized.
Checkpoints do not duplicate frozen Wan/UMT5 files. Preserve asset paths in configuration when changing machines;
the three resource arguments above can also override paths for fine-tuning. Fine-tuning from platform checkpoints restores all robot input/output layers.
The full base requires substantial GPU memory; the example batch size may not fit every GPU.

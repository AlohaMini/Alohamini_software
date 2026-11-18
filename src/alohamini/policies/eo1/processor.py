"""EO-1 robot-only prompt and Qwen pixel processing with exact placeholder counts."""

import torch
from transformers import Qwen2_5_VLProcessor

from alohamini.learning.processor import Processor

SPECIAL_TOKENS = [
    "<|action_start|>",
    "<|action_pad|>",
    "<|action_end|>",
    "<|state_start|>",
    "<|state_pad|>",
    "<|state_end|>",
    "<|vla|>",
]


class EO1Processor:
    def __init__(self, config, stats, device):
        self.config = config
        self.numeric = Processor.from_config(config, stats, device)
        self.vlm = Qwen2_5_VLProcessor.from_pretrained(
            config.assets_path,
            local_files_only=True,
            use_fast=False,
        )
        self.vlm.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
        self.vlm.image_processor.min_pixels = config.image_min_pixels
        self.vlm.image_processor.max_pixels = config.image_max_pixels

    def __call__(self, batch):
        cfg = self.config
        tasks = batch.get("task")
        if not isinstance(tasks, (tuple, list)) or len(tasks) != len(batch["observation.state"]):
            raise ValueError("EO-1 requires a task for every sample")
        if any(not isinstance(task, str) or not task.strip() for task in tasks):
            raise ValueError("EO-1 task strings cannot be empty")
        numeric = self.numeric(
            {k: v for k, v in batch.items() if k != "task" and k not in cfg.image_features}
        )
        messages, pictures = [], []
        for i, task in enumerate(tasks):
            images = []
            for key in cfg.image_features:
                image = batch[key][i].detach().cpu()
                if image.dtype != torch.uint8:
                    image = image.clamp(0, 1).mul(255).round().to(torch.uint8)
                images.append(image)
            pictures.extend(images)
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful physical assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": im} for im in images],
                        {
                            "type": "text",
                            "text": f"<|state_start|><|state_pad|><|state_end|>{task}<|vla|>",
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "<|action_start|>"
                            + "<|action_pad|>" * cfg.chunk_size
                            + "<|action_end|>",
                        }
                    ],
                },
            ]
            messages.append(
                self.vlm.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                )
            )
        self.vlm.tokenizer.padding_side = "right" if "action" in batch else "left"
        inputs = self.vlm(text=messages, images=pictures, padding=True, return_tensors="pt")
        if inputs["input_ids"].shape[1] > cfg.tokenizer_max_length:
            raise ValueError(
                "EO-1 prompt exceeds tokenizer_max_length; "
                "do not truncate state/action placeholders"
            )
        numeric.update(
            {k: v.to(self.numeric.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        )
        numeric["state_token_id"] = self.vlm.tokenizer.convert_tokens_to_ids("<|state_pad|>")
        numeric["action_token_id"] = self.vlm.tokenizer.convert_tokens_to_ids("<|action_pad|>")
        return numeric

    def action(self, tensor, *, context=None):
        return self.numeric.action(tensor)

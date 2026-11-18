"""Long-WAM prompt, numeric and pixel processing with local tokenizer assets."""

import torch
from torchvision.transforms import functional as TF
from transformers import AutoTokenizer

from alohamini.learning.processor import Processor

from .pixels import _pil_bilinear_resize


class LongWAMProcessor:
    def __init__(self, config, stats, device):
        self.config = config
        self.numeric = Processor.from_config(config, stats, device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, local_files_only=True)

    def __call__(self, batch):
        tasks = batch.get("task")
        if not isinstance(tasks, (list, tuple)) or any(
            not isinstance(task, str) or not task.strip() for task in tasks
        ):
            raise ValueError("Long-WAM requires nonempty task strings")
        if len(tasks) != len(batch["observation.state"]):
            raise ValueError("Long-WAM task and state batch sizes differ")
        result = self.numeric({k: v for k, v in batch.items() if k != "task"})
        height, width = self.config.camera_size
        for key in self.config.camera_keys:
            image = result[key]
            if image.ndim not in (4, 5):
                raise ValueError("Long-WAM images require [B,3,H,W] or [B,T,3,H,W]")
            if image.ndim == 5:
                # The released training transform uses torchvision Resize(H,W).
                shape = image.shape[:-3]
                image = TF.resize(image.flatten(0, 1), [height, width], antialias=True)
                result[key] = image.reshape(*shape, 3, height, width).mul(2).sub(1)
            else:
                # Released online processing resizes uint8 with PIL then center-crops.
                ih, iw = image.shape[-2:]
                ratio = max(height / ih, width / iw)
                rh, rw = round(ih * ratio), round(iw * ratio)
                image = _pil_bilinear_resize(image.mul(255).round().to(torch.uint8), rh, rw)
                top, left = (rh - height) // 2, (rw - width) // 2
                result[key] = (
                    image[..., top : top + height, left : left + width].float().mul(2 / 255).sub(1)
                )
        result["observation.state"] = result["observation.state"].clamp(-5, 5)
        if "action" in result:
            result["action"] = result["action"].clamp(-5, 5)
        inputs = self.tokenizer(
            [self.config.prompt_template.format(task=" ".join(task.split())) for task in tasks],
            padding="max_length",
            truncation=True,
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        result["observation.language.tokens"] = inputs["input_ids"].to(self.numeric.device)
        result["observation.language.attention_mask"] = inputs["attention_mask"].to(
            self.numeric.device
        )
        return result

    def action(self, tensor, *, context=None):
        return self.numeric.action(tensor)

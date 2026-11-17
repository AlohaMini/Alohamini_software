"""Local Florence pixel processor and author BART instruction tokenization."""

import torch
from transformers import AutoImageProcessor, AutoTokenizer

from alohamini.policies.chunk import TaskProcessor


class XVLAProcessor(TaskProcessor):
    def __init__(self, config, stats, device):
        super().__init__(config, stats, device)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.assets_path, local_files_only=True)
        self.pixels = AutoImageProcessor.from_pretrained(config.assets_path, local_files_only=True)
        self.tokenizer.padding_side = "right"

    def __call__(self, batch):
        tasks = batch.get("task")
        if (
            not isinstance(tasks, (tuple, list))
            or len(tasks) != len(batch[next(iter(self.config.image_features))])
            or any(not isinstance(text, str) or not text.strip() for text in tasks)
        ):
            raise ValueError("X-VLA requires a nonempty instruction per sample")
        result = self.numeric(
            {
                key: value
                for key, value in batch.items()
                if key != "task" and key not in self.config.image_features
            }
        )
        images = []
        for index in range(len(tasks)):
            for key in self.config.image_features:
                pixels = batch[key][index].detach().cpu()
                if pixels.dtype != torch.uint8:
                    if not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
                        raise ValueError("X-VLA expects raw RGB pixels or RGB in [0,1]")
                    pixels = pixels.mul(255).round().byte()
                images.append(pixels.numpy())
        values = self.pixels(images, return_tensors="pt")["pixel_values"]
        if values.ndim != 4:
            raise ValueError("X-VLA expects one Florence image tensor per selected view")
        ids = self.tokenizer(
            tasks,
            padding="max_length",
            max_length=self.config.tokenizer_max_length,
            truncation=False,
            return_tensors="pt",
        )["input_ids"]
        if ids.shape[-1] > self.config.tokenizer_max_length:
            raise ValueError("X-VLA instructions exceed tokenizer_max_length")
        result["input_ids"] = ids.to(self.numeric.device)
        result["image_input"] = values.reshape(
            len(tasks), len(self.config.image_features), *values.shape[1:]
        ).to(self.numeric.device)
        result["image_mask"] = torch.ones(
            (len(tasks), len(self.config.image_features)),
            dtype=torch.bool,
            device=self.numeric.device,
        )
        result["domain_id"] = torch.full(
            (len(tasks),),
            self.config.default_domain_id,
            dtype=torch.long,
            device=self.numeric.device,
        )
        return result

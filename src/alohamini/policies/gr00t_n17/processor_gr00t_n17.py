"""Normalize recorded coordinates and pack current RGB/language with the local Qwen processor."""

import re

import torch
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

from alohamini.learning.processor import Processor


class GR00TN17Processor:
    def __init__(self, config, stats, processor, device, model):
        self.config, self.vlm, self.model = config, processor, model
        self.numeric = Processor.from_config(config, stats, device)
        jitter = config.network.get(
            "color_jitter_params", dict(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)
        )
        self.jitter = ColorJitter(**(jitter or {}))

    def __call__(self, batch):
        cfg = self.config
        tasks = batch.get("task")
        if not isinstance(tasks, (tuple, list)) or any(
            not isinstance(t, str) or not t.strip() for t in tasks
        ):
            raise ValueError("GR00T requires a nonempty language task for every observation")
        if len(tasks) != len(batch["observation.state"]):
            raise ValueError("Task batch differs from the observations")
        numeric = self.numeric(
            {k: v for k, v in batch.items() if k not in cfg.image_features and k != "task"}
        )
        for key in ("observation.state", "action"):
            if key in numeric and self.numeric.modes[key] == "MIN_MAX":
                numeric[key] = numeric[key].clamp(-1, 1)
        pictures = [[] for _ in tasks]
        train = self.model.training and cfg.image_augmentation
        # Replay the same crop location and color factors across each sample's
        # views, while preserving independent samples and the trainer RNG state.
        augmentation = []
        if train:
            for _ in tasks:
                location = torch.rand(2).tolist()
                color = ColorJitter.get_params(
                    self.jitter.brightness,
                    self.jitter.contrast,
                    self.jitter.saturation,
                    self.jitter.hue,
                )
                augmentation.append((location, color))
        for key in cfg.image_features:
            values = batch[key]
            if values.ndim == 5:
                values = values[:, -1]
            for i, image in enumerate(values):
                image = image.detach().cpu()
                image = image.float() / 255 if image.dtype == torch.uint8 else image.float()
                image = TF.resize(image, cfg.shortest_image_edge, antialias=True)
                h, w = image.shape[-2:]
                ch, cw = max(1, int(h * cfg.crop_fraction)), max(1, int(w * cfg.crop_fraction))
                top = int(augmentation[i][0][0] * (h - ch + 1)) if train else (h - ch) // 2
                left = int(augmentation[i][0][1] * (w - cw + 1)) if train else (w - cw) // 2
                image = TF.resize(
                    TF.crop(image, top, left, ch, cw), cfg.shortest_image_edge, antialias=True
                )
                if train:
                    order, *factors = augmentation[i][1]
                    operations = (
                        TF.adjust_brightness,
                        TF.adjust_contrast,
                        TF.adjust_saturation,
                        TF.adjust_hue,
                    )
                    for index in order.tolist():
                        if factors[index] is not None:
                            image = operations[index](image, factors[index])
                pictures[i].append(image.clamp(0, 1))
        texts = []
        for task, images in zip(tasks, pictures, strict=True):
            task = re.sub(r"[^\w\s]", "", task.lower()) if cfg.formalize_language else task
            conversation = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": image} for image in images],
                        {"type": "text", "text": task},
                    ],
                }
            ]
            texts.append(
                self.vlm.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                )
            )
        inputs = self.vlm(
            text=texts,
            images=[im for ims in pictures for im in ims],
            padding=True,
            return_tensors="pt",
            do_rescale=False,
        )
        numeric.update(
            {k: v.to(self.numeric.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        )
        return numeric

    def action(self, tensor, *, context=None):
        if self.numeric.modes["action"] == "MIN_MAX":
            tensor = tensor.clamp(-1, 1)
        return self.numeric.action(tensor)

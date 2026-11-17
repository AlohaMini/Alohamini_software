"""DM0.5 chat template, raw square-padded RGB and pre-padding state bins."""

import torch
from torch.nn import functional as F
from transformers import AutoProcessor

from alohamini.policies.chunk import TaskProcessor

from .core.tokenization import DM05Tokenization, action_to_bin_tokens


class DM05Processor(TaskProcessor):
    def __init__(self, config, stats, device):
        super().__init__(config, stats, device)
        self.config = config
        self.assets = AutoProcessor.from_pretrained(
            config.assets_path, local_files_only=True, trust_remote_code=False
        )
        self.tokens = DM05Tokenization(
            self.assets, max_length=config.tokenizer_max_length, add_state=config.add_state
        )

    def __call__(self, batch):
        tasks = batch.get("task")
        if (
            not isinstance(tasks, (list, tuple))
            or len(tasks) != len(batch["observation.state"])
            or any(not isinstance(task, str) or not task.strip() for task in tasks)
        ):
            raise ValueError("DM05 requires one nonempty task per observation")
        result = self.numeric(
            {
                key: value
                for key, value in batch.items()
                if key != "task" and key not in self.config.image_features
            }
        )
        bins = action_to_bin_tokens(result["observation.state"]).tolist()
        pictures = []
        for key in self.config.image_features:
            image = batch[key].detach().cpu()
            if image.dtype == torch.uint8:
                image = image.float() / 255
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("DM05 requires RGB images in [0,1] or uint8")
            height, width = image.shape[-2:]
            side = max(height, width)
            top, left = (side - height) // 2, (side - width) // 2
            pictures.append(F.pad(image, (left, side - width - left, top, side - height - top)))
        samples = [
            dict(
                prompt=task,
                images=[image[i] for image in pictures],
                state_bins=bins[i],
                meta_data=dict(dataset_meta=dict(image_keys=list(self.config.image_features))),
            )
            for i, task in enumerate(tasks)
        ]
        encoded = self.tokens.tokenize_robot_batch(samples)
        result.update(
            {
                key: value.to(self.numeric.device)
                for key, value in encoded.items()
                if isinstance(value, torch.Tensor)
            }
        )
        return result

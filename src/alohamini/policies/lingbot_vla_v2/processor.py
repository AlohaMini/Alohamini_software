"""Official Qwen preprocessing with reversible, declared platform action fields."""

import torch
from torchvision.transforms.v2.functional import resize

from alohamini.policies.chunk import local_bpe_assets

from .semantic import validate_layout


class LingBotProcessor:
    decode_before_queue = True

    def __init__(self, config, stats, device):
        from transformers import AutoImageProcessor, AutoTokenizer

        validate_layout(config.layout)
        self.config, self.device = config, torch.device(device)
        path = local_bpe_assets(config.text_tokenizer_path or config.tokenizer_path)
        self.images = AutoImageProcessor.from_pretrained(
            config.image_processor_path or config.tokenizer_path,
            local_files_only=True,
            max_pixels=config.image_max_pixels,
            min_pixels=config.image_min_pixels,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.stats = {
            k: {
                n: torch.as_tensor(v[n], device=device, dtype=torch.float32)
                for n in ("mean", "std")
            }
            for k, v in stats.items()
            if k in ("action", "observation.state")
        }
        self.active = [i for i, slot in enumerate(config.layout["slots"]) if slot is not None]
        self.slots = [config.layout["slots"][i] for i in self.active]

    def _normalize(self, key, value):
        s = self.stats[key]
        return (value - s["mean"]) / (s["std"] + 1e-8)

    def _pack(self, value):
        result = value.new_zeros((*value.shape[:-1], 55))
        result[..., self.slots] = value[..., self.active]
        return result

    def __call__(self, batch):
        state = batch["observation.state"].to(self.device)
        tasks = batch.get("task")
        if (
            not isinstance(tasks, (list, tuple))
            or len(tasks) != len(state)
            or any(not t.strip() for t in tasks)
        ):
            raise ValueError("LingBot requires a nonempty task for each sample")
        result = {
            "_action_context": state,
            "observation.state": self._pack(self._normalize("observation.state", state)),
        }
        for key in ("_loss_weight", "_aux_loss_weight", "action_is_pad", "noise", "time"):
            if key in batch:
                result[key] = (
                    batch[key].to(self.device) if torch.is_tensor(batch[key]) else batch[key]
                )
        mask = torch.zeros(len(state), 55, dtype=torch.bool, device=self.device)
        mask[:, self.slots] = True
        result["state_joint_mask"] = result["action_joint_mask"] = mask
        if "action" in batch:
            action = batch["action"].to(self.device).clone()
            dims = self.config.layout["relative"]
            action[..., dims] -= state[:, None, dims]
            result["action"] = self._pack(self._normalize("action", action))
            result["joint_mask"] = mask[:, None] & ~result["action_is_pad"][..., None]
        frames = []
        for index, key in enumerate(self.config.image_features):
            value = batch[key].to(self.device)
            value = value.float() / 255 if value.dtype == torch.uint8 else value.float()
            value = resize(value, list(self.config.resize_imgs_with_padding), antialias=True)
            if value.ndim == 5:
                if index == 0:
                    result["teacher_images"] = (value[:, [0, -1]] * 255).round().clamp(0, 255) / 255
                value = value[:, 0]
            frames.append(value)
        values = torch.stack(frames, 1)
        processed = self.images(list((values * 255).flatten(0, 1)), return_tensors="pt")
        result["images"] = (
            processed["pixel_values"].to(self.device).unflatten(0, (len(state), len(frames), -1))
        )
        result["image_grid_thw"] = (
            processed["image_grid_thw"].to(self.device).view(len(state), len(frames), 3)
        )
        result["img_masks"] = torch.ones(
            len(state), len(frames), dtype=torch.bool, device=self.device
        )
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=False
            )
            for t in tasks
        ]
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        result["observation.language.tokens"] = tokens["input_ids"].to(self.device)
        result["observation.language.attention_mask"] = (
            tokens["attention_mask"].bool().to(self.device)
        )
        return result

    def action(self, tensor, *, context=None):
        if context is None:
            raise ValueError("LingBot decoding requires the observation anchor")
        single = tensor.ndim == 2
        value = tensor[:, None] if single else tensor
        result = context[:, None].expand(-1, value.shape[1], -1).clone().to(value)
        s = self.stats["action"]
        decoded = value[..., self.slots] * s["std"][self.active] + s["mean"][self.active]
        result[..., self.active] = decoded
        dims = self.config.layout["relative"]
        result[..., dims] += context[:, None, dims].to(value)
        return result[:, 0] if single else result

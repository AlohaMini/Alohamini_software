"""Current-frame Qwen prompts and explicitly selected future JEPA training clips."""

from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoProcessor, AutoVideoProcessor

from alohamini.learning.processor import Processor


def qwen_assets(config):
    return Path(config.assets_path) / "qwen"


def expand_tokens(processor, config):
    tokens = [config.special_action_token.format(i) for i in range(config.chunk_size * 4)]
    processor.tokenizer.add_tokens([*tokens, config.embodied_action_token], special_tokens=True)
    if len(processor.tokenizer) > config.qwen_config["text_config"]["vocab_size"]:
        raise ValueError("VLA-JEPA readout tokens exceed saved Qwen embedding capacity")
    return processor.tokenizer.convert_tokens_to_ids(
        tokens
    ), processor.tokenizer.convert_tokens_to_ids(config.embodied_action_token)


class VLAJEPAProcessor:
    def __init__(self, config, stats, device):
        self.config = config
        self.numeric = Processor.from_config(config, stats, device)
        self.vlm = AutoProcessor.from_pretrained(qwen_assets(config), local_files_only=True)
        self.vlm.tokenizer.padding_side = "left"
        self.action_ids, self.embodied_id = expand_tokens(self.vlm, config)
        self.video = (
            AutoVideoProcessor.from_pretrained(
                Path(config.assets_path) / "vjepa2", local_files_only=True
            )
            if config.enable_world_model
            else None
        )

    def __call__(self, batch):
        cfg = self.config
        tasks = batch.get("task")
        state = batch["observation.state"]
        if (
            not isinstance(tasks, (list, tuple))
            or len(tasks) != len(state)
            or any(not isinstance(v, str) or not v.strip() for v in tasks)
        ):
            raise ValueError("VLA-JEPA requires one nonempty task per observation")
        result = self.numeric({k: v for k, v in batch.items() if k != "task"})
        if cfg.resize_images_to:
            for key in cfg.image_features:
                value = result[key]
                if tuple(value.shape[-2:]) != tuple(cfg.resize_images_to):
                    lead = value.shape[:-3]
                    value = F.interpolate(
                        value.reshape(-1, *value.shape[-3:]), size=cfg.resize_images_to, mode="area"
                    )
                    result[key] = value.reshape(*lead, *value.shape[-3:])
        images = {
            key: value[:, 0] if value.ndim == 5 else value
            for key, value in result.items()
            if key in cfg.image_features
        }
        steps = cfg.num_video_frames // cfg.jepa_tubelet_size - 1
        action_prompt = "".join(
            cfg.special_action_token.format(i) * cfg.num_action_tokens_per_timestep
            for i in range(steps)
        )
        messages = [
            [
                dict(
                    role="user",
                    content=[
                        *[dict(type="image", image=images[key][i]) for key in cfg.image_features],
                        dict(
                            type="text",
                            text=cfg.prompt_template.format(
                                instruction=task,
                                actions=action_prompt,
                                e_actions=cfg.embodied_action_token
                                * cfg.num_embodied_action_tokens_per_instruction,
                            ),
                        ),
                    ],
                )
            ]
            for i, task in enumerate(tasks)
        ]
        text = [
            self.vlm.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        encoded = self.vlm(
            text=text,
            images=[images[key][i] for i in range(len(tasks)) for key in cfg.image_features],
            do_rescale=False,
            padding=True,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] > cfg.tokenizer_max_length:
            raise ValueError(
                "VLA-JEPA prompt exceeds tokenizer_max_length; readouts cannot be truncated"
            )
        result.update(
            {
                k: v.to(self.numeric.device)
                for k, v in encoded.items()
                if isinstance(v, torch.Tensor)
            }
        )
        ids = result["input_ids"]
        action_mask = torch.isin(ids, torch.tensor(self.action_ids, device=ids.device))
        embodied_mask = ids == self.embodied_id
        if (
            not (action_mask.sum(-1) == steps * cfg.num_action_tokens_per_timestep).all()
            or not (embodied_mask.sum(-1) == cfg.num_embodied_action_tokens_per_instruction).all()
        ):
            raise ValueError("VLA-JEPA prompt readouts differ from saved token geometry")
        result["action_token_mask"], result["embodied_token_mask"] = action_mask, embodied_mask
        if cfg.enable_world_model and "action" in batch:
            clips = torch.stack([result[key] for key in cfg.world_model_cameras], dim=1)
            if clips.ndim != 6 or clips.shape[2] != cfg.num_video_frames:
                raise ValueError("VLA-JEPA training requires complete configured video windows")
            b, v, t, c, h, w = clips.shape
            encoded_video = self.video(
                videos=list(clips.reshape(b * v, t, c, h, w)),
                return_tensors="pt",
                do_rescale=False,
                do_sample_frames=False,
            )
            result["video_pixels"] = encoded_video["pixel_values_videos"].to(self.numeric.device)
        return result

    def action(self, tensor, *, context=None):
        if self.config.clip_normalized_actions:
            tensor = tensor.clamp(-1, 1)
        return self.numeric.action(tensor)

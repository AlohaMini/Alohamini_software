"""Original WALL flow prompt, Qwen vision patches and explicit named robot slots."""

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLProcessor

from alohamini.learning.processor import Processor
from alohamini.policies.slots import NamedSlots

ACTION_TOKEN = "<|action|>"
PROPRIO_TOKEN = "<|propri|>"
CAMERA_LABELS = dict(
    face_view="front view", right_wrist_view="right wrist view", left_wrist_view="left wrist view"
)


class WallOSS05Processor:
    def __init__(self, config, stats, device):
        self.config = config
        self.numeric = Processor.from_config(config, stats, device)
        self.vlm = Qwen2_5_VLProcessor.from_pretrained(config.assets_path, local_files_only=True)
        self.vlm.tokenizer.add_tokens([PROPRIO_TOKEN, ACTION_TOKEN], special_tokens=True)
        self.vlm.tokenizer.padding_side = "left"
        self.action_token_id = self.vlm.tokenizer.convert_tokens_to_ids(ACTION_TOKEN)
        if self.action_token_id != config.core_config["action_token_id"]:
            raise ValueError("WALL action token ID does not match the saved model vocabulary")
        if (
            len(self.vlm.tokenizer)
            > config.core_config.get("text_config", config.core_config)["vocab_size"]
        ):
            raise ValueError("WALL processor vocabulary exceeds model embeddings")
        self.state_slots = NamedSlots(config.state_names, config.state_slots, 26)
        self.action_slots = NamedSlots(config.action_names, config.action_slots, 26)

    def image(self, image):
        image = image.detach().cpu()
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("WALL expects CHW RGB images")
        if image.dtype != torch.uint8:
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("WALL expects uint8 RGB or finite floats in [0,1]")
            image = image.mul(255).round().to(torch.uint8)
        value = Image.fromarray(image.permute(1, 2, 0).numpy())
        width, height = value.size
        edge = self.config.image_long_edge
        size = (
            (edge, max(1, int(edge * height / width)))
            if width > height
            else (max(1, int(edge * width / height)), edge)
        )
        return value.resize(size)

    def __call__(self, batch):
        cfg = self.config
        tasks = batch.get("task")
        if (
            not isinstance(tasks, (tuple, list))
            or len(tasks) != len(batch["observation.state"])
            or any(not isinstance(t, str) or not t.strip() for t in tasks)
        ):
            raise ValueError("WALL requires one nonempty task per sample")
        result = self.numeric(
            {k: v for k, v in batch.items() if k != "task" and k not in cfg.image_features}
        )
        state = self.state_slots.pack(result["observation.state"])
        # The original tokenizer digitizes into 512 bins; outliers saturate here only.
        # Numeric action targets are never clipped by this state tokenization.
        bins = np.linspace(-1, 1, 513)[:-1]
        discrete = np.digitize(state.detach().float().cpu().numpy(), bins=bins) - 1
        observation = "Observation:"
        for camera in cfg.camera_key_mapping.values():
            label = CAMERA_LABELS.get(camera, camera.replace("_", " "))
            observation += f" {label}: <|vision_start|><|image_pad|><|vision_end|>"
        prompts = [
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{observation}\nInstruction: {task}\n"
            "Predict the next action in robot action.\n"
            f"Proprioception: {' '.join(map(str, discrete[i]))}\n"
            "<|im_end|>\n<|im_start|>assistant\n" + ACTION_TOKEN * cfg.chunk_size
            for i, task in enumerate(tasks)
        ]
        images = [
            self.image(batch[key][i]) for i in range(len(tasks)) for key in cfg.camera_key_mapping
        ]
        encoded = self.vlm(text=prompts, images=images, padding=True, return_tensors="pt")
        if encoded["input_ids"].shape[1] > cfg.tokenizer_max_length:
            raise ValueError(
                "WALL prompt exceeds tokenizer_max_length; action/image tokens cannot be truncated"
            )
        counts = (encoded["input_ids"] == self.action_token_id).sum(-1)
        if not (counts == cfg.chunk_size).all():
            raise ValueError("WALL prompts must contain exactly chunk_size action tokens")
        result.update(
            {
                k: v.to(self.numeric.device)
                for k, v in encoded.items()
                if isinstance(v, torch.Tensor)
            }
        )
        result["moe_token_types"] = result["input_ids"] == self.action_token_id
        result["dof_mask"] = self.action_slots.mask(
            len(tasks), cfg.chunk_size, device=self.numeric.device
        )
        if "action" in result:
            result["action"] = self.action_slots.pack(result["action"])
        return result

    def action(self, tensor, *, context=None):
        return self.numeric.action(tensor)

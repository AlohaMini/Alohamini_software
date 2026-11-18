"""EO-1 core through the platform tensor policy and global loss-reduction contract."""

import torch
from torch.nn import functional as F
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from alohamini.policies.chunk import ChunkPolicy

from .core import EO1VisionFlowMatchingModel


class EO1Policy(ChunkPolicy):
    name = "eo1"

    def __init__(self, config):
        super().__init__()
        self.config = config
        vlm_cfg = Qwen2_5_VLConfig(**config.vlm_config)
        vlm_cfg._attn_implementation = config.attn_implementation
        vlm = Qwen2_5_VLForConditionalGeneration._from_config(
            vlm_cfg, dtype=getattr(torch, config.dtype)
        )
        self.model = EO1VisionFlowMatchingModel(config, vlm)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.reset()

    def inputs(self, batch):
        keys = (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
            "state_token_id",
            "action_token_id",
        )
        result = {key: batch[key] for key in keys if key in batch}
        result["states"] = F.pad(
            batch["observation.state"],
            (0, self.config.max_state_dim - batch["observation.state"].shape[-1]),
        )
        return result

    def forward(self, batch):
        action = F.pad(batch["action"], (0, self.config.max_action_dim - batch["action"].shape[-1]))
        output = self.model(
            **self.inputs(batch), action=action, action_is_pad=batch.get("action_is_pad")
        )
        loss = self.config.flow_loss_weight * output.flow_loss
        return loss * batch.get("_loss_weight", 1.0), {"flow_loss": output.flow_loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        self.eval()
        inputs = self.inputs(batch)
        # Validation batches carry training padding. Align valid tokens to the
        # right for cached action sampling without changing training tokenization.
        mask = inputs["attention_mask"]
        order = torch.argsort(mask, dim=1, stable=True)
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if key in inputs:
                inputs[key] = inputs[key].gather(1, order)
        return self.model.sample_actions(**inputs).float()[
            ..., : self.config.action_feature.shape[0]
        ]

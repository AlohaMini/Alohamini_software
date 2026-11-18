"""Native WALL joint experts with platform reduction, precision and action queues."""

from copy import deepcopy

import torch
from transformers import Qwen2_5_VLConfig

from alohamini.policies.chunk import ChunkPolicy
from alohamini.policies.slots import NamedSlots

from .core import WallOSS05Model, WallOSS05RMSNorm


class WallOSS05Policy(ChunkPolicy):
    name = "wall_oss_05"

    def __init__(self, config):
        super().__init__()
        self.config = config
        core = Qwen2_5_VLConfig(**deepcopy(config.core_config))
        core._attn_implementation = "sdpa"
        self.model = WallOSS05Model._from_config(core, dtype=getattr(torch, config.dtype))
        self.model.action_preprocessor.float()
        for module in self.model.modules():
            if isinstance(module, WallOSS05RMSNorm):
                module.float()
        self.model.model.gradient_checkpointing = config.gradient_checkpointing
        if config.gradient_checkpointing:
            self.model.visual.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.slots = NamedSlots(config.action_names, config.action_slots, 26)
        self.reset()

    @staticmethod
    def inputs(batch):
        return {
            key: batch[key]
            for key in (
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
                "moe_token_types",
                "dof_mask",
            )
            if key in batch
        }

    def forward(self, batch):
        size = len(batch["action"])
        scheduler = self.config.core_config.get("noise_scheduler", {})
        beta = torch.distributions.Beta(
            scheduler.get("beta_alpha", 1.5), scheduler.get("beta_beta", 1.0)
        )
        sample_time = (1 - beta.sample((size,))).to(batch["action"].device) * scheduler.get(
            "s", 0.999
        )
        mask = batch.get("action_is_pad")
        valid = (
            torch.ones(batch["action"].shape[:2], dtype=torch.bool, device=batch["action"].device)
            if mask is None
            else ~mask
        )
        output = self.model(
            **self.inputs(batch),
            action_chunk=batch["action"],
            sample_time=sample_time,
            flow_loss_mask=valid,
        )
        return output.flow_loss * batch.get("_flow_weight", 1.0), {
            "flow_loss": output.flow_loss.detach()
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        self.eval()
        result = self.model.generate_flow_action(
            **self.inputs(batch),
            action_horizon=self.config.chunk_size,
            action_dim=26,
            num_inference_timesteps=self.config.num_inference_timesteps,
        )
        return self.slots.unpack(result["predict_action"]).float()

"""DM0.5 action expert through platform loss reduction, assets and parameter storage."""

from copy import deepcopy

import torch
from torch.nn import functional as F

from alohamini.policies.chunk import ChunkPolicy

from .core.modeling import DM05CoreModelConfig, DM05ForCausalLM


class DM05Policy(ChunkPolicy):
    name = "dm05"

    def __init__(self, config):
        super().__init__()
        self.config = config
        payload = deepcopy(config.core_config)
        payload.update(
            chunk_size=config.chunk_size,
            bf16=config.dtype == "bfloat16",
            freeze_vlm_embedding=config.freeze_vlm_embedding,
            llm_attn_implementation=config.llm_attn_implementation,
            vision_attn_implementation=config.vision_attn_implementation,
            action_attn_implementation=config.action_attn_implementation,
            vlm_gradient_checkpointing=config.vlm_gradient_checkpointing,
            ae_gradient_checkpointing=config.ae_gradient_checkpointing,
            ae_gradient_checkpointing_layers=config.ae_gradient_checkpointing_layers,
        )
        core = DM05CoreModelConfig(**payload)
        self.model = DM05ForCausalLM._from_config(core, dtype=getattr(torch, config.dtype))
        self.reset()

    def inputs(self, batch):
        keys = ("input_ids", "attention_mask", "pixel_values", "token_type_ids", "position_ids")
        inputs = {key: batch[key] for key in keys if key in batch}
        width = self.model.config.action_dim
        mask = torch.zeros(
            (len(batch["input_ids"]), width), dtype=torch.bool, device=batch["input_ids"].device
        )
        mask[:, : self.config.action_feature.shape[0]] = True
        inputs["action_dim_mask"] = mask
        return inputs

    def forward(self, batch):
        inputs = self.inputs(batch)
        width = self.model.config.action_dim
        actions = F.pad(batch["action"], (0, width - batch["action"].shape[-1])).to(
            self.model.model.action_in_proj.weight.dtype
        )
        result = self.model(
            **inputs,
            actions=actions,
            action_is_pad=batch.get("action_is_pad"),
            has_actions=torch.ones(len(actions), dtype=torch.bool, device=actions.device),
        )
        loss = result.fm_loss
        return loss * batch.get("_loss_weight", 1.0), {"flow_loss": loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        return self.model.inference_action(
            **self.inputs(batch), diffusion_steps=self.config.diffusion_steps
        ).float()[..., : self.config.action_feature.shape[0]]

    def save_assets(self, path):
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(
            self.config.assets_path, local_files_only=True, trust_remote_code=False
        ).save_pretrained(path)

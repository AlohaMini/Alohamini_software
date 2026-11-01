"""Official N1.7 action head with local Qwen3-VL and native action chunks."""

from copy import deepcopy

import torch
from torch import nn
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature

from alohamini.policies.chunk import ChunkPolicy, local_assets

from ._vendor.action_head import Gr00tN1d7ActionHead
from ._vendor.configuration import Gr00tN1d7Config


class LocalQwenBackbone(nn.Module):
    def __init__(self, config, head):
        super().__init__()
        options = deepcopy(config.backbone_config)
        options["text_config"]["num_hidden_layers"] = head.select_layer
        backbone = Qwen3VLConfig(**options)
        backbone._attn_implementation = "sdpa"
        self.model = Qwen3VLForConditionalGeneration(backbone).to(
            dtype=getattr(torch, config.backbone_dtype)
        )
        self.model.model.language_model.requires_grad_(config.tune_llm)
        self.model.model.visual.requires_grad_(config.tune_visual)
        if config.tune_top_llm_layers:
            for layer in self.model.model.language_model.layers[-config.tune_top_llm_layers :]:
                layer.requires_grad_(True)
        if self.model.lm_head.weight is not self.model.get_input_embeddings().weight:
            self.model.lm_head.requires_grad_(False)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def forward(self, batch):
        inputs = {
            k: batch[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        }
        output = self.model.model(**inputs, use_cache=False)
        return BatchFeature(
            dict(
                backbone_features=output.last_hidden_state.float(),
                backbone_attention_mask=batch["attention_mask"].bool(),
                image_mask=batch["input_ids"] == self.model.config.image_token_id,
            )
        )


class GR00TN17Policy(ChunkPolicy):
    name = "gr00t_n17"

    def __init__(self, config):
        super().__init__()
        self.config = config
        head = Gr00tN1d7Config(**config.network)
        head.action_horizon = config.chunk_size
        for key in ("tune_projector", "tune_diffusion_model", "tune_vlln"):
            setattr(head, key, getattr(config, key))
        if head.state_history_length != 1:
            raise ValueError(
                "The native N1.7 recorded-state path currently uses one state observation"
            )
        self.backbone = LocalQwenBackbone(config, head)
        self.action_head = Gr00tN1d7ActionHead(head)
        self.action_head.model.gradient_checkpointing = config.gradient_checkpointing
        if hasattr(self.action_head.vl_self_attention, "gradient_checkpointing"):
            self.action_head.vl_self_attention.gradient_checkpointing = (
                config.gradient_checkpointing
            )
        self.vlm_processor = AutoProcessor.from_pretrained(
            local_assets(config.assets_path),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=False,
        )
        self.vlm_processor.tokenizer.padding_side = "left"
        self.register_buffer(
            "state_slots",
            torch.tensor([config.state_slot_map[n] for n in config.state_names]),
            persistent=False,
        )
        self.register_buffer(
            "action_slots",
            torch.tensor([config.action_slot_map[n] for n in config.action_names]),
            persistent=False,
        )
        self.reset()

    def train(self, mode=True):
        super().train(mode)
        if not self.config.tune_llm and not self.config.tune_top_llm_layers:
            self.backbone.model.model.language_model.eval()
        if not self.config.tune_visual:
            self.backbone.model.model.visual.eval()
        return self

    def _head_input(self, batch):
        cfg, head = self.config, self.action_head.config
        state = batch["observation.state"]
        state = state[:, -1] if state.ndim == 3 else state
        padded = state.new_zeros(len(state), 1, head.max_state_dim)
        padded[:, 0, self.state_slots] = state
        mask = state.new_zeros(len(state), cfg.chunk_size, head.max_action_dim)
        mask[:, :, self.action_slots] = 1
        values = dict(
            state=padded,
            embodiment_id=torch.full(
                (len(state),), cfg.embodiment_id, device=state.device, dtype=torch.long
            ),
            action_mask=mask,
        )
        if "action" in batch:
            action = state.new_zeros(mask.shape)
            action[:, :, self.action_slots] = batch["action"]
            values["action"] = action
            values["action_mask"] = mask * (~batch["action_is_pad"])[..., None]
        return BatchFeature(values)

    def forward(self, batch):
        output = self.action_head(self.backbone(batch), self._head_input(batch))
        loss = output["loss"]
        return loss * batch.get("_loss_weight", 1.0), {"flow_loss": loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        # Supervised actions are never used as an inference prefix.
        observation = {k: v for k, v in batch.items() if k not in ("action", "action_is_pad")}
        output = self.action_head.get_action(
            self.backbone(observation), self._head_input(observation)
        )
        return output.action_pred[:, :, self.action_slots].float()

    def save_assets(self, path):
        self.vlm_processor.save_pretrained(path)
        self.backbone.model.config.save_pretrained(path)

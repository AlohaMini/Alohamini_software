"""Cosmos3 MoT through platform losses, parameter groups and action chunks."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.modeling_utils import local_torch_dtype

from alohamini.policies.chunk import ChunkPolicy

from .core import FlowState, OmniMoTModel


class Cosmos3Policy(ChunkPolicy):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = "cosmos3_" + config.variant
        with local_torch_dtype(getattr(torch, config.dtype)):
            self.model = OmniMoTModel(config.backend())
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, local_files_only=True)
        if config.gradient_checkpointing:
            self.model.network.gradient_checkpointing_enable()
        if config.compile_model:
            self.model.network.compile_decoder_layers(config.compile_mode)
            self.model.network.compile_vae_encoder(config.compile_mode)
        self.last_predicted_latents = None
        self.reset()
        self.model.network.vae.eval()

    def train(self, mode=True):
        super().train(mode)
        self.model.network.vae.eval()
        return self

    def get_optim_params(self):
        groups = [
            dict(name="backbone", params=[]),
            dict(
                name="action",
                params=[],
                lr_scale=self.config.optimizer_action_lr_multiplier,
            ),
        ]
        for name, value in self.named_parameters():
            if value.requires_grad:
                is_action = any(
                    part in self.model.network._action_modules for part in name.split(".")
                )
                groups[int(is_action)]["params"].append(value)
        return [group for group in groups if group["params"]]

    def _inputs(self, batch, *, sampling=False):
        result = {
            key: batch[key]
            for key in (
                "input_ids",
                "attention_mask",
                "mm_token_type_ids",
                "pixel_values_videos_gen",
                "video_grid_thw_gen",
                "second_per_grid_ts_gen",
            )
        }
        if sampling:
            # A preprocessed training clip may be supplied; inference sees only its current frame.
            result["pixel_values_videos_gen"] = result["pixel_values_videos_gen"][:, :, :1]
            result.update(
                {
                    key: batch[key]
                    for key in (
                        "negative_input_ids",
                        "negative_attention_mask",
                        "negative_mm_token_type_ids",
                    )
                }
            )
        result["state"] = (
            F.pad(
                batch["observation.state"],
                (0, self.config.max_action_dim - self.config.original_action_dim),
            )
            if self.config.use_state
            else None
        )
        return result

    def forward(self, batch):
        valid = torch.ones(len(batch["action"]), device=batch["action"].device, dtype=torch.bool)
        for key, value in batch.items():
            if key.endswith("_is_pad"):
                valid &= ~value.flatten(1).any(1)
        if not valid.all():
            raise ValueError("Cosmos3 training requires complete windows without episode padding")
        actions = F.pad(
            batch["action"], (0, self.config.max_action_dim - self.config.original_action_dim)
        )
        loss, terms = self.model(**self._inputs(batch), actions=actions)
        return loss * batch.get("_loss_weight", 1.0), terms

    @torch.no_grad()
    def predict_action_chunk(self, batch, *, noise=None):
        if noise is not None and not isinstance(noise, FlowState):
            raise ValueError("Cosmos3 fixed noise needs FlowState(video_latents, actions)")
        actions, video = self.model.generate(**self._inputs(batch, sampling=True), noise=noise)
        if self.config.save_predicted_video:
            self.last_predicted_latents = video.cpu()
        return actions[..., : self.config.original_action_dim]

    def save_assets(self, path):
        self.tokenizer.save_pretrained(path / "tokenizer")

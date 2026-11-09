"""Official XR1 model, weighted loss terms and platform chunk interface."""

import torch

from alohamini.policies.chunk import ChunkPolicy


class XiaomiPolicy(ChunkPolicy):
    name = "xiaomi_robotics1"

    def __init__(self, config):
        super().__init__()
        from ._vendor.model import xr1

        self.config = config
        self.network = xr1(
            vlm_config=config.vlm_config,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            chunk_size=config.chunk_size,
            dtype=getattr(torch, config.dtype),
            attention_implementation=config.attention_implementation,
            freq_coefficient=config.freq_coefficient,
            ffn_gradient_checkpointing=config.gradient_checkpointing,
            async_train=config.async_train,
        )
        self.network.training_repeat = config.training_repeat
        self.network.num_steps = config.num_steps
        self.reset()

    def forward(self, batch):
        payload = {
            k: v for k, v in batch.items() if k != "inference_inputs" and not k.startswith("_")
        }
        result = self.network(payload, return_loss=True)
        loss = (
            0.5 * result["loss_mse"] * batch.get("_loss_weight", 1)
            + self.config.freq_coefficient * result["loss_freq"] * batch.get("_freq_weight", 1)
            + 0.5 * (result["loss_l1"] + result["loss_score"]) * batch.get("_aux_loss_weight", 1)
        )
        return loss, {k: v.detach() for k, v in result.items() if k != "loss"}

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        payload = dict(batch["inference_inputs"])
        mask = torch.zeros_like(batch["action_mask"])
        # A target tail mask is training information and must not affect inference.
        from .semantic import XiaomiFields

        mask[..., XiaomiFields(self.config.kinematic_fields).action_slots] = True
        payload.update(
            state=batch["state"], action=torch.zeros_like(batch["action"]), action_mask=mask
        )
        return self.network(payload, return_loss=False)

    def save_assets(self, path):
        from transformers import AutoProcessor, Qwen3VLConfig

        AutoProcessor.from_pretrained(
            self.config.assets_path, local_files_only=True
        ).save_pretrained(path)
        Qwen3VLConfig(**self.config.vlm_config).save_pretrained(path)

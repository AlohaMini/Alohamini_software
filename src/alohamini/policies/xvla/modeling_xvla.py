"""X-VLA core with native parameter groups and checkpoint asset ownership."""

import torch
from torch.nn import functional as F

from alohamini.policies.chunk import ChunkPolicy

from .core import XVLAModel


class XVLAPolicy(ChunkPolicy):
    name = "xvla"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = XVLAModel(
            config, config.get_florence_config(), config.max_state_dim if config.use_proprio else 0
        )
        self.model.vlm.language_model.config.use_cache = False
        self.model.transformer.gradient_checkpointing = config.gradient_checkpointing
        if config.gradient_checkpointing:
            self.model.vlm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.reset()

    def get_optim_params(self):
        groups = {name: [] for name in ("vlm", "transformer_core", "soft_prompts", "action_heads")}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("model.vlm."):
                group = "vlm"
            elif name.startswith("model.transformer.soft_prompt_hub."):
                group = "soft_prompts"
            elif name.startswith(
                ("model.transformer.action_encoder.", "model.transformer.action_decoder.")
            ):
                group = "action_heads"
            else:
                group = "transformer_core"
            groups[group].append(parameter)
        if not any(groups.values()):
            raise ValueError("X-VLA has no trainable parameters")
        return [
            dict(
                name=name,
                params=parameters,
                lr_scale=self.config.learning_coef if name in ("vlm", "soft_prompts") else 1.0,
            )
            for name, parameters in groups.items()
        ]

    def inputs(self, batch):
        state = batch.get("observation.state")
        if self.config.use_proprio:
            state = F.pad(state, (0, self.config.max_state_dim - state.shape[-1]))
        else:
            state = batch["image_input"].new_zeros((len(batch["input_ids"]), 0))
        return dict(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
            domain_id=batch["domain_id"],
            proprio=state,
        )

    def forward(self, batch):
        actions = F.pad(
            batch["action"], (0, self.config.max_action_dim - batch["action"].shape[-1])
        )
        terms = self.model(**self.inputs(batch), action=actions)
        return sum(terms.values()) * batch.get("_loss_weight", 1.0), {
            key: value.detach() for key, value in terms.items()
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        return self.model.generate_actions(
            **self.inputs(batch), steps=self.config.num_denoising_steps
        ).float()

    def save_assets(self, path):
        from transformers import AutoImageProcessor, AutoTokenizer

        AutoTokenizer.from_pretrained(
            self.config.assets_path, local_files_only=True
        ).save_pretrained(path)
        AutoImageProcessor.from_pretrained(
            self.config.assets_path, local_files_only=True
        ).save_pretrained(path)

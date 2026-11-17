# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for native Alohamini integration in 2026.
"""Native G0.5 with platform optimizer groups and independently reduced objectives."""

from copy import deepcopy
from pathlib import Path
from shutil import copy2

import torch
from transformers.modeling_utils import local_torch_dtype

from alohamini.policies.chunk import ChunkPolicy

from .core import G05NativeBackend
from .tokenizer_g05 import G05_RUNTIME_PREDICT_COT


class G05Policy(ChunkPolicy):
    name = "g05"

    def __init__(self, config):
        super().__init__()
        self.config = config
        core = deepcopy(config.author_model_config)
        core.update(
            continuous_action=config.train_flow_actions,
            discrete_action=config.train_discrete_actions,
            predict_cot=config.predict_cot,
            embodiment=config.embodiment,
            checkpoint_vlm=config.gradient_checkpointing,
            checkpoint_action_expert=config.gradient_checkpointing,
        )
        with local_torch_dtype(getattr(torch, config.dtype)):
            self.backend = G05NativeBackend.from_config(core, config.assets_path)
        self.backend.apply_fp32_params()
        vision = self.backend.model.vision_tower
        if config.gradient_checkpointing:
            vision.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if config.freeze_vision or config.freeze_vlm:
            vision.requires_grad_(False)
        if config.freeze_vlm:
            self.backend.model.vlm.requires_grad_(False)
            self.backend.model.proprio_embedder.requires_grad_(False)
        if config.train_discrete_actions and self.backend.action_tokenizer is None:
            raise FileNotFoundError("G0.5 discrete training requires action_tokenizer.safetensors")
        self.last_cot = None
        self.reset()

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_vision or self.config.freeze_vlm:
            self.backend.model.vision_tower.eval()
        if self.config.freeze_vlm:
            self.backend.model.vlm.eval()
            self.backend.model.proprio_embedder.eval()
        return self

    def get_optim_params(self):
        cfg = self.config
        groups = self.backend.get_optim_param_groups(
            lr=1.0,
            weight_decay=cfg.optimizer_weight_decay,
            apply_decay_on_norm_and_bias=cfg.optimizer_apply_decay_on_norm_and_bias,
            backbone_lr_multiplier=cfg.optimizer_backbone_lr_multiplier,
            vision_lr_multiplier=cfg.optimizer_vision_lr_multiplier,
        )
        for group in groups:
            group["lr_scale"] = group.pop("lr")
        return [group for group in groups if group["params"]]

    def forward(self, batch):
        if self.backend.action_tokenizer is not None:
            self.backend.action_tokenizer.to(next(self.parameters()).device)
        _, terms = self.backend(batch)
        loss = sum(
            value * batch.get("_flow_weight" if name == "fm_loss" else "_text_weight", 1.0)
            for name, value in terms.items()
        )
        return loss, {name: value.detach() for name, value in terms.items()}

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        self.eval()
        if self.backend.action_tokenizer is not None:
            self.backend.action_tokenizer.to(next(self.parameters()).device)
        # Validation may supply already tokenized targets. Prefix-only encoding
        # prevents both text targets and action-code targets leaking into inference.
        sequence = self.backend.processor.encode_inference(
            batch["samples"], device=next(self.parameters()).device
        )
        from .tokenizer_g05 import G05_INPUT_IDS, G05_LABELS, G05_SPLIT_INDEX, G05_TOKEN_TYPES

        inputs = {
            **batch,
            "g05.action_head": self.config.action_head,
            G05_INPUT_IDS: sequence.input_ids,
            G05_LABELS: sequence.labels,
            G05_TOKEN_TYPES: sequence.token_types,
            G05_SPLIT_INDEX: sequence.split_index,
        }
        # CoT availability is a per-sample/runtime decision made by the processor.
        inputs[G05_RUNTIME_PREDICT_COT] = batch.get(G05_RUNTIME_PREDICT_COT, False)
        result = self.backend.predict_action(inputs)
        self.last_cot = result.get("cot_text")
        return result["ar_action" if self.config.action_head == "actioncodec" else "action"]

    def save_assets(self, path):
        path = Path(path)
        self.backend.processor.tokenizer.save_pretrained(path / "hf_processor")
        if self.config.train_discrete_actions:
            copy2(self.backend.action_tokenizer_path, path / "action_tokenizer.safetensors")
        for name in ("LICENSE-G0.5", "NOTICE"):
            source = Path(self.config.assets_path) / name
            if not source.is_file():
                source = Path(__file__).parent / name
            if source.is_file():
                copy2(source, path / name)

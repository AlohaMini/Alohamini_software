# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for native Alohamini integration in 2026.
"""Platform normalization and named slots into the author G0.5 sample/token format."""

import torch
from torchvision.transforms.functional import resize

from alohamini.datasets.recipe import MESSAGES_RENDERED
from alohamini.learning.processor import Processor
from alohamini.policies.slots import NamedSlots

from .samples import prepare_g05_policy_batch
from .tokenizer_g05 import (
    G05_INPUT_IDS,
    G05_LABELS,
    G05_SPLIT_INDEX,
    G05_TOKEN_TYPES,
    G05Tokenizer,
)


def prepare_samples(config, numeric, *, metadata_only=False):
    result = dict(numeric)
    action_slots = NamedSlots(config.action_names, config.action_slots, config.policy_action_dim)
    state_slots = NamedSlots(config.state_names, config.state_slots, config.policy_state_dim)
    result["observation.state"] = state_slots.pack(result["observation.state"])
    if result["observation.state"].ndim == 2:
        result["observation.state"] = result["observation.state"][:, None]
    if "action" in result:
        result["action"] = action_slots.pack(result["action"])
    size = len(result["observation.state"])
    result["action_dim_is_pad"] = ~action_slots.mask(
        size, device=result["observation.state"].device
    )
    if metadata_only:
        for key in config.camera_order:
            result[key] = torch.empty(size, config.n_obs_steps, 3, 1, 1)
    run_cot = config.predict_cot and (
        config.runtime_system == "system2" or MESSAGES_RENDERED in result
    )
    return prepare_g05_policy_batch(config, result, predict_cot=run_cot)


def render_recipe(config, batch):
    if not config.use_language_recipe or "action" not in batch:
        return batch
    from alohamini.datasets.language_render import render_training
    from alohamini.datasets.recipe import TrainingRecipe

    rendered = render_training(TrainingRecipe.from_dict(config.recipe), batch)
    return {**batch, **{key: [row[key] for row in rendered] for key in rendered[0]}}


class G05Processor:
    def __init__(self, model, stats, device):
        self.config = model.config
        self.numeric = Processor.from_config(model.config, stats, device)
        self.tokenizer = model.backend.processor
        self.action_codec = model.backend.action_tokenizer
        self.decode_before_queue = self.numeric.decode_before_queue

    def __call__(self, batch):
        cfg = self.config
        numeric = self.numeric({k: v for k, v in batch.items() if k not in cfg.camera_order})
        for key in cfg.camera_order:
            value = batch[key].to(self.numeric.device)
            if value.dtype != torch.uint8 and cfg.quantize_rgb_before_resize:
                value = (value * 255).round().clamp(0, 255).to(torch.uint8)
            original = value.shape
            flat = value.reshape(-1, *original[-3:])
            target = tuple(cfg.camera_sizes[key])
            if tuple(flat.shape[-2:]) != target:
                flat = resize(flat, list(target))
            flat = flat.float() / 255 if flat.dtype == torch.uint8 else flat.float()
            mean, std = (
                flat.new_tensor(v).view(1, 3, 1, 1) for v in (cfg.image_mean, cfg.image_std)
            )
            value = ((flat - mean) / std).reshape(*original[:-3], 3, *target)
            if value.ndim == 4:
                value = value[:, None]
            numeric[key] = value
        numeric = render_recipe(cfg, numeric)
        prepared = prepare_samples(cfg, numeric)
        tokenizer = self.tokenizer
        if "action" in batch:
            if self.action_codec is not None:
                self.action_codec.to(self.numeric.device)
            sequence = tokenizer.encode_train(
                prepared["samples"],
                device=self.numeric.device,
                action_codec=self.action_codec,
                proprio_dropout_p=cfg.proprio_dropout_p if torch.is_grad_enabled() else 0.0,
            )
        else:
            sequence = tokenizer.encode_inference(prepared["samples"], device=self.numeric.device)
        prepared.update(
            {
                G05_INPUT_IDS: sequence.input_ids,
                G05_LABELS: sequence.labels,
                G05_TOKEN_TYPES: sequence.token_types,
                G05_SPLIT_INDEX: sequence.split_index,
            }
        )
        return prepared

    def action(self, tensor, *, context=None):
        slots = NamedSlots(
            self.config.action_names, self.config.action_slots, self.config.policy_action_dim
        )
        return self.numeric.action(slots.unpack(tensor))


class G05LossCounter:
    """Count text tokens and weighted action coordinates without image decoding."""

    def __init__(self, config, stats):
        self.config = config
        self.numeric = Processor.from_config(config, stats, "cpu")
        self.tokenizer = None
        self.codec = None

    def __call__(self, batch):
        cfg = self.config
        length = len(batch["action"])
        physical = len(cfg.action_names)
        padded = cfg.policy_action_dim - physical
        invalid = batch.get("action_is_pad", torch.zeros(length, cfg.chunk_size, dtype=torch.bool))
        padding_weight = cfg.author_model_config["fm"]["padding_action_weight"]
        flow = (
            (~invalid).sum() * physical
            + invalid.sum() * physical * padding_weight
            + length * cfg.chunk_size * padded * padding_weight
        ) * cfg.num_flow_samples
        text = 0
        if cfg.text_loss_weight and (cfg.train_discrete_actions or cfg.predict_cot):
            if self.tokenizer is None:
                from pathlib import Path

                from .action_codec_g05 import G05NativeActionCodec

                self.tokenizer = G05Tokenizer(
                    Path(cfg.assets_path) / "hf_processor", cfg.author_model_config
                )
                if cfg.train_discrete_actions:
                    self.codec = G05NativeActionCodec.load(
                        cfg.author_model_config["AT_CONFIG"],
                        action_token_begin=self.tokenizer.action_token_begin,
                        ckpt_path=Path(cfg.assets_path) / "action_tokenizer.safetensors",
                    )
            prepared = prepare_samples(
                cfg, render_recipe(cfg, self.numeric(batch)), metadata_only=True
            )
            sequence = self.tokenizer.encode_train(
                prepared["samples"],
                device=torch.device("cpu"),
                action_codec=self.codec,
                proprio_dropout_p=0.0,
            )
            text = int((sequence.labels[:, 1:] != -100).sum())
        return {
            "_flow_weight": float(flow) if cfg.train_flow_actions and cfg.flow_loss_weight else 0.0,
            "_text_weight": text,
        }

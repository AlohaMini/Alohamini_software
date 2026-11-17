# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Native FineART joint text/FAST/flow math, adapted from the official implementation.

One VLM prefix serves independent block-masked flow draws. Flow never sees FAST
targets, and KI detaches only the VLM K/V read by the action expert.
"""

import math

import torch
from torch.nn import functional as F

from alohamini.policies.chunk import ChunkPolicy
from alohamini.policies.pi05.configuration_pi05 import PI05Config
from alohamini.policies.pi05.modeling_pi05 import PI0Pytorch, make_att_2d_masks


def action_attention(prefix_valid, prefix_marks, non_fast_length, horizon, repeats):
    """Independent flow draws share a causal/bidirectional prefix and RoPE origin."""
    batch, length = prefix_valid.shape
    device = prefix_valid.device
    mask = torch.zeros(
        batch,
        length + repeats * horizon,
        length + repeats * horizon,
        dtype=torch.bool,
        device=device,
    )
    mask[:, :length, :length] = make_att_2d_masks(prefix_valid, prefix_marks)
    mask[:, length:, :non_fast_length] = prefix_valid[:, None, :non_fast_length]
    mask[:, length:, length:] = torch.block_diag(
        *[torch.ones(horizon, horizon, dtype=torch.bool, device=device) for _ in range(repeats)]
    )
    offset = prefix_valid[:, :non_fast_length].long().sum(-1, keepdim=True)
    positions = torch.cat(
        (
            prefix_valid.long().cumsum(-1) - 1,
            offset + torch.arange(horizon, device=device).repeat(repeats),
        ),
        dim=1,
    )
    return mask, positions


class FineARTNetwork(PI0Pytorch):
    def objective_outputs(self, batch, cfg, *, noise=None, time=None):
        obs = batch["observation"]
        images, masks, tokens, valid, state = self._preprocess_observation(obs, train=False)
        prefix, prefix_valid, prefix_marks = self.embed_prefix(images, masks, tokens, valid)
        labels = batch["text_labels"]
        prefix_marks = prefix_marks.clone()
        prefix_marks[:, -labels.shape[1] :] |= labels != -100
        non_fast_length = prefix.shape[1]
        if "fast_tokens" in batch:
            codes = self.paligemma_with_expert.embed_language_tokens(batch["fast_tokens"])
            prefix = torch.cat((prefix, codes * math.sqrt(codes.shape[-1])), dim=1)
            prefix_valid = torch.cat((prefix_valid, batch["fast_mask"]), dim=1)
            prefix_marks = torch.cat((prefix_marks, torch.ones_like(batch["fast_mask"])), dim=1)
        if cfg.flow_loss_weight > 0 and bool(batch["predict_actions"].any()):
            actions = batch["action"]
            b, h, d = actions.shape
            k = cfg.flow_num_repeats
            repeated = actions.repeat(k, 1, 1)
            if noise is None:
                noise = self.sample_noise(repeated.shape, repeated.device)
            if time is None:
                time = self.sample_time(k * b, repeated.device)
            latent = time[:, None, None] * noise + (1 - time[:, None, None]) * repeated
            suffix, _, _, condition = self.embed_suffix(state.repeat(k, 1), latent, time)
            suffix = suffix.view(k, b, h, -1).transpose(0, 1).reshape(b, k * h, -1)
            condition = condition.view(k, b, 1, -1).expand(k, b, h, -1)
            condition = condition.transpose(0, 1).reshape(b, k * h, -1)
            mask, positions = action_attention(prefix_valid, prefix_marks, non_fast_length, h, k)
            dtype = self.paligemma_with_expert.paligemma.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            (hidden, suffix_out), _ = self.paligemma_with_expert(
                inputs_embeds=[prefix.to(dtype), suffix.to(dtype)],
                attention_mask=self._prepare_attention_masks_4d(mask),
                position_ids=positions,
                adarms_cond=[None, condition],
                use_cache=False,
                detach_vlm=cfg.knowledge_insulation,
                fp32_attention=cfg.fp32_joint_attention,
            )
            predicted = self.action_out_proj(suffix_out.float()).view(b, k, h, d)
            target = (noise - repeated).view(k, b, h, d).transpose(0, 1)
            physical = cfg.action_feature.shape[0]
            flow = (predicted - target).square()[..., :physical].mean((1, 2, 3))
        else:
            # Text-only batches do not run the action expert or draw flow noise.
            (hidden, _), _ = self.paligemma_with_expert(
                inputs_embeds=[prefix, None],
                attention_mask=self._prepare_attention_masks_4d(
                    make_att_2d_masks(prefix_valid, prefix_marks)
                ),
                position_ids=prefix_valid.long().cumsum(-1) - 1,
                use_cache=False,
            )
            flow = hidden.sum((1, 2)) * 0
        language_hidden = hidden[:, non_fast_length - labels.shape[1] : non_fast_length]
        text = self.language_loss(
            language_hidden[:, :-1], labels[:, 1:], cfg.ce_chunk_size, cfg.text_ce_z_loss_weight
        )
        fast = flow * 0
        if "fast_tokens" in batch:
            fast_hidden = hidden[:, non_fast_length:]
            fast_labels = batch["fast_tokens"].masked_fill(~batch["fast_code_mask"], -100)
            fast_labels = fast_labels.masked_fill(~batch["predict_actions"][:, None], -100)
            fast = self.language_loss(
                fast_hidden[:, :-1], fast_labels[:, 1:], cfg.ce_chunk_size, per_sample=True
            )
        return flow, text, fast

    def language_loss(self, hidden, labels, chunk_size, z_weight=0.0, *, per_sample=False):
        valid = labels != -100
        selected, targets = hidden[valid], labels[valid]
        head = self.paligemma_with_expert.paligemma.lm_head
        rows = torch.arange(len(labels), device=labels.device)[:, None].expand_as(labels)[valid]

        def ce(features, targets):
            logits = head(features.to(head.weight.dtype)).float()
            values = F.cross_entropy(logits, targets, reduction="none")
            return values + z_weight * logits.logsumexp(-1).square()

        losses = [
            self._apply_checkpoint(ce, selected[i : i + chunk_size], targets[i : i + chunk_size])
            for i in range(0, len(targets), chunk_size)
        ]
        if not losses:
            return hidden.sum() * 0 if not per_sample else hidden.sum((1, 2)) * 0
        values = torch.cat(losses)
        if not per_sample:
            return values.mean()
        sums = torch.zeros(len(labels), device=hidden.device).index_add(0, rows, values)
        return sums / valid.sum(-1).clamp_min(1)


class FineARTVLAPolicy(ChunkPolicy):
    name = "fineart_vla"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.network = FineARTNetwork(PI05Config(**config.network))
        # The action expert is a feature model; its unused vocabulary projection is frozen.
        self.network.paligemma_with_expert.gemma_expert.lm_head.requires_grad_(False)
        if config.freeze_vision_encoder:
            vlm = self.network.paligemma_with_expert.paligemma
            vlm.vision_tower.requires_grad_(False)
            vlm.multi_modal_projector.requires_grad_(False)
        if config.gradient_checkpointing:
            self.network.gradient_checkpointing_enable()
        self.reset()

    def forward(self, batch):
        flow_values, text, fast_values = self.network.objective_outputs(batch, self.config)
        active = batch["predict_actions"].to(flow_values.dtype)
        flow = (flow_values * active).sum() / active.sum().clamp_min(1)
        fast = (fast_values * active).sum() / active.sum().clamp_min(1)
        cfg = self.config
        loss = (
            cfg.flow_loss_weight * flow * batch.get("_flow_weight", 1.0)
            + cfg.text_loss_weight * text * batch.get("_text_weight", 1.0)
            + cfg.fast_action_loss_weight * fast * batch.get("_fast_weight", 1.0)
        )
        return loss, {
            "flow_loss": flow.detach(),
            "text_loss": text.detach(),
            "fast_loss": fast.detach(),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        if not batch["predict_actions"].all() or (batch["text_labels"] != -100).any():
            raise ValueError(
                "FineART action inference requires explicit low-level prompts without targets"
            )
        return self.network.sample_actions(
            next(self.parameters()).device,
            batch["observation"],
            num_steps=self.config.num_inference_steps,
            **kwargs,
        )

    @torch.no_grad()
    def generate_text_tokens(self, batch, *, eos_id, blocked_range=None):
        """Explicit high-level greedy query; the caller owns task/subtask execution state."""
        if batch["predict_actions"].any() or (batch["text_labels"] != -100).any():
            raise ValueError("FineART text generation requires an explicit unlabeled text_query")
        if blocked_range is None:
            blocked_range = batch.get("text_blocked_range")
        obs = batch["observation"]
        images, masks, tokens, valid, _ = self.network._preprocess_observation(obs, train=False)
        prefix, valid, marks = self.network.embed_prefix(images, masks, tokens, valid)
        mask = make_att_2d_masks(valid, marks)
        backbone = self.network.paligemma_with_expert
        (hidden, _), cache = backbone(
            inputs_embeds=[prefix, None],
            attention_mask=self.network._prepare_attention_masks_4d(mask),
            position_ids=valid.long().cumsum(-1) - 1,
            use_cache=True,
        )
        last = (
            torch.arange(valid.shape[1], device=valid.device)
            .expand_as(valid)
            .masked_fill(~valid, -1)
            .max(-1)
            .values
        )
        hidden = hidden[torch.arange(len(valid), device=valid.device), last]
        results = [[] for _ in range(len(valid))]
        done = torch.zeros(len(valid), dtype=torch.bool, device=valid.device)
        for _ in range(self.config.max_decoding_steps):
            logits = backbone.paligemma.lm_head(hidden).float()
            if blocked_range:
                logits[:, blocked_range[0] : blocked_range[1]] = -torch.inf
            token = logits.argmax(-1)
            for i, value in enumerate(token.tolist()):
                if not done[i] and value != eos_id:
                    results[i].append(value)
            done |= token == eos_id
            if done.all():
                return results
            positions = valid.long().sum(-1, keepdim=True)
            valid = torch.cat((valid, torch.ones_like(done[:, None])), dim=1)
            embedded = backbone.embed_language_tokens(token[:, None])
            (step, _), cache = backbone(
                inputs_embeds=[embedded * math.sqrt(embedded.shape[-1]), None],
                attention_mask=self.network._prepare_attention_masks_4d(valid[:, None, :]),
                position_ids=positions,
                past_key_values=cache,
                use_cache=True,
            )
            hidden = step[:, -1]
        raise ValueError("FineART text query exceeded max_decoding_steps without EOS")

    def save_assets(self, path):
        import shutil
        from pathlib import Path

        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(
            self.config.assets_path, local_files_only=True
        ).save_pretrained(path)
        if self.config.fast_tokenizer_path:
            from alohamini.policies.pi05.fast_tokenizer import fast_asset_files

            target = Path(path) / "fast"
            target.mkdir()
            for name, source in fast_asset_files(self.config.fast_tokenizer_path).items():
                shutil.copyfile(source, target / name)
            if self.config.fast_fit_report:
                import json

                (target / "fit_report.json").write_text(
                    json.dumps(self.config.fast_fit_report, indent=2) + "\n"
                )

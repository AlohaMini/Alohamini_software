"""Native PaliGemma action-token CE and cached autoregressive decoding.

OpenPI models/pi0_fast.py and models/gemma_fast.py at SOURCE.json.
Uses the existing native SigLIP/Gemma implementation; no flow action expert.
"""

import math

import torch
from torch.utils.checkpoint import checkpoint

from alohamini.policies.chunk import ChunkPolicy
from alohamini.policies.pi05.configuration_pi05 import get_config
from alohamini.policies.pi05.gemma_pytorch import PaliGemmaWithExpertModel
from alohamini.policies.pi05.knowledge_insulation import attention_masks, token_cross_entropy


class PI0FastPolicy(ChunkPolicy):
    name = "pi0_fast"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.network = PaliGemmaWithExpertModel(
            get_config(config.paligemma_variant), None, precision=config.dtype
        )
        vlm = self.network.paligemma
        if config.freeze_vision_encoder:
            vlm.vision_tower.requires_grad_(False)
            vlm.multi_modal_projector.requires_grad_(False)
        if config.gradient_checkpointing:
            for module in (vlm.language_model, vlm.vision_tower):
                module.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        self.reset()

    def reset(self):
        super().reset()
        self.action_context = None

    def _checkpoint(self, function, *args):
        if self.training and self.config.gradient_checkpointing:
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def _embed_tokens(self, tokens):
        values = self.network.embed_language_tokens(tokens)
        return values * math.sqrt(values.shape[-1])

    def _prefix(self, batch):
        observation = batch["observation"]
        embeddings, masks = [], []
        for key in self.config.camera_order:
            image = self.network.embed_image(observation.images[key])
            embeddings.append(image)
            masks.append(observation.image_masks[key][:, None].expand(image.shape[:2]))
        embeddings.append(self._embed_tokens(observation.tokenized_prompt))
        masks.append(observation.tokenized_prompt_mask)
        return torch.cat(embeddings, dim=1), torch.cat(masks, dim=1)

    def _transform(self, embeddings, mask, positions, *, cache=None, use_cache=False):
        dtype = next(self.network.paligemma.language_model.parameters()).dtype
        additive = torch.where(mask[:, None], 0.0, torch.finfo(torch.float32).min)
        (hidden, _), cache = self.network(
            inputs_embeds=[embeddings.to(dtype), None],
            attention_mask=additive,
            position_ids=positions,
            past_key_values=cache,
            use_cache=use_cache,
        )
        return hidden, cache

    def forward(self, batch):
        prefix, context_mask = self._prefix(batch)
        tokens, targets_mask = batch["fast_tokens"], batch["fast_mask"]
        empty = torch.empty((len(tokens), 0), dtype=torch.bool, device=tokens.device)
        mask, positions = attention_masks(context_mask, targets_mask, empty)
        hidden, _ = self._transform(
            torch.cat((prefix, self._embed_tokens(tokens)), dim=1), mask, positions
        )
        loss = token_cross_entropy(
            hidden,
            context_mask,
            tokens,
            targets_mask,
            self.network.paligemma.lm_head,
            self._checkpoint,
            self.config.ce_chunk_size,
        ).mean()
        return loss * batch.get("_loss_weight", 1.0), {"fast_loss": loss.detach()}

    @torch.no_grad()
    def generate_tokens(self, batch, *, eos_id):
        """Greedy generation with the same prefix positions used in teacher forcing."""
        prefix, valid = self._prefix(batch)
        batch_size, length = valid.shape
        mask = valid[:, :, None] & valid[:, None, :]
        positions = valid.long().cumsum(-1) - 1
        hidden, cache = self._transform(prefix, mask, positions, use_cache=True)
        last = (
            torch.arange(length, device=valid.device)
            .expand_as(valid)
            .masked_fill(~valid, -1)
            .max(-1)
            .values
        )
        hidden = hidden[torch.arange(batch_size, device=valid.device), last]
        generated = [[] for _ in range(batch_size)]
        active = torch.ones(batch_size, dtype=torch.bool, device=valid.device)
        for index in range(self.config.max_decoding_steps):
            token = self.network.paligemma.lm_head(hidden).argmax(-1)
            for i in range(batch_size):
                if active[i]:
                    generated[i].append(token[i].item())
            active &= token.ne(eos_id)
            if not active.any():
                return generated
            valid = torch.cat((valid, active[:, None]), dim=1)
            hidden, cache = self._transform(
                self._embed_tokens(token[:, None]),
                valid[:, None, :],
                positions[:, -1:] + index + 1,
                cache=cache,
                use_cache=True,
            )
            hidden = hidden[:, -1]
        raise ValueError("FAST generation reached max_decoding_steps before EOS")

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        from .processor import FASTActionDecoder

        # Tokenizer assets are immutable; initialized once and excluded from model state.
        if not hasattr(self, "_decoder"):
            self._decoder = FASTActionDecoder(self.config)
        tokens = self.generate_tokens(batch, eos_id=self._decoder.sp.eos_id())
        return self._decoder.decode(tokens, device=next(self.parameters()).device)

    @torch.no_grad()
    def select_action(self, batch):
        if not self._actions:
            self.action_context = batch["_action_context"].detach().clone()
        return super().select_action(batch)

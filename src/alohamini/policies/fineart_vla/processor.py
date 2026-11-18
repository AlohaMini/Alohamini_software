"""Local tokenization and recipe routing; numeric transforms belong to the platform."""

from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from alohamini.datasets.recipe import TrainingRecipe
from alohamini.learning.processor import Processor
from alohamini.policies.pi05.fast_tokenizer import FASTTokenizer, pad_targets
from alohamini.policies.pi05.image_tools import resize_with_pad_torch

from .routing import render_training
from .spatial import spatial_messages


class _CodecVocabulary:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def vocab_size(self):
        return self.tokenizer.vocab_size


class FineARTTextEncoder:
    def __init__(self, config, stats):
        self.config = config
        self.recipe = TrainingRecipe.from_dict(config.recipe)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.assets_path,
            local_files_only=True,
            use_fast=True,
        )
        if not self.tokenizer.is_fast or self.tokenizer.eos_token is None:
            raise ValueError("FineART requires a local fast tokenizer with offsets and EOS")
        self.tokenizer.add_tokens([f"<loc{i:04d}>" for i in range(1024)])
        self.numeric = Processor.from_config(config, stats, "cpu")

    def __call__(self, batch, *, text_query=False):
        query = batch.get("_language_query")
        if query is not None:
            text_query = True
            if (
                "action" in batch
                or not isinstance(query, list)
                or len(query) != len(batch["observation.state"])
            ):
                raise ValueError("Language queries require unlabeled messages for each observation")
            rendered = [
                {
                    "messages_rendered": messages,
                    "message_streams": ["high_level"] * len(messages),
                    "target_message_indices": [],
                }
                for messages in query
            ]
        elif "action" in batch:
            rendered = render_training(self.recipe, batch)
        else:
            rendered = [
                {
                    "messages_rendered": [{"role": "user", "content": task}],
                    "message_streams": ["high_level" if text_query else "low_level"],
                    "target_message_indices": [],
                }
                for task in batch["task"]
            ]
        states = self.numeric({"observation.state": batch["observation.state"]})[
            "observation.state"
        ].numpy()
        prompts, spans, actions = [], [], []
        for item, state in zip(rendered, states, strict=True):
            messages = spatial_messages(
                item,
                cameras=self.config.image_features,
                fast_enabled=self.config.fast_action_loss_weight > 0,
                fast_skip_tokens=self.config.fast_skip_tokens,
            )
            predict_action = "low_level" in item["message_streams"]
            if predict_action:
                bins = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
                state_text = " ".join(str(int(value)) for value in bins)
                for message in reversed(messages):
                    if message["role"] == "user":
                        message["content"] = f"{message['content']}, State: {state_text};"
                        break
            targets = set(item["target_message_indices"])
            parts, bounds, cursor = [], [], 0
            for i, message in enumerate(messages):
                content = message.get("content")
                if not isinstance(content, str) or message.get("tool_calls"):
                    raise ValueError("FineART native text recipes require plain text turns")
                header = f"{message['role'].capitalize()}: "
                body = content + (self.tokenizer.eos_token if i in targets else "")
                if i in targets:
                    bounds.append((cursor + len(header), cursor + len(header) + len(body)))
                parts.append(header + body + "\n")
                cursor += len(parts[-1])
            if text_query:
                parts.append("Assistant:")
            prompts.append("".join(parts))
            spans.append(bounds)
            actions.append(predict_action)
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding_side="right",
        )
        if encoded.input_ids.shape[1] > self.config.network["max_token_len"]:
            raise ValueError("FineART prompt exceeds max_token_len; increase it without truncation")
        labels = torch.full_like(encoded.input_ids, -100)
        for i, bounds in enumerate(spans):
            offsets = encoded.offset_mapping[i]
            for start, end in bounds:
                selected = (offsets[:, 1] > start) & (offsets[:, 0] < end)
                selected &= encoded.attention_mask[i].bool()
                labels[i, selected] = encoded.input_ids[i][selected]
        return {
            "tokens": encoded.input_ids,
            "mask": encoded.attention_mask.bool(),
            "text_labels": labels,
            "predict_actions": torch.tensor(actions, dtype=torch.bool),
        }

    def counts(self, batch):
        encoded = self(batch)
        active = int(encoded["predict_actions"].sum())
        return {
            "_flow_weight": active if self.config.flow_loss_weight else 0,
            "_text_weight": int((encoded["text_labels"][:, 1:] != -100).sum())
            if self.config.text_loss_weight
            else 0,
            "_fast_weight": active if self.config.fast_action_loss_weight else 0,
        }


class FineARTProcessor:
    def __init__(self, config, stats, device):
        self.config = config
        self.text = FineARTTextEncoder(config, stats)
        self.numeric = Processor.from_config(config, stats, device)
        self.codec = None
        if config.fast_action_loss_weight:
            self.codec = FASTTokenizer(
                config.fast_tokenizer_path,
                _CodecVocabulary(self.text.tokenizer),
                config.max_action_tokens,
                skip_tokens=config.fast_skip_tokens,
            )
            if config.fast_fit_report:
                from .assets import file_hashes

                if file_hashes(config.fast_tokenizer_path) != config.fast_fit_report.get("files"):
                    raise ValueError("FAST assets do not match the saved fitting report")

    def __call__(self, batch, *, text_query=False):
        cfg, device = self.config, self.numeric.device
        tokens = {
            key: value.to(device) for key, value in self.text(batch, text_query=text_query).items()
        }
        numeric = self.numeric(batch)
        images = {}
        for key in cfg.image_features:
            value = numeric[key]
            if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                raise ValueError(f"{key}: expected RGB in [0,1]")
            images[key] = resize_with_pad_torch(value * 2 - 1, 224, 224)
        state = numeric["observation.state"]
        result = {
            "observation": SimpleNamespace(
                images=images,
                image_masks={
                    key: torch.ones(len(state), dtype=torch.bool, device=device) for key in images
                },
                state=F.pad(state, (0, cfg.network["action_dim"] - state.shape[-1])),
                tokenized_prompt=tokens["tokens"],
                tokenized_prompt_mask=tokens["mask"],
                token_ar_mask=None,
                token_loss_mask=None,
            ),
            "text_labels": tokens["text_labels"],
            "predict_actions": tokens["predict_actions"],
            "_action_context": numeric.get("_action_context"),
        }
        if self.codec is not None:
            capacity = (
                max(self.codec.vocab_size, cfg.fast_fit_vocab_size)
                if cfg.auto_fit_fast_tokenizer
                else self.codec.vocab_size
            )
            result["text_blocked_range"] = (
                max(0, self.codec.offset - capacity + 1),
                self.codec.offset + 1,
            )
        if "action" in numeric:
            action = numeric["action"]
            result["action"] = F.pad(action, (0, cfg.network["action_dim"] - action.shape[-1]))
            if self.codec is not None:
                tokenizer = self.text.tokenizer
                prefix = [tokenizer.bos_token_id] + tokenizer.encode(
                    "Action: ", add_special_tokens=False
                )
                end = tokenizer.encode("|", add_special_tokens=False) + [tokenizer.eos_token_id]
                codes = self.codec.encode_actions(action.detach().cpu().numpy())
                sequences = [
                    prefix + [self.codec.offset - code for code in row] + end for row in codes
                ]
                fast, mask = pad_targets(sequences, cfg.max_action_tokens, device=device)
                code_mask = torch.zeros_like(mask)
                for i, row in enumerate(codes):
                    code_mask[i, len(prefix) : len(prefix) + len(row)] = True
                result.update(fast_tokens=fast, fast_mask=mask, fast_code_mask=code_mask)
        return result

    def action(self, tensor, *, context=None):
        return self.numeric.action(
            tensor[..., : self.config.action_feature.shape[0]], context=context
        )

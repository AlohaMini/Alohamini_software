"""Narrow adapters for upstream interfaces used by the vendored policy models."""

from dataclasses import dataclass
from inspect import signature

from transformers.masking_utils import create_causal_mask as _create_causal_mask

_LEGACY_MASK = "cache_position" in signature(_create_causal_mask).parameters


def dataclass_config(cls):
    """Keep HF's subclass dataclass intact, including default factories."""
    if "__dataclass_fields__" in vars(cls):
        return cls
    return dataclass(cls)


def create_causal_mask(
    *, config, input_embeds, attention_mask, cache_position, past_key_values, position_ids
):
    """Adapt the mask signature for decoders with contiguous cache positions."""
    kwargs = dict(
        config=config,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    if _LEGACY_MASK:
        return _create_causal_mask(
            input_embeds=input_embeds, cache_position=cache_position, **kwargs
        )
    return _create_causal_mask(inputs_embeds=input_embeds, **kwargs)


def crop_cache_to_length(cache, length):
    """Restore each layer to its prefix without trimming cross-attention K/V.

    Transformers 5.21 accepts only negative crop lengths. Self-attention layers
    append action tokens, while cross-attention layers retain only the prefix,
    so a single negative offset for the whole cache would corrupt the latter.
    """
    for layer in cache.layers:
        extra = layer.get_seq_length() - length
        if extra > 0:
            layer.crop(-extra)

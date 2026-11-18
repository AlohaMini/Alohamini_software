"""UMT5 feature preparation; generic storage and writer-rank ownership stay in the platform."""

import json
from importlib.metadata import version
from pathlib import Path

import torch

from alohamini.learning.frozen_features import (
    FrozenFeatureCache,
    prepare_features,
    source_signature,
)
from alohamini.paths import WorkspacePaths


def feature_key(task, max_length):
    from .core import clean_prompt

    return json.dumps([clean_prompt(task), max_length], ensure_ascii=False)


def encode_features(keys, tokenizer, encoder):
    tasks = []
    length = None
    for key in keys:
        task, n = json.loads(key)
        if length is not None and length != n:
            raise ValueError("A text feature batch must use one token length")
        tasks.append(task)
        length = n
    inputs = tokenizer(
        tasks,
        padding="max_length",
        max_length=length,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
    device = next(encoder.parameters()).device
    with torch.no_grad():
        output = encoder(
            input_ids=inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device)
        ).last_hidden_state
        output = output * inputs.attention_mask.to(device).unsqueeze(-1)
    return list(output.detach().cpu())


def prepare_text_features(options, samples, *, validation_samples=None):
    if not options.get("precompute_text_features", True):
        return options
    from .core import load_text_encoder, load_tokenizer

    path = Path(options["wan_pretrained_path"])
    length = options.get("max_sequence_length", 512)
    tasks = {row["task"] for row in samples.rows} | {""}
    if validation_samples is not None:
        # Frozen input encodings are not fitted statistics or supervised targets.
        tasks.update(row["task"] for row in validation_samples.rows)
    tasks = sorted(tasks)
    keys = [feature_key(task, length) for task in tasks]
    dtype = options.get("dtype", "bfloat16")
    if options.get("text_feature_store_path"):
        prepared = FrozenFeatureCache(options["text_feature_store_path"], max_bytes=0)
        signature = prepared.manifest["identity"]["signature"]
        if (
            set(keys) <= prepared.manifest["entries"].keys()
            and signature.get("dtype") == dtype
            and signature.get("protocol") == 2
        ):
            return options
    signature = dict(
        source=source_signature([path / "text_encoder", path / "tokenizer"]),
        dtype=dtype,
        transformers=version("transformers"),
        torch=version("torch"),
        tokenizers=version("tokenizers"),
        ftfy=version("ftfy"),
        protocol=2,
    )
    holder = {}

    def produce(batch):
        if not holder:
            holder["tokenizer"] = load_tokenizer(path, subfolder="tokenizer")
            holder["encoder"] = load_text_encoder(
                path, getattr(torch, dtype), "cpu", subfolder="text_encoder"
            )
            holder["encoder"].eval().requires_grad_(False)
        return encode_features(batch, holder["tokenizer"], holder["encoder"])

    root = (
        options.get("text_feature_cache_dir")
        or WorkspacePaths().pretrained / "frozen_features" / "umt5"
    )
    prepared = prepare_features(root, signature, keys, produce)
    result = dict(options, text_feature_store_path=str(prepared))
    return result

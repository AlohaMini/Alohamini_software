"""Train-only FAST fitting with deterministic sampling and atomic local cache publication."""

import fcntl
import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import scipy
import tokenizers

from alohamini.learning.processor import Processor
from alohamini.paths import WorkspacePaths
from alohamini.policies.pi05.fast_fitting import fit_fast, validate_fast
from alohamini.policies.pi05.fast_tokenizer import fast_asset_files

from .configuration_fineart_vla import FineARTVLAConfig


def file_hashes(path):
    return {
        name: hashlib.sha256(file.read_bytes()).hexdigest()
        for name, file in fast_asset_files(path).items()
    }


def prepare_fast_assets(options, samples, stats):
    result = deepcopy(options)
    cfg = FineARTVLAConfig(**result)
    if not cfg.auto_fit_fast_tokenizer or cfg.fast_action_loss_weight == 0:
        return result
    if len(samples) < 2:
        raise ValueError("Automatic FAST fitting needs at least two training chunks for validation")
    seed = np.random.default_rng(cfg.fast_fit_seed)
    selected = seed.permutation(len(samples))
    validation_count = min(cfg.fast_validation_samples, max(1, len(samples) // 5))
    fit_count = min(cfg.fast_fit_samples, len(samples) - validation_count)
    validation_indices = selected[:validation_count]
    fit_indices = selected[validation_count : validation_count + fit_count]
    transform = getattr(samples, "field_transform", None)
    protocol = dict(
        version=2,
        source_sha256=samples.table_sha256,
        row_selection_sha256=hashlib.sha256(
            np.asarray(samples.sample_indices, dtype="<i8").tobytes()
        ).hexdigest(),
        windows=samples.delta_indices,
        sample_filter=samples.sample_filter,
        action_feature=getattr(samples, "model_features", {}).get(
            "action", samples.info["features"]["action"]
        ),
        field_transforms=transform.configuration if transform is not None else None,
        action_statistics=stats["action"],
        normalization=cfg.quantile_options["action"],
        fit_indices=fit_indices.tolist(),
        validation_indices=validation_indices.tolist(),
        scale=cfg.fast_fit_scale,
        vocab_size=cfg.fast_fit_vocab_size,
        max_action_tokens=cfg.max_action_tokens,
        max_rmse=cfg.fast_max_reconstruction_rmse,
        max_dim_rmse=cfg.fast_max_dim_rmse,
        libraries=dict(
            numpy=np.__version__, scipy=scipy.__version__, tokenizers=tokenizers.__version__
        ),
    )
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    root = (
        Path(cfg.fast_tokenizer_cache_dir).expanduser()
        if cfg.fast_tokenizer_cache_dir
        else (WorkspacePaths().pretrained / "action_tokenizers/fast")
    )
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / fingerprint
    with (root / f"{fingerprint}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            report = json.loads((destination / "fit_report.json").read_text())
            if report.get("fingerprint") != fingerprint or report.get("files") != file_hashes(
                destination
            ):
                raise ValueError(
                    "Fitted FAST cache changed; preserve it and select a new cache directory"
                )
        else:
            numeric = Processor.from_config(cfg, stats, "cpu")

            def chunks(indices):
                return [
                    numeric({"action": samples.action_metadata(int(index))["action"]})[
                        "action"
                    ].numpy()
                    for index in indices
                ]

            training = chunks(fit_indices)
            held_out = chunks(validation_indices)
            # Account for BOS, Action: and |+EOS; the codec limit applies to complete chunks.
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(cfg.assets_path, local_files_only=True)
            overhead = (
                2
                + len(tokenizer.encode("Action: ", add_special_tokens=False))
                + len(tokenizer.encode("|", add_special_tokens=False))
            )
            budget = cfg.max_action_tokens - overhead
            if budget < 1:
                raise ValueError("max_action_tokens does not leave room for FAST codes")
            with tempfile.TemporaryDirectory(prefix=".fit-", dir=root) as temporary:
                fit_fast(
                    training,
                    temporary,
                    scale=cfg.fast_fit_scale,
                    vocab_size=cfg.fast_fit_vocab_size,
                )
                limits = dict(
                    max_rmse=cfg.fast_max_reconstruction_rmse,
                    max_dim_rmse=cfg.fast_max_dim_rmse,
                    max_tokens=budget,
                )
                training_report = validate_fast(temporary, training, **limits)
                validation_report = validate_fast(temporary, held_out, **limits)
                report = dict(
                    fingerprint=fingerprint,
                    protocol=protocol,
                    fit_chunks=fit_count,
                    training=training_report,
                    validation=validation_report,
                    files=file_hashes(temporary),
                )
                (Path(temporary) / "fit_report.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                os.replace(temporary, destination)
    result.update(fast_tokenizer_path=str(destination), fast_fit_report=report)
    return result

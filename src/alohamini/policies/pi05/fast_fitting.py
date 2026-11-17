"""Native fitting of the FAST DCT/BPE codec; never executes downloaded Python.

Algorithm: physical-intelligence/fast at ec4d7aa71691cac0b8bed6942be45684db2110f4.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.fft import dct
from tokenizers import ByteLevelBPETokenizer, pre_tokenizers
from tokenizers.trainers import BpeTrainer

from .fast_tokenizer import FASTTokenizer


def fit_fast(actions, path, *, scale=10.0, vocab_size=1024):
    """Fit coefficients and BPE on explicitly selected [H,D] action chunks."""
    if not np.isfinite(scale) or scale <= 0 or type(vocab_size) is not int or vocab_size < 1:
        raise ValueError("FAST fitting requires positive scale and integer vocab_size")
    if not actions:
        raise ValueError("FAST fitting requires action chunks")
    shape = np.shape(actions[0])
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("FAST fitting requires [time, action] chunks")
    coefficients = []
    for values in actions:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError("FAST fitting chunks must be finite and have the same shape")
        coefficients.append(np.around(dct(values, axis=0, norm="ortho").flatten() * scale))
    minimum = int(min(values.min() for values in coefficients))
    maximum = int(max(values.max() for values in coefficients))
    alphabet_size = maximum - minimum + 1
    if alphabet_size > vocab_size or alphabet_size > 0x110000:
        raise ValueError(
            f"FAST coefficient alphabet needs {alphabet_size} entries, vocabulary={vocab_size}; "
            "increase fast_fit_vocab_size or inspect the action normalization"
        )

    def strings():
        for values in coefficients:
            yield "".join(map(chr, (values - minimum).astype(np.int64)))

    bpe = ByteLevelBPETokenizer()
    # Coefficient characters are byte-escaped by ByteLevel. Registering raw
    # Unicode alone lets unseen held-out byte symbols disappear during encode.
    alphabet = sorted(
        set(map(chr, range(alphabet_size))) | set(pre_tokenizers.ByteLevel.alphabet())
    )
    if len(alphabet) > vocab_size:
        raise ValueError(f"FAST byte alphabet needs {len(alphabet)} entries; increase vocabulary")
    bpe._tokenizer.train_from_iterator(
        strings(),
        trainer=BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=False,
            special_tokens=[],
            initial_alphabet=alphabet,
            max_token_length=10000,
        ),
    )
    actual_size = bpe.get_vocab_size()
    if actual_size > vocab_size:
        raise ValueError("FAST byte alphabet exceeds the requested vocabulary")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    bpe._tokenizer.save(str(path / "tokenizer.json"))
    config = dict(
        scale=float(scale),
        min_token=minimum,
        vocab_size=actual_size,
        time_horizon=shape[0],
        action_dim=shape[1],
    )
    (path / "processor_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def validate_fast(path, actions, *, max_rmse=0.1, max_dim_rmse=0.2, max_tokens=256):
    """Held-out reconstruction and length checks reject truncation or dropped coefficients."""
    if (
        not np.isfinite([max_rmse, max_dim_rmse]).all()
        or min(max_rmse, max_dim_rmse) <= 0
        or type(max_tokens) is not int
        or max_tokens < 1
    ):
        raise ValueError("FAST validation requires positive finite error limits and token budget")
    if not actions:
        raise ValueError("FAST validation requires held-out training chunks")
    vocabulary = SimpleNamespace(vocab_size=lambda: 1_000_000)
    codec = FASTTokenizer(path, vocabulary)
    targets = np.asarray(actions, dtype=np.float32)
    if targets.ndim != 3 or not np.isfinite(targets).all():
        raise ValueError("FAST validation chunks must be finite [batch,H,D]")
    codes = codec.encode_actions(targets)
    if max(map(len, codes)) > max_tokens:
        raise ValueError("Fitted FAST codes exceed max_action_tokens; no action was truncated")
    reconstructed = codec.decode_actions(
        codes, horizon=targets.shape[1], action_dim=targets.shape[2]
    )
    error = np.square(reconstructed.astype(np.float64) - targets)
    dimension_rmse = np.sqrt(error.mean(axis=(0, 1)))
    rmse = float(np.sqrt(error.mean()))
    worst = float(dimension_rmse.max())
    if not np.isfinite(error).all() or rmse > max_rmse or worst > max_dim_rmse:
        raise ValueError(
            f"FAST reconstruction exceeds configured limits: rmse={rmse:.6g}, "
            f"max_dim_rmse={worst:.6g}; no training was started"
        )
    return dict(
        reconstruction_rmse=rmse,
        max_dim_rmse=worst,
        max_code_tokens=max(map(len, codes)),
        validation_chunks=len(actions),
    )

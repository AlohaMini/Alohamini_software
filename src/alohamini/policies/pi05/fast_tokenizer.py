"""Local FAST DCT/BPE encoding; no downloaded Python code is executed.

Algorithm and assets: physical-intelligence/fast at FAST_REVISION.
PaliGemma token mapping: OpenPI models/tokenizer.py (skip 128 special tokens).
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.fft import dct, idct
from tokenizers import Tokenizer

FAST_REPO = "physical-intelligence/fast"
FAST_REVISION = "ec4d7aa71691cac0b8bed6942be45684db2110f4"
FAST_FILES = ("tokenizer.json", "processor_config.json")


def fast_asset_files(path):
    """Resolve both released UniversalActionProcessor and flat native codec assets."""
    path = Path(path).expanduser()
    tokenizer = path / "tokenizer.json"
    if not tokenizer.is_file():
        tokenizer = path / "bpe_tokenizer/tokenizer.json"
    result = {"tokenizer.json": tokenizer, "processor_config.json": path / "processor_config.json"}
    if any(not file.is_file() for file in result.values()):
        raise FileNotFoundError(f"FAST tokenizer/configuration assets are missing: {path}")
    return result


class FASTTokenizer:
    def __init__(self, path, sentencepiece, max_tokens=256, *, skip_tokens=128):
        files = fast_asset_files(path)
        config = json.loads(files["processor_config.json"].read_text())
        self.bpe = Tokenizer.from_file(str(files["tokenizer.json"]))
        self.scale, self.min_token = float(config["scale"]), int(config["min_token"])
        self.vocab_size = int(config["vocab_size"])
        if (
            self.scale <= 0
            or not np.isfinite(self.scale)
            or self.vocab_size != self.bpe.get_vocab_size()
        ):
            raise ValueError("Invalid FAST tokenizer configuration")
        self.sp, self.max_tokens = sentencepiece, max_tokens
        if type(skip_tokens) is not int or skip_tokens < 0:
            raise ValueError("FAST skip_tokens must be a nonnegative integer")
        self.offset = sentencepiece.vocab_size() - 1 - skip_tokens
        if self.offset - self.vocab_size < 0:
            raise ValueError("FAST vocabulary exceeds PaliGemma reserved range")

    def encode_actions(self, actions):
        actions = np.asarray(actions)
        if actions.ndim != 3 or not np.isfinite(actions).all():
            raise ValueError("FAST requires finite normalized [batch, time, action] targets")
        coefficients = np.around(dct(actions, axis=1, norm="ortho") * self.scale)
        strings = []
        for chunk in coefficients:
            codepoints = np.maximum(chunk.flatten() - self.min_token, 0)
            if (codepoints > 0x10FFFF).any():
                raise ValueError(
                    "FAST coefficients exceed Unicode range; inspect action normalization"
                )
            strings.append("".join(map(chr, codepoints.astype(int))))
        encoded = self.bpe.encode_batch(strings)
        return [item.ids for item in encoded]

    def targets(self, actions, *, device):
        # The existing Pi05 prompt already ends with 'Action: '. Predict only
        # FAST tokens and the official '|'+EOS terminator after that context.
        end = self.sp.encode("|", add_eos=True)
        sequences = [
            [self.offset - token for token in row] + end for row in self.encode_actions(actions)
        ]
        return pad_targets(sequences, self.max_tokens, device=device)

    def decode_actions(self, codes, *, horizon, action_dim):
        """Invert the author's DCT/BPE codec, rejecting malformed generated actions."""
        if type(horizon) is not int or horizon < 1 or type(action_dim) is not int or action_dim < 1:
            raise ValueError("FAST decoding requires positive horizon and action_dim")
        chunks = []
        for row in codes:
            if not row or any(type(i) is not int or not 0 <= i < self.vocab_size for i in row):
                raise ValueError("Generated FAST action contains invalid BPE tokens")
            text = self.bpe.decode(row, skip_special_tokens=False)
            values = np.asarray([ord(char) + self.min_token for char in text], np.float64)
            if len(values) != horizon * action_dim:
                raise ValueError(
                    f"Generated FAST coefficients have length {len(values)}; "
                    f"expected {horizon * action_dim}"
                )
            coefficients = values.reshape(horizon, action_dim) / self.scale
            chunks.append(idct(coefficients, axis=0, norm="ortho").astype(np.float32))
        if not chunks:
            raise ValueError("FAST decoding requires at least one action sequence")
        return np.stack(chunks)


def pad_targets(sequences, max_tokens, *, device):
    if not sequences or any(not row or len(row) > max_tokens for row in sequences):
        raise ValueError(
            f"Language/FAST target exceeds max tokens ({max_tokens}); "
            "increase the limit, never truncate actions"
        )
    length = max(map(len, sequences))
    tokens = torch.zeros((len(sequences), length), dtype=torch.long, device=device)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for i, row in enumerate(sequences):
        tokens[i, : len(row)] = torch.as_tensor(row, dtype=torch.long, device=device)
        mask[i, : len(row)] = True
    return tokens, mask

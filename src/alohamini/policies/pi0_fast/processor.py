"""Official FAST task/state prefix and action postfix, with local tokenizer assets."""

import numpy as np
import torch

from alohamini.policies.pi05.adapter import PI05BatchProcessor
from alohamini.policies.pi05.fast_tokenizer import FASTTokenizer, pad_targets
from alohamini.policies.pi05.processor_pi05 import PI05Processor


class FASTPromptTokenizer:
    def __init__(self, tokenizer, max_length):
        self.sp, self.max_length = tokenizer, max_length

    def tokenize(self, prompt, state):
        cleaned = prompt.lower().strip().replace("_", " ")
        bins = np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1
        text = f"Task: {cleaned}, State: {' '.join(map(str, bins))};\n"
        ids = self.sp.encode(text, add_bos=True)
        if len(ids) > self.max_length:
            raise ValueError("FAST task/state prefix exceeds max_token_len; increase the limit")
        return np.pad(ids, (0, self.max_length - len(ids))), np.arange(self.max_length) < len(ids)


class FASTActionDecoder:
    def __init__(self, config):
        import sentencepiece

        self.config = config
        self.sp = sentencepiece.SentencePieceProcessor(model_file=config.tokenizer_path)
        self.codec = FASTTokenizer(config.fast_tokenizer_path, self.sp, config.fast_max_tokens)

    def decode(self, generated, *, device):
        prefix = self.sp.encode("Action: ")
        end = self.sp.encode("|", add_eos=True)
        codes = []
        for sequence in generated:
            if sequence[: len(prefix)] != prefix:
                raise ValueError("Generated FAST action is missing its Action: prefix")
            if sequence[-len(end) :] != end:
                raise ValueError("Generated FAST action is missing its complete terminator")
            codes.append([self.codec.offset - i for i in sequence[len(prefix) : -len(end)]])
        return torch.as_tensor(
            self.codec.decode_actions(codes, horizon=self.config.chunk_size, action_dim=18),
            device=device,
        )


class PI0FastProcessor(PI05BatchProcessor, FASTActionDecoder):
    def __init__(self, config, stats, device):
        FASTActionDecoder.__init__(self, config)
        self.device = device
        self.transform = PI05Processor(
            stats,
            FASTPromptTokenizer(self.sp, config.max_token_len),
            action_dim=18,
            delta_dims=config.delta_dims,
            cameras=config.camera_order,
        )

    def __call__(self, batch):
        result = self.prepare_observations(batch)
        if "action" in result:
            codes = self.codec.encode_actions(result["action"].detach().cpu().numpy())
            prefix = self.sp.encode("Action: ")
            end = self.sp.encode("|", add_eos=True)
            sequences = [prefix + [self.codec.offset - i for i in row] + end for row in codes]
            result["fast_tokens"], result["fast_mask"] = pad_targets(
                sequences, self.config.fast_max_tokens, device=self.device
            )
        return result

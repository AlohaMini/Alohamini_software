"""Task tokenization and shared normalization; geometry remains explicitly declared."""

import torch

from alohamini.learning.processor import Processor


class ACDiTProcessor:
    def __init__(self, config, stats, tokenizer, device):
        self.config, self.tokenizer = config, tokenizer
        self.numeric = Processor.from_config(config, stats, device)
        normalized = config.normalization_mapping.get("STATE") == "MEAN_STD"
        state_stats = stats["observation.state"]
        std = torch.tensor(state_stats["std"], dtype=torch.float32, device=device)
        self.state_noise_std = std / (std + 1e-8) if normalized else std
        self.state_mean = torch.tensor(
            [0.0] * len(config.state_names) if normalized else state_stats["mean"],
            dtype=torch.float32,
            device=device,
        )

    def __call__(self, batch):
        tasks = batch.get("task")
        if (
            not isinstance(tasks, (list, tuple))
            or len(tasks) != len(batch["observation.state"])
            or any(not isinstance(t, str) or not t.strip() for t in tasks)
        ):
            raise ValueError("AC-DiT requires one nonempty task per observation")
        tokens = self.tokenizer(
            list(tasks),
            padding="max_length",
            max_length=self.config.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        result = self.numeric({key: value for key, value in batch.items() if key != "task"})
        result["observation.language.tokens"] = tokens["input_ids"].to(self.numeric.device)
        result["observation.language.attention_mask"] = (
            tokens["attention_mask"].to(self.numeric.device).bool()
        )
        result["_state_noise_std"] = self.state_noise_std
        result["_state_mean"] = self.state_mean
        return result

    def action(self, tensor, *, context=None):
        return self.numeric.action(tensor)

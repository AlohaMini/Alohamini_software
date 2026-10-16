"""CLIP task tokenization and shared numeric/image normalization."""

from alohamini.learning.processor import Processor


class MultiTaskDiTProcessor(Processor):
    def __init__(self, config, stats, tokenizer, device):
        features = {**config.input_features, **config.output_features}
        super().__init__(
            stats,
            device,
            modes={
                key: config.normalization_mapping.get(feature.type, "IDENTITY")
                for key, feature in features.items()
            },
        )
        self.config = config
        self.tokenizer = tokenizer

    def __call__(self, batch):
        tasks = batch.get("task")
        if (
            not isinstance(tasks, (list, tuple))
            or len(tasks) != batch["observation.state"].shape[0]
            or any(not isinstance(task, str) or not task.strip() for task in tasks)
        ):
            raise ValueError("Multi-Task DiT requires one nonempty task instruction per sample")
        tokens = self.tokenizer(
            list(tasks),
            padding=self.config.tokenizer_padding,
            max_length=self.config.tokenizer_max_length,
            truncation=self.config.tokenizer_truncation,
            return_tensors="pt",
        )
        if tokens["input_ids"].shape[1] > self.config.tokenizer_max_length:
            raise ValueError("Task instruction exceeds tokenizer_max_length")
        return super().__call__(
            {
                **{key: value for key, value in batch.items() if key != "task"},
                "observation.language.tokens": tokens["input_ids"],
                "observation.language.attention_mask": tokens["attention_mask"],
            }
        )

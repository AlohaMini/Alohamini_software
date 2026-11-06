"""Official multi-view prompt, numeric normalization and auxiliary action tokens."""

import torch
from PIL import Image

from alohamini.policies.chunk import local_bpe_assets

from ._vendor.io import resize_image
from .semantic import XiaomiFields


class XiaomiProcessor:
    def __init__(self, config, stats, device):
        from transformers import AutoProcessor

        self.config, self.device = config, torch.device(device)
        self.fields = XiaomiFields(config.kinematic_fields)
        special = {
            "score": "<score>",
            "state": "<state>",
            **{f"a_{i}": f"<a_{i}>" for i in range(60)},
        }
        self.processor = AutoProcessor.from_pretrained(
            local_bpe_assets(config.assets_path), local_files_only=True
        )
        self.processor.tokenizer.add_tokens(list(special.values()), special_tokens=True)
        ids = self.processor.tokenizer.convert_tokens_to_ids(
            ["<score>", "<state>", "<a_0>", "<a_59>"]
        )
        if ids != [151669, 151670, 151671, 151730]:
            raise ValueError(f"Xiaomi action-token IDs differ from released weights: {ids}")
        self.processor.tokenizer.padding_side = "right"
        self.stats = {
            k: {n: torch.tensor(v, device=device, dtype=torch.float32) for n, v in values.items()}
            for k, values in stats["xiaomi"].items()
        }

    @staticmethod
    def messages(task, images):
        content = [
            {
                "type": "text",
                ("text"): (
                    "The following observations are captured from multiple views.\n# Ego View\n"
                ),
            }
        ]
        for image, following in zip(
            images,
            (
                "\n# Left-Wrist View\n",
                "\n# Right-Wrist View\n",
                f"\nGenerate robot actions for the task:\n{task} /no_cot",
            ),
            strict=True,
        ):
            content += [{"type": "image", "image": image}, {"type": "text", "text": following}]
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]

    def _prompt(self, messages):
        result = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
            do_resize=False,
        )
        return {k: v.to(self.device) for k, v in result.items() if torch.is_tensor(v)}

    def __call__(self, batch):
        raw = batch["observation.state"].detach().cpu().numpy()
        tasks = batch.get("task")
        if (
            not isinstance(tasks, (list, tuple))
            or len(tasks) != len(raw)
            or any(not t.strip() for t in tasks)
        ):
            raise ValueError("Xiaomi requires a nonempty task per sample")
        state = torch.tensor(self.fields.state(raw), device=self.device)[:, None]
        q = self.stats["state"]
        span = q["q99"] - q["q01"]
        state = torch.where(span > 0, 2 * (state - q["q01"]) / (span + 1e-6) - 1, 0).clamp(-1, 1)
        messages = []
        for index, task in enumerate(tasks):
            images = []
            for key in self.config.image_features:
                value = batch[key][index].detach().cpu()
                if value.dtype != torch.uint8:
                    value = (value * 255).round().clamp(0, 255).byte()
                images.append(
                    resize_image(
                        Image.fromarray(value.permute(1, 2, 0).numpy()),
                        max_pixels=self.config.image_max_pixels,
                    )
                )
            messages.append(self.messages(task, images))
        inference = self._prompt(messages)
        result = {"inference_inputs": inference, "state": state}
        mask = torch.zeros(
            len(raw), self.config.chunk_size, 60, dtype=torch.bool, device=self.device
        )
        mask[..., self.fields.action_slots] = True
        if "action" in batch:
            values = self.fields.action(raw, batch["action"].detach().cpu().numpy())
            action = torch.tensor(values, device=self.device)
            s = self.stats["action"]
            action = (action - s["mean"]) / (s["std"] + 1e-6)
            valid = ~batch["action_is_pad"].to(self.device)
            mask &= valid[..., None]
            lengths = valid.sum(-1)
            if (lengths < 1).any():
                raise ValueError("Xiaomi auxiliary action tokens require a valid action target")
            # Action tokens represent a contiguous prefix of each episode window.
            if not torch.equal(
                valid, torch.arange(valid.shape[1], device=self.device)[None] < lengths[:, None]
            ):
                raise ValueError("Xiaomi action padding must be a contiguous episode tail")
            prompts = [
                m
                + [
                    {"role": "user", "content": [{"type": "text", "text": "Robot state: <state>"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "".join(f"<a_{i}>" for i in range(int(n))) + "<score>",
                            }
                        ],
                    },
                ]
                for m, n in zip(messages, lengths.tolist(), strict=True)
            ]
            result.update(self._prompt(prompts))
            ids = result["input_ids"]
            condition = []
            for row in ids:
                state_index = (row == 151670).nonzero()[0, 0]
                turns = (row[:state_index] == 151644).nonzero().flatten()
                condition.append(int(turns[-1]))
            result["condition_lengths"] = torch.tensor(condition, device=self.device)
            result.update(
                action=action,
                action_mask=mask,
                vlm_action_actual_length=lengths,
                vlm_action_target=action[valid],
                vlm_action_mask=mask[valid],
            )
        else:
            result.update(action=torch.zeros_like(mask, dtype=torch.float32), action_mask=mask)
        return result

    def action(self, tensor, *, context=None):
        if context is None:
            raise ValueError("Xiaomi decoding requires the raw joint observation anchor")
        single = tensor.ndim == 2
        values = tensor[:, None] if single else tensor
        s = self.stats["action"]
        # Preserve truly constant labels exactly, including a stationary base.
        values = values * s["std"][: values.shape[1]] + s["mean"][: values.shape[1]]
        result = self.fields.decode(
            context.detach().cpu().numpy(), values.detach().float().cpu().numpy()
        )
        result = torch.tensor(result, device=tensor.device)
        return result[:, 0] if single else result

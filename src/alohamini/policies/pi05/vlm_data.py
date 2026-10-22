"""Optional local VQA/caption/planning examples for KI co-training."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from .fast_tokenizer import pad_targets
from .image_tools import resize_with_pad
from .processor_pi05 import CAMERAS


def manifest_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class VLMCoTrainingData:
    """One independently sampled VLM example per robot example.

    Uses the trainer's checkpointed CPU torch RNG. Keeps only JSONL offsets in
    memory; image paths are relative to the manifest. Never used during inference
    or robot validation. The manifest hash is part of the saved policy config.
    """

    def __init__(self, config, tokenizer):
        self.config, self.tokenizer = config, tokenizer
        self.path = Path(config.vlm_data_path).expanduser().resolve()
        if not config.vlm_data_sha256 or manifest_digest(self.path) != config.vlm_data_sha256:
            raise ValueError("VLM co-training manifest changed or has no saved SHA256")
        self.offsets = []
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError("VLM co-training manifest is empty")

    def sample(self, batch_size, *, device):
        prompts, answers, images = [], [], {name: [] for name in CAMERAS}
        image_masks = {name: [] for name in CAMERAS}
        indices = torch.randint(len(self.offsets), (batch_size,)).tolist()
        with self.path.open("rb") as stream:
            for index in indices:
                stream.seek(self.offsets[index])
                row = json.loads(stream.readline())
                if not all(
                    isinstance(row.get(key), str) and row[key].strip()
                    for key in ("prompt", "answer")
                ):
                    raise ValueError("VLM rows require nonempty prompt and answer strings")
                cameras = row.get("images", {})
                if (
                    not isinstance(cameras, dict)
                    or "forward" not in cameras
                    or set(cameras) - set(CAMERAS.values())
                ):
                    raise ValueError(
                        "VLM images require forward and optional wrist_left/wrist_right paths"
                    )
                # No dummy robot state is added to general VLM examples.
                prompts.append(
                    self.tokenizer._tokenizer.encode(row["prompt"], add_bos=True)
                    + self.tokenizer._tokenizer.encode("\n")
                )
                answers.append(self.tokenizer._tokenizer.encode(row["answer"], add_eos=True))
                for target, source in CAMERAS.items():
                    present = source in cameras
                    if present:
                        path = Path(cameras[source]).expanduser()
                        if not path.is_absolute():
                            path = self.path.parent / path
                        with Image.open(path) as image:
                            rgb = resize_with_pad(np.asarray(image.convert("RGB")), 224, 224)
                        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 127.5 - 1
                    else:
                        tensor = torch.full((3, 224, 224), -1.0)
                    images[target].append(tensor)
                    image_masks[target].append(present)
        prompt, prompt_mask = pad_targets(
            prompts, self.config.network["max_token_len"], device=device
        )
        targets, target_mask = pad_targets(answers, self.config.vlm_max_tokens, device=device)
        observation = SimpleNamespace(
            images={key: torch.stack(value).to(device) for key, value in images.items()},
            image_masks={
                key: torch.tensor(value, device=device) for key, value in image_masks.items()
            },
            state=torch.zeros(batch_size, self.config.network["action_dim"], device=device),
            tokenized_prompt=prompt,
            tokenized_prompt_mask=prompt_mask,
            token_ar_mask=None,
            token_loss_mask=None,
        )
        return dict(observation=observation, targets=targets, target_mask=target_mask)

"""Recorded numeric fields and explicit camera layout into Cosmos3 model inputs."""

import json
import math
import random

import torch

from alohamini.learning.processor import Processor

from .canvas import VIDEO_CANVAS_SIZES, VideoCanvas


class Cosmos3Processor:
    def __init__(self, model, stats, device):
        self.config = model.config
        if model.config.use_state:
            # The clean state token lives in the same normalized coordinates as the action tokens.
            stats = {**stats, "observation.state": stats["action"]}
        self.numeric = Processor.from_config(model.config, stats, device)
        self.tokenizer = model.tokenizer
        self.canvas = VideoCanvas(model.config.canvas_hw)
        token = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        if (
            token is None
            or token == self.tokenizer.unk_token_id
            or self.tokenizer.eos_token is None
        ):
            raise ValueError("Cosmos3 tokenizer requires end-of-turn and vision-start tokens")

    def caption(self, task, height, width):
        cfg = self.config
        if not task:
            return ""
        task = task.strip()
        if not task.endswith((".", "!", "?")):
            task += "."
        duration = (cfg.chunk_size + 1) / cfg.fps
        rounded = round(duration)
        ratios = [
            ratio
            for sizes in VIDEO_CANVAS_SIZES.values()
            for ratio, size in sizes.items()
            if size == (width, height)
        ]
        divisor = math.gcd(height, width)
        ratio = ratios[0] if ratios else f"{width // divisor},{height // divisor}"
        description = cfg.view_description
        if not description:
            roles = [
                cfg.camera_roles.get(k, k.removeprefix("observation.images."))
                for k in cfg.camera_keys
            ]
            if len(roles) == 1:
                description = f"The view is from camera {roles[0]}."
            elif len(roles) == 2:
                description = (
                    f"The left view is from camera {roles[0]}; "
                    f"the right view is from camera {roles[1]}."
                )
            else:
                description = (
                    f"The top view is from camera {roles[0]}; the lower left and right "
                    f"views are from cameras {roles[1]} and {roles[2]}."
                )
        framing = (
            "This video contains concatenated views from multiple camera perspectives. "
            if len(cfg.camera_keys) > 1
            else ""
        ) + description
        return json.dumps(
            dict(
                cinematography=dict(framing=framing),
                actions=[
                    dict(
                        time=f"0:00-{rounded // 60}:{rounded % 60:02d}",
                        description=task,
                        idle_frame=f"0 out of {cfg.chunk_size + int(cfg.use_state)}.",
                    )
                ],
                duration=f"{int(duration)}s",
                fps=float(int(cfg.fps)),
                resolution=dict(H=height, W=width),
                aspect_ratio=ratio,
            )
        )

    def encode(self, prompts):
        suffix = self.tokenizer.eos_token + "<|vision_start|>"
        texts = [
            self.tokenizer.apply_chat_template(
                [dict(role="user", content=[dict(type="text", text=prompt)])],
                tokenize=False,
                add_generation_prompt=True,
            )
            + suffix
            for prompt in prompts
        ]
        return self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            padding_side="right",
            return_tensors="pt",
        )

    def __call__(self, batch):
        cfg = self.config
        result = self.numeric(batch)
        training_clip = "action" in result
        tasks = result.get("task")
        if (
            not isinstance(tasks, (list, tuple))
            or not tasks
            or any(not isinstance(t, str) or not t.strip() for t in tasks)
        ):
            raise ValueError("Cosmos3 requires a nonempty task per sample")
        views = []
        for key in cfg.camera_keys:
            view = result[key]
            if view.ndim == 5 and not training_clip:
                view = view[:, 0]
            view = self.canvas.prepare_view(view, cfg.view_size)
            if training_clip and view.shape[1] != cfg.chunk_size + 1:
                raise ValueError("Cosmos3 training requires current plus complete future video")
            views.append(view)
        frames = (self.canvas.compose(views) * 255).clamp(0, 255).to(torch.uint8)
        canvas, (height, width) = self.canvas.fit_canvas(frames)
        size, device = canvas.shape[0], canvas.device
        if len(tasks) != size:
            raise ValueError("Cosmos3 task count differs from image batch size")
        spatial = cfg.vae.get("scale_factor_spatial") or 1
        temporal = cfg.vae.get("scale_factor_temporal") or 1
        if height < spatial or width < spatial:
            raise ValueError("Cosmos3 camera content must contain at least one VAE latent cell")
        result.update(
            pixel_values_videos_gen=canvas.permute(0, 2, 1, 3, 4).float() / 127.5 - 1,
            video_grid_thw_gen=torch.tensor(
                [1 + cfg.chunk_size // temporal, height // spatial, width // spatial], device=device
            ).expand(size, 3),
            second_per_grid_ts_gen=torch.full((size,), temporal / cfg.fps, device=device),
        )
        prompts = [self.caption(t, *canvas.shape[-2:]) for t in tasks]
        if training_clip and torch.is_grad_enabled() and cfg.cfg_dropout_rate:
            prompts = ["" if random.random() < cfg.cfg_dropout_rate else text for text in prompts]
        encoded = self.encode(prompts)
        result.update(
            input_ids=encoded.input_ids.to(device), attention_mask=encoded.attention_mask.to(device)
        )
        result["mm_token_type_ids"] = torch.zeros_like(result["input_ids"])
        negative = self.encode([""])
        result.update(
            negative_input_ids=negative.input_ids.expand(size, -1).to(device),
            negative_attention_mask=negative.attention_mask.expand(size, -1).to(device),
            negative_mm_token_type_ids=torch.zeros_like(negative.input_ids)
            .expand(size, -1)
            .to(device),
        )
        return result

    def action(self, value, *, context=None):
        return self.numeric.action(value)

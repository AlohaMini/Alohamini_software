"""Native Qwen readouts, flow DiT and a frozen latent-video target encoder."""

from copy import deepcopy

import torch
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, VJEPA2Config, VJEPA2Model

from alohamini.policies.chunk import ChunkPolicy

from .action_head import VLAJEPAActionHead
from .world_model import ActionConditionedVideoPredictor


def world_validity(config, batch):
    tubelet = config.jepa_tubelet_size
    mask = None
    for key in config.world_model_cameras:
        value = batch.get(key + "_is_pad")
        if value is None:
            raise ValueError("VLA-JEPA video supervision requires temporal padding masks")
        valid = (~value).reshape(len(value), -1, tubelet).all(-1)[:, 1:]
        mask = valid if mask is None else mask & valid
    return mask


class VLAJEPAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        qwen = Qwen3VLConfig(**deepcopy(config.qwen_config))
        qwen._attn_implementation = "sdpa"
        self.qwen = nn.Module()
        self.qwen.model = Qwen3VLForConditionalGeneration._from_config(
            qwen, dtype=getattr(torch, config.dtype)
        )
        # HF4.57 records block outputs before the final norm. Keep this original readout,
        # rather than using the changed HF5 hidden_states collection convention.
        self.qwen.model.model.language_model.norm.requires_grad_(False)
        if self.qwen.model.lm_head.weight is not self.qwen.model.get_input_embeddings().weight:
            self.qwen.model.lm_head.requires_grad_(False)
        if config.gradient_checkpointing:
            self.qwen.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if config.freeze_qwen:
            self.qwen.requires_grad_(False)
        width = qwen.text_config.hidden_size
        self.action_model = VLAJEPAActionHead(config, cross_attention_dim=width)
        self.action_model.model.gradient_checkpointing = config.gradient_checkpointing
        if config.enable_world_model:
            video = VJEPA2Config(**deepcopy(config.jepa_config))
            self.video_encoder = VJEPA2Model._from_config(video, dtype=torch.float32)
            self.video_encoder.requires_grad_(False).eval()
            patch = video.patch_size
            if isinstance(patch, (list, tuple)):
                if len(set(patch)) != 1:
                    raise ValueError("VLA-JEPA predictor requires square V-JEPA patch geometry")
                patch = patch[0]
            self.video_predictor = ActionConditionedVideoPredictor(
                num_frames=config.num_video_frames // video.tubelet_size,
                img_size=(video.crop_size, video.crop_size),
                patch_size=patch,
                tubelet_size=1,
                embed_dim=video.hidden_size * len(config.world_model_cameras),
                action_embed_dim=width,
                predictor_embed_dim=video.hidden_size,
                depth=config.predictor_depth,
                num_heads=config.predictor_num_heads,
                mlp_ratio=config.predictor_mlp_ratio,
                num_action_tokens_per_step=config.num_action_tokens_per_timestep,
                dropout=config.predictor_dropout,
                rope_legacy=config.world_model_rope_legacy,
            )
            self.video_predictor.gradient_checkpointing = config.gradient_checkpointing
            if config.retain_unused_predictor_heads:
                # Published policy weights retain these unused projections. Freeze them
                # and preserve their exact names/shapes for strict portable loading.
                self.video_predictor.state_encoder = nn.Linear(width, video.hidden_size)
                self.video_predictor.extrinsics_encoder = nn.Linear(width - 1, video.hidden_size)
                self.video_predictor.state_encoder.requires_grad_(False)
                self.video_predictor.extrinsics_encoder.requires_grad_(False)
        else:
            self.video_encoder, self.video_predictor = None, None

    def train(self, mode=True):
        super().train(mode)
        if self.video_encoder is not None:
            self.video_encoder.eval()
        if self.config.freeze_qwen:
            self.qwen.eval()
        return self

    def readouts(self, batch):
        captured = []

        def capture(module, args, output):
            captured.append(output[0] if isinstance(output, tuple) else output)

        layer = self.qwen.model.model.language_model.layers[-1]
        handle = layer.register_forward_hook(capture)
        try:
            self.qwen.model.model(
                **{
                    key: batch[key]
                    for key in (
                        "input_ids",
                        "attention_mask",
                        "pixel_values",
                        "image_grid_thw",
                        "mm_token_type_ids",
                    )
                    if key in batch
                },
                use_cache=False,
            )
        finally:
            handle.remove()
        hidden = captured[0]
        size, width = hidden.shape[0], hidden.shape[-1]
        embodied = hidden[batch["embodied_token_mask"]].reshape(size, -1, width)
        action = hidden[batch["action_token_mask"]].reshape(size, -1, width)
        return embodied, action

    @staticmethod
    def merge_views(features, batch, views):
        n, d = features.shape[1:]
        return features.reshape(batch, views, n, d).permute(0, 2, 1, 3).reshape(batch, n, views * d)

    def world_loss(self, batch, readouts):
        cfg = self.config
        size, views = len(batch["action"]), len(cfg.world_model_cameras)
        pixels = batch["video_pixels"]
        tubelet = cfg.jepa_tubelet_size
        positions = cfg.num_video_frames // tubelet
        with torch.no_grad():
            full = self.video_encoder.get_vision_features(pixel_values_videos=pixels)
            tokens = full.shape[1] // positions
            target = self.merge_views(full, size, views)[:, tokens:]
            if cfg.world_model_context == "causal":
                leading = []
                for i in range(positions - 1):
                    feature = self.video_encoder.get_vision_features(
                        pixel_values_videos=pixels[:, : (i + 1) * tubelet]
                    )
                    leading.append(feature[:, -tokens:])
                context = self.merge_views(torch.cat(leading, dim=1), size, views)
            else:
                context = self.merge_views(full, size, views)[:, : tokens * (positions - 1)]
        with torch.autocast(pixels.device.type, enabled=False):
            prediction = self.video_predictor(context.float(), readouts.float())
            loss = (
                (prediction - target.float())
                .abs()
                .reshape(size, positions - 1, tokens, -1)
                .mean((2, 3))
            )
            valid = world_validity(cfg, batch)
            per_sample = (loss * valid).sum(-1) / valid.sum(-1).clamp_min(1)
            active = valid.any(-1)
            return (per_sample * active).sum() / active.sum().clamp_min(1)


class VLAJEPAPolicy(ChunkPolicy):
    name = "vla_jepa"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = VLAJEPAModel(config)
        self.reset()

    def get_optim_params(self):
        cfg = self.config
        groups = []
        for name, module in (
            ("qwen", self.model.qwen),
            ("action", self.model.action_model),
            ("world", self.model.video_predictor),
        ):
            if module is not None:
                params = [p for p in module.parameters() if p.requires_grad]
                if params:
                    groups.append(dict(name=name, params=params, lr_scale=cfg.lr_scales[name]))
        return groups

    def forward(self, batch):
        cfg = self.config
        embodied, readouts = self.model.readouts(batch)
        repeats = cfg.repeated_diffusion_steps
        with torch.autocast(embodied.device.type, enabled=False):
            flow = self.model.action_model(
                embodied.float().repeat(repeats, 1, 1),
                batch["action"].float().repeat(repeats, 1, 1),
                batch["observation.state"].float()[:, None].repeat(repeats, 1, 1),
                batch["action_is_pad"].repeat(repeats, 1),
                reduction="none",
            ).mean()
        loss = flow * batch.get("_loss_weight", 1.0)
        terms = dict(action_loss=flow.detach())
        if cfg.enable_world_model:
            world = self.model.world_loss(batch, readouts)
            loss = loss + cfg.world_model_loss_weight * world * batch.get("_world_weight", 1.0)
            terms["wm_loss"] = world.detach()
        return loss, terms

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        self.eval()
        embodied, _ = self.model.readouts(batch)
        with torch.autocast(embodied.device.type, enabled=False):
            return self.model.action_model.predict_action(
                embodied.float(), batch["observation.state"].float()[:, None]
            ).float()

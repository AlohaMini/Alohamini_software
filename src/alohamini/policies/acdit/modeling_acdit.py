"""Native two-stage AC-DiT; the frozen mobility rollout conditions the body DiT."""

import logging
from pathlib import Path

import torch
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, SiglipConfig, SiglipImageProcessor, SiglipModel

from alohamini.policies.chunk import ChunkPolicy, local_assets

from .point_encoder import Lift3DEncoder
from .transformer import CoordinationDiT, PerceptionAdapter


def projector(inputs, outputs, depth):
    layers = [nn.Linear(inputs, outputs)]
    for _ in range(depth - 1):
        layers.extend((nn.GELU(approximate="tanh"), nn.Linear(outputs, outputs)))
    return nn.Sequential(*layers)


class ConditionedHead(nn.Module):
    def __init__(self, config, feature_dim, language_dim, *, body=False):
        super().__init__()
        self.config = config
        width = config.hidden_dim if body else config.mobility_hidden_dim
        self.model = CoordinationDiT(
            config.padded_dim,
            config.chunk_size,
            width,
            config.num_layers if body else config.mobility_num_layers,
            config.num_heads if body else config.mobility_num_heads,
            config.num_inference_steps if body else 0,
        )
        self.model.gradient_checkpointing = config.gradient_checkpointing
        self.lang_adaptor = projector(language_dim, width, 2)
        self.img_adaptor = projector(feature_dim, width, 2)
        self.state_adaptor = projector(config.padded_dim * 2, width, 3)
        self.lift3d_adaptor = projector(feature_dim, width, 2) if config.use_pointcloud else None
        self.in_context_conditions_adaptor = (
            projector(config.context_dim, width, 2) if config.context_dim else None
        )
        self.latent_mobility_feature = (
            projector(config.mobility_hidden_dim, width, 3) if body else None
        )

    def forward(
        self, state, state_mask, action, action_mask, frequency, timestep, features, latent=None
    ):
        language, language_mask, images, points, context = features
        values = torch.cat((state, action), dim=1)
        mask = torch.cat((state_mask, action_mask[:, None].expand(-1, action.shape[1], -1)), dim=1)
        x = self.state_adaptor(torch.cat((values, mask), dim=-1))
        conditions = [self.lang_adaptor(language), self.img_adaptor(images.flatten(1, 3))]
        masks = [
            language_mask,
            torch.ones(conditions[-1].shape[:2], device=x.device, dtype=torch.bool),
        ]
        if points is not None:
            conditions.append(self.lift3d_adaptor(points.flatten(1, 2)))
            masks.append(torch.ones(conditions[-1].shape[:2], device=x.device, dtype=torch.bool))
        context = (
            self.in_context_conditions_adaptor(context)
            if self.in_context_conditions_adaptor is not None
            else x.new_zeros(x.shape[0], 1, self.model.hidden_size)
        )
        latent = self.latent_mobility_feature(latent) if latent is not None else None
        return self.model(x, frequency, timestep, context, conditions, masks, latent)

    def sample(self, state, state_mask, action_mask, frequency, features, *, noise=None):
        cfg = self.config
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_schedule=cfg.beta_schedule,
            prediction_type=cfg.prediction_type,
        )
        scheduler.set_timesteps(cfg.num_inference_steps, device=state.device)
        action = (
            torch.randn(state.shape[0], cfg.chunk_size, cfg.padded_dim, device=state.device)
            if noise is None
            else noise.float().clone()
        )
        hidden = []
        for timestep in scheduler.timesteps:
            predicted, token = self(
                state, state_mask, action, action_mask, frequency, timestep.reshape(1), features
            )
            hidden.append(token)
            action = scheduler.step(predicted.float(), timestep, action).prev_sample
        return action * action_mask[:, None], torch.stack(hidden, dim=1).flatten(1, 2)

    def load_rdt(self, path):
        """Import matching RDT core tensors; report intentionally new adapters."""
        path = Path(path).expanduser()
        weights = load_file(path / "model.safetensors" if path.is_dir() else path)
        own = self.state_dict()
        compatible = {
            key: value
            for key, value in weights.items()
            if key in own and own[key].shape == value.shape
        }
        required = {key for key in own if key.startswith("model.blocks.")}
        if not required <= compatible.keys():
            raise ValueError("RDT initializer must contain every matching transformer block")
        self.load_state_dict(compatible, strict=False)
        logging.getLogger(__name__).info(
            "Imported %d RDT tensors; %d tensors use native initialization",
            len(compatible),
            len(own) - len(compatible),
        )


class ACDiTPolicy(ChunkPolicy):
    name = "acdit"

    def __init__(self, config):
        super().__init__()
        self.config = config
        assets = local_assets(config.assets_path)
        encoder_config = SiglipConfig.from_pretrained(assets, local_files_only=True)
        if encoder_config.text_config.hidden_size != encoder_config.vision_config.hidden_size:
            raise ValueError("AC-DiT perception adapter needs aligned SigLIP text/vision widths")
        if config.tokenizer_max_length > encoder_config.text_config.max_position_embeddings:
            raise ValueError("tokenizer_max_length exceeds the local SigLIP position table")
        self.siglip = (
            (
                SiglipModel.from_pretrained(assets, local_files_only=True)
                if config.load_pretrained_encoders
                else SiglipModel(encoder_config)
            )
            .requires_grad_(False)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
        self.image_processor = SiglipImageProcessor.from_pretrained(assets, local_files_only=True)
        if tuple(self.image_processor.image_mean) != (0.5, 0.5, 0.5) or tuple(
            self.image_processor.image_std
        ) != (0.5, 0.5, 0.5):
            raise ValueError("AC-DiT requires SigLIP RGB mean/std 0.5")
        width = encoder_config.vision_config.hidden_size
        self.point_encoder = (
            Lift3DEncoder(self.siglip.vision_model, config) if config.use_pointcloud else None
        )
        self.mobility = ConditionedHead(config, width, width)
        self.body = (
            ConditionedHead(config, width, width, body=True) if config.stage == "full" else None
        )
        self.perception = (
            PerceptionAdapter(width, use_pointcloud=config.use_pointcloud)
            if config.stage == "full"
            else None
        )
        if self.body is not None:
            self.mobility.requires_grad_(False).eval()
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
            clip_sample=config.clip_sample,
        )
        self.register_buffer(
            "action_slots",
            torch.tensor([config.action_slot_map[n] for n in config.action_names]),
            persistent=False,
        )
        self.register_buffer(
            "state_slots",
            torch.tensor([config.state_slot_map[n] for n in config.state_names]),
            persistent=False,
        )
        base = torch.zeros(config.padded_dim)
        base[[config.action_slot_map[n] for n in config.base_action_names]] = 1
        full = torch.zeros_like(base)
        full[self.action_slots] = 1
        self.register_buffer("base_mask", base, persistent=False)
        self.register_buffer("action_mask", full, persistent=False)
        self.reset()

    def train(self, mode=True):
        super().train(mode)
        self.siglip.eval()
        if self.body is not None:
            self.mobility.eval()
        return self

    def save_assets(self, path):
        self.siglip.config.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.image_processor.save_pretrained(path)

    def _encode(self, batch):
        cfg = self.config
        pictures = []
        for key in cfg.image_features:
            value = batch[key]
            if value.ndim == 4:
                if cfg.n_obs_steps != 1:
                    raise ValueError("AC-DiT requires an explicit RGB history window")
                value = value[:, None]
            if value.shape[1] != cfg.n_obs_steps:
                raise ValueError("AC-DiT RGB history differs from n_obs_steps")
            pictures.append(value)
        images = torch.stack(pictures, dim=2)
        batch_size, history, cameras = images.shape[:3]
        images = images.flatten(0, 2)
        size = self.siglip.config.vision_config.image_size
        if images.shape[-2:] != (size, size):
            images = F.interpolate(
                images, (size, size), mode="bilinear", align_corners=False, antialias=True
            )
        if self.training and cfg.image_augmentation:
            brightness = torch.empty(len(images), 1, 1, 1, device=images.device).uniform_(0.8, 1.2)
            images = (images * brightness).clamp(0, 1)
        if self.training and cfg.condition_mask_prob:
            images = images.masked_fill(
                (torch.rand(len(images), 1, 1, 1, device=images.device) < cfg.condition_mask_prob),
                0.5,
            )
        with torch.no_grad():
            image_features = self.siglip.vision_model(pixel_values=images * 2 - 1).last_hidden_state
            language = self.siglip.text_model(
                input_ids=batch["observation.language.tokens"],
                attention_mask=batch["observation.language.attention_mask"],
            ).last_hidden_state
        image_features = image_features.reshape(
            batch_size, history, cameras, -1, image_features.shape[-1]
        )
        points = None
        if cfg.use_pointcloud:
            cloud = batch[cfg.pointcloud_field]
            if cloud.ndim != 4 or cloud.shape[1] != cfg.n_obs_steps:
                raise ValueError(
                    "AC-DiT requires a point-cloud history ending at the current frame"
                )
            points = self.point_encoder(cloud.flatten(0, 1)).reshape(
                batch_size, history, -1, image_features.shape[-1]
            )
        context = (
            torch.cat(
                [batch[key].reshape(batch_size, -1).float() for key in cfg.context_fields], dim=-1
            )[:, None]
            if cfg.context_fields
            else language.new_zeros(batch_size, 1, 0)
        )
        return (
            language,
            batch["observation.language.attention_mask"],
            image_features,
            points,
            context,
        )

    def _state(self, batch):
        state = batch["observation.state"]
        if state.ndim == 3:
            state = state[:, -1]
        if self.training:
            if self.config.state_noise_snr is not None:
                std = batch.get("_state_noise_std", torch.ones_like(state))
                state = state + torch.randn_like(state) * std / (
                    10 ** (self.config.state_noise_snr / 20)
                )
            if self.config.condition_mask_prob:
                state = torch.where(
                    torch.rand(len(state), 1, device=state.device)
                    < self.config.condition_mask_prob,
                    batch.get("_state_mean", torch.zeros_like(state)),
                    state,
                )
        padded = state.new_zeros(len(state), 1, self.config.padded_dim)
        padded[:, 0, self.state_slots] = state
        mask = torch.zeros_like(padded)
        mask[:, 0, self.state_slots] = 1
        frequency = state.new_full((len(state),), self.config.control_frequency)
        if self.training and self.config.condition_mask_prob:
            frequency = frequency.masked_fill(
                torch.rand(len(state), device=state.device) < self.config.condition_mask_prob, 0
            )
        return padded, mask, frequency

    def _body_conditions(self, features):
        language, mask, images, points, context = features
        images, points, weights = self.perception(images, points, language, mask)
        return (language, mask, images, points, context), weights

    def forward(self, batch, *, noise=None, timestep=None, mobility_noise=None):
        cfg = self.config
        features = self._encode(batch)
        state, state_mask, frequency = self._state(batch)
        mask = (self.base_mask if self.body is None else self.action_mask)[None].expand(
            len(state), -1
        )
        targets = state.new_zeros(len(state), cfg.chunk_size, cfg.padded_dim)
        targets[:, :, self.action_slots] = batch["action"]
        targets = targets * mask[:, None]
        latent = None
        if self.body is not None:
            with torch.no_grad():
                _, latent = self.mobility.sample(
                    state,
                    state_mask,
                    self.base_mask[None].expand_as(mask),
                    frequency,
                    features,
                    noise=mobility_noise,
                )
            features, _ = self._body_conditions(features)
        noise = torch.randn_like(targets) if noise is None else noise
        timestep = (
            torch.randint(cfg.num_train_timesteps, (len(state),), device=state.device)
            if timestep is None
            else timestep
        )
        noisy = self.noise_scheduler.add_noise(targets, noise, timestep)
        predicted, _ = (self.mobility if self.body is None else self.body)(
            state, state_mask, noisy, mask, frequency, timestep, features, latent
        )
        errors = (
            predicted.float() - (targets if cfg.prediction_type == "sample" else noise).float()
        ).square()
        if cfg.mask_padding_loss:
            valid = (~batch["action_is_pad"])[..., None].to(errors.dtype)
            loss = (errors * valid).sum() / (valid.sum() * cfg.padded_dim).clamp_min(1)
        else:
            loss = errors.mean()
        return loss * batch.get("_loss_weight", 1), {"diffusion_loss": loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch, *, noise=None, mobility_noise=None):
        if self.body is None:
            raise ValueError(
                "Mobility pretraining checkpoint is not a full-body deployment "
                "policy; train stage=full"
            )
        features = self._encode(batch)
        state, state_mask, frequency = self._state(batch)
        mask = self.action_mask[None].expand(len(state), -1)
        _, latent = self.mobility.sample(
            state,
            state_mask,
            self.base_mask[None].expand_as(mask),
            frequency,
            features,
            noise=mobility_noise,
        )
        features, _ = self._body_conditions(features)
        cfg = self.config
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_schedule=cfg.beta_schedule,
            prediction_type=cfg.prediction_type,
        )
        scheduler.set_timesteps(cfg.num_inference_steps, device=state.device)
        action = (
            torch.randn(len(state), cfg.chunk_size, cfg.padded_dim, device=state.device)
            if noise is None
            else noise.float().clone()
        )
        for timestep in scheduler.timesteps:
            predicted, _ = self.body(
                state, state_mask, action, mask, frequency, timestep.reshape(1), features, latent
            )
            action = scheduler.step(predicted.float(), timestep, action).prev_sample
        return action[:, :, self.action_slots]

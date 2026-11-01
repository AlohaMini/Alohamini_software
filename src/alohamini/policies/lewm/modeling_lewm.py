"""End-to-end JEPA learning with global SIGReg and visual-goal CEM control."""

import torch
import torch.distributed as dist
from torch import nn
from transformers import ViTConfig, ViTModel

from alohamini.policies.chunk import ChunkPolicy

from ._vendor.jepa import JEPA
from ._vendor.module import MLP, ARPredictor, Embedder, SIGReg


class GlobalSIGReg(SIGReg):
    def forward(self, proj):
        if self.training and dist.is_available() and dist.is_initialized():
            from torch.distributed.nn.functional import all_gather

            proj = torch.cat(all_gather(proj.contiguous()), dim=1)
            directions = torch.randn(proj.shape[-1], self.num_proj, device=proj.device)
            dist.broadcast(directions, src=0)
            directions = directions / directions.norm(dim=0)
            with torch.autocast(proj.device.type, enabled=False):
                x = (proj.float() @ directions.float()).unsqueeze(-1) * self.t.float()
                err = (x.cos().mean(-3) - self.phi.float()).square() + x.sin().mean(-3).square()
                return ((err @ self.weights.float()) * proj.shape[-2]).mean()
        # Keep characteristic functions/integration in FP32 under autocast.
        with torch.autocast(proj.device.type, enabled=False):
            return super().forward(proj.float())


class LeWMPolicy(ChunkPolicy):
    name = "lewm"

    def __init__(self, config):
        super().__init__()
        self.config = config
        size = next(iter(config.image_features.values())).shape[-1]
        encoder = ViTModel(
            ViTConfig(
                image_size=size,
                patch_size=config.patch_size,
                hidden_size=config.embed_dim,
                num_hidden_layers=config.encoder_depth,
                num_attention_heads=config.encoder_heads,
                intermediate_size=config.encoder_mlp_dim,
            ),
            add_pooling_layer=False,
            use_mask_token=False,
        )
        if config.gradient_checkpointing:
            encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.world = JEPA(
            encoder=encoder,
            predictor=ARPredictor(
                num_frames=config.history_size,
                input_dim=config.embed_dim,
                hidden_dim=config.embed_dim,
                output_dim=config.embed_dim,
                depth=config.predictor_depth,
                heads=config.predictor_heads,
                dim_head=config.predictor_head_dim,
                mlp_dim=config.predictor_mlp_dim,
                dropout=config.dropout,
            ),
            action_encoder=Embedder(
                input_dim=config.action_block * config.action_feature.shape[0],
                emb_dim=config.embed_dim,
            ),
            projector=MLP(
                config.embed_dim,
                config.projector_hidden_dim,
                config.embed_dim,
                norm_fn=nn.BatchNorm1d,
            ),
            pred_proj=MLP(
                config.embed_dim,
                config.projector_hidden_dim,
                config.embed_dim,
                norm_fn=nn.BatchNorm1d,
            ),
        )
        self.sigreg = GlobalSIGReg(config.sigreg_knots, config.sigreg_projections)
        self.world.predictor.transformer.gradient_checkpointing = config.gradient_checkpointing
        self._goal = None
        self.reset()

    def _encode(self, batch):
        images = []
        for key in self.config.image_features:
            value = batch[key]
            images.append(value[:, None] if value.ndim == 4 else value)
        images = torch.stack(images, dim=2)
        b, t, cameras = images.shape[:3]
        features = self.world.encoder(
            images.flatten(0, 2), interpolate_pos_encoding=True
        ).last_hidden_state[:, 0]
        return self.world.projector(features).reshape(b, t, cameras, -1).mean(2)

    def forward(self, batch):
        cfg = self.config
        embeddings = self._encode(batch)
        if embeddings.shape[1] != cfg.history_size + 1:
            raise ValueError("LeWM training needs history_size + 1 image observations")
        actions = batch["action"][:, :-1].reshape(len(embeddings), cfg.history_size, -1)
        predicted = self.world.predict(embeddings[:, :-1], self.world.action_encoder(actions))
        pad = batch["action_is_pad"]
        valid = ~pad[:, :-1].reshape(len(pad), cfg.history_size, cfg.action_block).any(-1)
        valid = valid & ~pad[:, cfg.action_block :: cfg.action_block]
        error = (predicted.float() - embeddings[:, 1:].float()).square().mean(-1)
        prediction_loss = (error * valid).sum() / valid.sum().clamp_min(1)
        regularizer = self.sigreg(embeddings.transpose(0, 1))
        loss = prediction_loss * batch.get(
            "_prediction_weight", 1.0
        ) + cfg.sigreg_weight * regularizer * batch.get("_sigreg_weight", 1.0)
        return loss, {
            "prediction_loss": prediction_loss.detach(),
            "sigreg_loss": regularizer.detach(),
        }

    def set_goal(self, images):
        if set(images) != set(self.config.image_features):
            raise ValueError("Goal must contain exactly the checkpoint cameras")
        batches = set()
        for key, value in images.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 4
                or tuple(value.shape[1:]) != self.config.image_features[key].shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Prepared goals must be finite BCHW tensors matching the cameras")
            batches.add(value.shape[0])
        if len(batches) != 1 or 0 in batches:
            raise ValueError("Goal camera batch sizes must match and be nonempty")
        self._goal = {k: v.detach().clone() for k, v in images.items()}
        self.reset()

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        if self.training:
            raise ValueError("Set LeWM to eval mode before planning")
        if self._goal is None:
            raise ValueError("LeWM requires explicit goal images; call policy.set_goal(images)")
        cfg = self.config
        current = {}
        for key in cfg.image_features:
            value = batch[key]
            if value.ndim == 5:
                # Training windows contain future supervision; it is never an
                # implicit inference goal or part of the current observation.
                index = cfg.history_size - 1 if value.shape[1] == cfg.history_size + 1 else -1
                value = value[:, index]
            current[key] = value
        initial = self._encode(current)[:, 0]
        goal = self._encode({k: v.to(initial.device) for k, v in self._goal.items()})[:, 0]
        if len(goal) not in (1, len(initial)):
            raise ValueError("Goal batch must have size 1 or match observations")
        goal = goal.expand(len(initial), -1)
        b, samples, horizon = len(initial), cfg.cem_samples, cfg.planning_horizon
        width = cfg.action_block * cfg.action_feature.shape[0]
        mean = initial.new_zeros(b, horizon, width)
        std = torch.full_like(mean, cfg.cem_initial_std)
        for _ in range(cfg.cem_iterations):
            plans = mean[:, None] + std[:, None] * torch.randn(
                b, samples, horizon, width, device=initial.device, dtype=initial.dtype
            )
            if cfg.constrain_actions:
                low = batch["_action_low"].repeat(cfg.action_block)
                high = batch["_action_high"].repeat(cfg.action_block)
                plans = plans.clamp(low, high)
            flat = plans.flatten(0, 1)
            latent = initial[:, None, None].expand(b, samples, 1, -1).flatten(0, 1)
            for t in range(horizon):
                start = max(0, t + 1 - cfg.history_size)
                context_actions = self.world.action_encoder(flat[:, start : t + 1])
                nxt = self.world.predict(latent[:, -cfg.history_size :], context_actions)[:, -1:]
                latent = torch.cat((latent, nxt), dim=1)
            final = latent[:, -1].reshape(b, samples, -1)
            cost = (final.float() - goal[:, None].float()).square().sum(-1)
            indices = cost.topk(cfg.cem_elites, largest=False, dim=1).indices
            elite = plans.gather(1, indices[..., None, None].expand(-1, -1, horizon, width))
            mean, std = elite.mean(1), elite.std(1, unbiased=False).clamp_min(1e-6)
        return mean.reshape(b, cfg.chunk_size, cfg.action_feature.shape[0]).float()

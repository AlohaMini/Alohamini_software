# Adapted from PKU-HMI-Lab/AC-DiT (MIT); see LICENSE.acdit.
"""RDT blocks and mobility-conditioned DiT with dynamic sequence lengths."""

import math

import numpy as np
import torch
from timm.layers import Mlp, RmsNorm
from timm.models.vision_transformer import Attention
from torch import nn
from torch.nn import functional as F


def position_embedding(width, positions):
    positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    frequency = 1 / 10000 ** (np.arange(width // 2, dtype=np.float64) / (width / 2))
    angle = positions[:, None] * frequency[None]
    return np.concatenate((np.sin(angle), np.cos(angle)), axis=1)


def multimodal_positions(width, lengths):
    result = []
    modality = position_embedding(width // 2, np.arange(len(lengths)))
    for i, length in enumerate(lengths):
        result.append(
            np.concatenate(
                (
                    np.repeat(modality[i : i + 1], length, axis=0),
                    position_embedding(width // 2, np.arange(length)),
                ),
                axis=1,
            )
        )
    return torch.from_numpy(np.concatenate(result)).float()[None]


class TimestepEmbedder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, timestep):
        frequency = torch.exp(
            -math.log(10000) * torch.arange(128, dtype=torch.float32, device=timestep.device) / 128
        )
        angle = timestep.reshape(-1, 1).float() * frequency[None]
        return self.mlp(torch.cat((angle.cos(), angle.sin()), dim=-1).to(self.mlp[0].weight.dtype))


class CrossAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.num_heads, self.head_dim = heads, width // heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, width * 2)
        self.q_norm, self.k_norm = RmsNorm(self.head_dim), RmsNorm(self.head_dim)
        self.proj = nn.Linear(width, width)

    def forward(self, x, conditions, mask):
        batch, length, width = x.shape
        q = self.q(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        kv = (
            self.kv(conditions)
            .reshape(batch, -1, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        output = F.scaled_dot_product_attention(
            self.q_norm(q), self.k_norm(k), v, attn_mask=mask[:, None, None, :]
        )
        return self.proj(output.transpose(1, 2).reshape(batch, length, width))


class RDTBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = [RmsNorm(width, eps=1e-6) for _ in range(3)]
        self.attn = Attention(
            width, num_heads=heads, qkv_bias=True, qk_norm=True, norm_layer=RmsNorm
        )
        self.cross_attn = CrossAttention(width, heads)
        self.ffn = Mlp(
            width, hidden_features=width, act_layer=lambda: nn.GELU(approximate="tanh"), drop=0
        )

    def forward(self, x, conditions, mask):
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), conditions, mask)
        return x + self.ffn(self.norm3(x))


class FinalLayer(nn.Module):
    def __init__(self, width, outputs):
        super().__init__()
        self.norm_final = RmsNorm(width, eps=1e-6)
        self.ffn_final = Mlp(
            width,
            hidden_features=width,
            out_features=outputs,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

    def forward(self, x):
        return self.ffn_final(self.norm_final(x))


class CoordinationDiT(nn.Module):
    def __init__(self, outputs, horizon, width, layers, heads, latent_steps=0):
        super().__init__()
        self.horizon, self.hidden_size = horizon, width
        self.t_embedder, self.freq_embedder = TimestepEmbedder(width), TimestepEmbedder(width)
        # Keep the released checkpoint's reserved positions while slicing by the
        # actual token count, rather than its hard-coded six-token forward path.
        self.x_pos_embed = nn.Parameter(multimodal_positions(width, [1, 1, 1, 1, horizon, 61]))
        if latent_steps:
            self.guess_action_cond_pos_embed = nn.Parameter(
                torch.from_numpy(
                    position_embedding(width, np.arange(horizon * latent_steps))
                ).float()[None]
            )
        self.blocks = nn.ModuleList([RDTBlock(width, heads) for _ in range(layers)])
        self.final_layer = FinalLayer(width, outputs)
        self.apply(self._initialize)
        for embedder in (self.t_embedder, self.freq_embedder):
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.final_layer.ffn_final.fc2.weight)
        nn.init.zeros_(self.final_layer.ffn_final.fc2.bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, frequency, timestep, context, conditions, masks, latent=None):
        batch = x.shape[0]
        time = self.t_embedder(timestep).unsqueeze(1).expand(batch, -1, -1)
        freq = self.freq_embedder(frequency).unsqueeze(1)
        x = torch.cat((time, freq, context, x), dim=1)
        x = x + self.x_pos_embed[:, : x.shape[1]]
        conditions, masks = list(conditions), list(masks)
        if latent is not None:
            conditions.append(latent + self.guess_action_cond_pos_embed)
            masks.append(torch.ones(latent.shape[:2], dtype=torch.bool, device=x.device))
        condition = torch.cat(conditions, dim=1)
        mask = torch.cat(masks, dim=1)
        for block in self.blocks:
            if (
                getattr(self, "gradient_checkpointing", False)
                and self.training
                and torch.is_grad_enabled()
            ):
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(block, x, condition, mask, use_reentrant=False)
            else:
                x = block(x, condition, mask)
        return self.final_layer(x)[:, -self.horizon :], x[:, -self.horizon :]


class PerceptionAdapter(nn.Module):
    def __init__(self, width, projection=256, use_pointcloud=True):
        super().__init__()

        def projector():
            return nn.Sequential(
                nn.Linear(width, projection), nn.ReLU(), nn.Linear(projection, projection)
            )

        self.image_proj, self.lang_proj = projector(), projector()
        self.pc_proj = projector() if use_pointcloud else None

    def forward(self, images, points, language, language_mask):
        # (B, history, cameras, patches, width), (B, history, tokens, width).
        valid = language_mask[..., None].to(language.dtype)
        lang = self.lang_proj((language * valid).sum(1) / valid.sum(1).clamp_min(1))
        image_global = self.image_proj(images.mean(-2))
        image_weights = (
            F.cosine_similarity(image_global.float(), lang[:, None, None].float(), dim=-1) * 0.5 + 1
        )
        weighted_images = images * image_weights[..., None, None].to(images.dtype)
        if points is None:
            return weighted_images, None, image_weights
        point_global = self.pc_proj(points.mean(-2))
        point_weights = (
            F.cosine_similarity(point_global.float(), lang[:, None].float(), dim=-1) * 0.5 + 1
        )
        return (
            weighted_images,
            points * point_weights[..., None, None].to(points.dtype),
            torch.cat((image_weights.flatten(1), point_weights), dim=1),
        )

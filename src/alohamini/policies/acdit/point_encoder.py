# Adapted from PKU-HMI-Lab/LIFT3D (MIT); see LICENSE.lift3d.
"""Lift3D tokenizer, six-view positional lifting and SigLIP q/k/v LoRA."""

import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from alohamini.datasets.pointcloud import normalize_points, resample_points
from alohamini.policies.lora import LoRALinear

from .point_tokenizer import FPS_kNN, Point_PN_scan


def view_rotations():
    angles = [(i * math.pi / 2, 0, math.pi / 2) for i in range(4)]
    angles += [(0, -math.pi / 2, math.pi / 2), (0, math.pi / 2, math.pi / 2)]
    matrices = []
    for x, y, z in angles:
        angle = torch.tensor([x, y, z], dtype=torch.float32)
        cx, cy, cz = angle.cos()
        sx, sy, sz = angle.sin()
        rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        matrices.append((rx @ ry @ rz).T)
    return torch.stack(matrices)


class Lift3DEncoder(nn.Module):
    def __init__(self, vision, config):
        super().__init__()
        self.config = config
        self.vision = deepcopy(vision).requires_grad_(False)
        width = vision.config.hidden_size
        for name, module in list(self.vision.named_modules()):
            if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in (
                "q_proj",
                "k_proj",
                "v_proj",
            ):
                parent, _, attr = name.rpartition(".")
                setattr(
                    self.vision.get_submodule(parent),
                    attr,
                    LoRALinear(module, config.lora_rank, config.lora_alpha, config.lora_dropout),
                )
        self.patch_embed = Point_PN_scan(
            input_points=config.pointcloud_points,
            num_stages=config.pointcloud_stages,
            embed_dim=config.pointcloud_embed_dim,
            k_neighbors=config.pointcloud_knn,
        )
        for module in self.patch_embed.modules():
            if isinstance(module, FPS_kNN):
                module.sampling = config.pointcloud_sampling
        self.token_dim_transform = nn.Linear(
            config.pointcloud_embed_dim * 2**config.pointcloud_stages, width
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, width))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.register_buffer("rotations", view_rotations(), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()
        for module in self.vision.modules():
            if isinstance(module, LoRALinear):
                module.train(mode)
        return self

    def positions(self, centers):
        batch, groups, _ = centers.shape
        projected = centers[:, None].float() @ self.rotations.float()[None]
        projected = projected + projected.new_tensor([0, 0, 1.6])
        size = self.config.pointcloud_projection_size
        pixels = ((projected[..., :2] / (projected[..., 2:] + 1e-12) + 1) * size / 2 - 0.5) % size
        image_size = self.vision.config.image_size
        grid = (pixels / (image_size - 1) * 2 - 1).reshape(batch * 6, groups, 1, 2)
        weights = self.vision.embeddings.position_embedding.weight
        side = image_size // self.vision.config.patch_size
        positional = weights.float().T.reshape(1, -1, side, side).expand(batch * 6, -1, -1, -1)
        lifted = F.grid_sample(
            positional, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return lifted[..., 0].transpose(1, 2).reshape(batch, 6, groups, -1).mean(1)

    def forward(self, points):
        points = normalize_points(resample_points(points, self.config.pointcloud_points))
        center, tokens = self.patch_embed(points.transpose(1, 2).contiguous(), points)
        tokens = self.token_dim_transform(tokens.transpose(1, 2))
        batch = points.shape[0]
        tokens = torch.cat((self.cls_token.expand(batch, -1, -1), tokens), dim=1)
        positions = torch.cat((self.cls_pos.expand(batch, -1, -1), self.positions(center)), dim=1)
        output = self.vision.encoder(inputs_embeds=tokens + positions).last_hidden_state
        return self.norm(output)[:, 1:]

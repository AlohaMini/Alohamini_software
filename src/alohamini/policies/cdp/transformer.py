"""Chunk-causal attention with separate clean-history projections and KV caches.

Native implementation of the CDP2 architecture described by GaavaMa/CDP.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class CachedAttention(nn.Module):
    def __init__(self, width, heads, *, history):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(width, width, bias=False)
        self.current_kv = nn.Linear(width, 2 * width, bias=False)
        self.history_kv = nn.Linear(width, 2 * width, bias=False) if history else None
        self.output = nn.Linear(width, width)

    def forward(self, query, current, *, past=None, cache=None, mask=None):
        kv = self.current_kv(current)
        if past is not None:
            kv = torch.cat((self.history_kv(past), kv), 1)
        if cache is not None:
            kv = torch.cat((cache, kv), 1)
        k, v = kv.chunk(2, -1)
        q = self.query(query)

        def split(x):
            return x.reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)

        value = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=mask)
        value = value.transpose(1, 2).reshape_as(query)
        return self.output(value), kv


class CausalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.self_attention = CachedAttention(d, config.num_heads, history=True)
        self.cross_attention = CachedAttention(d, config.num_heads, history=False)
        self.dropout = nn.Dropout(config.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.Mish(), nn.Dropout(config.dropout), nn.Linear(4 * d, d)
        )

    def forward(self, x, cond, history_size, mask, cache=None):
        old_self, old_obs = (None, None) if cache is None else cache
        norm = self.norms[0](x)
        residual, self_kv = self.self_attention(
            norm,
            norm[:, history_size:],
            past=norm[:, :history_size],
            cache=old_self,
            mask=mask,
        )
        x = x + self.dropout(residual)
        residual, obs_kv = self.cross_attention(self.norms[1](x), cond, cache=old_obs)
        x = x + self.dropout(residual)
        return x + self.mlp(self.norms[2](x)), (self_kv, obs_kv[:, :-1])


class CausalTransformer(nn.Module):
    def __init__(self, config, observation_width):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.action = nn.Linear(config.action_feature.shape[0], d)
        self.observation = nn.Linear(observation_width, d)
        self.time = nn.Sequential(
            nn.Linear(config.future_steps * d, 4 * d), nn.Mish(), nn.Linear(4 * d, d)
        )
        frequency = torch.exp(-math.log(10000) * torch.arange(d // 2).float() / (d // 2 - 1))
        self.register_buffer("time_frequency", frequency)
        phase = torch.arange(config.horizon)[:, None] * (
            10000 ** (-torch.arange(d // 2).double() / (d / 2))
        )
        self.register_buffer("position", torch.cat((phase.sin(), phase.cos()), -1).float())
        chunks = torch.arange(config.horizon) // config.n_action_steps
        past = torch.arange(config.horizon) < config.n_obs_steps
        visible = torch.where(
            past[:, None], chunks[:, None] == chunks[None], chunks[:, None] >= chunks[None]
        )
        self.register_buffer("mask", visible, persistent=False)
        self.blocks = nn.ModuleList([CausalBlock(config) for _ in range(config.num_layers)])
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, config.action_feature.shape[0]))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, actions, timesteps, observations, start, *, history_size=None, caches=None):
        cfg = self.config
        history_size = cfg.n_obs_steps if history_size is None else history_size
        offsets = torch.arange(actions.shape[1], device=actions.device)
        ids = (start[:, None] + offsets) % cfg.horizon
        x = self.action(actions) + self.position[ids]
        obs_ids = (
            start[:, None] + torch.arange(observations.shape[1], device=actions.device)
        ) % cfg.horizon
        cond = self.observation(observations) + self.position[obs_ids]
        phase = timesteps[..., None] * self.time_frequency
        time = self.time(torch.cat((phase.sin(), phase.cos()), -1).flatten(1))
        cond = torch.cat((cond, time[:, None]), 1)
        mask = self.mask[-actions.shape[1] :]
        next_cache = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, cond, history_size, mask, None if caches is None else caches[i])
            next_cache.append((cache[0][:, : cfg.n_obs_steps], cache[1]))
        return self.output(x), next_cache

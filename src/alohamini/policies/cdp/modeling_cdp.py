"""CDP2 training and cached rolling DDPM inference."""

from collections import deque

import torch
from diffusers import DDPMScheduler
from torch import nn

from alohamini.policies.resnet_encoder import ResNetEncoder

from .transformer import CausalTransformer


class CDPPolicy(nn.Module):
    name = "cdp"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision = ResNetEncoder(config, pooled=True, groups=None)
        width = self.vision.width * len(config.image_features) + config.robot_state_feature.shape[0]
        self.transformer = CausalTransformer(config, width)
        scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps, beta_schedule="squaredcos_cap_v2"
        )
        self.register_buffer("alphas_cumprod", scheduler.alphas_cumprod)
        self.reset()

    def reset(self):
        self._observations = {}
        self._actions = deque()
        self._buffer = self._times = self._cache = None
        self._position = 0

    def get_optim_params(self):
        decay, no_decay = [], []
        for parameter in self.transformer.parameters():
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        return [
            {"params": decay},
            {"params": no_decay, "weight_decay": 0.0},
            {"params": self.vision.parameters(), "weight_decay": self.config.vision_weight_decay},
        ]

    def encode(self, batch):
        return torch.cat(
            [*(x.squeeze(2) for x in self.vision(batch)), batch["observation.state"]], -1
        )

    def _alpha(self, times):
        return torch.where(times >= 0, self.alphas_cumprod[times.clamp(min=0)], 1.0)[..., None]

    def add_noise(self, actions, noise, times):
        alpha = self._alpha(times)
        return alpha.sqrt() * actions + (1 - alpha).sqrt() * noise

    def _ramp(self, batch_size, device):
        cfg = self.config
        chunks = cfg.future_steps // cfg.n_action_steps
        ids = torch.arange(cfg.future_steps, device=device) // cfg.n_action_steps
        return (((ids + 1) * cfg.num_train_timesteps // chunks) - 1).expand(batch_size, -1)

    def training_timesteps(self, batch_size, device):
        cfg = self.config
        regime = int(torch.multinomial(torch.tensor(cfg.noise_regime_weights), 1))
        n, h = cfg.num_train_timesteps, cfg.horizon
        if regime == 0:
            times = torch.randint(n, (batch_size, 1), device=device).expand(-1, h)
        elif regime == 1:
            times = torch.linspace(0, n - 1, h, device=device).long().expand(batch_size, -1)
        elif regime == 2:
            times = torch.randint(n, (batch_size, h), device=device)
        else:
            ramp = self._ramp(batch_size, device)
            offset = torch.randint(
                n // (cfg.future_steps // cfg.n_action_steps), (batch_size, 1), device=device
            )
            times = torch.cat((ramp[:, :1].expand(-1, cfg.n_obs_steps), ramp), 1) - offset
        return times[:, cfg.n_obs_steps :]

    def forward(self, batch):
        cfg = self.config
        action, past = batch["action"], batch["past_action"]
        cond = self.encode(batch)
        # The reference draws noise for history and future together.
        noise = torch.randn(len(action), cfg.horizon, action.shape[-1], device=action.device)
        times = self.training_timesteps(len(action), action.device)
        history = past.masked_fill(batch["past_action_is_pad"][..., None], 0)
        history = history + cfg.history_noise_scale * noise[:, : cfg.n_obs_steps]
        noisy = self.add_noise(action, noise[:, cfg.n_obs_steps :], times)
        start = (batch["sequence_index"] - cfg.n_obs_steps).clamp(min=0)
        predicted, _ = self.transformer(torch.cat((history, noisy), 1), times, cond, start)
        target = action if cfg.prediction_type == "sample" else noise[:, cfg.n_obs_steps :]
        error = (predicted[:, cfg.n_obs_steps :] - target).square()
        if cfg.do_mask_loss_for_padding:
            error = error * (~batch["action_is_pad"])[..., None]
        # CDP averages over the entire trajectory, including the zero-loss history prefix.
        loss = error.sum() / (len(action) * cfg.horizon * action.shape[-1])
        return loss * batch.get("_loss_weight", 1.0), {"diffusion_loss": loss.detach()}

    def _step(self, prediction, times, sample):
        alpha, previous = self._alpha(times), self._alpha(times - 1)
        current = alpha / previous
        beta = 1 - current
        clean = (
            prediction
            if self.config.prediction_type == "sample"
            else (sample - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
        )
        if self.config.clip_sample:
            clean = clean.clamp(-1, 1)
        denominator = (1 - alpha).clamp_min(1e-20)
        mean = previous.sqrt() * beta / denominator * clean
        mean = mean + current.sqrt() * (1 - previous) / denominator * sample
        variance = ((1 - previous) / denominator * beta).clamp_min(1e-20)
        value = mean + (times > 0)[..., None] * variance.sqrt() * torch.randn_like(sample)
        return torch.where((times >= 0)[..., None], value, sample)

    def _denoise(self, buffer, times, cond, start, cache, *, complete):
        cfg = self.config
        h, chunk = cfg.n_obs_steps, cfg.n_action_steps
        first = True
        limit = cfg.future_steps if complete else chunk
        # Each chunk traverses an equal number of training noise levels.
        steps = int(times[:, :limit].max()) + 1
        for _ in range(steps):
            history_size = (h if cache is None else chunk) if first else 0
            offset = h - history_size
            sample = buffer[:, offset:]
            observations = cond[:, offset:h]
            prediction, new_cache = self.transformer(
                sample,
                times,
                observations,
                start + offset,
                history_size=history_size,
                caches=cache,
            )
            if first:
                cache = new_cache
            buffer[:, h:] = self._step(prediction[:, history_size:], times, buffer[:, h:])
            times = (times - 1).clamp(min=-1)
            first = False
        return buffer, times, cache

    def _initial_buffer(self, cond, past=None, padding=None):
        cfg = self.config
        shape = (len(cond), cfg.horizon, cfg.action_feature.shape[0])
        noise = torch.randn(shape, device=cond.device, dtype=cond.dtype)
        times = self._ramp(len(cond), cond.device).clone()
        future = self.add_noise(
            torch.zeros_like(noise[:, cfg.n_obs_steps :]), noise[:, cfg.n_obs_steps :], times
        )
        history = torch.zeros_like(noise[:, : cfg.n_obs_steps]) if past is None else past
        if padding is not None:
            history = history.masked_fill(padding[..., None], 0)
        history = history + cfg.history_noise_scale * noise[:, : cfg.n_obs_steps]
        return torch.cat((history, future), 1), times

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        # Independent offline queries never change rollout buffers or read future labels.
        cond = self.encode(batch)
        buffer, times = self._initial_buffer(
            cond, batch["past_action"], batch["past_action_is_pad"]
        )
        start = (batch["sequence_index"] - self.config.n_obs_steps).clamp(min=0)
        buffer, _, _ = self._denoise(buffer, times, cond, start, None, complete=True)
        return buffer[:, self.config.n_obs_steps :]

    @torch.no_grad()
    def select_action(self, batch):
        cfg = self.config
        for key in cfg.input_features:
            queue = self._observations.setdefault(key, deque(maxlen=cfg.n_obs_steps))
            if not queue:
                queue.extend([batch[key]] * cfg.n_obs_steps)
            else:
                queue.append(batch[key])
        if not self._actions:
            history = {key: torch.stack(list(q), 1) for key, q in self._observations.items()}
            cond = self.encode(history)
            if self._buffer is None:
                self._buffer, self._times = self._initial_buffer(cond)
            else:
                self._cache = [
                    (a[:, cfg.n_action_steps :], o[:, cfg.n_action_steps :]) for a, o in self._cache
                ]
            start = torch.full((len(cond),), self._position, device=cond.device, dtype=torch.long)
            result, times, self._cache = self._denoise(
                self._buffer,
                self._times,
                cond,
                start,
                self._cache,
                complete=False,
            )
            h, chunk = cfg.n_obs_steps, cfg.n_action_steps
            self._actions.extend(result[:, h : h + chunk].clone().unbind(1))
            fresh = torch.randn_like(result[:, :chunk])
            self._buffer = torch.cat((result[:, chunk:], fresh), 1)
            self._times = torch.cat(
                (times[:, chunk:], torch.full_like(times[:, :chunk], cfg.num_train_timesteps - 1)),
                1,
            )
            self._position += chunk
        return self._actions.popleft()

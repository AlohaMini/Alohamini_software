"""Native DiT-Block policy with layer-wise observation conditioning and DDIM."""

from collections import deque

import torch
from diffusers import DDIMScheduler
from torch import nn

from alohamini.policies.resnet_encoder import ResNetEncoder

from .transformer import _DiTNoiseNet


class DiTPolicy(nn.Module):
    name = "dit_policy"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision = ResNetEncoder(config, pooled=config.pooled_vision, groups=16)
        self.state_token = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(config.robot_state_feature.shape[0], self.vision.width)
        )
        self.token_projection = nn.Sequential(
            nn.Linear(self.vision.width, config.hidden_dim)
            if self.vision.width != config.hidden_dim
            else nn.Identity(),
            nn.Dropout(config.dropout),
        )
        self.noise_net = _DiTNoiseNet(
            config.action_feature.shape[0],
            config.horizon,
            time_dim=config.time_dim,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_layers,
            dropout=config.dropout,
            dim_feedforward=config.feedforward_dim,
            nhead=config.num_heads,
        )
        self.scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=config.clip_sample,
            prediction_type="epsilon",
            set_alpha_to_one=True,
        )
        self.reset()

    def reset(self):
        self._observations = {}
        self._actions = deque()

    def get_optim_params(self):
        return self.parameters()

    def encode(self, batch):
        # The reference replay buffer orders images newest first within each camera.
        tokens = [x.flip(1).flatten(1, 2) for x in self.vision(batch)]
        state = batch["observation.state"]
        if state.ndim == 3:
            state = state[:, -1]
        tokens.append(self.state_token(state)[:, None])
        return self.token_projection(torch.cat(tokens, dim=1))

    def forward(self, batch):
        action = batch["action"]
        time = torch.randint(self.config.num_train_timesteps, (len(action),), device=action.device)
        noise = torch.randn_like(action)
        noisy = self.scheduler.add_noise(action, noise, time)
        _, predicted = self.noise_net(noisy, time, self.encode(batch))
        mask = ~batch["action_is_pad"]
        # Reference objective sums time, then averages over batch and action dimensions.
        loss = ((predicted - noise).square() * mask[..., None]).sum(1).mean()
        return loss * batch.get("_loss_weight", 1.0), {"diffusion_loss": loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        cond = self.noise_net.forward_enc(self.encode(batch))
        action = torch.randn(
            len(cond[0][0]),
            self.config.horizon,
            self.config.action_feature.shape[0],
            device=cond[0].device,
            dtype=cond[0].dtype,
        )
        self.scheduler.set_timesteps(self.config.num_inference_steps, device=action.device)
        for time in self.scheduler.timesteps:
            predicted = self.noise_net.forward_dec(action, time.expand(len(action)), cond)
            action = self.scheduler.step(predicted, time, action).prev_sample
        return action

    @torch.no_grad()
    def select_action(self, batch):
        for key in self.config.input_features:
            queue = self._observations.setdefault(key, deque(maxlen=self.config.n_obs_steps))
            if not queue:
                queue.extend([batch[key]] * self.config.n_obs_steps)
            else:
                queue.append(batch[key])
        if not self._actions:
            history = {key: torch.stack(list(q), 1) for key, q in self._observations.items()}
            action = self.predict_action_chunk(history)[:, : self.config.n_action_steps]
            self._actions.extend(action.unbind(1))
        return self._actions.popleft()

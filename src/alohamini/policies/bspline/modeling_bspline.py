"""Diffusion over BSP parameters, retaining a separate physical action contract."""

from copy import deepcopy

import torch
from torch import nn

from alohamini.policies.configuration import PolicyFeature
from alohamini.policies.diffusion.modeling_diffusion import DiffusionPolicy


class BSplinePolicy(nn.Module):
    name = "bspline"

    def __init__(self, config):
        super().__init__()
        self.config = config
        network = deepcopy(config)
        network.output_features["action"] = PolicyFeature(
            "ACTION", (config.action_feature.shape[0] + 1,)
        )
        self.generator = DiffusionPolicy(network)

    def reset(self):
        self.generator.reset()

    def get_optim_params(self):
        return self.generator.get_optim_params()

    def forward(self, batch):
        target = batch["trajectory"]
        return self.generator(
            {
                **batch,
                "action": target,
                "action_is_pad": torch.zeros(
                    target.shape[:2], dtype=torch.bool, device=target.device
                ),
            }
        )

    @torch.no_grad()
    def predict_parameters(self, batch):
        return self.generator.predict_action_chunk(batch)

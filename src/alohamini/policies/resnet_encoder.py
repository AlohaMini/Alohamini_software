"""ResNet spatial tokens and pooled features for native visuomotor policies."""

from copy import deepcopy

import torch
from torch import nn
from torchvision import models, transforms


class ResNetEncoder(nn.Module):
    def __init__(self, config, *, pooled, groups):
        super().__init__()
        backbone = models.get_model(
            config.vision_backbone, weights=config.pretrained_backbone_weights
        )
        self.width = backbone.fc.in_features

        def replace(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    count = groups if groups else child.num_features // 16
                    norm = nn.GroupNorm(count, child.num_features)
                    with torch.no_grad():
                        norm.weight.copy_(child.weight)
                        norm.bias.copy_(child.bias)
                    setattr(module, name, norm)
                else:
                    replace(child)

        replace(backbone)
        trunk = nn.Sequential(*list(backbone.children())[:-2])
        self.keys = list(config.image_features)
        self.shared = config.share_rgb_encoder
        self.encoders = nn.ModuleList(
            [trunk] + [deepcopy(trunk) for _ in range(0 if self.shared else len(self.keys) - 1)]
        )
        self.pooled = pooled
        self.crop_shape = config.crop_shape
        self.resized_crop = getattr(config, "random_resized_crop", False)
        self.blur = getattr(config, "gaussian_blur", False)

    def forward(self, batch):
        result = []
        for index, key in enumerate(self.keys):
            image = batch[key]
            if image.ndim == 4:
                image = image[:, None]
            b, t = image.shape[:2]
            image = image.flatten(0, 1)
            if self.training and self.resized_crop:
                # Dataset transforms in dit-policy draw independently per example,
                # sharing the augmentation across that example's history frames.
                views = []
                for view in image.unflatten(0, (b, t)):
                    if self.crop_shape:
                        view = transforms.RandomResizedCrop(
                            self.crop_shape,
                            scale=(0.9, 1.0),
                            antialias=False,
                        )(view)
                    if self.blur:
                        size = int(0.05 * view.shape[-1]) // 2 * 2 + 1
                        view = transforms.GaussianBlur(size)(view)
                    views.append(view)
                image = torch.stack(views).flatten(0, 1)
            if self.crop_shape:
                if self.resized_crop and not self.training:
                    image = transforms.Resize(self.crop_shape, antialias=False)(image)
                elif not self.resized_crop and self.training:
                    # CDP samples a separate crop for every encoded observation.
                    height, width = self.crop_shape
                    y = torch.randint(image.shape[-2] - height + 1, (b * t,), device=image.device)
                    x = torch.randint(image.shape[-1] - width + 1, (b * t,), device=image.device)
                    rows = (
                        y[:, None, None] + torch.arange(height, device=image.device)[None, :, None]
                    )
                    cols = (
                        x[:, None, None] + torch.arange(width, device=image.device)[None, None, :]
                    )
                    image = image.permute(0, 2, 3, 1)[
                        torch.arange(b * t, device=image.device)[:, None, None],
                        rows,
                        cols,
                    ].permute(0, 3, 1, 2)
                elif not self.resized_crop:
                    image = transforms.CenterCrop(self.crop_shape)(image)
            if self.training and self.blur and not self.resized_crop:
                size = int(0.05 * image.shape[-1]) // 2 * 2 + 1
                image = transforms.GaussianBlur(size)(image)
            features = self.encoders[0 if self.shared else index](image)
            if self.pooled:
                features = features.mean((-2, -1), keepdim=True)
            result.append(features.flatten(2).transpose(1, 2).reshape(b, t, -1, self.width))
        return result

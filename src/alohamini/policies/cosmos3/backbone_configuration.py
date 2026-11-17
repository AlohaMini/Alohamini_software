# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# ruff: noqa: E501
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Published Cosmos3 MoT and Wan geometry, independent of trainer configuration."""

from dataclasses import dataclass, field
from typing import Literal

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig

# Per-channel mean and std of the Wan2.2 latents (`vae/config.json` of the released checkpoints).
WAN2PT2_LATENTS_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813, 0.0891,
    0.157, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123, -0.0313, -0.1649, 0.0117, 0.0723,
    -0.2839, -0.2083, -0.052, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]  # fmt: skip
WAN2PT2_LATENTS_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
    0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]  # fmt: skip

TimeDistribution = Literal["uniform", "logitnormal", "waver"]
# Values of `Cosmos3Config.mode` the policy supports.
SUPPORTED_MODES = ("wam",)
# Resolution tiers and the largest shorter side of each (native `get_vision_data_resolution`).
RESOLUTION_TIER_LIMITS = (("256", 256), ("480", 640), ("720", 960))


@strict
class Qwen3VLMoTVisionConfig(Qwen3VLVisionConfig):
    """Qwen3-VL vision config of the Cosmos3 MoT backbone, defaults from ``nvidia/Cosmos3-Nano``."""

    out_hidden_size: int = 4096


@strict
class Qwen3VLMoTTextConfig(Qwen3VLTextConfig):
    """
    Qwen3-VL text config of the Cosmos3 MoT backbone (Cosmos3 Nano and Super). The fields set here are the
    ``text_config`` of ``nvidia/Cosmos3-Nano`` where it differs from Qwen3-VL.
    """

    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_key_value_heads: int = 8
    max_position_embeddings: int = 262144
    rope_parameters: dict | None = field(
        default_factory=lambda: {
            "rope_type": "default",
            "rope_theta": 5_000_000,
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
        }
    )
    # Joint attention over the understanding and generation pathways. Only "two_way" is supported.
    joint_attn_implementation: str = "two_way"

    def validate_architecture(self):
        super().validate_architecture()
        if self.joint_attn_implementation != "two_way":
            raise NotImplementedError(
                f"joint_attn_implementation={self.joint_attn_implementation!r} is not supported, only 'two_way'."
            )


@strict
class Qwen3VLMoTConfig(Qwen3VLConfig):
    """Config of the Qwen3-VL MoT backbone (Cosmos3 Nano and Super), defaults from ``nvidia/Cosmos3-Nano``."""

    sub_configs_defaults = {
        "vision_config": Qwen3VLMoTVisionConfig,
        "text_config": Qwen3VLMoTTextConfig,
    }


@strict
class Nemotron3DenseVLMoTTextConfig(PretrainedConfig):
    """Text config of the Nemotron 3 Dense VL MoT backbone, defaults from ``nvidia/Cosmos3-Edge``."""

    model_type = "nemotron_3_dense_vl_text"
    base_config_key = "text_config"
    default_theta = 100_000_000.0
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    vocab_size: int = 131072
    hidden_size: int = 2048
    intermediate_size: int = 9216
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "relu2"
    mlp_bias: bool = False
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    rope_parameters: dict | None = field(
        default_factory=lambda: {
            "rope_type": "default",
            "rope_theta": 100_000_000,
            "mrope_section": [24, 20, 20],
        }
    )
    # Padding index of the token embedding.
    pad_token_id: int | None = 0
    # Joint attention over the understanding and generation pathways. Only "two_way" is supported.
    joint_attn_implementation: str = "two_way"

    def validate_architecture(self):
        super().validate_architecture()
        if self.joint_attn_implementation != "two_way":
            raise NotImplementedError(
                f"joint_attn_implementation={self.joint_attn_implementation!r} is not supported, only 'two_way'."
            )


@strict
class Nemotron3DenseVLVisionConfig(Siglip2VisionConfig):
    """SigLIP2 vision config of the Nemotron 3 Dense VL backbone, defaults from ``nvidia/Cosmos3-Edge``."""

    model_type = "nemotron_3_dense_vl_vision"

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    # Side of the square patch blocks the projector merges into one token.
    spatial_merge_size: int = 2


@strict
class Nemotron3DenseVLProjectorConfig(PretrainedConfig):
    """Patch merger config of the Nemotron 3 Dense VL backbone, defaults from ``nvidia/Cosmos3-Edge``."""

    model_type = "nemotron_3_dense_vl_projector"
    base_config_key = "projector_config"

    input_hidden_size: int = 1152
    merger_intermediate_size: int = 11520
    out_hidden_size: int = 2048
    spatial_merge_size: int = 2
    use_postshuffle_norm: bool = False


@strict
class Nemotron3DenseVLMoTConfig(PretrainedConfig):
    """Config of the Nemotron 3 Dense VL MoT backbone (Cosmos3 Edge), defaults from ``nvidia/Cosmos3-Edge``."""

    model_type = "nemotron_3_dense_vl"
    sub_configs_defaults = {
        "vision_config": Nemotron3DenseVLVisionConfig,
        "projector_config": Nemotron3DenseVLProjectorConfig,
        "text_config": Nemotron3DenseVLMoTTextConfig,
    }

    text_config: dict | PretrainedConfig | None = None
    vision_config: dict | PretrainedConfig | None = None
    projector_config: dict | PretrainedConfig | None = None
    image_token_id: int = 19
    video_token_id: int = 18
    vision_start_token_id: int = 20
    vision_end_token_id: int = 21
    tie_word_embeddings: bool = False

    def validate_architecture(self):
        super().validate_architecture()
        if (
            isinstance(self.vision_config, Nemotron3DenseVLVisionConfig)
            and isinstance(self.projector_config, Nemotron3DenseVLProjectorConfig)
            and self.vision_config.spatial_merge_size != self.projector_config.spatial_merge_size
        ):
            raise ValueError(
                f"vision_config.spatial_merge_size ({self.vision_config.spatial_merge_size}) must match "
                f"projector_config.spatial_merge_size ({self.projector_config.spatial_merge_size})."
            )


@dataclass
class Wan2pt2VAEConfig:
    """
    Config of the Wan2.2 VAE (`diffusers.AutoencoderKLWan`) of the released Cosmos3 checkpoints: 48 latent
    channels, 16x spatial and 4x temporal compression. The fields are the arguments of
    `AutoencoderKLWan.__init__`, the defaults the `vae/config.json` of the released checkpoints.

    Args:
        base_dim: Channels of the first encoder stage.
        decoder_base_dim: Channels of the first decoder stage.
        z_dim: Latent channels.
        dim_mult: Channel multiplier of every stage.
        num_res_blocks: Residual blocks per stage.
        attn_scales: Scales with self attention.
        temperal_downsample: Whether every downsampling stage also halves the frames.
        dropout: Dropout of the residual blocks.
        latents_mean: Mean of every latent channel, `z_dim` values.
        latents_std: Standard deviation of every latent channel, `z_dim` values.
        is_residual: Residual down and up sampling blocks of the Wan2.2 VAE.
        in_channels: Input channels after the pixel patchify, 3 colors times `patch_size**2`.
        out_channels: Output channels before the pixel unpatchify.
        patch_size: Pixel patch size of the input patchify.
        scale_factor_temporal: Temporal compression from frames to latents.
        scale_factor_spatial: Spatial compression from pixels to latents.
    """

    base_dim: int = 160
    decoder_base_dim: int | None = 256
    z_dim: int = 48
    dim_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    attn_scales: list[float] = field(default_factory=list)
    temperal_downsample: list[bool] = field(default_factory=lambda: [False, True, True])
    dropout: float = 0.0
    latents_mean: list[float] = field(default_factory=lambda: list(WAN2PT2_LATENTS_MEAN))
    latents_std: list[float] = field(default_factory=lambda: list(WAN2PT2_LATENTS_STD))
    is_residual: bool = True
    in_channels: int = 12
    out_channels: int = 12
    patch_size: int | None = 2
    scale_factor_temporal: int | None = 4
    scale_factor_spatial: int | None = 16

    def __post_init__(self):
        if len(self.latents_mean) != self.z_dim or len(self.latents_std) != self.z_dim:
            raise ValueError(
                f"latents_mean and latents_std need z_dim={self.z_dim} values, got {len(self.latents_mean)} "
                f"and {len(self.latents_std)}."
            )

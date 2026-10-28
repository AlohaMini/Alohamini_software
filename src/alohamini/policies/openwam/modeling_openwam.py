"""OpenWAM-alpha tensor path, using the platform's optimizer and distributed runtime."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from alohamini.policies.chunk import ChunkPolicy

from ._vendor.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from ._vendor.model.architectures.utils.common import downsample_video_mask_to_latent
from ._vendor.model.video_backbone.wan.models.dit import WanModel
from ._vendor.model.video_backbone.wan.models.text_encoder import (
    HuggingfaceTokenizer,
    WanTextEncoder,
)
from ._vendor.model.video_backbone.wan.models.vae import WanVideoVAE38
from ._vendor.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
from ._vendor.model.video_backbone.wan_backbone import Wan22Ti2v


class AlphaArchitecture(DualSystemSelfAttnArchitecture):
    def _init_video_backbone(self, cfg):
        spec = cfg["native_config"]
        holder = SimpleNamespace(
            dit=WanModel(**spec.wan),
            vae=WanVideoVAE38(**spec.vae),
            text_encoder=WanTextEncoder(**spec.text_encoder),
            tokenizer=HuggingfaceTokenizer(
                spec.assets_path,
                seq_len=512,
                clean="whitespace",
                local_files_only=True,
                trust_remote_code=False,
            ),
            scheduler=FlowMatchScheduler("Wan"),
            height_division_factor=32,
            width_division_factor=32,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.video_backbone = Wan22Ti2v(holder, shift_video=spec.shift)


class OpenWAMPolicy(ChunkPolicy):
    name = "openwam"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.network = AlphaArchitecture(
            dict(
                native_config=config,
                dim=config.action_hidden_dim,
                ffn_dim=config.action_ffn_dim,
                action_dim=config.action_feature.shape[0],
                state_dim=config.robot_state_feature.shape[0],
                use_proprioception=True,
                bridge_interval=1,
                shift_action=config.shift,
                attention_mask_mode="mutual",
                video_attention_mask_mode="first_frame_causal",
                mot_checkpoint_mixed_attn=config.gradient_checkpointing,
            )
        )
        self.network.to(dtype=torch.bfloat16)
        for module in (self.network.video_backbone.vae, self.network.video_backbone.text_encoder):
            module.requires_grad_(False).eval()
        self.network.init_training_schedulers()
        self.reset()

    def train(self, mode=True):
        super().train(mode)
        self.network.video_backbone.vae.eval()
        self.network.video_backbone.text_encoder.eval()
        return self

    def _runtime(self):
        ref = next(self.parameters())
        # Forward-time metadata only: FSDP owns parameter movement and materialization.
        self.network._device, self.network._dtype = ref.device, ref.dtype
        for backbone in self.network.backbones.values():
            backbone._device, backbone._dtype = ref.device, ref.dtype
        return ref

    def _canvas(self, batch):
        height, width = self.config.canvas_size
        frames = next(batch[key] for key in self.config.camera_layout if key is not None)
        if frames.ndim == 4:
            frames = frames[:, None]
        b, t = frames.shape[:2]
        canvas = frames.new_zeros(b, t, 3, height, width)
        split = height * 2 // 3
        slots = [
            (0, split, 0, width),
            (split, height, 0, width // 2),
            (split, height, width // 2, width),
        ]
        for key, (top, bottom, left, right) in zip(self.config.camera_layout, slots, strict=True):
            if key is None:
                continue
            value = batch[key]
            if value.ndim == 4:
                value = value[:, None]
            if value.shape[:2] != (b, t):
                raise ValueError("All OpenWAM cameras must share a temporal window")
            value = F.interpolate(
                value.flatten(0, 1),
                (bottom - top, right - left),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            canvas[..., top:bottom, left:right] = value.reshape(b, t, 3, bottom - top, right - left)
        return canvas.permute(0, 2, 1, 3, 4).mul(2).sub(1)

    @torch.no_grad()
    def _encode(self, batch):
        ref = self._runtime()
        vb = self.network.video_backbone
        pixels = self._canvas(batch).to(dtype=ref.dtype)
        latents = vb.vae.batch_encode(pixels, device=ref.device)
        ids, mask = vb._tokenizer(batch["task"], return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(ref.device), mask.to(ref.device)
        context = vb.text_encoder(ids, mask).masked_fill(~mask.bool()[..., None], 0)
        return dict(
            input_latents=latents,
            context=context,
            seq_lens=mask.sum(-1),
            first_frame_latents=latents[:, :, :1],
            fuse_vae_embedding_in_latents=True,
        )

    def forward(self, batch):
        inputs = self._encode(batch)
        pad = torch.stack(
            [batch[key + "_is_pad"] for key in self.config.camera_layout if key is not None]
        ).any(0)
        inputs.update(
            actions=batch["action"],
            proprio=batch["observation.state"],
            action_is_pad=batch["action_is_pad"],
            video_is_pad=downsample_video_mask_to_latent(pad, skip_first=True),
            use_gradient_checkpointing=self.config.gradient_checkpointing,
        )
        output = self.network.compute_loss(
            **inputs, lambda_video=self.config.lambda_video, lambda_action=self.config.lambda_action
        )
        return output["loss"] * batch.get("_loss_weight", 1.0), {
            key: output[key] for key in ("loss_video", "loss_action")
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        # Only current observations condition inference. Never encode future supervision.
        batch = dict(batch)
        for key in self.config.image_features:
            if batch[key].ndim == 5:
                batch[key] = batch[key][:, 0]
            elif batch[key].ndim != 4:
                raise ValueError("OpenWAM requires BCHW images or a training video window")
        inputs = self._encode(batch)
        first = inputs.pop("input_latents")
        b, c, _, h, w = first.shape
        nlatent = 1 + self.config.chunk_size // (4 * self.config.video_stride)
        video = torch.randn(b, c, nlatent, h, w, dtype=first.dtype, device=first.device)
        action = torch.randn(
            b,
            self.config.chunk_size,
            self.config.action_feature.shape[0],
            dtype=first.dtype,
            device=first.device,
        )
        sigma = torch.linspace(1, 0, self.config.num_steps + 1, device=first.device)
        sigma = self.config.shift * sigma / (1 + (self.config.shift - 1) * sigma)
        for now, after in zip(sigma[:-1], sigma[1:], strict=True):
            video[:, :, :1] = first
            t = (now * 1000).expand(b).to(first.dtype)
            v_video, v_action = self.network(
                action,
                t,
                proprio=batch["observation.state"].to(first.dtype),
                latents=video,
                timestep=t,
                **inputs,
            )
            video = video + (after - now) * v_video
            action = action + (after - now) * v_action
        return action.float()

    def save_assets(self, path):
        self.network.video_backbone._tokenizer.tokenizer.save_pretrained(path)

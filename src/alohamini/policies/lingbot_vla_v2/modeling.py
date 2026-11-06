"""LingBot's upstream flow/MoE/distillation core under the shared trainer."""

from copy import deepcopy
from pathlib import Path

import torch

from alohamini.policies.chunk import ChunkPolicy


class LingBotPolicy(ChunkPolicy):
    name = "lingbot_vla_v2"

    def __init__(self, config):
        super().__init__()
        from ._vendor.model import FlowMatchingV2

        self.config = config
        self.model = FlowMatchingV2(config)
        if not config.use_lm_head:
            del self.model.qwenvl_with_expert.qwenvl.lm_head
        del self.model.qwenvl_with_expert.qwen_expert.lm_head
        self.model.to(getattr(torch, config.dtype))
        self._teachers = None
        self.reset()

    def get_optim_params(self):
        base, expert = [], []
        for name, p in self.named_parameters():
            if p.requires_grad:
                (expert if ".mlp.experts." in name else base).append(p)
        groups = [{"params": base}] if base else []
        if expert:
            scale = (
                (self.config.token_num_experts / self.config.token_top_k) ** 0.5
                if self.config.use_moe_expert_lr
                else 1
            )
            groups.append({"params": expert, "lr_scale": scale})
        return groups

    def _inputs(self, batch):
        dtype = next(self.parameters()).dtype
        return (
            batch["images"].to(dtype),
            batch["img_masks"],
            batch["observation.language.tokens"],
            batch["observation.language.attention_mask"],
            batch["observation.state"].to(dtype),
        )

    def forward(self, batch):
        targets = None
        if self.model.use_align:
            from ._vendor.model import DistillationTeachers

            if "teacher_images" not in batch:
                raise ValueError("LingBot training/validation losses require teacher frames")
            if self._teachers is None:
                params = deepcopy(self.config.align_params)
                root = Path(self.config.teacher_root).expanduser()
                for section, key, relative in (
                    ("depth", "moge_path", "moge/model.pt"),
                    ("depth", "morgbd_path", "depth/model.pt"),
                    ("video", "ckpt_path", "dino_video/teacher_step_10000.pth"),
                    ("video", "config_path", "dino_video/config.yaml"),
                ):
                    if section in params:
                        params[section][key] = str(root / relative)
                self._teachers = DistillationTeachers(params, next(self.parameters()).device)
            frames = batch["teacher_images"]
            fps = self.config.dataset_fps / max(1, self.config.chunk_size - 1)
            targets = self._teachers.targets(frames[:, 0], frames[:, -1], fps)
        losses, balance, router, _, align = self.model(
            *self._inputs(batch),
            batch["action"].to(next(self.parameters()).dtype),
            image_grid_thw=batch["image_grid_thw"],
            noise=batch.get("noise"),
            time=batch.get("time"),
            collect_metrics=False,
            align_targets=targets,
        )
        mask = batch["joint_mask"]
        flow = (losses * mask).sum() / mask.sum().clamp_min(1)
        loss = flow * batch.get("_loss_weight", 1)
        metrics = {"flow_loss": flow.detach()}
        aux = batch.get("_aux_loss_weight", 1)
        for key, term in (("seq_wise_loss", balance), ("router_z_loss", router)):
            term = torch.as_tensor(term, device=flow.device)
            loss = loss + term * aux
            metrics[key] = term.detach()
        for name, (weighted, value) in align.items():
            loss = loss + weighted * aux
            metrics[name] = value.detach()
        return loss, metrics

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        return self.model.sample_actions(
            *self._inputs(batch), noise=batch.get("noise"), image_grid_thw=batch["image_grid_thw"]
        )

    def save_assets(self, path):
        from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer

        source = Path(self.config.tokenizer_path)
        AutoConfig.from_pretrained(source, local_files_only=True).save_pretrained(path)
        AutoImageProcessor.from_pretrained(
            self.config.image_processor_path or source, local_files_only=True
        ).save_pretrained(path)
        AutoTokenizer.from_pretrained(
            self.config.text_tokenizer_path or source, local_files_only=True
        ).save_pretrained(path)

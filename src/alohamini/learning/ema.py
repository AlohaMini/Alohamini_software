"""Trainer-owned EMA for replicated or FSDP2 parameters."""

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class EMAConfig:
    enable: bool = False
    decay: float | None = None
    update_after_step: int = 0
    inv_gamma: float = 1.0
    power: float = 0.75
    min_decay: float = 0.0
    max_decay: float = 0.9999
    use_for_eval: bool = True

    def __post_init__(self):
        for name in ("enable", "use_for_eval"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"ema.{name} must be a boolean")
        if type(self.update_after_step) is not int or self.update_after_step < 0:
            raise ValueError("ema.update_after_step must be a nonnegative integer")
        for name in ("inv_gamma", "power", "min_decay", "max_decay", "decay"):
            value = getattr(self, name)
            if name == "decay" and value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"ema.{name} must be finite")
        if self.inv_gamma <= 0 or self.power <= 0:
            raise ValueError("ema.inv_gamma and ema.power must be positive")
        if not 0 <= self.min_decay <= self.max_decay < 1:
            raise ValueError("Expected 0 <= ema.min_decay <= ema.max_decay < 1")
        if self.decay is not None and not 0 <= self.decay < 1:
            raise ValueError("Expected 0 <= ema.decay < 1")
        if self.decay is not None and (self.min_decay != 0 or self.max_decay != 0.9999):
            raise ValueError("Use ema.decay or ema.min_decay/max_decay, not both")

    @classmethod
    def from_dict(cls, settings=None):
        if isinstance(settings, cls):
            return settings
        if settings is not None and not isinstance(settings, dict):
            raise ValueError("ema must be a configuration object")
        try:
            return cls(**(settings or {}))
        except TypeError as exc:
            raise ValueError(f"Invalid EMA configuration: {exc}") from exc


def validate_ema_setup(settings):
    config = EMAConfig.from_dict(settings.get("ema"))
    return config


def _tensors(model):
    # Frozen base weights (e.g. VLM/VAE) need no second copy. Use names, not zip order,
    # so a changed trainable set cannot silently load the wrong shadow tensors.
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    persistent = model.state_dict().keys()
    buffers = {name: b for name, b in model.named_buffers() if name in persistent}
    return params, buffers


class EMAManager:
    """FP32 parameter shadows with copied persistent buffers, including BatchNorm.

    Construct after preparation on rank zero (DDP) or every rank (FSDP2).
    DTensor shadows retain the parameter's mesh/placement and store only its shard.
    Update once per *successful*
    optimizer step. Buffers are copied, not averaged; no BN recalibration is implied.
    Frozen parameters remain in the live model and are included normally on export.
    """

    def __init__(self, model, config):
        self.config = EMAConfig.from_dict(config)
        if not self.config.enable:
            raise ValueError("Cannot construct EMA with ema.enable=false")
        params, buffers = _tensors(model)
        if not params:
            raise ValueError("EMA requires trainable parameters")
        self.sharded = any(hasattr(p, "placements") for p in params.values())
        self.parameters = {name: p.detach().float().clone() for name, p in params.items()}
        self.buffers = {name: b.detach().clone() for name, b in buffers.items()}
        self.num_updates = 0
        self._active = False

    def _model_tensors(self, model):
        params, buffers = _tensors(model)
        for live, shadow in ((params, self.parameters), (buffers, self.buffers)):
            if live.keys() != shadow.keys() or any(
                live[name].shape != shadow[name].shape for name in shadow
            ):
                raise ValueError("EMA model parameter/buffer names or shapes changed")
        return params, buffers

    def get_decay(self, updates):
        cfg = self.config
        if updates <= cfg.update_after_step:
            return 0.0
        if cfg.decay is not None:
            return cfg.decay
        step = updates - cfg.update_after_step - 1
        if step <= 0:
            return 0.0
        return min(cfg.max_decay, max(cfg.min_decay, 1 - (1 + step / cfg.inv_gamma) ** -cfg.power))

    @torch.no_grad()
    def update(self, model):
        if self._active:
            raise RuntimeError("Cannot update EMA while its weights are applied")
        params, buffers = self._model_tensors(model)
        decay = self.get_decay(self.num_updates + 1)
        for name, parameter in params.items():
            self.parameters[name].lerp_(parameter.detach().float(), 1 - decay)
        for name, buffer in buffers.items():
            self.buffers[name].copy_(buffer)
        self.num_updates += 1

    @contextmanager
    def average_parameters(self, model):
        """Temporarily apply EMA for eval/export; restore live tensors even on failure."""
        if self._active:
            raise RuntimeError("EMA weight contexts cannot be nested")

        def reshard():
            if self.sharded:
                from torch.distributed.fsdp import FSDPModule

                for module in model.modules():
                    if isinstance(module, FSDPModule):
                        module.reshard()

        reshard()
        params, buffers = self._model_tensors(model)
        pairs = [(params, self.parameters), (buffers, self.buffers)]
        backups = [
            {name: value.detach().clone() for name, value in live.items()} for live, _ in pairs
        ]
        modes = {module: module.training for module in model.modules()}
        self._active = True
        try:
            with torch.no_grad():
                for live, shadow in pairs:
                    for name, value in live.items():
                        value.copy_(shadow[name])
            model.eval()
            yield
        finally:
            # Inference can retain gathered root weights. Invalidate them before
            # restoring local shards so the next forward cannot reuse EMA weights.
            reshard()
            with torch.no_grad():
                for (live, _), backup in zip(pairs, backups, strict=True):
                    for name, value in live.items():
                        value.copy_(backup[name])
            for module, training in modes.items():
                module.training = training
            self._active = False

    def state_dict(self):
        def snapshot(value):
            # FSDP2 state is serialized collectively by distributed.checkpoint.
            return value.detach().clone() if self.sharded else value.detach().cpu().clone()

        return {
            "version": 1,
            "config": asdict(self.config),
            "num_updates": self.num_updates,
            "buffer_policy": "copy",
            "parameters": {name: snapshot(p) for name, p in self.parameters.items()},
            "buffers": {name: snapshot(b) for name, b in self.buffers.items()},
        }

    def load_state_dict(self, state):
        if self._active:
            raise RuntimeError("Cannot load EMA while its weights are applied")
        if state.get("version") != 1 or state.get("buffer_policy") != "copy":
            raise ValueError("Unsupported EMA state format")
        if state.get("config") != asdict(self.config):
            raise ValueError("EMA configuration changed since checkpoint")
        updates = state.get("num_updates")
        if type(updates) is not int or updates < 0:
            raise ValueError("Invalid EMA update count")
        for key, shadow in (("parameters", self.parameters), ("buffers", self.buffers)):
            saved = state.get(key, {})
            if saved.keys() != shadow.keys() or any(
                not isinstance(saved[name], torch.Tensor)
                or saved[name].shape != shadow[name].shape
                or saved[name].dtype != shadow[name].dtype
                for name in shadow
            ):
                raise ValueError(f"EMA {key} names, shapes or dtypes differ from checkpoint")
        # Validate everything before mutating any shadow.
        for key, shadow in (("parameters", self.parameters), ("buffers", self.buffers)):
            for name, value in shadow.items():
                value.copy_(state[key][name].to(value.device))
        self.num_updates = updates

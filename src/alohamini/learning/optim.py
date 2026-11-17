"""Resolved optimizer/scheduler recipes, independent of the training backend."""

import math
from functools import partial

import torch


def resolve_optimization(settings, policy_config):
    """Policy defaults < explicit training overrides; preserve backbone parameter groups."""
    cfg = dict(settings)
    supplied = dict(cfg.get("optimizer", {}))
    kind = supplied.get("type", "adamw")
    if kind not in ("adam", "adamw", "sgd"):
        raise ValueError("optimizer.type must be adam, adamw or sgd")
    optimizer = dict(
        type=kind,
        lr=policy_config.optimizer_lr,
        weight_decay=policy_config.optimizer_weight_decay,
        grad_clip_norm=cfg.get(
            "grad_clip_norm", getattr(policy_config, "optimizer_grad_clip_norm", 10.0)
        ),
    )
    if kind == "sgd":
        optimizer.update(momentum=0.0, dampening=0.0, nesterov=False)
    else:
        optimizer.update(
            betas=list(getattr(policy_config, "optimizer_betas", (0.9, 0.999))),
            eps=getattr(policy_config, "optimizer_eps", 1e-8),
        )
    if supplied.keys() - optimizer.keys():
        raise ValueError(
            f"Unsupported {kind} options: {sorted(supplied.keys() - optimizer.keys())}"
        )
    optimizer.update(supplied)
    for key in ("lr", "weight_decay", "grad_clip_norm"):
        if not math.isfinite(optimizer[key]) or optimizer[key] < 0:
            raise ValueError(f"optimizer.{key} must be finite and nonnegative")
    cfg["optimizer"] = optimizer
    preset = (
        dict(
            type=getattr(policy_config, "scheduler_type", "cosine"),
            warmup_steps=policy_config.scheduler_warmup_steps,
            decay_steps=policy_config.scheduler_decay_steps,
            decay_lr=policy_config.scheduler_decay_lr,
        )
        if hasattr(policy_config, "scheduler_decay_steps")
        else dict(type="none")
    )
    if getattr(policy_config, "scheduler_type", None) == "staged_cosine":
        preset = dict(
            type="staged_cosine",
            start_step=policy_config.scheduler_start_step,
            warmup_steps=policy_config.scheduler_warmup_steps,
            decay_steps=cfg["steps"],
            decay_lr=0.0,
        )
    supplied_scheduler = cfg.get("scheduler", {})
    if supplied_scheduler is None or supplied_scheduler.get("type") == "none":
        scheduler = {"type": "none"}
    else:
        scheduler = {**preset, **supplied_scheduler}
    kind = scheduler.get("type", "none")
    allowed = {"type"} if kind == "none" else {"type", "warmup_steps", "decay_steps", "decay_lr"}
    if kind in ("staged_cosine", "constant_with_warmup", "warmup_linear"):
        allowed.update(("start_step", "before_start"))
    if (
        kind
        not in (
            "none",
            "cosine",
            "warmup_cosine",
            "diffusers_cosine",
            "staged_cosine",
            "constant_with_warmup",
            "warmup_linear",
        )
        or scheduler.keys() - allowed
    ):
        raise ValueError("Unsupported scheduler configuration")
    if kind in ("cosine", "warmup_cosine", "diffusers_cosine"):
        scheduler = {"warmup_steps": 0, "decay_steps": cfg["steps"], "decay_lr": 0.0, **scheduler}
        if (
            type(scheduler["warmup_steps"]) is not int
            or type(scheduler["decay_steps"]) is not int
            or not 0 <= scheduler["warmup_steps"] < scheduler["decay_steps"]
            or not math.isfinite(scheduler["decay_lr"])
            or not 0 <= scheduler["decay_lr"] <= optimizer["lr"]
            or optimizer["lr"] <= 0
        ):
            raise ValueError("Invalid cosine warmup/decay configuration")
    if kind in ("staged_cosine", "constant_with_warmup", "warmup_linear"):
        scheduler = {"warmup_steps": 0, "decay_steps": cfg["steps"], "decay_lr": 0.0, **scheduler}
        if (
            type(scheduler["warmup_steps"]) is not int
            or scheduler["warmup_steps"] < 0
            or type(scheduler["decay_steps"]) is not int
            or scheduler["decay_steps"] < 1
            or not 0 <= scheduler["decay_lr"] <= optimizer["lr"]
            or optimizer["lr"] <= 0
        ):
            raise ValueError("Invalid warmup/staged schedule")
        scheduler.setdefault("start_step", 0)
        if type(scheduler["start_step"]) is not int or scheduler["start_step"] < 0:
            raise ValueError("Staged schedules require nonnegative integer start_step")
        before_start = scheduler.get("before_start", {})
        if not isinstance(before_start, dict) or any(
            not isinstance(name, str)
            or not name
            or not isinstance(factor, (int, float))
            or isinstance(factor, bool)
            or not math.isfinite(factor)
            or factor < 0
            for name, factor in before_start.items()
        ):
            raise ValueError("scheduler.before_start requires named nonnegative finite LR factors")
    cfg["scheduler"] = scheduler
    return cfg


def make_optimizer_and_scheduler(cfg, model):
    options = dict(cfg["optimizer"])
    kind = options.pop("type")
    options.pop("grad_clip_norm")
    params = list(model.get_optim_params())
    # Relative groups follow the resolved trainer LR, including explicit overrides.
    if params and isinstance(params[0], dict):
        params = [dict(group) for group in params]
        for group in params:
            if "lr_scale" in group:
                group["lr"] = options["lr"] * group.pop("lr_scale")
    optimizer = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[
        kind
    ](params, **options)
    return optimizer, make_scheduler(cfg, optimizer)


def make_scheduler(cfg, optimizer):
    schedule = cfg["scheduler"]
    if schedule["type"] == "none":
        return None
    # Retain the official SmolVLA cosine convention, including short-run scaling.
    warmup, decay = schedule["warmup_steps"], schedule["decay_steps"]
    if schedule["type"] == "cosine" and cfg["steps"] < decay:
        warmup = int(warmup * cfg["steps"] / decay)
        decay = cfg["steps"]
    alpha = schedule["decay_lr"] / cfg["optimizer"]["lr"]

    def multiplier(step):
        if schedule["type"] in ("staged_cosine", "constant_with_warmup", "warmup_linear"):
            start = schedule.get("start_step", 0)
            if step < start:
                return 1.0
            relative = step - start
            if relative < warmup:
                return relative / max(1, warmup)
            if schedule["type"] == "constant_with_warmup":
                return 1.0
            progress = min(1.0, (relative - warmup) / max(1, decay - start - warmup))
            if schedule["type"] == "warmup_linear":
                return alpha + (1 - alpha) * (1 - progress)
            return alpha + (1 - alpha) * (1 + math.cos(math.pi * progress)) / 2
        if schedule["type"] == "diffusers_cosine":
            if step < warmup:
                return step / max(1, warmup)
            progress = min(1.0, (step - warmup) / max(1, decay - warmup))
            return alpha + (1 - alpha) * (1 + math.cos(math.pi * progress)) / 2
        if schedule["type"] == "warmup_cosine":
            if step < warmup:
                initial = 1 / (warmup + 1)
                return initial + (1 - initial) * step / warmup
            progress = min(1.0, (step - warmup) / max(1, decay - warmup))
            return alpha + (1 - alpha) * (1 + math.cos(math.pi * progress)) / 2
        if step < warmup:
            return (
                1 / (warmup + 1) if step <= 0 else (1 / (warmup + 1) - 1) * (1 - step / warmup) + 1
            )
        return (1 - alpha) * 0.5 * (1 + math.cos(math.pi * min(step, decay) / decay)) + alpha

    before_start = schedule.get("before_start", {})
    if not before_start:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    names = {group.get("name") for group in optimizer.param_groups}
    if before_start.keys() - names:
        raise ValueError(
            "scheduler.before_start names unknown parameter groups: "
            f"{sorted(before_start.keys() - names)}"
        )

    def grouped_multiplier(name, step):
        if step < schedule["start_step"]:
            return before_start.get(name, 1.0)
        return multiplier(step)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [partial(grouped_multiplier, group.get("name")) for group in optimizer.param_groups],
    )

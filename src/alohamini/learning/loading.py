"""Shared loader settings for training and ordered offline evaluation."""

import importlib.util
import multiprocessing

import torch
from torch.utils.data import DataLoader


def resolve_data_pipeline(cfg):
    """Prefer installed TorchCodec and preserve explicit loader settings."""
    cfg = dict(cfg)
    if "video_backend" not in cfg:
        cfg["video_backend"] = "torchcodec" if importlib.util.find_spec("torchcodec") else "pyav"
    cfg.setdefault("camera_workers", 3)
    cfg.setdefault("video_cache_size", 32)
    cfg.setdefault("return_uint8", True)
    cfg.setdefault("dataloader_multiprocessing_context", "spawn")
    return cfg


def loader_options(cfg, *, evaluation=False):
    result = {
        "num_workers": cfg.get("num_workers", 0),
        "prefetch_factor": cfg.get("prefetch_factor", 4),
        "persistent_workers": cfg.get("persistent_workers", True),
        "multiprocessing_context": cfg.get("dataloader_multiprocessing_context", "spawn"),
    }
    if evaluation:
        for key in result:
            if cfg.get(f"eval_{key}") is not None:
                result[key] = cfg[f"eval_{key}"]
    return result


def make_loader(
    samples,
    *,
    device="cpu",
    batch_size=8,
    batch_sampler=None,
    num_workers=0,
    prefetch_factor=4,
    persistent_workers=True,
    multiprocessing_context="spawn",
    generator=None,
):
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(prefetch_factor) is not int or prefetch_factor < 1:
        raise ValueError("prefetch_factor must be a positive integer")
    if type(persistent_workers) is not bool:
        raise ValueError("persistent_workers must be a boolean")
    if multiprocessing_context is not None and (
        multiprocessing_context not in multiprocessing.get_all_start_methods()
    ):
        raise ValueError("Unsupported dataloader_multiprocessing_context")
    if batch_sampler is None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    return DataLoader(
        samples,
        **(
            {"batch_sampler": batch_sampler}
            if batch_sampler is not None
            else {"batch_size": batch_size, "shuffle": False, "drop_last": False}
        ),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers else None,
        persistent_workers=persistent_workers and num_workers > 0,
        # CUDA/NCCL and decoder thread pools must not be inherited through fork.
        multiprocessing_context=multiprocessing_context if num_workers else None,
        pin_memory=torch.device(device).type == "cuda",
        generator=generator,
    )

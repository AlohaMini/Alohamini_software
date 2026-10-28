"""Pinned source recipe and explicit robot adaptation."""

from copy import deepcopy


def apply_preset(settings):
    cfg = deepcopy(settings)
    cfg.setdefault("state", "auto")
    cfg.setdefault("image_size", [480, 640])
    cfg.setdefault("mixed_precision", "bfloat16")
    cfg.setdefault("batch_size", 8)
    cfg.setdefault("steps", 60000)
    cfg["paper_preset"] = {
        "name": "hyvla05-recorded-v1",
        "paper": "https://arxiv.org/abs/2606.14409v2",
        "source": "https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA",
        "revision": "af57e7507ec5964b52fdf6296741e553cbcd3288",
        "reference_recipe": {
            "dataset": "RoboTwin HDF5",
            "images": "6 frames/camera, raw row stride 15; padded resize to 224x224, [-1,1]",
            "state": "20D two-arm xyz/rotation6d/gripper, padded to 32; native RoboTwin frame",
            "actions": "20 future poses, stride 3; wrist RT-relative then absolute: 40 tokens",
            "units": "metres, dimensionless rotation6d, gripper open=1/closed=0",
            "normalization": "state mean/std; action mean/std per horizon and representation",
            "language": "Hy tokenizer, 64 tokens, assistant suffix",
            "flow": "Beta(1.5,1) time; 10 Euler steps",
        },
        "adaptations": {
            "actions": "20 consecutive recorded physical targets; one representation",
            "state": "selected recorded feedback with its names and units",
            "normalization": "training split mean/std in recorded coordinates",
            "cameras": "ordered dataset/CLI camera list; no inferred camera identity",
            "time": "history stride 15 and action stride 1 in recorded rows; no resampling",
            "optimizer": "AdamW 2.5e-5, betas [0.9,0.95], decay 1e-10; 1k warmup, 40k decay",
            "scope": "supervised flow training; FlowPRO and Bézier execution are not enabled",
        },
        "overrides": deepcopy(
            settings.get("paper_preset", {}).get(
                "overrides", {k: v for k, v in settings.items() if k != "paper_preset"}
            )
        ),
    }
    return cfg

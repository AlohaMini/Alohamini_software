"""Pinned paper and released-script protocols; robot adaptations stay explicit."""

from copy import deepcopy

REVISION = "90ad00a926f34da04816ed9c3312aaf3bc845b7f"
SOURCE = "https://github.com/PKU-HMI-Lab/AC-DiT/tree/" + REVISION
PAPER = "https://proceedings.neurips.cc/paper_files/paper/2025/file/5c6c63b7ec141825ff1327563e699f76-Paper-Conference.pdf"


def apply_preset(settings):
    cfg = deepcopy(settings)
    if cfg.get("resume") and cfg.get("paper_preset"):
        return cfg
    model = cfg.setdefault("model", {})
    stage, recipe = model.get("stage", "full"), model.get("recipe", "paper")
    mobility = stage == "mobility"
    steps = (
        (20_000 if mobility else 30_000) if recipe == "paper" else (30_000 if mobility else 20_000)
    )
    defaults = dict(
        state="auto",
        image_size=[384, 384],
        image_resize_mode="pad",
        image_pad_value=0.5,
        batch_size=16 if recipe == "paper" or not mobility else 24,
        steps=steps,
        mixed_precision="bfloat16",
        trainable_dtype="float32",
        seed=42,
    )
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    cfg["optimizer"] = {
        "type": "adamw",
        "lr": 1e-4 if mobility else 1e-5,
        **cfg.get("optimizer", {}),
    }
    cfg["scheduler"] = {"type": "none", **(cfg.get("scheduler") or {})}
    cfg["ema"] = {"enable": False, **(cfg.get("ema") or {})}
    if cfg.get("pretrained_path") and not mobility:
        cfg.setdefault("normalization", "checkpoint")
    cfg["paper_preset"] = {
        "name": f"acdit-{recipe}-{stage}-alohamini-v1",
        "paper": PAPER,
        "paper_sections": ["3", "B.3", "G"],
        "implementation": SOURCE,
        "reference_revision": REVISION,
        "paper_recipe": {
            "mobility_steps": 20000,
            "full_steps": 30000,
            "batch_size": 16,
            "hardware": "8 A100 80GB; not a local hardware requirement",
            "mobility_width": 1024,
            "mobility_depth": 14,
            "body_width": 2048,
            "body_depth": 28,
        },
        "released_recipe": {
            "mobility_steps": 30000,
            "full_steps": 20000,
            "mobility_batch_per_rank": 24,
            "full_batch_per_rank": 16,
            "lr": [1e-4, 1e-5],
            "optimizer": "AdamW",
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "scheduler": "constant",
            "history": 2,
            "action_chunk": 2,
            "diffusion_train_steps": 1000,
            "sampling": "DPM-Solver++ 5 steps; sample prediction; cosine betas",
            "image": "SigLIP SO400M patch14 384, RGB mean/std 0.5",
            "pointcloud": (
                "1024 xyzrgb points in meters; Lift3D 128 tokens, kNN64, 3 stages; "
                "released grouping samples random centers"
            ),
            "language": (
                "frozen SigLIP tokens; YAML advertises 1024 but pretrained text positions allow 64"
            ),
            "coordinates": (
                "MSHab Fetch 13-D named state/action in 128 semantic slots; relative "
                "joints, absolute base velocity; 18-D goal/grasp/object/TCP context"
            ),
            "normalization": (
                "released MSHab state/actions are unnormalized; state std used for SNR40 noise"
            ),
            "trainable": (
                "point tokenizer and q/k/v LoRA, projectors, current stage head; SigLIP frozen"
            ),
        },
        "native_adaptations": [
            "Paper B.3 reports batch 16 without specifying global/per-rank; "
            "native batch_size is per process",
            "Native heads start from random initialization unless an explicit "
            "RDT initializer and semantic slot maps are supplied",
            (
                "Recorded names/units and explicit slot maps replace Fetch "
                "coordinates; native packing follows declared names, never "
                "vector-width semantics"
            ),
            (
                "Default state/action MEAN_STD uses training episodes only; set "
                "IDENTITY in JSON for data already in reference units"
            ),
            (
                "Dataset FPS is retained; windows use episode-local rows with "
                "repeated boundaries, no resampling; source state is current, "
                "visual/point histories end at current row"
            ),
            (
                "Observed cameras only, in recorded order; missing cameras are not "
                "synthesized; use_pointcloud=false is an explicit 2D adaptation"
            ),
            "Goal/grasp/object/TCP context is off unless named recorded fields are supplied",
            (
                "Entire mobility branch and point encoder transfer between stages; "
                "complete mobility branch is frozen in stage two; unweighted "
                "perception feeds mobility consistently in training and inference"
            ),
            (
                "Native FP32 trainable storage with BF16 autocast prevents BF16 "
                "update loss; frozen encoders retain their constructed types"
            ),
            (
                "Text length respects local SigLIP positions; variable camera/token "
                "counts replace hard-coded source reshapes; native augmentations are "
                "recorded config, not original PIL equivalence"
            ),
        ],
        "overrides": {k: deepcopy(v) for k, v in settings.items() if k != "paper_preset"},
    }
    return cfg

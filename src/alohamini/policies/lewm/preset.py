"""Pinned released LeWM recipe and explicit robot adaptations."""

from copy import deepcopy

REVISION = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"


def apply_preset(settings):
    cfg = deepcopy(settings)
    if cfg.get("resume") and cfg.get("paper_preset"):
        return cfg
    for key, value in dict(
        state="none",
        image_size=[224, 224],
        batch_size=128,
        steps=30000,
        seed=3072,
        mixed_precision="bfloat16",
        trainable_dtype="float32",
    ).items():
        cfg.setdefault(key, value)
    cfg["ema"] = {"enable": False, **cfg.get("ema", {})}
    cfg["scheduler"] = {"type": "none", **(cfg.get("scheduler") or {})}
    cfg["paper_preset"] = dict(
        name="lewm-released-recorded-v1",
        paper="https://arxiv.org/abs/2603.19312v1",
        source="https://github.com/lucas-maes/le-wm/tree/" + REVISION,
        reference_recipe=dict(
            fields="pixels/action; proprio/state are recorded but not model inputs",
            units=(
                "Benchmark actions follow each environment; native preserves recorded action units"
            ),
            data="PushT: HDF5/Lance pixels and actions; 224 RGB ImageNet normalization",
            state="No proprioception input; goal is an image, not language",
            time="history 3, frame skip/action block 5, next-step prediction",
            architecture="ViT-Tiny 192/12/3 patch14; causal AdaLN predictor 6 layers, 16x64 heads",
            loss=(
                "prediction MSE + .09 SIGReg in released config; paper lambda=.1; no "
                "stop-gradient/EMA"
            ),
            normalization="action z-score; released loader uses sample standard deviation",
            training="AdamW lr=5e-5 wd=1e-3, batch128, bf16, 100 epochs, epoch warmup/cosine",
            planning="CEM 300 candidates, 30 elites, 30 iterations, 5 blocks of 5 actions",
        ),
        adaptations=[
            (
                "LeRobot v3/recorded coordinates replace benchmark HDF5, without "
                "inferred action semantics"
            ),
            "Explicit 30000-step constant-lr budget replaces dataset-dependent epoch schedule",
            (
                "Platform fits population statistics on training episodes only; "
                "original field units retained"
            ),
            (
                "Shared encoder/projector across cameras; multiview embeddings "
                "averaged, single-view unchanged"
            ),
            "Episode padding excluded from prediction loss; final action_block rows omitted",
            (
                "Distributed SIGReg uses differentiable per-forward global batch; "
                "accumulation remains microbatch-wise"
            ),
            (
                "Planning starts from the current image and grows a causal history; "
                "explicit goal images required"
            ),
            (
                "CEM constrained to training action limits by default; "
                "constrain_actions=false restores unbounded sampling"
            ),
        ],
        overrides=deepcopy(settings),
    )
    return cfg

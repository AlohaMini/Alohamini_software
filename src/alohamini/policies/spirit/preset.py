"""Versioned reference protocol and explicitly separate AlohaMini adaptation."""

from copy import deepcopy

SOURCE = (
    "https://github.com/Spirit-AI-Team/spirit-v1.5/tree/6d67377f16f67d6575447a1f29fdd83a2e0099d9"
)


def apply_preset(settings):
    cfg = deepcopy(settings)
    for key, value in dict(
        state="joint_position,base_velocity,lift_height",
        image_size=[240, 320],
        batch_size=32,
        steps=40000,
        mixed_precision="bfloat16",
    ).items():
        cfg.setdefault(key, value)
    cfg["paper_preset"] = dict(
        name="spirit-v1.5-6d67377-alohamini-v1",
        source=SOURCE,
        reference_recipe=dict(
            model_revision="9886fa7552de70b097c4214b7e40e66ff937a6fd",
            frames="Franka example: image at i, state at max(i-1,0), future measured poses from i",
            action="delta xyz; rotvec(R_target @ R_anchor.T); absolute gripper; padded 14 to 32",
            images="RGB 240x320 then Qwen3-VL image processor; three fixed slots",
            language="Qwen chat: image tokens and robot type in user turn, task in assistant turn",
            normalization="separate state/action p1/p99 mapped to [-1,1], epsilon 1e-8",
            window="60 future steps at reference 30 Hz; one state anchor; 10 Euler flow steps",
            optimizer="full AdamW lr 2.5e-5, betas [0.9,0.95], warmup 1000, cosine decay",
        ),
        adaptation=dict(
            action_representation=cfg.get("model", {}).get("action_representation", "eef_delta"),
            source="recorded action requests; observation/action/image pairing preserved",
            coordinates="explicit kinematic descriptor or explicitly declared recorded fields",
            extra_axes="retain all named base/lift action fields as absolute recorded values",
            timing="dataset fps and original pairing; no timestamp resampling",
            language_head="exclude an unused independent vocabulary head; train tied input embeddings",
            gripper="descriptor explicitly declares physical or canonical units; no geometry inferred",
            padding="mask repeated episode-tail targets and padded action coordinates",
            images="selected named cameras mapped to slots; absent slots black; dataset image pipeline",
            statistics="exact training-split p1/p99 over all selected windows, including repeated tails",
        ),
        overrides=deepcopy(
            settings.get("paper_preset", {}).get(
                "overrides", {k: v for k, v in settings.items() if k != "paper_preset"}
            )
        ),
    )
    return cfg

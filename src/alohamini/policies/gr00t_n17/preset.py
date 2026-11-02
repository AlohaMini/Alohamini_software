"""Official N1.7 fine-tuning settings; recorded robot coordinates remain explicit."""

from copy import deepcopy

REVISION = "c56f54128d92f743889b7e4cecfc1ceafd88739e"


def apply_preset(settings):
    cfg = deepcopy(settings)
    if cfg.get("resume") and cfg.get("paper_preset"):
        return cfg
    for key, value in dict(
        state="auto",
        image_size=[480, 640],
        batch_size=8,
        steps=10000,
        mixed_precision="bfloat16",
        trainable_dtype="float32",
        seed=42,
    ).items():
        cfg.setdefault(key, value)
    cfg["ema"] = {"enable": False, **cfg.get("ema", {})}
    cfg["scheduler"] = {
        "type": "diffusers_cosine",
        "warmup_steps": int(cfg["steps"] * 0.05),
        "decay_steps": cfg["steps"],
        "decay_lr": 0.0,
        **(cfg.get("scheduler") or {}),
    }
    cfg["paper_preset"] = dict(
        name="gr00t-n17-recorded-v1",
        model_card="https://huggingface.co/nvidia/GR00T-N1.7-3B",
        architecture_paper="https://arxiv.org/abs/2503.14734",
        source="https://github.com/NVIDIA/Isaac-GR00T/tree/" + REVISION,
        reference_recipe=dict(
            fields=(
                "Embodiment video/state/action keys and language; native images/state/action/task"
            ),
            units=(
                "Per-embodiment schema; native retains declared units without inferring coordinates"
            ),
            architecture=(
                "Cosmos-Reason2-2B/Qwen3-VL, first16 LLM layers; 32-layer alternating VL DiT"
            ),
            state_action=(
                "132-wide padded embodiment-specific projections; native robot "
                "coordinates are named"
            ),
            relative_eef=(
                "Per-key RELATIVE+EEF: T_observed^-1 @ T_target; XYZ_ROT6D uses first "
                "two matrix rows. DROID also uses relative joints and absolute gripper; "
                "other embodiments may use NON_EEF/absolute actions"
            ),
            images=(
                "RGB, fraction crop .95, shortest edge256, Qwen3-VL processor; all selected views"
            ),
            language="Lowercase/punctuation formalization; Qwen3-VL chat template and left padding",
            normalization="Per-field q01/q99 limits by default; optional recorded min/max",
            time=(
                "Current images/state; 40 future actions; Beta(1.5,1) flow time, "
                "noise_s=.999; Euler4"
            ),
            training=(
                "AdamW lr1e-4 wd1e-5, 5% warmup/cosine; fine-tune10000 steps, global "
                "batch64; pretraining defaults30000 steps, global batch1024"
            ),
            finetuning=(
                "Official fine-tune defaults freeze LLM/vision and train "
                "projectors/DiT; state dropout .2"
            ),
        ),
        adaptations=[
            (
                "Shared native LeRobot v3/Host reader, optimizer, EMA, DDP/FSDP2 and "
                "portable checkpoints"
            ),
            (
                "Local GR00T weights plus local Cosmos config/processor; no network "
                "during model load"
            ),
            ("Per-rank batch8 is a resource adaptation, not the official fine-tune global batch64"),
            (
                "New embodiment31 projectors reinitialized by default; recorded names "
                "packed into explicit slots"
            ),
            (
                "Recorded actions by default; optional named FK, TCP deltas in base/tool "
                "frame and rotvec/rotation_6d via data.*; no inferred FK or sin/cos expansion"
            ),
            (
                "Quantiles fitted exactly in float64 after field transforms; relative "
                "action windows pooled across horizon (official relative stats are per-step); "
                "inputs and inverse outputs clipped to limits"
            ),
            (
                "Torchvision crop/color augmentation replaces Albumentations; not "
                "claimed pixel-identical"
            ),
            (
                "Strict VL/action padding masks enabled for new native training; "
                "attention uses PyTorch SDPA"
            ),
            ("LM logits omitted from the action path; VLM trainability is configured explicitly"),
        ],
        overrides=deepcopy(settings),
    )
    return cfg

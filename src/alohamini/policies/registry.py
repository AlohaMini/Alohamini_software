"""Lazy algorithm registration; no model loading at import time."""

from importlib import import_module

ALGORITHMS = {
    "g05": ("alohamini.policies.g05.adapter", "G05Algorithm"),
    "cosmos3_nano": ("alohamini.policies.cosmos3.adapter", "Cosmos3NanoAlgorithm"),
    "cosmos3_edge": ("alohamini.policies.cosmos3.adapter", "Cosmos3EdgeAlgorithm"),
    "lingbot_va": ("alohamini.policies.lingbot_va.adapter", "LingBotVAAlgorithm"),
    "vla_jepa": ("alohamini.policies.vla_jepa.adapter", "VLAJEPAAlgorithm"),
    "wall_oss_05": ("alohamini.policies.wall_oss_05.adapter", "WallOSS05Algorithm"),
    "fineart_vla": ("alohamini.policies.fineart_vla.adapter", "FineARTVLAAlgorithm"),
    "dm05": ("alohamini.policies.dm05.adapter", "DM05Algorithm"),
    "xvla": ("alohamini.policies.xvla.adapter", "XVLAAlgorithm"),
    "pi0_fast": ("alohamini.policies.pi0_fast.adapter", "PI0FastAlgorithm"),
    "evo1": ("alohamini.policies.evo1.adapter", "Evo1Algorithm"),
    "eo1": ("alohamini.policies.eo1.adapter", "EO1Algorithm"),
    "pi0": ("alohamini.policies.pi0.adapter", "PI0Algorithm"),
    "vqbet": ("alohamini.policies.vqbet.adapter", "VQBeTAlgorithm"),
    "longwam": ("alohamini.policies.longwam.adapter", "LongWAMAlgorithm"),
    "bspline": ("alohamini.policies.bspline.adapter", "BSplineAlgorithm"),
    "xiaomi_robotics1": ("alohamini.policies.xiaomi_robotics1.adapter", "XiaomiAlgorithm"),
    "lingbot_vla_v2": ("alohamini.policies.lingbot_vla_v2.adapter", "LingBotAlgorithm"),
    "gr00t_n17": ("alohamini.policies.gr00t_n17.adapter", "GR00TN17Algorithm"),
    "lewm": ("alohamini.policies.lewm.adapter", "LeWMAlgorithm"),
    "acdit": ("alohamini.policies.acdit.adapter", "ACDiTAlgorithm"),
    "hyvla05": ("alohamini.policies.hyvla05.adapter", "HyVLA05Algorithm"),
    "openwam": ("alohamini.policies.openwam.adapter", "OpenWAMAlgorithm"),
    "cdp": ("alohamini.policies.cdp.adapter", "CDPAlgorithm"),
    "dit_policy": ("alohamini.policies.dit_policy.adapter", "DiTPolicyAlgorithm"),
    "multi_task_dit": ("alohamini.policies.multi_task_dit.adapter", "MultiTaskDiTAlgorithm"),
    "diffusion": ("alohamini.policies.diffusion.adapter", "DiffusionAlgorithm"),
    "fastwam": ("alohamini.policies.fastwam.adapter", "FastWAMAlgorithm"),
    "act": ("alohamini.policies.act.adapter", "ACTAlgorithm"),
    "am_act": ("alohamini.policies.act.adapter", "AMACTAlgorithm"),
    "smolvla": ("alohamini.policies.smolvla.adapter", "SmolVLAAlgorithm"),
    "pi05": ("alohamini.policies.pi05.adapter", "PI05Algorithm"),
    "spirit": ("alohamini.policies.spirit.adapter", "SpiritAlgorithm"),
}


def algorithm(name):
    try:
        module, cls = ALGORITHMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown policy: {name}") from exc
    return getattr(import_module(module), cls)()

"""Lazy algorithm registration; no model loading at import time."""

from importlib import import_module

ALGORITHMS = {
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

"""Explicit PI0.5 parameter ownership; LoRA keeps base checkpoint key names."""

from torch import nn

from alohamini.policies.lora import LoRALinear as LoRALinear

# Vision includes the image-to-language projector. Heads/time conditioning belong
# to the action expert and remain fully trainable, like OpenPI's LoRA recipe.
FINETUNING_MODES = {
    "full": ("full", "full", "full"),
    "a": ("frozen", "lora", "lora"),
    "b": ("frozen", "frozen", "full"),
    "c": ("full", "lora", "full"),
    "d": ("lora", "lora", "full"),
}


def _adapt(module, mode, rank, alpha):
    module.requires_grad_(mode == "full")
    if mode == "lora":
        # Attention/MLP projections, excluding norms, adaRMS conditioning, and
        # SigLIP's unused pooling head. Vision's projector is a single Linear.
        targets = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "fc1",
            "fc2",
            "linear",
        }
        for name, child in list(module.named_modules()):
            if (
                isinstance(child, nn.Linear)
                and name.rsplit(".", 1)[-1] in targets
                and ".head." not in name
            ):
                parent, _, attr = name.rpartition(".")
                setattr(module.get_submodule(parent), attr, LoRALinear(child, rank, alpha))


def configure_finetuning(network, config):
    vision, language, expert = FINETUNING_MODES[config.finetuning_mode]
    if config.finetuning_mode == "a":
        vision = config.a_vision_mode
    combined = network.paligemma_with_expert
    vlm = combined.paligemma
    network.requires_grad_(True)
    _adapt(vlm.vision_tower, vision, config.lora_rank, config.lora_alpha)
    _adapt(vlm.multi_modal_projector, vision, config.lora_rank, config.lora_alpha)
    _adapt(vlm.language_model, language, config.lora_rank, config.lora_alpha)
    # PaliGemma ties the output head to its input embeddings; never LoRA-wrap it.
    vlm.lm_head.requires_grad_(language == "full")
    _adapt(combined.gemma_expert.model, expert, config.expert_lora_rank, config.expert_lora_alpha)
    # The action expert never predicts vocabulary tokens.
    combined.gemma_expert.lm_head.requires_grad_(False)
    return parameter_summary(network)


def parameter_summary(network):
    groups = {key: {"total": 0, "trainable": 0} for key in ("vision", "llm", "expert")}
    for name, param in network.named_parameters():
        if any(part in name for part in ("vision_tower", "multi_modal_projector")):
            group = "vision"
        elif name.startswith("paligemma_with_expert.paligemma."):
            group = "llm"
        else:
            group = "expert"
        groups[group]["total"] += param.numel()
        if param.requires_grad:
            groups[group]["trainable"] += param.numel()
    return groups


def load_base_weights(network, weights):
    """Only newly introduced adapters may be absent from an OpenPI base."""
    from safetensors.torch import load_model

    missing, unexpected = load_model(network, weights, strict=False, device="cpu")
    adapters = {
        name for name, _ in network.named_parameters() if name.endswith((".lora_A", ".lora_B"))
    }
    if set(missing) != adapters or unexpected:
        raise RuntimeError(
            f"Invalid PI0.5 base weights: missing={sorted(missing - adapters)}, "
            f"unexpected={sorted(unexpected)}; "
            "use native checkpoint loading for trained LoRA weights"
        )

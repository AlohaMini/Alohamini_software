"""FSDP2 placement for native policies, without algorithm-specific execution paths."""

import logging

import torch
from torch import nn


def sharding_units(model, preferred=()):
    """Prefer declared blocks; otherwise shard called projections and normalizers.

    Functional transformer implementations may bypass decoder-layer.forward, so
    their Linear/Conv calls are valid boundaries even without block declarations.
    MultiheadAttention accesses out_proj.weight directly and must stay one unit.
    Shared parameters stay with their common ancestor (normally the policy root).
    """
    primitive = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.MultiheadAttention,
    )
    owners = {}
    for module in model.modules():
        for param in module.parameters(recurse=False):
            owners.setdefault(id(param), set()).add(module)
    claimed, units = set(), []

    def eligible(module):
        members = set(module.modules())
        parameters = [p for p in module.parameters() if id(p) not in claimed]
        return (
            bool(parameters)
            and all(owners[id(p)] <= members for p in parameters)
            and len({p.dtype for p in parameters if p.requires_grad}) <= 1
        )

    def add(module):
        units.append(module)
        claimed.update(id(p) for p in module.parameters())

    def visit(module):
        declared = type(module).__name__ in preferred
        atomic = isinstance(module, nn.MultiheadAttention)
        if module is not model and (declared or atomic) and eligible(module):
            add(module)
            return
        for child in module.children():
            visit(child)
        if module is not model and (declared or isinstance(module, primitive)) and eligible(module):
            add(module)

    visit(model)
    remaining = [p for p in model.parameters() if id(p) not in claimed and p.requires_grad]
    if len({p.dtype for p in remaining}) > 1:
        raise ValueError(
            "FSDP2 root has mixed trainable dtypes; declare the called normalization/"
            "projection modules in fsdp_classes so each unit has one trainable dtype"
        )
    return units


def prepare_sharded_model(model, optimizer, *, preferred=()):
    """Shard one CPU unit at a time and keep optimizer references and native dtypes."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import Shard

    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Construct the FSDP2 model on CPU before preparing it")
    names = {id(p): name for name, p in model.named_parameters()}
    groups = [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]
    units = sharding_units(model, preferred)
    # Preserve BF16 backbones, FP32 norms/adapters, and batch target dtypes. AMP
    # remains controlled by Execution.autocast rather than FSDP input casting.
    precision = MixedPrecisionPolicy(reduce_dtype=torch.float32, cast_forward_inputs=False)
    world_size = torch.distributed.get_world_size()

    def placement(parameter):
        # A one-row embedding would leave empty dim-0 shards. Prefer an evenly
        # divisible dimension when available, including positional/camera tokens.
        if parameter.shape[0] < world_size:
            for dimension, size in enumerate(parameter.shape):
                if size >= world_size and size % world_size == 0:
                    return Shard(dimension)
        return Shard(0)

    for module in units:
        fully_shard(
            module,
            reshard_after_forward=True,
            mp_policy=precision,
            shard_placement_fn=placement,
        )
    fully_shard(
        model,
        reshard_after_forward=False,
        mp_policy=precision,
        shard_placement_fn=placement,
    )
    # Parameters are already local CUDA shards. Run normal device hooks as well:
    # some policies own frozen runtime assets outside their registered state.
    model.to(device=torch.device("cuda", torch.cuda.current_device()))
    parameters = dict(model.named_parameters())
    for group, group_names in zip(optimizer.param_groups, groups, strict=True):
        group["params"] = [parameters[name] for name in group_names]
    logging.getLogger(__name__).info("FSDP2: %d compute units plus policy root", len(units))
    return model

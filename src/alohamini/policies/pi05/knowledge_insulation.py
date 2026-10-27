"""KI attention and token losses (pi.website/download/pi05_KI.pdf, §5)."""

import torch
from torch.nn import functional as F


def attention_masks(context_mask, target_mask, action_mask):
    """Two independent output branches sharing a bidirectional observation prefix.

    Action positions deliberately exclude teacher-forced token positions, matching
    the prefix KV-cache used by flow inference.
    """
    batch, context_len = context_mask.shape
    target_len = target_mask.shape[1]
    vlm_len = context_len + target_len
    valid = torch.cat((context_mask, target_mask, action_mask), dim=1)
    mask = torch.zeros(
        (batch, valid.shape[1], valid.shape[1]), dtype=torch.bool, device=valid.device
    )
    mask[:, :context_len, :context_len] = True
    mask[:, context_len:vlm_len, :context_len] = True
    mask[:, context_len:vlm_len, context_len:vlm_len] = torch.ones(
        (target_len, target_len), dtype=torch.bool, device=valid.device
    ).tril()
    mask[:, vlm_len:, :context_len] = True
    mask[:, vlm_len:, vlm_len:] = True
    mask &= valid[:, :, None] & valid[:, None, :]
    context_positions = context_mask.long().cumsum(-1) - 1
    offset = context_mask.sum(-1, keepdim=True)
    positions = torch.cat(
        (
            context_positions,
            offset + target_mask.long().cumsum(-1) - 1,
            offset + action_mask.long().cumsum(-1) - 1,
        ),
        dim=1,
    )
    return mask, positions


def token_cross_entropy(hidden, context_mask, targets, target_mask, head, checkpoint, chunk_size):
    """Per-example mean CE, with bounded vocabulary-logit activation memory.

    Predict the first target from the last valid context token and every subsequent
    target from its predecessor. Prompt padding is never a prediction position.
    """
    batch, length = targets.shape
    if target_mask.shape != targets.shape or not target_mask.any(dim=1).all():
        raise ValueError("Each language target must contain at least one valid token")
    context_len = context_mask.shape[1]
    positions = torch.arange(context_len, device=hidden.device).expand_as(context_mask)
    last_context = positions.masked_fill(~context_mask, -1).max(dim=1).values
    if (last_context < 0).any():
        raise ValueError("Language prediction requires a nonempty context")
    predictors = torch.cat(
        (
            hidden[torch.arange(batch, device=hidden.device), last_context][:, None],
            hidden[:, context_len : context_len + length - 1],
        ),
        dim=1,
    )
    rows = torch.arange(batch, device=hidden.device)[:, None].expand_as(targets)[target_mask]
    selected, labels = predictors[target_mask], targets[target_mask]

    def cross_entropy(inputs, labels):
        return F.cross_entropy(head(inputs).float(), labels, reduction="none")

    losses = []
    for start in range(0, len(labels), chunk_size):
        losses.append(
            checkpoint(
                cross_entropy,
                selected[start : start + chunk_size],
                labels[start : start + chunk_size],
            )
        )
    sums = torch.zeros(batch, device=hidden.device).index_add(0, rows, torch.cat(losses))
    return sums / target_mask.sum(dim=1)

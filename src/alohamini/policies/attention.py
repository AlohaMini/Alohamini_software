"""PyTorch attention for packed model inputs without a FlashAttention dependency."""

import torch
import torch.nn.functional as F


def packed_attention(q, k, v, *, softmax_scale=None, causal=False, dropout_p=0.0):
    """Attend B,L,H,D tensors, using FlashAttention's bottom-right causal alignment."""
    mask = None
    if causal:
        nq, nk = q.shape[1], k.shape[1]
        mask = torch.arange(nk, device=q.device)[None] <= (
            torch.arange(nq, device=q.device)[:, None] + nk - nq
        )
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask,
        dropout_p=dropout_p,
        scale=softmax_scale,
        enable_gqa=q.shape[2] != k.shape[2],
    )
    return output.transpose(1, 2)


def variable_attention(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *,
    softmax_scale=None,
    causal=False,
    dropout_p=0.0,
):
    """Attend each packed sequence independently; boundaries never share keys."""
    qs, ks = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    if len(qs) != len(ks) or not qs or qs[0] != 0 or ks[0] != 0:
        raise ValueError("Packed attention requires matching cumulative sequence boundaries")
    if qs[-1] != len(q) or ks[-1] != len(k) or len(k) != len(v):
        raise ValueError("Packed attention boundaries do not cover the supplied tensors")
    outputs = []
    for a, b, c, d in zip(qs[:-1], qs[1:], ks[:-1], ks[1:], strict=True):
        if not 0 <= b - a <= max_seqlen_q or not 0 <= d - c <= max_seqlen_k:
            raise ValueError("Invalid packed sequence length")
        outputs.append(
            packed_attention(
                q[a:b][None],
                k[c:d][None],
                v[c:d][None],
                softmax_scale=softmax_scale,
                causal=causal,
                dropout_p=dropout_p,
            )[0]
        )
    return torch.cat(outputs, dim=0) if outputs else q.clone()

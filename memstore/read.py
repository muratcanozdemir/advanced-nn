"""The one lookup operation:  f(x) = sum_i n_i v_i,  w_i = sigma(score(x, k_i)),  n = normalize(w).

Normalisation modes:
  "none"   memories are independent (FFN, linear attention).
  "sum"    memories compete over the whole pool (softmax attention when sigma = exp).
  "split"  memories compete only within their own pool; pool outputs are summed.
           `pools` gives the pool sizes along the memory axis, e.g. (T_context, N_persistent).
           "sum" over several pools is the joint softmax of all-attention.
Optional `topk` keeps only the k highest-scoring memories per pool before normalising.
"""

import torch
import torch.nn.functional as F

from memstore.kernels import Exp, Kernel


def dot_score(x: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """x: (..., Q, d), K: (..., M, d) -> (..., Q, M)."""
    return x @ K.transpose(-1, -2)


def _topk_mask(s: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, s.shape[-1])
    thresh = s.topk(k, dim=-1).values[..., -1:]
    return s >= thresh


def normalize(
    s: torch.Tensor,
    kernel: Kernel,
    mode: str = "none",
    pools: tuple[int, ...] | None = None,
    mask: torch.Tensor | None = None,
    topk: int | None = None,
) -> torch.Tensor:
    """Scores (..., Q, M) -> normalised weights (..., Q, M). `mask` is True where a memory may be read."""
    pools = pools or (s.shape[-1],)
    if sum(pools) != s.shape[-1]:
        raise ValueError(f"pools {pools} do not cover {s.shape[-1]} memories")
    if mode not in ("none", "sum", "split"):
        raise ValueError(f"unknown normalisation {mode!r}")

    keep = torch.ones_like(s, dtype=torch.bool) if mask is None else mask.expand_as(s)
    if topk is not None:
        chunks = s.masked_fill(~keep, float("-inf")).split(pools, dim=-1)
        keep = keep & torch.cat([_topk_mask(c, topk) for c in chunks], dim=-1)

    groups = [s.shape[-1]] if mode == "sum" else list(pools)
    s_parts, keep_parts = s.split(groups, dim=-1), keep.split(groups, dim=-1)

    out = []
    for sp, kp in zip(s_parts, keep_parts):
        if mode == "none":
            out.append(kernel(sp) * kp)
        elif kernel.log_weight is not None:
            logits = kernel.log_weight(sp).masked_fill(~kp, float("-inf"))
            # A fully masked row (e.g. an empty pool) gets zero weight instead of NaN.
            out.append(torch.softmax(logits, dim=-1).nan_to_num(0.0))
        else:
            w = kernel(sp) * kp
            out.append(w / w.sum(-1, keepdim=True).clamp_min(1e-12))
    return torch.cat(out, dim=-1)


def read(
    x: torch.Tensor,
    K,
    V: torch.Tensor,
    kernel: Kernel,
    norm: str = "none",
    score=dot_score,
    pools: tuple[int, ...] | None = None,
    mask: torch.Tensor | None = None,
    topk: int | None = None,
) -> torch.Tensor:
    """x: (..., Q, d) queries; K: keys accepted by `score`; V: (..., M, d_v). Returns (..., Q, d_v)."""
    if isinstance(kernel, Exp) and norm in ("sum", "split") and score is dot_score and topk is None:
        return _fused_softmax_read(x, K, V, kernel.beta, norm, pools, mask)
    n = normalize(score(x, K), kernel, norm, pools, mask, topk)
    return n @ V


def _fused_softmax_read(x, K, V, beta, norm, pools, mask):
    """Same result as the general path for exp kernel + softmax, via PyTorch's fused attention.

    Batch dims of x, K, V and the mask are broadcast first; SDPA requires matching batch dims.
    """
    batch = torch.broadcast_shapes(x.shape[:-2], K.shape[:-2], V.shape[:-2])
    x, K, V = x.expand(*batch, *x.shape[-2:]), K.expand(*batch, *K.shape[-2:]), V.expand(*batch, *V.shape[-2:])
    groups = [K.shape[-2]] if norm == "sum" or pools is None else list(pools)
    out, start = 0, 0
    for size in groups:
        m = None if mask is None else mask[..., start : start + size]
        y = F.scaled_dot_product_attention(
            x, K[..., start : start + size, :], V[..., start : start + size, :], attn_mask=m, scale=beta
        )
        # A query with no readable memory in this pool contributes nothing, as in `normalize`.
        out = out + (y.nan_to_num(0.0) if m is not None else y)
        start += size
    return out

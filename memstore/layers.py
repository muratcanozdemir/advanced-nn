"""Grid rows as layers, each a configuration of `memstore.read`.

Shapes: x is (B, T, d). Biases are omitted throughout, as in the FFN-as-attention conversion.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from memstore import kernels
from memstore.read import read


class FFN(nn.Module):
    """Memory source: weights. Keys are rows of W1, values are columns of W2. No normalisation.

    With kernel=Exp and norm="sum" this is the "FF-attn" arm of Sukhbaatar et al.
    """

    def __init__(self, d: int, hidden: int, kernel: kernels.Kernel = kernels.relu, norm: str = "none"):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(hidden, d) / math.sqrt(d))
        self.values = nn.Parameter(torch.randn(hidden, d) / math.sqrt(hidden))
        self.kernel, self.norm = kernel, norm

    def forward(self, x):
        return read(x, self.keys, self.values, self.kernel, self.norm)


def _swiglu_score(x, K):
    keys, gates = K
    return F.silu(x @ keys.T) * (x @ gates.T)


class SwiGLUFFN(nn.Module):
    """Memory source: weights. The weight of memory i is silu(k_i . x) * (g_i . x)."""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(hidden, d) / math.sqrt(d))
        self.gates = nn.Parameter(torch.randn(hidden, d) / math.sqrt(d))
        self.values = nn.Parameter(torch.randn(hidden, d) / math.sqrt(hidden))

    def forward(self, x):
        return read(x, (self.keys, self.gates), self.values, kernels.identity, score=_swiglu_score)


class _Heads(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        if d % n_heads:
            raise ValueError("d must be divisible by n_heads")
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def split(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        return q, k, v  # each (B, H, T, dh)

    def merge(self, y):
        B, H, T, dh = y.shape
        return self.out(y.transpose(1, 2).reshape(B, T, H * dh))


class Attention(_Heads):
    """Memory source: context, one memory per token. Kernel exp(s / sqrt(dh)), softmax over the context."""

    def __init__(self, d: int, n_heads: int, causal: bool = True):
        super().__init__(d, n_heads)
        self.causal = causal

    def forward(self, x):
        q, k, v = self.split(x)
        T = x.shape[1]
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device)
        if self.causal:
            mask = mask.tril()
        y = read(q, k, v, kernels.Exp(1 / math.sqrt(self.dh)), "sum", mask=mask)
        return self.merge(y)


class AllAttention(_Heads):
    """Sukhbaatar et al. 2019: context memories plus N learned persistent memories per head.

    norm="sum"   joint softmax over both pools (all-attn).
    norm="split" separate softmax per pool, outputs summed (attn-split).
    """

    def __init__(self, d: int, n_heads: int, n_persistent: int, norm: str = "sum", causal: bool = True):
        super().__init__(d, n_heads)
        self.n, self.norm, self.causal = n_persistent, norm, causal
        # Sukhbaatar et al. scale persistent keys by sqrt(dh) and values by sqrt(N) at init; unit-variance init here.
        self.pkeys = nn.Parameter(torch.randn(n_heads, n_persistent, self.dh))
        self.pvalues = nn.Parameter(torch.randn(n_heads, n_persistent, self.dh))

    def forward(self, x):
        q, k, v = self.split(x)
        B, T = x.shape[:2]
        K = torch.cat([k, self.pkeys.expand(B, -1, -1, -1)], dim=2)
        V = torch.cat([v, self.pvalues.expand(B, -1, -1, -1)], dim=2)
        ctx = torch.ones(T, T, dtype=torch.bool, device=x.device)
        if self.causal:
            ctx = ctx.tril()
        mask = torch.cat([ctx, torch.ones(T, self.n, dtype=torch.bool, device=x.device)], dim=1)
        y = read(q, K, V, kernels.Exp(1 / math.sqrt(self.dh)), self.norm, pools=(T, self.n), mask=mask)
        return self.merge(y)


class AdditiveScore(nn.Module):
    """Bahdanau et al. 2014: score(x, k) = v . tanh(W x + U k), an MLP used as the similarity function.

    Pass as `score=` to `read`. Unlike a dot product it cannot be computed as one matmul over all pairs:
    it materialises a (Q, M, hidden) tensor.
    """

    def __init__(self, d_query: int, d_key: int, hidden: int):
        super().__init__()
        self.W = nn.Linear(d_query, hidden, bias=False)
        self.U = nn.Linear(d_key, hidden, bias=True)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, x, K):
        return self.v(torch.tanh(self.W(x).unsqueeze(-2) + self.U(K).unsqueeze(-3))).squeeze(-1)

"""Byte-level LM arms for the Sukhbaatar et al. (2019) ablation, built from memstore layers.

Every arm has the same depth and width. Persistent memories per layer are sized N = ffn_hidden, so
2 N d persistent parameters equal the FFN's 2 d h: all arms except "n0" have the same parameter count
at the same depth. (In the paper, FF-attn changed depth to match parameters; here nothing has to.)

  ffn         attention + ReLU FFN                     standard transformer
  ff-attn     attention + softmax FFN                  the FFN's ReLU swapped for softmax, separate sublayer
  all-attn    attention over context + persistent       one joint softmax, no FFN
  attn-split  same memories, one softmax per pool       outputs summed
  head-split  half the heads read context only, half read persistent only (2N each, same params)
  n0          attention only, no FFN, no persistent memories
"""

import math

import torch
import torch.nn as nn

from memstore import kernels, read
from memstore.layers import FFN, AllAttention, Attention, _Heads

ARMS = ("ffn", "ff-attn", "all-attn", "attn-split", "head-split", "n0")


class HeadSplitAttention(_Heads):
    def __init__(self, d, n_heads, n_persistent):
        super().__init__(d, n_heads)
        self.hc = n_heads // 2
        hp = n_heads - self.hc
        # Persistent heads get the persistent memory budget of all heads: N * H / hp memories each.
        self.np = n_persistent * n_heads // hp
        self.pkeys = nn.Parameter(torch.randn(hp, self.np, self.dh))
        self.pvalues = nn.Parameter(torch.randn(hp, self.np, self.dh))

    def forward(self, x):
        q, k, v = self.split(x)
        T = x.shape[1]
        kern = kernels.Exp(1 / math.sqrt(self.dh))
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        yc = read(q[:, : self.hc], k[:, : self.hc], v[:, : self.hc], kern, "sum", mask=causal)
        yp = read(q[:, self.hc :], self.pkeys, self.pvalues, kern, "sum")
        return self.merge(torch.cat([yc, yp], dim=1))


class Block(nn.Module):
    def __init__(self, arm, d, n_heads, hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        if arm in ("ffn", "ff-attn", "n0"):
            self.attn = Attention(d, n_heads)
        elif arm in ("all-attn", "attn-split"):
            self.attn = AllAttention(d, n_heads, hidden, norm="sum" if arm == "all-attn" else "split")
        elif arm == "head-split":
            self.attn = HeadSplitAttention(d, n_heads, hidden)
        else:
            raise ValueError(arm)
        self.ffn = None
        if arm == "ffn":
            self.ffn = FFN(d, hidden)
        elif arm == "ff-attn":
            self.ffn = FFN(d, hidden, kernel=kernels.Exp(1.0), norm="sum")
        if self.ffn is not None:
            self.ln2 = nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if self.ffn is not None:
            x = x + self.ffn(self.ln2(x))
        return x


class ByteLM(nn.Module):
    def __init__(self, arm, layers=4, d=256, n_heads=4, hidden=1024, context=256):
        super().__init__()
        self.tok = nn.Embedding(256, d)
        self.pos = nn.Embedding(context, d)
        self.blocks = nn.ModuleList(Block(arm, d, n_heads, hidden) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 256, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

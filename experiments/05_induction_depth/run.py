"""Depth, not one store: induction needs two composed attention reads.

Task: a random segment of L tokens (L ~ U[8, 24], vocab 128) repeated to fill 64 positions. Every token
after the first period is predictable by copying: find the previous occurrence of the current token
and output the token that followed it. Because L varies, position alone cannot solve it.

One attention read cannot do this: it can find the earlier occurrence of the current token but reads
that token back, not its successor. Two layers can: layer 1 writes "my previous token" into each
position (a previous-token head), layer 2 matches on it (the induction head).

Models: attention-only transformers (memstore.layers.Attention, no FFN), 1 and 2 layers, d = 64, 2 heads.
Reported: accuracy on the predictable positions, and for the 2-layer model the mean attention weight
each layer-1 head puts on the previous position.
"""

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn

from memstore import kernels, normalize
from memstore.layers import Attention

VOCAB, SEQ, D, HEADS = 128, 64, 64, 2


class AttnOnly(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D)
        self.pos = nn.Embedding(SEQ, D)
        self.norms = nn.ModuleList(nn.LayerNorm(D) for _ in range(layers))
        self.attns = nn.ModuleList(Attention(D, HEADS) for _ in range(layers))
        self.final = nn.LayerNorm(D)
        self.unembed = nn.Linear(D, VOCAB, bias=False)

    def forward(self, idx, return_weights=False):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        weights = []
        for norm, attn in zip(self.norms, self.attns):
            h = norm(x)
            if return_weights:
                q, k, _ = attn.split(h)
                T = idx.shape[1]
                weights.append(
                    normalize(
                        q @ k.transpose(-1, -2),
                        kernels.Exp(1 / math.sqrt(attn.dh)),
                        "sum",
                        mask=torch.ones(T, T, dtype=torch.bool).tril(),
                    )
                )
            x = x + attn(h)
        logits = self.unembed(self.final(x))
        return (logits, weights) if return_weights else logits


def batch(n, gen):
    lengths = torch.randint(8, 25, (n,), generator=gen)
    segs = torch.randint(VOCAB, (n, 24), generator=gen)
    pos = torch.arange(SEQ)
    idx = segs[torch.arange(n)[:, None], pos[None] % lengths[:, None]]
    predictable = pos[None, 1:] >= lengths[:, None]  # target at t+1 is a repeat
    return idx, predictable


def loss_and_acc(model, idx, predictable):
    logits = model(idx)[:, :-1]
    target = idx[:, 1:]
    ce = nn.functional.cross_entropy(logits[predictable], target[predictable])
    acc = (logits.argmax(-1)[predictable] == target[predictable]).float().mean()
    return ce, acc


def train(layers, steps, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = AttnOnly(layers)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    curve = []
    for step in range(steps):
        loss, acc = loss_and_acc(model, *batch(64, gen))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            curve.append((step, loss.item(), acc.item()))
    with torch.no_grad():
        idx, pred = batch(2000, torch.Generator().manual_seed(10_000 + seed))
        _, acc = loss_and_acc(model, idx, pred)
        _, weights = model(idx[:200], return_weights=True)
        prev_token = weights[0].diagonal(offset=-1, dim1=-2, dim2=-1).mean(dim=(0, 2)).tolist()
    return model, acc.item(), prev_token, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("results/05_induction_depth"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    rows, curves = [], {}
    for layers in (1, 2):
        for seed in range(args.seeds):
            _, acc, prev, curve = train(layers, args.steps, seed)
            rows.append(
                {
                    "layers": layers,
                    "seed": seed,
                    "accuracy": round(acc, 4),
                    "layer1_prev_token_weight": " ".join(f"{p:.2f}" for p in prev),
                }
            )
            curves[(layers, seed)] = curve
            print(
                f"{layers} layer(s), seed {seed}: copy accuracy {acc:6.1%}; "
                f"layer-1 heads' weight on previous position: {', '.join(f'{p:.2f}' for p in prev)}",
                flush=True,
            )

    with open(args.out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nchance accuracy {1 / VOCAB:.1%}")
    plot(curves, args.seeds, args.out / "curves.png")
    print(f"wrote {args.out}")


def plot(curves, seeds, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, layers in enumerate((1, 2)):
        for seed in range(seeds):
            c = curves[(layers, seed)]
            ax.plot(
                [s for s, _, _ in c],
                [a for _, _, a in c],
                color=f"C{i}",
                alpha=0.7,
                label=f"{layers} layer{'s' if layers > 1 else ''}" if seed == 0 else None,
            )
    ax.set(xlabel="step", ylabel="copy accuracy (train batch)", title="attention-only, induction task")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()

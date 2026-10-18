"""Trained FFN capacity: how many random key -> value pairs an FFN memorises by gradient descent.

The capacities in the grid (ReLU ~ 128, GLU ~ 2,700) are for constructed stores: one hidden unit per
memory, keys and values written in directly. Here the same layers are trained instead, at matched
parameter counts (2 h d), and recall is measured as in experiment 02 (nearest stored value).

Layers, built from memstore:
  relu      FFN, ReLU kernel, no normalisation               hidden h
  swiglu    SwiGLU FFN                                       hidden 2h/3 (three matrices)
  softmax   FFN with exp kernel and sum normalisation        hidden h    (FF-attn, attention over weights)

Also reported: the constructed store with h = M (keys and values written in, untrained), for comparison.
"""

import argparse
import csv
import time
from pathlib import Path

import torch

from memstore import kernels, read
from memstore.layers import FFN, SwiGLUFFN

D = 64


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def make_layer(kind, h):
    if kind == "relu":
        return FFN(D, h)
    if kind == "swiglu":
        return SwiGLUFFN(D, (2 * h) // 3)
    if kind == "softmax":
        return FFN(D, h, kernel=kernels.Exp(1.0), norm="sum")
    raise ValueError(kind)


def recall(out, V):
    return ((unit(out) @ V.T).argmax(-1) == torch.arange(len(V))).float().mean().item()


def train(kind, h, K, V, steps, lr, seed):
    torch.manual_seed(seed)
    layer = make_layer(kind, h)
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        loss = ((layer(K) - V) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        out = layer(K)
        return recall(out, V), ((out - V) ** 2).sum(-1).mean().item()


CONSTRUCTED = {"relu": kernels.relu, "relu^2 (GLU approx.)": kernels.relu2, "softmax beta=8": kernels.Exp(8.0)}


def constructed(M, gen):
    K = unit(torch.randn(M, D, generator=gen))
    V = unit(torch.randn(M, D, generator=gen))
    return {name: recall(read(K, K, V, k), V) for name, k in CONSTRUCTED.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--widths", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--max-m", type=int, default=16384)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/03_trained_ffn_capacity"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator().manual_seed(args.seed)
    ms = [2**p for p in range(5, args.max_m.bit_length())]
    rows = []
    for h in args.widths:
        for kind in ("relu", "swiglu", "softmax"):
            fails = 0
            for M in ms:
                K = unit(torch.randn(M, D, generator=gen))
                V = unit(torch.randn(M, D, generator=gen))
                t0 = time.time()
                acc, mse = train(kind, h, K, V, args.steps, args.lr, args.seed)
                rows.append(
                    {"layer": kind, "h": h, "params": 2 * h * D, "M": M, "recall": round(acc, 4), "mse": round(mse, 4)}
                )
                print(
                    f"h={h:<5} {kind:<8} M={M:<6} recall {acc:6.1%}  mse {mse:.3f}  ({time.time() - t0:.0f}s)",
                    flush=True,
                )
                fails = fails + 1 if acc < 0.5 else 0
                if fails == 2:
                    break

    with open(args.out / "trained.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("\nLargest M recalled at >= 95%, and memories per hidden unit of the ReLU-width budget:")
    for h in args.widths:
        cells = []
        for kind in ("relu", "swiglu", "softmax"):
            ok = [r["M"] for r in rows if r["layer"] == kind and r["h"] == h and r["recall"] >= 0.95]
            best = max(ok, default=0)
            cells.append(f"{kind} {best:>6} ({best / h:5.1f}/unit)")
        print(f"h={h:<5} params={2 * h * D:>7} | " + " | ".join(cells))

    print("\nConstructed store (h = M, keys and values written in, untrained): recall")
    crows = []
    for M in ms:
        c = constructed(M, gen)
        crows.append({"M": M, **c})
        print(f"M={M:<6} " + "  ".join(f"{n} {a:6.1%}" for n, a in c.items()))
    with open(args.out / "constructed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=crows[0].keys())
        w.writeheader()
        w.writerows(crows)

    plot(rows, args.widths, args.out / "trained_ffn.png")
    print(f"\nwrote {args.out}")


def plot(rows, widths, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(widths), figsize=(5 * len(widths), 4), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, h in zip(axes, widths):
        for kind in ("relu", "swiglu", "softmax"):
            rs = [r for r in rows if r["layer"] == kind and r["h"] == h]
            ax.plot([r["M"] for r in rs], [r["recall"] for r in rs], marker="o", ms=3, label=kind)
        ax.axvline(h, color="gray", lw=0.8, ls=":")
        ax.set(xscale="log", xlabel="stored pairs M", title=f"hidden {h} ({2 * h * D:,} params)")
    axes[0].set_ylabel("nearest-value recall")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()

"""Bahdanau's additive (MLP) score vs a dot-product score, as the similarity inside one softmax read.

Task: T keys in d = 8; the query is one key plus Gaussian noise; the read must put its weight on that key.
The scorer is trained with cross-entropy on the read weights (values = one-hot positions).

Two regimes:
  equal norms    keys on the unit sphere. Nearest key = most aligned key; a bilinear score suffices.
  varied norms   key norms ~ U(0.3, 3). Nearest key needs -|k|^2, which a bilinear q^T W k cannot express.

Scorers (all with the exp kernel and sum normalisation):
  dot        (W_q q) . (W_k k) / sqrt(d_h)                       Vaswani-style
  additive   v . tanh(W q + U k + b)                             Bahdanau 2014
  neg-L2     -beta |q - k|^2, beta learned                       distance kernel, as in kNN-LM

Also timed: one forward pass of each scorer at T = 1,024 keys.
"""

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from memstore import kernels, normalize
from memstore.layers import AdditiveScore

D, T, NOISE = 8, 16, 0.3


class DotScore(nn.Module):
    def __init__(self, dh=16):
        super().__init__()
        self.q, self.k, self.dh = nn.Linear(D, dh, bias=False), nn.Linear(D, dh, bias=False), dh

    def forward(self, x, K):
        return self.q(x) @ self.k(K).transpose(-1, -2) / math.sqrt(self.dh)


class NegL2Score(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_beta = nn.Parameter(torch.zeros(()))

    def forward(self, x, K):
        return -self.log_beta.exp() * torch.cdist(x, K) ** 2


SCORERS = {"dot": DotScore, "additive": lambda: AdditiveScore(D, D, hidden=32), "neg-L2": NegL2Score}


def batch(n, varied, gen):
    K = torch.randn(n, T, D, generator=gen)
    K = K / K.norm(dim=-1, keepdim=True)
    if varied:
        K = K * (0.3 + 2.7 * torch.rand(n, T, 1, generator=gen))
    target = torch.randint(T, (n,), generator=gen)
    q = K[torch.arange(n), target] + NOISE * torch.randn(n, D, generator=gen)
    return q.unsqueeze(1), K, target


def accuracy_and_loss(scorer, q, K, target):
    n = normalize(scorer(q, K), kernels.Exp(1.0), "sum").squeeze(1)
    loss = -n[torch.arange(len(target)), target].clamp_min(1e-12).log().mean()
    return (n.argmax(-1) == target).float().mean(), loss


def run(name, varied, steps, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    scorer = SCORERS[name]()
    opt = torch.optim.Adam(scorer.parameters(), lr=3e-3)
    curve = []
    for step in range(steps):
        _, loss = accuracy_and_loss(scorer, *batch(256, varied, gen))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            curve.append(loss.item())
    with torch.no_grad():
        q, K, target = batch(20_000, varied, torch.Generator().manual_seed(10_000 + seed))
        acc, loss = accuracy_and_loss(scorer, q, K, target)
        # Oracle: nearest key under L2, the best any scorer can do with this noise.
        oracle = (torch.cdist(q, K).squeeze(1).argmin(-1) == target).float().mean()
    return acc.item(), loss.item(), oracle.item(), curve


def time_forward(name, keys=1024, queries=256, reps=5):
    scorer = SCORERS[name]()
    q, K = torch.randn(1, queries, D), torch.randn(1, keys, D)
    with torch.no_grad():
        scorer(q, K)
        t0 = time.perf_counter()
        for _ in range(reps):
            scorer(q, K)
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("results/04_bahdanau_vs_dot"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    rows, curves = [], {}
    for varied in (False, True):
        regime = "varied norms" if varied else "equal norms"
        for name in SCORERS:
            for seed in range(args.seeds):
                acc, loss, oracle, curve = run(name, varied, args.steps, seed)
                rows.append(
                    {
                        "regime": regime,
                        "scorer": name,
                        "seed": seed,
                        "accuracy": round(acc, 4),
                        "nll": round(loss, 4),
                        "oracle": round(oracle, 4),
                    }
                )
                curves[(regime, name, seed)] = curve

    with open(args.out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"Accuracy picking the source key (T = {T}, d = {D}, noise {NOISE}), mean over {args.seeds} seeds")
    for regime in ("equal norms", "varied norms"):
        rs = [r for r in rows if r["regime"] == regime]
        print(f"\n{regime} (L2-nearest oracle {sum(r['oracle'] for r in rs) / len(rs):.1%})")
        for name in SCORERS:
            accs = [r["accuracy"] for r in rs if r["scorer"] == name]
            nlls = [r["nll"] for r in rs if r["scorer"] == name]
            print(
                f"  {name:<9} acc {sum(accs) / len(accs):6.1%} (min {min(accs):.1%})  nll {sum(nlls) / len(nlls):.3f}"
            )

    print("\nForward time, 256 queries x 1,024 keys:")
    for name in SCORERS:
        print(f"  {name:<9} {time_forward(name):7.2f} ms")

    plot(curves, args.seeds, args.out / "curves.png")
    print(f"\nwrote {args.out}")


def plot(curves, seeds, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, regime in zip(axes, ("equal norms", "varied norms")):
        for i, name in enumerate(SCORERS):
            for seed in range(seeds):
                c = curves[(regime, name, seed)]
                ax.plot(
                    [100 * j for j in range(len(c))], c, color=f"C{i}", alpha=0.7, label=name if seed == 0 else None
                )
        ax.set(title=regime, xlabel="step", yscale="log")
    axes[0].set_ylabel("training NLL")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()

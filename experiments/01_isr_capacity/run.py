"""Interference-to-signal ratio of one memory read, kernel x store size M x query cosine.

Store M random unit keys and random unit values in d_k = d_v = 64. Query with the target key k_0,
optionally noised to cosine c. With w_i = sigma(k_i . x):

    signal       = w_0 v_0
    interference = sum_{i>0} w_i v_i
    ISR          = |interference| / |signal|          (recall fails above 1)

Normalisation rescales every weight by the same factor, so it does not change the ISR.
The analytic estimate is ISR ~ sqrt((M - 1) E[sigma(s)^2]) / sigma(c), s ~ N(0, 1/d_k).
For exp(beta s) this gives the capacity M ~ exp(2 beta (c - beta / d_k)).

Replaces the `isr.py` referenced in memory-store-grid.md section 6.
"""

import argparse
import csv
import math
from pathlib import Path

import torch

from memstore import kernels

D = 64
KERNELS = [kernels.linear, kernels.relu, kernels.relu2, kernels.relu3, kernels.Exp(8.0)]
COSINES = [1.0, 0.89, 0.71]
MS = [2**p for p in range(4, 17)]

# memory-store-grid.md section 6: (kernel, cosine) -> {M: median ISR}
REFERENCE = {
    ("linear", 1.0): {64: 0.99, 128: 1.38, 2048: 5.73, 16384: 16.19},
    ("relu", 1.0): {64: 0.65, 128: 0.99, 2048: 3.92, 16384: 10.96},
    ("relu^2", 1.0): {64: 0.14, 128: 0.21, 2048: 0.83, 16384: 2.41},
    ("relu^3", 1.0): {64: 0.04, 128: 0.05, 2048: 0.22, 16384: 0.65},
    ("exp(beta=8)", 1.0): {64: 0.01, 128: 0.01, 2048: 0.04, 16384: 0.11},
    ("linear", 0.89): {64: 1.09, 128: 1.56, 2048: 6.28, 16384: 18.08},
    ("relu^2", 0.89): {64: 0.18, 128: 0.25, 2048: 1.06, 16384: 2.93},
    ("exp(beta=8)", 0.89): {64: 0.01, 128: 0.02, 2048: 0.09, 16384: 0.27},
    ("linear", 0.71): {64: 1.40, 128: 1.94, 2048: 8.09, 16384: 22.40},
    ("relu^2", 0.71): {64: 0.29, 128: 0.41, 2048: 1.68, 16384: 4.70},
    ("exp(beta=8)", 0.71): {64: 0.06, 128: 0.10, 2048: 0.45, 16384: 1.19},
}


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def noisy_query(k0, c, gen):
    """Unit query at cosine c to k0: c k0 + sqrt(1 - c^2) u, with u a unit vector orthogonal to k0."""
    u = torch.randn(k0.shape, generator=gen, dtype=k0.dtype)
    u = unit(u - (u @ k0) * k0)
    return c * k0 + math.sqrt(1 - c * c) * u


def isr_curves(trials: int, ms: list[int], seed: int = 0):
    """Returns {(kernel, cosine): {M: [ISR per trial]}}. Each trial shares one store across all settings."""
    gen = torch.Generator().manual_seed(seed)
    out = {(k.name, c): {m: [] for m in ms} for k in KERNELS for c in COSINES}
    for _ in range(trials):
        keys = unit(torch.randn(max(ms), D, generator=gen, dtype=torch.float64))
        values = unit(torch.randn(max(ms), D, generator=gen, dtype=torch.float64))
        for c in COSINES:
            x = noisy_query(keys[0], c, gen)
            s = keys @ x
            for kern in KERNELS:
                w = kern(s)
                # Prefix sums give the interference vector for every M in one pass.
                contrib = torch.cumsum(w[1:, None] * values[1:], dim=0)
                signal = (w[0] * values[0]).norm().item()
                for m in ms:
                    out[(kern.name, c)][m].append(contrib[m - 2].norm().item() / signal)
    return out


def analytic(kern, c, m):
    return math.sqrt((m - 1) * kern.second_moment(D)) / kern(torch.tensor(c, dtype=torch.float64)).item()


def median(xs):
    return torch.tensor(xs).median().item()


def median_se(xs, resamples: int = 2000, seed: int = 0):
    """Bootstrap standard error of the median."""
    t = torch.tensor(xs)
    idx = torch.randint(len(xs), (resamples, len(xs)), generator=torch.Generator().manual_seed(seed))
    return t[idx].median(dim=1).values.std().item()


def agrees(got, se, ref):
    """Both medians are noisy (the notes' run is another sample of the same size), and the notes round to 0.01."""
    return abs(got - ref) <= 3 * math.sqrt(2) * se + 0.01


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/01_isr_capacity"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # The per-trial ops are small; many threads spend more time synchronising than computing.
    torch.set_num_threads(4)

    curves = isr_curves(args.trials, MS, args.seed)
    by_name = {k.name: k for k in KERNELS}

    with open(args.out / "isr.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "cosine", "M", "median_isr", "analytic_isr"])
        for (name, c), row in curves.items():
            for m, xs in row.items():
                w.writerow([name, c, m, f"{median(xs):.4f}", f"{analytic(by_name[name], c, m):.4f}"])

    print(f"{'kernel':<12} {'cos':>5} | " + " ".join(f"{'M=' + str(m):>16}" for m in (64, 128, 2048, 16384)))
    print(f"{'':<12} {'':>5} | " + " ".join(f"{'run / notes':>16}" for _ in range(4)))
    failures = []
    for (name, c), ref in REFERENCE.items():
        cells = []
        for m, r in ref.items():
            xs = curves[(name, c)][m]
            got, se = median(xs), median_se(xs)
            ok = agrees(got, se, r)
            if not ok:
                failures.append((name, c, m, got, r, se))
            cells.append(f"{got:7.2f} / {r:6.2f}{' ' if ok else '!'}")
        print(f"{name:<12} {c:>5} | " + " ".join(cells))
    print(f"\n{len(failures)} of {sum(map(len, REFERENCE.values()))} cells outside 3*sqrt(2)*SE + 0.01 of the notes")
    for name, c, m, got, r, se in failures:
        print(f"  {name} cos {c} M={m}: run {got:.3f}, notes {r:.2f}, SE {se:.3f}")

    for c in COSINES:
        beta = 8.0
        pred = math.exp(2 * beta * (c - beta / D))
        xs = curves[("exp(beta=8)", c)]
        crossing = next((m for m in MS if median(xs[m]) > 1.0), None)
        print(f"softmax beta=8, cos {c}: formula capacity ~ {pred:,.0f}; first M (power of 2) with ISR > 1: {crossing}")

    plot(curves, args.out / "isr.png")
    print(f"\nwrote {args.out / 'isr.csv'} and {args.out / 'isr.png'}")


def plot(curves, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(COSINES), figsize=(5 * len(COSINES), 4), sharey=True)
    for ax, c in zip(axes, COSINES):
        for kern in KERNELS:
            ax.plot(MS, [median(curves[(kern.name, c)][m]) for m in MS], marker="o", ms=3, label=kern.name)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax.set(xscale="log", yscale="log", xlabel="stored memories M", title=f"query cosine {c}")
    axes[0].set_ylabel("median interference / signal")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()

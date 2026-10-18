"""Write rules for a fixed d_v x d_k store: Hebbian (linear attention), delta (DeltaNet), least squares.

Three measurements, d_k = d_v = 64, random unit keys and values:

1. Capacity: store M pairs, read every key back. Report median ISR (as in experiment 01: the component of
   the readout off the target value over the component along it) and nearest-value recall accuracy.
2. Recency: for M = 256, recall accuracy by write position. The delta rule should favour recent writes.
3. Overwrite: store 128 pairs, then rewrite a quarter of the keys with new values. Report the cosine of
   the readout to the new value and to the stale one.
"""

import argparse
import csv
from pathlib import Path

import torch

from memstore import writes

D = 64
MS = [16, 32, 64, 96, 128, 192, 256, 512, 1024]
RULES = {
    "hebbian": writes.hebbian,
    "delta beta=1": lambda K, V: writes.delta(K, V, beta=1.0),
    "delta beta=0.1 x20": lambda K, V: writes.delta(K, V, beta=0.1, passes=20),
    "least squares": writes.least_squares,
}


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def store(M, gen):
    return (
        unit(torch.randn(M, D, generator=gen, dtype=torch.float64)),
        unit(torch.randn(M, D, generator=gen, dtype=torch.float64)),
    )


def evaluate(S, K, V):
    """Per-pair ISR and whether the readout's nearest stored value is the target."""
    R = K @ S.T  # readouts (M, d_v)
    along = (R * V).sum(-1)  # component along the target value
    off = (R - along[:, None] * V).norm(dim=-1)
    isr = off / along.abs().clamp_min(1e-12)
    hit = (unit(R) @ V.T).argmax(-1) == torch.arange(len(V))
    return isr, hit


def capacity(trials, gen):
    rows = []
    for M in MS:
        for name, rule in RULES.items():
            isr, acc = [], []
            for _ in range(trials):
                K, V = store(M, gen)
                i, h = evaluate(rule(K, V), K, V)
                isr.append(i.median().item())
                acc.append(h.double().mean().item())
            rows.append(
                {"M": M, "rule": name, "median_isr": torch.tensor(isr).median().item(), "recall": sum(acc) / len(acc)}
            )
    return rows


def recency(trials, gen, M=256, bins=8):
    out = {}
    for name, rule in RULES.items():
        hits = torch.zeros(M, dtype=torch.float64)
        for _ in range(trials):
            K, V = store(M, gen)
            hits += evaluate(rule(K, V), K, V)[1].double()
        out[name] = (hits / trials).view(bins, -1).mean(-1).tolist()
    return out


def overwrite(trials, gen, M=128, frac=0.25):
    out = {}
    n = int(M * frac)
    for name, rule in RULES.items():
        new_cos, old_cos = [], []
        for _ in range(trials):
            K, V = store(M, gen)
            V_new = unit(torch.randn(n, D, generator=gen, dtype=torch.float64))
            S = rule(torch.cat([K, K[:n]]), torch.cat([V, V_new]))
            R = unit(K[:n] @ S.T)
            new_cos.append((R * V_new).sum(-1).mean().item())
            old_cos.append((R * V[:n]).sum(-1).mean().item())
        out[name] = (sum(new_cos) / trials, sum(old_cos) / trials)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/02_write_rules"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    gen = torch.Generator().manual_seed(args.seed)

    rows = capacity(args.trials, gen)
    with open(args.out / "capacity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("1. Capacity: median ISR / nearest-value recall")
    print(f"{'M':>5} | " + " | ".join(f"{n:>18}" for n in RULES))
    for M in MS:
        cells = [r for r in rows if r["M"] == M]
        print(f"{M:>5} | " + " | ".join(f"{r['median_isr']:9.2f} / {r['recall']:6.1%}" for r in cells))

    rec = recency(args.trials, gen)
    print("\n2. Recency, M = 256: recall by write position (oldest eighth -> newest eighth)")
    for name, accs in rec.items():
        print(f"{name:>20} | " + " ".join(f"{a:5.0%}" for a in accs))

    ow = overwrite(args.trials, gen)
    print("\n3. Overwrite: 128 pairs, then 32 keys rewritten. Cosine of readout to new / stale value")
    for name, (new, old) in ow.items():
        print(f"{name:>20} | new {new:5.2f}   stale {old:5.2f}")

    plot(rows, rec, args.out / "write_rules.png")
    print(f"\nwrote {args.out}/capacity.csv and write_rules.png")


def plot(rows, rec, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4))
    for name in RULES:
        rs = [r for r in rows if r["rule"] == name]
        a.plot([r["M"] for r in rs], [r["median_isr"] for r in rs], marker="o", ms=3, label=name)
        b.plot([r["M"] for r in rs], [r["recall"] for r in rs], marker="o", ms=3, label=name)
        c.plot(range(1, len(rec[name]) + 1), rec[name], marker="o", ms=3, label=name)
    a.axhline(1.0, color="gray", lw=0.8, ls="--")
    for ax in (a, b):
        ax.axvline(D, color="gray", lw=0.8, ls=":")
        ax.set(xscale="log", xlabel="stored pairs M")
    a.set(
        yscale="log",
        ylim=(1e-2, 30),
        ylabel="median ISR (least squares is ~1e-15 below d_k)",
        title="interference (d_k = 64 dotted)",
    )
    b.set(ylabel="nearest-value recall", title="recall")
    c.set(xlabel="write position (eighths, oldest -> newest)", ylabel="recall", title="recency, M = 256")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()

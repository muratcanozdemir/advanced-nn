"""Summarise the 06 batch: per-arm mean and spread of val/test bpc over seeds, and validation curves."""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import ARMS  # noqa: E402

# Sukhbaatar et al. 2019, text8 ablation ordering, best to worst (section 3.2 of the notes).
PAPER_ORDER = "all-attn beats attn-split; head-split worse; FF-attn worse than all-attn; N=0 poor"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/06_persistent_memory"))
    args = ap.parse_args()

    results = [json.loads(p.read_text()) for p in sorted(args.out.glob("*_s*/result.json"))]
    if not results:
        print("no finished runs yet")
        return
    by_arm = {a: [r for r in results if r["arm"] == a] for a in ARMS}

    print(f"{'arm':<11} {'params':>10} {'seeds':>5} {'val bpc':>16} {'test bpc':>16}  per-seed test")
    rows = []
    for arm, rs in sorted(
        by_arm.items(), key=lambda kv: statistics.mean(r["test_bpc"] for r in kv[1]) if kv[1] else 99
    ):
        if not rs:
            print(f"{arm:<11} {'—':>10} {0:>5}")
            continue
        val = [r["val_bpc"] for r in rs]
        test = [r["test_bpc"] for r in rs]
        sd = lambda xs: statistics.stdev(xs) if len(xs) > 1 else float("nan")  # noqa: E731
        print(
            f"{arm:<11} {rs[0]['params']:>10,} {len(rs):>5} {statistics.mean(val):8.4f} ± {sd(val):.4f}"
            f" {statistics.mean(test):8.4f} ± {sd(test):.4f}  " + " ".join(f"{t:.4f}" for t in test)
        )
        rows.append(
            {
                "arm": arm,
                "params": rs[0]["params"],
                "seeds": len(rs),
                "val_mean": statistics.mean(val),
                "val_sd": sd(val),
                "test_mean": statistics.mean(test),
                "test_sd": sd(test),
            }
        )
    print(f"\npaper (text8 ablation): {PAPER_ORDER}")

    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    plot(args.out)
    print(f"wrote {args.out}/summary.csv and curves.png")


def plot(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, arm in enumerate(ARMS):
        for j, log in enumerate(sorted(out.glob(f"{arm}_s*/log.csv"))):
            with open(log) as f:
                rs = list(csv.DictReader(f))
            if rs:
                ax.plot(
                    [int(r["tokens"]) / 1e6 for r in rs],
                    [float(r["val_bpc"]) for r in rs],
                    color=f"C{i}",
                    alpha=0.7,
                    label=arm if j == 0 else None,
                )
    ax.set(xlabel="training bytes (M)", ylabel="val bpc (256 KB sample)", title="enwik8, ~3.3M params, 4 layers")
    ax.set_ylim(top=3.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=120)


if __name__ == "__main__":
    main()

"""Train one arm on enwik8 bytes; resumable. Writes <out>/<arm>_s<seed>/{ckpt.pt, log.csv, result.json}.

Splits follow the standard 90M / 5M / 5M. Evaluation uses non-overlapping 256-byte windows (no sliding
context), so bpc is higher than a sliding-window evaluation would report.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from model import ARMS, ByteLM  # noqa: E402

LN2 = math.log(2)


def load_splits(path):
    data = torch.frombuffer(bytearray(Path(path).read_bytes()), dtype=torch.uint8)
    return data[:90_000_000], data[90_000_000:95_000_000], data[95_000_000:]


def random_batch(data, batch, context, gen):
    starts = torch.randint(len(data) - context - 1, (batch,), generator=gen)
    idx = starts[:, None] + torch.arange(context + 1)
    chunk = data[idx].long()
    return chunk[:, :-1], chunk[:, 1:]


@torch.no_grad()
def evaluate(model, data, context, max_bytes, batch=32):
    model.eval()
    n = min(len(data) - 1, max_bytes) // context
    total, count = 0.0, 0
    for i in range(0, n, batch):
        starts = torch.arange(i, min(i + batch, n)) * context
        idx = starts[:, None] + torch.arange(context + 1)
        chunk = data[idx].long()
        logits = model(chunk[:, :-1])
        total += F.cross_entropy(logits.reshape(-1, 256), chunk[:, 1:].reshape(-1), reduction="sum").item()
        count += chunk[:, 1:].numel()
    model.train()
    return total / count / LN2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", type=float, default=10e6, help="training bytes")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-bytes", type=int, default=256 * 1024)
    ap.add_argument("--final-val-bytes", type=int, default=5_000_000)
    ap.add_argument("--test-bytes", type=int, default=2 * 1024 * 1024)
    ap.add_argument("--ckpt-minutes", type=float, default=10)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--data", type=Path, default=Path("data/enwik8"))
    ap.add_argument("--out", type=Path, default=Path("results/06_persistent_memory"))
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    run_dir = args.out / f"{args.arm}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "result.json").exists():
        print(f"{run_dir} already done")
        return

    train, val, test = load_splits(args.data)
    torch.manual_seed(args.seed)
    model = ByteLM(args.arm, args.layers, args.d, args.heads, args.hidden, args.context)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    steps = int(args.tokens // (args.batch * args.context))
    warmup = min(200, steps // 10)

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        return args.lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    gen = torch.Generator().manual_seed(1000 + args.seed)
    step, elapsed = 0, 0.0
    ckpt = run_dir / "ckpt.pt"
    if ckpt.exists():
        state = torch.load(ckpt, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        gen.set_state(state["gen"])
        step, elapsed = state["step"], state["elapsed"]
        print(f"resumed {run_dir} at step {step}/{steps}")

    log = open(run_dir / "log.csv", "a")
    if step == 0:
        log.write("step,tokens,train_bpc,val_bpc,elapsed_s\n")
    print(f"{args.arm} seed {args.seed}: {params:,} params, {steps} steps of {args.batch}x{args.context}", flush=True)

    t0, last_ckpt, running = time.time(), time.time(), []
    while step < steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = random_batch(train, args.batch, args.context, gen)
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        running.append(loss.item() / LN2)
        if not math.isfinite(running[-1]):
            raise RuntimeError(f"non-finite loss at step {step}")

        if step % args.eval_every == 0 or step == steps:
            v = evaluate(model, val, args.context, args.val_bytes)
            now = elapsed + time.time() - t0
            log.write(
                f"{step},{step * args.batch * args.context},{sum(running) / len(running):.4f},{v:.4f},{now:.0f}\n"
            )
            log.flush()
            print(
                f"  step {step}/{steps} train {sum(running) / len(running):.3f} val {v:.3f} bpc "
                f"({step * args.batch * args.context / now:,.0f} bytes/s)",
                flush=True,
            )
            running = []

        if time.time() - last_ckpt > args.ckpt_minutes * 60 or step == steps:
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "gen": gen.get_state(),
                    "step": step,
                    "elapsed": elapsed + time.time() - t0,
                },
                ckpt,
            )
            last_ckpt = time.time()

    elapsed += time.time() - t0
    result = {
        "arm": args.arm,
        "seed": args.seed,
        "params": params,
        "tokens": steps * args.batch * args.context,
        "val_bpc": evaluate(model, val, args.context, args.final_val_bytes),
        "test_bpc": evaluate(model, test, args.context, args.test_bytes),
        "train_seconds": round(elapsed),
        "config": {k: str(v) for k, v in vars(args).items()},
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"done: val {result['val_bpc']:.4f} test {result['test_bpc']:.4f} bpc", flush=True)


if __name__ == "__main__":
    main()

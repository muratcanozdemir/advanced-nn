"""Run the 06 batch: arms x seeds, seed-major, on one or more pinned workers sharing a claim-file queue.

Launch one process per worker, e.g.
    run_all.py --cpus 0-7  --threads 8 &
    run_all.py --cpus 8-15 --threads 8 &
Each worker claims the next unfinished run by creating <run_dir>/claim exclusively, runs train.py under
taskset, and moves on. Runs resume from their checkpoint; a claim whose owner process is gone is retaken.
Extra arguments after `--` are passed to train.py.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import ARMS  # noqa: E402

TRAIN = Path(__file__).parent / "train.py"


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def claim(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "claim"
    if path.exists():
        try:
            owner = int(path.read_text().strip() or 0)
        except ValueError:
            owner = 0
        if alive(owner):
            return False
        path.unlink(missing_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpus", required=True, help="taskset CPU list, e.g. 0-7")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", type=Path, default=Path("results/06_persistent_memory"))
    args, extra = ap.parse_known_args()
    extra = [a for a in extra if a != "--"]

    for seed in range(args.seeds):
        for arm in args.arms:
            run_dir = args.out / f"{arm}_s{seed}"
            if (run_dir / "result.json").exists() or not claim(run_dir):
                continue
            cmd = [
                "taskset",
                "-c",
                args.cpus,
                sys.executable,
                str(TRAIN),
                "--arm",
                arm,
                "--seed",
                str(seed),
                "--threads",
                str(args.threads),
                "--out",
                str(args.out),
                *extra,
            ]
            print(f"[{args.cpus}] start {arm} seed {seed}", flush=True)
            with open(run_dir / "stdout.log", "a") as log:
                code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
            print(f"[{args.cpus}] end {arm} seed {seed}: exit {code}", flush=True)
            (run_dir / "claim").unlink(missing_ok=True)


if __name__ == "__main__":
    main()

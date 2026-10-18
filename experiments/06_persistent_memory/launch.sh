#!/usr/bin/env bash
# Overnight batch: two pinned workers (P-cores 0-7, E-cores 8-15), 6 arms x 3 seeds, 16M bytes per run.
# Detached from the terminal and holding a sleep inhibitor. Rerunning resumes unfinished runs.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/06_persistent_memory
mkdir -p "$OUT"
PY=.venv/bin/python
ARGS=(--seeds 3 --out "$OUT" -- --tokens 16e6 --final-val-bytes 1000000)
setsid nohup systemd-inhibit --what=sleep:idle --who=advanced-nn --why="experiment 06 overnight batch" \
  bash -c "$PY experiments/06_persistent_memory/run_all.py --cpus 0-7 --threads 8 ${ARGS[*]} &
           $PY experiments/06_persistent_memory/run_all.py --cpus 8-15 --threads 8 ${ARGS[*]} &
           wait; echo batch finished at \$(date)" >> "$OUT/batch.log" 2>&1 < /dev/null &
echo "launched; follow with: tail -f $OUT/batch.log"

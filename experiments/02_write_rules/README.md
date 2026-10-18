# 02 — Write rules on a fixed store

`uv run python experiments/02_write_rules/run.py` (~15 s)

Fixed d_v × d_k = 64 × 64 linear store, random unit keys and values, 16 trials.

- **Grid capacities confirmed.** Median ISR crosses 1 at M ≈ 64 = d_k for Hebbian (linear attention) and
  at M = 128 = 2d_k for least squares. Least squares is exact (ISR ~ 1e-15) up to M = d_k.
- **ISR = 1 is not where recall fails.** Nearest-value recall stays ≥ 98% until M ≈ 128 (Hebbian) and
  ≈ 192 (least squares). Interference spreads over 64 dimensions, while decoding only has to beat the
  other stored values. A "capacity" figure depends on the decoder as much as on the store.
- **Delta rule, β = 1, one pass (DeltaNet-style)** has the lowest capacity (90% at M = 64) but is the only
  rule that overwrites: after rewriting 32 keys, the readout's cosine is 0.80 to the new value and 0.02 to
  the stale one. Hebbian and least squares blend them (≈ 0.5 / 0.5). Its recall is a recency window of
  roughly the last d_k writes (99% for the newest eighth, 0% for the oldest four eighths at M = 256).
- **Delta rule, β = 0.1, 20 passes** approaches least squares in capacity, with a milder recency tilt.

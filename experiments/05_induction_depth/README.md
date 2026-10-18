# 05 — Induction needs depth

`uv run python experiments/05_induction_depth/run.py` (~10 min)

Attention-only transformers built from `memstore.layers.Attention` (no FFN), d = 64, 2 heads, 2,000 steps.
A random segment of 8–24 tokens is repeated to fill 64 positions; scored on the predictable positions.

| Layers | Copy accuracy (3 seeds) | Layer-1 weight on previous position (per head) |
|---|---|---|
| 1 | 18.4%, 19.0%, 19.1% | 0.02–0.04 |
| 2 | 96.3%, 96.4%, 96.6% | one head per seed at 0.39–0.41; the other 0.06–0.26 |

Chance is 0.8%.

- The 2-layer models jump from ~10% to ~85% between steps 250 and 350 in every seed: the induction phase
  transition. In each seed one layer-1 head puts ~40% of its weight on the previous position, which a
  causal head spreading weight uniformly would not (~1/position, mean ≈ 0.05).
- For §5 of the notes: no single-read capacity number predicts this. Both models have enough store; what
  the 1-layer model lacks is a second read that can use the output of the first.
- Not investigated: what the 1-layer model's 19% comes from (it is well above chance, so some partial
  strategy, e.g. position-based guesses for common segment lengths).

# 01 — Interference-to-signal ratio

`uv run python experiments/01_isr_capacity/run.py` (~45 s; writes `results/01_isr_capacity/`)

Reproduces the §6 table of `memory-store-grid.md`: 43 of 44 cells agree within 3·√2·SE + 0.01.

- **Off cell:** softmax β = 8, cosine 0.71, M = 2,048: run 0.39 (seeds 0–3: 0.39–0.41), notes 0.45,
  analytic 0.42. Building the query without orthogonalising the noise gives 0.37–0.43. The notes'
  value is the likely outlier.
- **Capacity formula confirmed:** softmax at cosine 0.71 crosses ISR = 1 between M = 8,192 and 16,384;
  the formula `exp(2β(c − β/d_k))` gives 11.6k.
- **Not in the notes:** with query noise, the sharp-kernel advantage collapses. At cosine 1.0, softmax
  interferes ~50× less than relu³; at cosine 0.71, under 2× less.
- Medians sit below the √mean analytic estimate for heavy-tailed kernels (relu³, exp at small M).
  `tests/test_isr.py` therefore checks mean(ISR²), which equals the analytic value exactly in expectation.

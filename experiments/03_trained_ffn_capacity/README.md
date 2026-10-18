# 03 — Trained FFN capacity

`uv run python experiments/03_trained_ffn_capacity/run.py` (~60 min on this CPU; `--widths 64 --max-m 2048` for a 2-minute run)

Random unit key → value pairs, d = 64, full-batch Adam for 2,000 steps, matched parameter count 2hd
(SwiGLU gets hidden 2h/3). Recall = the readout's nearest stored value is the target.

| Hidden h (params) | ReLU | SwiGLU | Softmax-FFN | Exact fit (MSE < 0.01), all three |
|---|---|---|---|---|
| 64 (8k) | 512 | 512 | 512 | 64–128 |
| 256 (33k) | 2,048 | 2,048 | 2,048 | 512 |
| 1,024 (131k) | 8,192 | 8,192 | 8,192 | 2,048 |

The figures under the layer names are the largest M recalled at ≥ 95%.

- **Trained, the kernel barely matters; capacity scales with parameters.** All three layers recall about
  8 pairs per hidden unit (one per 16 parameters) at every width. Exact fit stops at M ≈ 2h = params/d,
  which is the counting bound: M·d equations for 2hd unknowns.
- **Constructed stores are kernel-bound instead.** With one hidden unit per memory and keys and values
  written in (no training), ReLU recall falls to 92% at M = 256 and 25% at M = 1,024. ReLU² holds to
  ~4,096, and softmax (β = 8) holds past 16,384. This is the ordering the grid's capacity column describes.
- **For §4.3 of the notes:** the post's case that a sharper kernel buys FFN capacity is about stores
  written one memory per unit. Under gradient descent, memories are spread across units and the advantage
  disappears at matched parameters. That fits Sukhbaatar et al.'s FF-attn result (ReLU → softmax alone
  did not help).
- Past capacity the curves separate slightly (h = 1,024, M = 16,384: ReLU 42%, SwiGLU 60%, softmax 72%).
  That may reflect the fixed 2,000-step budget rather than capacity; not tested.

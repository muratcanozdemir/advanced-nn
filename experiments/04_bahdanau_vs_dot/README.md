# 04 — Bahdanau additive score vs dot product

`uv run python experiments/04_bahdanau_vs_dot/run.py` (~2 min)

One softmax read over 16 keys in d = 8; the query is a key plus noise (σ = 0.3). Three seeds.

| Scorer | Equal norms | Varied norms (0.3–3) | Forward, 256 × 1,024 |
|---|---|---|---|
| dot, (W_q q)·(W_k k) | 83.0% | 60.1% | 0.3 ms |
| additive, v·tanh(Wq + Uk + b) | 82.2% | 92.0% | 51 ms |
| −β‖q − k‖² | 83.0% | 92.4% | 0.7 ms |
| L2-nearest oracle | 83.0% | 92.4% | — |

- With equal key norms, every scorer reaches the oracle: nearest key = most aligned key.
- With varied norms, retrieval needs the −‖k‖² term. A bilinear score can't express it, so dot product
  stalls at 60% across all seeds. The MLP score learns it.
- The cost of the MLP score is that it can't be one matmul: it materialises a (queries × keys × hidden)
  tensor, ~170× slower here. (Timings were taken while another experiment shared the CPU; the ratio is
  indicative.)
- For §3.1: the 2014 MLP score was strictly more expressive per read. Dot product won on cost, and
  transformers recover what it loses through learned projections, multiple heads and layer norm
  (which equalises norms). The distance kernel gets both, and it is the kNN-LM row's kernel.

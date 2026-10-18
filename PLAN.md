# Plan: runnable examples for "FFN and Attention as One Lookup Operation"

Companion code for `memory-store-grid.md`. Every layer in the design-space grid is built from one
primitive, `memstore.read`, configured along the four axes: memory source, kernel, normalisation, write rule.

## Machine constraints

- Intel Core Ultra 5 125H, 18 threads, AVX2 + AVX-VNNI, 15 GB RAM. CPU is the compute (~150–300 GFLOPS fp32).
- Intel Arc iGPU present but no Level Zero runtime and user not in `render`; PyTorch `xpu` is optional, later.
- No CUDA: no flash-attention, no Triton kernels. DeltaNet is written in plain PyTorch.
- Environment: uv project, CPU-only torch wheel pinned in `pyproject.toml`.

## Experiments

| # | Experiment | Tests | CPU time | Status |
|---|---|---|---|---|
| 01 | ISR capacity, kernel × M × query cosine | §6 table, noisy-query formula | minutes | ✅ 43/44 cells match; see experiments/01_isr_capacity/README.md |
| 02 | Write rules on a fixed state: Hebbian, delta, least squares | outer-product / optimal linear / DeltaNet rows | minutes | ✅ Hebbian ISR = 1 at d_k, least squares at 2d_k |
| 03 | Trained FFN capacity: ReLU, SwiGLU, softmax-FFN memorising random pairs | whether constructed capacities hold under gradient descent | ~60 min | ✅ trained: ~8 pairs/unit for every kernel |
| 04 | Bahdanau additive score vs dot-product on toy retrieval | §3.1 | ~2 min | ✅ dot fails when key norms vary (60% vs 92%) |
| 05 | Induction heads, 2-layer attention-only | §5 depth caveat | ~10 min | ✅ 1 layer 19%, 2 layers 96% |
| 06 | Persistent-memory ablation on char-level enwik8/text8, ~2M params: all-attn, attn-split, head-split, FF-attn (matched params and matched depth), N=0, standard FFN; 3 seeds | §3.2 ordering at tiny scale, plus run-to-run variance | ~1 h/arm, overnight | |
| 07 | kNN-LM over hidden states of the 06 checkpoints | kNN-LM row | minutes after 06 | |

Optional: product-key memory and MoE top-k as FFN-slot arms in 06; Arc GPU via PyTorch `xpu`.

## Layout

```
memstore/      kernels.py read.py (kernel, normalisation, read) layers.py writes.py
experiments/   NN_name/run.py, each writes results/NN_name/
tests/         equivalence tests: each layer equals its reference implementation
data/          downloaded corpora (gitignored)
results/       CSV + plots, seeds recorded
```

## Order

1. ✅ Core + tests + experiment 01 (acceptance: §6 table within sampling error).
2. ✅ Experiments 02–05.
3. Char-LM harness, smoke runs, then the 06 overnight batch.
4. Experiment 07.
5. Optional Arc GPU.

## Known limits

- 06 at ~2M params tests ordering only; disagreement with the paper is a scale result, not a failed replication.
- FF-attn in the paper changed depth to match parameters; 06 runs both matched-params and matched-depth variants.

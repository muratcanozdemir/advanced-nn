# 06 — Persistent-memory ablation (Sukhbaatar et al. 2019) at tiny scale

Byte-level LM on enwik8. 4 layers, d = 256, 4 heads, context 256, FFN hidden / persistent memories 1,024.
About 3.3M parameters per arm (n0: 1.2M). 16M training bytes per run, 3 seeds.

| Arm | What changes |
|---|---|
| ffn | standard: attention + ReLU FFN |
| ff-attn | FFN with softmax instead of ReLU, still a separate sublayer |
| all-attn | no FFN; context and 1,024 persistent memories per head in one softmax |
| attn-split | same memories, one softmax per pool, outputs summed |
| head-split | half the heads read context only, half read persistent only |
| n0 | attention only: no FFN, no persistent memories |

Because persistent memories per layer are sized N = FFN hidden, all arms except n0 have the same parameter
count *and* the same depth. The paper's FF-attn arm changed depth to match parameters; this setup has no
such confound.

## Running

```
data/enwik8                                          # 100,000,000 bytes, from mattmahoney.net/dc/enwik8.zip
experiments/06_persistent_memory/launch.sh           # detached two-worker batch (~8 h); rerun to resume
tail -f results/06_persistent_memory/batch.log
uv run python experiments/06_persistent_memory/summarize.py
```

Workers: P-cores (CPUs 0-7) and E-cores (8-15), 8 threads each, ~5k bytes/s each. One 4-thread job is
faster than one 8- or 12-thread job on this hybrid CPU, but two pinned 8-thread workers give the most total
throughput. Softmax reads go through PyTorch's fused attention (`memstore.read` fast path); without it,
the persistent arms ran at half speed.

## Differences from the paper

- ~3.3M params and 16M bytes, against 39–114M params and full enwik8: this tests the ordering only.
- Learned absolute positions; no relative positions, no adaptive span, no Transformer-XL cache.
- Evaluation on non-overlapping 256-byte windows, 1 MB of validation and 2 MB of test: bpc is higher than
  a sliding-window evaluation would give.
- Persistent memories initialised with unit variance; the paper rescales keys by √d_h and values by √N.

## Results

Pending: the batch started 2026-10-18 at 00:03 CEST.

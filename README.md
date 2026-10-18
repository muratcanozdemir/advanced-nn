# advanced-nn

Runnable companion to [`memory-store-grid.md`](memory-store-grid.md): FFN and attention layers treated as one
operation, a similarity-weighted readout of stored values,

```
f(x) = Σ_i n_i · v_i      where  w_i = σ(k_i · x)  and  n = normalize(w)
```

configured along four axes: memory source, kernel σ, normalisation, and write rule. Every layer here is built
from one primitive, `memstore.read`, and each experiment tests one claim from the notes on a CPU.

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12. Torch comes from the CPU-only wheel index (pinned in
`pyproject.toml`), so no CUDA download.

```
uv sync --locked
uv run pytest -q                                       # 85 equivalence and capacity tests, ~6 s
uv run python experiments/01_isr_capacity/run.py       # ~45 s, writes results/01_isr_capacity/
```

## Layout

```
memstore/        kernels.py, read.py (kernel, normalisation, read), layers.py (FFN, attention), writes.py
experiments/     NN_name/run.py + README.md with the findings; each writes results/NN_name/
tests/           each layer equals its reference implementation; ISR matches the analytic value
results/         CSV, plots and logs, seeds recorded (checkpoints are gitignored)
data/            downloaded corpora, gitignored (enwik8 for experiment 06)
```

## Experiments

| # | Question | CPU time | Finding |
|---|---|---|---|
| [01](experiments/01_isr_capacity) | Does the §6 interference-to-signal table hold? | ~45 s | 43 of 44 cells within sampling error; query noise erases most of the sharp-kernel advantage |
| [02](experiments/02_write_rules) | Hebbian vs delta vs least-squares writes | ~15 s | capacities d_k and 2d_k confirmed; only the delta rule overwrites |
| [03](experiments/03_trained_ffn_capacity) | Do constructed FFN capacities survive training? | ~60 min | no: trained, every kernel stores ~8 pairs per hidden unit |
| [04](experiments/04_bahdanau_vs_dot) | Additive (Bahdanau) score vs dot product | ~2 min | dot product stalls at 60% when key norms vary; additive and −‖q−k‖² reach 92% |
| [05](experiments/05_induction_depth) | Does induction need two layers? | ~10 min | 1 layer 19%, 2 layers 96%, with a phase transition near step 300 |
| [06](experiments/06_persistent_memory) | Persistent-memory ablation (Sukhbaatar et al. 2019), ~3.3M params on enwik8 | ~8 h, two workers | standard FFN best (2.40 test bpc); all-attn worst (3.30), the reverse of the paper's ordering at this scale |
| 07 | kNN-LM over the 06 checkpoints | minutes | not started |

Experiment 06 at ~3M parameters and 16M training bytes tests ordering only: a disagreement with the paper is
a result about scale, not a failed replication. `uv run python experiments/06_persistent_memory/summarize.py`
prints the per-arm table. [`PLAN.md`](PLAN.md) has the plan and machine constraints; [`RUNBOOK.md`](RUNBOOK.md)
covers running the long batches, CI, and releases.

## CI and supply chain

| Workflow | Runs on | Does |
|---|---|---|
| `ci.yml` | every push and PR | ruff lint and format check; pytest on Python 3.12 and 3.14; experiments 01, 02, 04, 05 at toy settings |
| `osv-scanner.yml` | push to main, PRs, Mondays 06:17 UTC | scans `uv.lock` against [osv.dev](https://osv.dev); PRs fail only on newly introduced vulnerabilities; findings land in Security › Code scanning |
| `release.yml` | `v*` tags, manual dispatch | builds wheel and sdist, exports a CycloneDX 1.5 SBOM from `uv.lock`, and signs a SLSA v1 build-provenance attestation and an SBOM attestation for each file; tags also publish a GitHub release |

Verify a downloaded release file:

```
gh attestation verify advanced_nn-0.1.0-py3-none-any.whl --repo muratcanozdemir/advanced-nn
gh attestation verify advanced_nn-0.1.0-py3-none-any.whl --repo muratcanozdemir/advanced-nn \
  --predicate-type https://cyclonedx.org/bom
```

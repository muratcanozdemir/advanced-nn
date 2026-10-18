# Runbook

Procedures for running the experiments, keeping CI green, triaging vulnerability reports, and cutting a
release. Commands run from the repository root.

## Setup

```
uv sync --locked          # creates .venv with the CPU-only torch wheel
uv run pytest -q          # 85 tests, ~6 s
```

`--locked` fails if `uv.lock` no longer matches `pyproject.toml`. After changing dependencies, run `uv lock` and
commit both files.

## Short experiments (01–05)

Each `experiments/NN_name/run.py` writes CSV, plots and a log to `results/NN_name/`, overwriting the previous
run. All take `--out`; see `--help` for the toy-size flags CI uses.

| # | Full run | Quick check |
|---|---|---|
| 01 | `uv run python experiments/01_isr_capacity/run.py` (~45 s) | `--trials 2` |
| 02 | `uv run python experiments/02_write_rules/run.py` (~15 s) | `--trials 2` |
| 03 | `uv run python experiments/03_trained_ffn_capacity/run.py` (~60 min) | `--widths 64 --max-m 2048` (~2 min) |
| 04 | `uv run python experiments/04_bahdanau_vs_dot/run.py` (~2 min) | `--steps 20 --seeds 1` |
| 05 | `uv run python experiments/05_induction_depth/run.py` (~10 min) | `--steps 20 --seeds 1` |

Timings are for the Core Ultra 5 125H in `PLAN.md`; anything else sharing the CPU slows them and skews 04's
timing column. When a rerun changes a number, update the experiment's README in the same commit.

## Experiment 06 batch (~8 h)

### Data

```
mkdir -p data && curl -L -o data/enwik8.zip http://mattmahoney.net/dc/enwik8.zip
unzip -o data/enwik8.zip -d data && stat -c %s data/enwik8    # must print 100000000
```

### Launch

```
experiments/06_persistent_memory/launch.sh
tail -f results/06_persistent_memory/batch.log
```

`launch.sh` detaches two workers with `setsid nohup` under `systemd-inhibit` (no sleep while running), pinned
to CPUs 0-7 and 8-15 with 8 threads each. Each worker claims the next unfinished arm × seed by creating
`<run_dir>/claim` with its PID. It uses `.venv/bin/python` directly, so run `uv sync --locked` first.

On a machine with a different core layout, edit the `--cpus` and `--threads` arguments in `launch.sh`. Two
pinned workers gave the most throughput on the hybrid P/E-core CPU; on a homogeneous CPU, one worker per
half of the cores is a reasonable start.

### Monitor

| What | Command |
|---|---|
| Start/end of each run | `tail -f results/06_persistent_memory/batch.log` |
| One run's training log | `tail -f results/06_persistent_memory/<arm>_s<seed>/stdout.log` |
| Workers alive | `pgrep -af 06_persistent_memory` |
| Finished runs | `ls results/06_persistent_memory/*/result.json \| wc -l` (18 when done) |
| Throughput | ~5k bytes/s per worker; a 16M-byte run takes ~55 min |

`batch finished at …` in `batch.log` marks the end.

### Stop, crash, reboot

```
pkill -f experiments/06_persistent_memory/    # stops both workers and their train.py children
experiments/06_persistent_memory/launch.sh    # resumes
```

`train.py` checkpoints every 10 minutes (`ckpt.pt`), so at most ~10 minutes per in-flight run is lost.
Rerunning `launch.sh` skips runs with a `result.json`, resumes the rest from their checkpoints, and retakes
claims whose PID is no longer alive. No manual cleanup is needed after a crash or reboot.

| Symptom | Cause | Fix |
|---|---|---|
| A run is never picked up and no worker is on it | its `claim` holds a PID that was reused by an unrelated live process | delete `<run_dir>/claim`, rerun `launch.sh` |
| A run exits non-zero in `batch.log` | see the end of its `stdout.log` | fix, then rerun `launch.sh`; it resumes from the checkpoint |
| Out of memory | another heavy job on the machine (15 GB total) | stop the other job, or run one worker |
| A changed setting does not take effect | the run resumed from an old checkpoint | delete that run's directory (or the whole `results/06_persistent_memory/<arm>_s*`) and relaunch |

### Summarise

```
uv run python experiments/06_persistent_memory/summarize.py
```

Prints per-arm mean ± sd of validation and test bpc over seeds and writes `summary.csv` and `curves.png`.
Commit those plus each run's `log.csv`, `result.json` and `stdout.log`; `ckpt.pt` is gitignored but keep it
locally, since experiment 07 reads the checkpoints.

## CI failures (`ci.yml`)

| Job | Reproduce locally | Usual fix |
|---|---|---|
| lint | `uvx ruff check . && uvx ruff format --check .` | `uvx ruff check --fix . && uvx ruff format .` |
| test | `uv run pytest -q` | `uv sync --locked` failing means `uv.lock` is stale: `uv lock`, commit |
| smoke | the four commands in `ci.yml`'s smoke step | usually a `memstore` API change that a `run.py` still uses the old way |

Test runs on 3.12 and 3.14; a failure on only one is usually a dependency without wheels for that version yet.
The smoke job uploads its outputs as the `smoke-results` artifact.

## Vulnerability reports (`osv-scanner.yml`)

The scanner reads `uv.lock`. Push and weekly runs report to Security › Code scanning; a PR check fails only
when the PR adds a vulnerability that main does not have.

1. Open the alert and note the package, fixed version, and whether it is a runtime dependency
   (`uv tree --invert --package <name>`). Dev-only (pytest) and build-only findings matter less: nothing here
   is deployed as a service.
2. If a fixed version exists: `uv lock --upgrade-package <name>`, then `uv run pytest -q`, commit `uv.lock`.
   Torch upgrades come from the CPU index; check that the smoke experiments still run.
3. If there is no fix, or it does not apply (e.g. a torch issue in a feature not used here, such as
   `torch.load` of untrusted files — this repo only loads its own checkpoints), record the decision in
   `osv-scanner.toml` at the repo root, with an expiry so it gets reviewed again:

   ```toml
   [[IgnoredVulns]]
   id = "GHSA-xxxx-xxxx-xxxx"
   ignoreUntil = 2027-01-31
   reason = "torch.load on untrusted input; we only load our own checkpoints"
   ```

Reproduce a scan locally with the osv-scanner binary: `osv-scanner scan source -r .`.

## Releases (`release.yml`)

A release packages `memstore` as a wheel and sdist with provenance and an SBOM. The experiments and results
are not in the wheel.

```
uv version --bump minor          # or patch / major; edits pyproject.toml and uv.lock
git commit -am "Release v$(uv version --short)"
git tag "v$(uv version --short)" && git push origin main --tags
```

The workflow refuses a tag that does not match the `pyproject.toml` version. It then:

1. builds `dist/*.whl` and `dist/*.tar.gz` with `uv build` (no restored cache);
2. exports `sbom.cdx.json` (CycloneDX 1.5, runtime dependencies, from `uv.lock`);
3. signs a SLSA v1 build-provenance attestation and a CycloneDX SBOM attestation for each dist file through
   GitHub artifact attestations (Sigstore public-good instance, since the repo is public);
4. creates the GitHub release with the dist files and the SBOM attached.

Running the workflow by hand (Actions › Release › Run workflow) does steps 1–3 and uploads the files as the
`dist` artifact without publishing a release.

### Verify

Needs GitHub CLI 2.49 or later (`gh attestation`).

```
gh release download v0.2.0 --repo muratcanozdemir/advanced-nn --dir /tmp/rel
gh attestation verify /tmp/rel/advanced_nn-0.2.0-py3-none-any.whl --repo muratcanozdemir/advanced-nn
gh attestation verify /tmp/rel/advanced_nn-0.2.0-py3-none-any.whl --repo muratcanozdemir/advanced-nn \
  --predicate-type https://cyclonedx.org/bom
```

The first checks the provenance (built by `release.yml` in this repo, from which commit and ref); the second
checks the SBOM attestation. Add `--signer-workflow muratcanozdemir/advanced-nn/.github/workflows/release.yml`
to pin the signing workflow.

| Symptom | Fix |
|---|---|
| "tag vX does not match pyproject version" | delete the tag (`git push --delete origin vX && git tag -d vX`), bump correctly, retag |
| Attestation step fails with a permissions error | the job needs `id-token: write` and `attestations: write`; check they were not removed |
| `gh release create` fails because the release exists | delete the release or tag it fresh; never re-upload different files under an attested version |
| `gh attestation verify` finds nothing | the file was not built by this workflow, or was modified; compare `sha256sum` with the release asset |

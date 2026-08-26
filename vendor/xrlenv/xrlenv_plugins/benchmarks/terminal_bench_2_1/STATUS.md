# terminal-bench-2-1 — oracle sweep status

Corpus-quality gate: run harbor's `OracleAgent` against the **patched** tasks per task
**on the xrlenv cluster** (`xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster`) and
confirm each earns a positive `reward`. An oracle failure is a corpus defect (reward
ceiling 0 → poison for RL); for this dataset it usually means an unpinned dependency
drifted — fixed by a curated `solve.sh` dep-pin in `build_cache.py` (`--stage patch`).

## Gate config (current)

`run_full_sweep.sh`: green set = **present tasks − `EXCLUDE`** (exclusion, so a
re-populate auto-picks up new tasks); the count is asserted. Concurrency via `--max-workers`
(default 32; tb2.1 is all-runc so it can go to 64); the oracle gate is `reward > 0`. The
cache defaults to cpuset-sizing ON (so `make -j$(nproc)` oracles see the declared cores,
not all 192 host cores). See the README §"One-command green sweep". Launch via
[Reproduce](#reproduce).

## Results

> **Run metadata pending the next full green sweep.** Passed/Total + run metadata (job id,
> wall-clock, `n_errored_trials`/`n_retries`) will be filled from the upcoming authoritative
> sweep. The green SET and the exclusion below are stable.

| Bucket | Count | Meaning |
|---|---:|---|
| ✅ **Green set** | **88** | present tasks minus the 1 operational exclude — the sweep asserts this count |
| 🐢 **Excluded (operational)** | **1** | `caffe-cifar-10` — see below |
| **Total (present)** | **89** | full populated `terminal-bench-2-1` shard |

### Excluded — `caffe-cifar-10` (operational, NOT a broken oracle)

Its CIFAR-10 dataset host is very slow, so the oracle busts its wall-clock on the
**download**, not on our infra. `run_full_sweep.sh` prints this exclusion on every launch.
Drop it from `EXCLUDE` once the dataset is pre-seeded into the image.

### Curated dep-pins (`build_cache.py --stage patch`)

Faithful `solve.sh` version pins that keep non-hermetic oracles reproducible (e.g.
`build-cython-ext` → `planarity==0.6`; was `FAIL`/reward-0 before the pin). Each pin has an
`anchor` that fails loudly if upstream moved the solve script — see the `PATCHES` table +
the README §"Adding a pin".

## Reproduce

```bash
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example
export XRLENV_GRPC_HOST=<control-plane-host>   # + XRLENV_CONSUMER_TOKEN if the CP has auth
export XRLENV_GRPC_PORT=50051

# THE GATE — (re)builds+patches the cache, asserts the green-set count, runs the sweep:
bash xrlenv_plugins/benchmarks/terminal_bench_2_1/run_full_sweep.sh          # 88 tasks
bash xrlenv_plugins/benchmarks/terminal_bench_2_1/run_full_sweep.sh --max-workers 64

# targeted subset / resource ablation (if an oracle can't pass):
python xrlenv_plugins/benchmarks/terminal_bench_2_1/run_oracle_sweep.py \
    --tasks build-cython-ext,build-pmars --max-workers 1 --jobs-dir ./tmp
```

Exit code is `0` only if every oracle solved, so the sweep is CI-usable. Per-trial
artifacts (`agent/oracle.txt`, `verifier/reward.txt`, `result.json`) land under
`--jobs-dir/<job-id>/` (default `tmp/tb21-oracle-sweep/`).

## Notes

- **Pass gate:** `reward > 0` per task. `build-cython-ext` must report `PASS` (reward 1) —
  it was `FAIL` before the `planarity` pin.
- **Image resolution** falls through to the per-task `docker_image` (prebuilt
  `alexgshaw/<task>:<tag>`) — no template/registry; the cluster pulls lazily on first acquire
  (warm eagerly via `build_plan_gen --all` + `xrlenv build apply` if desired).
- When a sweep finds a new task broken by an unpinned dependency, add a `PATCHES` row in
  `build_cache.py` and re-run `--stage patch` — never edit the upstream solve script.

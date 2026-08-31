# DeepSWE — oracle sweep status (all 113)

Corpus-quality gate: run pier's `OracleAgent` (applies each task's `solution/` +
commits) per task **on the xrlenv cluster** via the pier plug-in
(`xrlenv_plugins.pier:XrlenvPierEnvironmentCluster`) and confirm each earns a
positive `reward`. Under the oracle a non-passing task is a plumbing/content bug —
its reward ceiling is 0 → poison for RL.

## Gate config (current)

Each task runs at its **native `timeout_sec`** (no `--timeout-multiplier`),
concurrency **32**, and the two default retry layers — `--retries 6` (per-trial,
infra-transient only) + `--content-retries 2` (per-task, outcome-keyed). See the README
§"Two retry layers — and why". Launch via the [Reproduce](#reproduce) command below.

## Full sweep — **113 / 113 GREEN** at **native timeout budget** (2026-07-21)

Native-budget (1.0) re-confirmation of the corpus: every task passes at its own
declared `timeout_sec` with **no** inflated headroom, so no task was silently relying
on a `--timeout-multiplier`. (Supersedes the 2026-07-18 run, which was measured at
`--timeout-multiplier 2`.)

| Bucket | Count | Meaning |
|---|---:|---|
| ✅ **Passed** | **113** | every task `reward=1.0` (f2p + p2p fully green) |
| ❌ **Failed** | **0** | — |
| **Total** | **113** | fully accounted |

- **Run:** `deepswe-full-sweep-2026-07-21_22-09-16`, conc **32**, `timeout_multiplier
  = 1.0` (native), `retry.max_retries = 6` (infra-only). Wall-clock **~20 min**
  (22:09:19 → 22:29:24). Per the job `result.json` stats: **`n_errored_trials = 0`**,
  **`n_retries = 0`**, `n_cancelled = 0` — no infra errors, no retries consumed, no
  content-retry rounds. Aggregate eval metric across all 113: `f2p = p2p = partial =
  reward = 1.0`.
- **No pre-warm.** Images were **not** pre-staged — the cluster's dynamic image cache
  (lazy-pull-on-acquire + LRU eviction + image-affinity) pulled each task's public-ECR
  image on first acquire and evicted under disk pressure. This is the guard that makes
  a full sweep safe without a 113×~2.5 GiB pre-warm — held even at conc-32.
- **All 5 languages green:** go, python, typescript, rust, javascript.
- **Separate-verifier seam exercised for all 113** (`environment_mode="separate"`): the
  plug-in resolves the verifier base image from `tests/Dockerfile` `FROM` / the parent
  task's top-level `docker_image` and uploads `/tests` itself (pier hardcodes
  `skip_tests_upload=True`), then the reward round-trips to the host via `download_dir`
  (`capabilities.mounted=False`).

## Reproduce

```bash
set -a; . ./.env; set +a                 # XRLENV_GRPC_HOST + tokens
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example

# Full gate via the entrypoint — (re)builds the cache, then runs all 113 at NATIVE
# timeout budget (no --timeout-multiplier), concurrency 32, with both default retry
# layers (--content-retries 2, --retries 6). Images lazy-pull on first acquire (no
# pre-warm needed). Run under nohup/background: a SWE oracle+verifier trial exceeds a
# foreground shell/tool timeout; the whole sweep is ~25-30 min.
nohup bash xrlenv_plugins/benchmarks/deep_swe/run_full_sweep.sh \
    --max-workers 32 --job-id deepswe-native-113 > tmp/native113.log 2>&1 &

# (Low-level equivalent, if you want to bypass the wrapper's content-retry loop:)
#   nohup .venv/bin/python xrlenv_plugins/benchmarks/deep_swe/run_oracle_sweep.py \
#       --max-workers 32 --retries 6 \
#       --jobs-dir ./tmp --job-id deepswe-native-113 > tmp/native113.log 2>&1 &
```

Exit code is `0` only if every oracle solved, so the sweep is CI-usable. Per-trial
artifacts (`agent/`, `verifier/reward.json`, trial logs) land under
`--jobs-dir/<job-id>/`. Resource-ablation knobs (`--cpus-multiplier`,
`--memory-multiplier`, `--cpu-pinning`, `--override-*`) mirror the tb2.1 / TW sweeps.

## Notes

- **Pass gate keys on `reward` only** — DeepSWE `reward.json` also carries
  f2p/p2p totals + fractions + `partial` that can be legitimately 0; the gate is
  `float(rewards["reward"]) > 0` with the rest reported as side metrics.
- If a future sweep surfaces a broken oracle (content/dep drift), add a curated
  overlay under `patches/<task_id>/` and re-run `build_cache.py --stage patch`
  (the hook exists; empty today — DeepSWE grades behaviorally against baked tests,
  so drift risk is lower than tb2.1's live-pip oracles).

# [B] DeepSWE

[DeepSWE](https://github.com/datacurve-ai/deep-swe) is a software-engineering
benchmark from Datacurve AI. It ships **113 tasks across 5 languages** (Go,
Python, TypeScript, Rust, JavaScript), each with a prebuilt per-task image on
public ECR, a `solution/` reference fix, and a separate verifier container that
grades with `reward.json`. The corpus uses the pier task format
(`task.toml` / `solution/` / `tests/`) and grades via
`environment_mode="separate"` — the verifier runs in its own fresh container
after the agent trial.

As of **2026-07-18 the full corpus is 113 / 113 GREEN** (oracle gate).

This page covers:

- [Prerequisites](#prerequisites) — install extras, env vars.
- [Step 1: build the task-dir cache](#step-1-build-the-task-dir-cache)
  (`build_cache.py`).
- [Step 2: warm images (optional)](#step-2-warm-images-optional)
  (`build_plan_gen.py` + `xrlenv build apply`).
- [Step 3: run the oracle sweep](#step-3-run-the-oracle-sweep)
  (`run_oracle_sweep.py` or `run_full_sweep.sh`).
- [Image route](#image-route) — why direct public-ECR pull works with no
  re-push.
- [Separate-verifier details](#separate-verifier-details).
- [Pass gate](#pass-gate) — why the gate keys on `reward` only.
- [Resource knobs](#resource-knobs).
- [Curated patches](#curated-patches).

## Prerequisites

Install the `deep-swe` extra (pulls in `datacurve-pier==0.3.0`):

```bash
pip install -e '.[deep-swe]'
```

Then boot the control plane and set cluster connection config:

```bash
xrlenv up                          # start the control plane

# In .env (auto-loaded by xrlenv on import):
export XRLENV_GRPC_HOST=<cp-host>
export XRLENV_GRPC_PORT=50051
export XRLENV_CONSUMER_TOKEN=<token>   # required if CP has auth
```

## Step 1: build the task-dir cache

`build_cache.py` materializes the 113 task directories into a shared cache
shard at `$XRLENV_BENCHMARK_CACHE/deep-swe/`. It is idempotent — re-running skips
tasks already present.

```bash
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example

# populate from git + apply curated patches (the default):
.venv/bin/python xrlenv_plugins/benchmarks/deep_swe/build_cache.py --stage all

# populate only (needs network + git):
.venv/bin/python xrlenv_plugins/benchmarks/deep_swe/build_cache.py --stage populate

# apply patches only (populate must have run first):
.venv/bin/python xrlenv_plugins/benchmarks/deep_swe/build_cache.py --stage patch
```

Two sources are available via `--source` (default `git`):

| Source | Requires | How |
|---|---|---|
| `git` (default) | network + `git` | Shallow-clone `datacurve-ai/deep-swe` at `main` |
| `hf` | network + `huggingface_hub` | Snapshot `datacurve/deep-swe` from the HF hub |

After populate the shard coexists with other benchmark shards (terminal-bench-2,
terminalworld) under the same cache root — no collision by design.

**Env overrides for populate:**

| Variable | Default | Description |
|---|---|---|
| `DEEPSWE_REPO_URL` | `https://github.com/datacurve-ai/deep-swe` | git remote |
| `DEEPSWE_REPO_REF` | `main` | git ref / branch |
| `DEEPSWE_HF_REPO` | `datacurve/deep-swe` | HF dataset id (for `--source hf`) |
| `DEEPSWE_SHARD` | `deep-swe` | Shard subdirectory name |

## Step 2: warm images (optional)

DeepSWE tasks ship a prebuilt public-ECR image per task (`public.ecr.aws/d3j8x8q7/swe-bench-202605:<id>-v1.1`). The cluster's dynamic image cache pulls
each image on first acquire and evicts under disk pressure, so a full 113-task
sweep is safe at low concurrency (validated at `--max-workers 8`) **without
any pre-warm step**.

For large sweeps where you want all images resident before the run starts,
generate and apply the warm plan:

```bash
# Generate the plan from task.toml docker_image fields:
XRLENV_BENCHMARK_CACHE=/path/to/cache \
.venv/bin/python -m xrlenv_plugins.benchmarks.deep_swe.build_plan_gen \
    --all --output xrlenv_plugins/benchmarks/deep_swe/deepswe_build_plan.yaml

# Apply it (node-side docker pull, FFD bin-packed across nodes). --plan + --connect-host are
# REQUIRED (the CLI dials the control plane); without --connect-host it exits 2.
xrlenv build apply \
    --plan xrlenv_plugins/benchmarks/deep_swe/deepswe_build_plan.yaml --fill-missing \
    --connect-host <control-plane-host>

# Calibrate sizes after the first warm (probe is OFF by default for public-ECR refs; writes a
# SEPARATE *.calibrated.yaml — diff before promoting):
xrlenv build calibrate \
    --plan xrlenv_plugins/benchmarks/deep_swe/deepswe_build_plan.yaml \
    --output xrlenv_plugins/benchmarks/deep_swe/deepswe_build_plan.calibrated.yaml \
    --connect-host <control-plane-host>
```

A committed plan (`deepswe_build_plan.yaml`) already exists in the repo for
re-use.

## Step 3: run the oracle sweep

### One-command gate

```bash
# Load .env (control-plane host, tokens, cache root), run the full 113-task corpus:
set -a; . ./.env; set +a
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example

bash xrlenv_plugins/benchmarks/deep_swe/run_full_sweep.sh
```

`run_full_sweep.sh` builds the cache (unless `--skip-build-cache`), runs
`run_oracle_sweep.py`, and content-retries any non-passing tasks up to
`--content-retries` times (default 2). Exit code is non-zero if any task is
still failing after retries.

Config flags (all optional). Run knobs are **flags**, not env vars — a stale
exported var must never silently change a sweep; only the cache root comes from
the environment:

| Flag | Default | Description |
|---|---|---|
| `--max-workers N` | `8` | Trial concurrency |
| `--job-id LABEL` | `deepswe-full-sweep` | Run label under `tmp/` |
| `--content-retries N` | `2` | Re-run non-passing tasks up to N times |
| `--jobs-dir DIR` | `./tmp` | Per-trial artifact root |
| `--skip-build-cache` | — | Skip `build_cache.py` (use the cache as-is) |
| `--list-green` | — | Print the green set (`present − EXCLUDE`) and exit |
| `XRLENV_BENCHMARK_CACHE` *(env)* | `/path/to/xrlenv_benchmark_cache` | Cache root |

Extra CLI args pass through to `run_oracle_sweep.py`:

```bash
bash xrlenv_plugins/benchmarks/deep_swe/run_full_sweep.sh \
    --max-workers 16 --timeout-multiplier 2
```

### Direct sweep (fine-grained control)

```bash
# Run all 113 tasks, concurrency 8, 30-min timeout headroom, up to 6 infra retries:
.venv/bin/python xrlenv_plugins/benchmarks/deep_swe/run_oracle_sweep.py \
    --max-workers 8 \
    --timeout-multiplier 2 \
    --retries 6 \
    --jobs-dir ./tmp \
    --job-id deepswe-full-113

# Run a subset:
.venv/bin/python xrlenv_plugins/benchmarks/deep_swe/run_oracle_sweep.py \
    --tasks fastapi-implicit-head-options,go-zero-github-issues

# For long sweeps (~25-30 min) run under nohup:
nohup .venv/bin/python xrlenv_plugins/benchmarks/deep_swe/run_oracle_sweep.py \
    --max-workers 8 --timeout-multiplier 2 --retries 6 \
    --jobs-dir ./tmp --job-id deepswe-full-113 \
    > tmp/full113.log 2>&1 &
```

Per-trial artifacts (agent output, `verifier/reward.json`, trial logs) land
under `--jobs-dir/<job-id>/<task-name>__<suffix>/` (harbor appends a generated
`__<suffix>` to the trial dir stem — the canonical task id is read from
`config.task.path`, not the dir name). Each content-retry round writes a
**sibling** `<job-id>-retryN/` directory (a task passing in ANY round counts), so a
retried task's artifacts live under the `-retryN` sibling, not the base job dir.

### Infra retries

The sweep retries only infra-transient errors — never task-content failures:

| Exception | Meaning |
|---|---|
| `CapacityExhausted` | Admission queue timed out waiting for a runtime slot |
| `ControlPlaneLost` | CP restarted under the run |
| `NodeLost` | Node dropped its stream mid-acquire |
| `NodeCommandTimeout` | A node RPC deadline (teardown / exec) tripped |

A task-content failure (agent timeout, verifier error, wrong reward) is never
retried — the oracle solving a task must be a clean single pass.

## Image route

DeepSWE images are on **public ECR** and are anonymous-pullable:
`public.ecr.aws/d3j8x8q7/swe-bench-202605:<ext_id>-v1.1`. The cluster resolves
each task's image from `task_env_config.docker_image` (precedence #2 in the
pier adapter's resolution chain — see {doc}`pier_framework`) and pulls on
first acquire.

**Do not re-push these images to the private registry.** Direct ECR pull
(Path 1) works without any new infrastructure. If you need a pull-through
mirror (Path 2) to reduce cross-region egress, rewrite the ref host to your
ECR-upstream proxy — that is the only change needed.

The dynamic image cache (lazy-pull + LRU eviction + image-affinity) makes
the full 113-task corpus safe at `--max-workers 8` without pre-warming: the
oracle gate ran clean at that concurrency with 0 infra errors and 0 retries
consumed.

## Separate-verifier details

All 113 DeepSWE tasks use `environment_mode="separate"`: pier grades each trial
in a fresh verifier container. The pier adapter handles this transparently:

- **Image**: the verifier's `[verifier.environment]` block carries no
  `docker_image`. The adapter resolves the base image from the `tests/Dockerfile`
  `FROM` (the prebuilt ECR image) or the parent `task.toml`'s top-level
  `docker_image`. No sweep flag is needed.
- **Tests upload**: pier hardcodes `skip_tests_upload=True` on the assumption
  the grader is baked into the verifier image. It is not — the ECR base is the
  task image, not a pre-baked grader. The adapter uploads the task's `tests/`
  directory (minus `Dockerfile`) to `/tests` inside the verifier container on
  `start()`, making `test.sh` and the supporting grader files present.
- **Reward round-trip**: `capabilities.mounted=False` — the cluster node is
  remote, so `reward.json` round-trips to the host via `download_dir`.

## Pass gate

The gate keys on the `reward` field of `verifier/reward.json` only:

```python
reward = float(vr.rewards["reward"])
passes = reward > 0
```

DeepSWE's `reward.json` also carries `f2p_total`, `f2p_fraction`, `p2p_total`,
`p2p_fraction`, and `partial`. These can be legitimately zero on a passing task
(e.g. a task with no passing-to-passing tests). An "all values > 0" gate would
false-fail such tasks. The sweep prints these as side metrics for context but
does not include them in the pass/fail decision.

## Resource knobs

The sweep's resource flags compose with each task's `task.toml` declarations:

| Flag | Description |
|---|---|
| `--override-cpus N` | Force every task to N CPUs (ignores task.toml). |
| `--override-memory-mb N` | Force every task to N MiB of memory. |
| `--cpus-multiplier F` | Scale each task's declared CPUs by F. Composes with `--override-cpus`. |
| `--memory-multiplier F` | Scale each task's declared memory by F. |
| `--cpu-pinning` | Opt the job into cpuset pinning (`nproc` == declared CPUs). Useful on large hosts where uncapped `nproc` would trigger `make -j$(nproc)` OOMs. |
| `--timeout-multiplier F` | Scale pier's agent/verifier timeouts. `2.0` gives slow nodes headroom. |

## Curated patches

The `patches/` directory beside `build_cache.py` holds curated full-file
overlays applied per task after extraction. It is **empty today** — DeepSWE
grades behaviorally against baked tests, so unpinned-dependency drift risk is
lower than benchmarks with live-pip oracles. The hook exists for when the
oracle sweep surfaces broken content: add `patches/<task_id>/<rel>` and
re-run `build_cache.py --stage patch` (idempotent, no re-download needed).

## Status

113 / 113 GREEN as of 2026-07-18. All 5 languages (go, python, typescript,
rust, javascript) pass with `reward=1.0`. Wall-clock ~26 min at `--max-workers 8
--timeout-multiplier 2 --retries 6`, 2 nodes, 0 infra errors.

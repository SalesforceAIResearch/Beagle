# XRLEnv benchmark integration harness

Manually-run harnesses that drive **real benchmark oracle sweeps** through a
running xrlenv cluster and fail LOUDLY. This is the corpus-quality / onboarding
gate: it confirms each benchmark's shipped reference solution still earns a
positive reward end-to-end, across the SDK → control plane → node → sandbox →
upstream-verifier boundary that unit tests can't see by construction.

Distinct from the two sibling layers:

| Layer | What it catches | Runs on `pytest -q`? |
|---|---|---|
| `tests/unit/` | logic regressions (pure, no cluster) | yes |
| `tests/smoke/` | wiring regressions against a live daemon / CP | no (`--ignore`d) |
| **`xrlenv_plugins/benchmarks/tests/`** (here) | corpus / onboarding regressions — a benchmark's oracle no longer clears its reward bar on a real cluster | no (real sweeps, minutes–hours) |

These sweeps do real cluster work (pull images, run oracles, grade). They are
launched **by hand** — either as a full green-set gate or with a small
deterministic sample for CI. They are **not** part of the default test run.

## Layout

```
xrlenv_plugins/benchmarks/tests/integration/
├── README.md                    ← you are here
├── run_benchmarks.py            ← config-driven sweep runner (all onboarded plug-in benchmarks)
├── benchmarks.yaml              ← the suite config it reads (benchmarks + per-cluster profiles)
├── test_run_benchmarks.py       ← offline unit tests for run_benchmarks.py's pure logic
└── test_wrapper_env_ordering.py ← offline guard: each run_full_sweep.sh sources .env first
```

The unit tests for `run_benchmarks.py`'s pure logic (config merge, deterministic
sampler, command construction) and the wrapper env-ordering guard are **co-located
here** and **do** run on `pytest -q` (collected via the `xrlenv_plugins` testpath).

## Coverage

`run_benchmarks.py` drives every benchmark that ships the `run_full_sweep.sh` /
`run_oracle_sweep.py` sweep contract: the 5 harbor/pier plug-ins (`deep_swe`,
`lhtb`, `seta`, `terminal_bench_2_1`, `terminalworld`) **and** `swebench_verified`
(a docker-py drop-in — swebench's own harness, cache-backed, wearing the same flag
interface and `resolved`-based gate).

`evoclaw` + `webarena-infinity` are plug-ins with bespoke entrypoints and are run
manually (not wired into the runner yet).

## `run_benchmarks.py` — config-driven sweep runner

One command runs the selected benchmarks, each in one of two modes, and exits
`0` iff **every** selected sweep passed (so it's CI-usable). Both modes run
**only `present − EXCLUDE`** — each benchmark's own `run_full_sweep.sh` owns its
`EXCLUDE` / blacklist (the single source of known-failing tasks); the runner
never re-invents that list.

| Mode | What it does |
|---|---|
| `full` | runs the benchmark's `run_full_sweep.sh` over its whole **green set** (`present − EXCLUDE`, **never** all present tasks), with the sweep's own content-retry layer |
| `sample` | reads the green set via `run_full_sweep.sh --list-green`, picks `k` tasks **deterministically** (seeded — same set every run; change the seed to rotate), then `run_oracle_sweep.py --tasks <sample>` |

### Quickstart

Run from the repo root, with the harbor cache pointed at the shared tree and a
control plane already up (see **Prerequisites**):

```bash
export XRLENV_BENCHMARK_CACHE=/path/to/benchmark-cache

# the manual full gate — whole green set per benchmark:
.venv/bin/python xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py --profile full-prod

# fast CI signal — a deterministic k-task sample per benchmark:
.venv/bin/python xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py --profile ci

# a subset of benchmarks:
.venv/bin/python xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py --profile ci --benchmark lhtb,seta

# see exactly what would run (+ the sampled task ids) without executing:
.venv/bin/python xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py --profile ci --dry-run
```

### Flags

| Flag | Meaning |
|---|---|
| `--profile` (required) | a profile in `benchmarks.yaml` (e.g. `full-prod`, `ci`, `selected-coding-prod`, `selected-coding-dev`) |
| `--config PATH` | config file (default: the co-located `benchmarks.yaml`) |
| `--benchmark a,b` | comma-separated subset of the config's benchmarks (default: all) |
| `--jobs-dir DIR` | **parent** artifact root (default: `./tmp`); the run is grouped under `<jobs-dir>/<run-name>/` |
| `--run-name NAME` | group this invocation's artifacts under one dir; must be a single path component beneath `--jobs-dir` (default: `<profile>-<timestamp>-<pid><rand>`, always unique) |
| `--overwrite` | if the run dir already exists, clear it first (default: refuse, so a reused dir can't feed stale artifacts to the coverage gate) |
| `--max-concurrent-tasks N` | concurrency budget; wins over the profile's / top-level `max_concurrent_tasks` (default: `benchmarks.yaml`, else 128; `0` = sequential) |
| `--seed N` | override the deterministic-sample seed for all benchmarks |
| `--dry-run` | print the resolved commands (+ sampled task ids) and exit `0` |

**Parallel scheduling.** Benchmarks run concurrently, gated by a **concurrency budget**
(`max_concurrent_tasks`, default 128; set top-level in `benchmarks.yaml`, per-profile —
at the profile level or under `overrides:` — or via `--max-concurrent-tasks`). The
runner admits benchmarks **in config order** while the sum of their in-flight **costs**
stays ≤ the budget — a benchmark whose cost alone exceeds the budget runs by itself (no
deadlock). A benchmark's cost is its **real peak concurrency** — `min(task-count,
workers)`, *not* its task count — because it runs at most `workers` rollouts at once.
So a full sweep's 500-task/8-worker `swebench_verified` costs 8, not 500, and OVERLAPS
the other benchmarks from the start instead of running alone at the end; a 128 budget
packs e.g. deep_swe(32) + tb2.1(32) + terminalworld(32) + swebench(8) = 104 concurrently.
Each benchmark streams to `<run-dir>/<name>-<profile>.log` (the runner prints
`▶ start …N task(s) (≤M concurrent)` / `✓ PASS` / `✗ FAIL`) and a single liveness
banner tracks live per-task progress; `--max-concurrent-tasks 0` restores sequential,
live-streamed output.

Every invocation is grouped under `<jobs-dir>/<run-name>/` (default run-name
`<profile>-<timestamp>-<pid><rand>`, e.g. `tmp/ci-2026-07-31_09-14-02-4821-b3a1/`, always
unique) so back-to-back runs don't flatten and collide at the top of `--jobs-dir`. A
pre-existing run dir is refused unless `--overwrite`. Both modes land there — each
benchmark under its own `<name>-<profile>/` subdir, plus a
`benchmarks-<profile>-summary.json`:

```
tmp/ci-2026-07-31_09-14-02-4821-b3a1/
├── deep_swe-ci/  seta-ci/  swebench_verified-ci/  ...   # per-benchmark artifacts
└── benchmarks-ci-summary.json                            # the run's PASS/FAIL tally
```

The process exit code is `0` iff every selected benchmark passed.

> **Note:** `swebench_verified` needs its cache materialized first —
> `python xrlenv_plugins/benchmarks/swebench_verified/build_cache.py --stage all --all`
> — before `--list-green` (and therefore `sample` mode) can enumerate it.

### The config — `benchmarks.yaml`

Three sections, merged per benchmark as
`defaults ← benchmarks[name] ← profiles[profile] ← profiles[profile].overrides[name]`:

```yaml
max_concurrent_tasks: 64   # top-level default budget (see "Parallel scheduling")

defaults:
  content_retries: 2      # full-mode per-task re-run rounds (the sweep's own layer)
  seed: 0                 # deterministic k-sample seed (change to rotate coverage)

benchmarks:               # per-benchmark knobs: workers (concurrency) + retries
  deep_swe:           { workers: 32, retries: 6 }   # sample-mode --retries N (infra-only)
  seta:               { workers: 32, retries: 6 }
  swebench_verified:  { workers: 32, retries: 6 }   # heavy SWE trials + big images
  # ... lhtb / terminal_bench_2_1 / terminalworld likewise

profiles:
  full-prod:                         # whole green set per benchmark, prod budget
    mode: full
    max_concurrent_tasks: 192        # per-profile budget (see note below)
  ci:                                # deterministic k-task sample
    mode: sample
    k: 2
    overrides:                       # heavier suites sample fewer
      swebench_verified:  { k: 3 }
  selected-coding-prod:              # a curated FULL sweep of just a subset
    mode: full
    only: [deep_swe, terminal_bench_2_1, terminalworld, swebench_verified]
    max_concurrent_tasks: 192
  selected-coding-dev:               # same subset, smaller dev-cluster budget
    mode: full
    only: [deep_swe, terminal_bench_2_1, terminalworld, swebench_verified]
    max_concurrent_tasks: 64
```

A profile sets the `mode` (`full` / `sample`) plus its knobs. Optional profile-level
keys shape the run:

- **`max_concurrent_tasks: N`** — this profile's scheduling budget, overriding the
  top-level default. Accepted at the profile level (canonical) **or** inside the
  profile's `overrides:` block — both are honored (profile level wins). Precedence:
  `--max-concurrent-tasks` (CLI) > profile > top-level > 128.
- **`workers: N`** — trial concurrency for *all* benchmarks in this profile; or set it
  per benchmark under `overrides:` (e.g. `overrides: { swebench_verified: { workers: 24 } }`).
- **`only: [names]`** — restrict the profile to a picked subset of benchmarks
  (config order is preserved; unknown names fail loud). Absent → all benchmarks.
- The CLI **`--benchmark a,b`** always overrides a profile's `only`, so you can run
  one benchmark from any profile ad hoc.

So `--profile selected-coding-prod` runs only those four at a 192 budget; `--profile
full-prod` runs all six. Name/rework profiles freely per cluster (dev vs prod).

### Prerequisites

`run_benchmarks.py` is a **read-only gate** — *every* profile lists and runs with
`--skip-build-cache` and never builds a cache or an image (it can't build/push all
benchmark images anyway, so it must not half-bootstrap). Prepare the prerequisites once,
then run the gate:

- **A control plane is up** and reachable. The sweeps read `XRLENV_GRPC_HOST` /
  token / registry from the repo-root `.env` (the runner auto-loads it,
  never overwriting an already-set var).
- **The harbor cache is populated** at `XRLENV_BENCHMARK_CACHE` (the sweep scripts default
  to `/path/to/benchmark-cache`). An absent/incomplete cache makes the gate
  **fail loud** (the wrapper's `--list-green` exits non-zero with a `build_cache.py` recipe)
  rather than silently building it.
- The interpreter is `.venv/bin/python` (`uv sync --all-extras`).

#### One-time bring-up (fresh cluster)

The runner assumes prepared prerequisites. To stand up a new cluster, do the bring-up once —
sequentially, from **one** operator host (`build_cache.py` is a sequential single-writer; do
not run it concurrently from multiple nodes for the same benchmark):

1. **Build + push (or pre-warm) the required images.** Benchmarks that ship a `Dockerfile`
   (seta, terminalworld) or need rebuilt images (lhtb REBUILD tasks) are built + pushed to
   the private registry by the `xrlenv_plugins/images_build/…` scripts. Prebuilt-image
   benchmarks (deep_swe, tb2.1, swebench_verified) need nothing here.
2. **Build the task-data caches** — for each benchmark:
   `bash xrlenv_plugins/benchmarks/<name>/run_full_sweep.sh --list-green`  *(without*
   `--skip-build-cache`*, so it populates)*, or run its `build_cache.py --stage all` directly.
   For SWE-bench that materializes all 500 Verified instances from HuggingFace.
3. **Run the gate:** `run_benchmarks.py --profile ci` / `full-prod`.

### What "pass" means

Each benchmark's `run_full_sweep.sh` / `run_oracle_sweep.py` is the authority: it
exits non-zero if any task in its green set fails to clear its reward bar after
the content-retry rounds. The runner surfaces that per-benchmark and aggregates.
An oracle FAIL is a **corpus / plumbing defect** (a reward-0 ceiling — or, for
swebench_verified, a gold patch that didn't `resolve` — poison for RL), not a model
signal; inspect the per-trial verifier output under the run's `tmp/` job dir.

### Adding a benchmark

The runner drives every benchmark through one **uniform flag interface** on its
`run_full_sweep.sh`. Run knobs are **flags, never env vars** — a stale exported
`SKIP_BUILD=1` / `LIST_GREEN=1` must never silently turn a real sweep into a
no-op. To onboard a new one:

1. Its `run_full_sweep.sh` must accept the standard flags:
   - `--list-green` — print its green set (`present − EXCLUDE`), one id per line,
     and exit, running nothing. This is how sample mode learns the set *without*
     re-implementing the benchmark's exclusion logic (the sweep owns it, as the
     single source of truth).
   - `--skip-build-cache` — enumerate/run against the already-populated cache without a
     (slow) rebuild. The runner is a **read-only gate** over prepared cache/image
     prerequisites (audit M12): it passes this flag on **every** invocation — both
     `--list-green` planning *and* full-mode execution — so a profile run never builds a
     cache or image. Bring-up (populate the cache, build+push images) is a separate,
     one-time operator step; see the bring-up section above.
   - `--max-workers N`, `--content-retries N`, `--job-id LABEL`, `--jobs-dir DIR` —
     the run knobs full mode passes; forward any unrecognized arg to
     `run_oracle_sweep.py`.
   - (`XRLENV_BENCHMARK_CACHE` stays env-driven — it's a deployment path, not a run knob.)
2. Its `run_oracle_sweep.py` must accept the uniform sample-mode flags — `--tasks`,
   `--max-workers`, `--jobs-dir`, `--job-id`, and `--retries N` (infra-transient
   retries only). The runner is uniform: there's no Python table to edit.
3. Add the benchmark to `benchmarks.yaml` (`workers` + `retries`, plus any profile
   override).
4. Extend `tests/unit/integration/test_run_benchmarks.py` if it adds a shape the
   existing tests don't cover.

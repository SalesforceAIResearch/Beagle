# [H] SWE-rebench

[SWE-rebench](https://hub.harborframework.com/datasets/swe-rebench/swe-rebench-leaderboard/latest)
(Nebius AI R&D) is a harbor-format corpus of **860 curated Python SWE tasks**,
continuously refreshed with monthly splits. Every task ships a prebuilt container
image on Docker Hub, a `tests/` verifier that writes `reward.txt` to
`/logs/verifier/`, and a `solution/solve.sh` reference — so, unlike FrontierSWE,
**the whole corpus is oracle-gateable**.

xrlenv runs SWE-rebench on the **harbor golden path**: the sweep reuses
`XrlenvHarborEnvironmentCluster` with zero adapter code, identical to
terminal-bench-2. Grading is stock harbor end-to-end — no grade-from-artifact
seam, no re-implemented resolution rule.

This page covers:

- [Two distinctive traits](#two-distinctive-traits) — Hub-package distribution
  and the `docker_image` repin.
- [Prerequisites](#prerequisites)
- [Step 1: build the task-dir cache](#step-1-build-the-task-dir-cache)
- [Step 2: prepare images](#step-2-prepare-images)
- [Step 3: run the oracle sweep](#step-3-run-the-oracle-sweep)
- [Pass gate](#pass-gate)
- [Resource knobs](#resource-knobs)
- [Status](#status)

## Two distinctive traits

### It is a Harbor Hub *package* dataset

Most onboarded corpora come from a git clone or a Hugging Face snapshot.
SWE-rebench is published as a Harbor Hub **package dataset**, so harbor is both
the harness *and* the download client: `build_cache.py` drives harbor's own
`PackageDatasetClient` (dataset → task list) and `TaskClient` (per-task archive
download + extract) in export mode. The dataset is public, so this is anonymous —
no `HARBOR_API_KEY`, no git, no HF token.

A practical consequence: the harbor version pin is **doubly** load-bearing. It
fixes the harness *and* the Hub client's wire contract, so a bump is a
deliberate, reviewed edit plus an oracle-sweep re-run.

### Tasks declare no `docker_image` — the cache repins it

A SWE-rebench `task.toml` carries no `[environment] docker_image`. Upstream ships
the prebuilt image per task and expresses it as the `FROM` of a three-line
`environment/Dockerfile`:

```dockerfile
FROM swerebench/sweb.eval.x86_64.berriai_1776_litellm-14715:latest
ENV _JAVA_OPTIONS=""
RUN (curl -fsSL "$UV_URL" || wget -qO- "$UV_URL") | UV_INSTALL_DIR=/usr/local/bin sh
RUN mkdir -p /logs
```

The xrlenv cluster environment resolves an image ref at acquire; it does not
build on acquire. So `build_cache.py --stage repin` writes the authoritative
upstream ref into each `task.toml`, turning the corpus into a pull-on-demand
`type: registry` plan with **nothing to build**. The ref comes from the task's own
`tests/config.json` and is cross-checked against the Dockerfile's `FROM` — a
mismatch fails loud rather than silently pinning the wrong repo snapshot.

All three dropped lines are inert in practice. Harbor itself creates
`/logs/verifier` before the verifier phase, and nothing reads `_JAVA_OPTIONS`.
The third — installing `uv` — looked load-bearing for the 17 tasks whose verifier
runs `uv run pytest`, but **the base images already ship `uv`**: measured
2026-09-01, all 17 ran on-cluster against the plain upstream image and 16 solved,
the 17th failing on an unrelated upstream packaging defect, with no trial
producing `uv: command not found`. Nothing needs building.

## Prerequisites

```bash
uv pip install -e '.[swe-rebench]'     # harbor==0.20.0
```

A running control plane, plus the usual environment:

```bash
export XRLENV_GRPC_HOST=<control-plane-host>
export XRLENV_GRPC_PORT=50051
export XRLENV_CONSUMER_TOKEN=<token>
export XRLENV_BENCHMARK_CACHE=/path/to/xrlenv_benchmark_cache
```

For the size probe in step 2, also export `DOCKERHUB_USER` / `DOCKERHUB_TOKEN`.
All of these normally live in `.env`.

## Step 1: build the task-dir cache

```bash
.venv/bin/python xrlenv_plugins/benchmarks/swe_rebench/build_cache.py    # --stage all
```

`--stage all` runs populate → repin → patch, each idempotent. Tasks land at
`<cache>/swe-rebench/<instance_id>/`, and a `.dataset-version.json` records the
resolved dataset content hash for reproducibility.

Other stages: `--stage populate` (download + normalize only), `--stage repin`
(rewrite the image pins), and `--stage patch` (curated content fixes —
`patches/` overlays plus the resource routing described below).

### Resource routing

harbor applies a CFS cpu quota and a memory cap but never a cpuset, so `nproc`
inside a `cpus = 1` container reports the host's core count. On a 192-core node
anything sizing a pool from `os.cpu_count()` — joblib/loky, pytest-xdist
`-n auto`, dask/ray, OpenMP/BLAS — fans out ~192 ways inside an 8 GB cap and is
SIGKILL'd; that cost 16 tasks in the first full sweep. `--stage patch`
writes a per-task `[environment.env] XRLENV_CPU_PINNING = "1"` marker so `nproc`
matches the task budget, plus a memory override for the four that still need it.

A memory override is permitted only where upstream declared no memory, and the
build enforces that in code: SWE-rebench sets `harbor_cpus`/`harbor_memory` in
`tests/config.json` when it has an intent (10 of 860), and for the rest the `8G`
is the converter's default. Raising a converter default keeps the resource
envelope the benchmark intended; overriding an upstream-declared value would
not, so it raises.

### Hermeticity routing

A second, separate table of `[environment.env]` keys, also written by
`--stage patch`. It exists because the fairness question is different: these
variables change how a verifier resolves its **own dependencies**, never how much
CPU or memory it gets, so the memory guard above does not apply.

One task needs it today. `CQCL__guppylang-1259` grades with `uv run pytest`, and
`uv run` re-resolves the workspace on every invocation. PEP-517 *build*
requirements are not covered by a lockfile, so that resolve pulls whatever
hatchling is currently on PyPI — which since 2026-08 rejects the task's
`readme = "../README.md"`, so the package never builds and every graded test
reports `NOT_FOUND`. Setting `UV_NO_SYNC = "1"` makes `uv run` use the
environment the image already ships, which both restores the grade and removes a
live PyPI dependency from the verify phase.

The general point is worth carrying to other benchmarks: a verifier that resolves
dependencies at grade time is not hermetic, and it will break the moment an
unpinned transitive dependency publishes a release. Pin the resolution off rather
than patch the symptom.

Because upstream publishes monthly splits, the corpus grows over time. The sweep
wrapper pins the expected task count and refuses to define a green set if the
populate disagrees — an intentional refresh means re-pinning it.

## Step 2: prepare images

**Nothing is built.** Every task's image is an upstream Docker Hub prebuilt,
pulled lazily on first acquire. Warming is optional:

```bash
xrlenv build apply --plan xrlenv_plugins/benchmarks/swe_rebench/swe_rebench_build_plan.yaml \
    --fill-missing --connect-host "$XRLENV_GRPC_HOST"
```

Mind the scale: the committed plan totals **~1.65 TB** compressed (mean 1.9 GB
per image), so prefer warming only the subset you are about to run. Do not push
these into the private registry — mirroring the corpus would add ~1.1 TB to a
store with no quota and no automatic GC.

Regenerating the plan probes Docker Hub for real compressed sizes. Docker Hub
rate-limits an 860-image sweep even authenticated, so probing is **resumable**:

```bash
PLAN=xrlenv_plugins/benchmarks/swe_rebench/swe_rebench_build_plan.yaml
.venv/bin/python -m xrlenv_plugins.benchmarks.swe_rebench.build_plan_gen \
    --all --reuse-sizes "$PLAN" --output "$PLAN"     # re-run until 0 fallbacks
```

`--reuse-sizes` keeps every already-measured (`registry-probe`) size and probes
only the entries still on the heuristic.

## Step 3: run the oracle sweep

The correctness gate runs each task's shipped reference solution under harbor's
`OracleAgent` and confirms it earns positive reward. A task the oracle cannot
solve is poison for RL — its reward ceiling is 0 — so a non-passing task is a
plumbing or content bug, not a model signal.

**Start with the risk-ranked smoke set.** The corpus is unusually homogeneous —
every task grades through the same byte-identical `parser.py`, none has an empty
`FAIL_TO_PASS` — so a random sample mostly re-tests the same easy path.
`scripts/smoke_30tasks.txt` instead lists the tasks that differ on the signals
which actually vary (bespoke test runners, network-touching tests, huge
`PASS_TO_PASS` suites, heavy-resource tasks, oversized images, multi-file test
patches): 30 tasks over 30 repos at ~65 GB instead of ~1.6 TB.

```bash
set -a; source ./.env; set +a
S=xrlenv_plugins/benchmarks/swe_rebench

# 1 task, a few minutes
bash $S/run_full_sweep.sh --tasks prometheus__client_python-1134 --max-workers 1

# the smoke 30 — content proof, then platform proof
bash $S/run_full_sweep.sh --tasks-file $S/scripts/smoke_30tasks.txt --max-workers 8
bash $S/run_full_sweep.sh --tasks-file $S/scripts/smoke_30tasks.txt --max-workers 32

# the real gate
nohup bash $S/run_full_sweep.sh --max-workers 32 > /tmp/swe-rebench-g1.log 2>&1 &
```

Run the smoke set at **both** concurrencies: a network-touching task that passes
solo and fails under load is not flaky, it is a non-hermetic dependency
surfacing, and the fix is hermeticity — never a lower concurrency.

The wrapper exits 0 **iff** every task in the green set passes. A full 860-task
SWE sweep far exceeds any foreground shell timeout — background it and poll.
`--tasks a,b,c` / `--tasks-file PATH` narrow any run; both are **intersected**
with the green set, so an excluded task is skipped with a note rather than
smuggled in.

### Monthly splits

Upstream organises the corpus into 15 monthly splits (`2025_01` … `2026_03`,
added continuously) whose union is exactly the 860 tasks. The Harbor Hub package
is the flat `test` split and records no month, so the kit ships the mapping as
`scripts/monthly_splits.json` and exposes it as `--split`:

```bash
bash $S/run_full_sweep.sh --list-splits
bash $S/run_full_sweep.sh --split 2026_03
bash $S/run_full_sweep.sh --split 2025_01,2025_02
```

The index comes from the HF datasets-server, which is authoritative. It is **not**
derivable from `config.json`'s `created_at`: that is the upstream PR date, and
the newest split absorbs every recently-collected task, so `created_at` misfiles
64 of 860. Regenerate after a corpus refresh with
`scripts/fetch_monthly_splits.py` (`--check` verifies without writing).

`--list-green` prints the green set and exits without running anything; that is
the seam CI uses to sample tasks without re-implementing the exclusion logic.

Two retry layers, both inside `run_oracle_sweep.py`:

| Layer | Granularity | Retries on | Purpose |
|---|---|---|---|
| `--retries` (6) | per-trial | `CapacityExhausted`, `ControlPlaneLost`, `NodeLost`, `NodeCommandTimeout`, `SessionReaped` | absorb capacity pacing; cannot mask a flaky task |
| `--content-retries` (**0**) | per-task | a reward-0 *outcome* | off by default: a task that only passes on a re-run is a finding, not a pass |

A content-retry round is visible as a folded `<job-id>-retryN` set of artifacts,
so a flaky task is surfaced rather than silently greened. Timeouts run at
**native budget** — the gate never inflates them.

## Pass gate

Each `tests/test.sh` runs the repo's test command, hands the log to the
corpus-wide `tests/parser.py` (byte-identical across all 860), and writes a flat
`/logs/verifier/reward.txt` of `0` or `1` — resolved iff every `FAIL_TO_PASS`
**and** `PASS_TO_PASS` test passes. harbor parses that into
`rewards={"reward": <float>}`, and the sweep requires **every** reward key `> 0`.

On a failure, `verifier/report.json` names exactly which tests did not pass.

## Resource knobs

Tasks declare 1 cpu / 8 GB / 16 GB storage (850 tasks) or 2 cpu / 16 GB (10
tasks). The sweep exposes the standard ablation knobs — `--override-cpus`,
`--override-memory-mb`, `--cpus-multiplier`, `--memory-multiplier`,
`--cpu-pinning`, `--timeout-multiplier` — but the **gate** runs at declared
values and native timeouts.

Callers may request any `--max-workers`; xrlenv paces capacity through fail-fast
acquire plus the infra-only retries. Never lower concurrency to turn a red run
green — that hides a latent bug instead of fixing it.

## Status

Per-run results, the current green set, exclusions with their evidence, and the
exact reproduce command live in
`xrlenv_plugins/benchmarks/swe_rebench/STATUS.md`.

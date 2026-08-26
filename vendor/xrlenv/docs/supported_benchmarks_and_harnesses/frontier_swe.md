# [H] FrontierSWE

[FrontierSWE](https://github.com/Proximal-Labs/frontier-swe) (Proximal Labs) is a
harbor-format corpus of **17 ultra-long-horizon technical challenges** spanning
performance engineering, computational science, and ML research. The public
leaderboard is at [frontierswe.com](https://frontierswe.com).

Each task ships a prebuilt container image on public GHCR, a `tests/` verifier
that writes `reward.json` to `/logs/verifier/`, and — for oracle-gateable tasks —
a `solution/solve.sh` reference (the standard harbor filesystem contract). xrlenv
runs FrontierSWE on the **harbor golden path**: the sweep reuses
`XrlenvHarborEnvironmentCluster` with zero adapter code, identical to terminal-bench-2.

This page covers:

- [Corpus structure](#corpus-structure) — which tasks run and why.
- [Two distinctive traits](#two-distinctive-traits) — grade-from-artifact and
  run-time oracle mode.
- [Prerequisites](#prerequisites)
- [Step 1: build the task-dir cache](#step-1-build-the-task-dir-cache)
- [Step 2: warm images (optional)](#step-2-warm-images-optional)
- [Step 3: run the oracle sweep](#step-3-run-the-oracle-sweep)
- [Pass gate](#pass-gate)
- [Resource knobs](#resource-knobs)
- [Status](#status)

## Corpus structure

FrontierSWE ships 17 tasks. The oracle sweep discovers tasks that have a
`solution/solve.sh` reference; upstream-withheld tasks are normally invisible to
the sweep (not EXCLUDEd — just not discovered). One withheld task
(`notebook-compression`) is in the green set via an xrlenv-authored `patches/`
overlay that supplies the missing `solve.sh` — see
[Corpus structure](#corpus-structure) and
[Xrlenv-authored solution](#xrlenv-authored-solution-notebook-compression).

| Bucket | Count | Notes |
|---|---|---|
| Total tasks | 17 | |
| In catalog (oracle-gateable + xrlenv-authored overlay) | 12 | 11 ship `solution/solve.sh`; `notebook-compression` added via xrlenv-authored patch |
| **Green set** (CPU-hermetic, G1 verified) | **7** | 5 upstream oracles: `ffmpeg-swscale-rewrite`, `git-to-zig`, `libexpat-to-x86asm`, `revideo-perf-opt`, `dart-style-haskell`; + 1 upstream oracle fixed via curated patch: `dependent-type-checker`; + 1 xrlenv-authored: `notebook-compression` |
| Excluded — defects surfaced by G1 (see [below](#why-1-task-remains-excluded-after-g1)) | 1 | `cranelift-codegen-opt` |
| GPU tasks (EXCLUDEd on CPU-only clusters) | 4 | `gpus=1` in `task.toml`; revisit when GPU nodes exist |
| Solution withheld by upstream, not gateable | 5 | Live-leaderboard anti-leakage; no `solution/solve.sh` → not oracle-derivable; `notebook-compression` was one such task but is now green via an xrlenv-authored solution (see below) |

The 4 GPU tasks (`granite-mamba2-inference-optimization`,
`inference-system-optimization`, `optimizer-design`, `pcqm4mv2-autoresearch`) are
not broken — they are excluded only because the dev cluster is CPU-only. Drop them
from `run_full_sweep.sh`'s `EXCLUDE` and re-pin the catalog counts once GPU nodes
are available.

## Two distinctive traits

### Grade-from-artifact

FrontierSWE's `reward.json` carries a richer schema than harbor's standard
`rewards: dict[str, float|int]`: it includes a `subscores` list and an
`additional_data` dict that harbor 0.20's strict `VerifierResult` validator
rejects. Harbor still downloads the file to disk before the parse error fires.

The oracle sweep reads the **downloaded `reward.json` directly** — the same file
that upstream's `scripts/score_from_reward.py` consumes — instead of using
harbor's parsed result. A task passes if `reward` (falling back to `score`) `> 0`.
The harbor `ValidationError` is treated as expected-and-ignored whenever a
gradeable `reward.json` is present on disk; a **missing** `reward.json` (verifier
produced no output at all) is the only failure the sweep counts as an infra error.

This approach requires no change to harbor, the verifier, or xrlenv-core — it
delegates grading to the same upstream artifact that the official leaderboard
scorer reads. The G1 sweep confirmed this end-to-end: all green tasks passed by reading the
downloaded `reward.json` despite harbor's expected `ValidationError`.

**Reward values are graded quality scores, not binary.** Performance tasks report
approximately the speedup ratio (can exceed 1.0, uncapped); implementation tasks
report a test-pass fraction. Sub-1 rewards are normal — the shipped references are
known-good, not maximal.

### Run-time oracle mode

FrontierSWE's verifiers run an anti-cheat / anti-wrapper source scan that
legitimately flags the reference solution (which wraps or links the library the
task asks the agent to reimplement). The sweep injects `HARBOR_ORACLE_MODE=1` at
run time via the job's `environment.env` and `verifier.env` — harbor threads the
latter to the verifier as `override_env`, relaxing the scan.

This flag is **never baked into `task.toml`**, so a real agent evaluation is
completely unaffected.

## Prerequisites

FrontierSWE uses the generic harbor extra (no benchmark-specific package):

```bash
pip install -e '.[harbor]'
```

Boot the control plane and configure cluster credentials:

```bash
xrlenv up                              # start the control plane

# In .env (auto-loaded by xrlenv on import):
export XRLENV_GRPC_HOST=<cp-host>
export XRLENV_GRPC_PORT=50051
export XRLENV_CONSUMER_TOKEN=<token>   # required if CP has auth enabled
```

## Step 1: build the task-dir cache

`build_cache.py` shallow-clones `Proximal-Labs/frontier-swe` and materializes
each task directory into a shared cache shard at
`$XRLENV_BENCHMARK_CACHE/frontier-swe/`. It is idempotent — re-running skips
tasks already present.

```bash
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example

# populate (git clone) and apply any curated patches — the default:
.venv/bin/python xrlenv_plugins/benchmarks/frontier_swe/build_cache.py --stage all

# populate only (requires network):
.venv/bin/python xrlenv_plugins/benchmarks/frontier_swe/build_cache.py --stage populate

# apply patches only (no network; populate must have run first):
.venv/bin/python xrlenv_plugins/benchmarks/frontier_swe/build_cache.py --stage patch
```

The shard coexists with other benchmark shards (deep-swe, terminal-bench-2-1,
…) under the same cache root — no collision by design.

**Env overrides for populate:**

| Variable | Default | Description |
|---|---|---|
| `FRONTIER_SWE_REPO_URL` | `https://github.com/Proximal-Labs/frontier-swe` | git remote (accepts a local clone path) |
| `FRONTIER_SWE_REPO_REF` | `main` | git ref / branch |
| `FRONTIER_SWE_SHARD` | `frontier-swe` | Shard subdirectory name |
| `XRLENV_BENCHMARK_CACHE` | *(required)* | Cache root |

**Curated patches.** The `patches/` directory contains two overlays applied at
`--stage patch`:

- `patches/dependent-type-checker/` — re-paths the upstream reference implementation
  that `solve.sh` reads, making an existing oracle pass (see
  [Fixed via curated patch](#fixed-via-curated-patch-confirmed-on-cluster)).
- `patches/notebook-compression/` — supplies a complete xrlenv-authored
  lossless-lzma `/app/run`; this task has no upstream reference (withheld), so
  the patch is an authored solution, not a derived oracle (see
  [Xrlenv-authored solution](#xrlenv-authored-solution-notebook-compression)).

FrontierSWE's oracle verifiers are deterministic local computations (no live-pip
installs during the oracle run), so unpinned-dependency drift risk is low. Add
`patches/<task-id>/<relative-path>` files if the oracle sweep surfaces broken
content; re-run `--stage patch` (idempotent, no re-clone needed).

## Step 2: warm images (optional)

Every FrontierSWE task ships a **prebuilt image on public GHCR**
(`ghcr.io/proximal-labs/frontier-swe/<id>:<tag>`, anonymous-pullable). The
cluster pulls each image on first acquire and evicts under disk pressure — no
pre-warm step is needed for a standard sweep.

For large runs where you want all images resident before the sweep starts,
generate and apply the warm plan:

```bash
# regenerate the committed green plan whenever tags change:
GREEN=$(bash xrlenv_plugins/benchmarks/frontier_swe/run_full_sweep.sh --list-green | paste -sd,)
XRLENV_BENCHMARK_CACHE=/path/to/shared/cache \
.venv/bin/python -m xrlenv_plugins.benchmarks.frontier_swe.build_plan_gen \
    --tasks "$GREEN" \
    --output ./xrlenv_plugins/benchmarks/frontier_swe/frontier_swe_build_plan.yaml

# eager-warm across the cluster (FFD bin-packed onto nodes):
xrlenv build apply \
    --plan ./xrlenv_plugins/benchmarks/frontier_swe/frontier_swe_build_plan.yaml \
    --connect-host "$XRLENV_GRPC_HOST" --fill-missing
```

A committed plan (`frontier_swe_build_plan.yaml`) covering the green-7 set is
already in the repo for re-use.

After images are materialized on nodes, refine the heuristic size hints to true
on-disk sizes:

```bash
export XRLENV_OPERATOR_TOKEN=<operator-token>
xrlenv build calibrate \
    --plan ./xrlenv_plugins/benchmarks/frontier_swe/frontier_swe_build_plan.yaml \
    --output ./xrlenv_plugins/benchmarks/frontier_swe/frontier_swe_build_plan.calibrated.yaml \
    --connect-host "$XRLENV_GRPC_HOST"
```

Diff the calibrated plan against the committed one before promoting it.

## Step 3: run the oracle sweep

### One-command gate

```bash
set -a; source ./.env; set +a
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example

bash xrlenv_plugins/benchmarks/frontier_swe/run_full_sweep.sh
```

`run_full_sweep.sh` builds the cache (unless `--skip-build-cache`), computes the
green set (asserts 12 present / 7 green), runs `run_oracle_sweep.py` over those
7 tasks, and content-retries any non-passing tasks up to `--content-retries`
times (default 2). Exit code is non-zero if any task still fails after retries.

**FrontierSWE tasks are heavy** — up to 128 GB / 16 CPUs, with oracle runtimes
of 1–2 hours. Run the sweep under `nohup` and poll the log:

```bash
nohup bash xrlenv_plugins/benchmarks/frontier_swe/run_full_sweep.sh \
    --max-workers 4 --job-id frontier-swe-g1 \
    > tmp/frontier-swe-g1.log 2>&1 &
tail -f tmp/frontier-swe-g1.log
```

Config flags (all optional):

| Flag | Default | Description |
|---|---|---|
| `--max-workers N` | `8` | Trial concurrency |
| `--job-id LABEL` | `frontier-swe-full-sweep` | Run label under `tmp/` |
| `--content-retries N` | `2` | Re-run non-passing tasks up to N times |
| `--jobs-dir DIR` | `./tmp` | Per-trial artifact root |
| `--skip-build-cache` | — | Skip `build_cache.py` (use cache as-is) |
| `--list-green` | — | Print the 7-task green set and exit (12 present → 7 green) |
| `XRLENV_BENCHMARK_CACHE` *(env)* | `/path/to/xrlenv_benchmark_cache` | Cache root |

Extra CLI args pass through to `run_oracle_sweep.py`:

```bash
bash xrlenv_plugins/benchmarks/frontier_swe/run_full_sweep.sh \
    --max-workers 4 --timeout-multiplier 1.5
```

### Direct sweep

```bash
.venv/bin/python xrlenv_plugins/benchmarks/frontier_swe/run_oracle_sweep.py \
    --max-workers 4 \
    --retries 6 \
    --jobs-dir ./tmp \
    --job-id frontier-swe-direct

# Run a subset:
.venv/bin/python xrlenv_plugins/benchmarks/frontier_swe/run_oracle_sweep.py \
    --tasks git-to-zig,libexpat-to-x86asm
```

### Retry layers

| Layer | Granularity | Retries on | Purpose |
|---|---|---|---|
| `--retries` (default 6) | per-task attempt (fresh container) | `CapacityExhausted`, `ControlPlaneLost`, `NodeLost`, `NodeCommandTimeout` only | absorb capacity pacing at high concurrency |
| `--content-retries` (default 2, via wrapper) | per-task | reward-0 outcome — non-passing tasks only | catch one-off environmental flakes |

The reward-schema `ValidationError` from harbor is deliberately **not** in the
infra-retry set — it is expected and handled by grade-from-artifact, not retried.

Per-trial artifacts (agent output, `verifier/reward.json`, logs) land under
`<jobs-dir>/<job-id>/<task-name>__<suffix>/`. A content-retry round writes a
sibling `<job-id>-retryN/` directory; a task passing in any round counts as
solved.

## Pass gate

The sweep grades each trial from the **downloaded** `reward.json`:

```python
# priority: "reward" key, then "score" key
reward = float(artifact.get("reward", artifact.get("score", 0)))
passes = reward > 0
```

A missing `reward.json` (verifier produced no output) is counted as an infra
failure. The harbor `ValidationError` on the parsed result is ignored whenever
the file is present and gradeable. Timeouts run at the task's **native budget**
(no `--timeout-multiplier` in the default gate) — a reference solution that
cannot fit its own declared timeout should fail loud.

## Resource knobs

| Flag | Description |
|---|---|
| `--override-cpus N` | Force every task to N CPUs (ignores `task.toml`). |
| `--override-memory-mb N` | Force every task to N MiB of memory. |
| `--cpus-multiplier F` | Scale each task's declared CPUs by F. |
| `--memory-multiplier F` | Scale each task's declared memory by F. |
| `--cpu-pinning` | Opt the job into cpuset pinning (`nproc` == declared CPUs). Recommended on large-CPU hosts. |
| `--timeout-multiplier F` | Scale harbor's agent/verifier timeouts. |

## Status

**G1 oracle sweep: GREEN 7/7 — 5 upstream oracles + 1 upstream oracle fixed via
curated patch + 1 xrlenv-authored solution.**
The grade-from-artifact design is validated end-to-end — every green task passes
by reading the downloaded `reward.json` despite harbor's expected
`ValidationError`. The first G1 run (2026-08-05, `--max-workers 8`, native
timeouts) greened 4 and surfaced 3 defects in the initial 7-task scope;
`dependent-type-checker` was then fixed via a curated `patches/` overlay
(confirmed on-cluster 2026-08-06, reward 1.0005); `notebook-compression` — a
previously withheld task — was solved by an xrlenv-authored lossless-lzma
`/app/run` (confirmed on-cluster 2026-08-06, reward 0.3175); `dart-style-haskell`
was recovered after the xrlenv-core `put_archive` 128 MiB limit was fixed
(chunked, heartbeat-safe — see [below](#xrlenv-core-put_archive-fix)),
confirmed on-cluster 2026-08-07 (reward 1.0), bringing the gate to 7/7.

| Gate | State |
|---|---|
| Offline unit tests | 31 passed |
| `ruff` / `mypy` (kit modules) | clean |
| `build_cache.py` populate (17 tasks) + patch (2 overlays → 12 gateable) | verified |
| `--list-green` (12 present → 7 green) | verified |
| Committed `frontier_swe_build_plan.yaml` (7 GHCR entries) | generated |
| **G1 — live oracle sweep (reward > 0 for all green)** | **GREEN 7/7 — 5 upstream oracles + `dependent-type-checker` (patch) + `notebook-compression` (xrlenv-authored)** |

### G1 results (2026-08-05)

| Task | Result | Reward | Notes |
|---|---|---|---|
| `ffmpeg-swscale-rewrite` | PASS | 0.9939 | correctness 30/30; reward = geo-mean speedup |
| `git-to-zig` | PASS | 0.7307 | weighted test-pass fraction (reference is partial) |
| `libexpat-to-x86asm` | PASS | 1.0002 | correctness full; uncapped speedup ratio (>1) |
| `revideo-perf-opt` | PASS | 0.9742 | correctness 8/8; reward = geo-mean speedup |
| `dart-style-haskell` | PASS | 1.0 | upstream oracle; oracle wraps a 340 MB Dart SDK; required the xrlenv-core chunked `put_archive` fix (see [below](#xrlenv-core-put_archive-fix)); confirmed on-cluster 2026-08-07 |
| `dependent-type-checker` | PASS | 1.0005 | broken oracle at G1; fixed via curated `patches/` overlay (see below); re-run 2026-08-06 confirmed |
| `notebook-compression` | PASS | 0.3175 | **xrlenv-authored solution** (no upstream oracle); lossless lzma `/app/run`; round-trip lossless on 80 hidden files; confirmed on-cluster 2026-08-06 (see [below](#xrlenv-authored-solution-notebook-compression)) |
| `cranelift-codegen-opt` | EXCLUDE | 0.0 | placeholder oracle (see below) |

Rewards above 1.0 (`libexpat-to-x86asm`: 1.0002) are expected — performance
tasks report an uncapped speedup ratio, so a reference that is marginally faster
than parity scores just above 1.

### Fixed via curated patch (confirmed on-cluster)

**`dependent-type-checker` — oracle reference-path fix.** At G1 this task
scored 0.0 because `solve.sh` did `cd "$(dirname solve.sh)/../tests"` to read
`reference_impl/{Cargo.toml,src/main.rs}`, but `/tests` is only mounted during
*verification*, not during *solve* — a contract upstream documents in its own
sibling oracles (`cranelift/solve.sh`: "Cannot access /tests/ — bundle any
needed resources under solution/"; `libexpat/solve.sh`: "/tests/ is only mounted
during verification"). So `cd /tests` failed, no checker was built, and the
verifier rejected all 174 valid programs (`accept 0/174`).

The fix (`patches/dependent-type-checker/`) bundles the byte-identical
`reference_impl/{Cargo.toml,src/main.rs}` — sha256-matched to the copy in the
verifier's own `tests/tests-bundle.tar.gz` — under `solution/`, and points
`solve.sh` at that sibling instead of `/tests`, following upstream's own
documented pattern. The reference source is unchanged; `solution/` is
oracle-only (no agent leak); `HARBOR_ORACLE_MODE=1` keeps the verifier's
anti-cheat oracle-aware. This is the smallest overlay that lifts the ceiling.

The fix is **offline-validated** (overlay applies cleanly; `solve.sh` is
bash-valid; bundled `main.rs` sha256 matches the verifier bundle) **and confirmed
on-cluster** — a re-run (2026-08-06, `--tasks dependent-type-checker`) built the
reference checker and scored **reward 1.0005** (correctness full). The task is in
the green set.

### Xrlenv-authored solution: `notebook-compression`

**`notebook-compression` — authored lossless-lzma compressor (confirmed on-cluster).**
This task has no upstream `solution/solve.sh` — the reference is withheld for
live-leaderboard anti-leakage. It cannot be *derived* from `tests/` because
`tests/` ships only a hidden test set and a `scoring_core.py` grader (a
correctness spec, not a reference solution). However, unlike the other withheld
tasks, a tractable static solution exists: the task asks for a lossless notebook
compressor, which is an authorable program — not multi-hour frontier research.

The `patches/notebook-compression/` overlay supplies an xrlenv-authored `/app/run`
that compresses with Python's `lzma` module and decompresses before scoring.
Round-trip lossless on all 80 hidden files. This patch is **clearly labelled as an
authored solution** in `patches/README.md` and is kept strictly separate from the 5
upstream oracle patches — it is never presented as a derived oracle.

**Confirmed on-cluster** (2026-08-06): reward **0.3175**; round-trip lossless on 80
hidden notebooks. The task is in the green set.

**Why 0.3175, not higher?** The grader rewards compression ratio — the authored
`lzma` solution is lossless and correct, but not the most aggressive compressor
possible. The reward reflects compression quality, not correctness. Sub-1 rewards
are normal for performance tasks; what matters for the gate is `reward > 0`.

### Xrlenv-core put_archive fix

**`dart-style-haskell` was previously excluded** because its oracle bundles a
340 MB Dart SDK, producing a 639 MB tar upload that exceeded the unary
`ContainerPutArchive` gRPC message limit (128 MiB). The `RESOURCE_EXHAUSTED`
error was also mislabelled `CapacityExhausted`. `container_put_archive` is now
chunked (mirroring the shipped `get_archive` streaming, ~4 MiB per frame,
heartbeat-safe, with a `NodeHello` capability + unary fallback for old peers)
— a general xrlenv-core fix that unblocks any future upload above 128 MiB.
`dart-style-haskell` confirmed on-cluster 2026-08-07 (reward 1.0) and is now a
normal upstream-oracle green.

### Why 1 task remains excluded after G1

The G1 sweep identified three tasks that could not pass under any agent; two
defects have since been fixed or resolved (see above). The remaining one has no
tractable fix:

**`cranelift-codegen-opt` — upstream oracle unimplemented.** `solution/solve.sh`
is a 7-line placeholder (`echo "Oracle solution placeholder — implement me."`).
It optimizes nothing, so the verifier scores the untouched baseline: correctness
subscore 1.0, performance subscore 0.0 (`target_speedup=0.5` unmet), gated
`score=0`. There is no reachable positive ceiling under this oracle. Revisit if
upstream ships a real reference.

### Solution withheld — 5 tasks not gateable

Five tasks ship no `solution/solve.sh` (live-leaderboard anti-leakage). They are
not EXCLUDEd — they simply do not appear in the green set:

| Task | Also GPU? | Why not authorable as a static solution |
|---|---|---|
| `lua-native-compiler` | No | Requires authoring a Lua→native compiler; the correctness spec is the standard Lua interpreter, not a reference compiler |
| `postgres-sqlite-wire-adapter` | No | Requires implementing full PostgreSQL wire-protocol compatibility; graded against the PostgreSQL 18 regression suite |
| `pyright-type-checking-optimization` | No | Performance task — optimizing pyright; same class as `cranelift` (no reference optimization exists to re-path) |
| `modular-stack-wan21` | Yes (GPU) | Reference-output frames are generated by a GPU reference stack; also GPU-only |
| `frogsgame-rl` | No | Graded by the external Tinker API (paid key + stochastic); non-hermetic; no local reference |

`notebook-compression` was previously in this group; it is now green via an
xrlenv-authored lossless-lzma solution (see
[above](#xrlenv-authored-solution-notebook-compression)).

**On derivability vs. authorability.** A withheld task cannot be *derived* — there
is no reference solution in `tests/` to re-path (only correctness specs, baselines,
or external graders). However, one withheld task (`notebook-compression`) was
*authorable* as a static solution: the task asks for a lossless compressor, which
a short `lzma`-based program can satisfy, and it is now green. The other 4
non-GPU withheld tasks are multi-hour frontier research challenges (compiler
implementation, protocol compatibility, ML optimization) that a static `solve.sh`
cannot crack — they stay not-gateable.

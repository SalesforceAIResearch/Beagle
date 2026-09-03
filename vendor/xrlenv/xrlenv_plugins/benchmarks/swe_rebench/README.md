# SWE-rebench

[SWE-rebench](https://hub.harborframework.com/datasets/swe-rebench/swe-rebench-leaderboard/latest)
is a harbor-format corpus of **860 curated Python SWE tasks** from Nebius AI R&D,
namespaced under the `swe-rebench/` shard of the shared benchmark cache. Its
xrlenv shape is the **harbor golden path** — the same one-line `import_path` swap
terminal-bench-2 uses:

```diff
 environment:
-  import_path: harbor.environments.docker.docker:DockerEnvironment
+  import_path: xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster
```

Two distinctive traits:

- **It is a Harbor Hub *package* dataset**, not a git repo or an HF snapshot — so
  harbor is both the harness *and* the download client. `build_cache.py` drives
  harbor's own `PackageDatasetClient` + `TaskClient` (anonymously; the dataset is
  public). No clone, no HF token.
- **Every one of the 860 tasks ships a reference solution**, so the whole corpus
  is oracle-gateable — unlike FrontierSWE, nothing is withheld.
- **Upstream is organised into monthly splits** (`2025_01` … `2026_03`), added
  continuously. The Harbor Hub package flattens them, so the kit restores the
  grouping from a committed index — see [§3](#3-run-the-oracle-sweep).

## What's here

| File | Role |
|---|---|
| `build_cache.py` | Populate the shard from the Harbor Hub, normalize + **repin** each `task.toml` to its prebuilt image, write the **resource routing** + **hermeticity env** + overlays — `--stage all` (default: populate → repin → patch), every stage idempotent |
| `build_plan_gen.py` | Emit the image warm plan (one `type: registry` entry per task, read from `task.toml`) |
| `swe_rebench_build_plan.yaml` | The committed 860-entry warm plan (regeneratable via `--all`) |
| `patches/` | Curated per-task content fixes — **currently empty**; `patches/README.md` |
| `run_oracle_sweep.py` | The correctness gate — OracleAgent per task on the cluster; owns BOTH retry layers |
| `run_full_sweep.sh` | The one-command entrypoint / CI gate — cache → green set (856 = 860 − 4 `EXCLUDE`d) → sweep. `--tasks` / `--tasks-file` narrow the run (intersected with the green set) |
| `scripts/smoke_30tasks.txt` | The **risk-ranked smoke set** — 30 task ids (one per repo, ~65 GB) covering every measurable failure mode. Feed it to `run_full_sweep.sh --tasks-file` before the full gate; the file documents itself |
| `scripts/monthly_splits.json` | The **`instance_id` → monthly-split index** — upstream's 15 monthly splits, which the flat Harbor Hub package drops. Drives `run_full_sweep.sh --split` (§3) |
| `STATUS.md` | Current oracle-sweep disposition + reproduce command |
| `tests/` | Offline unit tests for the files above |

```
Harbor Hub package dataset ──①build_cache──▶ <cache>/swe-rebench/<id>/  (+ docker_image repin)
                                                  │
                            ②build_plan_gen ──▶ type: registry warm plan ──▶ xrlenv build apply
                                                  │
                            ③run_full_sweep ──▶ per-task reward>0 PASS/FAIL on the cluster
```

## 1. Build the cache

`build_cache.py` (default **`--stage all`** = populate → repin → patch) yields a
correct cache in one command:

```bash
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example
.venv/bin/python xrlenv_plugins/benchmarks/swe_rebench/build_cache.py      # --stage all
```

Each task lands at `<cache>/swe-rebench/<instance_id>/` (e.g.
`BerriAI__litellm-14715`) with the harbor filesystem contract:

```
task.toml            resources + timeouts (+ the docker_image repin writes)
instruction.md       the agent-facing problem statement
environment/         Dockerfile (FROM the upstream prebuilt image)
tests/               test.sh · parser.py · config.json (FAIL_TO_PASS / PASS_TO_PASS)
solution/solve.sh    the reference patch — present for all 860
```

Everything is idempotent (an already-present task is skipped; repin and overlays
re-apply), so re-running is safe. The shard is self-contained, so it never
collides with the other benchmarks under the same `XRLENV_BENCHMARK_CACHE`.
`run_full_sweep.sh` runs this same `--stage all` as its step 1 — **there is no
separate step to remember**. A populate writes `.dataset-version.json` recording
the resolved dataset content hash, which `STATUS.md` pins for reproducibility.

`--dest` overrides the cache root (defaults to `$XRLENV_BENCHMARK_CACHE`);
`SWE_REBENCH_DATASET` / `SWE_REBENCH_DATASET_REF` / `SWE_REBENCH_SHARD` override
the dataset identity so a fork or a pinned snapshot swaps in without a code edit.

### Task-level cache fixes — the `docker_image` repin

SWE-rebench tasks carry **no** `[environment] docker_image`. Upstream ships a
prebuilt image per task on Docker Hub
(`swerebench/sweb.eval.x86_64.<slug>:latest`) and expresses it as the `FROM` of a
**three-line** `environment/Dockerfile`:

```dockerfile
FROM swerebench/sweb.eval.x86_64.berriai_1776_litellm-14715:latest
ENV _JAVA_OPTIONS=""
RUN (curl -fsSL "$UV_URL" || wget -qO- "$UV_URL") | UV_INSTALL_DIR=/usr/local/bin sh
RUN mkdir -p /logs
```

The xrlenv harbor cluster environment resolves an image ref at acquire; it does
not build on acquire. So `--stage repin` writes the authoritative upstream ref
into each `task.toml`, turning the corpus into a pull-on-demand `type: registry`
plan with **nothing to build**. The ref is read from the task's own
`tests/config.json` (upstream's declared `docker_image`) and **cross-checked
against the Dockerfile's `FROM`** — a mismatch fails loud rather than silently
pinning the wrong repo snapshot. This is programmatic, not a frozen overlay, so
it tracks a re-populate automatically.

**What the repin drops, and why it is safe.** Two of the three lines are
provably inert under harbor 0.20:

| Dockerfile line | Who depends on it | Verdict |
|---|---|---|
| `RUN mkdir -p /logs` | nobody — harbor's `empty_dirs` runs `mkdir -p /logs/verifier && chmod 777` in-container before the verifier phase (`harbor/environments/base.py`, called from `trial/trial.py`), and all 860 `tests/test.sh` open with their own `mkdir -p /logs/verifier` | inert |
| `ENV _JAVA_OPTIONS=""` | nobody — a guard against a JVM echoing `Picked up _JAVA_OPTIONS` into parsed test output; 0 of 860 `test.sh`/`solve.sh` reference it and no base image sets it | inert |
| the `uv` install | 17 tasks' `tests/test.sh` runs `uv run pytest` / `uv pip install` — but **the base images already ship `uv`**. Measured 2026-09-01: all 17 run on-cluster against the plain upstream image, 16 solved; the 17th (`CQCL__guppylang-1259`) failed on an unrelated upstream packaging defect since fixed by the hermeticity env below, and no trial produced `uv: command not found` | inert |

So the repin is **lossless in practice** and nothing needs building. The `uv`
row is the one that had to be measured rather than reasoned about — see
STATUS.md for the run.

### Task-level cache fixes — resource routing (`--stage patch`)

harbor applies a CFS cpu quota + a hard memory cap but **never a cpuset**, so
inside a `cpus = 1` container `nproc` reports the **host's** core count (192 on
this fleet). Any pool sized from `os.cpu_count()` — joblib/loky, pytest-xdist
`-n auto`, dask/ray, OpenMP/BLAS — then fans out ~192 ways inside an 8 GB cap
and is SIGKILL'd. This cost 16 tasks in the first full sweep.

`--stage patch` writes the surgical remedy the harbor plug-in documents: a
per-task `[environment.env] XRLENV_CPU_PINNING = "1"` marker
(`CPU_PINNING_TASKS`) that sizes the affinity mask to `ceil(cpus)` so `nproc`
matches the task budget, while the quota and memory cap still apply. Four tasks
also need `MEMORY_OVERRIDES`. All 16 are re-verified green through the markers
alone; the evidence trail is in STATUS.md.

**Fairness.** A memory override is allowed **only where upstream declared no
memory**, and `_assert_memory_override_is_fair` enforces that in code — it
raises if `tests/config.json` carries a `harbor_memory`. Upstream states
resource intent when it has one (10 of 860 are explicitly 2 cpu / 16 G); for the
rest the `8G` is the dataset converter's blanket default. Every overridden task
is in the default group, and the upstream-sized tasks needed pinning only.

Grow either set the same way it was built: a task must have failed a real
sweep, reproduced at low concurrency (so it is not contention), and flipped to
passing under the change.

### Task-level cache fixes — hermeticity env (`--stage patch`)

A separate table from the resource routing, because the fairness question is
different: `HERMETICITY_ENV` changes how a verifier resolves its **own
dependencies**, never how much CPU or memory it gets, so no envelope guard
applies. It writes plain `[environment.env]` keys.

One task uses it. `CQCL__guppylang-1259` runs `uv run pytest`, and `uv run`
re-resolves the workspace on every invocation. PEP-517 **build** requirements are
not covered by the lockfile, so the resolve pulls whatever hatchling is current
on PyPI — since 2026-08 that is 1.32.0, which rejects the task's
`readme = "../README.md"`. The package never builds and every `FAIL_TO_PASS` plus
33 `PASS_TO_PASS` report `NOT_FOUND`; the task was authored 2025-09 against a
hatchling that accepted it. Setting `UV_NO_SYNC` = `"1"` tells `uv run` to use
the environment the image already ships. Measured: reward 1, F2P PASSED, 33/33
P2P, and zero download/build lines — so it fixes the grade *and* removes a live
PyPI dependency from the verify phase.

That second property is the general lesson: a verifier that resolves
dependencies at grade time is not hermetic, and will rot the moment an unpinned
transitive dependency publishes. Prefer pinning the resolution off to patching
the symptom.

## 2. Prepare the images

**Nothing is built.** Every one of the 860 tasks pins an upstream prebuilt on
Docker Hub, so the cluster pulls each on first acquire and the plan is 100 %
`type: registry`. Do not push these into the private registry: a 60-image sample
measured ~2.3 GB compressed each at ~1.67× layer dedup, so mirroring the corpus
would add **~1.1 TB** to a store with no quota and no automatic GC.

Warming is optional (§4). Regenerating the committed plan:

```bash
.venv/bin/python -m xrlenv_plugins.benchmarks.swe_rebench.build_plan_gen \
    --all --output xrlenv_plugins/benchmarks/swe_rebench/swe_rebench_build_plan.yaml
```

Size probing is **on by default** here (unlike deep_swe / frontier-swe, whose
images are on GHCR/ECR): these images *are* on Docker Hub, so the shared probe
returns real compressed sizes for the FFD bin-packer. Export `DOCKERHUB_USER` /
`DOCKERHUB_TOKEN` (in `.env`) first — Docker Hub rate-limits an 860-image sweep
even authenticated (~600 succeed, then HTTP 429), and every fallback
over-reserves disk. Probing is therefore **resumable**: re-run with
`--reuse-sizes <the plan>` after the rate window resets (~6 h) until it reports
0 fallbacks. `--no-probe` skips it entirely.

## 3. Run the oracle sweep (validate the cache)

The correctness gate: run each task's shipped `solution/solve.sh` under harbor's
`OracleAgent` and confirm it earns positive reward. **A task the oracle can't
solve is poison for RL** — its reward ceiling is 0 — so under the oracle a
non-passing task is a plumbing/content bug, not a model signal. Exit 0 **iff**
every task passes.

**Start with the risk-ranked smoke set, not the full gate.**
[`scripts/smoke_30tasks.txt`](scripts/smoke_30tasks.txt) lists 30 tasks (~65 GB,
one per repo) covering every measurable failure mode in the corpus — bespoke test
runners, network-touching tests, huge `PASS_TO_PASS` suites, heavy-resource
tasks, oversized images. If the plumbing is broken it breaks there first, ~25×
cheaper. The file explains itself; the signal populations are in `STATUS.md`.

```bash
set -a; source ./.env; set +a          # XRLENV_GRPC_HOST + XRLENV_CONSUMER_TOKEN
S=xrlenv_plugins/benchmarks/swe_rebench

# one task first — proves cache + repin + cluster + grading end to end
bash $S/run_full_sweep.sh --tasks prometheus__client_python-1134 --max-workers 1

# then the 30, at BOTH concurrencies (see below)
bash $S/run_full_sweep.sh --tasks-file $S/scripts/smoke_30tasks.txt --max-workers 8
bash $S/run_full_sweep.sh --tasks-file $S/scripts/smoke_30tasks.txt --max-workers 32

# only then the real gate
bash $S/run_full_sweep.sh --max-workers 32
```

Run the smoke set at **both** concurrencies: a network-touching task that passes
solo and fails under load is not flaky — it is a non-hermetic dependency
surfacing, and the fix is hermeticity, never a lower concurrency.

### Running one monthly split

Upstream publishes 15 monthly splits (`2025_01` … `2026_03`) whose union is
exactly this 860-task corpus, and adds one each month. The Harbor Hub package
this kit downloads is the flat `test` split — every task's `source` reads
`…::test/<id>` with no month — so the mapping is restored from the committed
`scripts/monthly_splits.json` (written by `scripts/fetch_monthly_splits.py`) and
exposed as `--split`:

```bash
bash $S/run_full_sweep.sh --list-splits            # names + task counts
bash $S/run_full_sweep.sh --split 2026_03          # the newest 110 tasks
bash $S/run_full_sweep.sh --split 2025_01,2025_02  # 185 tasks
```

`--split` resolves to ids and then flows through the ordinary `--tasks`
intersection, so `EXCLUDE` still applies and it composes with `--tasks` /
`--tasks-file` (the union is taken). An unknown split fails loud and lists the
valid ones.

**Do not derive the split from `config.json`'s `created_at`.** That is the
upstream PR date; it agrees with the split name for only 14 of the 15. The
newest split absorbs every recently-collected task, so `created_at` misfiles
64 of 860 into `2026_04` / `2026_05`, which are not splits at all. Regenerate
the index after a corpus refresh — run `scripts/fetch_monthly_splits.py`.

A full 860-task SWE oracle sweep far exceeds any foreground shell timeout — run
it under `nohup`/background and poll. `--tasks a,b,c` / `--tasks-file PATH`
narrow any run; both are intersected with the green set, so an `EXCLUDE`d task is
skipped with a note rather than smuggled in.

**The pass gate.** Every task's `tests/test.sh` runs the repo's test command,
hands the log to the corpus-wide `tests/parser.py` (byte-identical across all
860), and writes a flat `/logs/verifier/reward.txt` of `0` or `1` — resolved iff
every `FAIL_TO_PASS` **and** `PASS_TO_PASS` test passes. harbor 0.20 parses that
into `rewards={"reward": <float>}`, and `_trial_passes` requires **every** reward
key `> 0`. Grading is stock harbor end-to-end: no grade-from-artifact seam, no
re-implemented resolution rule (`parser.py` stays upstream's). On a failure,
`verifier/report.json` names exactly which tests did not pass.

**The two-retry-layer design.** Both layers live in `run_oracle_sweep.py`; the
wrapper just passes the flags.

| Layer | Granularity | Retries on | Purpose |
|---|---|---|---|
| `--retries` (default 6) | per-**trial** | the infra-transient set only — `CapacityExhausted`, `ControlPlaneLost`, `NodeLost`, `NodeCommandTimeout`, `SessionReaped` | absorb capacity pacing; **cannot** mask a flaky task |
| `--content-retries` (default **0**) | per-**task** | a reward-0 *outcome* (re-runs ONLY the non-passing tasks) | **off by default** — a task that only passes on a re-run is a *finding*, not a pass: non-deterministic reward is noise an RL run inherits. Raise it to confirm a suspected flake, never to green a gate |

A content-retry round writes a sibling `<job-id>-retryN/` dir which is then folded
back into the main dir, so a flaky task is **surfaced in `STATUS.md`, not silently
greened**. Pass `--content-retries 0` for a zero-tolerance gate.

**Timeouts run at native budget** (3000 s agent + 3000 s verifier; 4 tasks get
6000 s). The gate never passes `--timeout-multiplier` — a task whose own
reference solution can't fit its own `timeout_sec` should fail loud, not be
rescued by inflated headroom.

**Concurrency is a trigger, not a cause.** Callers may request any
`--max-workers`; xrlenv paces capacity via fail-fast acquire plus the infra-only
`--retries`. Never lower it to turn a red run green — that hides a latent bug
instead of fixing it.

`--list-green` prints the green set (present − EXCLUDE) and exits without running
anything; it implies `--skip-build-cache`. This is the seam the integration
runner uses to sample tasks for CI without re-implementing the exclusion logic.

## 4. Warm the image and Calibrate the image size (optional)

```bash
xrlenv build apply --plan xrlenv_plugins/benchmarks/swe_rebench/swe_rebench_build_plan.yaml \
    --fill-missing --connect-host "$XRLENV_GRPC_HOST"
```

**Optional** — the cluster's dynamic image cache (lazy pull-on-acquire + LRU
eviction + image-affinity) means you can skip pre-warming entirely. Pre-warm only
to amortize the first-acquire pull across a big run. Note the scale: warming all
860 pulls on the order of **1 TB** through the `:5010` pull-through mirror, so
prefer warming the subset you are about to run (`--tasks "$(bash …
run_full_sweep.sh --list-green | paste -sd,)"`).

```bash
xrlenv build calibrate --plan .../swe_rebench_build_plan.yaml \
    --output .../swe_rebench_build_plan.calibrated.yaml --connect-host "$XRLENV_GRPC_HOST"
```

`calibrate` queries each node for the **actual on-disk** size and writes a
separate `.calibrated.yaml` so you can diff before promoting. Those true sizes
feed the FFD bin-packer for tighter placement. Note this is the *uncompressed*
on-disk size — not the registry's compressed size the probe reports; the two
differ substantially for these images.

## See also

- [`../GUIDELINE_onboard_benchmarks.md`](../GUIDELINE_onboard_benchmarks.md) — the onboarding convention this kit follows
- [`../../harbor/README.md`](../../harbor/README.md) — the shared harbor cluster environment
- [`STATUS.md`](STATUS.md) — current results + the exact reproduce command
- `docs/supported_benchmarks_and_harnesses/swe_rebench.md` — the Sphinx user page

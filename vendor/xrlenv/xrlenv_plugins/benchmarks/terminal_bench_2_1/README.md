# terminal-bench-2-1

terminal-bench-2-1 is a **harbor-format** corpus: each task unpacks to a
self-contained container image, a `solution/solve.sh` reference, and a verifier
that writes a reward file — the harbor filesystem contract. Its xrlenv shape is
the **harbor golden path**: the sweep reuses the shared cluster environment
(`xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster`) with zero adapter code,
so this directory is a self-contained ops kit (cache builder, warm-plan
generator, oracle gate) on top of generic core.

## What's here

| File | Role |
|---|---|
| `build_cache.py` | populate a faithful copy of the corpus into the shared cache, then patch xrlenv's curated pins (solve.sh dep pins + `[environment.env]` cpuset markers) |
| `build_plan_gen.py` | emit the `type: registry` image-warmup plan (per-task `docker_image` read from `task.toml`) |
| `build_plan_89_full.yaml` | committed `--all` warmup plan (89 entries, registry-probe sizes) |
| `build_plan_89_full.calibrated.yaml` | the plan after `calibrate` (true on-disk sizes) |
| `run_oracle_sweep.py` | oracle-per-task correctness gate on the xrlenv cluster (owns both retry layers) |
| `run_full_sweep.sh` | thin one-command entrypoint: build cache → green set → invoke the sweep |
| `tests/` | offline unit tests for the pure patch / build-plan / pass-gate logic |
| `STATUS.md` | point-in-time oracle-sweep disposition + reproduce command |

```
build_cache.py ─▶ <cache>/terminal-bench-2-1/<task>/  ─▶ build_plan_gen.py (type: registry warm) ─▶ run_oracle_sweep.py (reward>0 PASS/FAIL)
```

## 1. Build the cache

`build_cache.py --stage all` takes a fresh box from nothing to a ready-to-use
cache in two idempotent stages:

1. **POPULATE** — materialize a faithful copy of the upstream tasks into
   `<cache>/terminal-bench-2-1/`. The default `--source registry` pulls the
   frozen upstream dataset via harbor's own downloader (slow, ~1 min/task; needs
   network + harbor registry reachability). `--source seed --seed-dir <dir>`
   populates offline from an existing harbor export (the dataset is
   content-frozen, so a prior export is byte-identical to a fresh pull).
2. **PATCH** — apply xrlenv's curated pins on top (see below).

`--dest` is the shared harbor cache **root** and defaults to
`$XRLENV_BENCHMARK_CACHE`; the dataset lands under `<dest>/terminal-bench-2-1/` so
future datasets sit beside it under the same root. The shared patched cache
lives at `<shared-cache-root>` — point every xrlenv
consumer's `XRLENV_BENCHMARK_CACHE` at it.

```bash
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example

# populate (if missing) + patch. Idempotent; safe to re-run.
.venv/bin/python xrlenv_plugins/benchmarks/terminal_bench_2_1/build_cache.py --stage all

# populate only (network box) / patch only (no network):
.venv/bin/python .../build_cache.py --stage populate
.venv/bin/python .../build_cache.py --stage patch
```

> **Path source only.** These pins reach a consumer that resolves the task by
> local *path* (`TaskConfig(path=...)` → harbor `LocalTaskId`, used as-is —
> harbor never re-downloads it). A job that resolves by registry *source*
> (`source: terminal-bench/terminal-bench-2-1` + a `sha256` ref) reads harbor's
> own `packages/<hash>/` cache and will **not** see them — switch such a job to
> the path-based source.

### Task-level cache fixes

Several tb2.1 oracle *solutions* are **not hermetic** — their `solve.sh` runs a
live `pip install` (so a transitive dep can drift under a frozen task), or their
build scales parallelism to `nproc` (so it fans out to the host core count on a
big node). Left alone, a previously-green oracle silently scores 0, which is
poison for RL (reward ceiling 0). Both fixes are **task/cache-level** — applied
by `--stage patch` on top of an otherwise-faithful copy, **never** by rebuilding
an image — so they survive re-populate and §2 stays thin. There are two kinds,
both declared as auditable in-code tables in `build_cache.py`:

- **`PATCHES` — one-line `solve.sh` dependency pins.** For oracles that fetch a
  dep that drifted. Concrete example: **`build-cython-ext`** — `planarity 1.0.0`
  (2026-06-29) dropped the networkx-graph `pos` attribute pyknotid 0.5.3's test
  suite needs, so the unpinned `pip install -e .` regressed to reward 0; the
  patch inserts `pip install 'planarity==0.6'` (last-good) before it, restoring
  `11 passed`. Each pin is inserted before an `anchor` line and guarded by a
  `sentinel` for idempotency; a missing anchor **fails loud** (upstream changed
  the solve script's shape) rather than silently no-op'ing.
- **`ENV_PATCHES` — per-task `[environment.env]` cpuset-pinning markers.** harbor
  honors a task's `cpus`/`memory` as a CFS quota + hard memory cap but never sets
  cpuset, so `nproc` inside the container reports the *host* core count. Oracles
  that scale to it (`make -j$(nproc)`, ninja's auto `-j`, OpenBLAS/OMP pools) fan
  out to ~host-count workers and OOM under their declared memory cap (proven for
  `install-windows-3.11`'s QEMU build → `cc1` SIGKILL). The marker sets
  `XRLENV_CPU_PINNING = "1"` under the task's `[environment.env]` table; the
  harbor plugin reads it and sizes the affinity mask to the declared `cpus` so
  `nproc` matches the task budget — quota + memory cap still enforced. Currently
  marked: `install-windows-3.11`, `caffe-cifar-10`, `build-pov-ray`,
  `rstan-to-pystan`, `sqlite-with-gcov`.

**Faithfulness:** each fix is the smallest overlay that restores the last-good
result, logged on every run, and lives in the benchmark content — xrlenv core is
untouched. To add a pin when a sweep surfaces another drift victim, add a row to
`PATCHES` (or `ENV_PATCHES`) and re-run `--stage patch`.

## 2. Prepare the images

**Thin — nothing is built.** Every tb2.1 task ships a **prebuilt** registry image
(Docker Hub, `alexgshaw/<task>:<tag>`), and its per-task ref is **read from
`task.toml`'s `[environment] docker_image`** — never synthesized. Most tasks are
on `:20251031`; a handful were rebuilt at newer tags (`:20260403` / `:20260430`),
so reading the ref per-task (rather than hard-coding one tag) warms the exact
image the eval resolves on acquire. A task with no `docker_image` fails loud.

The cluster's lazy image cache pulls each task's image on first acquire, so
warming is **optional**. `build_plan_gen.py --all` emits a `type: registry` warm
plan (each entry lowers to a node-side `docker pull` under `xrlenv build apply`) —
see §4 to eager-warm it across the fleet.

## 3. Run the oracle sweep (validate the cache)

The proof the cache is good: run harbor's OracleAgent for each task **on the
xrlenv cluster** and confirm every task earns a positive reward. `run_full_sweep.sh`
is the thin gate — it (1) sources `./.env` for the CP host + token, (2) rebuilds
the cache (`build_cache.py --stage all`), (3) computes the green set = **present
tasks − `EXCLUDE`** and asserts the count, and (4) invokes `run_oracle_sweep.py`
once over that set, trusting its exit code.

```bash
set -a; source ./.env; set +a                      # XRLENV_GRPC_HOST + token
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example

bash xrlenv_plugins/benchmarks/terminal_bench_2_1/run_full_sweep.sh   # 88 green tasks
# override concurrency: ... run_full_sweep.sh --max-workers 64
```

**Pass gate.** tb2.1's `_trial_passes` requires the trial to complete with
verifier rewards fully populated and **every reward value > 0** (a single-key
reward in practice). Exit code is 0 **iff every oracle solved**, so the sweep is
CI-usable. An oracle FAIL is a corpus defect (usually a drifted unpinned dep —
add a `PATCHES` row), not a model signal.

**The two retry layers** both live in `run_oracle_sweep.py` (so every driver —
the wrapper and the `xrlenv_plugins/benchmarks/tests/integration/` ci runner — gets them from one place, no
bash re-implementation):

| Layer | Granularity | Retries on | Purpose |
|---|---|---|---|
| `--retries` (default 6) | per-**task attempt** (fresh container each) | the 4 infra-transient exceptions only (`CapacityExhausted`, `ControlPlaneLost`, `NodeLost`, `NodeCommandTimeout`) | absorb capacity pacing at high `--max-workers`. **Final stats = one result per task** (a retried-then-passed task counts once, never double-counted); a content outcome is never re-rolled. Common case is a fail-fast **acquire**; a post-acquire infra error re-runs the whole attempt in a new container, so `solve.sh` can *execute* more than once — matters only for **external** side effects, not the recorded result |
| `--content-retries` (default 2 via the wrapper) | per-**task** | a reward-0 *outcome* — re-runs ONLY the non-passing tasks | catch a one-off environmental flake (transient DNS / verifier blip) that surfaced as reward-0 rather than a typed exception; a task is solved if ANY attempt passes |

Both layers report their counts, so a task that only passes on a re-run surfaces
as *flaky*, not silently greened. Pass `--content-retries 0` for a zero-tolerance
gate. **Timeouts run at native budget** (no `--timeout-multiplier` in the gate) —
a reference solution that can't fit its own `timeout_sec` should fail loud, not
be rescued by inflated headroom. tb2.1 additionally passes `--cpu-pinning` (opt
every task in the job into cpuset pinning) as an ablation knob for
timing-sensitive tasks.

**Other flags.** `run_full_sweep.sh` accepts `--max-workers N` (default 32; tb2.1
is all-runc, no sysbox cap, so it can go to 64), `--skip-build-cache` (reuse a
cache built this session), `--list-green` (print the green set and exit — the
seam `xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py` samples), `--job-id` / `--jobs-dir`,
and forwards anything unrecognized to `run_oracle_sweep.py`.

`EXCLUDE` currently holds one **operational** exclude — **`caffe-cifar-10`**,
whose CIFAR-10 dataset host is very slow (the oracle busts its wall-clock on the
download, not on our infra) — so the green set is **88 of 89**. This is *not* a
broken oracle; drop it once the dataset is pre-seeded into the image. Per-trial
artifacts (`agent/oracle.txt`, `verifier/reward.txt`, `result.json`) land under
`<jobs-dir>/<job-id>/`, downloaded back to the launch machine; each content-retry round writes
a **sibling** `<job-id>-retryN/` dir, so a retried task's artifacts live there, not under the
base job.

## 4. Warm the image and Calibrate the image size (optional)

Warming is optional because the cluster pulls each image lazily on first acquire;
pre-warm only to amortize the first-acquire pull across a big run. Generate the
plan for the whole populated shard, then eager-warm every image across the fleet:

```bash
# regenerate the committed plan (registry-probed sizes) whenever the shard's tags change:
# XRLENV_BENCHMARK_CACHE is read from .env
.venv/bin/python -m xrlenv_plugins.benchmarks.terminal_bench_2_1.build_plan_gen \
    --all --output ./xrlenv_plugins/benchmarks/terminal_bench_2_1/build_plan_89_full.yaml

# eager-warm across the cluster (FFD bin-packed onto nodes):
xrlenv build apply \
    --plan ./xrlenv_plugins/benchmarks/terminal_bench_2_1/build_plan_89_full.yaml \
    --connect-host "$XRLENV_GRPC_HOST" --fill-missing
```

The committed plan's `size_hint_bytes` are Docker-Hub **registry-probe**
(compressed manifest) sizes. After the images are materialized on the nodes,
`xrlenv build calibrate` refines them to true on-disk `cluster-reported` sizes in
a **separate** `*.calibrated.yaml` (diff before promoting):

```bash
export XRLENV_OPERATOR_TOKEN=<operator token>
xrlenv build calibrate \
    --plan ./xrlenv_plugins/benchmarks/terminal_bench_2_1/build_plan_89_full.yaml \
    --output ./xrlenv_plugins/benchmarks/terminal_bench_2_1/build_plan_89_full.calibrated.yaml \
    --connect-host "$XRLENV_GRPC_HOST"
```

## See also

- `xrlenv_plugins/benchmarks/GUIDELINE_onboard_benchmarks.md` — the onboarding
  convention this kit follows (§3 golden-path file contracts, §4 workflow, §5
  image/registry mechanics).
- `xrlenv_plugins/harbor/README.md` — the shared harbor cluster environment that
  runs these tasks.
- `docs/supported_benchmarks_and_harnesses/harbor_framework.md` — the harbor
  framework Sphinx page (the `terminal-bench-2` section).
- [`STATUS.md`](STATUS.md) — current oracle-sweep disposition (green set +
  reproduce command).

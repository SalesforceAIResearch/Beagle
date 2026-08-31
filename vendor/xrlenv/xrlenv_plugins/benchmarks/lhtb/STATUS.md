# LHTB — oracle sweep status (surprises + fixes only)

The corpus-quality gate runs harbor's OracleAgent (each task's shipped `solution/`
reference) per task **on the xrlenv cluster** and reads the dense-float reward. The
paper's oracle-success bar is **`R ≥ τ`, τ = 0.95**; the plumbing gate in
`run_full_sweep.sh` is the looser `reward > 0` (a positive reward = the reference
produced a gradable result).

**This file records only the surprises.** Tasks that naturally reach `R ≥ 0.95` with
the shipped `solution/` and needed no intervention are **omitted** (that's the
expected case — 14 of 46: `alp-paper-reproduction`, the 4 `apex-*-matter`,
`audio-visual-event-alignment`, `foldseek-paper-reproduction`, `langchain-version-migration`,
`microscopy-cell-count-qc-audit`, `riscv-core-debug`, `rush_hour_campaign`,
`satellite-flood-change-detection-audit`, `su2-airfoil-regression`, `unison-paper-reproduction`).
Everything below fell into at least one of:

- **(a)** we modified the task cache / grader / image to get a non-zero reward,
- **(b)** flagged by [issue #2](https://github.com/zli12321/LHTB/issues/2) (the TBD set — run, but tautological), or
- **(c)** `R < 0.95` after all fixes (can't reach the paper's success bar no matter what).

> **Headline** (three sets, two modes — don't conflate). Every task is GREEN, TBD, or
> BLACKLIST. The **default (§2) sweep** runs GREEN + TBD = **43 of 46** — 31 GREEN
> (incl. the 4 rebuilt tasks on their §2 private images) + 12 TBD (issue-#2); the
> `--use-upstream-image` gate runs **37** (27 GREEN + 10 TBD — the 6 rebuild tasks drop
> out on their broken docker.io images). Only the **3 BLACKLIST** tasks (`super-mario`,
> `sudoku-recovery`, `apex-openroad-ibex-signoff` — upstream defects) are never run.
> `duckdb` + `robotics-slam` used to be green via a `solve.sh` workaround; the root
> (in-image) fix made them rebuild-gated. **What actually succeeds:** the honest
> references reach the paper's `R ≥ 0.95` bar on the solvable tasks; the rest are below
> 0.95 **by design** (band-graded games + dense metrics — see (c1)/(c2)). Zero xrlenv defects.

## Gate config (current)

`run_full_sweep.sh`, DEFAULT **§2 mode** (the 6 REBUILD tasks on their rebuilt private
images). Concurrency via `--max-workers` (8 = content proof, 32 = platform proof);
tasks run at their native `timeout_sec` (`--timeout-multiplier` is an ablation knob,
not the gate — but note the current default sweep invocation uses `1.5`; see the
[Reproduce](#reproduce) caveat). Two retry layers, both reported: `--retries`
(per-*trial*, infra-transient exceptions only) + content-retries (per-*task*, reward-0
outcome, `--content-retries`, default 2). Gate rule = `reward > 0`; the paper's success
bar is the stricter `R ≥ 0.95`. README §3 has the full walkthrough.

| Bucket | Count | Meaning |
|---|---:|---|
| ✅ **GREEN** | **31** | reference clears the bar (band-pass or `R ≥ 0.95`); incl. the 6 REBUILD tasks green on their §2 private images |
| 🔍 **TBD (issue #2)** | **12** | run but tautological (graded vs an unpublished private reference) — tracked apart from GREEN |
| ⛔ **BLACKLIST** | **3** | never run — upstream defects (`super-mario`, `sudoku-recovery`, `apex-openroad-ibex-signoff`) |
| **Total** | **46** | default §2 sweep runs GREEN+TBD = **43**; `--use-upstream-image` runs **37** (drops the 6 rebuilds) |

> Last authoritative full sweep recorded: `lhtb-authoritative-0718-100002` (conc-8,
> 2026-07-18) — **37/46 reward>0**; the green/TBD/blacklist disposition above reflects
> the 2026-07-21+ refinements (worker-generated game refs, `commit0` sealing, the duckdb
> root-cause). A fresh `run_full_sweep.sh` `result.json` would refresh the run metadata
> (job id, wall-clock, `n_errored_trials`/`n_retries`).

## (a) Fixes applied, by task

Every task we modified to get a gradable or higher reward — **16 tasks fixed**; the
other 30 ran as shipped. All idempotent `build_cache.py --stage patch` steps, faithful
(repair the benchmark's own reference — no re-implementation). Most write into the
cached *task dir* and green the oracle out-of-box. **🔁 = build-context fix** (Dockerfile
/ harness) that's baked into the image on rebuild — persistent for the oracle *and* a
real agent, but the task passes only after the §2 build+push+repin (`duckdb`, the 3
`patch`-less audits — `climate`/`materials`/`robotics-slam`). **commit0** is the one
sweep-side fix (see its row). The `also` column flags overlap with **(b)** issue-#2 /
**(c)** still-`< 0.95` / **rebuild-required**.

| Task | R after | Fix | also |
|---|---:|---|---|
| dicom-radiology-audit | 1.00 | `recover_files_based_oracles` — ships a reference under `solution/files/` but no `solve.sh`; synth `solve.sh` (`cp -a solution/files/. /app/`) | (b) |
| epa-swmm-stormwater-regression-audit | 1.00 | `recover_files_based_oracles` | (b) |
| epidemic-inverse-control-audit | 1.00 | `recover_files_based_oracles` | (b) |
| nrel-pysam-hybrid-renewables-audit | 1.00 | `recover_files_based_oracles` | (b) |
| opensees-seismic-structural-regression-audit | 1.00 | `recover_files_based_oracles` | (b) |
| gdal-proj-raster-regression | 1.00 | `recover_files_based_oracles` | — (issue's fair counterexample) |
| modflow6-groundwater-regression-audit | 1.00 | `recover_files_based_oracles` | — |
| climate-netcdf-extreme-event-audit | 1.00 🔁 | `bake_patch_binary` — image ships no GNU `patch`; add `RUN apt-get install -y patch` to `environment/Dockerfile` (baked on rebuild) | (b) |
| materials-phase-diagram-audit | 1.00 🔁 | `bake_patch_binary` | (b) |
| robotics-slam-benchmark-repair | 1.00 🔁 | `bake_patch_binary` | rebuild-required |
| duckdb-optimizer-closure | 0.77 🔁 | `fix_nproc_scaling_oracles` — `ninja -j{os.cpu_count()}` = `-j192` OOM; rewrite to `sched_getaffinity` in the **baked harness** (`environment/harness/`) + verifier, with cpuset pinning ON (`XRLENV_CPU_PINNING=1`, boolean) so affinity = `ceil(cpus)=4` cores → `sched_getaffinity` reads 4 → `-j4`. No `solve.sh` sed — the fix ships in the rebuilt image. **Reward ~0.77 is HARNESS-CAPPED, not trampling** — the no-op oracle → geomean 1.0006× → `min(1, /1.3)`; per-query full-CSV-reload + `threads=1` make optimizer edits invisible (README §c2a, root-caused 2026-07-30) | rebuild-required, (c) |
| vector-db-iterative-build | 0.86 | `raise_slow_verifier_timeouts` — `[verifier] timeout_sec` 600→2400 (heavy HNSW rebuild) | (c) |
| sokoban | 0.59 | `regenerate_sokoban_reference` — replay the moves shipped as data in `gen/reference_solutions.json` (pure-Python, seconds) | (c) |
| 2048 | 0.60 | committed `patches/…` HQ reference (full expectimax game, **band 6 = the 2048 tile**, worker-generated); `regenerate_2048_reference` (band 4, ~0.36) is the fallback. Reward `raw/11` capped by design (win=0.55, 1.0=65536 tile) | (c) |
| snake_maze_campaign | 0.52 | committed `patches/…` HQ reference (**43 foods → band 8**; best of a 90-seed worker sweep); `regenerate_snake_maze_reference` (25 foods, ~0.18) is the fallback. Reward `min(1, foods·(foods+1)/2 / 1830)`, 60 foods = 1.0 | (c) |
| commit0-multilib-tdd | 1.00 | **native seal** (harbor 0.20): `[environment] allow_internet=false` → `NO_NETWORK` shim + the cluster env's `_apply_baseline_network_policy` seal the container via `apply_egress` before grading. Anti-cheat sees a sealed net (`no egress (sandbox sealed)`) → 784/784. Cluster-confirmed via the native path alone (reward 1.0) | — |

> **Scope:** `bake_patch_binary` touches only the 3 images confirmed to lack `patch`
> (`climate`, `materials`, `robotics-slam`); the other audit tasks already ship it and
> stay on docker.io — no rebuild. Game `reference_*.log` files
> carry the "never in training corpora" canary: the strong `2048` (band 6) and
> `snake_maze_campaign` (band 8) references are **committed under `patches/`**
> (worker-generated; see `patches/README.md`), with the in-build `regenerate_*` as the
> low-band fallback; `sokoban` regenerates in-cache from shipped data; `super-mario`
> stays an offline docker gen.

## (b) Issue #2 — grading against a private reference → SKIP by default

[Issue #2](https://github.com/zli12321/LHTB/issues/2) (author `zhaoyb1990`) raises a
**valid content-validity concern**: a cluster of science/engineering tasks grade a
**near-exact per-field match** to a reference implementation's output, but the
required **output schema (column names, JSON keys), magic constants, and
unit/algorithm conventions are NOT stated in `instruction.md` and NOT present in the
container**. In several the grader's embedded reference is byte-for-byte identical to
`solution/`. So the only path to a high score is to **reproduce the author's exact
implementation**, not to do the task correctly — a numerically/physically correct
submission can score ~0 for naming a column `pos_x_km` instead of `x_km`, or using a
defensible-but-different convention.

**Consequence for us:** a passing *oracle* on these tasks is a **tautology** (the
oracle runs the same private reference), so it validates nothing about solvability.
**We skip them by default, even for the oracle eval** — they are in
`run_full_sweep.sh` EXCLUDE. All 12 currently score `1.0`; the `fixed?` column marks
the ones we *also* had to repair (see the (a) table) just to make the tautology run.

| Task | R | fixed? | note |
|---|---:|---|---|
| epidemic-inverse-control-audit | 1.00 | files-based | |
| dicom-radiology-audit | 1.00 | files-based | |
| epa-swmm-stormwater-regression-audit | 1.00 | files-based | |
| nrel-pysam-hybrid-renewables-audit | 1.00 | files-based | |
| opensees-seismic-structural-regression-audit | 1.00 | files-based | |
| climate-netcdf-extreme-event-audit | 1.00 🔁 | `bake_patch_binary` (rebuild) | |
| materials-phase-diagram-audit | 1.00 🔁 | `bake_patch_binary` (rebuild) | issue's softer "additional potential bug" |
| matpower-opf-regression | 1.00 | — | |
| spice-ephemeris-regression | 1.00 | — | |
| great-expectations-audit | 1.00 | — | |
| document-table-layout-reconstruction | 1.00 | — | `*-reconstruction` |
| scientific-figure-data-reconstruction | 1.00 | — | `*-reconstruction` |

> **Not flagged:** the issue explicitly names `gdal-proj-raster-regression` as a
> **fair counterexample** (kept in the green set), and praises `riscv-core-debug`,
> `poc-exploit-craft`, `apex-openroad-ibex-signoff`, and the `apex-*-matter` tasks as
> clearly fair.

## (c) Below τ = 0.95 after all fixes — two very different kinds

The paper's `R ≥ 0.95` success bar is on the **normalized** reward. But some tasks
*cannot* reach it by construction, and split into two groups that must be read
differently:

### (c1) Band-graded games — normalized reward capped by design; **PASS via band** (GREEN)

The **game / campaign** tasks (`2048`, `sokoban`, `snake_maze_campaign`, and
`super-mario` once generated) grade with a **dual scheme**:

- a **normalized** `reward = raw / MAX_REWARD` (0–1, what our sweep reads and what the
  paper thresholds), where `MAX_REWARD` is a deliberately **impractical ceiling** — the
  2048 65536-tile (band 11), all 155 sokoban levels, 60 snake foods, Mario world 2+ —
  so the normalized value **cannot reach 0.95** even for a strong/perfect reference;
  **plus**
- a `raw_reward` (band + fractional progress, in `raw_reward.txt`) and an integer
  **band**, and the **actual PASS/FAIL is `band ≥ PASS_BAND`** — which the reference
  *does* meet.

So these are **legitimate green passes** (they clear the band bar); the low normalized
reward is an artifact of the impractical denominator, not a weak reference. **Report
both** — the sweep now prints `reward=<normalized> raw=<raw>` (from `raw_reward.txt`),
and the table below shows band + PASS_BAND. **Document, keep green, don't treat the
sub-0.95 normalized as a failure.**

| Task | normalized R | raw_reward | band | PASS_BAND | pass? | note |
|---|---:|---:|---:|---:|:--:|---|
| 2048 | 0.604 | 6.64 | 6 | 1 | ✅ | band 6 = the classic **2048 win**; `/11` (65536 tile). HQ worker-generated full game; fallback band 4 (0.36) |
| sokoban | 0.594 | 92.0 | 92 | 1 | ✅ | **92 / 155 levels** solved (the shipped move data's ceiling); `/155` |
| snake_maze_campaign | 0.517 | (43 foods) | 8 | 3 | ✅ | 43 foods (best of a 90-seed worker sweep); `/1830`-weight (60 foods = 1.0). No `raw_reward.txt`; reward is band-derived |
| super-mario | — | — | — | 1 | ⛔ | same scheme (`/3` = world 2+) but **not yet generated** — needs a torch+net docker gen image (excluded) |

### (c2) Other sub-0.95 tasks — **need a final review**

Not band-graded games. The dense ones are *continuous performance metrics* an honest
reference genuinely maxes below 0.95 (verified from each grader); the rest are still
0 / blocked. These are the set to eyeball before calling the corpus done.

| Task | R | kind | note |
|---|---:|---|---|
| grammar-fuzz-coverage-hunt | 0.92 | dense (green) | `0.6·line+0.4·branch` coverage; last ~9% hard-to-reach |
| spot-scheduler-traces | 0.90 | dense (green) | online policy vs offline-DP hindsight optimum |
| vector-db-iterative-build | 0.86 | dense (green) · **(a)** | `raise_slow_verifier_timeouts` 600→2400 s; heavy HNSW rebuild |
| poc-exploit-craft | 0.80 | dense (green) | reward pinned 0.8 for reference-sized PoC (issue calls it fair) |
| generals-bot-arena | 0.79 | dense (green) | mean win-rate; bot genuinely loses ~20% vs tough tiers |
| nbody-accel-iterative | 0.16 | dense (green) | speedup / 20× cap; oracle hits a real 5×, correct |
| tabular-data-feature-covshift | 0.05 | dense (green) | back-weighted² reward + anti-leak guard zeroes high scores |
| sudoku-recovery | 0.00 | unrecovered | upstream ref/harness-model conflict (needs root; harbor runs oracle as `agent`) |
| chess-mate | — | build+push route | multi-service compose; `game` sidecar built + pushed to the private registry via `build_plan_gen --all` → `deploy/registry/build_and_push_images.py` (README §2). Runs in the ordinary sweep after the rebuild — the sidecar namespace is derived from the repinned main ref (no `IMAGE_TEMPLATE`); excluded only from the out-of-box docker.io gate |
| unknown-config-semantics | 0.00 | stale image | pinned `:20260615` bakes an old daemon w/o `nonce` → rebuild+repin |
| apex-openroad-ibex-signoff | 0.00 | upstream ref | reference `solve.sh` never applies the `config.mk` fixes it documents |

## Upstream image defects — the 🔁 REBUILD tasks

Baked into the prebuilt images, so a `solve.sh` workaround only greens the *oracle*; a
**real agent** hits them. `build_cache --stage patch` now writes the fix into the
**build context** (Dockerfile / harness), so §2's rebuild bakes it — persistent for
the oracle AND a real agent. Flow: `build_cache --stage patch` (bakes the fix) →
`build_plan_gen --all` → `deploy/registry/build_and_push_images.py` → `build_cache --stage
repin` (README §2). All are in `REBUILD_TASKS`.

| Image | Task(s) | Defect | Build-context fix (`build_cache`) |
|---|---|---|---|
| `zli12321/lhtb-duckdb-optimizer-closure:20260615` | duckdb | baked `duckdb_harness.py` builds `ninja -j{os.cpu_count()}` → `-j192` OOM | `fix_nproc_scaling_oracles`: `os.cpu_count()` → `len(os.sched_getaffinity(0))` in `environment/harness/` (baked) + `tests/` + cpuset pin |
| `zhongzhi660/lhtb-{climate-netcdf-extreme-event-audit,materials-phase-diagram-audit,robotics-slam-benchmark-repair}:20260709` | climate, materials, robotics-slam | no GNU `patch` → `patch < fix_audit.patch` "command not found" | `bake_patch_binary`: `RUN apt-get install -y patch` into `environment/Dockerfile` |
| `zli12321/lhtb-unknown-config-semantics:20260615` | unknown-config-semantics | stale image: old daemon `status` reply lacks `nonce` the current client reads | none needed — the shipped `environment/Dockerfile` is self-consistent; a plain rebuild bakes the fixed daemon |
| `chess-mate` `game` sidecar (`Dockerfile.game`) | chess-mate | image published nowhere (a `build:`-only sidecar), not a *defect* | none — `build_plan_gen --all` emits chess-mate's main + game as `type: local` entries in the unified plan; the sidecar namespace is derived from the repinned main ref, so it runs in the ordinary sweep — no `IMAGE_TEMPLATE` (§2) |

`apex-openroad-ibex-signoff` and `sudoku-recovery` are **not** image defects — the
first is an unfinished upstream reference (`solve.sh` never applies its documented
`config.mk` edits), the second an upstream reference/harness-model conflict (needs
root; harbor runs the oracle as `[agent] user`). Neither a rebuild nor an xrlenv
change helps.

## Reproduce

```bash
set -a; . ./.env; set +a
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example
python xrlenv_plugins/benchmarks/lhtb/build_cache.py --stage all --registry "$XRLENV_PRIVATE_REGISTRY_HOST"  # clone + LFS + fixes + repin the 6 REBUILD tasks (§2 path; --use-upstream-image instead for the out-of-box gate)
# the gate — DEFAULT §2 mode runs GREEN + TBD = 43 (drops only the 3 BLACKLIST); add
# --use-upstream-image for the docker.io gate (37: drops the 6 rebuilds too). Shouts the excluded set:
bash xrlenv_plugins/benchmarks/lhtb/run_full_sweep.sh --max-workers 8
# a full 46-task diagnostic sweep (includes the blacklisted tasks; heavy oracles run long):
nohup python xrlenv_plugins/benchmarks/lhtb/run_oracle_sweep.py \
    --max-workers 8 --timeout-multiplier 1.5 --jobs-dir ./tmp --job-id lhtb-sweep > tmp/lhtb.log 2>&1 &
```

Notes: harbor **schema 1.1 / `allow_internet`** (no version conflict). Images prebuilt
on **docker.io** (36 `zli12321/`, 10 `zhongzhi660/`) — resolved directly from each
task's `docker_image`, no registry-resolver change. Gate keys on the dense `reward`.

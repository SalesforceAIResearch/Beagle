# LHTB (Long-Horizon Terminal-Bench)

LHTB is a 46-task, harbor-format corpus of long-horizon terminal tasks
([zli12321/LHTB](https://github.com/zli12321/LHTB)) committed **in-git** under
`tasks/<name>/` with large assets (`*.zip`/`*.mp4`/`*.gif`) as git-lfs objects. Its
xrlenv shape is the **harbor golden path** — each task unpacks to `task.toml` +
`instruction.md` + `environment/Dockerfile` + `solution/solve.sh` + `tests/`, the same
contract tb2.1 / deep-swe use, so `xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster`
runs them with **no code change** — plus a distinctive **mixed image plan** (most tasks
pull a prebuilt public Docker Hub image; the 6 REBUILD tasks are built + pushed to our
private registry). Offline tasks are sealed by harbor 0.20's **native network policy**
(its `[environment] allow_internet = false` → `NO_NETWORK` shim, enforced by the cluster
env's `_apply_baseline_network_policy` via `apply_egress`), so an offline task's anti-cheat
probe passes with no sweep-side wrapper.

## What's here

| File / dir | Role |
| --- | --- |
| `build_cache.py` | populate (git clone + `git lfs pull`) → normalize `task.toml` → **task-level fixes** (curated overlays + programmatic) → repin the REBUILD tasks (§1) |
| `patches/` | curated **full-file overlays** applied by `--stage patch` (the pre-generated `2048` / `snake_maze_campaign` game references — see `patches/README.md`) |
| `build_plan_gen.py` | emit **the** plan (`--all`) — the single source of truth for where each task's image comes from: `type: local` build entries (the 6 rebuilds + compose sidecars) + `type: registry` for the ~40 prebuilt docker.io images |
| `lhtb_build_plan.yaml` | the generated **mixed** plan (gitignored — every `image_ref` bakes in your private-registry host; regenerate it with `--all` before use) |
| `deploy/registry/build_and_push_images.py` (repo-root `scripts/`) | build the plan's `type: local` entries + push to the `:5011` private registry; skip `type: registry` (§2) |
| `run_full_sweep.sh` | the CI gate: (re)build cache → compute the run set by 3-set exclusion → invoke the oracle sweep once |
| `run_oracle_sweep.py` | the oracle sweep on the xrlenv cluster (dense-reward gate; owns both retry layers) |
| `STATUS.md` | full 46-task disposition + upstream-image defects + every task-level fix |
| `tests/` | offline unit tests for the pure cache-build / fix / plan-gen logic |

```
zli12321/LHTB (git + LFS) ─(1)build_cache─▶ <cache>/lhtb/<task>/  (+ task-level fixes, repin)
                                                 │
              (2)build_plan_gen ─▶ pull the ~40 prebuilt images + build/push the 6 we build
                                                 │
                          (3)run_full_sweep ─▶ per-task dense reward on the cluster
```

## 1. Build the cache

`build_cache.py` materializes the corpus into the shared harbor cache (idempotent,
`--stage`-driven; `all` runs `populate` → `patch` → `repin`):

```bash
# XRLENV_BENCHMARK_CACHE (the shared root) is read from .env — see .env.example

# populate (git clone + git-lfs pull + task.toml normalize) + task-level fixes + repin.
# `--stage all` REFUSES to guess how to handle the 6 REBUILD tasks — pick one:
#   --registry <host>      repin them at your private registry (the full rebuild flow —
#                          then build+push them in §2). Recommended: their public images
#                          are broken/unpublished, so we build them ourselves.
#   --use-upstream-image   keep the docker.io refs (the out-of-box gate path; the 6 are
#                          excluded from that gate anyway)
# host: $XRLENV_PRIVATE_REGISTRY_HOST from .env (ephemeral per cluster; a bare host gets
# :5011 appended)
set -a; source .env; set +a
.venv/bin/python xrlenv_plugins/benchmarks/lhtb/build_cache.py \
    --stage all --registry "$XRLENV_PRIVATE_REGISTRY_HOST"
```

Stages (each idempotent): **`populate`** shallow-clones `zli12321/LHTB` + `git lfs pull`
(the large task assets — needs network + `git-lfs`), copies each `tasks/<name>/` into
`<cache>/lhtb/<name>/`, and normalizes its `task.toml`; **`patch`** applies the curated
`patches/` overlays and the programmatic task-level fixes below; **`repin`** re-points the
6 REBUILD tasks' `docker_image` at `<registry>/lhtb/<task>:main` (needs `--registry`; `all`
does this by default and requires `--registry` **or** `--use-upstream-image` — no silent
default). Tasks land at `<XRLENV_BENCHMARK_CACHE>/lhtb/<task>/`; point every xrlenv consumer's
`XRLENV_BENCHMARK_CACHE` at this shared root.

> The HF mirror `IntelligenceLab/Long-Horizon-Terminal-Bench` intentionally withholds
> `tests/` and `solution/`; the git repo is the only complete source (we need both for the
> verifier + the oracle sweep).

### Task-level cache fixes

Several LHTB tasks ship a **complete** reference that a naive run scores at 0 — because of
an *environment/harness* gap, not a partial reference. Every fix is idempotent, faithful
(installs/repairs the benchmark's own reference — no re-implementation, smallest overlay),
and logged in `STATUS.md`. They split by **where the gap lives**, which decides whether an
image rebuild is needed.

**Curated overlays (`patches/<name>/`)** — full-file replacements copied onto the task dir
by `apply_all_patches` (survive re-populate; exec bits preserved). Today these are the
pre-generated strong **game references** the public repo doesn't ship: `patches/2048/`
(band 6 = the 2048 tile, `reward≈0.60`) and `patches/snake_maze_campaign/` (43 foods →
band 8, `reward≈0.52`) — each produced offline on a dev worker because a *cheap* in-build
run only reaches a low band. `apply_all_patches` runs **before** the programmatic regen, and
each regen skips when a non-empty `reference_moves.log` already exists, so the committed
patch wins and the regen is a low-band fallback. These logs carry the benchmark's "never in
training corpora" canary — see `patches/README.md`.

**Programmatic fixes (in `build_cache.py --stage patch`)** — the task-dir gaps that green
the oracle **out-of-box, no rebuild**:

- `recover_files_based_oracles` — 7 `*-audit`/`*-regression` tasks ship a reference under
  `solution/files/` but **no `solve.sh`** → synthesize a files-install `solve.sh`
  (`cp -a solution/files/. /app/`).
- `regenerate_sokoban_reference` — `sokoban`'s moves ship as *data* in
  `gen/reference_solutions.json`; replay them (pure-Python, seconds) → `reward≈0.59`.
- `regenerate_2048_reference` / `regenerate_snake_maze_reference` — the low-band **fallbacks**
  behind the committed `patches/` references above (a bounded run when no patch is present).
- `raise_slow_verifier_timeouts` — `vector-db-iterative-build`'s HNSW-rebuild grader runs
  ~20 min → raise `[verifier] timeout_sec` 600 → 2400 → `reward=0.86`.

**IMPORTANT — some fixes are IMAGE-LEVEL, not just task files.** For the **6 REBUILD tasks**
the gap is baked into the *image* (Dockerfile / build harness), beyond the task dir's reach
at run time — a `solve.sh` workaround would hide it for the oracle while a **real agent**
still lands in the broken image. So `build_cache` writes these into the task's **build
context**, and they take effect only once the image is **rebuilt + pushed in §2**:

- `fix_nproc_scaling_oracles` (`duckdb-optimizer-closure`) — the big-node `nproc` trap:
  `ninja -j{os.cpu_count()}` = `-j192` OOMs in the 8 GiB cap → rewrite to
  `len(os.sched_getaffinity(0))` in the baked build harness (`environment/harness/`) + the
  uploaded verifier, and mark `XRLENV_CPU_PINNING="1"` so the affinity mask is the declared
  `ceil(cpus)` cores. The rewritten harness only bites in the **rebuilt image**.
- `bake_patch_binary` (3 `zhongzhi660` audits — `climate`, `materials`, `robotics-slam`) —
  the image ships no GNU `patch`, so the task's `patch < fix_audit.patch` fails "command not
  found" → add `RUN apt-get install -y patch` to `environment/Dockerfile`.
- Two more round out `REBUILD_TASKS` with **no `--stage patch` edit**:
  `unknown-config-semantics` (its shipped Dockerfile is self-consistent; only the *published*
  docker.io image is stale — a plain rebuild fixes it) and `chess-mate`'s `game` compose
  sidecar (published nowhere — built, not fixed).

With `--registry`, repin re-points each REBUILD task's `docker_image` to
`<host>/lhtb/<task>:main`, so in §2 they become `type: local` **build** entries. Full
per-task disposition: [`STATUS.md`](STATUS.md).

## 2. Prepare the images

The plan is **mixed** — one file, the single source of truth, typed by each task's own
repinned `docker_image`.

**THIN part (most tasks).** ~40 tasks ship a prebuilt public Docker Hub image
(`docker_image = zli12321/lhtb-<task>:<date>`, a few `zhongzhi660/`), read from `task.toml`
→ `type: registry`. Nothing is built for them: they pull on acquire through the **`:5010`
docker.io pull-through mirror** (same path `xrlenv build apply` warms). Warming is optional
→ §4.

**BUILD part (the 6 REBUILD tasks).** `chess-mate`'s `game` compose sidecar (published
nowhere) + the baked-defect images (duckdb's `-j{os.cpu_count()}`, the `patch`-less audits,
`unknown-config`'s stale daemon) are `type: local` build entries (+ one per compose sidecar)
**built and pushed to the `:5011` PRIVATE registry** by `deploy/registry/build_and_push_images.py`.
§1's repin already pointed their `docker_image` at `<host>:5011/lhtb/<task>:main`; those refs
**404 until you build + push them here**. The `:5011` registry has **no Docker-Hub fallback**,
and the **prod-colocated registry is off-limits**.

If you ran §1 with `--use-upstream-image`, the 6 REBUILD tasks stay on their (broken)
docker.io refs and are excluded from the out-of-box gate — **skip to §3.** For the full /
real-agent path (§1 `--registry <host>`), run the command sequence:

```bash
# XRLENV_BENCHMARK_CACHE (the cache §1 wrote) is read from .env — see .env.example
export XRLENV_PRIVATE_REGISTRY_HOST=<private-registry-host>           # :5011 = the private registry

# (a) repin the 6 REBUILD tasks so the plan types them as type: local (idempotent).
.venv/bin/python xrlenv_plugins/benchmarks/lhtb/build_cache.py \
    --stage all --registry "$XRLENV_PRIVATE_REGISTRY_HOST"

# (b) generate THE plan (mixed): 6 type: local (built) + ~40 type: registry (docker.io).
#     Gitignored (host-specific); (re)generate before use and whenever the repinned cache changes.
.venv/bin/python -m xrlenv_plugins.benchmarks.lhtb.build_plan_gen \
    --all --output xrlenv_plugins/benchmarks/lhtb/lhtb_build_plan.yaml

# (c) build the type: local entries + push to :5011; type: registry entries are skipped
#     (served via the :5010 mirror). Run on a BUILD HOST (docker daemon + the private
#     registry in insecure-registries — the login/CP box has neither; use a worker node).
#     Idempotent: HEADs each manifest, skips if present (--force rebuilds).
.venv/bin/python deploy/registry/build_and_push_images.py \
    --plan xrlenv_plugins/benchmarks/lhtb/lhtb_build_plan.yaml \
    --registry "$XRLENV_PRIVATE_REGISTRY_HOST:5011"
```

(Warming the images onto the nodes — `xrlenv build apply` — is the optional §4 step.)

## 3. Run the oracle sweep (validate the cache)

The corpus-quality gate: run harbor's oracle per task **on the xrlenv cluster** and confirm
the shipped reference produces a gradable dense reward. Under the oracle a non-passing task
is a plumbing/content bug, not a model signal. LHTB grades with a **dense float** (partial
credit), so the gate keys on the canonical **`reward` key** of the rewards dict (`reward > 0`
= the reference produced a gradable result), falling back to `max()` over the dict **only if
a verifier writes no `reward` key** — never a blind `max()`, which would grab a raw diagnostic
metric (chess-mate writes `reference_white_moves=120` alongside `reward=0.9`). Some tasks
legitimately max **below 1.0** (band-graded games, inherent dense metrics — root-caused in
`STATUS.md`, none a grading bug). The oracle here is harbor's default **`OracleAgent`**; an
offline task (e.g. `commit0-multilib-tdd`) is sealed by harbor 0.20's native network policy
before grading, so its anti-cheat probe passes.

```bash
set -a; source ./.env; set +a   # XRLENV_GRPC_HOST + XRLENV_CONSUMER_TOKEN + registry from .env
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example

# THE GATE — run_full_sweep.sh. Two paths, one flag apart; both apply the §1 task-cache
# fixes and both run the same TBD set. They differ ONLY in the 6 REBUILD tasks (image-level).
# (a) DEFAULT — §2 path: the 6 REBUILD tasks run on their rebuilt :5011 images (needs the
#     §2 build+push done). Runs 43 = GREEN (31) + TBD (12).
bash xrlenv_plugins/benchmarks/lhtb/run_full_sweep.sh --max-workers 32 --jobs-dir ./tmp --job-id lhtb-s2

# (b) --use-upstream-image — raw docker.io images, no rebuild: the 6 REBUILD tasks drop out.
#     Runs 37 = GREEN (27) + TBD (10; climate + materials are TBD *and* need a rebuild).
bash xrlenv_plugins/benchmarks/lhtb/run_full_sweep.sh --use-upstream-image --max-workers 32 --jobs-dir ./tmp --job-id lhtb-s1

# --skip-ultra-long drops the >30 min oracles (unknown-config-semantics ~88 min,
# nbody-accel-iterative ~45 min) for faster iteration; --list-green prints the run set + exits.
bash xrlenv_plugins/benchmarks/lhtb/run_full_sweep.sh --max-workers 32 --skip-ultra-long --jobs-dir ./tmp --job-id lhtb-quick

# --- targeted subsets go through run_oracle_sweep.py directly ---
.venv/bin/python xrlenv_plugins/benchmarks/lhtb/run_oracle_sweep.py --tasks 2048,sokoban   # smoke
```

**Two retry layers, both in `run_oracle_sweep.py`.** `--retries` (default 6) retries only
**infra-transient** errors (`CapacityExhausted` / `ControlPlaneLost` / `NodeLost` /
`NodeCommandTimeout`) — a genuinely-failed task is never re-rolled. `--content-retries`
(the wrapper passes 1) re-runs only the tasks that came back non-passing (`reward=0` /
missing on the `reward` key) up to N more times; a task is solved if **any** attempt rewards
`> 0`, catching nondeterministic `reward=0` flakes. `run_full_sweep.sh` is a thin wrapper:
it (re)builds the cache mode-aware, computes the run set, and invokes the sweep **once**,
trusting its exit code (0 iff every task solved after its content-retries). Timeouts use each
task's **native budget** (`timeout_multiplier` defaults to 1.0). Per-trial artifacts
(`agent/oracle.txt`, `verifier/reward.txt`, `result.json`) land under `<jobs-dir>/<job-id>/`;
each content-retry round writes a **sibling** `<job-id>-retryN/` dir (a task passing in ANY
round counts), so a retried task's artifacts live under the `-retryN` sibling, not the base job.

**The 3-set exclusion** (every one of the 46 tasks is in exactly one set; `run_full_sweep.sh`
SHOUTS the excluded set every run — full disposition in `STATUS.md`):

- **GREEN** — expected `reward > 0` under the oracle. In the §2 path this includes the 6
  REBUILD tasks (validated on their rebuilt images); with `--use-upstream-image` they drop
  out (broken/unpublished on docker.io), so GREEN is 27 not 31.
- **TBD** — the 12 [issue-#2](https://github.com/zli12321/LHTB/issues/2)
  grade-against-a-private-reference tasks (output schema / magic constants not shipped, in
  several cases the grader's embedded reference is byte-identical to `solution/`). They
  **run** (the oracle produces a result) but are **tracked apart** — a passing oracle there
  is a tautology, so it validates nothing about solvability.
  `gdal-proj-raster-regression` is the issue's named **fair counterexample** → GREEN, not
  TBD.
- **BLACKLIST — never run** (upstream defects; may still be fixable): `super-mario` (game
  reference log not shipped; regen needs a torch+net gen image), `sudoku-recovery` (upstream
  reference needs root; harbor runs the oracle as `[agent]`),
  `apex-openroad-ibex-signoff` (upstream `solve.sh` never applies its documented `config.mk`
  fixes).

`run_full_sweep.sh` flags: `--use-upstream-image`, `--max-workers`, `--job-id`, `--jobs-dir`,
`--content-retries`, `--skip-build-cache`, `--skip-ultra-long`, `--list-green`; anything else
forwards to `run_oracle_sweep.py` (e.g. `--timeout-multiplier` / `--cpus-multiplier` /
`--memory-multiplier`).

## 4. Warm the image and Calibrate the image size (optional)

Warming pulls every plan entry onto the nodes ahead of first acquire, and calibration
replaces the plan's conservative `size_hint_bytes` placeholders with true on-disk sizes,
sharpening the capacity estimator's FFD packing. **Optional** — the heuristic is safe and the
sweep runs fine without it (the §3 sweep also materializes images on first acquire).

```bash
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example
export XRLENV_OPERATOR_TOKEN=<operator-token>

# (a) warm — apply the SAME unified plan; warms BOTH the registry pulls (docker.io via the
#     :5010 mirror) AND the 6 pushed-local images (from :5011).
.venv/bin/python -m xrlenv_plugins.benchmarks.lhtb.build_plan_gen \
    --all --output xrlenv_plugins/benchmarks/lhtb/lhtb_build_plan.yaml
xrlenv build apply --plan xrlenv_plugins/benchmarks/lhtb/lhtb_build_plan.yaml \
    --fill-missing --connect-host <control-plane-host>

# (b) calibrate — overwrite the size hints with cluster-reported on-disk sizes (a SEPARATE
#     file so you diff before promoting). Warm first — calibrate measures materialized images.
xrlenv build calibrate \
    --plan xrlenv_plugins/benchmarks/lhtb/lhtb_build_plan.yaml \
    --output xrlenv_plugins/benchmarks/lhtb/lhtb_build_plan.yaml.calibrated.yaml \
    --connect-host <control-plane-host>
```

> `--all` reads each task's current `docker_image`. On the full path the 6 REBUILD refs
> point at your private registry, so warm them only **after §2** built + pushed them (else
> they 404); on the out-of-box path they're docker.io refs — always safe.

## See also

- `xrlenv_plugins/benchmarks/GUIDELINE_onboard_benchmarks.md` — the onboarding design
  convention; §3 (golden-path file contracts), §4 (workflow + gates), §5 (image & registry
  mechanics — the `:5010` mirror vs `:5011` private registry, `calibrate`), §8 (multi-service
  / **compose** sidecars, the shape chess-mate's `game` service uses).
- `xrlenv_plugins/harbor/README.md` + `xrlenv_plugins/harbor/compose.py` — the harbor cluster
  plug-in that runs these tasks (compose sidecar naming/namespacing shared build-time ↔
  run-time).
- `docs/supported_benchmarks_and_harnesses/harbor_framework.md` (the `LHTB` section) — the
  harbor framework Sphinx page.
- [`STATUS.md`](STATUS.md) — the current 46-task disposition (green / TBD / blacklist),
  per-task fixes, the sub-1.0 reward root-cause table, and the reproduce commands.

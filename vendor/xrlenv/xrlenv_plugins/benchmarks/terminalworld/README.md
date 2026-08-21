# TerminalWorld (verified split)

Onboards the **`verified`** split of
[`EuniAI/TerminalWorld`](https://huggingface.co/datasets/EuniAI/TerminalWorld)
(200 human-reviewed harbor-shape tasks) onto the xrlenv cluster. The xrlenv shape
is the **harbor golden path** (`build_cache → build_plan_gen → run_oracle_sweep`,
reusing `xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster` with no adapter and
no manifest) **plus** the two extra axes this benchmark is the reference for:
**multi-service docker-compose** stacks and **sysbox DinD** privileged runners.
TerminalWorld tasks do **not** ship prebuilt images — each ships an
`environment/Dockerfile` that is built locally from the cache shard on shared FSx
and pushed to the private (`:5011`) registry, then resolved by namespace.

## What's here

| File | Role |
|---|---|
| `build_cache.py` | populate (HF download + `task.toml` normalize) + patch + opt-in sysbox markers into the shared cache |
| `build_plan_gen.py` | emit the `type: local` image build plan (one entry per task **plus** one per compose sub-service build context) |
| `scripts/build_and_push_images.py` *(shared)* | build each `type: local` Dockerfile once + push to the `:5011` PRIVATE registry |
| `patches/` | curated per-task full-file overlays for broken upstream packaging (see `patches/README.md`) |
| `run_full_sweep.sh` | the entrypoint / CI gate: (re)build cache → compute green set → invoke the oracle sweep |
| `run_oracle_sweep.py` | oracle sweep on the xrlenv cluster (PASS/FAIL correctness gate; owns both retry layers) |
| `tw_build_plan.yaml` | the generated build plan (regeneratable — `type: local` entries embed absolute cache paths, so it's rebuilt on demand) |
| `STATUS.md` | point-in-time per-task disposition (passed / flaky / failed / blocking) + reproduce command |
| `tests/` | OFFLINE unit tests for the pure normalize / sysbox-marker / build-plan logic |

```
EuniAI/TerminalWorld (verified, HF) ─(1)build_cache─▶ <cache>/terminalworld-verified/<id>/
                                                           │
              (2)build_plan_gen + build_and_push_images ─▶ <registry>:5011/terminalworld-verified/<id>:main
                                                           │        (+ one <id>-<service>:main per compose sub-build)
                                       (3)run_oracle_sweep ─▶ per-task PASS/FAIL on the cluster
```

One name does double duty: **`terminalworld-verified`** is both the cache shard
(`<cache>/terminalworld-verified/<id>/`) and the image namespace
(`<registry>:5011/terminalworld-verified/<id>:main`). Multi-service compose tasks
add one **sidecar** entry per sub-directory build context
(`<id>-<service>:main`), carrying an `xrlenv.compose_service` label.

## 1. Build the cache

```bash
export XRLENV_BENCHMARK_CACHE=/path/to/benchmark-cache   # shared root

# the full setup: populate (HF download + normalize) + patch + sysbox markers
.venv/bin/python xrlenv_plugins/benchmarks/terminalworld/build_cache.py --stage all
```

`--stage` (each idempotent; `all` runs them in order `populate` → `patch` →
`sysbox`, sysbox last so a `patch` overlay can't clobber a marker):

- **`populate`** — pull the `verified` task list from `EuniAI/TerminalWorld` on
  the Hub, download only those `artifacts/<id>.tar.gz`, extract each into
  `<cache>/terminalworld-verified/<id>/`, and **normalize** its `task.toml`.
  Needs network (`datasets` / `huggingface_hub`).
- **`patch`** — overlay the curated `patches/<id>/` full-file fixes (+ the
  `CPU_PINNING_TASKS` / `COMPOSE_DROP_PRIVILEGED` in-cache task.toml/compose
  edits). No network — safe on any cluster; this is the **runc-only** entry point.
- **`sysbox`** — write the `SYSBOX_TASKS` routing markers (see §1 Task-level cache
  fixes). A marked task hard-fails on a cluster with no sysbox node
  (`BackendCapabilityMissing`), so a **runc-only** cache should use `--stage
  patch`, not `all`.
- **`all`** — the default; all three, for a full run on a sysbox-capable cluster.

**`task.toml` normalization is required, not cosmetic.** TerminalWorld's
generation pipeline emits BOTH harbor's deprecated `memory` / `storage` strings
and the canonical `_mb` integers, and ~half the verified set sets them
inconsistently (e.g. `memory = "2G"` alongside `memory_mb = 4096`). harbor's
`EnvironmentConfig` rejects that conflict outright, so **97 of the 200 verified
tasks won't load at all** — `populate` drops the deprecated duplicate wherever the
canonical `_mb` field is present (a surgical, line-level edit that preserves the
rest of the file). The logic is a pure, unit-tested function
(`normalize_task_toml_text`).

### Task-level cache fixes

`patches/<task_id>/` carries curated **full-file overlays** for tasks whose
upstream packaging is broken — a deliberately-partial reference `solve.sh`, a
missing verifier user, a non-hermetic dependency pin. Concrete examples:
`tw_245733`'s shipped `solve.sh` pulls `ubuntu:latest` but never writes the
`/app/result.txt` the verifier reads; `tw_655577`'s `task.toml` gets a
`[verifier] user = "root"` because the image bakes `USER delicate` but `test.sh`
needs root for `apt`/`uv`. Overlays are applied by `--stage patch` *after*
extraction + normalization, so they survive re-populate. **Faithfulness:** each
overlay is the smallest change that lifts the oracle's reward ceiling to 1 (or
restores loadability), and every override is logged per task (`[patch] <id>:
overrode […]`). Full per-patch table + rationale: `patches/README.md`.

The **DinD / sysbox markers** are task-level *routing*, not content edits — they
are written by `--stage sysbox` into a marked task's `task.toml`
(`[environment.env] XRLENV_CONTAINER_RUNTIME="sysbox-runc"`, plus companions
`XRLENV_INNER_DOCKERD` / `XRLENV_INSTALL_DOCKERD` / `XRLENV_DOCKERD_LEGACY_STORE`
/ `XRLENV_SYSTEMD_INIT` and `agent_user`/`verifier_user`). The curated
`SYSBOX_TASKS` set lives in `build_cache.py`, grown one proven task at a time; the
decisive probe is `tw_245733` (its `solve.sh` is `docker pull ubuntu:latest`, its
image ships a docker daemon, single-service, non-privileged):

```bash
.venv/bin/python xrlenv_plugins/benchmarks/terminalworld/build_cache.py --stage sysbox --tasks tw_245733
.venv/bin/python xrlenv_plugins/benchmarks/terminalworld/run_oracle_sweep.py --tasks tw_245733
```

Routing is **case-by-case, via `task.toml`** — never a global default (an
accidental cluster-wide sysbox switch would hard-fail every task on a cluster with
no sysbox node). For a marked task the harbor cluster plug-in threads
`container_runtime="sysbox-runc"` into `acquire_container` (the control-plane
`KwargsPolicy` gates it and the scheduler pins the acquire to a sysbox-capable
node) and, for a DinD task (`XRLENV_INNER_DOCKERD="1"`), brings up a nested
`dockerd` and waits for its socket before `solve.sh` runs — the faithful
substrate. We never edit `solve.sh`; we only bring the daemon up the way the VM's
init would.

**Compose sidecars are IMAGE-level**, not task-level cache fixes — the extra
sub-service build contexts are emitted by `build_plan_gen.py` and built in
[§2](#2-prepare-the-images).

## 2. Prepare the images

TerminalWorld tasks have **no registry image** — each ships an
`environment/Dockerfile` that must be built. `build_plan_gen.py --all` emits one
`context_source: {type: local}` entry per task, pointing at that task's
`environment/` dir in the cache shard on shared FSx (built **in place** — no
clone, no tarball, no copy). `type: local` requires `shared_fs` (default
`hyperpod`): the assertion that every build node mounts the same FSx path. Image
refs are `terminalworld-verified/<task_id>:main`, and the private-registry host
prefixes every ref at push time (`<host>:5011/terminalworld-verified/<id>:main`)
— **exactly** the ref `run_oracle_sweep.py` constructs as the
`xrlenv_image_template` per-run kwarg, so the image the build pushes is the image
the eval acquires.

```bash
# pre-req: the :5011 private registry is up

# generate the plan (type: local, from the populated cache shard):
.venv/bin/python -m xrlenv_plugins.benchmarks.terminalworld.build_plan_gen \
    --all --output ./xrlenv_plugins/benchmarks/terminalworld/tw_build_plan.yaml

# build once + push to the :5011 PRIVATE registry (idempotent — HEADs each
# manifest and skips a present ref). Run on a build host (or ssh to a worker):
export XRLENV_PRIVATE_REGISTRY_HOST=node-host
.venv/bin/python scripts/build_and_push_images.py \
    --plan ./xrlenv_plugins/benchmarks/terminalworld/tw_build_plan.yaml \
    --registry "$XRLENV_PRIVATE_REGISTRY_HOST:5011"
```

The `:5011` registry is the **PRIVATE** writable one: refs are addressed directly
and a miss has **no Docker-Hub fallback** (do not confuse it with the `:5010`
pull-through mirror). Never point the build/push at the prod-colocated registry
pair — that is off-limits.

**Multi-service compose tasks** (e.g. `tw_188260`'s `solr-node` / `ambari-server`)
additionally emit **one `type: local` entry per sub-directory build context**,
named `<id>-<service>:main` with an `xrlenv.compose_service` label. Naming is
delegated to `compose.subdir_build_services` + `default_image_refs` (not
hand-rolled), and the sidecar registry **namespace is derived from the repinned
main ref** via `compose.registry_namespace_and_tag` — the *same* helper at build
time and run time. A compose task uses the SAME per-run `xrlenv_image_template` as
any other task (it repins the *main* ref); its sidecars derive their refs from that
repinned main ref with **no separate per-sidecar override** (a sidecar-specific
override would wrongly repin the main image too).
Services that build from `.` reuse the task's canonical `<id>` image;
`image:`-only sidecars (`postgres:14`) are pulled, not built.

See the `build_plan_gen.py` module docstring for the `--tasks` subset invocation
and the `type: local` / `shared_fs` details.

## 3. Run the oracle sweep (validate the cache)

The corpus-quality gate: run harbor's OracleAgent per task **on the xrlenv
cluster** and confirm each earns a positive reward. Under the oracle a non-passing
task is a plumbing/content bug (its reward ceiling is 0 → poison for RL), not a
model signal. The pass gate is **all rewards `> 0`** (harbor's `_trial_passes`).

```bash
export XRLENV_BENCHMARK_CACHE=/path/to/benchmark-cache

# control plane (required) + private registry (for image resolution)
export XRLENV_GRPC_HOST=<control-plane-host>       # the CP host (see slurm_scripts/clusters.yaml)
export XRLENV_GRPC_PORT=50051
export XRLENV_CONSUMER_TOKEN=<token>
export XRLENV_PRIVATE_REGISTRY_HOST=<private-registry-host>   # (:_PORT defaults to 5011)

# THE FULL GATE — run_full_sweep.sh is the entrypoint. It (re)builds the cache,
# computes the green set (present tasks − EXCLUDE, asserts 192), runs it, and
# CONTENT-RETRIES any reward-0 flake. Prefer it over a raw run over all 200 (that
# would run the known-failing EXCLUDE tasks too AND skip the retry reads).
bash xrlenv_plugins/benchmarks/terminalworld/run_full_sweep.sh \
    --max-workers 32 --jobs-dir ./tmp --job-id tw-oracle-sweep__192

# print the green set (present − EXCLUDE) and exit — no run, no cache rebuild:
bash xrlenv_plugins/benchmarks/terminalworld/run_full_sweep.sh --list-green --skip-build-cache

# --- targeted subsets go through run_oracle_sweep.py directly ---
# smoke (2 tasks):
.venv/bin/python xrlenv_plugins/benchmarks/terminalworld/run_oracle_sweep.py --tasks tw_100459,tw_101703
# the multi-service compose tasks:
.venv/bin/python xrlenv_plugins/benchmarks/terminalworld/run_oracle_sweep.py \
  --tasks tw_522753,tw_299387,tw_188260,tw_304270,tw_304271,tw_305044 \
  --max-workers 6 --jobs-dir ./tmp --job-id tw-compose-validate
```

**The two retry layers (both live in `run_oracle_sweep.py`, so every driver — the
wrapper AND the ci runner — gets them from one place):**

- `--retries` (infra-transient only, default 6) — retries **only** the four infra
  exceptions `{CapacityExhausted, ControlPlaneLost, NodeLost, NodeCommandTimeout}`
  via harbor's `RetryConfig(include_exceptions=…)`. **The final stats record one result
  per task** — a retried task that then passes counts once, never double-counted. A
  content outcome is never re-rolled. In the common case the infra failure is a fail-fast
  **acquire** (before `solve.sh`); a **post-acquire** infra error (e.g.
  `NodeCommandTimeout` on an exec) re-runs the whole attempt in a **fresh container**, so
  `solve.sh` can *execute* more than once — this only matters for **external** side
  effects, not the recorded result. This is how xrlenv absorbs high concurrency against a
  capacity-capped runtime — see the compose/sysbox pacing below.
- `--content-retries` (default 2, passed by `run_full_sweep.sh`) — per-**task**:
  after a run, re-run ONLY the non-passing tasks (by `_trial_passes`) up to N more
  times; a task is solved if ANY attempt passes. Catches a nondeterministic
  reward-0 flake (a transient DNS / verifier blip) that `--retries` deliberately
  never re-rolls. Both layers report their counts, so a task that only passes on a
  re-run surfaces as *flaky*, not silently greened.

**Timeouts run at native budget** (no `--timeout-multiplier` in the gate) — a task
whose own reference solution can't fit its own `timeout_sec` should fail loud, not
be rescued by inflated headroom (this is why `tw_528959` is excluded). Exit code
is `0` only if every oracle solved, so the sweep is CI-usable. Per-trial artifacts
(`agent/oracle.txt`, `verifier/reward.txt`, `trial.log`) land under
`--jobs-dir/<job-id>/`; each content-retry round writes a **sibling** `<job-id>-retryN/` dir,
so a retried task's artifacts live there, not under the base job.

**Compose / sysbox specifics (the §8 axes):**

- **Concurrency is unbounded from the caller's side** — request any
  `--max-workers` (8, 32, …). xrlenv paces the capacity-capped sysbox runtime: the
  per-node sysbox cap (`SYSBOX_MAX_CONCURRENT`, default ~4) is enforced
  server-side, and the harbor adapter fails an at-cap acquire **fast** with
  `CapacityExhausted` (before harbor's 360 s setup window;
  `XRLENV_HARBOR_ACQUIRE_QUEUE_TIMEOUT_S`, default 240 s) so the trial queue
  retries it until a slot frees. More sysbox nodes raise steady-state throughput
  but are not required for correctness.
- **sysbox creates *and* destroys are serialized** (`raw_sysbox_create_concurrency=1`,
  `raw_sysbox_destroy_concurrency=1`) — concurrent sysbox-fs mounts/unmounts wedge
  under load and leak containers.
- **Multi-service compose tasks run under runc** (compose and sysbox are mutually
  exclusive); the whole-stack footprint is reserved via `place(reserve=…)` at
  runtime and every sidecar is capped so the reservation is enforced (an uncapped
  sidecar can OOM the node). A pinned static subnet makes the stack node-exclusive.

> **Image resolution.** Because TW tasks have no prebuilt `docker_image`, the
> sweep composes `<registry>/terminalworld-verified/{task_id}:main` from
> `--registry` / `$XRLENV_PRIVATE_REGISTRY_HOST[:_PORT]` and passes it as the
> `xrlenv_image_template` kwarg via `EnvironmentConfig` (a scoped per-run handoff,
> not a process-global env var). Compose sidecars derive their namespace from the
> repinned main ref, so the template covers them with no extra flag.

> **AS OF 2026-07-17: green set = 192 of 200.** `run_full_sweep.sh` EXCLUDEs the 8
> non-green: the 6 substrate/broken-oracle **Failed** tasks (incl. `tw_488034`),
> `tw_222108` (netns-DNS, deep investigation), and `tw_528959` (excluded
> 2026-07-08 — its own `timeout_sec=2700` can't fit a CPython from-source
> `make -j2` build even uncontended). Full per-task disposition + the EXCLUDE
> rationale: [`STATUS.md`](STATUS.md).

## 4. Warm the image and Calibrate the image size (optional)

Both steps are **optional**: the cluster's dynamic image cache
(lazy-pull-on-acquire + LRU eviction + image-affinity) means the sweep works with
no pre-warm — you warm only to amortize the first-acquire pull across a big run.

```bash
# warm the pushed images onto the cluster (--fill-missing tops up any absent ref):
xrlenv build apply \
    --plan ./xrlenv_plugins/benchmarks/terminalworld/tw_build_plan.yaml \
    --fill-missing --connect-host <control-plane-host>

# calibrate the true on-disk sizes AFTER the first warm:
source .venv/bin/activate
export XRLENV_OPERATOR_TOKEN=<operator token>
xrlenv build calibrate \
    --plan ./xrlenv_plugins/benchmarks/terminalworld/tw_build_plan.yaml \
    --output ./xrlenv_plugins/benchmarks/terminalworld/tw_build_plan.yaml.calibrated.yaml \
    --connect-host <control-plane-host>
```

The plan ships a conservative `size_hint_source: heuristic` — a local Dockerfile
build **can't be probed before it runs**, so the plan can only guess. `calibrate`
queries each node for the actual on-disk (uncompressed) size, takes the max per
`image_ref`, and writes a **separate** `*.calibrated.yaml` (so you diff before
promoting) whose measured sizes feed the FFD bin-packer for tighter placement on
the next apply. At runtime a multi-service compose task additionally reserves its
whole-stack footprint via `place(reserve=…)`, so accurate sizes matter most for
those.

## See also

- `../GUIDELINE_onboard_benchmarks.md` — the onboarding convention; **§8** is the
  multi-service compose + sysbox/DinD axis this benchmark is the reference for.
- `xrlenv_plugins/harbor/README.md` + `xrlenv_plugins/harbor/compose.py` — the
  harbor cluster plug-in that runs these tasks and the compose helpers
  (`subdir_build_services`, `default_image_refs`, `registry_namespace_and_tag`).
- `docs/supported_benchmarks_and_harnesses/harbor_framework.md` — the
  `TerminalWorld` section of the harbor framework Sphinx page.
- [`STATUS.md`](STATUS.md) — current per-task disposition (passed / flaky / failed
  / blocking) + the EXCLUDE rationale + reproduce command.
- Sibling shards / patterns: `seta` (build-from-Dockerfile git-source sibling),
  `lhtb` (mixed plan + its own compose task), `terminal_bench_2_1`
  (prebuilt-image sibling).

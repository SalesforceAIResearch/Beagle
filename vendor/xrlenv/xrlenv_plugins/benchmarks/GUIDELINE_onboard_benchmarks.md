# Onboarding a new benchmark to xrlenv — design convention

> **Audience:** an engineer or agent onboarding a new agentic benchmark onto the
> xrlenv cluster. Read this **before** you create any files. It captures the
> convention distilled from every benchmark already under
> `xrlenv_plugins/benchmarks/` (deep_swe, lhtb, seta, terminal_bench_2_1,
> terminalworld, evoclaw, webarena-infinity) so a new onboarding lands on the
> same rails and gives users one consistent experience.
>
> This file is **stable convention**. Point-in-time results live in each
> benchmark's `STATUS.md`; the mechanics of the platform live in the specs
> (`specs/14`, `15`, `21`) and `CLAUDE.md`. If those ever disagree with this
> file, they win — tell us so we can fix the guideline.

---

## 0. The three laws (read `CLAUDE.md` first, then these)

Everything below is downstream of three project invariants. If a decision ever
seems to conflict with one of these, **stop and surface it** — it usually means
xrlenv core is missing a hook, not that your benchmark needs a workaround.

1. **xrlenv only manages containers and images.** The benchmark's own harness —
   its parser, grader, agent loop, report generator — runs **unmodified**. You
   integrate at the container/image boundary, never by editing benchmark logic.

2. **Don't reinvent benchmark-side wheels.** Delegate parsing/grading/reporting
   to upstream's published API or filesystem contract. If upstream writes
   `reward.json` at `/logs/verifier/`, honor that path **byte-for-byte** so
   upstream's own aggregator consumes your output with no translation. If you
   find you *can't* delegate, that's a signal to surface, not to work around.

3. **Mechanism, not policy.** xrlenv core ships primitives (`task_key`,
   `group_id`, fleet reservation, `cancel_group`, `RuntimeLimits.cpu_pinning`,
   `image_pin_mode`). Benchmark-specific logic (image redirection, retry loops,
   content fixes) lives **in the benchmark's plug-in directory**, never as a
   change to xrlenv core.

A corollary that runs through every benchmark here: **a task the oracle can't
solve is poison for RL** (its reward ceiling is 0). So the correctness gate for
onboarding is an **oracle sweep** — run each task's shipped reference solution
and confirm it earns positive reward. Under the oracle, a non-passing task is a
plumbing/content bug, not a model signal.

---

## 1. Mental model: how a benchmark actually attaches to xrlenv

Two things are called "adapter" in this codebase. Keep them apart:

- **Framework cluster environment (spec 14 "case-3")** — the mechanism *every*
  onboarded benchmark uses today. It is a subclass of the **upstream harness's**
  own environment class (harbor's / pier's `DockerEnvironment`) that reroutes
  container ops onto the xrlenv cluster. These live in
  `xrlenv_plugins/harbor/` and `xrlenv_plugins/pier/` and are **shared** — you
  usually reuse one with zero code change.

- **`EnvAdapter` (spec 14 "case-1", `xrlenv/envs/base.py`)** — an in-sandbox
  `setup/step/teardown` protocol for **RL-training step loops** (gym-shaped
  `reset/step`). Only the abstract base + `SyncEnvAdapter` exist today; **no
  onboarded benchmark uses this path.** Ignore it unless you are specifically
  onboarding a gym-style step-loop environment.

**Crucial consequence — there is no `manifest.yaml`, no template registration.**
The onboarded benchmarks are *not* xrlenv templates and xrlenv core is
deliberately ignorant of them. The wiring is entirely on the **upstream
harness's** config: harbor/pier expose an `import_path` escape hatch (the same
one a harbor user uses to pick `e2b`/`modal`/`daytona`), and the benchmark's
sweep script points it at the xrlenv cluster environment:

```python
# in run_oracle_sweep.py — this one line is the whole wiring
ENV_IMPORT_PATH = "xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster"   # or pier
EnvironmentConfig(import_path=ENV_IMPORT_PATH, ...)
```

xrlenv is then reached only through the gRPC `Client` the cluster environment
lazy-builds from `XRLENV_GRPC_HOST` / `XRLENV_GRPC_PORT` / `XRLENV_CONSUMER_TOKEN`
/ `XRLENV_GRPC_SECURE`.

> **Note on `xrlenv_plugins/README.md`:** it describes an *aspirational*
> `manifest.yaml` + `adapter.py` per-benchmark layout. That is **not** how the
> real onboarded benchmarks work — they reuse a shared framework adapter and
> carry an ops kit instead. Follow *this* guideline; the plug-ins README
> describes the case-1 template path, which is unused today.

---

## 2. Decision tree — which integration shape is yours?

Answer these top-to-bottom. The first match is your path.

```
Q1. Does each task unpack to a self-contained container the platform can drive
    from OUTSIDE — i.e. an image + a solve.sh-style command + a reward file at
    /logs/verifier/ that the platform reads back?  (the "harbor filesystem contract")
    │
    ├─ YES, and its native harness is harbor (harbor==0.8.0 compatible)
    │        → GOLDEN PATH · reuse  xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster
    │          (examples: lhtb, seta, terminal_bench_2_1)   →  §3, §4
    │
    ├─ YES, and its native harness is pier / it needs a separate grading
    │        container (environment_mode="separate") or filtered egress
    │        → GOLDEN PATH · reuse  xrlenv_plugins.pier:XrlenvPierEnvironmentCluster
    │          (example: deep_swe)                           →  §3, §4
    │
    └─ NO ↓

Q2. Does its harness hardcode a container driver with NO pluggable seam — e.g.
    it shells out to the `docker` CLI via subprocess, so import_path can't slot in?
    │
    └─ YES → INTERCEPTOR pattern: monkeypatch the subprocess/docker boundary
             in-process and reroute through the xrlenv docker-py-compat client.
             (example: evoclaw)                              →  §7.1

Q3. Is the unit of work NOT host-gradable — the whole agent+verify loop must run
    INSIDE the container because there's no host-reachable service (e.g. a
    port-less browser/web env)?
    │
    └─ YES → RUNNER-SHIM pattern: a host runner-wrapper that reuses upstream's
             own runner + an in-container runner that does agent+verify on
             localhost. (example: webarena-infinity)         →  §7.2

Q4. Is the harness a DIFFERENT RL framework with its own BaseEnvironment Protocol
    that neither harbor nor pier's DockerEnvironment satisfies?
    │
    └─ YES → Write a NEW framework cluster adapter under xrlenv_plugins/<framework>/
             (pier itself is the worked example — a harbor fork that needed one).
             Follow xrlenv_plugins/harbor/README.md.          →  §7.3

Q5. Is it a gym-style RL-training step loop (reset/step, in-sandbox reward)?
    └─ YES → spec-14 case-1 EnvAdapter. Not yet exercised — design-first with
             the maintainers before writing.
```

**Bias hard toward Q1.** Most agentic benchmarks either already are harbor/pier-shape
or can be *made* so cheaply, and the golden path is the only one that gives you
the full ops kit (cache, build plan, oracle gate, calibrate) for free. The
outlier paths (§7) exist because a benchmark's coupling genuinely couldn't be
expressed through `import_path`; reach for them only after Q1 fails.

---

## 3. The golden path — directory layout & file contracts

A golden-path benchmark is a **self-contained ops kit** under
`xrlenv_plugins/benchmarks/<name>/`. It ships **no adapter and no manifest** — it
reuses the shared harbor/pier cluster environment and leans only on generic core
(`xrlenv build apply` / `xrlenv build calibrate`).

```
xrlenv_plugins/benchmarks/<name>/
├── build_cache.py           # populate the task corpus into the shared cache (+ normalize + patch)
├── build_plan_gen.py        # emit the image build/warm plan YAML from the populated corpus
├── <name>_build_plan.yaml   # the committed build/warm plan (regeneratable)
├── <name>_build_plan.calibrated.yaml   # optional: post-warm true on-disk sizes
├── run_oracle_sweep.py      # the correctness gate — OracleAgent per task on the cluster; owns both retry layers (--retries infra + --content-retries per-task)
├── run_full_sweep.sh        # one-command entrypoint (thin): build cache → green set → invoke run_oracle_sweep --content-retries
├── patches/                 # optional curated per-task content fixes (see §6)
├── tests/                   # OFFLINE unit tests for the four files above
├── README.md                # stable how-to + design rationale
└── STATUS.md                # point-in-time results + reproduce command
```

**Naming:** use an **importable** dir name (underscores, e.g. `deep_swe`,
`terminal_bench_2_1`, `webarena_infinity`) so
`python -m xrlenv_plugins.benchmarks.<name>.build_plan_gen` works — this holds for
outliers too (webarena-infinity was renamed hyphen→underscore for consistency; its
call-site scripts still ship under `copy_to_call_site/` and are invoked by path from
the original repo). `xrlenv_plugins/benchmarks/` is a PEP-420 namespace package, so no
`__init__.py` is required.

**The shard constant does double duty.** Pick one string (`"deep-swe"`,
`"lhtb"`, `"seta-env"`, `"terminal-bench-2-1"`) that is *both* the cache
sub-directory (`$XRLENV_BENCHMARK_CACHE/<shard>/<task_id>/`) *and* the namespace
every consumer enumerates. Define it once as a module constant.

### 3.1 `build_cache.py` — materialize the corpus

**Role:** download the upstream dataset and land each task under
`<cache_root>/<shard>/<task_id>/`, normalizing each `task.toml` and applying any
curated patch overlays.

**Contract:**
- CLI: `--stage {all,populate,patch}` (each **idempotent**; `all` = populate →
  patch), `--dest <cache root>` (defaults to `$XRLENV_BENCHMARK_CACHE`), plus a
  `--source {git,hf}` where the dataset has both.
- **Idempotent:** a task already present is skipped; re-runs are safe. A dir is a
  valid task iff it carries the anchor file (`task.toml`).
- **`task.toml` normalization is a pure, unit-tested function.** harbor/pier
  reject a task that sets both the deprecated `memory`/`storage` strings *and*
  the canonical `memory_mb`/`storage_mb` — strip the deprecated duplicate where
  the canonical is present. Keep this a byte-preserving line-level edit.
- **Fail loud** on unexpected upstream layout — never silently produce an empty
  or half-populated shard.
- Env overrides for dataset identity (repo URL / ref / HF repo / shard name) so
  a fork or mirror can be swapped without a code edit.
- Network-dependent `populate` is **not** unit-tested (it's covered by a live
  `--stage populate`); the normalize/patch logic **is**.

### 3.2 `build_plan_gen.py` — emit the image plan

**Role:** produce a `BuildPlan` YAML (`entries: [...]`), one entry per task.

**The load-bearing invariant:** read each entry's `image_ref` **from the task's
own `[environment] docker_image` in `task.toml` — never synthesize it.** A task
with no `docker_image` **fails loud**. (A synthesized ref drifts from the ref the
cluster resolves at acquire time — this has bitten before.)

**Contract (stable SEAMS, not a byte-identical CLI).** The generators share the same *shape*
but their selection flags + committed filenames differ per benchmark — do not assume one CLI
covers all; check the benchmark's own `--help`:
- A **full-corpus selector** and a **subset selector**, plus `--output <path|->` (default
  stdout); MOST also have a **size-probe toggle**. The spellings vary per benchmark — check
  `--help`: swebench_verified uses `--all | --instances`; **seta** uses
  `--remote | --starter | --range | --tasks` (its full-corpus selector is `--remote`, **not**
  `--all`); deep_swe / lhtb / terminal_bench_2_1 / terminalworld use `--all | --tasks`. The
  probe toggle is `--no-probe` on most (probe on by default) and `--probe` on deep_swe (probe
  off by default) — but **seta and terminalworld have no probe flag at all** (seta always
  builds from git; terminalworld sizes locally).
- Emit the canonical YAML shape (see §5.1). Pick the `context_source.type` per §5.2
  (`registry` / `git` / `local` / `tarball`).
- The committed plan is the full-corpus output checked into git — regeneratable at any time
  (the full-corpus selector is `--all` for most, `--remote` for seta). Its filename is
  per-benchmark (e.g. `swebench_verified_build_plan.yaml`, tb2.1's `build_plan_89_full.yaml`,
  terminalworld's `tw_build_plan.yaml`), so consult the README.

### 3.3 `run_oracle_sweep.py` — the correctness gate

**Role:** run the harness's default **OracleAgent** (applies each task's shipped
`solution/` reference and commits) for every task **on the xrlenv cluster**, and
report which tasks the oracle solves.

**Contract:**
- Sets `EnvironmentConfig(import_path="xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster")`
  (or pier). Requires `XRLENV_GRPC_HOST` (fail loud if unset).
- CLI: `--tasks <csv>` (default all), `--max-workers N`, `--jobs-dir`, `--job-id`,
  `--retries N` (**infra-transient only**), `--content-retries N` (**per-task
  content-retry** — see below), resource-ablation knobs (`--cpus-multiplier`,
  `--memory-multiplier`, `--override-cpus`, `--override-memory-mb`, `--cpu-pinning`,
  `--timeout-multiplier`).
- **BOTH retry layers live HERE (not in the shell wrapper), so every driver —
  `run_full_sweep.sh` AND the `xrlenv_plugins/benchmarks/tests/integration/` ci runner — gets them from one
  place, no re-implementation:**
  - `--retries` — **infra-transient only**, per **task attempt** (one full acquire → solve →
    verify, each in a **fresh container**). The retry set is exactly
    `{CapacityExhausted, ControlPlaneLost, NodeLost, NodeCommandTimeout}` (swebench_verified
    also includes `PinCapacityExhausted`), wired via the harness
    `RetryConfig(include_exceptions=...)`. **The final stats record ONE result per task** — a
    retried task that then passes counts once, never double-counted (the harness retries a
    trial internally and emits its final outcome). A genuinely-failed task (a content/reward
    outcome) is never re-rolled. In the common case the infra failure is a fail-fast **acquire**
    (before `solve.sh` runs). A **post-acquire** infra error (e.g. `NodeCommandTimeout` on an
    exec) does re-run the whole attempt in a NEW container, so `solve.sh` can *execute* more than
    once — this only matters for **external** side effects, not the recorded result. This lets
    the caller request any concurrency while xrlenv paces the capacity cap via fail-fast acquire.
  - `--content-retries` — **per-task content-retry.** After a run, re-run ONLY the
    non-passing tasks (by this benchmark's own `_trial_passes` gate) up to N more times;
    a task is solved if ANY attempt passes. Catches a nondeterministic reward=0 flake (a
    transient DNS / verifier blip) that `--retries` deliberately never re-rolls. Default
    0; the drivers pass 2.
- **Exit 0 iff every oracle solved** (CI-usable). Harbor/Pier per-trial artifacts (`agent/`,
  `verifier/reward.json|reward.txt`, logs) land under `<jobs-dir>/<job-id>/`; a `--content-retries`
  re-run writes a SIBLING `<jobs-dir>/<job-id>-retryN/` dir (the integration runner globs
  `<job-id>*` so a pass in any round counts). swebench_verified writes a single
  `<jobs-dir>/<job-id>/summary.json` (per-instance `resolved`), not per-task trial dirs — so
  don't assume one artifact layout across benchmarks.
- **The pass gate is a per-benchmark seam** — see §6. deep_swe keys on the
  `reward` key only (`float(rewards["reward"]) > 0`); tb2.1/seta require *all*
  rewards `> 0`; lhtb keys on the `reward` key when present (`> 0`, partial-credit) and falls
  back to `max()` of the rewards dict ONLY when no `reward` key is written. Pick the rule that
  matches your reward semantics, call the benchmark's actual `_trial_passes` (don't restate the
  rule in prose), and unit-test it.

### 3.4 `run_full_sweep.sh` — the entrypoint / CI gate

**Role: a THIN wrapper — it owns cache + green-set, and DELEGATES the run + both
retries to `run_oracle_sweep.py`.** It (1) sources `./.env` for CP host + token,
(2) (re)builds the cache, (3) computes the run set as **present tasks minus an
EXCLUDE list** (exclusion, not inclusion — so a re-populate auto-picks up new
tasks), and (4) invokes `run_oracle_sweep.py --tasks <green> --content-retries N`
**once**, trusting its exit code. Under `set -e`, a persistent failure aborts the
wrapper with `run_oracle_sweep`'s own exit code + its `X/N solved` + failed-list
summary; the wrapper only prints its GREEN line on success. It does **not**
re-implement the retry loop (a past version did — the bash `_failed_tasks` reader
duplicated the `.py` pass gate and drifted; keep the loop in the `.py`).

**The two-retry-layer design (get this right — it's what keeps the gate honest).**
BOTH layers live inside `run_oracle_sweep.py`; the wrapper (and the ci runner) just
pass the flags:

| Layer | Granularity | Retries on | Purpose |
|---|---|---|---|
| `--retries` (default 6) | per-**trial**, in `run_oracle_sweep.py` | the 4 infra exceptions only | absorb capacity pacing; **cannot** mask a flaky task |
| `--content-retries` (default 2) | per-**task**, in `run_oracle_sweep.py` | a reward-0 *outcome* (re-runs ONLY the non-passing tasks) | catch a one-off environmental flake that surfaced as reward-0 rather than a typed exception |

A `--content-retries` re-run is **visible** as a `<job-id>-retryN` sibling artifact dir, and
STATUS.md records which tasks only passed on a re-run — so a flaky task is surfaced, not
silently greened. (Note: the summary JSON reports the pass/resolve outcome, not a separate
retry-count field — read the per-round `<job-id>*` dirs for the retry history.) Pass
`--content-retries 0` for a zero-tolerance gate. **Timeouts run at native budget**
(no `--timeout-multiplier` in the gate) — a task whose own reference solution
can't fit its own `timeout_sec` should fail loud, not be rescued by inflated
headroom.

**A uniform FLAG interface — run knobs are flags, never env vars.** A stale
exported `SKIP_BUILD=1` / `LIST_GREEN=1` must never silently turn a real sweep
into a no-op, so every run knob is a flag: `--max-workers N`, `--content-retries N`,
`--job-id LABEL`, `--jobs-dir DIR`, `--skip-build-cache`; any unrecognized arg
passes through to `run_oracle_sweep.py`. Only **deployment** config
(`XRLENV_BENCHMARK_CACHE`, and the `.env` CP host / token / registry) comes from the
environment — never a per-run knob.

**`--list-green` is required.** The script MUST honor `--list-green` by printing
its green set (`present − EXCLUDE`), one task id per line, then exiting — running
nothing. This is the seam the integration runner
(`xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py`) uses to sample `k` tasks for CI **without
re-implementing the exclusion logic** (the sweep stays the single source of
truth). It is always paired with `--skip-build-cache` so a listing never triggers
a slow cache rebuild. See `xrlenv_plugins/benchmarks/tests/integration/README.md`.

### 3.5 `README.md` vs `STATUS.md`

- **`README.md` — stable how-to + rationale.** Audience: operators + future
  onboarders. **Every benchmark README uses the same canonical skeleton** so an
  operator who has read one can drive them all. Follow these headings *exactly*
  (all six onboarded benchmarks already do — read `swebench_verified` for the
  thinnest and `lhtb` / `terminalworld` for the fullest):

  ```
  # <Benchmark name>
  <1–3 sentence intro: what it is, upstream source + corpus size, its xrlenv
   shape (harbor / pier / drop-in), and any one distinctive trait>

  ## What's here
  <a file-role table for the ops kit + a one-line ASCII pipeline
   build_cache → build_plan_gen → run_oracle_sweep>

  ## 1. Build the cache
  <the build_cache.py --stage all command, the <cache>/<shard>/<id>/ layout,
   what lands per task, idempotency, XRLENV_BENCHMARK_CACHE>

  ### Task-level cache fixes        # ONLY if the benchmark has patches/ or
  <the curated overlays + programmatic content fixes, WHY each exists, how they
   apply (--stage patch), the faithfulness rule (smallest overlay, logged in
   STATUS). If a fix is IMAGE-level (needs an image rebuild), say so and
   forward-reference §2 — it is built/pushed there, not just written to the cache>

  ## 2. Prepare the images
  # THIN branch — upstream ships prebuilt images (registry): nothing is built;
  #   the cluster pulls each on first acquire. Keep it short and point to §4.
  #   (swebench_verified, deep_swe, terminal_bench_2_1)
  # BUILD branch — upstream ships NO image: build_plan_gen emits type: git
  #   (built node-side from the upstream Dockerfile — seta) or type: local
  #   (built from the shard Dockerfile + pushed to the :5011 PRIVATE registry via
  #   deploy/registry/build_and_push_images.py — terminalworld). Give the build+push
  #   command sequence. Any IMAGE-level fix carried from §1 is (re)built + pushed
  #   here. A benchmark may MIX (lhtb: most registry, 6 local rebuilds).

  ## 3. Run the oracle sweep (validate the cache)
  <run_full_sweep.sh as the correctness gate (exit 0 iff every oracle solves);
   the two-retry-layer design (--retries infra-only set; --content-retries
   per-task reward-0 re-run); this benchmark's pass-gate rule; native-budget
   timeouts; --list-green / --skip-build-cache>

  ## 4. Warm the image and Calibrate the image size (optional)
  <xrlenv build apply --plan ... --fill-missing to warm (amortize the
   first-acquire pull); WHY optional (lazy pull-on-acquire + LRU);
   xrlenv build calibrate → <plan>.calibrated.yaml (true on-disk uncompressed
   size for tighter FFD packing, ≠ the registry compressed size)>

  ## See also
  <GUIDELINE, the shared harbor/pier README, the Sphinx page, STATUS.md>
  ```

  Keep the design-rationale (the two-retry-layer design, the provenance/fetch-path
  decision, any separate-verifier / native-network-policy / compose / sysbox seam)
  *inside* the relevant numbered section rather than as loose prose.
- **`STATUS.md` — point-in-time disposition.** Audience: "is it green right now,
  which tasks, and why." Contents: current gate config (concurrency, timeout
  mode, retry layers); a results bucket table (Passed/Failed/Total); the exact
  reproduce command; run metadata (job id, wall-clock, `n_errored_trials` /
  `n_retries` from `result.json`); and per-task failure dispositions.

Use `docs/supported_benchmarks_and_harnesses/<name>.md` for the **Sphinx user
page** (invoke the `sphinx-docs-writer` agent) — that's the external audience;
README/STATUS are operator-facing.

### 3.6 `tests/` — offline only

Ship `tests/{test_build_cache.py, test_build_plan_gen.py, test_run_oracle_sweep.py}`.
**Every test must be network-free and cluster-free** — exercise only the pure /
filesystem logic against a synthetic shard:

- `test_build_cache.py` — `task.toml` normalization (strips deprecated key iff
  canonical present; no-op otherwise), copy/idempotency, patch application.
- `test_build_plan_gen.py` — repo:tag splitting (a `host:port` prefix is **not**
  a tag), task discovery, `image_ref` read-from-toml, and the **fail-loud path**
  (`pytest.raises(SystemExit)` when a task has no `docker_image`).
- `test_run_oracle_sweep.py` — the **pass-gate logic** with fake trial results:
  pass when reward > 0 even if side metrics are 0; fail on reward-0 / exception /
  missing reward key; task resolution (all / subset / `SystemExit` on unknown).

Run `.venv/bin/python -m pytest xrlenv_plugins/benchmarks/<name>/tests -q` and
invoke the `qa-test-engineer` agent once the four files stabilize.

---

## 4. The onboarding workflow (command sequence + gates)

The canonical sequence, all from the repo root:

```bash
# 0) the shared cache root (one shared root across all benchmarks) —
#    XRLENV_BENCHMARK_CACHE is read from .env, see .env.example

# 1) build the cache: download + normalize [+ patch]
.venv/bin/python xrlenv_plugins/benchmarks/<name>/build_cache.py --stage all

# 2) generate + warm the image plan (OPTIONAL for prebuilt-image benchmarks — see below)
.venv/bin/python -m xrlenv_plugins.benchmarks.<name>.build_plan_gen \
    --all --output xrlenv_plugins/benchmarks/<name>/<name>_build_plan.yaml
# --connect-host is REQUIRED (it dials the control plane); without it the CLI exits 2.
xrlenv build apply --plan .../<name>_build_plan.yaml --fill-missing --connect-host <cp-host>

# 3) THE GATE — oracle sweep (correctness)
set -a; source ./.env; set +a                # XRLENV_GRPC_HOST + XRLENV_CONSUMER_TOKEN
bash xrlenv_plugins/benchmarks/<name>/run_full_sweep.sh             # or ... --max-workers 32

# 4) (OPTIONAL) calibrate image sizes after the first warm
xrlenv build calibrate --plan .../<name>_build_plan.yaml \
    --output .../<name>_build_plan.calibrated.yaml --connect-host <cp-host>
```

**Step 2.** It is optional if the benchmark alrady has a pre-built image. The cluster's dynamic image cache (lazy-pull-on-acquire +
LRU eviction + image-affinity) means you can skip pre-warming entirely —
deep_swe's 113/113 run had no pre-warm even at conc-32. Pre-warm only to
amortize the first-acquire pull across a big run. If the benchmark does not have a pre-built image (or you need to rebuild the images for some reason), you should have the build plan to build the image from the Dockerfile
and push it to the private registry. See  seta, lhtb, terminalworld for examples.

**The two gates:**

- **G1 — correctness (oracle sweep).** Every non-excluded task passes under its
  own reference solution at native timeout. This is the bar for "onboarded". Run
  it at low concurrency first (conc-4/8) to prove content, then again at high
  concurrency (conc-32+) to prove the platform carries it.
- **G2 — scale (live sweep).** Only after G1: a real multi-node run at target
  concurrency confirms no infra regressions (node-loss, capacity collapse). This
  is where compose/sysbox/fleet interactions surface (§8).

A SWE-style oracle+verifier trial exceeds a foreground shell/tool timeout — run
the full sweep under `nohup`/background and poll.

---

## 5. Image & registry mechanics

### 5.1 Build-plan YAML shape

```yaml
version: 1
name: <benchmark>-<N>-task           # operator label (excluded from plan_id hash)
replication: 1
budget: {reserved_runtime_gb: 30, buffer_gb: 10}
entries:
- image_ref: <the exact ref consumers pass to acquire_container(image=...)>
  context_source: {type: registry}   # or git / local / tarball — see §5.2
  placement:
    preferred_home_count: 1
    size_hint_bytes: <int>           # feeds FFD bin-packing across nodes
    size_hint_source: heuristic      # heuristic | registry-probe | cluster-reported
  pinned: false                      # true = never evict
  priority: 0                        # higher = built first under budget pressure
  labels: {}                         # extra docker labels (e.g. xrlenv.compose_service)
```

`plan_id` is a SHA-256 over the canonicalized plan (drops `name`); re-applying an
unchanged YAML is a no-op. **`calibrate` changes the `plan_id`** (it rewrites
`size_hint_bytes`), so the next apply re-dispatches every entry — cheap, because
the docker layer cache keeps wall-clock low.

### 5.2 The four `context_source` types — pick by where the image comes from

| type | When | What the cluster does | Example |
|---|---|---|---|
| `registry` | task ships a **prebuilt** image ref | node-side `docker pull` (warm only, nothing built) | deep_swe (public ECR), tb2.1 (Docker Hub) |
| `git` | task ships only a **Dockerfile in a git repo** | build node-side from `{repo, ref, subdir, dockerfile}` | seta |
| `local` | built from a **shard Dockerfile on shared FS**, pushed to the private registry | `deploy/registry/build_and_push_images.py` builds once + pushes to `:5011`; needs `shared_fs` | terminalworld, lhtb's 6 rebuild tasks |
| `tarball` | small self-contained build context | context bytes travel in the plan (`content_b64`) | rare |

A single benchmark can **mix** types (lhtb: most `registry`, 6 `local`).

### 5.3 The two registries — do not confuse them

- **`:5010` — pull-through MIRROR of docker.io.** Configured as a docker
  `registry-mirror`; applies only to `docker.io` pulls, image refs unchanged, and
  **falls back to Docker Hub if down**. This is what makes `type: registry`
  prebuilt images cheap to pull cluster-wide (dedup + Docker-Hub-rate-limit
  relief). Warming **does** populate it. Filled by `deploy/registry/warm_images.py` /
  `xrlenv build apply`.
- **`:5011` — PRIVATE writable registry.** Addressed **directly** by explicit
  refs (`<host>:5011/<repo>:<tag>`); a miss has **no Docker-Hub fallback**. This
  is where **we** push images we build (`type: local`/`git`). Filled by
  `deploy/registry/build_and_push_images.py`.

> **Prod colocation is off-limits.** Do not point a new benchmark's build/push at
> the prod-colocated registry pair. If your images live on a public registry
> (public ECR is anonymous-pullable, no punishing rate limit), pull them
> **directly** — no new infra needed (deep_swe's decision). Only stand up a
> pull-through mirror for Docker-Hub-rate-limited or egress-controlled cases.

### 5.3.1 Registry host: resolve at RUN TIME, never bake it into the cache ⚠

**The private + mirror registries are colocated with the prod control plane on one
node. Every time the CP is lost and replaced, that node's IP changes.** So any
absolute `<host>:<port>/<repo>` image ref that has been **frozen** into a persisted
artifact — a cached `task.toml`, a committed build plan, a compose file — silently
**drifts**: after the next CP replacement it points at a dead host, and the task
fails at acquire with `RegistryResolveError: registry unreachable`. This is a latent
time bomb, not a build bug — it lies dormant until the next CP swap.

**Rule: a benchmark MUST resolve the private-registry host at run time from `.env`
(`XRLENV_PRIVATE_REGISTRY_HOST` / `_PORT`) and MUST NOT persist it.**

- ✅ **Good — `seta` (immune by design).** `run_oracle_sweep.py::_registry_from_env`
  reads `XRLENV_PRIVATE_REGISTRY_HOST` at launch, builds
  `{registry}/seta-env/{task_id}:main`, and passes it as a per-run
  `xrlenv_image_template` kwarg (`EnvironmentConfig(kwargs=...)`). Its cached
  `task.toml` bake **no** registry host. The harbor plugin's `_resolve_image_ref`
  applies the template. Change the CP → update `.env` → next run just works.
- ✅ **`lhtb` (was the drift, now fixed the same way).** It's a *mixed* corpus (~40
  docker.io prebuilts + 6 private-registry rebuilds), so a single job-level template
  would clobber the docker.io tasks. Instead, `build_cache.py --stage repin` writes a
  **host-agnostic placeholder** into the 6 rebuild tasks' `docker_image`:
  `${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}/lhtb/<task>:main`.
  The harbor plugin's `_resolve_image_ref` (and `build_plan_gen`) `os.path.expandvars`
  it from `.env` at acquire / plan-gen — so the cache bakes **no** host. (Before this,
  `repin --registry <host>` baked an absolute host and a CP move
  `<old-registry-host>` → `<new-registry-host>` stranded the rebuild tasks on the dead registry.)

**The rule:** never persist a registry host. Use a per-run template (`seta`) or a
host-agnostic `${XRLENV_PRIVATE_REGISTRY_HOST}:${…PORT}/…` placeholder that the plugin
expands from `.env` (`lhtb`). A quick check for a new benchmark:
`grep -rE '(ip-[0-9-]+|[0-9.]+):5011/' <benchmark>/` should be empty (only the
placeholder is allowed).

### 5.4 `image_pin_mode` (spec 14/15, per-plan/template)

Three modes; default is right for almost everything:

- `registry_digest` **(default)** — a mutable tag `host:5011/repo:tag` is
  resolved to a content-addressed `repo@sha256:...` per acquire, so nodes
  materialize exactly the bytes the tag serves *now* (solves mutable-tag
  staleness — a rebuilt tag doesn't propagate to already-cached nodes otherwise;
  the escape hatch is `xrlenv images evict`).
- `per_node_local` — for images built per-host (no central digest to pin).
- `shared_storage` — reserved for phase-2 NFS/S3 mounts.

### 5.5 `calibrate` — why and when

The plan ships a conservative `size_hint_source: heuristic`. After the first
warm, `xrlenv build calibrate` queries each node for the **actual on-disk size**,
takes the max per `image_ref`, and writes `<plan>.calibrated.yaml` with
`size_hint_bytes` = measured and `size_hint_source: cluster-reported`. Those true
sizes feed the FFD bin-packer for tighter placement on the next apply. It writes
a **separate** file so you diff before promoting. (Calibrate measures on-disk
uncompressed size — *not* the registry's compressed size; the two differ.)

---

## 6. Customization seams — what varies per benchmark

The skeleton is invariant; these are the knobs each benchmark tuned. Expect to
touch a subset:

| Seam | What it controls | Examples |
|---|---|---|
| **Image plan type** | §5.2 — prebuilt vs build-from-Dockerfile vs private-push | deep_swe all-`registry`; seta all-`git`; lhtb mixed |
| **Populate source** | `git` clone vs `hf` snapshot vs registry export | deep_swe `--source {git,hf}` |
| **Pass gate rule** | which rewards make a task "pass" | deep_swe / lhtb key on the `reward` key (`> 0`; lhtb is partial-credit, falls back to `max()` only if a verifier writes no `reward` key); tb2.1 / seta require all rewards `> 0` (single-key in practice) |
| **Oracle agent** | default vs a subclass with extra setup | tb2.1 adds `--cpu-pinning`; offline-egress is sealed by harbor 0.20's native network policy (no oracle subclass needed) |
| **Task exclusion** | how the green set is bounded | deep_swe `EXCLUDE=()`; seta `black_list.txt`; lhtb 3 sets (GREEN/TBD/BLACKLIST) |
| **Patches** | curated per-task content fixes | tb2.1 dep-pins for non-hermetic oracles; lhtb rebuild overlays |
| **Image resolution override** | private-registry image template | seta sweep injects `xrlenv_image_template` kwarg via `EnvironmentConfig` (**no subclass needed**) |
| **Compose sidecars** | multi-service image entries | lhtb chess-mate, terminalworld — see §8 |

**Faithfulness rule for content fixes.** Every fix to broken benchmark content is
an **auditable overlay or exclusion in the benchmark dir**, never a change to
xrlenv core and never an unlogged edit:

- **`patches/<task_id>/<relative_path>`** — a curated *full-file* overlay applied
  by `build_cache.py --stage patch` (survives re-populate). Keep each the
  *smallest* diff that lifts the reward ceiling to passing — "complete the
  partial, don't re-author the task." A dep that drifted → pin it back to the era
  version, changing only that line. Record the line-delta so drift stays visible.
- **Exclusions** (`black_list.txt` / an `EXCLUDE` list / a TBD set) — for tasks
  upstream genuinely can't build or that need content xrlenv shouldn't fabricate.
  Log *why* each is excluded in `STATUS.md`.

Prefer redirecting an image ref (via the `xrlenv_image_template` kwarg injected by
the sweep driver, or the task's own `docker_image`) over subclassing the adapter —
this precedence mechanism exists exactly so a private-registry benchmark needs no
code (seta is the reference).

---

## 7. Outlier patterns (when Q1 fails)

Reach for these only when the harbor filesystem contract genuinely doesn't fit.
All three keep the same non-negotiable: **the benchmark's own harness runs
unmodified; integration happens at the container/image boundary.**

> **Outliers keep the canonical *shape* but not the golden-path sweep *contract*.**
> They are normalized to the same directory phases (README/STATUS with a "Canonical
> phases" section, `tests/`), with the executable half split by **where it runs**:
>
> ```
> <name>/
> ├── README.md · STATUS.md # canonical docs; "Canonical phases" maps each phase to a file and
> │                         #   documents any N/A phase (do NOT stub it)
> ├── (host-side build files, if any, stay top-level — e.g. wai's Dockerfile + build_plan.yaml,
> │    evoclaw's go-zero-gitfix.Dockerfile: the images xrlenv itself builds+pushes)
> └── copy_to_call_site/    # the payload copied/rsync'd into the ORIGINAL benchmark repo (and,
>     ├── run_full_sweep.sh #   for wai, injected into the container). `run_full_sweep.sh` lives
>     │                     #   HERE and runs the sweep FROM the call site (the sweep runs there,
>     │                     #   so the entrypoint does too); host-side prep (stage / build+push)
>     │                     #   is a documented PREREQUISITE, not inside it.
>     └── <driver>.py       #   the driver KEEPS its adapted name (run_all_xrlenv.py /
>                           #   run_eval_parallel_xrlenv.py): upstream's own entry point adapted
>                           #   by xrlenv, not a golden-path run_oracle_sweep.py.
> ```
>
> **Only phases that exist get files; a genuinely-absent phase is documented in the
> README, never stubbed** (evoclaw has no `build_plan.yaml` — the image plan is resolved
> at run time by the in-process `image_resolution.py`; wai has no per-task `build_cache.py`
> — the Dockerfile is the cache). Conversely, an image xrlenv *does* build stays top-level
> (evoclaw's `go-zero-gitfix.Dockerfile`, wai's `Dockerfile`) — that's a real host-side
> phase, separate from the call-site `run_full_sweep.sh`.
>
> Outliers are still **NOT** wired into the `xrlenv_plugins/benchmarks/tests/integration/`
> runner, and the §3.3 `--retries` / `--content-retries` layers do NOT apply — each
> outlier owns its own retry in its driver (evoclaw retries transient cluster loss +
> falls back to upstream's eval-retry; webarena-infinity relies on the upstream runner it
> reuses). If you later make an outlier golden-path-shaped, adopt §3.3's
> `run_oracle_sweep.py` to get both retry layers for free.

### 7.1 Interceptor (evoclaw) — harness hardcodes the `docker` CLI

The harness drives Docker through raw `subprocess.run(["docker", ...])`, so
`import_path` has no seam. Create the seam by monkeypatching the subprocess
boundary **in-process**:

- `docker_shim.install()` patches `subprocess.run` / `subprocess.Popen`; a call
  whose argv[0] basename is `docker` is rerouted to the xrlenv docker-py-compat
  client (`run→_run`, `exec→_exec`, `cp→_cp`, `rm/stop/...`); everything else
  falls through untouched.
- **Fail loud on uncovered surface** (`build`/`rmi`/`pull`/`tag` → rc 127) rather
  than silently dropping a capability. Track container names in an in-process
  registry (prefix `xrl-<pid>-` to avoid cross-rollout collisions). Emulate `-v`
  binds via `put_archive`/`get_archive`.
- Cross-process workers (`spawn`/`forkserver`) don't inherit the patch — re-apply
  it at interpreter startup via a `sitecustomize.py` on a `PYTHONPATH`-prepended
  bootstrap dir, gated by an env flag.
- **Faithfulness:** all correctness fixes are **runtime monkeypatches**, default
  **OFF**, so a stock run is byte-identical to upstream and leaderboard-comparable
  (a loud banner warns when fixes are on). Image redirection reads upstream's own
  source of truth (its `pull_images.sh` map), not a hardcoded table.

### 7.2 Runner-shim (webarena-infinity) — no host-reachable service

The whole agent+verify loop must run *inside* the container (port-less browser
env; the app server is on the container's `localhost`). There is **no
subprocess/docker interception** here:

- A **host runner-wrapper** (`run_eval_parallel_xrlenv.py`) mirrors upstream's own
  runner's CLI + output layout **byte-for-byte** and *reuses upstream's modules*
  (task loading, report generation, result merging) so the two paths can't drift.
  It acquires a container per worker via the xrlenv session API, injects the eval
  code once, and drives per-task via `session.exec` / `put_archive` / `get_archive`.
- An **in-container runner** (`xrlenv_runner.py`) runs the agent + verifier against
  `localhost` inside the container.
- A single **config module** (`xrlenv_config.py`) is the only place that reads
  `XRLENV_*` + LLM keys — config can't drift between files.
- **Answer isolation:** build an **answer-free substrate image** (strip all
  verifiers/solvers at image build; fail the build if any answer survives), and
  inject the verifier + answer only **after** the agent exits, deleting it before
  the next task. Use `xrlenv.group_id` for dashboard grouping and a **unique**
  `task_key` per container (a shared key caps a node at `max_runs_per_task`).

### 7.3 New framework cluster adapter — a different RL harness

If the harness is a different framework with its own `BaseEnvironment` Protocol,
add `xrlenv_plugins/<framework>/` subclassing that framework's **concrete**
environment class, overriding only the routing seams (`start`/`stop`/`exec`/
upload/download/`capabilities`). Pop xrlenv kwargs before `super().__init__()`.
Delegate grading to upstream's filesystem contract (`/logs/verifier/reward.json`)
— advertise `capabilities.mounted=False` and `download_dir` the verifier dir back
so upstream's aggregator reads reward exactly where it expects. Pin the framework
version exactly (`harbor==0.8.0`, `datacurve-pier==0.3.0`) — a floating harness
version silently drifts a frozen eval, so a bump is a deliberate, reviewed edit +
oracle-sweep re-run. Follow `xrlenv_plugins/harbor/README.md`.

---

## 8. Extra axis — multi-service / compose / DinD

If your tasks need more than one container, or the task's own `solve.sh` runs
`docker`/`systemctl`/`ip netns`, these apply **on top of** the golden path.
Reference: terminalworld (`build_plan_gen.py`, `build_cache.py`, `patches/`) and
the shared helpers in `xrlenv_plugins/harbor/compose.py`.

**docker-compose (multiple containers, one private network):**
1. Emit **one `type: local` entry per sub-dir build context** (or per
   root-context service with a custom `dockerfile:`), named `<id>-<service>:tag`,
   with an `xrlenv.compose_service` label. Delegate naming to
   `compose.subdir_build_services` + `default_image_refs` — don't hand-roll it.
2. Derive the sidecar registry namespace from the **repinned main ref** via
   `compose.registry_namespace_and_tag` (**same helper at build time and run
   time**) so the task runs in the ordinary sweep **without**
   a sweep-level `xrlenv_image_template` kwarg (which would wrongly override the
   main image too).
3. Guard build contexts with `is_safe_relative_context` (reject `..`/absolute
   escapes; fail loud on a missing service Dockerfile).
4. Runtime routes multi-service tasks (`is_multi_service`, >1 service) onto the
   compose-project path; the whole-stack **footprint** is reserved via
   `place(reserve=…)` and every sidecar is **capped** so the reservation is
   enforced (an uncapped sidecar can OOM the node).
5. A pinned static subnet makes the task **node-exclusive** (anti-affinity);
   prefer service-DNS-only where possible.
6. Compose is vetted CP-side against the *same* `KwargsPolicy` as single-acquire
   (reject-don't-strip). Avoid host bind-mounts (gated by `allowed_host_paths`).
7. **Delete any prior in-container sidecar-bootstrap workaround** once the real
   compose path works — the faithful `solve.sh` is the unpatched one (the
   `tw_299387` lesson: once the substrate can carry compose, the in-container
   hack becomes unfaithful).

**DinD / privileged (`solve.sh` runs docker/systemctl/netns):**
1. Route **case-by-case via `task.toml`**, never a global default (a cluster-wide
   switch hard-fails every task on a cluster with no sysbox node,
   `BackendCapabilityMissing`). Mark tasks with `build_cache.py --stage sysbox`.
2. Set `XRLENV_CONTAINER_RUNTIME="sysbox-runc"`; add `XRLENV_INNER_DOCKERD`,
   `XRLENV_INSTALL_DOCKERD`, `XRLENV_SYSTEMD_INIT`, `agent_user`/`verifier_user`
   as the substrate demands. Bring the daemon up the way the VM's init would —
   never edit `solve.sh` to fake it.
3. **Serialize sysbox creates *and* destroys** (`raw_sysbox_create_concurrency=1`,
   `raw_sysbox_destroy_concurrency=1`) and set a per-node cap
   (`max_concurrent_by_runtime` / deploy knob `SYSBOX_MAX_CONCURRENT`, ~4) —
   concurrent sysbox-fs mounts/unmounts wedge under load and leak containers.
4. Callers may request any concurrency; xrlenv paces the cap via fail-fast
   acquire (`XRLENV_HARBOR_ACQUIRE_QUEUE_TIMEOUT_S`) + infra-only `--retries`.
5. Grow the sysbox set **one proven task at a time**, starting with a decisive
   probe (image ships a daemon, single-service, non-privileged).
6. Compose and sysbox are **mutually exclusive** — a multi-service task runs under
   runc; sysbox markers on a compose task are rejected loudly.

---

## 9. Definition of done — the onboarding checklist

- [ ] **Shape chosen** via §2 and recorded in the README's opening paragraph
      (harbor / pier / interceptor / runner-shim / new-framework).
- [ ] **`build_cache.py`** — idempotent `--stage`, pure/tested `task.toml`
      normalizer, fail-loud on bad layout, dataset-identity env overrides.
- [ ] **`build_plan_gen.py`** — `image_ref` read from `task.toml` (never
      synthesized), fail-loud on missing image, canonical YAML shape.
- [ ] **`<name>_build_plan.yaml`** committed (regeneratable via `--all`).
- [ ] **No baked registry host** (§5.3.1) — private-registry images are resolved at
      run time from `.env` (seta's `xrlenv_image_template`, or lhtb's
      `${XRLENV_PRIVATE_REGISTRY_HOST}:${…PORT}/…` placeholder the plugin expands),
      never a frozen host in `task.toml` / the build plan. Check:
      `grep -rE '(ip-[0-9-]+|[0-9.]+):5011/' <benchmark>/` is empty (placeholder only).
- [ ] **`run_oracle_sweep.py`** — `import_path` wiring; exit-0-iff-all-pass;
      pass-gate rule chosen + **unit-tested**; **BOTH retry layers live here** —
      `--retries` (infra-only set) and `--content-retries` (per-task, re-runs only
      the non-passing tasks by the pass gate).
- [ ] **`run_full_sweep.sh`** — a THIN wrapper: build → green-set-by-exclusion →
      invoke `run_oracle_sweep.py --content-retries N` **once** (delegates both
      retries — no bash retry loop); native-budget timeouts; a **uniform flag
      interface** (`--max-workers` / `--content-retries` / `--job-id` / `--jobs-dir`
      / `--skip-build-cache` — run knobs never via env) including `--list-green`
      (print the green set + exit) for the CI sampler.
- [ ] **`tests/`** — offline, network+cluster-free; cover normalizer, plan
      generation (incl. fail-loud), and pass-gate logic. `pytest` green.
- [ ] **G1 oracle sweep passes** at low conc (content proof) *and* high conc
      (platform proof); the green set is `present − EXCLUDE`, every exclusion
      logged with a reason.
- [ ] **Content fixes are auditable overlays/exclusions** in the benchmark dir —
      zero edits to upstream harness logic, zero changes to xrlenv core.
- [ ] **`README.md`** (how-to + rationale) and **`STATUS.md`** (current results +
      reproduce command) written.
- [ ] **Sphinx page** `docs/supported_benchmarks_and_harnesses/<name>.md` added
      (invoke `sphinx-docs-writer`).
- [ ] **Per-benchmark deps** declared as a `pyproject.toml` extra named after the
      benchmark; upstream harness version **pinned exactly**.
- [ ] **A memory note** recorded (`~/.claude/.../memory/`) capturing anything
      non-obvious: version-conflict resolution, blacklist reasons, infra quirks.

---

## 10. Reference implementations (read these before starting)

| Benchmark | Path | Read it for |
|---|---|---|
| **deep_swe** | `benchmarks/deep_swe/` | the **cleanest golden path** — all-`registry`, pier, separate-verifier seam, the canonical README/STATUS + two-retry-layer writeup |
| **lhtb** | `benchmarks/lhtb/` | mixed plan (registry + local + compose sidecar), native offline-egress seal (harbor 0.20 network policy), 3-set exclusion |
| **seta** | `benchmarks/seta/` | all-`git` build-from-Dockerfile, `black_list.txt`, sweep-injected `xrlenv_image_template` kwarg (no subclass) |
| **terminal_bench_2_1** | `benchmarks/terminal_bench_2_1/` | non-hermetic-oracle dep-pins via `patches/`, cpuset/nproc handling |
| **terminalworld** | `benchmarks/terminalworld/` | **multi-service compose + sysbox** (§8), curated `patches/` |
| **evoclaw** | `benchmarks/evoclaw/` | **interceptor** pattern (§7.1) — subprocess/docker shim + sitecustomize |
| **webarena-infinity** | `benchmarks/webarena_infinity/` | **runner-shim** pattern (§7.2) — answer-free substrate + in-container runner |
| shared adapters | `xrlenv_plugins/{harbor,pier}/` | the framework cluster environments you reuse; `compose.py` helpers |

**See also:** `CLAUDE.md` (the three laws + "don't reinvent wheels"),
`specs/14` (adapters), `specs/15` (image cache), `specs/21` (node protocol),
`docs/supported_benchmarks_and_harnesses/writing_your_own_adapter.md`.

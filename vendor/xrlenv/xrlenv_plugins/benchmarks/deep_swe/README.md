# DeepSWE

DeepSWE onboards [datacurve-ai/deep-swe](https://github.com/datacurve-ai/deep-swe)
(HF mirror [datacurve/deep-swe](https://huggingface.co/datasets/datacurve/deep-swe)),
a **113-task pure harbor/pier-format** SWE corpus across 5 languages (go, python,
typescript, rust, javascript). Its xrlenv shape is the **PIER golden path** — each
task is the exact contract pier's `TaskConfig(path=…)` consumes, so the shared pier
cluster plug-in (`xrlenv_plugins.pier:XrlenvPierEnvironmentCluster`) runs them with
no code change — and it grades in a **separate** container (see §3, the
separate-verifier seam).

## What's here

| File | Role |
|---|---|
| `build_cache.py` | populate the 113 task dirs (git / HF) into the shared cache + normalize each `task.toml` (+ a no-op `patch` stage) |
| `build_plan_gen.py` | emit the `type: registry` image **warm** plan (per-task ECR refs read from `task.toml` — never synthesized) |
| `deepswe_build_plan.yaml` | the committed 113-entry warm plan |
| `run_oracle_sweep.py` | the correctness gate — pier's `OracleAgent` per task on the cluster (reward > 0 PASS/FAIL); owns both retry layers |
| `run_full_sweep.sh` | one-command entrypoint (thin): build cache → green set → invoke `run_oracle_sweep.py` once |
| `STATUS.md` | point-in-time per-corpus status (113/113 green as of 2026-07-18) |

The directory is **self-contained**: cache build, warm-plan generation, and the
oracle sweep all live here. The only shared pieces it leans on are generic core
(`xrlenv build apply` / `xrlenv build calibrate`) and the cluster plug-in
`xrlenv_plugins/pier/` (runs the tasks).

```
datacurve-ai/deep-swe (git; or datacurve/deep-swe HF)
        │
        ├─(1)build_cache──────▶ <cache>/deep-swe/<id>/   (task dirs)
        ├─(2)build_plan_gen───▶ type: registry warm plan (public.ecr.aws/... — OPTIONAL)
        └─(3)run_oracle_sweep─▶ per-task reward>0 PASS/FAIL on the cluster
```

## 1. Build the cache

```bash
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example

# the full setup: populate (download + normalize) + patch
.venv/bin/python xrlenv_plugins/benchmarks/deep_swe/build_cache.py --stage all
```

`--stage` (each idempotent; `all` runs `populate` → `patch`): `populate`
materializes the 113 task dirs into `<cache>/deep-swe/<id>/` — each unpacks to
`task.toml` + `instruction.md` + `environment/Dockerfile` + `tests/` + `solution/`
+ `pre_artifacts.sh` — and normalizes each `task.toml`; `patch` applies curated
`patches/<id>/` overlays; `all` (default) runs both. Idempotent throughout: a task
already present is skipped, so re-runs are safe. `--dest` defaults to
`$XRLENV_BENCHMARK_CACHE`.

`--source` picks how to populate: `git` (default; shallow-clone
`datacurve-ai/deep-swe`) or `hf` (snapshot the `datacurve/deep-swe` HF mirror).
Override the dataset identity via `DEEPSWE_REPO_URL` / `DEEPSWE_REPO_REF` /
`DEEPSWE_HF_REPO` / `DEEPSWE_SHARD`.

**`task.toml` normalization is defensive, not required here.** pier's
`EnvironmentConfig` (like harbor's) carries both the deprecated `memory`/`storage`
strings and the canonical `_mb` integers, and rejects a task that sets them
inconsistently. DeepSWE mostly uses `_mb` only, so the normalizer is usually a no-op
— but it's kept so a task that ever ships the conflict still loads (it drops the
deprecated duplicate where the canonical field is present).

> **The `patch` stage is available but currently a no-op** — deep_swe ships no
> curated patches today (no `patches/` dir), because it grades behaviorally against
> baked-in tests, so unpinned-dependency drift risk is low. If a future oracle sweep
> surfaces broken content, drop a full-file overlay under `patches/<id>/` and re-run
> `build_cache.py --stage patch`.

## 2. Prepare the images

Nothing is built on our side. Each task ships a **prebuilt per-task image on public
ECR** — `[environment] docker_image = public.ecr.aws/d3j8x8q7/swe-bench-202605:<ext_id>-v1.1`
— **read straight from `task.toml`** (a task with no `docker_image` fails loud; a
ref is never synthesized). So there's no source to build and no reason to copy it
into the xrlenv private registry.

Because these live on **public ECR** (anonymous-pullable, no punishing rate limit),
a direct node-side `docker pull` needs **no new infra** — this is **Path 1**, and
the classic reason to stand up a pull-through mirror (Docker Hub's anonymous-pull
throttling) simply doesn't apply. `build_plan_gen.py --all` reads each task's
authoritative ref and emits a `context_source: {type: registry}` **warm** plan:

```bash
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example

# (re)generate the committed plan from the populated shard:
.venv/bin/python -m xrlenv_plugins.benchmarks.deep_swe.build_plan_gen \
    --all --output xrlenv_plugins/benchmarks/deep_swe/deepswe_build_plan.yaml
```

Warming this plan is **optional** — the 113/113 run had **no pre-warm** even at
conc-32, relying on the cluster's dynamic image cache (lazy-pull-on-acquire + LRU
eviction + image-affinity). To warm ahead of time (and to calibrate the size hints),
see **§4**. The optional Path 2 (a pull-through mirror that rewrites the ref host to
an ECR-upstream proxy) is kept for egress control if ever needed, off by default;
see the `build_plan_gen.py` module docstring for the `--tasks` subset invocation and
`--probe`.

## 3. Run the oracle sweep (validate the cache)

The corpus-quality gate: run pier's `OracleAgent` (applies each task's `solution/`
reference and commits) per task **on the xrlenv cluster** and confirm each earns a
positive `reward`. Under the oracle, a non-passing task is a plumbing/content bug
(reward ceiling 0 → poison for RL), not a model signal.

`run_full_sweep.sh` is the entrypoint. It (1) sources `./.env` for CP host + token,
(2) (re)builds the cache, (3) computes the **green set** = present tasks − `EXCLUDE`
(`EXCLUDE=()` — the whole corpus is green, so a re-populate auto-picks up new tasks),
then (4) invokes `run_oracle_sweep.py` **once**, trusting its exit code. Exit is `0`
iff every oracle solves.

```bash
set -a; source ./.env; set +a            # XRLENV_GRPC_HOST + XRLENV_CONSUMER_TOKEN
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example

# THE FULL GATE:
bash xrlenv_plugins/benchmarks/deep_swe/run_full_sweep.sh
bash xrlenv_plugins/benchmarks/deep_swe/run_full_sweep.sh --max-workers 32

#   flags: --max-workers (default 8), --content-retries (default 2), --job-id,
#   --jobs-dir, --skip-build-cache, --list-green (print the green set + exit);
#   any extra arg passes through to run_oracle_sweep.py.

# --- targeted subsets / one-shot runs go through run_oracle_sweep.py directly ---
# NB: a SWE oracle+verifier trial exceeds a foreground shell/tool timeout — run the
# whole sweep under nohup/background (~25-30 min for all 113):
nohup .venv/bin/python xrlenv_plugins/benchmarks/deep_swe/run_oracle_sweep.py \
    --max-workers 32 --retries 6 \
    --jobs-dir ./tmp --job-id deepswe-native-113 > tmp/native113.log 2>&1 &

# a smoke subset:
.venv/bin/python xrlenv_plugins/benchmarks/deep_swe/run_oracle_sweep.py \
    --tasks fastapi-implicit-head-options --jobs-dir ./tmp --job-id deepswe-smoke
```

* **Pass gate keys on the `reward` key only.** DeepSWE `reward.json` also carries
  f2p/p2p totals + fractions + `partial` that can be legitimately 0; the gate is
  `float(rewards["reward"]) > 0`, with the rest reported as side metrics. (An
  "all values > 0" gate would false-FAIL a passing task whose `p2p_total` or
  `partial` is 0.)

* Exit code is `0` only if every oracle solved, so the sweep is CI-usable.
  Per-trial artifacts (`agent/`, `verifier/reward.json`, trial logs) land under
  `--jobs-dir/<job-id>/` (default the repo's gitignored `tmp/`); each content-retry round
  writes a **sibling** `<job-id>-retryN/` dir, so a retried task's artifacts live there, not
  under the base job.
  `--list-green` (paired with `--skip-build-cache`) prints the green set and exits
  — the seam `xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py` samples from.

* Resource-ablation knobs (`--cpus-multiplier`, `--memory-multiplier`,
  `--cpu-pinning`, `--timeout-multiplier`, `--override-*`) mirror the tb2.1 / TW
  sweeps. Full per-task disposition: [`STATUS.md`](STATUS.md).

> AS OF 2026-07-18: **Green set = 113 / 113** (`reward=1.0`; f2p + p2p fully green).
> conc-8, ~26 min, **0 infra errors / 0 content failures / 0 retries consumed**, all
> 5 languages green. `EXCLUDE` is empty.

### Two retry layers — and why

The gate has **two independent retry mechanisms** at different granularities. Both
live inside `run_oracle_sweep.py` (the wrapper just passes the flags); they catch two
*different* failure classes, and neither may re-roll a genuine eval result:

* **`--retries` (default 6) — per-*trial*, infra-only.** Wired via pier's
  `RetryConfig(include_exceptions=_INFRA_RETRY_EXCEPTIONS)`. Retries a trial *only*
  on the four infra-transient exceptions (`CapacityExhausted`, `ControlPlaneLost`,
  `NodeLost`, `NodeCommandTimeout`) — never on a content outcome. Its job is to
  absorb capacity pacing: the sweep deliberately requests more concurrency than the
  runtime cap, so an at-cap acquire fails *fast* with `CapacityExhausted` and
  re-queues. **The final stats record one result per task** — a retried task that then
  passes counts once, never double-counted. In the common case the infra failure is a
  fail-fast **acquire** (before `solve.sh` runs); a **post-acquire** infra error
  (e.g. `NodeCommandTimeout` on an exec) re-runs the whole attempt in a **fresh
  container**, so `solve.sh` can *execute* more than once — this only matters for
  **external** side effects, not the recorded result. By construction this layer
  *cannot* mask a flaky task (a content outcome is never re-rolled). `run_full_sweep.sh`
  sets it to 6.

* **`--content-retries` (default 2) — per-*task*, outcome-keyed.** After the sweep it
  re-runs *only* the tasks that finished `reward=0` (by this benchmark's own
  `_trial_passes` gate), up to N more times; a task is solved if ANY attempt passes.
  This is the coarse net for a flake that slipped through as a bad *reward* rather
  than a typed infra exception — a network blip during a dependency fetch, a node
  evicted mid-verify. It is **bounded and visible**: each content-retry round writes a
  **sibling `<job-id>-retryN/` directory**, so a task that only passes on a re-run has
  artifacts under a `-retryN` sibling (discoverable as *flaky*) rather than being silently
  greened — the summary reports the final pass/fail outcome, not a retry-count field. A task
  that fails all `1 + N` attempts reddens the gate (exit 1). Pass `--content-retries 0` for a
  zero-tolerance gate.

Why two: the **infra layer** keeps a single *trial* from failing on pacing (typed
exception, eval signal untouched); the **content layer** keeps the *corpus gate* from
red-flagging a one-off environmental hiccup that surfaced as a reward-0 instead of a
typed exception. Both report their counts, so neither hides a real regression.

**Timeouts run at the native budget.** The gate uses `--timeout-multiplier 1.0` (the
default): each task runs at its own declared `timeout_sec`, so a task whose reference
solution can't fit its own budget fails *loud* (a content bug), rather than being
silently rescued by inflated headroom. `--timeout-multiplier` stays available as an
ablation / contention-compensation knob, but it is **not** part of the gate.

### The separate-verifier seam

DeepSWE grades in a **fresh container**, not the solve container
(`environment_mode="separate"`) — the interesting mechanism worth knowing. No sweep
flag toggles this; the pier plug-in detects `separate` mode from `task.toml` and
wires it. For all 113 tasks the pier cluster env:

1. resolves the **verifier base image** from the tests `Dockerfile` `FROM` (falling
   back to the parent task's top-level `docker_image`);
2. **uploads `/tests` itself** — pier hardcodes `skip_tests_upload=True`, so the
   plug-in stages the test bundle into the verifier container;
3. round-trips the **reward back to the host** via `download_dir`
   (`capabilities.mounted=False` — nothing is bind-mounted on the shared cluster),
   so `verifier/reward.json` lands host-side for the gate to read.

Details: `xrlenv_plugins/pier/README.md`.

## 4. Warm the image and Calibrate the image size (optional)

Both steps are optional and only worth doing to amortize the first-acquire pull
across a big run — the cluster's dynamic image cache (lazy pull-on-acquire + LRU
eviction + image-affinity) means a sweep works with no pre-warm at all (§3).

**Warm** the plan onto nodes (Path 1 — direct public-ECR pull, no new infra):

```bash
# --connect-host is REQUIRED (it dials the control plane); without it the CLI exits 2.
xrlenv build apply \
    --plan xrlenv_plugins/benchmarks/deep_swe/deepswe_build_plan.yaml --fill-missing \
    --connect-host <control-plane-host>
```

**Calibrate** the size hints. The size probe is **OFF by default** — the shared probe
targets Docker Hub, not public ECR, so it can't size these refs; the plan ships a
conservative 2.5 GiB heuristic hint (`size_hint_source: heuristic`). After the first
warm, refine to true on-disk uncompressed (`cluster-reported`) sizes into a
**separate** `*.calibrated.yaml` (diff before promoting):

```bash
source .venv/bin/activate
export XRLENV_OPERATOR_TOKEN=<operator token>
xrlenv build calibrate \
    --plan xrlenv_plugins/benchmarks/deep_swe/deepswe_build_plan.yaml \
    --output xrlenv_plugins/benchmarks/deep_swe/deepswe_build_plan.calibrated.yaml \
    --connect-host <control-plane-host>
```

## See also

- `xrlenv_plugins/benchmarks/GUIDELINE_onboard_benchmarks.md` — the onboarding
  convention (deep_swe is its "cleanest golden path" reference).
- `xrlenv_plugins/pier/README.md` — the cluster plug-in that runs the tasks
  (including the separate-verifier wiring).
- `docs/supported_benchmarks_and_harnesses/deep_swe.md` — the Sphinx user page.
- [`STATUS.md`](STATUS.md) — current per-corpus status + reproduce command.

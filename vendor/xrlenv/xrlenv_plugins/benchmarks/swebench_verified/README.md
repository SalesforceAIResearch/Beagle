# SWE-bench Verified — xrlenv onboarding

SWE-bench Verified is the 500-instance human-validated corpus from the HF dataset
`SWE-bench/SWE-bench_Verified`. Its xrlenv shape is the **docker-py drop-in** path
(GUIDELINE §7.1 / §2 "already uses docker-py") — **not** the harbor/pier
`import_path` path: swebench ships its own harness
(`swebench.harness.run_evaluation`) that already drives containers through
docker-py, so `run_oracle_sweep.py` runs that **upstream harness unmodified** and
reroutes it onto the cluster via the xrlenv docker-py-compat client
(`docker.from_env()` → `xrlenv.from_env()`). Each instance's prediction is the
dataset's **gold patch** — the oracle — so upstream's grader (`get_eval_report`)
should mark every instance `resolved: true`. A non-resolving instance under the
oracle is a plumbing/content bug, not a model-eval signal.

## What's here

A self-contained ops kit — no adapter and no manifest (the harness carries its own
docker-py seam), reusing only generic core (`xrlenv build apply` /
`xrlenv build calibrate`).

| File | Role |
|---|---|
| `build_cache.py` | materialize the Verified **task data** into `<cache>/swebench-verified/<instance_id>/` — `instance.json` (the full upstream row, the anchor) + `problem_statement.md` (a real agent's prompt) + `gold_patch.diff` (the oracle's prediction). The **image** is not cached — pulled on first acquire. |
| `build_plan_gen.py` + `swebench_verified_build_plan.yaml` | emit the image plan — one `type: registry` entry per instance (Docker Hub `swebench/sweb.eval.x86_64.*`, pulled on first acquire). The committed YAML is the 8-instance smoke plan; regenerate with `--all`. |
| `verified_instance_ids.txt` | the vendored, digest-validated 500-id manifest — the authority the `--all` plan and the full-suite gate check membership against. |
| `run_oracle_sweep.py` | the gate — drive upstream `run_instance` + `get_eval_report` per instance via the docker-py drop-in; exit 0 iff every instance `resolved`; writes `summary.json`. Owns both retry layers. |
| `run_full_sweep.sh` | the one-command entrypoint — (re)build cache → compute green set → invoke `run_oracle_sweep.py` once; `--smoke` for the 8-instance subset. |
| `tests/` | offline unit tests (row normalization, image-ref derivation, pass-gate). |
| `README.md` / `STATUS.md` | this file / point-in-time results. |

```
build_cache.py          build_plan_gen.py         run_oracle_sweep.py
(HF rows → cache)   →   (registry image plan)  →  (docker-py drop-in: run_instance
                                                   per instance, gold patch = pred,
                                                   upstream get_eval_report → resolved)
```

## 1. Build the cache

Unlike harbor/pier benchmarks there are no `task.toml` task dirs — the corpus is
upstream's HF dataset. But we still **cache the task data** so the corpus is
self-contained and offline, and so a *real agent* run (not just the oracle) can
read the prompt from the cache. `build_cache.py` materializes each Verified row
under the shard:

```bash
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example
.venv/bin/python xrlenv_plugins/benchmarks/swebench_verified/build_cache.py --stage all --all
# or just the 8 smoke instances:
.venv/bin/python .../build_cache.py --stage all --smoke
```

```
<cache>/swebench-verified/<instance_id>/
├── instance.json         # the full upstream row (anchor, written last)
├── problem_statement.md   # the PROMPT a real agent reads
└── gold_patch.diff        # the GOLD patch the oracle applies
```

The cache root comes from `--dest` or `$XRLENV_BENCHMARK_CACHE`. There is **no
`patch` stage** — swebench content is clean upstream, so `--stage all` == `--stage
populate` (the CLI accepts both). The build is **idempotent**: a complete,
semantically-valid instance dir (all files present, a parseable anchor whose
`instance_id` agrees with the dir name, extracts matching the anchor — see
`_is_complete`) is skipped; an incomplete/corrupt one is re-materialized; re-runs
are safe. Membership is gated by the **vendored 500-id manifest**
(`verified_instance_ids.txt`, count/uniqueness/order + a mandatory `sha256(ids)`
digest): `--all` prefers complete cache entries and falls back to the validated
manifest, so a partial or drifted cache can never silently masquerade as the full
Verified corpus.

The **image** is not cached here — it's pulled from Docker Hub on first acquire by
the node's `ImageCacheManager`; `build_plan_gen.py` (§2) emits the registry plan.

## 2. Prepare the images

**Nothing is built here.** swebench publishes a prebuilt per-instance image on
Docker Hub — `swebench/sweb.eval.x86_64.<key>:latest`, where `<key>` mirrors
upstream's derivation (`__` → `_1776_`, lowercased) — for every Verified instance.
`build_plan_gen.py` emits a plan with **one `type: registry` entry per instance**;
the node's `ImageCacheManager` pulls each image on first acquire, so this step is
**optional** — you can go straight to the gate (§3).

```bash
.venv/bin/python -m xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen \
    --all --max-workers 8 \
    --output xrlenv_plugins/benchmarks/swebench_verified/swebench_verified_build_plan.yaml
# --smoke for the committed 8-instance plan; --no-probe to skip the Docker Hub size probes
```

Sizes are probed from Docker Hub's v2 manifest API
(`size_hint_source: registry-probe`); the committed YAML is the smoke plan.
Warming and calibrating the plan (to amortize the first-acquire pull and record
true on-disk sizes) is covered in **§4**.

## 3. Run the oracle sweep (validate the cache)

`run_full_sweep.sh` is the correctness gate. It (1) sources `./.env` for the
control-plane host + consumer token, (2) (re)builds the cache, (3) computes the
**green set** = present-complete cached instances − `EXCLUDE`, gating the full
suite on membership against the vendored 500-id manifest (reconciled so
`INCLUDE ∪ EXCLUDE` must equal the manifest — a held-out id is still a real
Verified instance, no fabrication/no silent subset), and (4) invokes
`run_oracle_sweep.py` **once** over the green set, trusting its exit code.
Exit 0 iff **every** instance reports `resolved: true` under the gold-patch oracle.

`EXCLUDE` holds only the **6 upstream-ungradeable** instances (green set = **494**),
each with a per-id root cause in `STATUS.md`: the gold-patch oracle itself can't
resolve under swebench 4.1.0 + the canonical images (astropy setuptools-68 distutils
collection error; sphinx 3.5/4.3 bare-checkout reverts the `pytest -rA` pre_install so
the parser sees no markers; astropy param-ID drift; django deterministic ERROR). These
are upstream defects, not cache/xrlenv bugs.

The **4 flaky `psf__requests`** are **kept IN the green set (not excluded)** and
documented as flaky: `test_requests.py` defaults `HTTPBIN` to external
`http://httpbin.org/` (no local pytest-httpbin, `HTTPBIN_URL` unset, no egress proxy),
so httpbin.org can return an nginx `503` under concurrent load; they pass in isolation.
A one-off 503 is a known, no-surprise flake; a **consistent** failure is a new signal
(httpbin.org down/changed, or an egress regression). Making them hermetic (a local
httpbin + `HTTPBIN_URL`) would remove the flakiness if it ever becomes persistent.

`scikit-learn__scikit-learn-14710` is **not** excluded — it was a real xrlenv bug
(root-caused + fixed, validated end-to-end on the cluster: 80/80, resolved). Its
OpenMP-heavy GBM tests hung under quota-only containers (threads sized to the host
core count, throttled to the CFS quota → thrash). Fix: per-task cpuset pinning via
`run_oracle_sweep.py`'s `_CPU_PINNED_INSTANCES` map (→ `cpu_isolation=REQUIRED`
through `rollout_metadata` → `acquire_container`), the docker-py-drop-in analogue of
harbor/tb2.1's `XRLENV_CPU_PINNING` marker — per-task, not global. See STATUS.md.

```bash
set -a; source ./.env; set +a                 # XRLENV_GRPC_HOST + XRLENV_CONSUMER_TOKEN
bash xrlenv_plugins/benchmarks/swebench_verified/run_full_sweep.sh                 # full 500
bash .../run_full_sweep.sh --max-workers 8                                          # concurrent
bash .../run_full_sweep.sh --smoke                                                  # 8-instance subset

# read-only: print the green set and exit, running nothing (the CI sampler's seam):
bash .../run_full_sweep.sh --list-green --skip-build-cache

# a subset / a local baseline, straight through run_oracle_sweep.py:
.venv/bin/python .../run_oracle_sweep.py --smoke                # the 8 smoke instances
.venv/bin/python .../run_oracle_sweep.py --smoke --local        # local docker daemon, no cluster
```

**Why the docker-py drop-in (not harbor/pier).** swebench's harness *is* the
grader — it applies the patch, runs the per-repo tests, and calls its own
`get_eval_report`. We must not vendor a parallel grader (GUIDELINE law #2), so we
keep the harness whole and only swap the docker client. The single xrlenv-specific
line is `client = xrlenv.from_env()` (`--local` uses `docker.from_env()` for a
baseline). The **pass gate** is upstream's own `resolved` field, read verbatim
from the per-instance `report.json` — we never invent a resolution rule.

**Two retry layers** (same design as the harbor/pier sweeps; both live *inside*
`run_oracle_sweep.py` — the shell wrapper only forwards the flags, so every driver
gets them from one place):

| Layer | Granularity | Retries on | Purpose |
|---|---|---|---|
| `--retries` (6 in the gate) | per-trial, in `run_oracle_sweep.py` | infra-transient exceptions only — `CapacityExhausted` / `PinCapacityExhausted` / `ControlPlaneLost` / `NodeLost` / `NodeCommandTimeout` | absorb capacity pacing; **cannot** mask a flaky instance |
| `--content-retries` (2 in the gate) | per-instance, in `run_oracle_sweep.py` | a non-`resolved` outcome (re-runs ONLY the unresolved instances) | catch a one-off environmental flake vs a real regression |

Both report their counts, and content failures are **never** re-rolled by the
infra layer. Because upstream swallows/wraps infra exceptions inside `run_instance`
(returning `completed: false`), the drop-in stamps a structured failure record at
the container-op boundary that the driver reads back — so a capacity/node blip is
retried as infra, not misreported as a content failure. **Timeouts run at
swebench's native 1800 s** (no inflated headroom in the gate).

## 4. Warm the image and Calibrate the image size (optional)

Warming and calibrating are **optional** — the cluster's dynamic image cache
(lazy pull-on-acquire + LRU eviction + image-affinity) means the gate works
without either. Reach for them only to make a big sweep cheaper.

```bash
# warm: pre-pull the plan's images so the first acquire doesn't pay the pull
xrlenv build apply --plan xrlenv_plugins/benchmarks/swebench_verified/swebench_verified_build_plan.yaml \
    --fill-missing --connect-host <cp-host>

# calibrate: record true on-disk sizes after the first warm
xrlenv build calibrate --plan .../swebench_verified_build_plan.yaml \
    --output .../swebench_verified_build_plan.calibrated.yaml --connect-host <cp-host>
```

**Warm** (`xrlenv build apply … --fill-missing`) amortizes the first-acquire pull
across a big run — useful, never required.

**Calibrate** (`xrlenv build calibrate`) queries each node for each image's actual
**on-disk (uncompressed)** size, takes the max per `image_ref`, and writes a
**separate** `*.calibrated.yaml` with `size_hint_bytes` = measured and
`size_hint_source: cluster-reported`. Those true sizes give the FFD bin-packer
tighter placement on the next apply — and they are **distinct** from the
registry-probe **compressed** sizes the generated plan ships (the two differ).

## See also

- `xrlenv_plugins/benchmarks/GUIDELINE_onboard_benchmarks.md` — the onboarding convention (this benchmark is the §7.1 docker-py drop-in worked example).
- `docs/supported_benchmarks_and_harnesses/swe_bench.md` — the Sphinx user page.
- `STATUS.md` — current results + the exact reproduce command.

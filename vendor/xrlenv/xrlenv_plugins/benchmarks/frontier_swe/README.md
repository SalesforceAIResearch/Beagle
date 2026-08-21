# frontier-swe

FrontierSWE ([Proximal-Labs/frontier-swe](https://github.com/Proximal-Labs/frontier-swe))
is a **harbor-format** corpus of ultra-long-horizon technical challenges
(performance engineering, computational science, ML research — 17 tasks). Each
task unpacks to a self-contained container image, a `tests/` verifier that writes
`reward.json` to `/logs/verifier`, and — for the gateable subset — a
`solution/solve.sh` reference (the harbor filesystem contract). Its xrlenv shape
is the **harbor golden path**: the sweep reuses the shared cluster environment
(`xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster`) with zero adapter code,
so this directory is a self-contained ops kit on top of generic core.

Two distinctive traits drive this kit (both documented in place below):

1. **Grade-from-artifact.** harbor 0.20's strict `VerifierResult` can't ingest
  FrontierSWE's *rich* `reward.json` (a `score`/`reward` PLUS a `subscores` list
   and an `additional_data` dict), so `run_oracle_sweep.py` grades from the
   **downloaded** `reward.json` on disk — the same file upstream's
   `scripts/score_from_reward.py` consumes — instead of harbor's parsed result.
2. **Run-time oracle mode.** The verifiers run an anti-cheat/anti-wrapper source
  scan that would FAIL the oracle; the sweep injects `HARBOR_ORACLE_MODE=1` at
   run time (never baked into `task.toml`).



## What's here


| File                           | Role                                                                                                                  |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `build_cache.py`               | git-clone-populate a faithful copy of the corpus into the shared cache (+ normalize `task.toml`, + a `patches/` hook) |
| `build_plan_gen.py`            | emit the `type: registry` image-warmup plan (per-task `docker_image` read from `task.toml`)                           |
| `frontier_swe_build_plan.yaml` | committed warm plan for the **green** set (GHCR refs, heuristic sizes)                                                 |
| `run_oracle_sweep.py`          | oracle-per-task correctness gate on the xrlenv cluster (owns both retry layers; grades from the downloaded artifact)  |
| `run_full_sweep.sh`            | thin one-command entrypoint: build cache → green set → invoke the sweep                                               |
| `tests/`                       | offline unit tests for the pure normalize / build-plan / pass-gate logic                                              |
| `STATUS.md`                    | point-in-time oracle-sweep disposition + reproduce command                                                            |


```
build_cache.py ─▶ <cache>/frontier-swe/<task>/ ─▶ build_plan_gen.py (type: registry warm) ─▶ run_oracle_sweep.py (reward>0 PASS/FAIL, from disk)
```



## 1. Build the cache

`build_cache.py --stage all` takes a fresh box to a ready cache in two idempotent
stages:

1. **POPULATE** — shallow `git clone` `Proximal-Labs/frontier-swe` and copy each
  `tasks/<id>/` into `<cache>/frontier-swe/<id>/`, normalizing each `task.toml`.
   `FRONTIER_SWE_REPO_URL` may point at a **local clone** for an offline
   re-populate (`git clone` handles a local source natively).
2. **PATCH** — apply any curated `patches/<id>/` overlays. Starts **empty**:
  FrontierSWE's oracle verifiers are deterministic local computations, so the
   unpinned-dep-drift risk that motivates tb2.1's pins is low here; the hook stays
   for when the oracle sweep surfaces broken content.

`--dest` is the shared cache **root** and defaults to `$XRLENV_BENCHMARK_CACHE`;
the dataset lands under `<dest>/frontier-swe/` beside the other shards.

```bash
export XRLENV_BENCHMARK_CACHE=/path/to/benchmark-cache

# populate (if missing) + patch. Idempotent; safe to re-run.
.venv/bin/python xrlenv_plugins/benchmarks/frontier_swe/build_cache.py --stage all

# populate only (network box) / patch only (no network):
.venv/bin/python xrlenv_plugins/benchmarks/frontier_swe/build_cache.py --stage populate
.venv/bin/python .xrlenv_plugins/benchmarks/frontier_swe/build_cache.py --stage patch
```

The full corpus is **17 tasks**; `build_cache` reports how many are
**gateable** (ship `solution/solve.sh`). FrontierSWE is a live leaderboard, so
upstream **withholds the reference solution** for 6 tasks (anti-leakage) — those
carry `tests/` + `task.toml` but no `solution/solve.sh`. One of the 6
(`notebook-compression`) is recovered by an xrlenv-authored solution overlay (§ Task-
level cache fixes → now gateable); the other 5 are not oracle-derivable *and* not
solvable by a static `solve.sh`, so they are never enumerated. See STATUS.md.

### Task-level cache fixes

`--stage patch` applies curated **full-file overlays** from `patches/<task_id>/` on
top of the faithful copy (idempotent, survive re-populate), logged in
`patches/README.md` + `STATUS.md`. xrlenv core is never touched. Two kinds, kept
strictly distinct: an **oracle fix** (re-path/complete the *upstream* reference) and
an **xrlenv-authored solution** (for a task whose reference is withheld — clearly
labelled, never passed off as an upstream oracle).

- **`dependent-type-checker` — oracle reference-path fix.** Its `solve.sh` copied the
  reference type-checker from `/tests/reference_impl` *during the solve phase*, but
  harbor mounts `/tests` only during **verify** (a contract upstream states in its own
  `cranelift`/`libexpat` oracles), so the build silently no-op'd and the empty checker
  rejected all 174 valid programs. The overlay bundles the **byte-identical**
  `reference_impl` (sha256-matched to the copy the verifier builds) under `solution/`
  and points `solve.sh` there — upstream's own documented pattern. **Confirmed
  on-cluster** (2026-08-06): the oracle builds and scores reward 1.0005.
- **`notebook-compression` — xrlenv-authored solution.** Upstream withholds the
  reference (no `solution/solve.sh`) and `tests/` ships only a hidden holdout + scorer
  (nothing to *derive*), so this overlay is an **xrlenv-authored** `/app/run`
  submission — a lossless per-file `lzma` compressor (stdlib only, no network) — added
  to prove the task is solvable end-to-end. **NOT an upstream oracle** (loudly
  labelled). **Confirmed on-cluster** (2026-08-06): reward 0.3175, round-trip lossless
  on 80 hidden notebooks.

## 2. Prepare the images

**Thin — nothing is built.** Every FrontierSWE task ships a **prebuilt** registry
image on **public GHCR** (`ghcr.io/proximal-labs/frontier-swe/<id>:<tag>`,
anonymous-pullable — verified), and its per-task ref is **read from** `task.toml`**'s**
`[environment] docker_image` — never synthesized (tags vary per task: `:v4`,
`:v5`, `:v6`). The cluster pulls each on first acquire, so warming is **optional**;
pull them **directly** (the deep_swe public-registry decision — no private registry,
no new infra). A task with no `docker_image` fails loud.

`build_plan_gen.py` emits a `type: registry` warm plan. The committed
`frontier_swe_build_plan.yaml` covers only the **green** set (the images the sweep
actually pulls) — the other corpus images are GPU (multi-GB CUDA, useless on a CPU
cluster), oracle-defective, or solution-withheld (never run), so warming them wastes
fleet disk. `--all` regenerates the full 17-task plan if needed. See §4 to eager-warm.

## 3. Run the oracle sweep (validate the cache)

The proof the cache is good: run harbor's OracleAgent for each gateable task **on
the xrlenv cluster** and confirm every green task earns a positive reward.
`run_full_sweep.sh` is the thin gate — it (1) sources `./.env` for the CP host +
token, (2) rebuilds the cache, (3) computes the green set = **present
gateable tasks −** `EXCLUDE` (asserts 12 present / 6 green), and (4) invokes
`run_oracle_sweep.py` once over that set, trusting its exit code.

```bash
set -a; source ./.env; set +a                      # XRLENV_GRPC_HOST + token
export XRLENV_BENCHMARK_CACHE=/path/to/benchmark-cache

bash xrlenv_plugins/benchmarks/frontier_swe/run_full_sweep.sh   # 5 green tasks
# override concurrency: ... run_full_sweep.sh --max-workers 16
```

**Pass gate (grade-from-artifact).** harbor 0.20's `VerifierResult` strictly
validates `rewards: dict[str, float|int]`, but FrontierSWE's `reward.json` carries
a `subscores` list + an `additional_data` dict, so harbor rejects it and
`verifier_result` comes back None with a `ValidationError` on **every** task.
harbor downloads the verifier dir to disk *before* that parse and catches the error
per-trial, so `run_oracle_sweep.py::_trial_passes` reads the **downloaded**
`reward.json` and passes iff `reward` (fallback `score`) `> 0`. The harbor
`ValidationError` is IGNORED whenever a gradeable `reward.json` is present; only a
**missing** reward.json (the verifier never produced output) counts as a real
failure. This delegates grading to upstream's own artifact — no harbor edit, no
verifier edit, no xrlenv-core change. Exit code is 0 **iff every green oracle
solved**, so the sweep is CI-usable.

**Oracle mode.** The sweep injects `HARBOR_ORACLE_MODE=1` via the job's
`environment.env` + `verifier.env` (harbor threads `verifier.env` to the verifier
as `override_env`). This relaxes the verifier's anti-cheat/anti-wrapper scan for the
reference solution (which legitimately wraps/links the library the task asks the
agent to reimplement). It is **run-time only** — never written into `task.toml`, so
a real agent eval is unaffected.

**The two retry layers** both live in `run_oracle_sweep.py` (so every driver — the
wrapper and the `xrlenv_plugins/benchmarks/tests/integration/` ci runner — gets them
from one place):


| Layer                                           | Granularity                                 | Retries on                                                                                                        | Purpose                                                                                                                                                                                                                                          |
| ----------------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--retries` (default 6)                         | per-**task attempt** (fresh container each) | the 4 infra-transient exceptions only (`CapacityExhausted`, `ControlPlaneLost`, `NodeLost`, `NodeCommandTimeout`) | absorb capacity pacing at high `--max-workers`. One result per task; a content outcome is never re-rolled. NB: the reward-schema `ValidationError` is deliberately NOT in this set — it's expected + handled by grade-from-artifact, not retried |
| `--content-retries` (default 2 via the wrapper) | per-**task**                                | a reward-0 *outcome* — re-runs ONLY the non-passing tasks                                                         | catch a one-off environmental flake surfaced as reward-0; a task is solved if ANY attempt passes                                                                                                                                                 |


**Timeouts run at native budget** (no `--timeout-multiplier` in the gate) — a
reference solution that can't fit its own `timeout_sec` should fail loud. Because
FrontierSWE oracles are heavy (up to 128 GB / 16 CPU) and long (oracle ~1-2 h), run
the sweep under `nohup`/background and poll.

**Other flags.** `run_full_sweep.sh` accepts `--max-workers N` (default 8),
`--skip-build-cache`, `--list-green` (print the green set and exit — the seam the ci
sampler uses), `--job-id` / `--jobs-dir`, and forwards anything unrecognized to
`run_oracle_sweep.py`.

## 4. Warm the image and Calibrate the image size (optional)

Warming is optional (lazy pull-on-acquire + LRU eviction); pre-warm only to amortize
the first-acquire pull across a big run.

```bash
# regenerate the committed green plan whenever tags change:
GREEN=$(bash xrlenv_plugins/benchmarks/frontier_swe/run_full_sweep.sh --list-green | paste -sd,)
XRLENV_BENCHMARK_CACHE=/path/to/benchmark-cache \
.venv/bin/python -m xrlenv_plugins.benchmarks.frontier_swe.build_plan_gen \
    --tasks "$GREEN" \
    --output ./xrlenv_plugins/benchmarks/frontier_swe/frontier_swe_build_plan.yaml

# eager-warm across the cluster (FFD bin-packed onto nodes):
xrlenv build apply \
    --plan ./xrlenv_plugins/benchmarks/frontier_swe/frontier_swe_build_plan.yaml \
    --connect-host "$XRLENV_GRPC_HOST" --fill-missing
```

The committed plan's `size_hint_bytes` are a conservative heuristic (the shared
Docker-Hub probe can't size GHCR refs). After the images are materialized on the
nodes, `xrlenv build calibrate` refines them to true on-disk `cluster-reported`
sizes in a **separate** `*.calibrated.yaml` (diff before promoting):

```bash
export XRLENV_OPERATOR_TOKEN=<operator token>
xrlenv build calibrate \
    --plan ./xrlenv_plugins/benchmarks/frontier_swe/frontier_swe_build_plan.yaml \
    --output ./xrlenv_plugins/benchmarks/frontier_swe/frontier_swe_build_plan.calibrated.yaml \
    --connect-host "$XRLENV_GRPC_HOST"
```



### Re-including a GPU / net task later

The 4 GPU tasks (`granite-mamba2-inference-optimization`,
`inference-system-optimization`, `optimizer-design`, `pcqm4mv2-autoresearch`) are
EXCLUDEd only because the dev cluster is CPU-only — drop them from
`run_full_sweep.sh`'s `EXCLUDE` and re-pin the catalog counts once GPU nodes exist.
`frogsgame-rl` is un-gateable for two independent reasons: it ships **no**
`solution/solve.sh`, AND it needs `allow_internet=true` + an external **Tinker API**
key (`TINKER_API_KEY`) whose stochastic 500-board inference grades the verifier
(non-hermetic). Enabling egress is trivial (harbor 0.20 native network policy honors
the task's `allow_internet`); the missing paid key + stochastic external grading is
the real blocker.

## See also

- `xrlenv_plugins/benchmarks/GUIDELINE_onboard_benchmarks.md` — the onboarding
convention this kit follows (§3 golden-path file contracts, §5 image mechanics).
- `xrlenv_plugins/harbor/README.md` — the shared harbor cluster environment.
- `docs/supported_benchmarks_and_harnesses/frontier_swe.md` — the Sphinx user page.
- `[STATUS.md](STATUS.md)` — current oracle-sweep disposition (green set +
reproduce command).


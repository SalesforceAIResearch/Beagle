# swebench-pro — xrlenv onboarding

SWE-bench Pro (ScaleAI/SWE-bench_Pro, 731 public instances across 11 repos; Go 280 / Python
266 / JS 165 / TS 20) is distributed as a HF dataset plus a per-instance evaluation kit in the
upstream harness repo (`run_scripts/<id>/{run_script.sh,parser.py}`,
`dockerfiles/{base,instance}_dockerfile/<id>/Dockerfile`) and a **prebuilt image per instance on
Docker Hub** (`jefzda/sweap-images:<dockerhub_tag>`, the tag is a dataset column). Its xrlenv
shape is the **harbor golden path** (GUIDELINE §2 Q1): each instance becomes a self-contained
harbor task dir (image + `solution/solve.sh` + a `tests/test.sh` that writes
`/logs/verifier/reward.txt`), and the sweep reuses
`xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster` with zero adapter code.

**Images are pulled, never rebuilt.** Upstream's own evaluator pulls `jefzda/sweap-images`
from Docker Hub; the Dockerfiles in the kit are the recipe only. Each task's `docker_image` is
pulled on first use; `build_plan_full.yaml` exists for optional eager warm-up. Sizes
(registry-probed): mean 1.47 GB, most 0.5–1.5 GB, `gravitational/teleport` ~2.5 GB,
`protonmail/webclients` 4–5.6 GB (the largest); 731 images ≈ 1071 GB.

This directory is the **full corpus**: all 731 public instances, the corpus as published, and
the gold-patch (oracle) sweep over it that is the onboarding gate. The derived selections — the
quality-filtered set (478), the repo-balanced subset-100 and the one-task smoke test — live in
[`scripts/`](scripts/README.md) together with the generators that produce them.

## Inputs

**Required:**

| Variable | What |
|---|---|
| `XRLENV_BENCHMARK_CACHE` | the benchmark cache ROOT; tasks are materialized under `<root>/swebench-pro/<instance_id>/` |
| `XRLENV_GRPC_HOST` / `_PORT` / `_TOKEN` | the cluster (not needed to build the cache) |

**Fetched for you** — both upstream inputs are public and ungated, so `build_cache.py` provisions
them itself and there is nothing to download by hand:

| Input | Where it comes from |
|---|---|
| the dataset | anonymous `snapshot_download("ScaleAI/SWE-bench_Pro")` into the shared HF cache |
| the upstream kit | shallow anonymous `git clone` of `scaleapi/SWE-bench_Pro-os`, cached once at `<cache root>/.upstream/` and reused by every entry point |

**Optional overrides** — set either to pin a local copy (an air-gapped box, or a specific snapshot):

| Variable | What |
|---|---|
| `SWEBENCH_PRO_PARQUET` | the dataset parquet, or the directory of a snapshot: `huggingface-cli download ScaleAI/SWE-bench_Pro --repo-type dataset --local-dir <dir>` |
| `SWEBENCH_PRO_HARNESS` | a checkout of the upstream kit: `git clone https://github.com/scaleapi/SWE-bench_Pro-os <dir>` (`run_scripts/`, `dockerfiles/`) |

A named location is honoured verbatim and **fails loud if wrong** — we never quietly download over
an operator's typo, because that would evaluate a different corpus than they asked for. `--parquet`
/ `--harness` do the same per-run.

Put any of these in this repo's `.env` (every entrypoint here loads it) or export them. `XRLENV_PY`
overrides the interpreter (default: this repo's `.venv`, from `uv sync --all-extras`, which
carries harbor + xrlenv + pyarrow + huggingface_hub). Optional: `DOCKERHUB_USER`/`DOCKERHUB_TOKEN`
lift the Docker Hub rate limit when the plan is regenerated with size probes.

## How to run the full gold-patch sweep

Two steps — materialize the 731 task dirs, run the oracle over them. **Nothing is ever built
and no warm-up is required**: each task's `[environment] docker_image` is a prebuilt Docker Hub
ref, pulled on first use.

```bash
# 1. the task dirs (idempotent; refreshes kit-rendered files in place when the renderer changed)
.venv/bin/python xrlenv_plugins/benchmarks/swebench_pro/build_cache.py --all

# 2. the sweep: every id must be materialized (else it refuses), then OracleAgent per task on the cluster
bash xrlenv_plugins/benchmarks/swebench_pro/run_full_sweep.sh --max-workers 16 --content-retries 1

# OPTIONAL pre-warm (operator, needs --connect-host + an operator token). Only moves the pulls
# earlier, so a wide sweep does not spend its first minutes pulling.
xrlenv build apply --plan xrlenv_plugins/benchmarks/swebench_pro/build_plan_full.yaml \
    --connect-host <admin-host> --connect-port 8080
```

`run_full_sweep.sh` rebuilds the cache itself (idempotent; `--skip-build-cache` to skip),
refuses to run unless every selected instance is materialized, and accepts `--list-green` (print
the task set and exit), `--max-workers`, `--content-retries`, `--job-id`, `--jobs-dir`; anything
else is forwarded to `run_oracle_sweep.py` (e.g. `--timeout-multiplier 1.5`). Every run gets its
own artifact dir `<jobs-dir>/<job-id>-<timestamp>/` (default jobs dir `./tmp/sanity-checks`, default
job id `swebench-pro-full-sweep`). The oracle is harbor's `OracleAgent` (`solution/solve.sh`
applies the gold patch): exit 0 iff every task rewards > 0, and an oracle FAIL is a corpus/plumbing
defect to fix or to exclude with a reason (the `EXCLUDE` list in `run_full_sweep.sh`), never a
model signal. Inspect a failure under `<jobs-dir>/<job-id>/<task>/verifier/{stdout.log,stderr.log,output.json,reward.json}`.
Results and the run log belong in `STATUS.md`.

Run knobs are flags, not env vars. `--max-workers` is trial concurrency; `--content-retries N`
re-runs the non-passing tasks up to N more times (a task is solved if ANY attempt passes), on top of
the infra-only per-trial retries inside `run_oracle_sweep.py`.

## What's here

| File | Role |
|---|---|
| `build_cache.py` | dataset parquet + upstream kit → `<cache>/swebench-pro/<instance_id>/` harbor task dirs; `--all` for the corpus (also `--smoke` = first 8 rows, `--ids-file`, `--instances`, and the partition flags documented in `scripts/`) |
| `run_full_sweep.sh` | the sweep entrypoint: build cache → green set (selection − `EXCLUDE`, every id present) → oracle sweep |
| `run_oracle_sweep.py` | the gate: OracleAgent per task on the cluster; `--retries` (infra) + `--content-retries` (per task); exit 0 iff every task rewards > 0 |
| `build_plan_full.yaml` | `type: registry` warm-up plan for the 731 images, every size registry-probed (~1071 GB compressed) |
| `STATUS.md` | the status of the full gold-patch sweep + reproduce commands |
| `scripts/` | the derived selections (filtered 478, subset-100, one-task smoke): manifests, plans, pinned sweep wrappers, generators — see [`scripts/README.md`](scripts/README.md) |
| `tests/` | offline unit tests (renderers, grade rule, selection, sampler, plan generation, pass gate, configuration consistency, kit layout, no-private-paths scan) |

```
$SWEBENCH_PRO_PARQUET (dataset)  +  $SWEBENCH_PRO_HARNESS/{run_scripts,dockerfiles}
        │ build_cache.py --all
        ▼
$XRLENV_BENCHMARK_CACHE/swebench-pro/<instance_id>/
├── task.toml                [environment] docker_image = jefzda/sweap-images:<tag>; cpus/memory by language; timeouts
├── instruction.md           problem statement + requirements + interface
├── instance.json            the dataset row (anchor)
├── environment/Dockerfile   FROM <image>
├── solution/{gold.patch,solve.sh}
└── tests/{test.sh,run_script.sh,parser.py,env.sh,f2p.json,p2p.json,grade.py}
```

## The verifier = upstream's entry script, on the working tree

Upstream (`swe_bench_pro_eval.create_entryscript`) grades a *patch file*: export the
Dockerfiles' `ENV` lines, `git reset --hard <base>`, `git apply` the patch, run the last line of
`before_repo_set_cmd` (checks the solution commit's test files out), run
`run_script.sh <selected files>` (one comma-joined argument), parse with `parser.py`, and
`resolved ⇔ FAIL_TO_PASS ∪ PASS_TO_PASS ⊆ {tests PASSED}`. `tests/test.sh` does exactly that,
with one prelude: it first captures the working tree the agent (or `solve.sh`) left in `/app`
as the submission (`git add -A && git diff --cached <base>` → `/logs/verifier/model.patch`),
so agents and the oracle are graded through the same path as upstream. Outputs:
`/logs/verifier/{model.patch,apply.log,stdout.log,stderr.log,output.json,reward.json,reward.txt}`.

One deliberate deviation from upstream's exact-equality rule, in `tests/grade.py`: a handful of
dataset `FAIL_TO_PASS` names are mangled (cut at an embedded `"`, or trailing-space drift; the
same strings appear in upstream's `helper_code/sweap_eval_full_v2.jsonl`, so upstream cannot pass
them either). A listed name also matches a parsed name equal after `rstrip()`, or — for a listed
name with an unbalanced quote — a parsed name that starts with it; every such match is recorded in
`grade_details.json.lenient_matches`, everything else stays exact. `reward.json` carries numeric
fields only (harbor parses it as `dict[str, float|int]`); the details live in `grade_details.json`.

Container sizing lives in `build_cache.py` (`RESOURCES` per language, `HEAVY_REPOS` overrides);
`--cpu-pinning` is on by default in the sweep (Go/JS suites scale their workers to nproc, and inside
a CFS-quota container nproc is the HOST core count).

## Plumbing notes (already handled by the kit)

- The images set `ENTRYPOINT ["/bin/bash"]`, which swallows the cluster's `sleep infinity`
  keep-alive as a script name and exits at once (every later exec 409s). `build_cache.py` writes the
  task marker `XRLENV_KEEPALIVE_ENTRYPOINT = "1"` into `[environment.env]`, and
  `xrlenv_plugins.harbor.environment` then starts the container with `entrypoint=["sleep"],
  command=["infinity"]` — the same override upstream's evaluator applies.
- Pro instance ids are ~100 chars, so a comma list of a few of them exceeds `NAME_MAX`: the sweep
  always hands `run_oracle_sweep.py` a tasks *file*, and nothing `stat()`s a string with a comma.
- `build_cache.py` refreshes the kit-rendered files (`grade.py`, `test.sh`, `solve.sh`) of an
  already-complete task dir whenever the renderer changed, so a grading fix propagates to a cache
  built before it without a rebuild.

# [B] SWE-bench Verified

The SWE-bench Verified plug-in lives at
`xrlenv_plugins/benchmarks/swebench_verified/` and is the canonical
way to run SWE-bench Verified on an xrlenv cluster. This page also
applies to any other harness that drives containers through
**docker-py** (`docker.from_env()`): OSWorld's docker provider,
ad-hoc evaluation scripts, or anything using
`swebench.harness.run_evaluation`.

## Integration shape: docker-py drop-in

SWE-bench ships its own harness (`swebench.harness.run_evaluation`)
that drives containers through docker-py. Onboarding onto xrlenv is
a **one-line swap** — the harness then runs **unmodified**, and every
docker-py call is rerouted to a cluster-picked node:

```diff
-import docker
-client = docker.from_env()
+import xrlenv
+client = xrlenv.from_env()
```

Connection config (`XRLENV_GRPC_HOST`, `XRLENV_GRPC_PORT`,
`XRLENV_CONSUMER_TOKEN`) lives in environment variables — the same
pattern `docker.from_env()` uses for `DOCKER_HOST`. There is no
`Client.acquire_container(...)`, no
`client.plan_image_distribution(...)`, no xrlenv-shaped pre-loop
setup.

Everything else in the harness — image probing, container creation,
`container.exec_run(...)`, `container.put_archive(...)`,
`container.get_archive(...)`, streaming exec, image management —
flows through xrlenv unchanged.

## Plug-in layout

| File | Role |
|---|---|
| `build_cache.py` | Materialize the Verified corpus into `<cache>/swebench-verified/<id>/` — `instance.json` (full row), `problem_statement.md` (the agent's prompt), `gold_patch.diff` (oracle's prediction). |
| `build_plan_gen.py` | Emit the image build plan — one `type: registry` entry per instance (`swebench/sweb.eval.x86_64.*` on Docker Hub). |
| `run_oracle_sweep.py` | Correctness gate — drive upstream `run_instance` per instance via the docker-py drop-in; exit 0 iff every instance `resolved`; writes `summary.json`. |
| `run_full_sweep.sh` | One-command entrypoint: build cache → 500-instance corpus gate → `run_oracle_sweep.py` (which owns the per-instance content-retry). `--smoke` runs the 8-instance subset. |

## The cache-backed corpus

Unlike harbor/pier benchmarks there are no `task.toml` task dirs —
the corpus is upstream's Hugging Face dataset. The plug-in
**materializes the task data** locally so runs are offline and
self-contained:

```
<cache>/swebench-verified/<instance_id>/
├── instance.json        # full upstream row (anchor)
├── problem_statement.md  # the PROMPT a real agent reads
└── gold_patch.diff       # the GOLD patch the oracle applies
```

The **image** is not cached here — it is pulled from Docker Hub on
first acquire by the node's `ImageCacheManager`; `build_plan_gen.py`
emits the registry plan.

## Oracle policy

The correctness gate submits each instance's **gold patch** as the
prediction. Upstream's own `resolved` field (from `get_eval_report`)
is read verbatim — xrlenv invents no parallel grader. Every instance
should resolve under the oracle; a non-resolving instance is a
plumbing or content bug, not a model-eval signal.

## Operator setup

```bash
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example

# 1. Materialize the Verified corpus locally.
.venv/bin/python xrlenv_plugins/benchmarks/swebench_verified/build_cache.py \
    --stage all --all

# 2. (Recommended for --all sweeps) FFD bin-pack and prefetch images.
.venv/bin/python -m xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen \
    --all --max-workers 8 \
    --output xrlenv_plugins/benchmarks/swebench_verified/swebench_verified_build_plan.yaml
xrlenv build apply \
    --plan xrlenv_plugins/benchmarks/swebench_verified/swebench_verified_build_plan.yaml \
    --connect-host <cp-host>

# 3. Boot the control plane and set connection env vars:
xrlenv up
export XRLENV_GRPC_HOST=127.0.0.1
export XRLENV_GRPC_PORT=50051
export XRLENV_CONSUMER_TOKEN=$(cat ~/.xrlenv/secrets/consumer.token)
```

For an 8-instance smoke the image prefetch step is **optional** —
reactive image-affinity scheduling handles distribution naturally.
For batch sweeps (~500 instances) it is recommended so first-acquire
pulls do not queue serially.

## Running the gate

```bash
# Full green-set sweep via the one-command entrypoint:
bash xrlenv_plugins/benchmarks/swebench_verified/run_full_sweep.sh

# Concurrent sweep (8 workers):
bash xrlenv_plugins/benchmarks/swebench_verified/run_full_sweep.sh \
    --max-workers 8

# Smoke subset (8 instances) directly through run_oracle_sweep.py:
.venv/bin/python xrlenv_plugins/benchmarks/swebench_verified/run_oracle_sweep.py \
    --smoke

# Baseline against local Docker (no cluster):
.venv/bin/python xrlenv_plugins/benchmarks/swebench_verified/run_oracle_sweep.py \
    --smoke --local
```

## Retry layers

Two retry layers absorb different failure modes:

| Layer | Granularity | Retries on | Purpose |
|---|---|---|---|
| `--retries` (6 in the gate) | per-trial, inside `run_oracle_sweep.py` | infra-transient exceptions only (`CapacityExhausted` / node loss) | Absorb capacity pacing; cannot mask a flaky instance. |
| `--content-retries` (2) | per-instance, inside `run_oracle_sweep.py` | a non-`resolved` outcome | Catch a one-off environmental flake vs a real regression. |

Content failures are never re-rolled by the infra layer.
Timeouts run at SWE-bench's native 1800 s.

## Image distribution

When the chosen cluster node does not yet have a requested image, its
`ImageCacheManager.ensure_present(image)` pulls from Docker Hub's
`swebench/` namespace on first acquire. Subsequent acquires for the
same image prefer nodes that already hold it; the LRU cache protects
in-use images from eviction.

## What's wired

The xrlenv docker-py drop-in covers the manager-level surface that
SWE-bench-shaped harnesses use:

- `client.containers.create(...)` / `containers.run(...)`
- `container.exec_run(...)` (batched + streaming via `stream=True`)
- `container.put_archive(...)` / `container.get_archive(...)`
- `container.start()` / `container.stop()` / `container.remove()`
- `client.images.get(...)` / `images.pull(...)` / `images.list(...)` /
  `images.remove(...)` / `image.history()`

Methods not yet wired raise `NotImplementedError` with a clear
message rather than failing on uninitialised state. Full coverage
list lives in `xrlenv/compat/docker_client.py:_CLUSTER_OVERRIDES`.

## Side pointer

If you're writing your own harness from scratch (not using
swebench's `run_evaluation` and not docker-py-shaped), the direct
xrlenv API is usually a better fit than the drop-in — see
{doc}`/build_with_xrlenv/work_with_xrlenv_managed_containers/index`.

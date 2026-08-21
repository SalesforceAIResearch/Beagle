# SWE-bench Verified onboarding

End-to-end runnable smoke that drives the **upstream** swebench
harness against either the local Docker daemon or an xrlenv
cluster. The only difference between modes is which docker client
the harness receives.

## The drop-in promise

The audience's harness contains **exactly one** xrlenv-specific
line:

```diff
-import docker
-client = docker.from_env()
+import xrlenv
+client = xrlenv.from_env()
```

Connection config (`XRLENV_GRPC_HOST`, `XRLENV_GRPC_PORT`,
`XRLENV_CONSUMER_TOKEN`) lives in environment variables that the
operator sets at deploy time — the same pattern `docker.from_env()`
uses for `DOCKER_HOST`. **No** `Client.acquire_container(...)`,
**no** `client.plan_image_distribution(...)`, **no** xrlenv-shaped
pre-loop setup. That's the contract.

The smoke driver in `smoke.py` itself contains literally one
xrlenv line: `client = xrlenv.from_env()` (when not in `--local`
mode). Everything else is upstream `swebench.harness`.

## How it works

```text
                    ┌────────────────────────┐
                    │ swebench.run_instance  │  unmodified
                    └───────────┬────────────┘
                                │ client.containers.create / .exec_run /
                                │ .put_archive / ...
                                ▼
   ┌──────────────────────────────────────────────────────────┐
   │  --local            │  default (cluster)                 │
   │  docker.from_env()  │  xrlenv.from_env()                 │
   │                     │  reads XRLENV_GRPC_HOST etc.       │
   └─────────┬──────────────────────────┬────────────────────┘
             │                          │
             ▼                          ▼
   local Docker daemon       xrlenv control plane → scheduler
                              (image-affinity, preferred_home) →
                              chosen node's ImageCacheManager
                              (pull or build on miss) →
                              docker on that node
```

When the chosen cluster node doesn't have an image yet, its
`ImageCacheManager.ensure_present(image)` pulls it from the
registry (for swebench, Docker Hub's `swebench/` namespace) on
first acquire — the same primitive case-1 sandboxes use, with the
same LRU eviction policy. Subsequent acquires for the same image
get steered back to that node by the image-affinity score.

## Operator setup

Per cluster bring-up, in order:

```bash
# 1. Boot the control plane (one-shot; idempotent):
xrlenv up

# 2. (Optional, recommended for --all sweeps) FFD bin-pack the
#    cluster's image bytes + eager prefetch:
xrlenv images plan \
    --refs examples/benchmarks-onboarding/swebench-verified/refs/smoke-8.txt \
    --eager-prefetch
# Or for the full Verified set (after generating
# refs/all-verified.txt — see refs/all-verified.txt.README.md):
xrlenv images plan \
    --refs examples/benchmarks-onboarding/swebench-verified/refs/all-verified.txt \
    --eager-prefetch

# 3. Set env vars in the consumer's shell (typically in the
#    operator's deploy script alongside `xrlenv up`):
export XRLENV_GRPC_HOST=127.0.0.1
export XRLENV_GRPC_PORT=50051
export XRLENV_CONSUMER_TOKEN=$(cat ~/.xrlenv/secrets/consumer.token)
```

For an 8-instance smoke the `xrlenv images plan` step is
**optional** — reactive image-affinity scheduling handles
distribution naturally. For batch sweeps (`--all`, ~500
instances), it's recommended so first-acquire pulls don't queue
serially.

## Running the smoke

```bash
# Local baseline (single host; swebench pulls images on demand
# from Docker Hub the first time):
.venv/bin/python examples/benchmarks-onboarding/swebench-verified/smoke.py \
    --local

# Cluster 8-instance smoke (env-var-driven; no xrlenv flags;
# default --max-workers=1 = serial):
.venv/bin/python examples/benchmarks-onboarding/swebench-verified/smoke.py

# Cluster 8-instance smoke, 4-way concurrent (only safe for
# thread-safe harnesses; swebench is — see "Concurrency" below):
.venv/bin/python examples/benchmarks-onboarding/swebench-verified/smoke.py \
    --max-workers 4

# Keep the harness's per-instance artifacts under <repo>/tmp/<job-id>/:
.venv/bin/python examples/benchmarks-onboarding/swebench-verified/smoke.py \
    --max-workers 4 --save-artifacts

# Custom instance list:
.venv/bin/python examples/benchmarks-onboarding/swebench-verified/smoke.py \
    --instances django__django-11099,sympy__sympy-13615

# Full Verified sweep, concurrent + archive artifacts:
.venv/bin/python examples/benchmarks-onboarding/swebench-verified/smoke.py \
    --all --max-workers 8 --save-artifacts
```

## Concurrency is the operator's choice

`--max-workers` defaults to **1 (serial)**. The smoke runs
multiple instances concurrently only when the operator opts in
explicitly. The xrlenv docker-py drop-in is **concurrency-neutral
by construction** — it shares no mutable state across calls, so
the operator picks whatever model fits the harness:

| Model | When | How |
|---|---|---|
| **Serial** (`--max-workers 1`, default) | Any harness; safe baseline | Built in. |
| **ThreadPoolExecutor** (`--max-workers N>1`) | Thread-safe harnesses (swebench is) | Built in. |
| **Multiprocessing** | Harnesses with thread-unsafe internals (e.g. OSWorld's in-container subprocess management) | Operator wraps the smoke externally — drive N instances per process, each with `--instances <subset>`. |
| **Asyncio** | Event-loop-shaped harnesses | Operator writes their own driver around `_run_one_instance` (importable). |

The drop-in ships no `threading.Lock`, no shared counter, no
default executor — concurrency is a driver-level policy decision,
not an xrlenv-core invariant.

## Artifact archiving

Without `--save-artifacts`, the smoke runs swebench in a tempdir
that's reaped at exit — nothing leaks into the working tree.
Pass `--save-artifacts` to keep the harness's native artifact
tree (`logs/run_evaluation/<run_id>/<model>/<instance>/`) under
`<repo>/tmp/<job-id>/`:

```bash
# Default: <repo>/tmp/smoke-YYYYMMDD-HHMMSS/
.venv/bin/python examples/benchmarks-onboarding/swebench-verified/smoke.py \
    --save-artifacts

# Explicit path:
.venv/bin/python examples/benchmarks-onboarding/swebench-verified/smoke.py \
    --save-artifacts ~/scratch/jobs

# Custom job id (useful for tagging model+version runs):
.venv/bin/python examples/benchmarks-onboarding/swebench-verified/smoke.py \
    --save-artifacts \
    --job-id claude-opus-4-7-50-v1
```

Layout matches `tests/smoke/test_swebench_drop_in.py`'s
`--save-artifacts` shape, so an operator navigating either
smoke's output finds the same tree:

```
<save-artifacts>/<job-id>/
├── logs/run_evaluation/<run-id>/<model>/<instance-id>/
│   ├── run_instance.log
│   ├── report.json
│   ├── test_output.txt
│   └── patch.diff
└── summary-<UTC-timestamp>.json    # this smoke's per-run tally
```

`<repo>/tmp/` is gitignored so artifacts never leak into commits.

## What the smoke does

For each instance:

1. Loads the upstream Verified row (instance metadata + gold patch).
2. Calls `swebench.harness.run_evaluation.make_test_spec(...)` with
   `namespace="swebench"` (flips `is_remote_image=True`, so the
   harness never tries to build images locally).
3. Builds a prediction whose `model_patch` is the dataset's
   **gold patch** — an oracle policy, not a model.
4. Calls upstream `run_instance(test_spec, pred, ..., client, ...)`.
   The harness:
   - probes image presence (in cluster mode the drop-in returns
     synthetic-positive; the pull happens at acquire time on the
     chosen node)
   - creates a container
   - copies the patch in via `put_archive`
   - runs `git apply` via `exec_run`
   - runs the per-repo `eval.sh` test command
   - parses the output via upstream's `get_eval_report`
   - writes `report.json` with `resolved: True/False`
5. Aggregates: every gold-patch-as-prediction should resolve. A
   non-resolved instance under the oracle policy is a plumbing
   bug, not a model-eval signal.

## Adapting to your own benchmark

If your benchmark already uses docker-py, the integration is just
the one-line swap above. The full set of docker-py methods the
xrlenv drop-in supports in cluster mode is documented in
`xrlenv/compat/docker_client.py:_CLUSTER_OVERRIDES` — the
manager-level surface (`client.containers.run`,
`container.exec_run`, `container.put_archive`,
`container.get_archive`, `container.remove`, streaming exec) is
wired for SWE-bench-shaped harnesses. Methods not yet wired raise
`NotImplementedError` with a clear message rather than failing on
uninitialised state.

For closed-set batch workloads you can ship a ref list with your
benchmark (like `refs/smoke-8.txt` here) and document that
operators run `xrlenv images plan --refs <your-file>` once at
cluster bring-up. For streaming / dynamic workloads the reactive
image-affinity scheduler handles distribution without any
operator pre-step — `containers.create(image=X)` "just works."

If your benchmark is **step-driven** (an `act → obs` state machine
the trainer drives) rather than docker-py-using, see the sibling
[`terminal-bench-2/`](../terminal-bench-2/) example.

## Pre-pull script (debug aid only)

`scripts/pre-pull-images.sh` is kept in the repo as a debug aid:

> "Verify Docker Hub auth + per-node disk + connectivity baseline
> before troubleshooting cluster-mode acquires."

It manually pulls the 8 smoke images on the local node — useful
when isolating "is this an xrlenv issue or a Docker Hub auth
issue?". The smoke itself does NOT require it; the cluster pulls
images automatically through `ensure_present`.

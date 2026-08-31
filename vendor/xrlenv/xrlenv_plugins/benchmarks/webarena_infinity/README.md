# WebArena-Infinity — xrlenv onboarding

xrlenv's onboarding of
[web-arena-x/webarena-infinity](https://github.com/web-arena-x/webarena-infinity)
(WAI). This directory ships **both halves** of running the benchmark on the
xrlenv cluster, versioned together so it's self-contained:

1. **The substrate image** — an *answer-free* env image (`Dockerfile` +
   `build_plan.yaml`) built and pushed to the cluster's private registry.
2. **The integration scripts** — three `*.py` files that drive the WAI eval
   against xrlenv containers instead of local app-server ports.

> **The `*.py` scripts run from inside the WAI checkout, not from xrlenv.** They
> import WAI's own `evaluation/` modules (`agents`, `run_eval_parallel`,
> `tasks`), so they live in **`copy_to_call_site/`** — you copy them into the WAI
> repo (step 2) and run them there. The copies here are the canonical source of
> truth.

## What's here (outlier layout)

WAI is an **outlier** benchmark (GUIDELINE §7.2, *runner-shim*): the eval runs from
inside the WAI checkout (the "call site"). The directory is normalized to the canonical
phases anyway, with a `copy_to_call_site/` payload for the call-site half:

| File | Runs on | Canonical phase / role |
|---|---|---|
| `Dockerfile` | build host | **build the cache** — multi-stage build of the ONE answer-free substrate image (WAI has no per-task cache). |
| `build_plan.yaml` | build host | **image plan** — one-entry `type: local` plan for `deploy/registry/build_and_push_images.py` (hand-written; `build_plan.calibrated.yaml` carries the cluster-measured size). |
| `README.md` · `STATUS.md` | — | canonical docs. |
| `copy_to_call_site/run_full_sweep.sh` | host (call site) | **the sweep entrypoint** — runs the **official 10-app real-tasks set** (`--model oracle`; `APPS=all` for the full 13) from the WAI checkout, looping the per-app runner (build+push is the separate prep step 1). |
| `copy_to_call_site/run_eval_parallel_xrlenv.py` | host (call site) | **run the oracle sweep** — orchestrator, same CLI + output as WAI's `run_eval_parallel.py`; each worker drives an xrlenv container. |
| `copy_to_call_site/xrlenv_config.py` | host (call site) | cluster coordinates + credentials (reads the WAI repo-root `.env`); holds `IMAGE_REF`, the substrate channel tag. |
| `copy_to_call_site/xrlenv_runner.py` | in container | injected + invoked per container by the orchestrator; you never run it by hand. |

There is **no** top-level `build_cache.py` / `build_plan_gen.py` (nothing per-task to
download or generate — the Dockerfile *is* the cache, and the one-entry plan is
hand-written) and **no** `patches/` (the corpus fix is the verifier-strip in the
Dockerfile). This is documented, not stubbed.

## How it works

A WAI container has **no reachable port** — the app server, browser, browser-use
agent, and verifier all run *inside* it (everything is `localhost` there). The
host process only orchestrates: it acquires containers, injects the scripts, runs
the agent, then injects the answer and runs the verifier, and pulls results back.
The answer is never baked into the image and lives in the container only for the
verifier step, so the substrate stays answer-free at runtime (audit H1/D6). See
[Per-run lifecycle](#per-run-lifecycle) for the full sequence.

## Prerequisites

- **`xrlenv` importable.** `xrlenv_config.py` imports `xrlenv.Client` — either
  `pip install` xrlenv into the WAI venv, or set `XRLENV_REPO=/path/to/xrlenv`.
- **`.env` at the WAI repo root** — read once (with `override=True`) by
  `xrlenv_config.py`:

  ```dotenv
  # cluster coordinates
  XRLENV_GRPC_HOST=<control-plane-host>      # your control-plane host (see slurm_scripts/clusters.yaml)
  XRLENV_GRPC_PORT=50051
  XRLENV_CONSUMER_TOKEN=<consumer-token>     # issue with: xrlenv tokens issue consumer
  XRLENV_PRIVATE_REGISTRY_HOST=<private-registry-host>
  XRLENV_PRIVATE_REGISTRY_PORT=5011
  # LLM keys — forwarded into each container
  OPENAI_API_KEY=...
  GOOGLE_API_KEY=...        # or GEMINI_API_KEY
  ANTHROPIC_API_KEY=...
  ```

- **Build host trusts the private registry** — needed only for step 1. The HTTP
  private registry requires `insecure-registries`, which every bootstrapped
  cluster node already has. See
  [`private_registry.md`](../../../docs/deploy/multi_node_deployment/private_registry.md).

## Run an eval, end to end

### 1. Build & push the substrate image

```bash
source .env
.venv/bin/python deploy/registry/build_and_push_images.py \
    --plan xrlenv_plugins/benchmarks/webarena_infinity/build_plan.yaml \
    --registry "${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}" \
    --force
```

Pushes `<registry>/xrlenv-webarena-infinity/substrate:dev`. The tool HEADs the
registry manifest and skips a ref already present, so re-pushing the **same
`:dev` channel tag requires `--force`** (`--dry-run` prints the plan without
building). After a node has materialized `:dev`, refresh the plan's size hint
with `xrlenv build calibrate` (writes `build_plan.calibrated.yaml`). What's in
the image, and the channel-tag model, are covered in
[The substrate image](#the-substrate-image).

### 2. Install the integration scripts into the WAI repo

Copy the `copy_to_call_site/` payload (the three `*.py` + `run_full_sweep.sh`) into the
WAI checkout's `evaluation/` directory (where the scripts can import `agents` /
`run_eval_parallel` / `tasks`):

```bash
cp /path/to/xrlenv-dev/xrlenv_plugins/benchmarks/webarena_infinity/copy_to_call_site/* \
   <path-to-webarena-infinity>/evaluation/
```

Re-copy to refresh — these files are the source of truth.

### 3. Run the sweep from the WAI repo

The canonical entrypoint is **`evaluation/run_full_sweep.sh`**, run from the WAI checkout.
By **default it runs the OFFICIAL 10-app real-tasks set** (the paper figure, ~1260 tasks —
the exact apps in coding-bench `configs/test_wai_monet_full_real-tasks.yaml`) under
`--model oracle`: the browserless answer-injection gate (no LLM/creds) that validates the
pipeline + every task's verifier end-to-end.

```bash
cd <path-to-webarena-infinity>
bash evaluation/run_full_sweep.sh                        # official 10 apps, real-tasks, oracle, 8 workers
APPS=all bash evaluation/run_full_sweep.sh               # the full 13-app HF corpus (1620 tasks)
```

Env knobs (script header): `MODEL` (default `oracle`; `gpt`/`gemini-pro`/… for a real
agent — needs LLM keys in `.env`), `TASK_SUITE` (default `real-tasks`; `function-tasks`/`all`),
`WORKERS`, `APPS` (`all`, or an explicit list), `WEB_APP` (one app). E.g. one app, real agent:

```bash
MODEL=gemini-pro WEB_APP=apps/gmail WORKERS=8 bash evaluation/run_full_sweep.sh
```

**Scheduling.** The runner takes one `--web-app`, so the script loops the apps
**sequentially** (parallel *within* an app via `--workers`; workers idle at app boundaries).
That's fine for the oracle gate.

The underlying per-app runner (what the script loops) is `run_eval_parallel_xrlenv.py` —
same CLI + output layout as WAI's own `run_eval_parallel.py`, plus the xrlenv coordinates;
call it directly for a single task / an explicit image / control-plane overrides:

```bash
.venv/bin/python evaluation/run_eval_parallel_xrlenv.py --model gpt --task-id task_e1 \
    --workers 1 --web-app apps/gmail \
    --image <private-registry-host>:5011/xrlenv-webarena-infinity/substrate:dev \
    --xrlenv-host <control-plane-host> --xrlenv-port 50051
```

`--workers N` = N containers in flight cluster-wide. Output (`results.json` + `report.html`,
multi-run merge, resume) is identical to WAI's local runner.

## Reference

### The substrate image

Pinned to WAI commit `1ca77813` via `WEBARENA_REF` in the `Dockerfile`
(`build_plan.yaml` carries no build-args, so the Dockerfile default *is* the
pin). It's tagged **`:dev`** — a stable distribution *channel* tag (prod uses
`:stable`), decoupled from the source commit: a rebuild re-pushes the same `:dev`
and the control plane resolves `:dev` → the current registry digest at acquire,
so the re-push reaches nodes without minting a new tag.

- **Base** — `public.ecr.aws/docker/library/python:3.12-slim` (ECR Public mirror
  of the Docker Official image; Docker Hub's anonymous pulls 429 on the cluster).
  Override with `--build-arg BASE_IMAGE=...`.
- **Agent stack** — `browser-use==0.11.9` (+ `cdp-use`, `requests`, `pillow`,
  `rich`, `google-genai`, `openai`), plus Node (the oracle runs the frontend's
  `data.js` through Node to generate seed state).
- **Browser baked** — `playwright install --with-deps chromium`; browser-use and
  the vision agents launch a Playwright-managed Chromium, so the in-container
  agent can't run without it.
- **Answer-free** — the default (last) `substrate` stage strips every task
  verifier (`apps/*/real-tasks/`, `apps/*/function-tasks/`) and oracle solver
  (`apps/*/sanity_check_*.py`), and the build fails if any survive. The `full`
  stage (answers present) is an intermediate build dependency only; nothing
  pushes it.

### Per-run lifecycle

The host process orchestrates; everything task-facing runs inside the container:

1. Acquire one container per worker (reused across many tasks).
2. Inject `evaluation/` (incl. `xrlenv_runner.py`) once per container.
3. Start the app server inside the container once (backgrounded, persistent).
4. Per task: run the agent (phase A) → inject the verifier + answer (only now) →
   run the verifier (phase B) → delete the answer → pull artifacts back.
5. Aggregate + report on the host (shared helpers from `run_eval_parallel.py`).

The answer is injected only *after* the agent exits and removed before the next
task, so the agent is never co-resident with an answer file — the substrate stays
answer-free at runtime (audit H1/D6).

### Freshness & deployment note

These scripts use the xrlenv **raw-container** acquire path. On a cluster whose
control plane runs the tag→digest resolver, `:dev` is resolved to the current
digest per acquire, so rebuilds propagate automatically. On a cluster still on
older code, `:dev` is a plain mutable tag — see the Sphinx page *Supported
benchmarks → WebArena-Infinity* for the freshness model and the deployment
caveat.

## See also

- `docs/supported_benchmarks_and_harnesses/webarena_infinity.md` — the Sphinx user page
  (channel-tag scheme, rebuild workflow, downstream consumer config, freshness model).
- `GUIDELINE_onboard_benchmarks.md` §7.2 — the **runner-shim** pattern this benchmark uses
  (answer-free substrate + in-container runner; no host-reachable service).
- `docs/deploy/multi_node_deployment/private_registry.md` — the private registry the
  substrate image is pushed to.
- [`STATUS.md`](STATUS.md) — current onboarding disposition + reproduce command.

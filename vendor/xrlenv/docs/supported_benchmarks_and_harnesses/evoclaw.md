# EvoClaw

[EvoClaw](https://github.com/EvoClaw-Bench/EvoClaw) is a software-evolution
benchmark where agents solve milestone-by-milestone code changes on real
open-source repositories. Each milestone is graded by EvoClaw's own evaluator
against the milestone image's test suite.

**XRLEnv's role is narrow:** manage EvoClaw's Docker containers and images on
the cluster. EvoClaw's own orchestrator, milestone DAG, agents, and evaluator
run unchanged.

## Why EvoClaw needs a different integration shape

Most benchmarks that speak docker-py can use the {doc}`xrlenv.from_env()
drop-in <../build_with_xrlenv/work_with_xrlenv_managed_containers/docker_py_dropin>`.
EvoClaw cannot: its harness shells out to the `docker` binary as raw
subprocesses (`subprocess.run(["docker", "exec", …])`,
`subprocess.Popen(["docker", "run", …])`) throughout
`container_setup.py`, `agent_runner.py`, `orchestrator.py`, and
`test_runner/core/docker.py`. No docker-py swap touches those call sites.

A PATH-level `docker` shim would also fail: the xrlenv compat client maps
cluster sessions by container ID, and a fresh process per `docker` call
would not see the sessions opened by earlier `docker run` calls.

The solution is an **in-process subprocess interceptor** (`docker_shim.py`)
that monkeypatches `subprocess.run`, `subprocess.Popen`, and related helpers
so that any argv beginning with `docker` is routed to the cluster — and every
other subprocess call passes through unchanged. Because everything runs in the
same Python process, the interceptor shares the live session map with every
subsequent `docker exec`, `docker cp`, and `docker rm`.

See `xrlenv_plugins/benchmarks/evoclaw/DESIGN.md` for the full design
rationale and `SHIM-SURFACE.md` for the exact docker subcommand and flag
coverage.

## Why `run_e2e`, not `run_all`

EvoClaw's `scripts/run_all.py` is a multi-repo orchestrator that spawns each
`run_e2e` call as a detached subprocess
(`subprocess.Popen(…, start_new_session=True)`). A monkeypatch installed in
the `run_all` process would not be inherited by those child interpreters.

The interceptor must therefore live in the `run_e2e` process. The xrlenv
wrapper `run_e2e_xrlenv.py` is the in-process equivalent of one `run_all`
worker: it installs the shim, registers the `oracle`/`noop` agents, and then
calls EvoClaw's own `harness.e2e.run_e2e.main()` unchanged.

## Setup

### Prerequisites

1. **Set up EvoClaw first** — follow EvoClaw's `README.md` §Setup steps 0–2
   (git clone + `uv sync` for the venv, `git lfs pull` for EvoClaw-data).
   Note the absolute path of the data directory.

2. **Copy the onboarding modules** into the EvoClaw checkout:

   ```bash
   mkdir -p ~/Github/EvoClaw-dev/xrlenv_onboard
   cp /path/to/xrlenv/xrlenv_plugins/benchmarks/evoclaw/copy_to_call_site/*.py \
      /path/to/xrlenv/xrlenv_plugins/benchmarks/evoclaw/run_oracle_smoke.sh \
      ~/Github/EvoClaw-dev/xrlenv_onboard/
   ```

3. **Install `xrlenv` into EvoClaw's venv:**

   ```bash
   cd ~/Github/EvoClaw-dev
   uv pip install -e /path/to/xrlenv
   ```

### Environment variables

EvoClaw uses `.env` (committed template) and `.env_private` (gitignored, your
real values) at the checkout root. Because the wrapper launches `run_e2e`
directly — bypassing `run_all.py`'s auto-load — it replicates EvoClaw's own
precedence (`shell > .env_private > .env`). Put all real values in
`.env_private`:

```bash
# ~/Github/EvoClaw-dev/.env_private   (gitignored — real values here)

# EvoClaw's own (configure per EvoClaw's docs):
EVOCLAW_DATA_ROOT=/home/you/EvoClaw-data   # REQUIRED
# EVOCLAW_WHEELHOUSE_DIR=...     # only for quarantine repos (e.g. scikit-learn)
# UNIFIED_API_KEY=...            # model auth for real-LLM runs
# UNIFIED_BASE_URL=...

# xrlenv cluster coordinates:
XRLENV_GRPC_HOST=internal-ip
XRLENV_GRPC_PORT=50051
XRLENV_CONSUMER_TOKEN=<token from xrlenv tokens issue consumer>

# Optional xrlenv knobs (see table below):
# XRLENV_IMAGE_REGISTRY=mirror.example.com:5000
# XRLENV_GROUP_ID=sweep-2026-07-04
```

#### EvoClaw-scoped xrlenv variables

| Variable | Required | Description |
|---|---|---|
| `XRLENV_GRPC_HOST` | yes | Control-plane host. |
| `XRLENV_GRPC_PORT` | no (default `50051`) | Control-plane gRPC port. |
| `XRLENV_CONSUMER_TOKEN` | when auth is enabled | Bearer token from `xrlenv tokens issue consumer`. |
| `XRLENV_IMAGE_REGISTRY` | no | Explicit Docker Hub mirror-host prefix (e.g. `mirror:5000`). When set, milestone image refs are formed as `<registry>/hyd2apse/<short>:<milestone>-<tag>`. When absent, nodes route `hyd2apse/*` pulls through the registry mirror configured in `registry-mirrors`. |
| `XRLENV_GROUP_ID` | no | Group tag injected on every `docker run` acquire. Use this to label concurrent sweeps so the admin panel can filter them and `cancel_group` can stop them together. |

The variables `EVOCLAW_IMAGE_TAG` (default `v0.9`) and `DOCKERHUB_ORG`
(default `hyd2apse`) are **EvoClaw's own** and are read from EvoClaw's
`image_version.py` — configure them per EvoClaw's docs, not here.

## Image resolution

EvoClaw's `scripts/pull_images.sh` retags milestone images from their
Docker Hub names (`hyd2apse/<short>:<milestone>-<v>`) to local multi-level
names (`<repo_full>/<milestone>:<v>`) that exist in no registry. The cluster
cannot pull local retags.

`image_resolution.py` monkeypatches `harness.e2e.image_version.resolve_image`
to return the original Docker Hub ref instead. The `short ↔ full` repository
map is **parsed from EvoClaw's own `pull_images.sh`** (its authoritative
source) rather than hardcoded — EvoClaw's short names are not derivable from
the full names (e.g. `apache_dubbo_…` → `dubbo`). The cluster's
`registry-mirrors` configuration routes `docker.io` (`hyd2apse/*`) pulls
through the pull-through mirror transparently.

The base image (the agent container) is passed directly as a pullable
`--image hyd2apse/<short>:base-<v>` ref, or derived automatically from
`EVOCLAW_DATA_ROOT` + `--repo-name`.

## Running the oracle smoke (acceptance gate)

The oracle smoke is the validation path: a host-injected golden solution
(the milestone `END` source) is applied through EvoClaw's own full
pipeline — `ContainerSetup` → `AgentRunner` streaming `docker exec` →
watcher → `PatchEvaluator`. A `resolved` result proves the entire
container path works on the cluster; only a real LLM agent remains
untested.

The agent container is answer-free by design — EvoClaw's `ContainerSetup`
deletes all tags and runs `git gc --prune=now --aggressive`, so the END
commit objects are gone from the container. The golden source survives
only as the `milestone-<mid>-end` tag inside the milestone image. The
oracle therefore extracts it from the host side (via `oracle_solution.py`)
before the run and injects it into the agent container via the shim's
bind-mount → `put_archive` translation.

```bash
cd ~/Github/EvoClaw-dev

# True oracle — applies the golden solution and expects `resolved`:
.venv/bin/python xrlenv_onboard/run_e2e_xrlenv.py \
  --agent oracle --model none --milestones 1 --force \
  --repo-name navidrome_navidrome_v0.57.0_v0.58.0
```

`--workspace-root`, `--srs-root`, and `--image` are derived automatically
from `EVOCLAW_DATA_ROOT` + `--repo-name`. Pass them explicitly to override.
The first run is slow — multi-GB base and milestone images are pulled and
cached on first acquire.

### Fast component checks

```bash
# Evaluator only (no agent pipeline) — checks out END in the milestone image
# and runs EvoClaw's test suite:
./xrlenv_onboard/run_oracle_smoke.sh navidrome

# Full pipeline, empty submission — proves the shim carries the agent
# surface; grades unresolved:
.venv/bin/python xrlenv_onboard/run_e2e_xrlenv.py \
  --agent noop --model none --milestones 1 --force \
  --repo-name navidrome_navidrome_v0.57.0_v0.58.0
```

### Real LLM agent

Use the same wrapper with a real `--agent` value (`claude-code`, `codex`,
`gemini-cli`, `openhands`) and the model auth vars
(`UNIFIED_API_KEY` / `UNIFIED_BASE_URL`) in `.env_private`. XRLEnv only
manages the containers — the agent itself runs inside them via EvoClaw's
standard `docker exec` path, now routed through the cluster.

## How the shim routes docker calls

The interceptor covers the docker surface that `run_e2e` exercises:

| docker call | Cluster mapping |
|---|---|
| `run -d --name N … IMG CMD` | `acquire_container(image, CMD, name=prefixed(N), …)` — records `N → container_id` |
| `exec [-w DIR] [-e K=V] N CMD` | `session.exec_run(CMD, workdir=DIR, environment=…, stream=…)` |
| `cp SRC N:DST` / `N:SRC DST` | `put_archive` / `get_archive` |
| `rm -f N` / `stop N` | `session.destroy()` + drop from name registry |
| `inspect -f {{.State.Running}} N` | answered from the in-process name registry |
| `ps --filter name=^N$` | answered from the in-process name registry |

**Bind mounts → `put_archive`:** EvoClaw passes `-v host_path:/ctr_path:ro`
for the oracle's golden directory and for agent credential files. The shim
resolves the host path on the consumer machine and uploads its contents via
`put_archive` so the host files reach the cluster container.

**Name namespacing:** EvoClaw uses a fixed `--name` per repo/milestone.
Under concurrent rollouts, unprefixed names collide. The interceptor
automatically prefixes every container name with `$EVOCLAW_RUN_PREFIX`
(defaults to `xrl-<pid>-`). The EvoClaw harness code is untouched.

**Labels:** `xrlenv.task_key` and (if `XRLENV_GROUP_ID` is set)
`xrlenv.group_id` are injected on every `run` so the admin panel and
`cancel_group` work without EvoClaw knowing.

## Grading

Grading is entirely EvoClaw's responsibility. The canonical metric is
`resolved` (the milestone's test suite passes). XRLEnv does not reimplement
EvoClaw's `convert_to_summary`, `report_parser`, or per-language grading
helpers — it only manages containers.

## Module layout

All files live in `xrlenv_plugins/benchmarks/evoclaw/` and are copied into
the EvoClaw checkout as `xrlenv_onboard/`:

| File | Role |
|---|---|
| `run_e2e_xrlenv.py` | Entry point — installs shim + image override, registers `oracle`/`noop`, calls EvoClaw's `run_e2e.main()` |
| `docker_shim.py` | Subprocess interceptor — argv routing, name → container ID registry, streaming wrap, bind-mount → `put_archive` |
| `image_resolution.py` | Monkeypatches `resolve_image` to return pullable Docker Hub refs |
| `oracle_agent.py` | The `oracle` agent framework — applies the host-injected golden source, commits, tags `agent-impl-<mid>` |
| `oracle_solution.py` | Host-side golden extraction — `git archive milestone-<mid>-end` from the milestone image |
| `noop_agent.py` | The `noop` agent framework — empty submission, grades unresolved |
| `oracle_eval_smoke.py` + `run_oracle_smoke.sh` | Fast evaluator-only check (no agent pipeline) |

## Known limitations

- **Quarantine not enforced in the wrapper.** EvoClaw's `load_quarantine_env()`
  (called by `run_all.py`) is not replicated in `run_e2e_xrlenv.py`. This
  affects only `scikit-learn` (the one repo with a quarantine policy) in
  real-LLM runs. The oracle is unaffected — there is nothing to quarantine
  when applying a host-injected solution. Replicating quarantine in the
  wrapper is a follow-up.
- **Golden covers `repo_src_dirs` only.** Root build files (e.g. `go.mod`)
  are not included. Harden for milestones that need them.
- **No preemption/resume.** EvoClaw's `resume.py` expects a long-lived
  container addressable by name across process restarts; xrlenv sessions
  are lifetime-capped and liveness-reaped. Resumable sessions require
  spec 18 (phase 3).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `get_agent_framework` / registration error | You are running `run_e2e` directly, not through the wrapper. Use `xrlenv_onboard/run_e2e_xrlenv.py`. |
| `XRLENV_GRPC_HOST is required` | `.env_private` not found or missing the cluster vars. Run from the EvoClaw checkout root, or set `EVOCLAW_SOURCE_ROOT`. |
| `run_e2e` demands `UNIFIED_API_KEY` for the oracle | Set a dummy value in `.env_private`: `UNIFIED_API_KEY=none`. |
| Acquire fails on a `docker run` kwarg | A flag in EvoClaw's `docker run` call is not yet mapped. Check `SHIM-SURFACE.md` and `docker_shim._run`. |

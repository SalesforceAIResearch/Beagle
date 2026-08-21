# terminal-bench-2 onboarding

End-to-end runnable smoke that drives the **upstream** `harbor`
runner against either the local Docker daemon or an xrlenv cluster.
The only difference between modes is which `import_path` the
harbor `EnvironmentConfig` points at.

## The drop-in promise

The audience's harbor `job.yaml` contains **exactly one**
xrlenv-specific line:

```diff
 environment:
-  import_path: harbor.environments.docker.docker:DockerEnvironment
+  import_path: xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster
```

Connection config (`XRLENV_GRPC_HOST`, `XRLENV_GRPC_PORT`,
`XRLENV_CONSUMER_TOKEN`) lives in environment variables that the
operator sets at deploy time — symmetric with the docker-py
drop-in's `xrlenv.from_env()`. **No** `Client.acquire_container(...)`,
**no** xrlenv-shaped pre-loop setup. That's the contract.

The smoke driver in `smoke.py` does the same `import_path` swap
programmatically (so a fresh checkout's `.venv/bin/python smoke.py`
runs without anyone editing YAML), but the underlying shape is
identical to what an end-user writes.

## How it works

```text
                  ┌────────────────────────┐
                  │ harbor.Job.run()       │  unmodified
                  └───────────┬────────────┘
                              │ environment.start / .exec /
                              │ .upload_dir / .download_dir / .stop
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │  --local                       │  default (cluster)      │
   │  XrlenvHarborEnvironment       │  XrlenvHarborEnviron-   │
   │  (= harbor's stock             │  mentCluster            │
   │   DockerEnvironment with       │  reads XRLENV_GRPC_HOST │
   │   xrlenv kwargs recorded for   │  + acquires a container │
   │   observability)               │  via xrlenv.Client      │
   └─────────┬───────────────────────────────┬────────────────┘
             │                               │
             ▼                               ▼
   local Docker daemon                xrlenv control plane → scheduler
   (docker compose up + cp)           (image-affinity) →
                                      chosen node's docker daemon
                                      (acquire raw container,
                                       then exec / put_archive /
                                       get_archive / destroy)
```

The cluster Environment overrides four method groups on harbor's
`DockerEnvironment`:

| harbor method | Cluster behavior |
|---|---|
| `is_mounted` | Returns `False` so harbor switches to the post-trial `download_dir` branch. |
| `start(force_build)` | `xrlenv.Client.acquire_container(image=..., command=["sleep", "infinity"])`; `chmod 777` on agent + verifier dirs. |
| `stop(delete)` | `session.destroy()`; client.close(). |
| `exec(...)` | `session.exec_stream(["bash", "-c", command], cwd=, env=, user=, timeout_s=)` — streamed for tasks running 1-2 hours. |
| `upload_file` / `upload_dir` | `session.put_archive(target_dir, tarball)` — local tarfile pack. |
| `download_file` / `download_dir` | `session.get_archive(source_path)` — un-tar locally. |

## Operator setup

Per cluster bring-up, in order:

```bash
# 1. Boot the control plane (one-shot; idempotent):
xrlenv up

# 2. Populate the harbor task-metadata cache (clones the upstream
#    terminal-bench-2 repo into ~/.cache/harbor/tasks/ — needed for
#    harbor to read each task's task.toml / solution/ / tests/):
.venv/bin/bash examples/benchmarks-onboarding/terminal-bench-2/scripts/populate-harbor-cache.sh

# 3. Set env vars in the consumer's shell (typically alongside `xrlenv up`):
export XRLENV_GRPC_HOST=127.0.0.1
export XRLENV_GRPC_PORT=50051
export XRLENV_CONSUMER_TOKEN=$(cat ~/.xrlenv/secrets/consumer.token)
```

**Image distribution.** The 8 phase-0 tasks each ship a prebuilt
`docker_image` in their `task.toml` pointing at
`alexgshaw/<task>:<rev>` on Docker Hub. The cluster's
`acquire_container(ensure_image_present=True)` pulls each image
on first acquire (cached by `ImageCacheManager` after that). No
local pre-build step required for the smoke.

For tasks **without** a prebuilt `docker_image` field (rare in
the upstream tb2 catalog today), the cluster has no way to
manufacture the image; the operator either builds + pushes to
their own registry, or waits for **P1.7.C.2**'s build-on-acquire
(currently deferred). Until then, an unbuilt task fails fast at
acquire with `ImageNotFound`.

## Running the smoke

```bash
# Local baseline (single host; harbor builds + runs against local
# Docker daemon):
.venv/bin/python examples/benchmarks-onboarding/terminal-bench-2/smoke.py \
    --local

# Cluster 8-task smoke (env-var-driven; default --max-workers=1):
.venv/bin/python examples/benchmarks-onboarding/terminal-bench-2/smoke.py

# Cluster 8-task smoke, 4-way concurrent:
.venv/bin/python examples/benchmarks-onboarding/terminal-bench-2/smoke.py \
    --max-workers 4

# Custom task list:
.venv/bin/python examples/benchmarks-onboarding/terminal-bench-2/smoke.py \
    --tasks fix-git,dna-insert

# Sweep across every cached task (cluster pulls each image on first acquire):
.venv/bin/python examples/benchmarks-onboarding/terminal-bench-2/smoke.py \
    --all --max-workers 8
```

## Concurrency is the operator's choice

`--max-workers` defaults to **1 (serial)**. The smoke threads the
value into harbor's native `JobConfig.n_concurrent_trials`, which
spawns N concurrent trials inside one harbor process.

harbor uses asyncio internally per-trial, so this is already an
event-loop-scoped concurrency model — no external
`ThreadPoolExecutor` or `multiprocessing` wrapper is required.
Concurrency >1 wins when N nodes are available to fan out across.

The xrlenv cluster Environment is **concurrency-neutral by
construction** — it carries no `threading.Lock`, no shared mutable
state across calls, no default executor. Whatever concurrency model
harbor picks (today asyncio) works because each trial gets its own
session.

## Artifact archiving

Always archives. harbor itself writes per-trial outputs
(`trial.log`, `result.json`, `verifier/`, `agent/`, `artifacts/`)
under `<jobs_dir>/<job_name>/<trial_name>/`.

Defaults:
- `jobs_dir = <repo>/tmp/` (gitignored)
- `job_name = smoke-terminal-bench-2-YYYYMMDD-HHMMSS` (UTC
  timestamped; lex-sortable)

Overrides:
- `--save-artifacts <PATH>` — different `jobs_dir`
- `--job-id <NAME>` — explicit `job_name` (useful for tagging
  runs with model/version labels)

Layout:

```
<jobs_dir>/<job_name>/
├── <task_name>__<trial-id>/
│   ├── trial.log
│   ├── result.json
│   ├── verifier/
│   │   ├── reward.txt
│   │   └── ...
│   ├── agent/
│   └── artifacts/
└── ...                           # per-trial dirs, one per task
```

`<repo>/tmp/` is gitignored so artifacts never leak into commits.

## What the smoke does

For each task (default 8 tasks, see `SMOKE_TASKS` in `smoke.py`):

1. Harbor reads the task definition from the local cache
   (`$XRLENV_BENCHMARK_CACHE`, default `~/.cache/harbor/tasks/`).
2. Harbor's `OracleAgent` copies `solution/solve.sh` into the
   container and runs it.
3. Harbor's verifier runs the per-task tests; writes
   `reward.txt` / `reward.json` to `/logs/verifier/`.
4. The cluster Environment's `download_dir` pulls the verifier
   outputs back to the consumer host.
5. Harbor records `trial_result.verifier_result.rewards` per trial.

Pass criterion: every trial has `verifier_result.rewards` fully
populated with positive values. A failing trial under the oracle
policy is a plumbing bug, not a model-eval signal.

## Agent integration patterns (two examples)

`smoke.py` runs harbor's stock oracle policy across many tasks. The
two examples below show **how a consumer wires their own agent**
onto xrlenv as a container substrate. Both are runnable end-to-end;
both are intentionally small.

### Outside the container — `agent_outside_container.py`

Pattern: **the agent lives in this process.** It predicts an
action, hands it to xrlenv, gets the observation back, decides
again. xrlenv contributes only `Client.acquire_container(...)`,
`session.exec(...)`, and `session.put_archive(...)` (for staging
the verifier) — nothing else. The example walks an `OracleAgent`
through a **real TB-2 task** (default `fix-git`): the oracle reads
`solution/solve.sh` from the harbor cache, emits it as a structured
action, xrlenv runs it inside the task's container, then we push
the task's `tests/` into the container and run the same verifier
script harbor's runner would. Pass/fail is the same
`/logs/verifier/reward.txt == 1` signal harbor's report aggregator
consumes.

Replace `OracleAgent.next_action(obs)` with whatever your training
stack's policy class looks like; everything else (task loading,
container acquire, verifier invocation) stays the same.

```bash
# Pre-req: harbor cache populated and the task image present on the
# chosen node. See "Operator setup" above.
export XRLENV_GRPC_HOST=127.0.0.1
.venv/bin/python examples/benchmarks-onboarding/terminal-bench-2/agent_outside_container.py
# or another task:
.venv/bin/python examples/benchmarks-onboarding/terminal-bench-2/agent_outside_container.py --task build-pov-ray
```

Use this pattern when the policy is GPU-resident in your trainer
process, when you already have an `act(obs)` / `predict(obs)`
shape, or when you want the trajectory's "intelligence" to live
outside the container.

### Inside the container — `agent_inside_container.py`

Pattern: **the agent runs inside the container.** A real CLI
agent — `claude-code`, `aider`, `codex`, `openhands`, etc. — is
installed in the container and drives the workspace autonomously.
The benchmark verifier judges the result at the end. xrlenv
provides the container; harbor provides the installed-agent class
that knows how to install + run it.

```bash
export ANTHROPIC_API_KEY=sk-ant-...           # or another vendor key
export XRLENV_GRPC_HOST=127.0.0.1
.venv/bin/python examples/benchmarks-onboarding/terminal-bench-2/agent_inside_container.py \
    --task fix-git
```

The script refuses to start if no LLM API key is found in the
environment (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / etc.). Use
this pattern when you're evaluating off-the-shelf agentic CLIs
against a benchmark, or when the agent's runtime needs to live in
the same workspace it's modifying.

The only difference from `smoke.py` is the `AgentConfig`: the
example sets `name="claude-code"` so harbor picks
`harbor.agents.installed.claude_code:ClaudeCode` from its
registry. xrlenv contributes zero install logic.

### Multi-key workloads — getting secrets into the container

Real workloads usually want >1 API key in the container (Anthropic
+ Tavily for web search + GitHub token for repo access, etc.).
xrlenv ships two harbor-agnostic helpers — `parse_dotenv()` and
`upload_dotenv()` — exported from `xrlenv.client`. Pick the shape
your in-container code expects:

| Shape | Helper | When |
|---|---|---|
| **Env vars** at container creation | `parse_dotenv(".env")` → pass to `acquire_container(environment=...)` | The agent / tooling reads from `os.environ` / `$VAR` (claude-code, most agentic CLIs, anything that uses `python-dotenv` won't matter — they'll see the var either way). |
| **File** copied into the container | `upload_dotenv(session, source=".env", target_dir="/workspace")` | Tooling that specifically wants a `.env` file to live on disk (some `python-dotenv` configurations, frameworks that watch for the file at a fixed path). |

Both shapes are operator-side primitives — they don't know about
harbor / claude-code / SWE-bench specifically. They just give you
either env-var injection at acquire time, or a file at a known
in-container path.

```python
from xrlenv import Client
from xrlenv.client.dotenv import parse_dotenv, upload_dotenv

# Shape 1: env vars at creation.
env = parse_dotenv(".env")
async with await client.acquire_container(
    image=task.docker_image,
    environment=env,
    command=["sleep", "infinity"],
) as session:
    # Shape 2 (optional): also copy the file in verbatim.
    await upload_dotenv(session, source=".env", target_dir="/workspace")
    ...
```

The outside-agent example accepts both as CLI flags:
`--env-file path/to/.env` (shape 1) and `--upload-env-file
path/to/.env` (shape 2). They're independent — combine them when
some tools want env vars and others want a file.

**Note for the inside pattern (harbor's Job orchestrator).**
Harbor's installed-agent classes own the `exec` calls and forward
their own allowlist of env vars (claude-code forwards
`ANTHROPIC_API_KEY` automatically, but not arbitrary keys).
The simplest fix is to `set -a; source .env; set +a` in your shell
before launching the harbor Job — harbor inherits the env, and
each installed-agent class forwards what it knows about. For full
arbitrary-key forwarding under harbor, subclass the agent and
override its env-forwarding (harbor-side, not xrlenv-side).

### When to use which

| Question | Outside | Inside |
|---|---|---|
| Where does the policy live? | This process | Inside the container |
| Who drives the loop? | Your code (act-step-act) | The agent (autonomously) |
| Who installs the agent? | n/a — agent is in-process | Harbor's installed-agent class |
| Trajectory shape | Per-step action + obs | Full agent session log + verifier rewards |
| Typical use | RL training, supervised eval with bespoke policies | Off-the-shelf coding-agent evals |
| xrlenv surface | `acquire_container` + `session.exec` | Same — harbor wraps it via `XrlenvHarborEnvironmentCluster` |

## Adapting to your own benchmark

If your benchmark is a harbor-shape task (subclass of
`harbor.BaseEnvironment`), the integration is just the one-line
`import_path` swap above. Populate the harbor cache, set the env
vars, point your `job.yaml` at
`xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster`. Make
sure each task's `task.toml` carries a registry-resolvable
`docker_image` (or wait for P1.7.C.2's build-on-acquire).

If your benchmark is **docker-py-using** (sync, `docker.from_env()`
at the top), see the sibling [`swebench-verified/`](../swebench-verified/)
example for the docker-py drop-in path — typically a one-line
swap of `docker.from_env()` → `xrlenv.from_env()`.

## Operator scripts

- `scripts/populate-harbor-cache.sh` — clones the upstream
  terminal-bench-2 task catalog into `$XRLENV_BENCHMARK_CACHE`
  (default `~/.cache/harbor/tasks/`). Idempotent; safe to re-run.
  Required on each consumer host (harbor reads `task.toml` /
  `solution/` / `tests/` from the cache regardless of where
  the image lives).

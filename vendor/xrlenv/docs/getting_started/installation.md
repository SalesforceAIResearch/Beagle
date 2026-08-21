# Installation

This page covers installing XRLEnv from source. After installing,
head to the {doc}`/getting_started/quickstart` for a 5-minute end-to-end walkthrough.

## System prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12 or newer | Strict requirement; XRLEnv uses 3.12+ syntax. |
| Docker | Engine 24+ | Must be reachable as the current user. `docker info` should succeed without `sudo`. On macOS, any Docker Engine front-end works (Docker.app, Colima, OrbStack, Lima). |
| OS | Linux, macOS | Linux uses Unix-domain-socket transport; macOS auto-falls back to TCP (handled by the platform). |
| Disk | ~5 GB free under `~/.xrlenv/` | Trajectory artifacts, SQLite state, image cache. Tunable via run-dir GC and image-cache size. |

[uv](https://docs.astral.sh/uv/) is the recommended package manager.
Plain `pip` works as an escape hatch (documented at the end).

## Get the source

```bash
git clone https://github.com/<your-org>/XRLEnv.git
cd XRLEnv
```

No PyPI release yet; install from a working tree.

## Install (uv, recommended)

```bash
uv sync
```

This creates `.venv/` and installs XRLEnv's runtime dependencies. To
add an optional extra (e.g. a benchmark's framework dep) after
`uv sync`:

```bash
uv pip install -e '.[<extra-name>]'      # e.g. .[swebench-verified], .[terminal-bench-2], .[docs]
```

Or install **everything** in one shot:

```bash
uv pip install -e '.[all]'               # every extra below (dev + docs + observability + benchmarks)
```

Available extras:

| Extra | What it pulls in | When to install |
|---|---|---|
| `all` | **Every extra below** — `dev` + `docs` + `observability` + `terminal-bench-2` + `swebench-verified` | One-shot full install (e.g. a workstation that runs every benchmark + builds docs). |
| `swebench-verified` | `swebench>=4.1`, `datasets>=4.8.5` | Running the SWE-bench onboarding example. |
| `terminal-bench-2` | `harbor>=0.5` | Running the terminal-bench-2 / seta-env onboarding examples or using `XrlenvHarborEnvironmentCluster`. |
| `observability` | `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-grpc` | Enabling OTel tracing (see {doc}`/observability/tracing`). Without this extra installed, `xrlenv` still runs — `get_tracer()` falls back to a noop tracer. |
| `dev` | pytest, mypy, ruff, etc. | Contributing to xrlenv core. |
| `docs` | sphinx, myst-parser, sphinxcontrib-mermaid | Building this documentation site. |

## How to run commands

`uv sync` creates `.venv/` but does **not** activate it. The docs
use the explicit-interpreter pattern for every Python invocation, so
commands work in any shell without needing `source .venv/bin/activate`:

```bash
.venv/bin/python -m pytest tests/              # explicit module
.venv/bin/sphinx-build -b html docs docs/_build/html
.venv/bin/xrlenv up --help                     # console scripts
```

If you'd rather have `python` / `xrlenv` / `pytest` resolve from
`PATH`, activate the venv once per shell:

```bash
source .venv/bin/activate
xrlenv up --help
```

Both forms are equivalent. The docs use `.venv/bin/...` so you can
copy-paste a recipe into a fresh shell without the activation step.

## Verify the install

```bash
.venv/bin/xrlenv --help
.venv/bin/xrlenv-node --help
.venv/bin/xrlenv-stub --help
```

All three should print their usage banner.

## Configuring environment variables (`.env` auto-load)

Most operator-facing commands and SDK entry points read a handful of
environment variables (`XRLENV_GRPC_HOST`, `XRLENV_CONSUMER_TOKEN`,
`XRLENV_BENCHMARK_CACHE`, plus LLM API keys for the in-container agent
examples). Setting these in every new shell got tedious enough that
xrlenv auto-loads a `.env` file at import time. **Set once, every
script picks it up.**

```bash
cp .env.example .env
# Edit .env to fill in YOUR values.
```

After that, `import xrlenv` (and any console script — `xrlenv up`,
the smokes, the agent examples) automatically populates `os.environ`
from `.env` — discovery walks from `CWD` upward to the first `.env`
it finds, mirroring the python-dotenv convention.

### Precedence

```
shell-exported env (highest — `export FOO=...`)
└─ .env file values (fallback)
```

Shell-exported values always win — `.env` only fills gaps. This lets
operators override per-shell without editing the file:

```bash
# .env says XRLENV_GRPC_HOST=127.0.0.1, but for this command:
XRLENV_GRPC_HOST=internal-ip .venv/bin/xrlenv build apply ...
```

### Required vs optional

`.env.example` (committed at the repo root, real `.env` is
gitignored) documents every recognized variable with its purpose,
grouped by use case:

| Group | Variables | When |
|---|---|---|
| Consumer SDK / drop-in | `XRLENV_GRPC_HOST`, `XRLENV_GRPC_PORT`, `XRLENV_CONSUMER_TOKEN` | Required to dial a control plane. |
| Benchmark harnesses | `XRLENV_BENCHMARK_CACHE` | Required for terminal-bench-2 / harbor. |
| Build-plan generation | `DOCKERHUB_USER`, `DOCKERHUB_TOKEN` | Lift Docker Hub's ~100 / 6h unauth probe rate-limit when generating large plans (e.g. SWE-bench Verified `--all`, 500 entries). Set `DOCKERHUB_TOKEN` to a Docker Hub [Personal Access Token](https://docs.docker.com/security/for-developers/access-tokens/). See {doc}`/technical_details/images/build_plan` § "Docker Hub probing and rate limits". |
| In-container LLM agents | `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / etc. | Required only when `agent_inside_container.py`-style examples drive a real model. |
| Operator (rare in consumer .env) | `XRLENV_CONTROL_PLANE`, `XRLENV_NODE_TOKEN`, `XRLENV_OPERATOR_TOKEN`, `XRLENV_VIEWER_TOKEN` | Mostly set by `xrlenv up` / `xrlenv-node` deploy config. |
| Knobs | `XRLENV_GRPC_SECURE`, `XRLENV_DOTENV` | Defaults usually fine. |

### Opting out

Set `XRLENV_DOTENV=off` (or `false` / `0` / `no` / `disabled`) in the
shell to disable the auto-load entirely. Useful for tests that need
strict env isolation or for operators preferring shell-only config:

```bash
export XRLENV_DOTENV=off
# .env in this directory is now ignored.
```

The auto-load is silent + idempotent: a missing or malformed `.env`
never raises from the import path, and subsequent imports don't
re-walk the filesystem.

## State and artifact locations

XRLEnv writes everything under `~/.xrlenv/`:

```
~/.xrlenv/
├── state.db            # SQLite state store (rollouts, sandboxes, audit, ...)
├── runs/<date>/<rollout_id>/
│   ├── meta.json
│   ├── trajectory.jsonl
│   └── coordinator.log
├── image-cache/        # Per-node image LRU cache (populated when xrlenv-node runs)
└── secrets/            # Operator-issued bearer tokens (one file per role, mode 0600)
    ├── node.token
    ├── consumer.token
    └── operator.token
```

Paths are configurable through environment variables; see
{doc}`/developer_guide/cli_reference`.

## Install with plain pip (escape hatch)

If you can't install uv, plain `pip` works:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

After this, every `uv pip install -e '.[X]'` recipe in the rest of
the docs becomes `pip install -e '.[X]'` (with the venv activated).
Every `.venv/bin/python ...` recipe still works as written.

## Next steps

- {doc}`/getting_started/quickstart` — boot a control plane and acquire one container.
- {doc}`/deploy/single_node_deployment` — pick a deployment shape (single host vs multi-node cluster).
- {doc}`/supported_benchmarks_and_harnesses/writing_your_own_adapter` — run a pre-wired benchmark (SWE-bench, terminal-bench-2).
- {doc}`/build_with_xrlenv/work_with_xrlenv_managed_containers/index` — use xrlenv-managed containers in your own code.

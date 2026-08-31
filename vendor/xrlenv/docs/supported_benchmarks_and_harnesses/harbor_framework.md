# [H] Harbor (tb2, seta …)

[Harbor](https://www.harborframework.com/docs) is a benchmark framework whose
tasks share one layout — a `Dockerfile`, `task.toml`, `solution/`, and `tests/` —
and whose trial driver loads a sandbox provider from `environment.import_path` in
`job.yaml`. XRLEnv ships a provider that routes Harbor's container operations
through the cluster, so **any Harbor-format benchmark** runs on XRLEnv with a
one-line config change and no fork of the benchmark.

This page covers:

- [General integration guide](#general-integration-guide) — the adapter, setup,
  and image resolution that apply to every Harbor-format benchmark.
- [terminal-bench-2](#terminal-bench-2) — the prebuilt-image case.
- [seta-env](#seta-env) — the build-it-yourself (Dockerfile → private registry)
  case.
- [LHTB](#lhtb) — Long-Horizon Terminal-Bench; a mixed plan (prebuilt docker.io + a
  few build-and-push) with a custom oracle seam.
- [TerminalWorld](#terminalworld) — multi-service compose (sidecars / private
  networks) under runc, with opt-in sysbox routing.
- [Onboarding a new Harbor-format benchmark](#onboarding-a-new-harbor-format-benchmark).

## General integration guide

### The one-line swap

Harbor's trial driver loads the environment class named in `job.yaml`. Point it at
XRLEnv's cluster provider:

```diff
 environment:
-  import_path: harbor.environments.docker.docker:DockerEnvironment
+  import_path: xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster
```

Everything else stays Harbor-owned: agents (including the **OracleAgent** that
runs `solution/solve.sh`), tasks, verifier config, metrics, reports, and output
layout. `XrlenvHarborEnvironmentCluster` subclasses Harbor's `DockerEnvironment`
and overrides only the container-touching methods:

| Harbor method | What the adapter does |
|---|---|
| `capabilities.mounted` | `False`, so Harbor uses its post-trial `download_dir` branch instead of host bind mounts (the container is on a remote node). |
| `start(force_build)` | Builds an `xrlenv.Client` from env vars, resolves the task image, acquires a managed container, creates the `/logs/*` dirs, and records artifact metadata for the admin panel. |
| `stop(delete)` | Destroys the XRLEnv container session and closes the client. |
| `exec(...)` | Streaming exec, so long-running tasks keep the gRPC path active. |
| `upload_file` / `upload_dir` | Creates the target dir and uploads a tar archive. |
| `download_file` / `download_dir` | Downloads a tar archive and extracts it locally. |

Source of truth:
[`xrlenv_plugins/harbor/environment.py`](https://github.com/Yutong-Dai/XRLEnv/blob/main/xrlenv_plugins/harbor/environment.py).

### Operator setup

```bash
# 1. Boot the control plane (and, for build-it-yourself benchmarks, the
#    registries — see the seta-env section).
xrlenv up

# 2. Populate the Harbor task-metadata cache for your benchmark (each onboarded
#    plug-in ships its own cache builder).
.venv/bin/python xrlenv_plugins/benchmarks/<benchmark>/build_cache.py
```

The consumer reads its cluster + registry config from `.env` (xrlenv auto-loads it
on import — no `source .env` needed). The relevant variables:

| Variable | Required | Description |
|---|---|---|
| `XRLENV_GRPC_HOST` | yes | Control-plane host. |
| `XRLENV_GRPC_PORT` | no (default `50051`) | Control-plane port. |
| `XRLENV_CONSUMER_TOKEN` | when the control plane runs with auth | Bearer token from `xrlenv tokens issue consumer`. |
| `XRLENV_GRPC_SECURE` | no (default `false`) | `true` / `1` / `yes` / `on` for TLS. |
| `XRLENV_PRIVATE_REGISTRY_HOST` / `_PORT` | build-it-yourself benchmarks | The registry the built images live in (see [seta-env](#seta-env)). |

### Image resolution

`start()` resolves the image the node acquires, in this precedence:

1. **`xrlenv_image_template` kwarg** — a `str.format` template with `{task_id}`
   (the task directory name). This is how a benchmark whose images live in a private
   registry under a derived name points the cluster at them — no per-task config, no
   subclass. Sweep drivers such as `seta/run_oracle_sweep.py` and
   `terminalworld/run_oracle_sweep.py` inject this kwarg via
   `EnvironmentConfig(kwargs={"xrlenv_image_template": template})`, composing the
   template from `--registry` / `$XRLENV_PRIVATE_REGISTRY_HOST`. There is no
   process-level environment variable to set.
2. **`task_env_config.docker_image`** — an upstream-published prebuilt (e.g.
   terminal-bench-2's `alexgshaw/<task>:<rev>`), or a private-registry ref written
   by `build_cache.py --stage repin` (e.g. LHTB).
3. **`hb__<environment_name>`** — the locally-built Harbor convention.

`acquire_container(ensure_image_present=True)` pulls the resolved image on first
acquire; an unresolvable image fails fast with `ImageNotFound`.

### Known limitations

- **Single-service tasks only.** Tasks needing a helper `db` / `redis` container
  aren't supported by the cluster adapter yet.
- **No build-on-acquire.** Images must exist (in a registry) before the run — the
  build-it-yourself flow (seta-env) builds + pushes them ahead of time.
- **`keep_containers=True` still destroys.** The session model releases the
  container on stop; the adapter logs a warning.

### Capacity pacing / backpressure

When you run more concurrent workers than the cluster can immediately place,
XRLEnv uses fail-fast backpressure rather than silently queuing indefinitely.
Each `acquire_container` call waits up to `XRLENV_HARBOR_ACQUIRE_QUEUE_TIMEOUT_S`
(default `240.0` s) for a capacity slot. If no slot opens in that window, the
acquire raises `CapacityExhausted` and the rollout is sealed as
`capacity_rejected`. The benchmark sweep retries these automatically via an
infra-retry loop. The **dominant** retry case is a fail-fast acquire *before*
`solve.sh` runs, so the task usually runs once; but a retry re-runs the WHOLE trial
in a FRESH container, so a *post-acquire* infra error (e.g. `NodeCommandTimeout` on
an exec) can re-execute the task body. The recorded statistics are still **one
outcome per task** (re-graded) — so this matters only for external side effects; a
waited-then-retried trial still reports as a clean single pass.

The timeout is deliberately shorter than Harbor's own ~360 s setup-cancel
window. This ordering guarantees that a paced acquire always surfaces as a
retriable `CapacityExhausted` rather than Harbor's non-retriable task cancel.

**What the admin panel shows.** `capacity_rejected` is categorised as a
capacity-pacing event, not a failure. The `/` overview counts these in a
separate "capacity-paced in last hour" tile and excludes them from the
"failed in last hour" tile and per-template failure rate. The `/users`
scoreboard excludes them from the `released ÷ total` success-rate denominator,
so a paced-then-retried run is not scored as a partial failure.

**Two operator levers:**

1. **Raise `XRLENV_HARBOR_ACQUIRE_QUEUE_TIMEOUT_S`** (export before launching
   the sweep — no control-plane redeploy). This makes each acquire wait longer
   for a slot. Keep it below ~360 s to preserve the fail-fast ordering
   described above. For sustained capacity pressure the better lever is fewer
   concurrent workers or more cluster nodes.
2. **Retry caller-side.** The sweep's infra-retry loop already does this. A
   fresh `acquire_container` re-queues from scratch; there is no session state
   to resume.

### Advanced container configuration — capabilities, devices, cpuset pinning

`XrlenvHarborEnvironmentCluster` accepts additional kwargs via
`environment.kwargs` in `job.yaml`. These are stripped before the kwargs
reach Harbor's constructor and forwarded to `acquire_container`:

| kwarg | Type | Description |
|---|---|---|
| `xrlenv_cap_add` | `list[str]` | Linux capabilities to add (e.g. `["NET_ADMIN", "SYS_ADMIN"]`). Allowed by default — no `nodes.yaml` change needed. |
| `xrlenv_devices` | `list[str]` | Host devices to expose inside the container (e.g. `["/dev/loop0:/dev/loop0"]`). |
| `xrlenv_privileged` | `bool` | Run the container `--privileged`. Default-denied; requires operator opt-in. |
| `xrlenv_cpu_pinning` | `bool` | Confine the container to a cpuset sized to `ceil(cpus)`. Off by default; opt in per-task via `[environment.env] XRLENV_CPU_PINNING = "1"` in `task.toml`, or job-wide via this kwarg. Useful on large hosts where `nproc` would otherwise report the full host core count. |
| `xrlenv_cpu_multiplier` | `float` | Multiply the task's effective CPU limit by this factor (default `1.0`). Applied on top of Harbor's `_effective_cpus` so the per-task relative sizing is preserved. |
| `xrlenv_mem_multiplier` | `float` | Multiply the task's effective memory limit by this factor (default `1.0`). Applied on top of Harbor's `_effective_memory_mb`. |

Example `job.yaml` fragment:

```yaml
environment:
  import_path: xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster
  kwargs:
    xrlenv_cap_add: ["NET_ADMIN"]
    xrlenv_cpu_multiplier: 2.0
```

```{note}
`xrlenv_privileged` and `xrlenv_devices` are gated operator-side by the
`nodes.yaml` policy (`allow_privileged`, `allowed_devices`, capability
denylist). The control plane rejects requests that violate the policy.
See {doc}`/deploy/multi_node_deployment/runbook` for the policy tiers.
```

## terminal-bench-2

terminal-bench-2 tasks ship a **prebuilt `docker_image`** in `task.toml`
(`alexgshaw/<task>:20251031` on Docker Hub), so image resolution falls through to
precedence #2 — no registry or template needed. The cluster pulls each image on
first acquire.

```bash
# populate + patch the cache, then run the oracle sweep against the cluster:
.venv/bin/python xrlenv_plugins/benchmarks/terminal_bench_2_1/build_cache.py       # populate + patch
.venv/bin/python xrlenv_plugins/benchmarks/terminal_bench_2_1/run_oracle_sweep.py  # oracle correctness gate
bash xrlenv_plugins/benchmarks/terminal_bench_2_1/run_full_sweep.sh                # full set
```

See the plug-in's README for the exact task-selection flags (`--tasks`,
`--max-workers`, retry layers).

Worked example:
[`xrlenv_plugins/benchmarks/terminal_bench_2_1/`](https://github.com/Yutong-Dai/XRLEnv/tree/main/xrlenv_plugins/benchmarks/terminal_bench_2_1)
— the onboarded plug-in in the GUIDELINE layout (`build_cache.py` +
`build_plan_gen.py` + `run_oracle_sweep.py` + `run_full_sweep.sh` +
`README.md`/`STATUS.md`/`tests/`).

## seta-env

[camel-ai/seta-env](https://github.com/camel-ai/seta-env)'s Harbor-Dataset tasks
ship a **`Dockerfile`, not a prebuilt `docker_image`**. The flow is therefore
*build once → push to a private registry → pull per-task*:

1. **Build + push** every task image to the FSx-backed private registry as
   `<registry>/seta-env/<id>:main` (bulk, sharded across nodes — see
   {doc}`/deploy/multi_node_deployment/private_registry`).
2. **Resolve via sweep-injected kwarg.** The sweep driver composes the template
   `<registry>/seta-env/{task_id}:main` from `XRLENV_PRIVATE_REGISTRY_HOST` / `_PORT`
   in `.env` and passes it as `xrlenv_image_template` in `EnvironmentConfig(kwargs=...)`
   — so precedence #1 maps each task to its built image — no `docker_image`, no subclass.
3. **Cache shard.** seta tasks live under a `seta-env/` shard of the shared Harbor
   cache (`~/.cache/harbor/tasks/seta-env/<id>/`), so they coexist with
   terminal-bench-2 without collision.

```bash
# populate (clones camel-ai/seta-env into the seta-env/ cache shard), then run:
.venv/bin/python xrlenv_plugins/benchmarks/seta/build_cache.py --stage populate
.venv/bin/python xrlenv_plugins/benchmarks/seta/run_oracle_sweep.py         # 8-task oracle smoke
.venv/bin/python xrlenv_plugins/benchmarks/seta/run_oracle_sweep.py --all   # every cached seta task
.venv/bin/python xrlenv_plugins/benchmarks/seta/run_oracle_sweep.py --local # baseline (builds Dockerfile locally)
```

The smoke reads `.env`, prints the resolved cluster config, and fails fast if a
required value is missing. A small set of upstream tasks have broken Dockerfiles
and are blacklisted (`xrlenv_plugins/benchmarks/seta/black_list.txt`);
`--all` skips them automatically.

Worked example:
[`xrlenv_plugins/benchmarks/seta/`](https://github.com/Yutong-Dai/XRLEnv/tree/main/xrlenv_plugins/benchmarks/seta).

## LHTB

[zli12321/LHTB](https://github.com/zli12321/LHTB) (Long-Horizon Terminal-Bench, 46
harbor-shape tasks) is a **mixed-image** case: ~40 tasks pull a prebuilt docker.io image
(precedence #2 — resolved from `task.toml`), while a handful with broken/unpublished
upstream images are rebuilt and repinned to the private registry by
`build_cache.py --stage repin`. It grades with a **dense float** (partial credit), so the
oracle gate keys on `reward > 0` (not `== 1.0`); the paper's success bar is `R ≥ 0.95`. Its
one custom seam is a `SealingOracleAgent` (in `run_oracle_sweep.py`) that seals egress for
offline tasks before grading.

```bash
# clone + git-lfs + task-level fixes + repin the rebuild tasks, then run the gate:
.venv/bin/python xrlenv_plugins/benchmarks/lhtb/build_cache.py --stage all --registry "$XRLENV_PRIVATE_REGISTRY_HOST"
bash xrlenv_plugins/benchmarks/lhtb/run_full_sweep.sh --max-workers 8   # green set + the issue-#2 TBD set
```

Worked example + full per-task disposition:
[`xrlenv_plugins/benchmarks/lhtb/`](https://github.com/Yutong-Dai/XRLEnv/tree/main/xrlenv_plugins/benchmarks/lhtb)
(`README.md` + `STATUS.md`).

## TerminalWorld

The [`verified`](https://huggingface.co/datasets/EuniAI/TerminalWorld) split (200
human-reviewed tasks) is the **multi-service compose** case: many tasks ship a
`docker-compose.yaml` (sidecars, private networks, static IPs). Like seta-env they ship a
`Dockerfile` (no prebuilt `docker_image`), so each is built + pushed to the private registry
as `<registry>/terminalworld-verified/<id>:main` and resolved via the sweep-injected
`xrlenv_image_template` kwarg. The compose stack runs under **runc** (the control-plane
policy gate makes multi-container safe without sysbox); redundant `privileged: true` flags
are stripped by `build_cache.py` (`COMPOSE_DROP_PRIVILEGED`). Tasks whose own `solve.sh`
runs `docker` / `systemd` are opt-in routed to a **sysbox-runc** node (`SYSBOX_TASKS`).

```bash
# populate + patch (+ sysbox markers), build+push the images on a build host, then the gate:
.venv/bin/python xrlenv_plugins/benchmarks/terminalworld/build_cache.py --stage all
bash xrlenv_plugins/benchmarks/terminalworld/run_full_sweep.sh --max-workers 32
```

Worked example + full per-task disposition:
[`xrlenv_plugins/benchmarks/terminalworld/`](https://github.com/Yutong-Dai/XRLEnv/tree/main/xrlenv_plugins/benchmarks/terminalworld)
(`README.md` + `STATUS.md`).

## Onboarding a new Harbor-format benchmark

A new Harbor-format benchmark reuses **everything above** — the same
`import_path`, the same adapter — and only differs in how its images are
distributed. Pick the case that matches:

| Your benchmark ships… | Pattern | Image resolution | Model it on |
|---|---|---|---|
| A **prebuilt `docker_image`** per task (on a registry the nodes can pull) | Pull on acquire | precedence #2 (`docker_image`) — nothing to set | [terminal-bench-2](#terminal-bench-2) |
| A **`Dockerfile`** per task (no prebuilt image) | Build once → push → pull | precedence #1 — sweep injects `xrlenv_image_template` kwarg | [seta-env](#seta-env) |

Steps either way:

1. **Populate the cache.** Write a `populate-harbor-cache.sh` that lands the tasks'
   `task.toml` / `solution/` / `tests/` under the Harbor cache (a benchmark shard,
   as seta-env does, keeps benchmarks from colliding).
2. **Make images resolvable.** Prebuilt: ensure each `task.toml` carries a pullable
   `docker_image`. Build-it-yourself: build + push (see
   {doc}`/deploy/multi_node_deployment/private_registry`) and have your sweep driver
   pass the template as `EnvironmentConfig(kwargs={"xrlenv_image_template": template})`.
3. **Run.** Drive `harbor.Job.run()` (or `harbor run`) with
   `environment.import_path: xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster`
   and `AgentConfig()` (defaults to the OracleAgent). The two onboarding examples
   are copy-paste starting points.

If your benchmark is **not** Harbor-format (its own `BaseEnvironment`-style
interface, or a docker-py harness), see {doc}`writing_your_own_adapter` for the
decision guide between a framework adapter, the `xrlenv.from_env()` docker-py
drop-in, and the direct API.

## Version compatibility

The plug-in loads on harbor 0.8.x as well as newer releases. Harbor 0.8.x
lacks the `NetworkMode` model introduced in later versions — importing it
unconditionally would raise a misleading "harbor is not installed" error on
every 0.8.x environment. The plug-in guards against this with a deferred
import:

```python
try:
    from harbor.models.task.config import NetworkMode
except ImportError:
    NetworkMode = None  # harbor 0.8.x
```

The internet-access decision (`_network_mode_for_task`) then branches on which
field the installed harbor actually provides: newer harbor's
`EnvironmentConfig.network_mode` enum takes precedence; 0.8.x falls back to
the legacy `allow_internet` boolean (default `True`). No feature that requires
`NetworkMode` is silently skipped — tasks that declare `no-network` are still
correctly isolated on both versions via the respective field.

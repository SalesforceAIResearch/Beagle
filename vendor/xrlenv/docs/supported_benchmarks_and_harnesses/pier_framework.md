# [H] Pier (DeepSWE)

[Pier](https://github.com/datacurve-ai/pier) (PyPI `datacurve-pier`) is a
harbor fork that reimplements the trial/verifier harness in-tree. It is the
harness for the **DeepSWE** benchmark. XRLEnv ships a pier adapter —
`xrlenv_plugins.pier` — that is the direct analog of the harbor adapter
retargeted at pier's classes. **Any pier-format task corpus runs on XRLEnv
with a one-line config change and no fork of pier.**

This page covers:

- [General integration guide](#general-integration-guide) — the adapter,
  setup, image resolution, and pier-specific behavior.
- [Pier vs harbor deltas](#pier-vs-harbor-deltas) — what changes between the
  two adapters and why.
- [Separate-verifier seam (DeepSWE)](#separate-verifier-seam-deepswe) — how
  the adapter handles `environment_mode="separate"`.
- [Capacity pacing and backpressure](#capacity-pacing-and-backpressure).
- [Advanced container configuration](#advanced-container-configuration).
- [DeepSWE benchmark](#deepswe-benchmark) — running the oracle sweep.

## General integration guide

### The one-line swap

Pier ships a first-class `import_path` escape hatch (the same one its built-in
`docker`, `modal`, and `daytona` environments use), so the adapter is selected
with no changes to the benchmark repo:

```diff
 environment:
-  import_path: pier.environments.docker.docker:DockerEnvironment
+  import_path: xrlenv_plugins.pier:XrlenvPierEnvironmentCluster
```

Equivalently, from the CLI:

```bash
pier run --environment-import-path xrlenv_plugins.pier:XrlenvPierEnvironmentCluster
```

Or programmatically:

```python
from pier.models.trial.config import EnvironmentConfig

env = EnvironmentConfig(
    import_path="xrlenv_plugins.pier:XrlenvPierEnvironmentCluster",
)
```

`XrlenvPierEnvironmentCluster` subclasses pier's `DockerEnvironment` and
overrides only the container-touching methods:

| Pier method | What the adapter does |
|---|---|
| `capabilities.mounted` | `False` — cluster nodes are remote, so pier uses its post-trial `download_dir` branch (the reward round-trips to the host). |
| `capabilities.preinstall_agents` | `False` — cluster mode clears `agent_install_spec` so pier installs each agent at runtime via `exec` instead of assuming a pre-baked binary. |
| `start(force_build)` | Builds an `xrlenv.Client` from env vars, resolves the task image, acquires a managed container (or compose project for multi-service tasks), creates `/logs/*` dirs, and records artifact metadata for the admin panel. |
| `stop(delete)` | Destroys the XRLEnv container session and closes the client. |
| `exec(...)` | Streaming exec, so long-running tasks keep the gRPC path active. |
| `upload_file` / `upload_dir` | Creates the target dir and uploads a tar archive. |
| `download_file` / `download_dir` | Downloads a tar archive and extracts it locally. |

Source of truth:
[`xrlenv_plugins/pier/environment.py`](https://github.com/Yutong-Dai/XRLEnv/blob/main/xrlenv_plugins/pier/environment.py).

### Operator setup

```bash
# 1. Install the pier extra.
pip install -e '.[deep-swe]'

# 2. Boot the control plane.
xrlenv up

# 3. Set cluster connection config in .env (auto-loaded by xrlenv on import).
```

The relevant environment variables:

| Variable | Required | Description |
|---|---|---|
| `XRLENV_GRPC_HOST` | yes | Control-plane host. |
| `XRLENV_GRPC_PORT` | no (default `50051`) | Control-plane port. |
| `XRLENV_CONSUMER_TOKEN` | when the CP runs with auth | Bearer token from `xrlenv tokens issue consumer`. |
| `XRLENV_GRPC_SECURE` | no (default `false`) | `true` / `1` / `yes` / `on` for TLS. |
| `XRLENV_PIER_IMAGE_TEMPLATE` | optional | Override image resolution (see below). |

### Image resolution

`start()` resolves the image the node acquires, in this precedence:

1. **`XRLENV_PIER_IMAGE_TEMPLATE`** — a `str.format` template with `{task_id}`
   (the task directory name) and `{environment_name}`. Use this when a
   benchmark's images live in a private registry under a derived name.
2. **`task_env_config.docker_image`** — an upstream-published prebuilt (e.g.
   DeepSWE's public-ECR ref `public.ecr.aws/d3j8x8q7/swe-bench-202605:<id>-v1.1`).
3. **Separate-verifier fallback** — when this is the verifier container and
   the resolved config carries no `docker_image`, resolve the base image from
   `tests/Dockerfile` `FROM` or the parent task's top-level `docker_image`
   (see [Separate-verifier seam](#separate-verifier-seam-deepswe)).
4. **`hb__<environment_name>`** — the locally-built pier/harbor convention.

## Pier vs harbor deltas

The cluster overrides (acquire, exec, file transfer, capacity pacing, compose)
port over from the harbor adapter essentially verbatim. The behavioral
differences are:

**`type()` returns a string.** Pier's `BaseEnvironment` declares `type()`
abstract as `-> str` precisely so third-party environments can return an
arbitrary identifier. The adapter returns `"xrlenv-cluster"`. (The harbor
adapter keyed off the harbor `EnvironmentType` enum — pier dropped that.)

**`agent_install_spec` cleared in cluster mode.** The cluster does not build
an agent-pre-installed image. Pier's installed-agent path treats a matching
`agent_install_spec` as "agent already baked in" and skips its runtime
`install()` — which would leave the agent binary absent on the cluster
container. `XrlenvPierEnvironmentCluster.__init__` drops `agent_install_spec`
and advertises `capabilities.preinstall_agents=False`, so pier runs each
installed agent's runtime `install()` via the adapter's `exec`. The
`XrlenvPierEnvironment` (local-Docker mode) keeps it — local mode can
genuinely build a pre-installed image. Moot for the `OracleAgent` (it has no
install spec); required for the installed-agent / egress path.

**Separate-verifier seam.** See the dedicated section below.

**`capabilities.filtered_egress=True` (Squid egress proxy).** Pier validates that
an offline task carrying an agent `network_allowlist` is matched by
`filtered_egress`. The adapter reproduces pier's Squid egress-proxy sidecar on
the cluster compose path: an offline task (`allow_internet=False`) with a
non-empty `network_allowlist` acquires a two-service compose — `main` on an
`internal: true` network (no direct egress) with the proxy its only route out,
the proxy on both that + the default (internet) bridge, and the allowlist passed
to squid as `dstdomain`. The proxy runs from a mirror-pullable `ubuntu:24.04` +
a runtime squid-install command (no build-on-node, no private-registry push).
pier's inherited `agent_process_env` injects the proxy `HTTP(S)_PROXY` into
**agent** commands only; verifier/task `exec` bypass it. The `OracleAgent`
carries no allowlist, so the proxy is never synthesized for the oracle gate.

## Separate-verifier seam (DeepSWE)

DeepSWE uses `environment_mode="separate"`, which tells pier to grade each
trial in a **fresh, isolated container** rather than running the grader inside
the agent container. Two details of how pier builds the verifier container
require adapter-side handling:

**No `docker_image` on the verifier env config.** DeepSWE's
`[verifier.environment]` block carries no `docker_image`. Pier in
`environment_mode="separate"` does not build the verifier's `tests/Dockerfile`
either — it passes the config to the environment's `start()` as-is. So a
plain `task_env_config.docker_image` lookup returns `None` and would fall
through to the nonexistent `hb__<env>` tag. The adapter detects a verifier
session by the `__verifier__` marker in `session_id` and resolves the base
image via, in order:

1. The `FROM` line of `<environment_dir>/Dockerfile` (the grader's base image,
   e.g. `public.ecr.aws/d3j8x8q7/swe-bench-202605:<id>-v1.1`).
2. The parent task's top-level `[environment] docker_image` from the sibling
   `task.toml`.

**`skip_tests_upload=True` hardcoded in pier.** Pier builds a verifier
`Verifier(skip_tests_upload=True)` on the assumption the grader is baked into
the verifier image. DeepSWE's verifier image is the prebuilt ECR base — `test.sh`
is not on it. The adapter reproduces the `tests/Dockerfile` COPY by uploading
the task's `tests/` directory (minus the `Dockerfile` itself) to `/tests`
inside the container on `start()`. This makes `/tests/test.sh` and the
supporting grader files present before pier runs the verifier.

Neither of these requires a sweep flag or a task change. The adapter handles
both automatically whenever it detects a verifier session.

## Capacity pacing and backpressure

When you run more concurrent workers than the cluster can immediately place,
XRLEnv uses fail-fast backpressure rather than silently queuing indefinitely.
Each `acquire_container` call waits up to `XRLENV_PIER_ACQUIRE_QUEUE_TIMEOUT_S`
(default `240.0` s) for a capacity slot. If no slot opens in that window, the
acquire raises `CapacityExhausted` and the trial queue retries via
`_INFRA_RETRY_EXCEPTIONS`. The acquire is pier's first setup step, so the
**dominant** retry case is a fail-fast acquire *before* `solve.sh` runs — no
container or agent setup is wasted, and the task usually runs once. A retry
re-runs the WHOLE trial in a FRESH container, so a *post-acquire* infra error
(e.g. a `NodeCommandTimeout` on an exec) can re-execute the task body; the
recorded statistics are still **one outcome per task** (re-graded), so this
matters only for external side effects, not the reported result.

The timeout is deliberately shorter than pier's own agent-setup cancel window
(~360 s). This ordering guarantees that a paced acquire surfaces as a
retriable `CapacityExhausted` rather than pier's non-retriable cancel.

**Two operator levers:**

1. **Raise `XRLENV_PIER_ACQUIRE_QUEUE_TIMEOUT_S`** (export before launching —
   no CP redeploy). Keep it comfortably below ~360 s.
2. **Retry caller-side.** The oracle sweep's `_INFRA_RETRY_EXCEPTIONS` set
   already does this. A fresh acquire re-queues from scratch.

## Advanced container configuration

`XrlenvPierEnvironmentCluster` accepts additional kwargs via
`environment.kwargs` in the job config. These are stripped before the kwargs
reach pier's constructor and forwarded to `acquire_container`:

| kwarg | Type | Description |
|---|---|---|
| `xrlenv_cap_add` | `list[str]` | Linux capabilities to add (e.g. `["NET_ADMIN", "SYS_ADMIN"]`). Allowed by default. |
| `xrlenv_devices` | `list[str]` | Host devices to expose (e.g. `["/dev/loop0:/dev/loop0"]`). |
| `xrlenv_privileged` | `bool` | Run the container `--privileged`. Default-denied; requires operator opt-in in `nodes.yaml`. |
| `xrlenv_cpu_pinning` | `bool` | Confine the container to a cpuset sized to `ceil(cpus)`. Useful on large hosts where `nproc` would otherwise report the full host core count. |
| `xrlenv_cpu_multiplier` | `float` | Multiply the task's effective CPU limit by this factor (default `1.0`). |
| `xrlenv_mem_multiplier` | `float` | Multiply the task's effective memory limit by this factor (default `1.0`). |

Example job config fragment:

```python
from pier.models.trial.config import EnvironmentConfig

env = EnvironmentConfig(
    import_path="xrlenv_plugins.pier:XrlenvPierEnvironmentCluster",
    kwargs={
        "xrlenv_cpu_pinning": True,
        "xrlenv_cpu_multiplier": 2.0,
    },
)
```

```{note}
`xrlenv_privileged` and `xrlenv_devices` are gated by the `nodes.yaml` policy
(`allow_privileged`, `allowed_devices`, capability denylist). The control plane
rejects requests that violate the policy.
See {doc}`/deploy/multi_node_deployment/runbook` for the policy tiers.
```

## Network and egress

For the DeepSWE oracle gate, offline tasks (`allow_internet=false`) acquire
`--network none` (loopback-only — external internet blocked). No allowlist is
needed for the oracle path.

The pier agent network-allowlist is enforced by a **Squid egress-proxy sidecar**
on the cluster compose path (`capabilities.filtered_egress=True`). When an offline
task (`allow_internet=false`) is constructed with a non-empty `network_allowlist`,
the adapter acquires a two-service compose: `main` on an `internal: true` network
(no direct egress) plus a `pier-egress-proxy` sidecar (from `ubuntu:24.04` +
runtime squid install — no build, no private-registry push) that is `main`'s only
route out and allows only the allowlisted `dstdomain`s. pier's inherited
`agent_process_env` injects the proxy into the agent's commands only; the
verifier/task commands bypass it. This mirrors pier's own Docker egress-proxy
mechanism, adapted to the cluster (no image build/push).

## DeepSWE benchmark

See {doc}`deep_swe` for the full onboarding guide, tool reference, image route
(direct public-ECR pull — no re-push), and reproduce recipe for the 113/113
oracle gate.

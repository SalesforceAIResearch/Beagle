# Python SDK reference

The public Python SDK is exposed from `xrlenv`:

```python
from xrlenv import Client, Deadline, from_env, rollout_metadata
```

Most integrations use one of two surfaces:

- `Client.acquire_container(...)` for new async Python code.
- `xrlenv.from_env()` for existing docker-py code.

Template rollouts (`Client.rollout`, `batch_rollout`, `replay`) are
still available for built-in and test environments, but they are an
advanced surface rather than the recommended starting point.

## Connect to a control plane

```python
from xrlenv import Client

client = Client.grpc(
    host="127.0.0.1",
    port=50051,
    token="<XRLENV_CONSUMER_TOKEN>",
)
```

`token` is required when the control plane runs with auth. Issue one
on the control-plane host:

```bash
xrlenv tokens issue consumer
```

For local tests that embed the control plane in the same Python
process, use `Client.in_process(runtime.service)`.

## Managed container sessions

`Client.acquire_container(...)` asks the control plane for one
container. The returned `ClusterContainerSession` tracks the XRLEnv
rollout id, Docker container id, selected node id, and cleanup state.

```python
async with await client.acquire_container(
    image="ubuntu:22.04",
    command=["sleep", "infinity"],
    labels={"workflow": "demo"},
) as session:
    result = await session.exec(["bash", "-lc", "echo hello"], timeout_s=10)
    print(result.exit_code)
    print(result.stdout.decode())
```

### `Client.acquire_container(...)`

| Parameter | Type | Description |
|---|---|---|
| `image` | `str` | Image reference to run, for example `ubuntu:22.04`. |
| `command` | `list[str] \| None` | Container command. Long-lived workflows usually use `["sleep", "infinity"]`. |
| `name` | `str \| None` | Optional Docker container name. |
| `labels` | `dict[str, str] \| None` | Labels stored on the raw rollout record and passed through to Docker. |
| `environment` | `dict[str, str] \| None` | Environment variables for the container. |
| `task_key` | `str \| None` | Scheduler fairness key for related acquires. |
| `ensure_image_present` | `bool` | When true, the selected node pulls or builds the image if needed. |
| `cpu_limit` | `float \| None` | Effective CPU limit (cores). A scheduling input: the cluster places the container on a node that can satisfy it, enforces a CFS quota, and pins `ceil(cpu_limit)` dedicated cores. `None` uses the raw-container default (2.0). |
| `mem_limit_bytes` | `int \| None` | Effective memory limit (bytes), same scheduling-input semantics. `None` uses the default (4 GiB). |
| `runtime_limits` | `RuntimeLimits \| None` | Container-shape limits (pids / shm / tmpfs / read-only rootfs / cpu pinning) that do **not** affect scheduling. `None` applies no constraint. |

See {doc}`/technical_details/resource_isolation` for the full resource
model and the operator-side knobs.

**`RuntimeLimits` fields** (all optional; unset = Docker default):

| Field | Type | Default | Description |
|---|---|---|---|
| `pids_limit` | `int \| None` | `None` | Maximum number of processes inside the container. |
| `shm_size_bytes` | `int \| None` | `None` | `/dev/shm` size override. |
| `tmpfs` | `dict[str, str]` | `{}` | tmpfs mounts: `{"/path": "options"}`. |
| `readonly_rootfs` | `bool` | `False` | Mount the container rootfs read-only. |
| `cpu_pinning` | `bool` | `False` | Pin the container to `ceil(cpu_limit)` dedicated host cores (cpuset). When `False`, the container receives a CFS quota only (burstable across all cores). Opt in only for timing-sensitive workloads; the harbor adapter leaves this `False` by default. |

### `ClusterContainerSession`

| Member | Description |
|---|---|
| `rollout_id` | XRLEnv id for this container lifecycle. |
| `container_id` / `container_name` | Docker identity on the selected node. |
| `node_id` | Node selected by the scheduler. |
| `exec(cmd, timeout_s=..., cwd=..., env=..., user=...)` | Run a command and return full stdout/stderr when it exits. |
| `exec_stream(cmd, timeout_s=..., cwd=..., env=..., user=...)` | Async iterator of stdout/stderr chunks for long-running commands. |
| `put_archive(target_dir, tarball)` | Upload a tar archive into the container. |
| `get_archive(source_path)` | Download a tar archive from the container. |
| `apply_egress(allowlist, *, dns_resolver=None)` | Install an iptables egress policy in the container's netns. `allowlist` is an `EgressAllowlist`; empty = block all external egress. Fail-closed: partial apply destroys the container. Raises `XRLEnvError` for shared-netns or privileged containers. See {doc}`security` for narrative and examples. |
| `destroy(force=True)` | Destroy the container. Idempotent. |

Use the session as an async context manager whenever possible; it
destroys the container on exit.

## Admin metadata hooks

Existing harnesses often write their own artifact directory. Use
`rollout_metadata(...)` to make that directory visible in the admin
panel:

```python
import xrlenv

with xrlenv.rollout_metadata(
    displayed_name="task-123",
    artifact_path="/path/to/my/task-123",
):
    async with await client.acquire_container(image="ubuntu:22.04") as session:
        ...
```

`displayed_name` appears in `/rollouts/raw`; `artifact_path` becomes
a link on the raw-rollout detail page.

## Docker SDK drop-in

`xrlenv.from_env()` mirrors the ergonomics of `docker.from_env()`.
When `XRLENV_GRPC_HOST` is set, it creates a cluster-backed Docker
client. When it is not set, it falls back to local Docker mode.

```python
import xrlenv

client = xrlenv.from_env()
container = client.containers.run(
    image="ubuntu:22.04",
    command=["bash", "-lc", "echo hello"],
    detach=True,
)
```

Environment variables:

| Variable | Description |
|---|---|
| `XRLENV_GRPC_HOST` | Control-plane host; setting this enables cluster mode. |
| `XRLENV_GRPC_PORT` | Control-plane port, default `50051`. |
| `XRLENV_CONSUMER_TOKEN` | Bearer token for authenticated control planes. |
| `XRLENV_GRPC_SECURE` | `true`, `1`, `yes`, or `on` for TLS. |

See
{doc}`/build_with_xrlenv/work_with_xrlenv_managed_containers/docker_py_dropin`.

## Template rollout API

Template rollouts drive an environment through a step loop and seal a
trajectory. This is useful for built-in templates, tests, and
advanced integrations that deliberately use XRLEnv's `EnvAdapter`
protocol.

```python
from xrlenv import Deadline

async with await client.rollout(
    template="hello-shell",
    init={"prompt": "hello"},
    deadline=Deadline(hard_s=60),
) as session:
    while not session.done:
        await session.step({"cmd": "echo hello"})

trajectory = session.trajectory
```

### `Client.rollout(...)`

| Parameter | Type | Description |
|---|---|---|
| `template` | `str` | Registered template name, for example `hello-shell`. |
| `init` | `dict \| None` | Template-specific setup data. |
| `deadline` | `Deadline \| None` | Per-rollout time budget. |
| `request_id` | `str \| None` | Idempotency key. |
| `task_key` | `str \| None` | Fairness key used by the scheduler. |
| `group_id` | `str \| None` | Opaque group label for group cancellation. |
| `reward_fn` | callable \| `None` | Required only for templates that declare `consumer_final` reward mode. |

### `RolloutSession`

`RolloutSession.step(action)` returns `StepResult` with `obs`,
`reward`, `done`, `truncated`, and `info`. `session.heartbeat()`
refreshes the idle timer without sending an action. `session.finish()`
seals the rollout and makes `session.trajectory` available.

## Batch, cancel, and replay

`Client.batch_rollout(...)` runs multiple template rollouts with a
concurrency limit and returns a `BatchRolloutResult` grouped into
`finished`, `truncated`, and `failed`.

`Client.cancel_rollout(rollout_id)` cancels one running rollout and
returns its sealed trajectory. `Client.cancel_group(group_id)`
cancels every non-terminal rollout carrying the group id.

`Client.replay(rollout_id)` reads a sealed trajectory by id. Local
deployments read from the run directory; multi-node deployments can
fetch from the owning node through the control plane.

## Node readiness

Use `wait_for_nodes(...)` when a script starts before nodes have
attached:

```python
await client.wait_for_nodes(min_nodes=1, timeout_s=30)
```

It polls `list_nodes()` until enough nodes are connected or the
timeout expires.

## Egress types

`EgressAllowlist` and `EgressRule` are importable from
`xrlenv.backends.egress`:

```python
from xrlenv.backends.egress import EgressAllowlist, EgressRule
```

| Type | Description |
|---|---|
| `EgressRule(cidr, ports=None)` | One allowed destination. `cidr` is an IPv4 CIDR string. `ports` is a tuple of destination port integers, or `None` to allow all ports. |
| `EgressAllowlist(rules=())` | Ordered set of `EgressRule`. An empty `EgressAllowlist()` means block all external egress. |

## Errors

| Exception | Module | Description |
|---|---|---|
| `XRLEnvError` | `xrlenv.errors` | Base for all XRLEnv runtime errors. |
| `FleetOverBudget` | `xrlenv.errors` | A fleet companion acquire would exceed the fleet's declared cpu/mem footprint. Raised at the control plane before any node command; the fleet's other containers and its reservation are untouched. The acquire path degrades gracefully (the task is not hard-failed; see commit a93cfb6). Fix: declare a larger `xrlenv.fleet_cpu_request` / `xrlenv.fleet_mem_request` in the fleet spec. |

## Public exports

The top-level package lazily exports:

- `Client`, `Template`
- `Deadline`, `Trajectory`, `Step`, `StepResult`
- `BatchRolloutResult`, `FailedRollout`, `CancelGroupReport`
- `RolloutStatus`
- `from_env`
- `rollout_metadata`

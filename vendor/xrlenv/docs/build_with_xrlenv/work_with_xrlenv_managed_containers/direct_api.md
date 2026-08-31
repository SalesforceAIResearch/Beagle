# Direct API for new code

If you're writing new code and don't need docker-py compatibility,
the direct API is the smallest surface that gets you a remote
container. It's an async shape (the underlying gRPC transport is
async).

## Connection

```python
from xrlenv import Client

client = Client.grpc(host="127.0.0.1", port=50051, token="<token>")
```

For ad-hoc connection params pass them explicitly. To pick up the
same env-var protocol the docker-py drop-in uses, see
{doc}`docker_py_dropin` — set `XRLENV_GRPC_HOST` etc. and use
`xrlenv.from_env()` for the universal Docker SDK shape.

## A worked example

```python
import asyncio
import io
import tarfile

from xrlenv import Client


async def main():
    client = Client.grpc(host="127.0.0.1", port=50051)

    # Acquire a container scoped to a fresh rollout. The
    # `async with` ensures destroy fires reliably on exception.
    async with await client.acquire_container(
        image="ubuntu:22.04",
        command=["sleep", "infinity"],
        labels={"my.workflow": "demo"},
    ) as session:

        # Run a command (batched — small output, short timeout).
        result = await session.exec(
            ["bash", "-c", "ls /"],
            timeout_s=30,
        )
        print(f"exit={result.exit_code}")
        print(result.stdout.decode())

        # Stream a long-running command (chunks as they arrive).
        async for chunk in session.exec_stream(
            ["bash", "-c", "for i in 1 2 3; do echo $i; sleep 1; done"],
            timeout_s=1800,
        ):
            if chunk.stdout:
                print(chunk.stdout.decode(), end="")
            if chunk.done:
                print(f"final exit={chunk.exit_code}")

        # Copy bytes in.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            info = tarfile.TarInfo(name="hello.txt")
            payload = b"hello\n"
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
        await session.put_archive(
            target_dir="/work",
            tarball=buf.getvalue(),
        )

        # Copy bytes out.
        tarball = await session.get_archive("/work/hello.txt")
        # ... un-tar locally with tarfile.

    # `destroy` fires automatically on context exit. To be explicit:
    # await session.destroy()

    await client.close()


asyncio.run(main())
```

## Recipes

| Want | Code |
|---|---|
| Acquire | `async with await client.acquire_container(image=..., command=..., labels=...) as session:` |
| Run a command | `await session.exec([...], timeout_s=30, cwd="/work", env={"FOO": "bar"}, user="agent")` |
| Stream a long command | `async for chunk in session.exec_stream([...], timeout_s=1800):` |
| Copy bytes in | `await session.put_archive(target_dir="/work", tarball=...)` |
| Copy bytes out | `tarball = await session.get_archive("/path/in/container")` |
| Destroy explicitly | `await session.destroy()` |

## Resource limits

`acquire_container` accepts the container's resource limits directly.
They are not just enforced at runtime — `cpu_limit` / `mem_limit_bytes`
are *scheduling inputs*: the cluster places the container on a node
that can satisfy them.

```python
from xrlenv.backends.base import RuntimeLimits

async with await client.acquire_container(
    image="my-grader:1",
    command=["sleep", "infinity"],
    cpu_limit=4.0,                       # 4 CPU: CFS quota + 4 pinned cores
    mem_limit_bytes=8 * 1024**3,         # 8 GiB hard memory cap
    runtime_limits=RuntimeLimits(        # container-shape, scheduling-neutral
        pids_limit=4096,
        shm_size_bytes=256 * 1024**2,
    ),
) as session:
    ...
```

- `cpu_limit` (cores) — a CFS quota **and** `ceil(cpu_limit)` dedicated
  host cores pinned to the container, so timing-sensitive tests aren't
  perturbed by CPU interleaving.
- `mem_limit_bytes` — a hard memory cgroup limit.
- `runtime_limits` — pids / shm / tmpfs / read-only rootfs. Optional;
  applied at creation, no scheduling effect.

Omitting them uses the raw-container default budget (2.0 CPU,
4&nbsp;GiB). See {doc}`/technical_details/resource_isolation` for the
full model, what's rejected, and the operator-side knobs.

## Admin UX hooks

To make your rollout show up nicely in the admin `/rollouts/raw`
view, wrap your work in `xrlenv.rollout_metadata(...)`:

```python
import xrlenv

with xrlenv.rollout_metadata(
    artifact_path="/path/where/this/workflow/saves/outputs",
    displayed_name="my-job-step-7",
):
    async with await client.acquire_container(...) as session:
        ...
```

`displayed_name` shows up as the row name; `artifact_path`
becomes a navigation pointer on the per-rollout detail page.

## OCI runtime selection (`container_runtime`)

`container_runtime` selects the OCI runtime for the container. **Which value do
I use?**

| I want… | Pass | Result |
|---|---|---|
| A **normal container** (the default — Terminal-Bench, SWE-bench, most tasks) | *nothing* — omit `container_runtime` (or `None`) | Docker's default runtime (`runc`); the path is completely unchanged |
| The same, but explicit | `container_runtime="runc"` | Identical to the default — `runc` is never treated as an override |
| **Docker-in-Docker**, `systemd` as PID 1, or kernel namespaces (`ip netns`, `iptables`, `ptrace`) — **unprivileged** | `container_runtime="sysbox-runc"` | Runs under [Sysbox](https://github.com/nestybox/sysbox); requires an operator-provisioned Sysbox node pool (see below) |

There is no separate "enable DinD" flag — **`sysbox-runc` *is* how you get
Docker-in-Docker** (plus systemd and netns), all without `--privileged`. Any
other runtime string is rejected unless the operator explicitly allow-lists it.

```python
# Normal container — no runtime argument (this is 99% of tasks):
async with await client.acquire_container(image="swe-task:1") as session:
    ...

# Docker-in-Docker / systemd / netns — opt into Sysbox:
async with await client.acquire_container(
    image="my-dind-task:1",
    command=["sleep", "infinity"],
    container_runtime="sysbox-runc",
) as session:
    # Inner Docker, systemd, ip netns, iptables, ptrace are available.
    result = await session.exec(["docker", "run", "--rm", "hello-world"])
    print(result.stdout.decode())
```

**What the cluster does with this:**

- **Policy check.** `KwargsPolicy.allowed_runtimes` on the cluster is empty by
  default — every non-`runc` runtime override is rejected with a clear error
  until an operator explicitly adds `sysbox-runc` to `nodes.yaml`
  `policy.allowed_runtimes`.
- **Placement.** The scheduler routes the acquire only to nodes whose Docker
  advertises `sysbox-runc` in `supported_runtimes`. If a runtime-capable node
  exists but is momentarily at capacity, the acquire queues until capacity frees.
  If **no connected node advertises the runtime at all** (e.g. `allowed_runtimes`
  was enabled before the Sysbox install/reconnect finished), the acquire **fails
  loud** with `BackendCapabilityMissing` rather than queueing — install Sysbox on
  a pool node (so it advertises `sysbox-runc` on reconnect) and retry.
- **Node verification.** Before `docker run`, the node confirms the runtime is
  registered. If it isn't, the node fails loudly — no silent fall-back to `runc`.
- **Init process.** A sysbox acquire skips the injected `tini` (`--init`) so
  `systemd` or another inner init can be PID 1. The normal path keeps `init=True`
  (zombie reaping) unchanged.
- **Egress restriction.** `apply_egress` refuses sysbox containers — the inner
  root controls its own network namespace and could flush `iptables` rules. Sysbox
  containers are always internet-on. Do not use `container_runtime="sysbox-runc"`
  for offline/egress-restricted grading tasks.

The normal `runc` path (no `container_runtime` argument) is byte-for-byte
unchanged — Terminal-Bench and SWE-bench Verified are unaffected.

**Prerequisites.** The operator must have set up a Sysbox node pool. See
{doc}`/deploy/multi_node_deployment/sysbox_pool` for the build / install /
`nodes.yaml` guide.

## Security: opting in to `userns-remap`

`acquire_container` defaults to `userns_mode="host"` — the
container runs with host UIDs (in-container root = host root).
This default keeps benchmark tasks working: lots of them assume
their agent runs as root inside the container and can write to
host-bind-mounted paths.

For workloads that don't need in-container root, opt in to the
docker daemon's `userns-remap` config on a per-acquire basis:

```python
async with await client.acquire_container(
    image="my-isolated-task:1",
    userns_mode="remap",   # honor daemon's userns-remap
) as session:
    ...
```

**Defense-in-depth.** When the daemon has `userns-remap`
configured, `userns_mode="remap"` makes the container's
processes run as a subordinate UID on the host even though they
appear to be root inside the container. If a process escapes
(kernel bug, mount race, etc.), it lands as an unprivileged user
on the host instead of host root.

**Prerequisites.** The daemon must have `userns-remap`
configured in `/etc/docker/daemon.json` separately — xrlenv
doesn't write that file. Without daemon-level config,
`userns_mode="remap"` is a silent no-op (the daemon ignores the
empty override and runs the container with host UIDs as usual).

**Why opt-in, not default.** Most benchmark images need
in-container root — for installing packages, chmod-ing
bind-mounted paths, running test harnesses that assume
privileged operations work. Defaulting to remap would silently
break them. Operators with a security-tighter posture can pick
`"remap"` per-acquire when they know the image doesn't need
in-container root.

## Secrets: getting API keys into the container

Real agentic workloads usually want multiple API keys inside the
container (one for the LLM, plus per-service keys for web search,
retrieval, GitHub, etc.). `xrlenv.client` ships two harbor-agnostic
helpers that cover the two shapes operator code typically expects.

```python
from xrlenv import Client
from xrlenv.client import parse_dotenv, upload_dotenv

# Shape 1 — keys as container env vars at creation time.
env = parse_dotenv(".env")
async with await client.acquire_container(
    image="ubuntu:22.04",
    command=["sleep", "infinity"],
    environment=env,        # ANTHROPIC_API_KEY, OPENAI_API_KEY, ... set inside.
) as session:
    # Shape 2 — copy the .env file itself into the container
    # (in addition to, or instead of, env vars). For in-container
    # tools that auto-load dotenv vs reading env directly.
    landed_at = await upload_dotenv(
        session, source=".env", target_dir="/workspace",
    )
    # landed_at == "/workspace/.env"
    ...
```

`parse_dotenv(path)` returns a `dict[str, str]`:

- Accepts `KEY=value`, `KEY="quoted value"`, `KEY='quoted value'`,
  and `export KEY=value` (the `export ` prefix is stripped).
- Skips comments (`#`) and blank lines.
- **Does not expand variables** — `KEY=$OTHER` lands as the literal
  `$OTHER`. The dict you get is exactly what the container sees;
  there's no surprise interpolation.
- Skips malformed lines silently (matches `set -a; source .env`'s
  forgiving behavior). Validate the returned dict against your
  expected key set if you want strictness.
- Raises `FileNotFoundError` when the source path is missing —
  fail-fast prevents an operator silently proceeding without
  secrets.

`upload_dotenv(session, source, target_dir="/workspace",
arcname=".env", mkdir=True)` copies the file via
`session.put_archive`:

- Pre-runs `mkdir -p <target_dir>` (Docker's `put_archive` returns
  a 404 if the target dir doesn't exist; the pre-mkdir closes that
  gotcha by default — pass `mkdir=False` to opt out).
- Returns the in-container path the file landed at, so you can log
  it or thread it into a follow-on `exec`.
- Raises `FileNotFoundError` if the local source isn't a file.

`load_dotenv(*, path=None, override=False)` populates **your
process's** `os.environ` from a `.env` file — the operator-side
counterpart to the in-container `upload_dotenv`. Useful when a
script needs the operator's secrets *before* connecting to the
control plane (e.g. reading `XRLENV_CONSUMER_TOKEN` from `.env` to
construct `Client.grpc(...)`). Returns the dict of keys actually
applied so callers can log what changed:

```python
from xrlenv.client import load_dotenv

applied = load_dotenv()          # discover .env upward from CWD
applied = load_dotenv(path="config/dev.env", override=True)

# Operators almost never need this directly: `import xrlenv` already
# calls `load_dotenv()` once at package import time. The explicit
# function is for explicit-reload scenarios (test fixtures, multi-env
# scripts) or for callers who pass a non-default `path=`.
```

- **Discovery (path=None)**: walks from `Path.cwd()` upward to the
  first `.env`. Returns `{}` when no file is found (silent).
- **Precedence**: existing `os.environ` values win unless
  `override=True`. Shell-exported vars beat the file by default —
  matches the import-time auto-load's contract documented in the
  {doc}`installation guide </getting_started/installation>`.
- **Always re-runs**: each call walks the filesystem and re-applies.
  Direct callers can invoke `load_dotenv()` repeatedly after editing
  `.env` between runs and expect every call to pick up the latest
  contents (subject to the precedence rule above — pass
  `override=True` if the keys are already in `os.environ`). The
  idempotency guard that keeps `import xrlenv` from re-walking on
  every re-import is internal to the package-init hook
  (`_maybe_auto_load_dotenv`); it does NOT short-circuit explicit
  `load_dotenv()` calls.

**Picking a shape.** Use Shape 1 (env vars) when your in-container
code reads from `os.environ` / `$VAR` — most agentic CLIs work this
way out of the box. Use Shape 2 (file copy) when a tool specifically
expects to find a `.env` file on disk (some `python-dotenv` configs,
frameworks that watch a fixed path). The two shapes are
independent and combinable; nothing forces a choice.

**Harbor / installed-agent caveat.** Harbor's installed-agent
classes own the `exec` calls inside the container and forward their
own allowlist of env vars (claude-code forwards `ANTHROPIC_API_KEY`
automatically, but not arbitrary custom keys). For Pattern-2
workloads driven by `harbor.Job.run()`, the cleanest workaround is
`set -a; source .env; set +a` in the operator shell before
launching the Job — harbor inherits the env, and each installed
agent forwards what it knows about. For full arbitrary-key
forwarding, subclass the harbor agent and override its
env-forwarding (harbor-side, not xrlenv-side).

## Fleet reservation for multi-container tasks

Some tasks spin up more than one container — a primary "agent" container and one
or more companion containers (e.g. a database, a browser, a tool server). Without
coordination, the scheduler places each container independently and they can end
up on different nodes, or the second container can be blocked by a full cluster
even though the node running the first container has room.

**Fleet reservation** is an opt-in mechanism that atomically reserves the full
resource footprint (CPU + memory) of all containers in the group on a single node
before any of them start. The opener declares the peak footprint; companions join
by referencing the same fleet id. Because the reservation pre-empts capacity
atomically, companions co-locate on the same node with no risk of oversubscription.

Fleet reservation is activated through labels on `acquire_container`. Omitting
the labels leaves behavior unchanged — this is purely additive.

```python
import asyncio
import uuid
from xrlenv import Client

async def main():
    client = Client.grpc(host="127.0.0.1", port=50051)

    # Pick a stable fleet id for this task run.
    fleet_id = f"my-task-{uuid.uuid4()}"

    # Opener: declare the TOTAL footprint for ALL containers in the fleet.
    # The scheduler reserves this on one node atomically before the first
    # container starts.
    async with await client.acquire_container(
        image="agent:latest",
        command=["sleep", "infinity"],
        cpu_limit=2.0,
        mem_limit_bytes=4 * 1024**3,
        labels={
            "xrlenv.fleet_id": fleet_id,
            "xrlenv.fleet_cpu_request": "6.0",   # total peak across all containers
            "xrlenv.fleet_mem_request": str(12 * 1024**3),  # 12 GiB total
        },
    ) as agent_session:

        # Companion: join the same fleet by fleet_id only (no footprint labels).
        # Placement is drawn from the reservation and guaranteed on the same node.
        async with await client.acquire_container(
            image="postgres:15",
            command=["postgres"],
            cpu_limit=2.0,
            mem_limit_bytes=4 * 1024**3,
            labels={
                "xrlenv.fleet_id": fleet_id,
            },
        ) as db_session:
            # Both containers now run on the same node.
            result = await agent_session.exec(["bash", "-c", "echo ready"])
            print(result.stdout.decode())

    await client.close()

asyncio.run(main())
```

**How it works:**

- The opener (`xrlenv.fleet_id` + both `xrlenv.fleet_cpu_request` and
  `xrlenv.fleet_mem_request` labels) atomically reserves the declared footprint
  on one node. It must succeed before any companion is launched.
- Companions (`xrlenv.fleet_id` only, no footprint labels) are drawn from that
  reservation and are guaranteed co-location.
- `fleet_id` is a distinct identity axis: it is separate from `task_key`
  (fairness) and from rollout `instance_id` (deduplication). Generate a fresh id
  per task run; reusing a fleet id across runs is a consumer ordering bug.

**On overflow:** If companions exceed the declared footprint (the task's actual
peak concurrency was higher than predicted), the acquire degrades gracefully —
the over-budget container is placed via the normal capacity-gated path on
whichever node has room, rather than hard-failing the acquire. A
`raw-container.coordinator.fleet-overflow` warning is logged; raise the declared
footprint labels to avoid it.

The scheduling and operator-side internals are covered in
{doc}`/technical_details/scheduling`.

### Admission queue during acquire

When the cluster is at capacity, `acquire_container` does **not** fail — it
blocks in the admission queue (default timeout 24 hours). While blocked, the
SDK logs a line every ~5 seconds:

```
INFO  acquire queued — position N of M in the cluster admission queue; waiting for capacity (not an error)
```

A long block is not a hang. `N` and `M` update in real time as slots open and
other requests are served. The acquire will proceed as soon as capacity is
available.

Timeout semantics (what happens if no capacity appears within the deadline) are
documented in {doc}`/developer_guide/timeouts`.

## When to pick this over the drop-in

- You're writing fresh code and want the cleanest async surface.
- You don't need docker-py compatibility (you're not porting
  existing code, your harness doesn't use docker-py internally).
- You want to manipulate `ClusterContainerSession` directly —
  e.g. one acquire that runs many `exec_stream` calls back to
  back, or finer control over `rollout_metadata` scoping.

If you're porting docker-py code, the drop-in
({doc}`docker_py_dropin`) is the smaller change.

## See also

- {doc}`/supported_benchmarks_and_harnesses/harbor_framework` —
  the harbor adapter is a subclass-override pattern that wraps
  the same `ClusterContainerSession` you're using here directly.
- {doc}`docker_py_dropin` — the alternative shape for code
  already using docker-py.

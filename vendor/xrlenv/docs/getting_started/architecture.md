# How XRLEnv works

XRLEnv is a small control plane for Docker-backed sandbox work. Your
code asks for work to run in an isolated container; XRLEnv chooses a
node, starts the container, tracks lifecycle and artifacts, and
exposes the result through the SDK, CLI, admin panel, metrics, and
logs.

The important boundary is simple: benchmark logic stays in your
workflow or upstream harness. XRLEnv owns container placement and
lifecycle.

## The runtime model

There are three moving pieces:

- **Your workflow** calls the Python SDK, the Docker SDK drop-in, or a
  framework/harness adapter.
- **The control plane** accepts requests, records state, schedules
  work, serves the admin panel, and exposes metrics.
- **Node daemons** run on Docker-capable hosts. Each node reports
  capacity and image state, then creates, execs into, and destroys
  containers when the control plane assigns work.

````{height-limit}
:height: 360px

```{mermaid}
flowchart LR
    workflow["Your workflow<br/>SDK, docker-py drop-in, or adapter"]
    control["Control plane<br/>scheduler, state, admin, metrics"]
    nodeA["xrlenv-node<br/>Docker host A"]
    nodeB["xrlenv-node<br/>Docker host B"]
    containerA["Managed containers"]
    containerB["Managed containers"]

    workflow <-->|"gRPC"| control
    nodeA <-->|"node-initiated stream"| control
    nodeB <-->|"node-initiated stream"| control
    nodeA --> containerA
    nodeB --> containerB
```
````

Node daemons dial out to the control plane. The control plane does
not need to open an inbound connection to each node, which keeps
cloud VM networking straightforward.

## The three supported workflow shapes

### Direct managed-container API

Use `Client.acquire_container(...)` when you are writing new async
Python code. The returned session gives you `exec`, streaming `exec`,
archive upload/download, and `destroy`:

```python
async with await client.acquire_container(
    image="ubuntu:22.04",
    command=["sleep", "infinity"],
) as session:
    result = await session.exec(["bash", "-lc", "echo hello"])
```

See {doc}`/build_with_xrlenv/work_with_xrlenv_managed_containers/direct_api`.

### Docker SDK drop-in

Use `xrlenv.from_env()` when your code or upstream harness already
uses docker-py. The goal is a one-line import change while the
harness keeps using `containers.create`, `exec_run`, `put_archive`,
`get_archive`, and image methods.

See {doc}`/build_with_xrlenv/work_with_xrlenv_managed_containers/docker_py_dropin`.

### Framework/harness adapter

Use a framework/harness adapter when a benchmark already has its own
environment interface. The adapter subclasses that interface and only
replaces the container-touching methods. Harbor/terminal-bench is the
current example: `XrlenvHarborEnvironmentCluster` keeps Harbor in
charge of tasks, agents, verifier output, and reports, while XRLEnv
runs the containers.

See {doc}`/supported_benchmarks_and_harnesses/harbor_framework`.

## Deployment shapes

The same SDK code works in both deployments:

- **Single host** runs the control plane and one node daemon on the
  same machine. This is what `xrlenv up` does by default and is the
  best way to develop or run a smoke.
- **Multi-node** runs the control plane on one host and node daemons
  on cloud VMs. Your workflow points at the control plane over gRPC.

See {doc}`/deploy/index`.

## What XRLEnv records

Every acquired container or template rollout gets an XRLEnv rollout
id. The platform records status, node, image, timestamps, labels,
artifact pointers, and lifecycle events. Workflow-specific outputs
remain where the workflow writes them; for example, a benchmark
harness can use `xrlenv.rollout_metadata(...)` to tell the admin
panel where its per-task artifact directory lives.

For template-based rollouts, XRLEnv also stores a sealed trajectory
that can be replayed later. Template rollouts are an advanced API
surface; they are not the primary path for new docs.

## Where to go next

- {doc}`quickstart` — run one local managed container.
- {doc}`/build_with_xrlenv/index` — write code against XRLEnv.
- {doc}`/supported_benchmarks_and_harnesses/index` — run supported
  benchmark harnesses.
- {doc}`/observability/admin_panel` — inspect a live cluster.

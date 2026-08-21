# Quickstart

Five minutes, one local host, one container. By the end of this
walkthrough you'll have:

1. Booted an xrlenv control plane on your local host.
2. Acquired one container through the cluster's direct API.
3. Run a command inside it and read the output.
4. Cleanly destroyed the container.

This is the smallest possible end-to-end exercise of the platform.
For benchmark-driven workflows (SWE-bench, terminal-bench-2), see
{doc}`/supported_benchmarks_and_harnesses/index`.

## Prerequisites

You've completed {doc}`/getting_started/installation`:

- Python 3.12+ + Docker Engine 24+ reachable as the current user.
- `.venv/` set up via `uv sync`.

Verify Docker:

```bash
docker info >/dev/null && echo "ok"
```

## Boot the control plane

In one shell:

```bash
.venv/bin/xrlenv up
```

You'll see a JSON log line like:

```text
{"ts": "...", "level": "INFO", "event": "control_plane.boot",
 "grpc_host": "127.0.0.1", "grpc_port": 50051, ...}
```

Leave this running.

In a second shell, verify the cluster is reachable:

```bash
.venv/bin/xrlenv nodes
```

You should see one node attached (the local host's `xrlenv-node`
the daemon spawns alongside `xrlenv up`).

## Acquire a container + run a command

Save this as `quickstart.py`:

```python
import asyncio
from xrlenv import Client


async def main():
    # Connect to the control plane on the local host.
    client = Client.grpc(host="127.0.0.1", port=50051)

    # Acquire a container scoped to a fresh rollout. The async-with
    # ensures destroy fires even if something below raises.
    async with await client.acquire_container(
        image="ubuntu:22.04",
        command=["sleep", "infinity"],
        labels={"quickstart.demo": "true"},
    ) as session:
        print(f"acquired container {session.container_id[:12]} on node {session.node_id}")

        # Run a command inside it.
        result = await session.exec(
            ["bash", "-c", "echo hello && uname -a"],
            timeout_s=10,
        )
        print(f"exit_code={result.exit_code}")
        print(result.stdout.decode())

    # Container is destroyed at this point.
    await client.close()


asyncio.run(main())
```

Run it:

```bash
.venv/bin/python quickstart.py
```

Expected output:

```
acquired container abc123def456 on node local-host
exit_code=0
hello
Linux abc123def456 6.2.0-...
```

## Inspect the rollout in the admin panel

While `xrlenv up` is still running, open `http://127.0.0.1:8080/`
in a browser. Navigate to **rollouts → raw containers**. You'll see
the rollout from your acquire — name, status (`released` after
destroy), node, container short-id, image, duration, age.

Click the row to see per-rollout details: labels, lifecycle
timestamps, and (if the script set them) artifact-path pointer.

## Stop the control plane

In the first shell, `Ctrl-C` to stop `xrlenv up`. Trajectory
artifacts persist under `~/.xrlenv/runs/<date>/<rollout_id>/`.

## What just happened

Three things layered together:

::::{height-limit}
:height: 250px

```{mermaid}
flowchart LR
    Script["<b>quickstart.py</b><br/><span style='font-size:11px'>your code on local host</span>"]
    Control["<b>xrlenv up</b><br/><span style='font-size:11px'>control plane<br/>(127.0.0.1:50051)</span>"]
    Node["<b>xrlenv-node</b><br/><span style='font-size:11px'>data plane on same host</span>"]
    Docker["<b>Docker</b><br/><span style='font-size:11px'>local engine</span>"]

    Script <==>|"gRPC"| Control
    Control <==>|"node-initiated stream"| Node
    Node -->|"container ops"| Docker

    classDef workflow fill:#ede7f6,stroke:#5e35b1,stroke-width:1.5px,color:#000;
    classDef control fill:#e3f2fd,stroke:#1565c0,stroke-width:1.5px,color:#000;
    classDef data fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.5px,color:#000;
    classDef external fill:#fff8e1,stroke:#ef6c00,stroke-width:1.5px,color:#000;

    class Script workflow;
    class Control control;
    class Node data;
    class Docker external;
```
::::

`xrlenv up` boots both the control plane (gRPC server on `:50051`
+ admin panel on `:8080`) and a single `xrlenv-node` daemon
attached to the local Docker engine. Your `quickstart.py` connects
as client code over gRPC; the **control plane** picks the only
attached node and forwards each operation; the **data plane**
(node daemon) talks to Docker; the container is real.

For multi-host deployments the only difference is that
`xrlenv-node` runs on separate cloud VMs and dials back to the
control plane. Your script's code stays exactly the same — it
still connects to the control-plane gRPC endpoint.

## Next steps

- {doc}`/build_with_xrlenv/work_with_xrlenv_managed_containers/direct_api`
  — full recipe set for the direct API (`exec_stream`, `put_archive`, `get_archive`,
  `rollout_metadata` for admin UX hooks).
- {doc}`/build_with_xrlenv/work_with_xrlenv_managed_containers/docker_py_dropin`
  — same primitives, drop-in shape for code already using docker-py.
- {doc}`/supported_benchmarks_and_harnesses/writing_your_own_adapter` — pre-wired adapters for SWE-bench + terminal-bench-2.
- {doc}`/deploy/multi_node_deployment/index` — set up a real multi-node cluster on cloud VMs.
- {doc}`/observability/admin_panel` — the full admin tour.

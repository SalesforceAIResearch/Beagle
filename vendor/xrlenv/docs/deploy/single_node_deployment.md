# Single-host deployment

A single-host deployment runs your workflow, the control plane, and
the node daemon all on one machine. This is the shape the
{doc}`/getting_started/quickstart` uses; it's also what local
integration tests and development work expect.

::::{height-limit}
:height: 200px

```{mermaid}
flowchart LR
    Workflow["<b>Your workflow</b><br/><span style='font-size:11px'>script, harness, or SDK</span>"]
    Up["<b>xrlenv up</b><br/><span style='font-size:11px'>control plane + xrlenv-node<br/>:50051 / :8080 / :9090</span>"]
    Docker["<b>Docker Engine</b><br/><span style='font-size:11px'>local sandboxes</span>"]

    Workflow <==>|"gRPC 127.0.0.1"| Up
    Up -->|"container ops"| Docker

    classDef workflow fill:#ede7f6,stroke:#5e35b1,stroke-width:1.5px,color:#000;
    classDef control fill:#e3f2fd,stroke:#1565c0,stroke-width:1.5px,color:#000;
    classDef external fill:#fff8e1,stroke:#ef6c00,stroke-width:1.5px,color:#000;

    class Workflow workflow;
    class Up control;
    class Docker external;
```
::::

## Bring it up

```bash
.venv/bin/xrlenv up
```

This single command boots both the control plane and one
`xrlenv-node` daemon attached to the local Docker engine. The
process exposes:

- **gRPC `127.0.0.1:50051`** — SDK and Docker SDK drop-in API.
- **HTTP `127.0.0.1:8080`** — admin panel
  ({doc}`/observability/admin_panel`).
- **HTTP `127.0.0.1:9090/metrics`** — Prometheus exposition
  ({doc}`/observability/metrics`).

Customize ports / state-db location via flags; see
{doc}`/developer_guide/cli_reference`.

## Connect your workflow

```python
from xrlenv import Client

client = Client.grpc(host="127.0.0.1", port=50051)
```

If `xrlenv up` was started with auth, set
`XRLENV_CONSUMER_TOKEN` (or pass `token=` explicitly) — issue the
token first via `xrlenv tokens issue consumer`. Auth defaults to
**off** for single-host loopback runs.

## Cross-platform notes

On macOS, the in-sandbox stub auto-falls back to TCP transport
because Docker Engine on macOS doesn't route Unix-domain-socket
bind-mounts through the host↔VM bridge. On Linux, the default is
UDS (lower overhead). No configuration needed — handled by
`DockerBackendConfig.stub_transport`.

## When to graduate to multi-node

- **Concurrency outgrows one machine.** A single local host can
  run ~4–8 sandboxes comfortably; serious evaluation runs need
  8+ VMs.
- **Image distribution becomes the bottleneck.** A multi-node
  cluster's image-affinity scheduler spreads cold-pull cost across
  nodes; one machine pays it serially.
- **The cluster needs to outlive your shell session.** A control
  plane on a dedicated host plus durable state is more reliable
  than `xrlenv up` in a tmux pane.

When you're ready, see {doc}`multi_node_deployment/index`.

## See also

- {doc}`/getting_started/quickstart` — first-run walkthrough.
- {doc}`multi_node_deployment/index` — multi-node topology.
- {doc}`/developer_guide/cli_reference` — `xrlenv up` flags.
- {doc}`multi_node_deployment/runbook` — `xrlenv bootstrap` usage and the
  `XRLENV_PYTHON` / `XRLENV_USER` / `XRLENV_VERSION` env-var equivalents
  (in the **Reference: bootstrap env-var equivalents** table under
  **Bootstrap node VMs**).

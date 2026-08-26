# Multi-node deployment

A multi-node xrlenv deployment runs the control plane on one host
and `xrlenv-node` daemons on N data-plane VMs. The current scripts
assume manually provisioned GCP or AWS VMs: no Terraform, managed
instance groups, or autoscaling required.

::::{height-limit}
:height: 320px

```{mermaid}
flowchart TB
    subgraph CP["Control-plane host"]
        direction TB
        Up["<b>xrlenv up</b><br/><span style='font-size:11px'>gRPC :50051 · /metrics :9090 · admin :8080</span>"]
    end

    subgraph GCP["GCP VM"]
        direction TB
        GNode["<b>xrlenv-node serve</b>"]
        GSbx["Docker sandboxes<br/><span style='font-size:11px'>in-sandbox stub (UDS / TCP)</span>"]
        GNode --> GSbx
    end

    subgraph AWS["AWS EC2"]
        direction TB
        ANode["<b>xrlenv-node serve</b>"]
        ASbx["Docker sandboxes<br/><span style='font-size:11px'>in-sandbox stub (UDS / TCP)</span>"]
        ANode --> ASbx
    end

    GNode -.->|"outbound bidi gRPC"| Up
    ANode -.->|"outbound bidi gRPC"| Up

    classDef control fill:#e3f2fd,stroke:#1565c0,stroke-width:1.5px,color:#000;
    classDef node    fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.5px,color:#000;
    classDef sandbox fill:#fff8e1,stroke:#ef6c00,stroke-width:1.5px,color:#000;
    class Up control;
    class GNode,ANode node;
    class GSbx,ASbx sandbox;
```
::::

The node transport is **outbound-only**: each `xrlenv-node` daemon
initiates the bidi stream to the control plane. The control plane
never opens an inbound connection to a node. This works correctly
behind NAT or restrictive egress firewalls — only the control-plane
host needs an inbound port reachable by the data-plane VMs and by
workflow hosts that dial into the cluster.

## What you'll do

1. Bring up data-plane VMs on your cloud provider — see
   {doc}`cloud_VM_providers/index`.
2. Declare the nodes in a `nodes.yaml` inventory file — see
   {doc}`inventory`.
3. Walk the end-to-end six-step deployment runbook —
   {doc}`runbook`.

## See also

- {doc}`/deploy/single_node_deployment` — same code path, single
  host, useful for development before paying for cloud VMs.

```{toctree}
:maxdepth: 2

inventory
cloud_VM_providers/index
runbook
registry_mirror
private_registry
scratch_registry
sysbox_pool
slurm_topology
```

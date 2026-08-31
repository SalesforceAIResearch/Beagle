---
orphan: true
---

# Deploy

XRLEnv has two deployment shapes. The SDK code you write is the same
in both; only where the control plane and node daemons run changes.

| Shape | Use it when | Start here |
|---|---|---|
| Single host | You are developing locally, running a small smoke, or validating an integration before using cloud VMs. | {doc}`single_node_deployment` |
| Multi-node | You have one control-plane host and one or more Docker-capable cloud VMs. | {doc}`multi_node_deployment/index` |

Deployment references:

- {doc}`single_node_deployment`
- {doc}`multi_node_deployment/index`
- {doc}`multi_node_deployment/inventory`
- {doc}`multi_node_deployment/cloud_VM_providers/index`
- {doc}`multi_node_deployment/runbook`
- {doc}`multi_node_deployment/registry_mirror`
- {doc}`multi_node_deployment/private_registry`

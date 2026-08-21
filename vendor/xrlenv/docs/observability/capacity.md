# Capacity and image state

Capacity signals answer two questions:

1. Which nodes can accept more containers?
2. Which images are already present, pinned, cold, or causing disk
   pressure?

## Admin views

| View | What it shows |
|---|---|
| `/capacity` | Per-template capacity **planning estimate** (not live health). |
| `/nodes` | Roster and connectivity; all-time rollout distribution across nodes. |
| `/sandboxes` | Containers backing template (case-1) rollouts; raw workloads create none. |
| `/images/cache` | Per-node image cache state and disk pressure. |
| `/images/catalog` | Distinct image refs and cluster-wide coverage. |
| `/builds` | Persisted image build/distribution plans. |

## CLI tools

Plan image placement when the image set is known ahead of time:

```bash
xrlenv images plan --refs image-refs.txt --eager-prefetch
```

Pre-pull individual images:

```bash
xrlenv warmup ubuntu:22.04
```

Apply a build plan:

```bash
xrlenv build apply --plan build-plan.yaml
```

For small runs, you can skip prefetching. XRLEnv pulls an image onto
the selected node on first acquire when `ensure_image_present=True`.

## Eviction

Each node's image cache evicts cold images by least-recently-used
order when disk pressure crosses the configured threshold. Images in
use are protected from eviction.

See {doc}`/technical_details/images/index` for the mechanics and
{doc}`/technical_details/scheduling` for how image state affects node
selection.

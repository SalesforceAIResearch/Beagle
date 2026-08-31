# Image management

XRLEnv treats image placement as part of scheduling. A node can run a
container only after the image is present locally, so the platform
tracks what each node has, pulls or builds missing images when
allowed, and uses cache state when choosing a node.

This section has three pages, ordered the way an operator
encounters the system:

- **On-demand image acquire** is the default runtime path: someone
  calls `acquire_container(image=...)`, the scheduler picks a node,
  and the node pulls or builds the image on miss. Start here — it's
  what happens when no planning is done at all.
- **Image cache and eviction** picks up where on-demand leaves off:
  acquires fill the per-node cache, and under disk pressure the
  cache reclaims space using a rebuild-cost-aware LRU. The
  {ref}`operator-driven evict command <xrlenv-images-evict>` is also
  documented here.
- **Registry tag freshness model** explains how the control plane
  resolves a mutable channel tag (`:dev`, `:stable`) to the registry's
  current digest at acquire time — so a rebuilt+re-pushed image reaches
  nodes without any consumer config change.
- **Image distribution and build planning** is the proactive
  optimization: when you know your image set ahead of time, you
  can prefetch and steer placement instead of paying first-acquire
  latency.
- **Bring-your-own-Dockerfile** is the self-service alternative to
  operator pre-builds: supply a Dockerfile in your template and the
  platform builds it on demand, once for the fleet, drift-free.


```{toctree}
:maxdepth: 1

on_demand
cache_eviction
registry_freshness
build_plan
bring_your_own_dockerfile
```

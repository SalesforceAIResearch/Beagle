# Registry tag freshness model

Docker's `ensure_present` short-circuits when an image is already local
— even when the registry tag was re-pushed with new content. This is
correct for immutable references like digest pins, but creates a silent
staleness hazard for **mutable tags**: `substrate:dev` rebuilt and
re-pushed is byte-for-byte different than what the node holds, yet the
node never re-pulls it because the tag string is unchanged.

The freshness model closes this gap. At raw-container-acquire time the
control plane resolves a registry-qualified **tag** to the registry's
current **digest** and dispatches the digest to the node. The node
materializes the digest-pinned image: if the rebuilt image has a new
digest, the node treats it as absent and pulls fresh. The node never
runs unintended stale content silently.

This page covers the freshness model internals for operators who need
to tune it, and explains the interaction with {doc}`cache_eviction` and
the {doc}`build_plan` channel-tag workflow.

## How it works

For every `raw_container` acquire the control plane calls
`RegistryDigestResolver.resolve(image_ref)`:

1. **Pass-through cases.** A ref that is already digest-pinned
   (`repo@sha256:…`) or is not registry-qualified (bare `name:tag`,
   Docker-Hub-relative `library/x:tag`) is returned unchanged. Nothing
   changes for workflows that already use explicit digests.
2. **Registry-qualified tag.** The resolver parses out `(host, repo,
   tag)`. If a fresh-enough cached resolution exists (within
   `XRLENV_REGISTRY_RESOLVE_TTL_S`, default 60 s) it is served
   immediately without a network round-trip — a burst of acquires for
   the same tag collapses into one registry probe.
3. **Registry probe.** On a cache miss, the resolver sends
   `HEAD /v2/<repo>/manifests/<tag>` to `<scheme>://<host>` with
   `Accept` headers for both OCI and Docker manifest formats and reads
   the `Docker-Content-Digest` response header. If the registry does
   not surface the digest on `HEAD` (403, 405, or missing header) it
   falls back to a `GET`. The resolved digest ref is cached.
4. **Dispatch.** The resolved `host/repo@sha256:…` ref replaces the
   tag in the `AcquireContainer` command sent to the node. The digest
   is also recorded on the raw session so every run is auditable.

```{note}
The freshness model is a **control-plane feature**. The node sees only
the digest ref the CP dispatches; no node code changed. An older
node-agent with a new CP resolves correctly. An older CP with a new
node-agent retains the mutable-tag behavior until the CP is updated.
```

## Failure semantics

The resolver distinguishes two failure classes:

| Failure class | Example | Resolver behavior |
|---|---|---|
| Transient (registry unreachable) | network blip, registry restart | Serve the **last-known-good** digest if it was resolved within `XRLENV_REGISTRY_RESOLVE_MAX_STALE_S` (default 900 s / 15 min). Otherwise raise `RegistryResolveError` and fail the acquire. |
| Permanent (4xx — tag missing or unauthorized) | tag deleted, wrong repo name | Raise `RegistryResolveError` immediately. **Never** serve a stale digest for a tag that the registry says does not exist. |

A `RegistryResolveError` surfaces to the consumer as a failed acquire.
The stale-window design preserves digest pinning and auditability across
a brief registry outage without stalling a whole training run, while
ensuring the control plane never silently runs an unverifiable image
once the window lapses.

## Configuration knobs

All knobs are read from the `xrlenv up` process environment at startup.

| Env var | Default | Description |
|---|---|---|
| `XRLENV_REGISTRY_DIGEST_RESOLVE` | `1` (on) | Kill-switch. Set to `0` / `false` / `off` to disable the freshness model entirely. Acquires pass the ref verbatim — the legacy mutable-tag behavior. Use this for clusters whose CP cannot reach the registry, or for non-HTTP registries not covered by the resolver. |
| `XRLENV_REGISTRY_SCHEME` | `http` | `http` or `https`. Use `http` for the private insecure registry on port 5011; `https` for public or TLS-terminated registries. |
| `XRLENV_REGISTRY_RESOLVE_TTL_S` | `60` | Seconds a cached resolution is considered fresh. A burst of acquires for the same tag within this window costs one registry probe. |
| `XRLENV_REGISTRY_RESOLVE_MAX_STALE_S` | `900` | On a transient registry outage, serve the last-known-good digest for up to this many seconds before failing the acquire. |
| `XRLENV_REGISTRY_RESOLVE_HOST_MAP` | (unset) | Comma-separated `ref-host:port=dial-host:port` pairs. Rewrites **only** the host:port the control plane dials when probing the manifest. The recorded digest ref always keeps the original ref host, so remote nodes pull from the externally routable address and digest pinning is unchanged. Malformed entries are skipped (not fatal). See [Co-located control plane and registry](#co-located-control-plane-and-registry). |

```{note}
`XRLENV_REGISTRY_SCHEME` applies to the manifest probe only. Node pulls
still use whatever scheme the node's Docker daemon is configured for
(the `insecure-registries` entry in `daemon.json` — see
{doc}`/deploy/multi_node_deployment/private_registry`).
```

## Co-located control plane and registry

The default production topology runs the control plane and the private
registry on the **same box**. The registry is addressed in image refs by
that box's external hostname or IP (e.g. `<registry-host>:5011/...`) so
that remote worker nodes can pull from it. The control plane's freshness
resolver therefore probes the registry by that same external address.

On some managed infrastructure (verified on SageMaker HyperPod / EFA
nodes), a box cannot reliably reach its own externally published Docker
port via its own external name. The observed failure path: Docker's `nat`
`OUTPUT` chain DNATs `<own-ip>:5011` toward the registry container, but
the packet's source address remains the box's own IP, which matches the
managed agent's policy-routing rule (`from <own-ip> lookup 101`). Table
101 has no route to the Docker bridge subnet, so the packet is sent out
the physical NIC to the VPC gateway instead of to `docker0` — a
connection timeout. The general principle is simpler: **a co-located
control plane often cannot reach its own externally published port; probe
loopback instead.** `curl http://localhost:5011/...` works fine on the
same box, which confirms the registry itself is healthy.

**Symptom.** Acquires fail with:

```
RegistryResolveError: cannot resolve '<registry-host>:5011/...' to a
digest: registry unreachable (ConnectTimeout: ) and no last-known-good
digest within 900s
```

while `curl http://localhost:5011/v2/<repo>/tags/list` on the registry
box returns the tags normally. Worker nodes are unaffected — off-box
pulls never take the hairpin path.

**Fix.** Point the control plane's manifest probe at loopback for refs
whose registry host is this box:

```bash
export XRLENV_REGISTRY_RESOLVE_HOST_MAP="<registry-host>:5011=127.0.0.1:5011"
```

The digest ref the resolver returns still carries `<registry-host>:5011`,
so remote nodes pull from the correct external address — only the
control plane's inbound probe is redirected.

**The shipped Slurm scripts handle this automatically.** Both
`slurm_scripts/generated/prod_xrlenv_control.sh` and
`slurm_scripts/generated/dev_xrlenv_control.sh` build and export a self-tuning map
before `exec xrlenv up`:

```bash
_hostmap=""
for _n in "$(hostname -s)" "$(hostname -I | awk '{print $1}')"; do
    for _p in 5011 5010; do
        _hostmap="${_hostmap:+$_hostmap,}${_n}:${_p}=127.0.0.1:${_p}"
    done
done
export XRLENV_REGISTRY_RESOLVE_HOST_MAP="$_hostmap"
```

This populates the map with all of this box's own names on both registry
ports. If the registry is co-located, image refs carry this box's name
and the entries match — the CP probes loopback. If the registry is on a
different box, no image ref names this box as the registry host and the
entries never match — the CP dials the external address directly. A fresh
co-located deploy using these scripts needs no manual configuration step.

**Why not an iptables hairpin rule?** On HyperPod the managed agent
re-applies table 101 on every boot, so a hairpin rule added after boot is
ephemeral and rots silently. The loopback override is structurally immune
(loopback is excluded from the `OUTPUT` DNAT entirely) and requires no
elevated privileges beyond setting an environment variable.

See also {doc}`/deploy/multi_node_deployment/private_registry` for the
registry server setup and the broader multi-node topology.

## What passes through unchanged

The resolver is conservative: when in doubt it does nothing.

- **Digest refs** (`host:5011/repo@sha256:…`): already pinned, no probe.
- **Bare refs** (`name:tag`, `my-repo:v1`): no registry host segment,
  no probe. Docker resolves these against its configured mirrors /
  Docker Hub.
- **Docker-Hub-relative repos** (`library/python:3.12`, `ubuntu:22.04`):
  the first path segment is not a host (no `.` / `:` / `localhost`),
  no probe.

Only `host:port/repo:tag` and `hostname.tld/repo:tag` shapes are
probed — the shapes used by XRLEnv's private registry.

## Interaction with `xrlenv images evict`

The freshness model is the **routine** path: every acquire for a
channel tag automatically pins the current digest. The evict command
is the **proactive cleanup** escape hatch: when you want nodes to drop
a cached old digest immediately rather than waiting for a new acquire
to pull in the updated digest.

Typical rebuild workflow:

```bash
# 1. Rebuild and push the new image under the channel tag.
.venv/bin/python deploy/registry/build_and_push_images.py \
    --plan xrlenv_plugins/benchmarks/webarena_infinity/build_plan.yaml \
    --registry <REGISTRY_HOST>:5011 --registry-scheme http --force

# 2. Optional: evict the old cached image from nodes immediately.
#    Without this, nodes serve the old image for any in-flight rollout
#    and pull the new digest on the next acquire.
xrlenv images evict xrlenv-webarena-infinity/substrate:dev \
    --connect-host <admin-host>

# 3. Next acquire dispatches the new digest automatically (no config change).
```

If you skip step 2, nodes will continue to serve the old digest for
in-flight rollouts (which is correct — evict would disrupt them).
New acquires after the resolver's TTL expires (`XRLENV_REGISTRY_RESOLVE_TTL_S`,
default 60 s) will see the new digest.

## Auditability

The resolved digest is recorded on the raw session. In the admin panel,
the `/rollouts/<id>` view shows the exact digest the node materialized.
A moving channel tag `:dev` is therefore fully auditable: you always
know which digest ran for each rollout, even after the tag has been
re-pushed.

## Deployment caveat

The freshness model lives in the control plane (`RegistryDigestResolver`
in `xrlenv/control/registry_resolver.py`). A cluster running old
control-plane code retains the legacy mutable-tag behavior regardless
of node-agent version. Update the control plane with `xrlenv up` (new
code) to activate the freshness model; no node-agent restart is needed.

## See also

- {doc}`cache_eviction` — operator-driven `xrlenv images evict` for
  proactive cache cleanup.
- {doc}`build_plan` — channel-tag scheme and the rebuild workflow.
- {doc}`/supported_benchmarks_and_harnesses/webarena_infinity` —
  webarena-infinity uses `:dev` / `:stable` channel tags and relies on
  this model to propagate rebuilds.

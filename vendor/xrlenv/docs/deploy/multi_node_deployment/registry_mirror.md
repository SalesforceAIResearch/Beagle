# Registry mirror (optional)

By default each node pulls benchmark images straight from Docker Hub.
That works, but it has two costs at cluster scale:

- **Rate limits.** Anonymous Docker Hub allows ~100 pulls per 6 hours
  per source IP; even with per-node Docker Hub auth (see the runbook's
  **Bootstrap node VMs** step), a large sweep funnels a lot of traffic upstream.
- **Redundant re-pulls.** When a node evicts a cold image under disk
  pressure and later needs it again, it re-fetches the whole thing from
  Docker Hub — once per node, every time.

An **optional pull-through registry mirror** removes both. It is a
single `registry:3` process running in *proxy (pull-through cache)*
mode in front of Docker Hub, with its blob store on a shared filesystem
(NFS / FSx / Lustre). The first cluster-wide pull of an image caches its
layers on that shared store; every later pull — including after a node
evicts the image — is served from the mirror over the LAN. Cache misses
and a down mirror fall back to Docker Hub automatically, so the mirror
is a bounded-risk accelerator, not a new hard dependency.

It needs **no code or scheduler changes**. Node pulls already go through
`dockerd`, and Docker's `registry-mirrors` setting applies transparently
to `docker.io` image references — which is exactly the benchmark images.

:::{note}
The mirror is an accelerator layered under the per-node image cache, not
a replacement for it. Each node still keeps its own local Docker image
cache and evicts cold images under disk pressure
({doc}`/technical_details/images/cache_eviction`); the mirror only makes
the *re-pull after eviction* cheap. You do not need the mirror to run a
cluster — turn it on when Docker Hub pulls become a bottleneck.
:::

## Two roles — keep them straight

A registry mirror has a **server** side and a **client** side, and they
run on different machines. The single most common mistake is running the
registry on a worker; don't.

| | **Server (the mirror)** | **Client (each worker node)** |
|---|---|---|
| What it does | Runs the `registry:3` proxy; caches layers on the shared store. | Points its Docker daemon's pulls at the mirror URL. |
| Where it runs | The **control-plane host** or a **dedicated registry VM** — **never a worker**. | **Every** worker node. |
| What it touches | Starts one container; writes blobs to the shared store. | Edits `/etc/docker/daemon.json` only. Runs **no** registry. |
| Script | `deploy/registry/run-registry-mirror.sh` | `deploy/registry/configure_docker_registry.sh` |

The client side is pure daemon config — it adds one `registry-mirrors`
URL. **No registry process ever runs on a worker.**

## Bring up the mirror (server — once, on the control-plane host)

Run on the control-plane host (or a dedicated registry VM that all
workers and the shared store can reach):

```bash
bash deploy/registry/run-registry-mirror.sh
```

In the bundled Slurm scripts this runs from
`slurm_scripts/generated/prod_xrlenv_control.sh` (alongside `run-registry-private.sh`), so
the designated registry box brings up both registries when the prod control plane
starts. Other clusters point at it via `.env` and don't start their own.

The script reads its config from a single `.env` file (repo-root `.env`
by default; override with `REGISTRY_ENV_FILE`). All keys are optional:

| Key | Default | Purpose |
|---|---|---|
| `DOCKERHUB_USER` / `DOCKERHUB_TOKEN` | unset (anonymous) | Upstream Docker Hub auth for the proxy. **The same two keys the node bootstrap and `refresh.sh` already use** — no separate secrets file. Anonymous works but Docker Hub rate-limits it, and the mirror funnels the whole cluster through one identity, so set a Pro/Team [Personal Access Token](https://docs.docker.com/security/for-developers/access-tokens/) here **before** warming the full image set. |
| `XRLENV_MIRROR_REGISTRY_STORAGE` | `/fsx/home/$USER/xrlenv-registry/proxy` | Blob-store path. **Must be a shared mount** (NFS / FSx / Lustre) reachable cluster-wide. This is the registry blob store only — never point it at the Docker data-root. *(Deprecated alias: `XRLENV_REGISTRY_STORAGE` — still accepted with a warning.)* |
| `XRLENV_MIRROR_REGISTRY_PORT` | `5010` | Host port the mirror listens on. *(Deprecated alias: `XRLENV_REGISTRY_PORT`.)* |

Re-running the script any time re-applies changed `.env` values: it
recreates the container, and the shared blob store persists across
restarts.

:::{note}
**Cache retention** is config-driven via `proxy.ttl` in
`deploy/registry/config-mirror.yml` (default 90 days). A clean redeploy keeps
cached content for that long with no manual steps. Changing the TTL only
affects *future* pulls — to re-stamp content already cached, run
`deploy/registry/restamp_registry_ttl.sh`.
:::

Verify the mirror is up from the control-plane host:

```bash
curl -s http://127.0.0.1:5010/v2/ && echo "  mirror OK"
```

## Point workers at the mirror (client — on each worker node)

### Existing workers

Run once per worker, via `sudo`:

```bash
sudo MIRROR_URL=http://<control-plane-ip>:5010 \
    bash deploy/registry/configure_docker_registry.sh --restart
```

This merges a `registry-mirrors` entry into `/etc/docker/daemon.json`
without clobbering existing keys (such as the relocated `data-root`),
then restarts `dockerd` **only if the config actually changed** (which
bounces running containers — the script lists them first) and restarts
`xrlenv-node`. It is idempotent: re-running with the same URL is a no-op
and does not bounce Docker a second time.

Drop `--restart` to write the config without restarting Docker; the
daemon picks it up on its next restart.

A later `deploy/refresh.sh` does **not** touch `daemon.json`, so the
mirror setting persists across xrlenv upgrades — you do not re-run this
on every release.

### New workers (set it at bootstrap)

Export `XRLENV_REGISTRY_MIRROR` in the **same shell that runs the
bootstrap**, alongside the node token and Docker Hub credentials, and
the bootstrap applies the identical client config for you:

```bash
sudo \
    XRLENV_NODE_TOKEN='<paste-token-here>' \
    DOCKERHUB_USER='<your-docker-hub-handle>' \
    DOCKERHUB_TOKEN='<dckr_pat_...>' \
    XRLENV_REGISTRY_MIRROR='http://<control-plane-ip>:5010' \
    bash deploy/bootstrap-aws.sh --hyperpod <control-plane-ip>:50051 "aws-$(hostname -s)"
```

The two positional args are the **control-plane gRPC address** and this
**worker's node id** — *not* a second copy of the control-plane address. Pass the
node id as `aws-$(hostname -s)` so it matches the roster `xrlenv nodes-from-slurm`
generates; drop `--hyperpod` if the node isn't a SageMaker HyperPod instance. In
practice the bundled `slurm_scripts/{dev,prod}_xrlenv_node.sh` issue this exact
command (with both `XRLENV_REGISTRY_MIRROR` and `XRLENV_PRIVATE_REGISTRY`) for
every worker — see {doc}`private_registry` (the **Allow plain-HTTP push and pull**
step).

`XRLENV_REGISTRY_MIRROR` is an operator-set shell variable read **once,
at bootstrap time** — it is not stored in inventory and is not
auto-discovered. Leave it unset to keep a node pulling straight from
Docker Hub. The value is the mirror's full URL including the port from
**Bring up the mirror** above (`http://<host>:5010`). The bootstrap merges it into
`daemon.json` (Docker reads it on first start); it does **not** start a
registry on the worker.

:::{tip}
To make the setting survive re-provisioning, put the `export
XRLENV_REGISTRY_MIRROR=...` line in whatever provisioning script or
`EnvironmentFile` you use to launch the bootstrap, next to where you set
`XRLENV_NODE_TOKEN`. Unlike the node token, the bootstrap does not
persist `XRLENV_REGISTRY_MIRROR` itself — but the `daemon.json` edit it
produces *does* persist on the node, so you only need it set on the run
that bootstraps a fresh VM.
:::

## Warm the cache (optional pre-fill)

By default the mirror fills lazily on the cluster's first real pulls.
To pre-fill it before a sweep — so the first rollouts don't pay the
upstream-fetch latency — warm it from a build plan:

```bash
.venv/bin/python deploy/registry/warm_images.py <build_plan.yaml> --concurrency 16
```

This pulls each image's manifest and blobs **through the mirror's
registry API**, so the content lands on the shared store with no local
Docker extraction — it does not `docker pull`, so it neither fills the
warming box's Docker store nor bypasses the mirror. Run it on any box
that can reach the mirror (on the control-plane host it hits
`127.0.0.1:5010`); it needs no Docker Hub credentials of its own (the
mirror handles upstream auth, so set the mirror's PAT in **Bring up the mirror** first or
a large fetch will hit the anonymous rate limit). It is idempotent and
resumable — it skips blobs already on the store, so re-running after an
interruption only fetches what's missing.

**All flags:**

| Flag | Default | Purpose |
|---|---|---|
| `--concurrency N` | `8` | Parallel image streams. Raise for faster warming on a fast shared store; lower if the store is the bottleneck. |
| `--mirror <url>` | `http://127.0.0.1:5010` | Override the mirror URL. Useful when warming from a box that isn't the control-plane host. |
| `--store-path <path>` | `$XRLENV_MIRROR_REGISTRY_STORAGE` or `/fsx/home/$USER/xrlenv-registry/proxy` | Override the blob-store root used for the skip-existing check. Set this when the env var is absent or you want to target a non-default store. |
| `--no-skip` | off | Re-stream every blob even if it is already present on the store. **Cache repair path** — use together with `--store-path` when blobs on the store are known to be corrupt or incomplete. |
| `--limit N` | `0` (no limit) | Warm only the first N images from the plan. Useful for spot-checks or phased warming. |

The `--store-path` / `--no-skip` pair is the cache-repair path: point
`--store-path` at the live store root so the script can target the right
location, then pass `--no-skip` to force a full re-stream of every blob
regardless of whether the file already exists on disk.

## `restamp_registry_ttl.sh` — re-stamp TTL on existing cached content

`deploy/registry/restamp_registry_ttl.sh` rewrites the retention expiry on all
content already cached in the mirror's scheduler state, in place. Normally
you don't need this: the mirror's `proxy.ttl` setting (in
`deploy/registry/config-mirror.yml`, default 90 days) governs the retention of
*new* pulls, and a clean redeploy retains existing content for the configured
period. This script is for the niche case where you want to change the
policy for **content already cached** — for example, bumping 90d to 180d
immediately, or recovering from a registry version that did not honour the
config.

**Usage** (run on the registry host, as the user who owns the repo):

```bash
bash deploy/registry/restamp_registry_ttl.sh          # re-stamp to now + 90d (default)
bash deploy/registry/restamp_registry_ttl.sh 180      # re-stamp to now + 180d
```

**What it does:**

1. Stops the mirror container (the registry holds scheduler state in memory;
   editing the state file while it runs would be lost on the next flush).
2. Backs up `<store>/scheduler-state.json`.
3. Rewrites every `ExpiryData` field to `now + N days` via an inline Python
   script (requires `sudo` for the root-owned state file).
4. Restarts the mirror by re-running `deploy/registry/run-registry-mirror.sh`.

The restart incurs a brief (~10 s) pull-through outage; workers fall back to
Docker Hub automatically during this window.

**Cron example** (re-extend monthly so the cache never lapses):

```bash
0 4 1 * *  cd /path/to/xrlenv && \
    bash deploy/registry/restamp_registry_ttl.sh 90 >> /var/log/xrlenv-restamp.log 2>&1
```

## Operational notes

- **Availability.** If the mirror is down, `dockerd` falls back to
  Docker Hub automatically. Bounded risk — a slow or missing mirror
  degrades to today's behavior, it doesn't fail pulls.
- **Sizing.** Deletes are off (the whole point is to keep a warm cache).
  Size the shared store for the full compressed image set you intend to
  cache.
- **Concurrency.** Run exactly **one** mirror instance over a given
  shared-store path. Do not run two replicas over the same path.
- **Scope.** A `registry-mirrors` entry applies only to `docker.io`
  pulls. Images referenced by a non-Docker-Hub registry (or pulled by a
  box that has no `registry-mirrors` entry, such as the mirror host
  itself) bypass the mirror — that's expected.

## See also

- {doc}`runbook` — the end-to-end multi-node bring-up; its **Bootstrap node VMs**
  step covers the per-node Docker Hub auth that the mirror complements.
- {doc}`/technical_details/images/cache_eviction` — the per-node image
  cache the mirror sits under.
- `deploy/registry/README.md` in the source tree — the maintainer-facing
  reference for the same scripts.

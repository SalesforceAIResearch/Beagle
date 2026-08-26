# Scratch registry (build-on-demand)

The **scratch registry** (`:5012`) is the third registry that runs alongside
the pull-through mirror (`:5010`) and the durable private registry (`:5011`).
Unlike the private registry — which holds images an operator built and pushed
ahead of time — the scratch registry is populated **on demand at rollout time**:
when a template carries an `image_build:` block instead of a prebuilt `image:`,
the platform builds the Dockerfile on the node that picks up the work, pushes the
result to `:5012`, and every other node pulls it from there over the LAN.

The scratch registry is **quota-bounded and garbage-collected**. Build-on-demand
traffic never grows `XRLENV_PRIVATE_REGISTRY_STORAGE`. The GC is run on a
schedule (section 4 below) and always exempts digests that active runs still
reference.

For the user-facing template syntax, see
{doc}`/technical_details/images/bring_your_own_dockerfile`.

## Three registries at a glance

| | **Mirror** | **Private** | **Scratch** |
|---|---|---|---|
| Port | `:5010` | `:5011` | `:5012` |
| Script | `run-registry-mirror.sh` | `run-registry-private.sh` | `run-registry-scratch.sh` |
| Storage env | `XRLENV_MIRROR_REGISTRY_STORAGE` | `XRLENV_PRIVATE_REGISTRY_STORAGE` | `XRLENV_SCRATCH_REGISTRY_STORAGE` |
| Mode | Pull-through cache of docker.io | Writable, durable, operator-pushed | Writable, ephemeral, GC'd |
| Client config | `registry-mirrors` (docker.io routing) | `insecure-registries` (named refs) | `insecure-registries` (named refs) |
| Miss behavior | Falls back to Docker Hub automatically | **No fallback** | **No fallback** |
| GC | None (TTL-kept, deletes off) | Manual offline GC | Automatic TTL + quota GC |
| Docs | {doc}`registry_mirror` | {doc}`private_registry` | This page |

All three can run on the **same box** — they use distinct ports and distinct FSx
subdirectories and never overlap.

## Bring up the scratch registry (server — once, on the registry host)

Run on the control-plane host (or a dedicated registry VM):

```bash
bash deploy/registry/run-registry-scratch.sh
```

The script reads its config from the repo-root `.env` by default (override with
`REGISTRY_ENV_FILE`). All keys are optional:

| Key | Default | Purpose |
|---|---|---|
| `XRLENV_SCRATCH_REGISTRY_STORAGE` | `/fsx/home/$USER/xrlenv-registry/scratch` | FSx blob-store path. **Must be a shared mount** (NFS / FSx / Lustre) so pushed images are visible cluster-wide. Never point it at the Docker data-root. |
| `XRLENV_SCRATCH_REGISTRY_PORT` | `5012` | Host port the scratch registry listens on. |
| `XRLENV_SCRATCH_REGISTRY_HTTP_SECRET` | unset | A stable shared upload secret. Only needed when running two or more registry replicas behind a load balancer. Single-instance deploys omit this. |

Re-running the script any time re-applies changed `.env` values: it recreates
the container, and the FSx blob store persists across restarts.

Verify the registry is up from the registry host:

```bash
curl -s http://127.0.0.1:5012/v2/ && echo "  scratch registry OK"
```

:::{note}
**Running more than one cluster?** Follow the same pattern as the private
registry: run the scratch registry on **one** designated registry box and point
every cluster at it. Do **not** start a second instance over the same FSx blob
store (two writers on one path corrupts uploads).
:::

:::{note}
**Deletes are enabled.** This is required for GC to reclaim blobs. The scratch
registry holds only ephemeral content that the GC may reclaim at any time; the
private registry (`:5011`) holds content you want to keep.
:::

## Point the control plane at the scratch registry

Set two keys in the `.env` the control plane reads before running `xrlenv up`:

```bash
XRLENV_SCRATCH_REGISTRY_HOST=<registry-host-ip>   # or hostname
XRLENV_SCRATCH_REGISTRY_PORT=5012                  # if you kept the default, this is optional
```

`<registry-host-ip>` is the IP (or hostname) **as reachable by worker nodes**
over the LAN. The control plane forms refs of the shape
`<host>:<port>/scratch/<input_digest>` and hands them to nodes; the nodes then
push or pull from that address.

In a single-cluster setup `<registry-host-ip>` is the same box as
`<control-plane-ip>`. On a shared-registry setup where one box hosts all three
registries for multiple clusters, it is that shared box's IP.

## Allow plain-HTTP push and pull on worker nodes

Worker nodes need `<registry-host>:5012` in their Docker daemon's
`insecure-registries` list — exactly the same step as the private registry,
just with port `5012` added alongside `5011`. Pass it into bootstrap alongside
`XRLENV_PRIVATE_REGISTRY`:

```bash
sudo \
    XRLENV_NODE_TOKEN='<paste-token-here>' \
    DOCKERHUB_USER='<your-docker-hub-handle>' \
    DOCKERHUB_TOKEN='<dckr_pat_...>' \
    XRLENV_REGISTRY_MIRROR='http://<registry-host>:5010' \
    XRLENV_PRIVATE_REGISTRY='<registry-host>:5011' \
    XRLENV_SCRATCH_REGISTRY='<registry-host>:5012' \
    bash deploy/bootstrap-aws.sh --hyperpod <control-plane-ip>:50051 "aws-$(hostname -s)"
```

The bootstrap merges `insecure-registries` entries into `/etc/docker/daemon.json`
without clobbering existing keys and restarts Docker once. A later
`deploy/refresh.sh` does not touch `daemon.json`, so the setting persists across
xrlenv upgrades.

**To configure an already-running node:**

```bash
sudo PRIVATE_REGISTRY=<registry-host>:5012 \
    bash deploy/registry/configure_docker_registry.sh --restart
```

The helper accepts any `host:port` for the `insecure-registries` entry. Pass
both `5011` and `5012` together if the node is missing both:

```bash
sudo PRIVATE_REGISTRY="<registry-host>:5011 <registry-host>:5012" \
    bash deploy/registry/configure_docker_registry.sh --restart
```

## Run the GC on a schedule

`deploy/registry/scratch_registry_gc.py` reclaims scratch images by **TTL** (age out)
and **per-registry quota** (oldest-first). It **never** reclaims a digest that
an active run still references.

### Basic invocation (dry run first)

```bash
# See what would be reclaimed — nothing is deleted.
python deploy/registry/scratch_registry_gc.py \
    --registry 127.0.0.1:5012 \
    --ttl 72h \
    --quota-gb 500 \
    --exempt-url http://127.0.0.1:8080/api/scratch/active-digests \
    --dry-run

# Live run — deletes manifests and calls registry garbage-collect.
python deploy/registry/scratch_registry_gc.py \
    --registry 127.0.0.1:5012 \
    --storage-path /fsx/home/$USER/xrlenv-registry/scratch \
    --container xrlenv-registry-scratch \
    --ttl 72h \
    --quota-gb 500 \
    --exempt-url http://127.0.0.1:8080/api/scratch/active-digests
```

### Active-run exemption (required for correctness)

The `--exempt-url` flag points at the control plane's admin endpoint
`GET /api/scratch/active-digests`. The endpoint returns the set of scratch
digests that in-flight runs currently reference. The GC never deletes any of
those digests, so a mid-run GC pass cannot pull the image out from under a live
rollout.

Use operator auth when calling the endpoint from outside `localhost`:

```bash
--exempt-url "http://127.0.0.1:8080/api/scratch/active-digests"
```

Alternatively, pass `--exempt-file /run/xrlenv/scratch-active-digests.txt` if
you pre-write the digest list from another process.

:::{warning}
Running the GC without `--exempt-file` or `--exempt-url` is **safe only if
`--ttl` is set well beyond the longest run**. Without an exemption source the
GC prints a loud warning and proceeds with an empty exempt set. In practice,
always wire up the `--exempt-url` in production.
:::

### Scheduling with cron

Set `XRLENV_SCRATCH_GC_TTL` and `XRLENV_SCRATCH_REGISTRY_QUOTA_GB` in your
`.env` and schedule the GC periodically from cron or systemd:

```bash
# /etc/cron.d/xrlenv-scratch-gc  — run every 4 hours
0 */4 * * * <user> cd /path/to/xrlenv-dev && \
    .venv/bin/python deploy/registry/scratch_registry_gc.py \
    --registry 127.0.0.1:5012 \
    --storage-path /path/to/xrlenv-registry/scratch \
    --container xrlenv-registry-scratch \
    --ttl "${XRLENV_SCRATCH_GC_TTL:-72h}" \
    --quota-gb "${XRLENV_SCRATCH_REGISTRY_QUOTA_GB:-500}" \
    --exempt-url http://127.0.0.1:8080/api/scratch/active-digests \
    >> /var/log/xrlenv-scratch-gc.log 2>&1
```

### GC CLI reference

| Flag | Default | Purpose |
|---|---|---|
| `--registry` | (required) | `host:port` of the scratch registry. Use `127.0.0.1:5012` when the GC runs on the registry host. |
| `--scheme` | `http` | `http` or `https`. |
| `--ttl` | unset (disabled) | Reclaim images older than this. Accepts `72h`, `30m`, `90s`, or bare seconds. |
| `--quota-gb` | unset (disabled) | Per-registry soft cap in GiB. When the store exceeds this, oldest-unused images are evicted first. |
| `--storage-path` | unset | FSx blob-store root (for last-used mtime lookup). Omit to disable mtime-based TTL (falls back to wall-clock now). |
| `--container` | unset | Docker container name (`xrlenv-registry-scratch`). When set, the GC calls `docker exec <container> registry garbage-collect` after manifest deletion to reclaim blobs. |
| `--exempt-file` | unset | Path to a file of `sha256:...` digests active runs pin, one per line. |
| `--exempt-url` | unset | URL that returns the active digest set as JSON. The CP endpoint `GET /api/scratch/active-digests` serves exactly this. |
| `--dry-run` | off | Print reclaim targets and total GiB; delete nothing. **Always run this first in a new environment.** |

## Operational notes

- **No fallback.** Scratch refs (`<host>:5012/scratch/<digest>`) have no Docker
  Hub fallback. If the scratch registry is down, pulls for `image_build:` images
  fail. Keep it running on the always-on registry host.
- **Set the TTL well beyond your longest run.** The active-run exemption is the
  belt-and-suspenders backstop; TTL is the normal eviction trigger. If your
  longest rollout lasts 4 hours, set `--ttl 24h` or longer.
- **Content addressing means one build per unique Dockerfile+context.** Two
  runs with identical `image_build:` blocks share the same scratch image; GC
  reclaims it only when no run holds a reference and the TTL expires.
- **Node LRU evicts scratch images from disk like any other pulled image.** A
  later acquire re-pulls from `:5012` over the LAN — the registry blob store is
  the durable-until-GC copy; node disk is the hot set.
- **One registry instance per FSx path.** Do not run two scratch registry
  replicas over the same storage path.
- **TLS.** The scratch registry runs plain HTTP on the trusted cluster VPC, the
  same posture as the mirror and private registry. HTTPS via the mTLS CA is a
  planned future hardening.

## See also

- {doc}`registry_mirror` — the pull-through docker.io cache (`:5010`).
- {doc}`private_registry` — the durable operator-built registry (`:5011`).
- {doc}`/technical_details/images/bring_your_own_dockerfile` — how to write
  a template with `image_build:` so users need no operator pre-build step.
- {doc}`runbook` — end-to-end multi-node bring-up; registry configuration
  fits into the **Bootstrap node VMs** step.
- `deploy/registry/README.md` in the source tree — maintainer-facing reference
  for all three registry scripts.

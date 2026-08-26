# xrlenv pull-through registry mirror

A `registry:3` running in **proxy (pull-through cache) mode** in front of Docker
Hub, backed by FSx. It makes node image **re-pulls go over the internal network**
instead of Docker Hub, so node-local eviction is no longer catastrophic: the
first cluster-wide pull of an image caches its blobs on FSx, and every later pull
(including after a node evicts it) is served LAN-local. Cache misses and a down
mirror fall back to Docker Hub automatically.

It needs **no code
changes** on the node agent — node pulls already go through `dockerd`, and
Docker's `registry-mirrors` applies transparently to `docker.io` refs (exactly
the benchmark images, `jefzda/sweap-images:*`).

## Scripts in this directory

**Servers** (each starts a `registry:3`): `run-registry-mirror.sh` (`:5010`
pull-through), `run-registry-private.sh` (`:5011` writable),
`run-registry-scratch.sh` (`:5012` build-on-demand).

**Operations:**

| Script | Used when |
|---|---|
| `configure_docker_registry.sh` | Client-side: point a worker's Docker daemon at these registries (`daemon.json`). |
| `build_and_push_images.py` | Build a plan's images from Dockerfiles → push to the private registry (single-host; fleet-wide = `xrlenv build push`). |
| `warm_images.py` | Pre-warm the mirror's FSx cache from a build plan. |
| `restamp_registry_ttl.sh` | Re-stamp retention TTL on content **already cached** in the mirror. |
| `scratch_registry_gc.py` | GC the scratch registry (`:5012`) by TTL + per-namespace quota. |
| `retag_registry_images.py` | Retag private-registry images across repo namespaces in place, no rebuild. |

The mirror + private + build-push flow is documented below; the scratch registry
has its own guide at
[`docs/deploy/multi_node_deployment/scratch_registry.md`](../../docs/deploy/multi_node_deployment/scratch_registry.md).

## Two roles — keep them straight

| | **SERVER (the mirror)** | **CLIENT (each worker)** |
|---|---|---|
| what | runs the `registry:3` proxy | points its `docker pull` at the mirror |
| where | **control-plane box OR a dedicated registry node** — **never a worker** | every worker node |
| how | `deploy/registry/run-registry-mirror.sh` | `deploy/registry/configure_docker_registry.sh` |
| touches | starts a container, writes FSx blobs | edits `/etc/docker/daemon.json` only (runs **no** registry) |

The worker side is pure client config — it adds a `registry-mirrors` URL to
`daemon.json`. **No registry process ever runs on a worker.**

## 1. SERVER — start the mirror (once, on the CP box or a registry node)

```sh
bash deploy/registry/run-registry-mirror.sh
```

Config is read from a single `.env` (your source of truth — repo-root `.env` by
default; override with `REGISTRY_ENV_FILE`). Keys (all optional):

- `DOCKERHUB_USER` / `DOCKERHUB_TOKEN` — upstream Docker Hub auth, mapped to the
  registry's `REGISTRY_PROXY_USERNAME/PASSWORD` so the proxy isn't anonymous.
  **The same two keys bootstrap/refresh already use** — no separate secrets file.
  Anonymous works but Docker Hub rate-limits it (~100 pulls/6h), and the mirror
  funnels the whole cluster through one identity, so set a **rotated Pro/Team
  PAT** here before warming the full set, then re-run this script.
- `XRLENV_MIRROR_REGISTRY_STORAGE` — FSx blob-store path (default
  `/fsx/home/$USER/xrlenv-registry/proxy`; e.g.
  `/fsx/data/$USER/xrlenv-registry`). Must be a shared (Lustre/NFS) mount;
  it is the registry blob store **only**, never the dockerd data-root (overlay
  can't live on Lustre). *(The old `XRLENV_REGISTRY_STORAGE` name still works but
  warns — the mirror keys are namespaced `XRLENV_MIRROR_REGISTRY_*` to match the
  private/scratch registries.)*
- `XRLENV_MIRROR_REGISTRY_PORT` — host port (default `5010`; chosen over `5000`
  because an unexplained service answered on `:5000` during the audit).
  *(Deprecated alias: `XRLENV_REGISTRY_PORT`.)*

**Cache retention** is config-driven: `proxy.ttl` in `config-mirror.yml` (default
90 days) is the single source of truth and is honored by `registry:3`. A clean
redeploy retains content for that long with no manual steps. (Note: `registry:2`
ignored `proxy.ttl` and hardcoded 7 days — that's why we run `registry:3`.
Changing `ttl` only affects *future* pulls; to re-stamp content already cached,
see `deploy/registry/restamp_registry_ttl.sh`.)

Re-run the script any time to apply changed `.env` / `config-mirror.yml` values
(it recreates the container; the FSx blob store persists).

## 2. CLIENT — point workers at the mirror

```sh
sudo MIRROR_URL=http://<cp-or-registry-ip>:5010 bash deploy/registry/configure_docker_registry.sh --restart
```

- **Existing workers**: run the line above **once**. It edits `daemon.json` and
  restarts dockerd only if the config changed (idempotent). A later
  `deploy/refresh.sh` does **not** touch `daemon.json`, so the mirror setting
  persists on its own — you do not need to re-run this on every release.
- **New workers**: export `XRLENV_REGISTRY_MIRROR=http://<ip>:5010` for
  `deploy/bootstrap-aws.sh` and it applies the same client config at bootstrap
  (merge-only; dockerd reads it on first start). This is **client config**, not a
  registry.

## 3. Warm the cache (optional pre-fill)

```sh
python3 deploy/registry/warm_images.py <build_plan.yaml> --concurrency 16
```
Pulls each image's manifest + blobs **through the mirror via the registry API**
(`?ns=docker.io`) so the content is cached on FSx with **no local docker
extraction** — it does NOT `docker pull`, so it neither fills the warming box's
docker store nor bypasses the mirror. Run it on a box that can reach the mirror
(the CP box hits `127.0.0.1:5010`); it needs **no** Docker Hub creds (the mirror
handles upstream auth). The mirror still needs its PAT in place (step 1) or the
731-image fetch will hit the anonymous rate limit. Otherwise the cache fills
lazily on the eval's first pulls.

**Idempotent / resumable**: it skips blobs already on the FSx store (a fast
filesystem check via `--store-path`, default `$XRLENV_MIRROR_REGISTRY_STORAGE` or
`/fsx/home/$USER/xrlenv-registry/proxy`), so re-running after an interruption
only fetches what's missing. `--no-skip` forces a full re-stream (cache repair).
Live progress shows count, %, GB fetched, MB/s, img/s, elapsed, and ETA.

> Do NOT warm with a `docker pull` loop, and do NOT run it on a worker: that
> extracts every image locally (filling the box) and, on a box without
> `registry-mirrors` (e.g. the registry host itself), bypasses the mirror.

## Validation

1. Pull a cached image on a worker with Docker Hub blocked → succeeds (LAN).
2. Pull an uncached image with the **mirror's** Docker Hub egress blocked (the
   `DOCKER-USER` chain — container egress, not `OUTPUT`) → fails.

## Operational notes

- **Availability**: mirror down ⇒ dockerd falls back to Docker Hub. Bounded risk.
- **Retention / GC**: deletes are off (a cache we want to keep). Size FSx for the
  full compressed set (~1.2–1.5 TB). GC offline (read-only) only if needed.
- **Concurrency**: one registry instance over one FSx path (flock-safe). Do
  **not** run two replicas over the same path.
- Scheduler changes are intentionally **not** here.

---

# Private (writable) registry + bulk build-and-push

The proxy above only accelerates `docker.io` pulls and **cannot be pushed to**.
Some benchmarks ship a **Dockerfile, not a prebuilt image** (e.g.
[camel-ai/seta-env](https://github.com/camel-ai/seta-env)'s Harbor-Dataset). For
those, build the image **once**, push it to a private registry on FSx, and let
every worker pull a digest-pinnable ref over the LAN — instead of every node
rebuilding the same Dockerfile (slow, and no shared digest to pin).

Two registries run side by side on the control-plane box and **do not overlap**:

| | **PROXY** (mirror) | **PRIVATE** |
|---|---|---|
| port | `:5010` | `:5011` |
| script | `run-registry-mirror.sh` | `run-registry-private.sh` |
| storage | `~/xrlenv-registry/proxy` | `~/xrlenv-registry/private` |
| mode | pull-through cache of docker.io | writable; holds images we build |
| client config | `registry-mirrors` (docker.io routing) | `insecure-registries` (named refs) |
| miss behavior | falls back to Docker Hub | **no fallback** — verify before a run |

## 1. SERVER — start the private registry (on the CP box)

```sh
bash deploy/registry/run-registry-private.sh
```

Keys (all optional; from the repo `.env` or the calling env):
`XRLENV_PRIVATE_REGISTRY_STORAGE` (default `~/xrlenv-registry/private`),
`XRLENV_PRIVATE_REGISTRY_PORT` (default `5011`),
`XRLENV_PRIVATE_REGISTRY_HTTP_SECRET` (only for a multi-replica/LB deploy). Re-run
any time to apply changes; the FSx blob store persists. Deletes are **enabled**
(this is authored content, so an operator may GC retired refs offline).

## 2. CLIENT — allow plain-HTTP push/pull (one-time, via bootstrap)

**Why this step exists:** Docker refuses to talk to an HTTP-only registry unless
that host:port is in the daemon's `insecure-registries`. So any node that will
**push** (a build host) or **pull** (a worker running rollouts) the private set
needs `<cp-ip>:5011` listed there. (It is *not* a `registry-mirrors` entry —
named refs aren't mirror-routed.)

**You normally don't run anything by hand.** It's wired into bootstrap exactly
like the mirror — `slurm_scripts/{dev,prod}_xrlenv_node.sh` already export
`XRLENV_PRIVATE_REGISTRY` alongside `XRLENV_REGISTRY_MIRROR`, so every node they
bring up gets the `insecure-registries` entry automatically. The underlying call:

```sh
sudo \
    XRLENV_NODE_TOKEN=... DOCKERHUB_USER=... DOCKERHUB_TOKEN=... \
    XRLENV_REGISTRY_MIRROR='http://<cp-ip>:5010' \
    XRLENV_PRIVATE_REGISTRY='<cp-ip>:5011' \
    bash deploy/bootstrap-aws.sh --hyperpod <cp-ip>:50051 "aws-$(hostname -s)"
#                                           ^ control-plane addr  ^ this worker's node id
```

The 2nd positional is the node id (pass `aws-$(hostname -s)` so it matches the
`aws-{hostname}` roster from `xrlenv nodes-from-slurm`; omitting it auto-detects
`aws-<instance-id>`, which won't match). Drop `--hyperpod` off HyperPod.

The **only** time you run the helper directly is to fix an **already-running**
node that wasn't bootstrapped with it (it live-reloads dockerd, no container
bounce):

```sh
sudo PRIVATE_REGISTRY=<cp-ip>:5011 bash deploy/registry/configure_docker_registry.sh --restart
```

(Pass `MIRROR_URL` and `PRIVATE_REGISTRY` together to set both at once.)

## 3. BUILD + PUSH — bulk-build a plan's images into the registry

`deploy/registry/build_and_push_images.py` takes a per-image-ref `build-plan.yaml` (the
same shape `xrlenv build apply` uses — `type: git` / `tarball` / `registry`
entries), builds each image, and pushes it to `--registry`. It **skips images
already present** in the registry (idempotent / resumable / shard-overlap-safe),
and writes a per-shard JSON report. Because the cluster shares one home
filesystem, the git checkout for a repo is done **once for the whole campaign**
(cross-node-locked under `~/.xrlenv/build-context-cache`), not once per node.

Single host (build everything here):

```sh
python deploy/registry/build_and_push_images.py \
    --plan xrlenv_plugins/benchmarks/seta/build_plan.yaml \
    --registry <cp-ip>:5011
```

**Distribute across the cluster** (1000+ Dockerfiles on one node is slow). Use
`xrlenv build push` — the control-plane-orchestrated replacement for the old
Slurm batch script. It shards the plan across all currently-connected node
agents (size-aware, no hardcoded nodelist), builds from source on each node,
and pushes to `--registry`. Already-pushed refs are skipped by a registry HEAD
check before any build work starts — re-runs resume where they left off, and
overlapping dispatches never double-push.

```sh
xrlenv build push \
    --plan xrlenv_plugins/benchmarks/seta/build_plan_1376_full.yaml \
    --registry <cp-ip>:5011 \
    --connect-host <admin-host>
```

`--dry-run` prints the per-node shard assignment without building; `--force`
rebuilds already-present refs. `xrlenv build push` accepts only `git` and
`tarball` source entries — `registry`/`local` entries are rejected (they are
not built-from-source images).

## 4. Reference the private set + pin digests

Push the registry host into the plan's refs (keep a portable plan for dev) and
have the template use `image_pin_mode: registry_digest` — the catalog resolves +
pins each ref's digest at register time, so consumers never copy digests by hand
and a training run is reproducible. Full seta-env template wiring is a follow-up
step; the registry + build-push mechanism ship here.

## Operational notes (private registry)

- **SPOF**: named refs have **no Docker-Hub fallback**. If the private registry is
  down, those pulls fail — verify completeness before a run, or keep it on the
  always-on CP box.
- **GC**: deletes are enabled; run GC offline (registry in read-only mode) during
  a quiet window. Size FSx so routine GC isn't needed.
- **Concurrency**: many build hosts pushing to **one** registry instance is fine —
  the single registry process is the only FSx writer; build hosts talk HTTP, never
  touch the blob store directly. Do **not** run two registry replicas over one FSx
  path.
- **TLS**: the mirror's HTTP-on-trusted-VPC posture applies (`insecure-registries`).
  HTTPS via the mTLS CA is the same future hardening noted for the proxy.

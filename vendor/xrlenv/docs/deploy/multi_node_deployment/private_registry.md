# Private registry and bulk build-and-push

Some benchmarks ship a *Dockerfile*, not a prebuilt image. For example,
[camel-ai/seta-env](https://github.com/camel-ai/seta-env)'s Harbor-Dataset
defines each task environment as a Dockerfile rather than a published image.
Without a shared registry, every worker node would have to rebuild that
Dockerfile independently — which is slow, wastes CPU, and produces local images
with no shared digest to pin for reproducibility.

The **private registry** solves this: build each image once on one (or a few)
build hosts, push it to a writable FSx-backed registry on the control-plane
host, and let every worker pull a digest-pinnable named reference over the
internal network. Combined with `image_pin_mode: registry_digest` in your
template, the catalog resolves and pins each image's digest at register time, so
workers always run the exact same image content across an entire training run —
template manifests are immutable for the duration of a training run.

:::{note}
This page covers the private registry. For the pull-through cache of
Docker Hub images, see {doc}`registry_mirror`.
:::

## Two registries, side by side

Both registries run on the **same registry host** (typically a control-plane
box) and use distinct ports and distinct FSx subdirectories. They do not overlap,
and their client-side configuration is different. Throughout this page,
`<registry-host>` is that box — the same as `<control-plane-ip>` in a
single-cluster setup, but a single shared box when several clusters share one
registry (see the note under **Bring up the private registry** below).

| | **Proxy** ({doc}`registry_mirror`) | **Private** (this page) |
|---|---|---|
| Port | `:5010` | `:5011` |
| Script | `deploy/registry/run-registry-mirror.sh` | `deploy/registry/run-registry-private.sh` |
| Storage | `~/xrlenv-registry/proxy` | `~/xrlenv-registry/private` |
| Mode | Pull-through cache of Docker Hub | Writable; holds images you build |
| Client config | `registry-mirrors` (docker.io routing) | `insecure-registries` (named refs) |
| Miss behavior | Falls back to Docker Hub automatically | **No fallback** — verify completeness before a run |

The key difference: a proxy miss falls back to Docker Hub, so it is a
bounded-risk accelerator. A private-registry miss *fails the pull outright* —
the private registry holds images that don't exist anywhere else. Verify that all
required images are present before starting a training run.

The two registries complement each other. Build hosts typically have *both*
configured: the proxy mirror accelerates `FROM` base-image pulls during
`docker build`, and the private registry is where the finished images are pushed
to.

## Bring up the private registry (server — once, on the registry host)

Run on the registry host — typically a control-plane box (the same host that runs
`xrlenv up`):

```bash
bash deploy/registry/run-registry-private.sh
```

:::{note}
**Running more than one cluster (e.g. dev + prod)?** Run the registry on **one
designated box** and point every cluster at it — do **not** start a second
instance over the same FSx blob store (two writers on one path corrupts uploads).
Other clusters reach it by setting `XRLENV_MIRROR_REGISTRY_HOST` /
`XRLENV_PRIVATE_REGISTRY_HOST` in their `.env` to that box; their control-plane
wrappers do **not** start a registry. In the bundled Slurm scripts, only
`slurm_scripts/generated/prod_xrlenv_control.sh` starts the two registries, and the dev
cluster's `.env` points `XRLENV_{MIRROR,PRIVATE}_REGISTRY_HOST` at that same host.
:::

The script reads its config from the repo-root `.env` by default (override with
`REGISTRY_ENV_FILE`). All keys are optional:

| Key | Default | Purpose |
|---|---|---|
| `XRLENV_PRIVATE_REGISTRY_STORAGE` | `/fsx/home/$USER/xrlenv-registry/private` | FSx blob-store path. Must be a shared mount (NFS / FSx / Lustre) reachable cluster-wide. Never point it at the Docker data-root. |
| `XRLENV_PRIVATE_REGISTRY_PORT` | `5011` | Host port the registry listens on. |
| `XRLENV_PRIVATE_REGISTRY_HTTP_SECRET` | unset | A stable shared upload secret. A single-instance deploy does not need this — the registry generates a random secret internally. Set a stable value only if you run two or more replicas behind a load balancer so in-flight uploads routed to a different replica still validate. |

No Docker Hub credentials are needed here — the private registry is not a proxy
and never talks to Docker Hub. Build hosts pull their `FROM` base images through
the proxy mirror (`:5010`).

Re-running the script any time re-applies changed `.env` values: it recreates
the container, and the FSx blob store persists across restarts.

Verify the registry is up from the control-plane host:

```bash
curl -s http://127.0.0.1:5011/v2/ && echo "  private registry OK"
```

:::{note}
**Deletes are enabled.** Unlike the proxy cache (which you want to keep intact),
the private registry holds content you authored. An operator can garbage-collect
retired image references by running the registry in read-only mode during a quiet
window.
:::

## Allow plain-HTTP push and pull (client — one-time, via bootstrap)

**Why this step is needed:** Docker refuses to talk to an HTTP-only registry
unless that `host:port` is listed in the daemon's `insecure-registries`. Any
node that will *push* (a build host) or *pull* (a worker running rollouts) the
private image set needs `<registry-host>:5011` listed there. This is different from the
proxy mirror, which goes in `registry-mirrors` — named refs like
`<registry-host>:5011/seta-env/0:main` are addressed directly and are never
mirror-routed.

**You normally do not run anything by hand.** Set
`XRLENV_PRIVATE_REGISTRY=<registry-host>:5011` in the same shell that runs the bootstrap
alongside `XRLENV_REGISTRY_MIRROR`, and a freshly provisioned node gets the
`insecure-registries` entry automatically:

```bash
sudo \
    XRLENV_NODE_TOKEN='<paste-token-here>' \
    DOCKERHUB_USER='<your-docker-hub-handle>' \
    DOCKERHUB_TOKEN='<dckr_pat_...>' \
    XRLENV_REGISTRY_MIRROR='http://<registry-host>:5010' \
    XRLENV_PRIVATE_REGISTRY='<registry-host>:5011' \
    bash deploy/bootstrap-aws.sh --hyperpod <control-plane-ip>:50051 "aws-$(hostname -s)"
```

Here `<registry-host>` (the mirror/private registry box) and `<control-plane-ip>`
(this cluster's gRPC control plane) are the **same** box in a single-cluster
setup, but **differ** when one registry serves several clusters — point the
`XRLENV_*_REGISTRY_*` values at the shared registry box and the positional gRPC
address at this cluster's own control plane.

The two positional args are the **control-plane gRPC address** and this
**worker's node id** — *not* a second copy of the control-plane address. Pass the
node id explicitly as `aws-$(hostname -s)` so it matches the roster
`xrlenv nodes-from-slurm` generates (its default template is `aws-{hostname}`); if
you omit it, bootstrap auto-detects `aws-<instance-id>` from EC2 metadata, which
will *not* match that roster. Drop `--hyperpod` if the node isn't a SageMaker
HyperPod instance (the flag relocates Docker's data-root onto the EBS volume). In
practice the `slurm_scripts/{dev,prod}_xrlenv_node.sh` Slurm scripts already issue this exact
command — including `XRLENV_PRIVATE_REGISTRY` — for every node, so you rarely type
it by hand.

The bootstrap applies the same `configure_docker_registry.sh` logic it uses for
the mirror — it merges `insecure-registries` into `daemon.json` without
clobbering existing keys (such as the relocated `data-root`), and Docker reads
it on first start.

**The only time you run the helper directly** is to fix an already-running node
that was not bootstrapped with the private-registry setting. The helper
live-reloads the Docker daemon without bouncing running containers:

```bash
sudo PRIVATE_REGISTRY=<registry-host>:5011 \
    bash deploy/registry/configure_docker_registry.sh --restart
```

Pass both `MIRROR_URL` and `PRIVATE_REGISTRY` together to configure both in one
step. A later `deploy/refresh.sh` does not touch `daemon.json`, so the setting
persists across xrlenv upgrades.

## Build and push images (bulk build-and-push tool)

`deploy/registry/build_and_push_images.py` is the build glue. It takes a
`build-plan.yaml` in the same per-image-ref shape that `xrlenv build apply`
uses, builds each entry from its Dockerfile (or pulls a registry source), and
pushes the result to `--registry`. Think of it as the private-registry sibling
of `deploy/registry/warm_images.py` — `warm_images.py` fills the proxy by streaming
docker.io blobs; `build_and_push_images.py` fills the private registry by
building your own Dockerfiles.

### Single-host build (small plans, or testing)

```bash
.venv/bin/python deploy/registry/build_and_push_images.py \
    --plan xrlenv_plugins/benchmarks/seta/build_plan.yaml \
    --registry <registry-host>:5011
```

### Fan out across the cluster (recommended for large plans)

Building 1000+ Dockerfiles on a single node is slow. For large plans, use
`xrlenv build push` — the control-plane-orchestrated replacement for the old
Slurm batch script. It shards the build plan automatically across all
currently-connected node agents over the node-control bidi stream; each node builds
its shard from source and pushes the result to `--registry`. Re-runs are
resumable — already-pushed refs are skipped by a registry HEAD check before
any build work begins. No Slurm, no hardcoded nodelist, no drift.

```bash
xrlenv build push \
    --plan xrlenv_plugins/benchmarks/seta/build_plan_1376_full.yaml \
    --registry <registry-host>:5011 \
    --connect-host <admin-host>
```

Append `--dry-run` to see the per-node shard assignment without building
anything. Append `--force` to rebuild and repush images that are already in
the registry.

`xrlenv build push` accepts only `git` and `tarball` source entries from the
plan. `registry` and `local` entries are rejected — they are by definition
already in a registry or on-disk and do not need to be built and pushed. The
single-host fallback (`deploy/registry/build_and_push_images.py`) handles `local`
sources.

For the full flag reference see
{ref}`cli-build-push` in the CLI reference.

### Key behaviors

**Idempotent and resumable.** Before building each image, the tool probes the
registry with an HTTP HEAD request on the manifest. If the image is already
present, it is skipped. This means re-running after an interruption only builds
what is missing, and overlapping shards never double-push the same image. Use
`--force` to rebuild and repush an image that is already in the registry.

**Size-aware sharding.** The greedy partition assigns entries to shards by
`size_hint_bytes` from the plan, not by image count. Benchmark images are
heavy-tailed (a CPU-only seta-env task may be ~200 MB; a CUDA one ~3 GB), so
balancing by bytes produces more even wall-clock times across nodes.

**Clone-once on shared FSx.** Cluster nodes share one home filesystem, so
`~/.xrlenv/build-context-cache` is the same physical directory on every node.
When a build plan refers to a git repository (e.g. `type: git`), the first shard
to need that `(repo, ref)` clones it under a cross-node file lock and writes a
completion marker. Every other shard on any node sees the marker and reuses the
snapshot read-only — the entire fan-out shares a single clone of the repository.
Use `--refresh-context` to force a fresh clone when a moving ref like `main` has
advanced to a new commit.

**Disk-aware pruning.** Building hundreds of images fills Docker's build cache
and data-root fast. The tool runs `docker builder prune` periodically — every
`--prune-every` images built (default 25) **and** whenever free space on the
data-root drops below `--prune-min-free-gb` (default 30) — so a long shard
doesn't run a build host out of disk. Pass `--no-prune` to disable it.

**Per-shard JSON report.** After finishing, the tool writes
`<plan-dir>/build-push-report.shard<i>of<n>.json` with per-image status (`built`,
`pulled`, `skipped`, `failed`), pushed digest, and elapsed time. The report files
are gitignored (regenerated each run).

### CLI reference

| Flag | Default | Purpose |
|---|---|---|
| `--plan` | (required) | Path to a `build-plan.yaml` (per-image-ref shape). |
| `--registry` | (required) | Private registry `host:port`, e.g. `<registry-host>:5011`. |
| `--registry-scheme` | `http` | `http` or `https` for the registry API probes. |
| `--shard-index` | `$SLURM_PROCID` or `0` | This shard's 0-based index. |
| `--num-shards` | `$SLURM_NTASKS` or `1` | Total number of shards. |
| `--concurrency` | `4` | Concurrent `docker build` processes within this shard. |
| `--force` | off | Rebuild and repush even if the ref already exists. |
| `--no-prune` | off | Disable the periodic `docker builder prune` (see "Disk-aware pruning"). |
| `--prune-every` | `25` | Run a builder prune every N images built on this shard. |
| `--prune-min-free-gb` | `30` | Also prune whenever free disk on the data-root drops below this. |
| `--refresh-context` | off | Re-clone a git source even if a cached checkout exists. |
| `--build-timeout` | `3600` | Per-image build/pull/push timeout in seconds. |
| `--probe-timeout` | `15` | Per-request timeout for the registry manifest HEAD probe. |
| `--report` | next to the plan | JSON report path. |
| `--dry-run` | off | Print this shard's assignment and exit without building. |

## Reference the private set and pin digests

Once images are in the private registry, point your build plan's `image_ref`
values at the registry host and set `image_pin_mode: registry_digest` in the
template. The catalog resolves and pins each ref's digest at template-register
time, so workers always run the same image content for the lifetime of a training
run — you never have to copy digests by hand.

A portable ref in the plan (`seta-env/0:main`) becomes
`<registry-host>:5011/seta-env/0:main` when pushed by the build tool. Keep the
portable plan for development; the Slurm wrapper injects the registry host at
build time via `XRLENV_PRIVATE_REGISTRY_HOST`.

:::{note}
Full seta-env template wiring — including the production `build-plan.yaml` and
the annotated template config — is a follow-up. This page documents the registry
server and build-push mechanism that the template wiring builds on.
:::

## Operational notes

- **No fallback.** Named refs addressed to `<registry-host>:5011/...` have no Docker Hub
  fallback. If the private registry is down, those pulls fail. Keep it running on
  an always-on host (typically the control-plane box that runs `xrlenv up`), or
  pre-stage a snapshot before a run.
- **Concurrency.** Many build hosts pushing to one registry instance is safe —
  the single registry process is the sole FSx writer; build hosts push over HTTP
  and never touch the blob store directly. Do **not** run two registry replicas
  over the same FSx path.
- **Garbage collection.** Deletes are enabled. Run GC offline (restart the
  registry container in read-only mode) during a quiet window if you need to
  reclaim space after retiring a benchmark. Size the FSx store generously so
  routine GC is not needed.
- **TLS.** The private registry runs plain HTTP on the trusted cluster VPC, the
  same posture as the proxy mirror. HTTPS via the mTLS CA is a planned future
  hardening.

## Troubleshooting: control plane co-located with the registry

**Symptom.** After bring-up, acquires consistently fail with a
`RegistryResolveError` (ConnectTimeout) even though the registry is up and
worker nodes pull images without issue. Running on the control-plane box:

```bash
curl -s http://localhost:5011/v2/<repo>/tags/list
```

returns the expected JSON — confirming the registry process is healthy.

**Root cause.** The control plane's freshness resolver probes the registry by
the host embedded in the image ref (e.g. `<registry-host>:5011`) so that
the recorded digest ref is externally routable for worker nodes. When the
control plane runs on the **same box** as the registry, this probe targets the
box's own external address. On some managed infrastructure (verified on
SageMaker HyperPod / EFA nodes), a host cannot reach its own externally
published Docker port: the packet is policy-routed out the physical NIC to the
VPC gateway instead of toward the Docker bridge, resulting in a connection
timeout. Worker nodes pulling from that same address off-box are unaffected.

**Fix.** Set `XRLENV_REGISTRY_RESOLVE_HOST_MAP` to redirect the control
plane's manifest probe to loopback:

```bash
export XRLENV_REGISTRY_RESOLVE_HOST_MAP="<registry-host>:5011=127.0.0.1:5011,<registry-host>:5010=127.0.0.1:5010"
```

Replace `<registry-host>` with the short hostname (`hostname -s`) or the
primary IP (`hostname -I | awk '{print $1}'`) of the registry box — whichever
appears in your image refs. The digest ref the control plane returns still
carries the original external address, so nodes pull correctly.

**Fresh co-located deploys need no manual step.** The shipped Slurm control-plane
scripts (`slurm_scripts/generated/prod_xrlenv_control.sh` and `dev_xrlenv_control.sh`)
build and export this map automatically from the box's own hostname and IP before
starting `xrlenv up`. A co-located control plane therefore probes loopback out of
the box; a control plane on a different machine from the registry is unaffected
(its box name never appears in worker image refs, so the map entries never match).

See {doc}`/technical_details/images/registry_freshness` for the full list of
resolver knobs and the failure semantics.

## See also

- {doc}`registry_mirror` — the pull-through cache of Docker Hub; the
  two registries complement each other.
- {doc}`runbook` — the end-to-end multi-node bring-up; its **Bootstrap node VMs**
  step covers the Docker Hub auth that the mirror and private registry complement.
- `deploy/registry/README.md` in the source tree — the maintainer-facing
  reference for both registry scripts.

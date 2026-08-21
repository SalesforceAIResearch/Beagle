# Resource isolation

Raw (grading) containers run benchmark test suites where a wrong
verdict is expensive. Two failure modes motivate this page:

- **Oversubscription** — without a CPU cap a container bursts to every
  host core; pack enough of them on a node and a 200&nbsp;ms test
  deadline measures 219&nbsp;ms, flipping a pass to a fail.
- **Silent divergence** — a harness that caps CPU/memory against a
  local Docker daemon must get the *same* cap on the cluster, or
  cluster results stop matching local ones.

XRLEnv addresses both by treating a harness's resource request as a
first-class input: it drives scheduling, runtime cgroup limits, and
core pinning.

## The model

A raw container is constrained by two independent layers:

| Layer | Mechanism | What it bounds |
|---|---|---|
| **CFS quota** | `cpu_period` / `cpu_quota`, `mem_limit` | *Average* CPU time and a hard memory ceiling. |
| **cpuset pinning** | `cpuset_cpus` | Dedicated logical CPUs — bounds *wall-clock* latency, not just average share. Timing-sensitive tests need this. |

Two kinds of limit, kept deliberately separate:

- **`ResourceSpec`** (CPU + memory + isolation policy) — *scheduling-relevant*. The
  control plane derives it once from the harness request and threads
  the same object to the capacity estimator, the scheduler, and the
  node. The node turns it into cgroup limits and a cpuset.
- **`RuntimeLimits`** (pids / shm / tmpfs / read-only rootfs) —
  *container-shape, scheduling-neutral*. Applied at container creation
  only; never enters capacity accounting.

**Integer-core admission (pinned containers only).** When
`cpu_isolation` is `best_effort` or `required`, the node pins
`ceil(cpu_limit)` dedicated logical CPUs from its core ledger. A
node-process restart rebuilds the ledger from live containers so cores
are never leaked or double-assigned. Containers that do not opt in to
pinning use CFS quota only and do not consume the core ledger.

If a harness sets no CPU/memory request, the raw-container default
budget applies: **2.0 CPU, 4&nbsp;GiB memory**.

## Consumer knobs

A consumer expresses resource limits either through the docker-py
drop-in's `host_config` (the limits flow automatically — no code
change beyond `xrlenv.from_env()`) or through
`Client.acquire_container(...)` directly.

| docker-py `host_config` | `acquire_container(...)` arg | Effect on the cluster |
|---|---|---|
| `NanoCpus`, or `CpuQuota`+`CpuPeriod` | `cpu_limit` | CFS CPU quota. The scheduler places the task on a node with sufficient CPU slack. |
| `Memory` | `mem_limit_bytes` | Hard memory cgroup limit; counts toward node memory capacity. |
| `PidsLimit` | `runtime_limits.pids_limit` | `--pids-limit`. No scheduling effect. |
| `ShmSize` | `runtime_limits.shm_size_bytes` | `--shm-size`. No scheduling effect. |
| `Tmpfs` | `runtime_limits.tmpfs` | `--tmpfs` mounts. No scheduling effect. |
| `ReadonlyRootfs` | `runtime_limits.readonly_rootfs` | `--read-only` rootfs. No scheduling effect. |
| _(not in docker-py)_ | `cpu_isolation=CpuIsolation.BEST_EFFORT` | Opt-in cpuset pinning — see below. Scheduling-neutral. |
| _(not in docker-py)_ | `cpu_isolation=CpuIsolation.REQUIRED` | Pin or fail — scheduling-relevant; see below. |

**Asks-for-less vs. asks-for-more.** A request below the default
yields a smaller effective spec and frees scheduler capacity; a
request above the default makes the scheduler place the container
only on a node that can satisfy it. Either way the limit is honored,
never silently truncated.

**Rejected — hard error, never a silent drop:**

| Kwarg | Why rejected |
|---|---|
| `CpuShares` | A relative weight, not a hard cap — no deterministic isolation. Use `nano_cpus` / `cpu_quota`. |
| `MemoryReservation` | A *soft* limit; cannot stand in for hard memory enforcement. Use `mem_limit`. |
| `MemorySwap` (when ambiguous) | Daemon/cgroup-dependent semantics; only `MemorySwap == Memory` (swap disabled) is unambiguous. |
| `CpusetCpus`, `CpusetMems` | CPU/memory **placement** is cluster-owned — the node pins cores from its own ledger. A harness pin would collide with co-located containers. |
| `CgroupParent` | Overrides the cluster's cgroup accounting hierarchy. |

Each rejection is a four-part error: the requested value, the reason,
and a suggested action.

## CPU isolation modes

`cpu_isolation` is a `CpuIsolation` enum on `ResourceSpec` with three
modes. It is **scheduling-relevant**: it travels from the control-plane
ingress through the scheduler, the wire, the node, and the session
record.

| Mode | String value | Behavior |
|---|---|---|
| `CpuIsolation.OFF` | `"off"` | CFS `--cpus` quota only, burstable across all host cores. **Default** — matches how harbor runs its containers. |
| `CpuIsolation.BEST_EFFORT` | `"best_effort"` | Pin to `ceil(cpu_limit)` dedicated logical CPUs **if the node has free pinnable capacity**, else fall back to CFS quota with a warning. Scheduling-neutral — no placement constraint. This is the compat target of the legacy `RuntimeLimits(cpu_pinning=True)` alias. |
| `CpuIsolation.REQUIRED` | `"required"` | Pin or **fail**: the scheduler places only on an `isolation_capable` node with free pinnable cores, and node-side ledger exhaustion is a hard error — re-admitted on a sibling capable node, never a silent CFS degrade. |

Pass `cpu_isolation` directly to `Client.acquire_container`:

```python
import asyncio
from xrlenv import Client
from xrlenv.backends.base import CpuIsolation

async def run():
    async with Client.grpc("127.0.0.1") as client:
        # Best-effort: pin 2 dedicated cores if the node has them,
        # fall back to CFS quota otherwise.
        async with await client.acquire_container(
            image="my-benchmark:latest",
            cpu_limit=2.0,
            mem_limit_bytes=4 * 1024**3,
            cpu_isolation=CpuIsolation.BEST_EFFORT,
        ) as session:
            # session.exec(...) here
            pass

asyncio.run(run())
```

For tasks where reward correctness is sensitive to CPU contention
(e.g. a benchmark that measures wall-clock speedup against a frozen
baseline), use `REQUIRED` to guarantee pinning or surface a placement
failure:

```python
async with await client.acquire_container(
    image="my-benchmark:latest",
    cpu_limit=4.0,
    mem_limit_bytes=8 * 1024**3,
    cpu_isolation=CpuIsolation.REQUIRED,  # pin-or-fail
) as session:
    ...
```

**Legacy alias.** `RuntimeLimits(cpu_pinning=True)` is a backward-compat
alias for `CpuIsolation.BEST_EFFORT`. New code should use
`cpu_isolation=CpuIsolation.BEST_EFFORT` directly; the boolean alias
is kept for callers that have not migrated yet. `cpu_pinning=True`
can never express `REQUIRED`.

**Harbor adapter knobs.** Harbor tasks opt into pinning through two
channels, **both of which map to `BEST_EFFORT`** (via the legacy
`RuntimeLimits(cpu_pinning=True)` alias). Harbor task markers deliberately
cannot express `REQUIRED` — a hard placement constraint must be requested
explicitly on the `acquire_container(cpu_isolation=…)` path:

- **Per-task** — `[environment.env] XRLENV_CPU_PINNING = "1"` in a
  task's `task.toml` (the surgical channel the patched-cache pipeline
  uses to mark only the `nproc`-scaling oracles, e.g. the QEMU-build tasks).
- **Job-level** — `environment.kwargs: {xrlenv_cpu_pinning: true}` in
  the harbor job config (a blunt hint applied to every task in the job).

The two are OR'd together in the harbor `EnvAdapter`; either one present
sets `RuntimeLimits(cpu_pinning=True)` on the acquire.

### What CPU isolation does — the shared-parent cpuset

Opt-in pinning (`best_effort`, `required`) reserves disjoint logical
CPUs from the node's core ledger. Without CPU isolation, **unpinned** containers
had no cpuset at all — the kernel gave them the full host affinity mask,
which included the cores nominally "reserved" for pinned containers.
Under load the unpinned neighbors trampled the pinned container's
dedicated cores, degrading wall-clock performance even for the pinned
task.

CPU isolation closes this gap on capable nodes via a **shared-parent cpuset
cgroup** (`/sys/fs/cgroup/xrlenv-shared`) whose `cpuset.cpus` always
equals the *complement* of the currently-pinned cores:

- When a pinned container acquires cores `C`, the node writes
  `xrlenv-shared.cpuset.cpus = all − C`. Every unpinned runc container
  is placed under `xrlenv-shared` (`--cgroup-parent`), so the kernel
  enforces that they can never run on `C` — one write reconfines all
  unpinned children simultaneously.
- When the pinned container exits, `C` is restored to the shared pool
  before the ledger releases it.
- Docker Compose sidecars (harbor, pier, or others) are also placed
  under `xrlenv-shared` for runc services, so they are equally confined.

The complement is serialized under the same ledger lock as
`allocate`/`release`, so concurrent pins are always consistent.

**Floor.** The ledger enforces a `min_shared_cores` floor (an internal
default of 25% of logical CPUs, derived at node wiring — not currently an
operator-settable `nodes.yaml` field). Pinning
that would drop the shared pool below the floor is refused:
`best_effort` degrades to CFS quota with a warning; `required` is a
hard error (but placement should already have excluded the node once
`pinned_cpus_free` drops to zero, so hitting the floor node-side is a
rare stale-heartbeat/ledger race that fails loudly and triggers
re-admission).

**Logical CPUs, not physical cores.** The ledger allocates *logical*
CPU indices (`os.cpu_count()`). On an SMT host two logical siblings
share one physical core. Full physical-core isolation (topology-aware
sibling-pair allocation) is a v2 follow-up, gated on evidence that
SMT contention moves the score after logical pinning.

**Sysbox runtime is NOT covered by CPU isolation (v1).** The shared-parent scheme
is runc-only. Sysbox-runtime containers (e.g. harbor DinD tasks) are
treated like unpinned containers on a non-capable node — no
`cgroup_parent`, today's behavior. A per-runtime probe is a future
follow-up.

## Operator knobs

### Node capability — enabling CPU isolation

A node advertises `isolation_capable=true` only after a real
self-test passes at startup. The self-test gates on **two conditions**:

1. The docker cgroup driver must be **`cgroupfs`** (not the default
   `systemd`). Stock AL2023 and Ubuntu 22.04 nodes use `systemd` →
   non-capable by default, behave exactly as today.
2. A throwaway container probe must confirm that `cgroup_parent`
   cpuset propagation actually works on this node's docker + kernel
   combination. A node where the probe fails advertises `false` and
   is never given a `required` task.

**To enable a node** (root, maintenance window, on the worker itself):

```bash
sudo bash scripts/enable_cpu_isolation.sh
```

The script:
1. Merges `"native.cgroupdriver=cgroupfs"` into `/etc/docker/daemon.json`
   (idempotent; preserves other keys).
2. Builds a tiny probe image (`xrlenv-selftest:1`, `FROM busybox`)
   if it does not already exist.
3. Restarts docker (a cgroup-driver change is not SIGHUP-reloadable)
   — **this bounces all running containers on the node**.
4. Runs the real container probe to verify `cgroup_parent` cpuset
   propagation. If it fails, the script exits cleanly and the node
   stays non-capable; docker has already been restarted to `cgroupfs`
   (so retry-ability is preserved).
5. Creates `/sys/fs/cgroup/xrlenv-shared`, enables the `cpuset`
   controller on it, and **delegates** its `cpuset.cpus`,
   `cgroup.procs`, `cgroup.subtree_control`, and `cgroup.threads`
   to the non-root agent user (`xrlenv` by default). This is the cgroup
   delegation that allows the agent to manage `xrlenv-shared` without
   running as root.
6. Restarts the `xrlenv-node` agent so it re-runs its capability
   check (non-root, using the delegated cgroup).

After the node reconnects, `xrlenv nodes` shows `CPU_ISOLATION = yes N/M`
where `N` is free pinnable CPUs and `M` is total pinnable CPUs.

**Stage only, no bounce** — to configure without restarting docker
(useful to pre-stage before a planned maintenance window):

```bash
SKIP_RESTART=1 sudo bash scripts/enable_cpu_isolation.sh
```

**Revert** — remove the `exec-opts` entry from `daemon.json`,
delete `/etc/xrlenv/cpu_isolation.env`, run `rmdir
/sys/fs/cgroup/xrlenv-shared` (once idle), and restart docker. The
node returns to non-capable / today's behavior.

### Deploy-time: the `CPU_ISOLATION_POOL` knob

The deploy scripts (`slurm_scripts/deploy_dev.sh` /
`slurm_scripts/deploy_prod.sh`) have a `CPU_ISOLATION_POOL` array
that runs `enable_cpu_isolation.sh` over SSH on the listed nodes
during each deploy:

```bash
# In deploy_dev.sh / deploy_prod.sh — see each committed file for its
# current pool. An empty array leaves every node non-capable.
CPU_ISOLATION_POOL=(node-host)
# CPU_ISOLATION_POOL=()   # empty = no CPU-isolation pool on this cluster
```

The deploy step is **idempotent** — nodes already on `cgroupfs` are
skipped (no second docker bounce). An empty `CPU_ISOLATION_POOL`
leaves all nodes non-capable. On **prod**, enabling a node restarts
docker and kills its in-flight containers, so a prod pool is a
maintenance-window choice — the committed `deploy_prod.sh` populates
its own CPU-isolation pool (consult that file for the current set), not an empty
default.

:::{note}
**Do not overlap `CPU_ISOLATION_POOL` with `SYSBOX_POOL`.** Flipping
a sysbox node to `cgroupfs` can break its DinD workloads; the deploy
script fires a warning if it detects overlap.
:::

### Other operator knobs

| Knob | Where | Default | Effect |
|---|---|---|---|
| `XRLENV_RAW_CREATE_CONCURRENCY` | `/etc/xrlenv/node.env` (per node) | `4` | Caps concurrent `docker run` calls — bounds daemon pressure and image-extraction bursts. `0` disables. |
| `XRLENV_RAW_SYSBOX_CREATE_CONCURRENCY` | `/etc/xrlenv/node.env` (per node) | `1` | Tighter, separate create cap for **sysbox** (non-`runc`) containers — sysbox-fs pre-register is far slower than a plain runc create, so unbounded concurrent sysbox creates surface a transient `pre-register with sysbox-fs … DeadlineExceeded`. Serialising them (default `1`) prevents it at the source; the node-side create retry recovers any that still slip through. `0` falls back to `XRLENV_RAW_CREATE_CONCURRENCY`. |
| `XRLENV_RAW_DESTROY_CONCURRENCY` | `/etc/xrlenv/node.env` (per node) | `4` | Caps concurrent `docker rm -f` calls — bounds teardown-time daemon pressure and EBS I/O. `0` disables. |
| `XRLENV_RAW_ARCHIVE_CONCURRENCY` | `/etc/xrlenv/node.env` (per node) | `4` | Caps concurrent bulk container↔node tar transfers (`get_archive` / `put_archive`). `0` disables. |
| `XRLENV_MAX_GET_ARCHIVE_RELAY_BYTES` | `/etc/xrlenv/node.env` (per node) | `134217728` (128 MiB) | Max bytes a single `get_archive` may relay through the control plane before raising `ArchiveTooLarge`. `0` disables the cap. |
| `XRLENV_DISK_GUARD_ENABLED` | `/etc/xrlenv/node.env` (per node) | `true` | Master switch for the disk-pressure guard (see below). |
| `XRLENV_DISK_GUARD_INTERVAL_S` | `/etc/xrlenv/node.env` (per node) | `15.0` | Poll cadence of the disk-pressure guard in seconds. |
| `policy:` section | `nodes.yaml` | see {doc}`/developer_guide/security` | Cluster docker-kwarg policy. `cpuset_cpus`, `cpuset_mems`, `cgroup_parent` are Level&nbsp;3 (always blocked, no override). |

**Node resilience: archive concurrency and relay cap.** A task that
copies a large directory tree (e.g., whole `/testbed`) via
`get_archive` on the shared node bidi stream can flood the stream's
send queue. If the tar payload exceeds gRPC's send ceiling
(~128 MiB), the stream is torn down — and since the heartbeat shares
that stream, the control plane marks the node lost and seals every
in-flight rollout there as `node_lost`. `XRLENV_RAW_ARCHIVE_CONCURRENCY`
bounds concurrent transfers to limit the blast radius from one
tenant's copy wave; `XRLENV_MAX_GET_ARCHIVE_RELAY_BYTES` caps relay
size at the control plane and surfaces `ArchiveTooLarge` to the
caller instead of tearing the stream. Archive data is read off the
asyncio event loop in chunks so large copies never block the
heartbeat regardless of size.

**Disk-pressure guard.** `DiskPressureGuard` polls free disk every
`XRLENV_DISK_GUARD_INTERVAL_S` seconds. When free disk falls below
a critical threshold and image eviction alone cannot recover it, the
guard force-kills the largest runaway raw containers until the node
returns above the recovery target. When `XRLENV_DISK_GUARD_ENABLED=false`,
the guard is not started and disk pressure is handled only by image
eviction.

## Observability

`xrlenv nodes` includes a `CPU_ISOLATION` column:

```
NODE               ...   CPU_ISOLATION
node-host         yes 184/188
node-host         no
```

- `yes N/M` — isolation-capable; `N` pinnable CPUs currently free out
  of `M` total pinnable (total = all logical CPUs − floor).
- `no` — non-capable; behaves exactly as today (per-container best-effort
  pinning / CFS quota); `required` tasks are never placed here.

The admin panel's `/nodes` view renders the same fields.

## Known limitations (v1)

- **Sysbox containers are not isolated.** The shared-parent scheme is
  runc-only. Harbor DinD and TW sysbox tasks remain on full-host
  affinity on their non-pinned cores.
- **Not reboot-persistent.** The delegated `xrlenv-shared` cgroup
  does not survive a node reboot. After a reboot the node reverts to
  non-capable until `enable_cpu_isolation.sh` (or a re-deploy) re-runs.
  A boot-time oneshot to re-create and re-delegate `xrlenv-shared` is
  a v2 follow-up.
- **`systemd`-driver nodes stay non-capable.** Stock AL2023 / Ubuntu
  22.04 nodes use the `systemd` docker cgroup driver. CPU isolation (v1) does not
  support the `systemd` path (it would require transient systemd slice
  management). Enablement requires flipping to `cgroupfs`
  (`enable_cpu_isolation.sh`), which is a maintenance-window operation.
- **Logical-CPU isolation, not physical.** On SMT (hyperthreaded)
  hosts, two logical CPUs share one physical core. The ledger allocates
  logical CPU indices, so sibling logical CPUs on the same physical
  core may not both be exclusive. Full physical-core isolation is a v2
  follow-up.
- **Transition gap on first enable.** Unpinned runc containers that
  were already running when the node wired the shared parent are not
  migrated — they retain full-host affinity until they exit. The node
  counts these as a "legacy gap" and reports `pinned_cpus_free = 0`
  until they drain. `required` tasks are therefore not placed on a
  newly-enabled node until the legacy containers have all exited
  (the `enable_cpu_isolation.sh` path avoids this by restarting docker,
  which kills all live containers before the agent wires the parent).

## See also

- {doc}`/build_with_xrlenv/work_with_xrlenv_managed_containers/docker_py_dropin`
  — passing limits via `host_config`.
- {doc}`/build_with_xrlenv/work_with_xrlenv_managed_containers/direct_api`
  — passing limits via `acquire_container(...)`.
- {doc}`scheduling` — how the capacity estimator packs a node.

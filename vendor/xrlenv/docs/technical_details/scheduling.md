# Scheduling

The scheduler chooses a node for each managed container or template
rollout. It optimizes for feasible capacity first, then image
locality. Capacity is released only after the node confirms that the
container was destroyed.

This page is for operators reading `/capacity` and `/nodes` who need
to understand what they're seeing, and for developers embedding the
scheduler who need to know how scoring, reservations, and queueing
fit together. The implementation lives in
`xrlenv/control/scheduler.py`, with the capacity model in
`xrlenv/control/capacity.py`, image-presence in
`xrlenv/control/image_presence.py`, and the queueing layer in
`xrlenv/control/admission.py`.

## Placement inputs

For each request, the scheduler considers:

- connected nodes and their latest heartbeat
- backend support on each node (Docker, cubesandbox, …)
- CPU, memory, sandbox-writable disk, and maximum concurrent
  sandbox capacity
- running sandboxes already recorded in the state store
- pending reservations from concurrent `place()` calls
- requested image and per-node image presence
- registered preferred home from image planning, if any
- `task_key`, when provided, for fairness

## Placement flow

````{height-limit}
:height: 320px

```{mermaid}
flowchart TD
    request["Acquire or rollout request"]
    resolve["Resolve resources, backend, image"]
    presence["Fetch image presence per eligible node"]
    eligible["Filter hard eligibility"]
    score["Score resource slack and image affinity"]
    reserve["Reserve placement"]
    command["Send command to selected node"]
    release["Release after node-confirmed destroy"]

    request --> resolve --> presence --> eligible --> score --> reserve --> command --> release
```
````

Hard eligibility happens before scoring. Disconnected nodes, nodes
without the requested backend, nodes that cannot fit the requested
resources, and nodes over the per-task fairness cap are removed.

## Capacity and resource slack

Capacity is a model, not a measurement. The scheduler uses reserved
capacity, not live OS idleness: a sleeping container still consumes
its reservation until the node confirms destroy. That keeps
placement cheap and predictable, while Docker/cgroup limits handle
runtime bursts.

The capacity estimator builds a per-(node, template) cell with
three independent caps — **CPU**, **memory**, and **sandbox-
writable disk** — and the minimum wins. The *binding constraint*
label records which axis bottlenecked, so an operator can see at a
glance why a node looks small.

### Resource slack score

For scoring, the estimator computes the post-placement remaining
slack fraction on each axis and returns the minimum:

```text
slack_axis  = (cap_after_headroom - load - candidate) / cap_after_headroom
slack_score = min(slack_cpu, slack_mem, slack_disk)
```

Using the minimum axis makes the score bottleneck-aware. A node
with plenty of disk but almost no CPU should not look healthy for
a CPU-heavy request. Axes the candidate opts out of (`cpu_request
<= 0` or `mem_request_bytes <= 0`) are excluded — see Quick notes.

### Backend overhead

Each active sandbox carries a per-runtime fixed cost on top of its
declared request. Defaults today:

| Backend | CPU per sandbox | Memory per sandbox |
|---|---|---|
| Docker | 0.05 CPU | 50 MB |
| cubesandbox | 0.10 CPU | 128 MB |
| local-process-debug | 0 | 0 |

The estimator subtracts this overhead per running and per pending
sandbox before computing remaining slack. An operator who sees a
node with 8 vCPUs reporting fewer slots than the template's
`cpu_request` would naïvely imply should suspect the overhead and
the headroom reservation below — they always apply.

### Headroom reservation

A fraction of each axis is kept free for the operating system and
the node agent itself, so a fully-packed node still has room to
breathe:

| Axis | Reserved |
|---|---|
| CPU | 10% |
| Memory | 15% |
| Disk (OS reserve) | 5 GB |

Templates requesting more than available-minus-headroom are
rejected at the hard-eligibility gate. The reservation is
operator-tunable through `HeadroomConfig`, but the defaults are
conservative enough that "node went unresponsive" is rarely traced
back to capacity math.

### Multi-pool disk

Disk is split into two pools the rest of the platform treats
independently:

- **Image cache pool** — the Docker layer store, managed by the
  node-side `ImageCacheManager`. Subject to eviction (see
  {doc}`images/cache_eviction`).
- **Sandbox-writable pool** — per-sandbox COW + scratch space.
  This is what placement counts against.

The split defaults to 50/50 (`image_cache_pool_fraction = 0.5`)
after the OS reserve, which mirrors a typical SWE-bench / OSWorld
profile where image bytes and per-sandbox writable disk consume
comparable budgets. A pin-heavy operator doesn't accidentally
starve sandbox creation: the image cache manager evicts within
its own pool, and the capacity estimator only sizes
`disk:sandbox_writable` against the remaining fraction. Operators
tune this when their templates skew toward one side.

### Binding constraint labeling

When the estimator returns a cell, it records which axis
bottlenecked the slot count. This shows up as the
`binding_constraint` field on `/capacity` and as a tooltip on
`/nodes`. Values:

| Label | Meaning |
|---|---|
| `cpu` | CPU was the tightest axis after overhead + headroom. |
| `mem` | Memory was the tightest axis. |
| `disk:sandbox_writable` | The sandbox-writable disk pool was the limit (image cache pool is accounted separately). |
| `gpu` | Template requires a GPU and the node has none — hard reject. |
| `backend_missing` | Node doesn't advertise the requested backend — hard reject. |

When debugging "why is this node under-utilized?", read the
binding label first.

## Image affinity

Before scoring, the control plane asks each backend-capable node
whether it already has the requested image. The result is the
`{node_id: bool}` `image_present` map consumed by
`Scheduler.place(...)`.

| Condition | Affinity signal |
|---|---|
| Node has the image | Full image-affinity bonus. |
| No node has the image, but this node is the registered preferred home | Half-strength preferred-home bonus. |
| Node doesn't have the image and isn't a preferred home | No image bonus. |

The preferred-home bonus only applies while no node has the image
yet. As soon as any node materializes it, real image presence
takes over and the preferred-home flag is forced off everywhere.

### How `image_present` is gathered

The admission queue issues one `query_image` RPC per backend-
capable node concurrently (via `asyncio.gather`), then waits for
all of them to come back. The same primitive serves the template
admission path and the raw-container acquire path, so they share
calibration and failure handling:

- **Per-node timeout / error.** Logged and treated as "absent." A
  flaky node doesn't poison the placement decision — it just
  doesn't get the affinity bonus on this acquire.
- **Wall-clock per call.** Bounded by the slowest single RPC,
  typically ~50 ms cross-LAN. Faster on a local laptop.
- **Skipped entirely** when the operator turned off image-aware
  placement, when the manifest has no concrete image (Pattern A
  before the per-instance resolver fires), or when no backend-
  capable nodes are attached.

Future scale work may add a per-node heartbeat-cached presence
snapshot so the hot path doesn't fan out on every admit, but
today the cost is small enough to do per call.

### Preferred home

The build planner assigns each image a *preferred home* during
FFD bin-packing (see {doc}`images/build_plan`). For images that
have been registered in the plan but not yet materialized on any
node, that recorded home flows through to the scheduler as a soft
hint: it adds a half-strength image-affinity bonus to the
preferred node only.

Two gates apply:

1. **Cluster-wide gate.** The bonus disappears once any node
   reports the image present. Real warm-cache nodes always win
   over a preferred-home cold node.
2. **Image-aware gate.** Operators who turned off image-aware
   placement also opt out of the preferred-home signal — the
   scheduler doesn't read the build snapshot when affinity is
   off.

The lookup is a single indexed query on the build-plan snapshot
(`StateStore.find_registered_preferred_home`). If the state store
doesn't support it (older test doubles), the scheduler quietly
falls through to non-affinity scoring.

## Default score

The default score is a weighted sum:

```text
score = w_R * resource_slack + w_I * image_affinity
```

Both inputs are normalized to `[0, 1]`, and the weights sum to
`1.0` — so the resulting score is also in `[0, 1]`. Defaults:

- `resource_weight = 2/3`
- `image_affinity_weight = 1/3`

The resource term is heavier on the explicit justification that
capacity-to-actually-run is fundamentally heavier than one-time-
cost-amortized-over-rollout. An overloaded node either fails the
hard-eligibility gate or runs the rollout slowly throughout its
life; a cold-image node only pays a one-time pull/build cost at
start. Different categories of cost, weighted accordingly.

Operators embedding the scheduler can supply a custom pure-
arithmetic `score_fn`, but it should not perform I/O. Any
expensive feature should be prefetched once above the scoring
loop and passed in through `PlacementFeatures`. Weight validation
runs at the factory call site, so a misconfigured operator
weight fails at boot, not at placement time.

## Task fairness

`task_key` groups related work for scheduling. The scheduler
counts running and pending placements with the same key on each
node and rejects nodes at `max_runs_per_task` (default 4) before
doing any capacity arithmetic — a hard early rejection, not a soft
penalty.

`task_key` is not identity. The rollout id and sandbox id identify
the execution. `task_key` only helps spread related attempts
across the cluster so a single task can't monopolize one node.

## Pending reservations

Concurrent placement calls can race if they all read the same
state snapshot before any sandbox record is committed. To avoid
overpacking, `Scheduler.place(...)` atomically:

1. Picks the best candidate inside a lock.
2. Allocates a `reservation_id` (uuid).
3. Records the candidate in an in-memory pending set keyed by that
   id.

Other concurrent `place()` calls fold pending reservations into
their cluster-load view before scoring, so they see each in-flight
peer's anti-affinity and resource contribution immediately —
without waiting for any state mutation.

A reservation moves through three states:

| State | Trigger | Effect |
|---|---|---|
| `pending` | `Scheduler.place` succeeds. | Counted in cluster load for subsequent placements. |
| `committed` | Coordinator calls `commit_placement(p)` after the sandbox record is in `StateStore`. | Drops from pending; `list_sandboxes()` now covers it. |
| `released` | Coordinator calls `release_placement(p)` after create-sandbox fails. | Drops from pending; never reached state store. |

Both `commit_placement` and `release_placement` are idempotent —
unknown or already-resolved reservation ids are a no-op — so the
coordinator can call either without tracking commit state
separately. The admission queue takes care to release a placement
if the waiter has timed out between scheduler success and
result-delivery, otherwise the pending count would leak.

## Admission queue

When no eligible node can fit the request, the scheduler raises
`CapacityExhausted`. The scheduler itself does not queue — the
admission queue catches that signal, enqueues the request, and
returns a future to the caller. The request blocks on the future
(with `queue_timeout_s`, default 300) until the queue drains.

The queue rows live in `StateStore` so they survive a control-
plane restart; the in-memory waiter futures don't, which is fine
— a restart's recovery path re-issues admission for any rows
whose request_id is still valid.

### Drain triggers

The drain worker wakes on two signals:

1. **Event-driven** — `kick()` is called from every node-confirmed
   destroy, so freed capacity is picked up immediately.
2. **Periodic poll** — every `poll_interval_s` (default 5 s) as a
   safety net for any event the kick path missed.

### What the drain does

On each wakeup, the worker snapshots pending rows and, for each:

1. Re-fetches `image_present` and the preferred-home node (state
   may have changed since the row was enqueued — late-arriving
   images get picked up here).
2. Calls `scheduler.place(...)` with the fresh inputs.
3. On `CapacityExhausted`, leaves the row queued and moves on —
   it will be retried on the next wakeup.
4. On any other exception, fails the waiter's future.
5. On success, delivers the placement to the waiter — unless the
   waiter has already timed out, in which case the worker
   *releases* the reservation before discarding, so the pending
   count doesn't leak.

The two-signal design means an operator watching queue latency
should expect sub-millisecond drain after a node destroys
something, and at most `poll_interval_s` after any state change
that didn't go through the destroy path.

(adaptive-admission)=
## Adaptive admission controller

The static capacity estimator tells the scheduler how many containers fit
on a node given CPU, memory, and disk arithmetic. That model is conservative
and works well under normal conditions, but it cannot detect transient
docker-daemon degradation — high `docker run` latency, container-create
timeouts, or outright docker errors — that are invisible to resource
accounting.

The **health-derived adaptive admission controller** adds a second, dynamic
cap on top of the static estimator. When enabled with `--adaptive-admission`
on `xrlenv up`, each node's concurrent-rollout admission limit contracts when
health degrades and recovers as health returns. The scheduler reads each
node's current admission limit and rejects placement on any node whose
current running-container count is at or above the limit.

### Per-node state, not a cluster budget

The controller (`HealthAimdController` in `xrlenv/control/capacity.py`)
holds a per-node limit dictionary keyed by `node_id`:

```python
self._limits: dict[str, int]
```

`limit_for(node_id)` returns that node's limit, or the `initial_limit`
seed for a node the controller has never seen. **There is no cluster-wide
shared budget.** Each node independently slow-starts at `--aimd-initial-limit`
(default 16) and evolves on its own health signal. Adding nodes to a running
cluster is safe — newcomers slow-start independently; nothing is reallocated
from existing nodes. When a node disconnects, its state entry is dropped.

### AIMD rule

A background `AimdControlLoop` (`xrlenv/control/aimd_loop.py`) ticks every
15 seconds. On each tick it:

1. Reads each connected node's latest Stage-1 health snapshot
   (`create_p95_ms`, `docker_error_count`, `docker_timeout_count`) from the
   node-transport's last heartbeat.
2. Reads each node's current running-container count from the scheduler's
   load snapshot.
3. Calls `HealthAimdController.step(health=..., load=...)`, which runs one
   AIMD round per node:

**Bad tick** — any of the following makes a tick "bad":

- `docker_error_count > 0`
- `docker_timeout_count > 0`
- `create_p95_ms > p95_bad_threshold_ms` (i.e., `--aimd-p95-threshold-s × 1000`)

On a bad tick: `limit = max(1, floor(limit × 0.5))`. The floor of 1 ensures
a node never contracts below one in-flight acquire — it keeps making progress
even while health is poor.

**Good tick, node saturating its limit** — when all three health conditions
are clear AND the node's current running-container count equals the current
limit:

`limit = min(max_limit, limit + 1)`

The "exactly at its limit" condition is deliberate: an under-loaded healthy
node holds (a quiet node is no evidence it can take more work); a node that
just contracted and is draining down also holds (its count is below the new
lower limit).

**Good tick, node under-loaded** — hold. No change.

**No health data** — a node agent that has not yet reported Stage-1 health
returns `None`. The controller holds the limit unchanged. Newly connected
nodes hold at `initial_limit` until their first health report arrives.

### Configuration knobs

All four knobs are set at `xrlenv up` startup; there is no live-reload.

| Flag | Default | Meaning |
|------|---------|---------|
| `--adaptive-admission` | off | Master switch. No other `--aimd-*` flag has any effect when this is off. |
| `--aimd-initial-limit` | `16` | Slow-start seed. Right value is cluster-shape-dependent — a 32-vCPU node doing short SWE-bench tasks can typically sustain 32–48; start conservative and let the controller ramp. |
| `--aimd-p95-threshold-s` | `60.0` s | Latency threshold for a bad tick. 60 s is appropriate for tasks whose `docker run` completes in under a minute; raise it for image-heavy workloads with legitimately slow container-start times. |
| `--aimd-max-limit` | `64` | Runaway guardrail. The real bound is node health, not this number; `max_limit` only prevents a long quiet stretch from drifting the limit to an unsafe value before the next stress event. |

### Observability

- **Admin panel** — the "Cluster health" page shows each node's current
  admission limit and the monotonic timestamp of its last contraction.
- **Prometheus** — `xrlenv_node_admission_limit{node_id="..."}` is updated
  after every tick so you can graph the sawtooth behavior of the AIMD loop.

### Relationship to the static estimator

The static estimator (`StaticCapacityEstimator`) and the adaptive controller
operate at different layers. The static estimator enforces resource caps (CPU,
memory, disk) for hard-eligibility filtering at placement time. The adaptive
controller enforces a health-derived concurrent-rollout cap. A node must pass
both: it must have resource slack for the request, and its running-rollout count
must be below its current admission limit.

### Two AIMD controllers — disambiguation

XRLEnv uses AIMD in two distinct places. The table below summarizes both so
you can tell them apart:

| | Admission AIMD (this section) | Pull AIMD ({doc}`images/cache_eviction`) |
|--|--|--|
| **Where it runs** | Control plane, `HealthAimdController` | Each node agent, `PullAimdController` |
| **Enabled by** | `--adaptive-admission` on `xrlenv up` | Always on (disable via `pull_aimd_enabled=False` in node config) |
| **What it caps** | Max concurrent rollouts placed per node | Max concurrent image pulls per node |
| **Signal** | Node's `docker run` create p95 + docker error/timeout counts | Node's current in-use container count |
| **Decrease trigger** | Bad health tick (error, timeout, or p95 > threshold) | Busy node (in-use containers above `pull_busy_threshold`) |
| **Config surface** | `xrlenv up --aimd-*` flags | `XRLENV_PULL_CONCURRENCY[_CEILING/_INITIAL]` env vars on each node |

Both are per-node and use additive-increase / multiplicative-decrease, but
they operate at different layers and respond to different signals.

### Code pointers

- `xrlenv/control/capacity.py::HealthAimdController`, `AimdConfig`,
  `NodeHealthInput`
- `xrlenv/control/aimd_loop.py::AimdControlLoop`
- `xrlenv/cli/__main__.py` — `--adaptive-admission` and `--aimd-*` argument
  definitions

## Fleet reservation (multi-container tasks)

Some tasks need more than one container. Without coordination, each
container goes through the normal admission path independently, and
nothing stops them from landing on different nodes or together
oversubscribing a node that appeared to have room when only the first
was placed.

Fleet reservation is an opt-in mechanism that reserves a CPU + memory
footprint for the entire task atomically, so all companion containers
are guaranteed to land on the same node and within the declared budget.
Tasks that do not set a `fleet_id` label go through the existing
single-container path unchanged.

### Identity

`fleet_id` is a **third identity axis** distinct from `task_key` and
`instance_id`. `task_key` is for fairness spread; `instance_id` is
execution identity; `fleet_id` groups companion containers that must
co-locate.

### Wire protocol

The consumer declares a fleet through Docker labels on the
**fleet-opening container** (the first acquire for the task):

| Label | Role |
|---|---|
| `xrlenv.fleet_id` | Unique fleet identifier. Present on opener and all companions. |
| `xrlenv.fleet_cpu_request` | Total CPU footprint (float vCPUs) the task will consume at peak. Set only on the opener. |
| `xrlenv.fleet_mem_request` | Total memory footprint (int bytes) the task will consume at peak. Set only on the opener. |

Companion acquires carry only `xrlenv.fleet_id`. Companions are
co-located on the opener's node and drawn from the reservation rather
than going through capacity-gated placement.

### Overflow: graceful degradation

If the task's actual container concurrency at runtime exceeds the
declared footprint, the over-budget companion is admitted through the
normal capacity-gated path (charged its own resources, queued if the
cluster is full) rather than hard-failing the acquire. The reservation
still guarantees the declared footprint's worth of co-located slots;
anything beyond it is best-effort. A warning is logged with the fleet
id and the overrun amounts; raise `xrlenv.fleet_cpu_request` /
`xrlenv.fleet_mem_request` to cover the real peak.

### Restart safety and TTL reclaim

Each open fleet reservation is persisted as a small row in `StateStore`.
On control-plane restart the coordinator rebuilds the in-memory fleet
table from the live container set's `xrlenv.fleet_id` labels, so no
footprint is lost across restarts.

If a consumer crashes without destroying its containers, the reservation
is reclaimed after `XRLENV_FLEET_RESERVATION_TTL_S` seconds (default
600 s) of inactivity. The TTL resets on each new companion acquire, so
a long-running task does not get reclaimed mid-flight.

| Knob | Default | Effect |
|---|---|---|
| `XRLENV_FLEET_RESERVATION_TTL_S` | `600` | Seconds of inactivity before a leaked fleet reservation is reclaimed. Tune up for very long tasks; tune down for faster cleanup of crashed consumers. |

### Scheduler integration

`Scheduler.place(reserve=footprint)` is the one API change that
supports fleet. When a fleet-opening acquire is in flight, the
scheduler atomically picks the target node and records the footprint
as a pending reservation. Subsequent companion placements read that
pending reservation into their cluster-load view, so concurrent
admission calls see the full fleet footprint immediately without
waiting for any container record to commit.

## Tiebreaks and edge cases

| Situation | Behavior |
|---|---|
| Equal scores | Ties break by `node_id` for deterministic tests and admin output. |
| No nodes have the image | Placement falls back to resource slack unless a preferred home is registered. |
| Image-aware placement disabled by embedder | Score uses resource slack only; preferred-home signal also suppressed. |
| Preferred home is saturated | A non-preferred feasible node can win on slack alone. |
| Node disappears | Open work is marked lost/failed through the node-control layer; future placements exclude the disconnected node. |
| Candidate opts out of an axis (`cpu_request <= 0`) | Estimator returns a sentinel cap for that axis so it doesn't dominate the bottleneck — see Quick notes. |

## What the scheduler does not do

- It does not predict future image needs.
- It does not optimize for cloud cost, region price, or
  spot/on-demand mix.
- It does not do priority lanes or per-user rate limits.
- It does not scrape live OS utilization to reclaim idle
  reservations.

## Operator views

| Surface | What it shows | Concepts on this page |
|---|---|---|
| `xrlenv nodes` (CLI) | Connected nodes, heartbeat age, current usage. | Connectivity, headroom-adjusted slots. |
| `xrlenv rollouts` (CLI) | Recorded rollout state. | Placement outcomes. |
| `/nodes` | Connected nodes and resource usage. | Backend support, overhead-deducted slack. |
| `/capacity` | Per-node capacity estimates with binding-constraint labels. | All of "Capacity and resource slack". |
| `/sandboxes` | Running containers and their node placement. | Pending vs committed reservations. |
| `/rollouts/raw` | Raw per-rollout placement decisions. | Score breakdown, task-fairness rejections. |
| `/images/cache`, `/images/catalog` | Per-node and cluster-wide image state. | Image affinity, preferred home. |

When something looks off, the right entry point is usually
`/capacity` — the binding-constraint label tells you which axis
is the bottleneck without having to inspect each axis manually.

## Code pointers

- `xrlenv/control/scheduler.py::Scheduler.place`,
  `commit_placement`, `release_placement`
- `xrlenv/control/scheduler.py::PlacementFeatures`,
  `weighted_sum_score`, `DEFAULT_SCORE_FN`
- `xrlenv/control/capacity.py::StaticCapacityEstimator`,
  `HeadroomConfig`, `BackendOverhead`, `CapacityCell`
- `xrlenv/control/image_presence.py::query_image_presence`
- `xrlenv/control/admission.py::AdmissionQueue.acquire`,
  `kick`, `_drain_once`

## Quick notes

Nuanced or status-dependent details you only need when they come
up.

### Axes the candidate opts out of

A template that declares `cpu_request <= 0` or
`mem_request_bytes <= 0` is opting out of accounting on that
axis. Pure-Python adapters in trust mode that don't request CPU
or memory at all are the practical case. The estimator returns a
sentinel cap (effectively "unbounded") for those axes so they
don't artificially dominate the min in `slack_after_placement`.

A consequence: if a candidate opts out of every axis, there's no
signal to rank nodes by; the estimator returns slack `1.0` so the
deterministic `node_id` tiebreak decides. This is rare in
practice but valid.

### Online refinement (future work)

`CapacityEstimator.report_usage(node_id, template_id, peak)` is
defined on the protocol and is a no-op debug log today. The
intent is to fold a per-template EMA-of-p95 effective request
into the static prediction as the node agent ships usage
telemetry. Embedders can wire the call now; the surface won't
change when refinement lands.

### Custom score functions

Operators embedding the scheduler can replace the default with
any pure-arithmetic `ScoreFn` that takes `PlacementFeatures` and
returns a float. Two rules to keep things sane:

- The function must not perform I/O. The scheduler calls it
  inside the placement lock, once per candidate per acquire.
- If you change weight ratios, validate at boot — the
  `weighted_sum_score` factory raises on out-of-range or non-
  summing weights so misconfiguration fails before any rollout
  arrives.

### Stale queued rows after process restart

The queue rows in `StateStore` survive a control-plane restart,
but the in-memory `_Waiter` futures do not. On restart the drain
worker sees rows with no matching waiter and drops them. The
caller (whose `start_rollout` died with the previous process) has
to retry — the in-process API has no way to deliver a placement
to a process that no longer exists. Tooling that issues
`start_rollout` from outside the control plane should be
restart-tolerant on its side.

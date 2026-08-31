# Timeouts and deadlines

XRLEnv has timeouts and deadlines at several layers. This page is the
single reference for all of them: the ones you set per call, and the
system defaults the cluster applies when you don't.

There are three families:

- **Template rollout deadlines** — for `Client.rollout(...)` workflows.
- **Container exec timeouts** — for command execution inside a
  container.
- **Container acquire / lifetime bounds** — for
  `Client.acquire_container(...)` and the Docker SDK drop-in.

## Template rollouts

For `Client.rollout(...)`, pass a `Deadline` or set defaults in a
run-config:

```python
from xrlenv import Deadline

session = await client.rollout(
    "hello-shell",
    deadline=Deadline(
        hard_s=60,
        step_timeout_s=10,
        idle_ttl_s=120,
    ),
)
```

| Field | Meaning |
|---|---|
| `hard_s` | Outer wall-clock budget for the rollout. |
| `step_timeout_s` | Budget for one `session.step(...)`. |
| `setup_timeout_s` | Budget for environment setup. |
| `teardown_timeout_s` | Budget for cleanup. |
| `init_timeout_s` | Budget for initialization. |
| `idle_ttl_s` | Maximum idle time between `step()` or `heartbeat()` calls. |

If the hard deadline fires, XRLEnv truncates the rollout and returns
the partial trajectory. If the idle TTL fires, the next step reports
that the rollout was truncated because the client stopped making
progress.

## Container exec

For both `Client.acquire_container(...)` sessions and the Docker SDK
drop-in, exec timeouts live on the operation:

```python
result = await session.exec(
    ["bash", "-lc", "pytest -q"],
    timeout_s=600,
)
```

Use `exec_stream(...)` for long-running commands that should keep
stdout/stderr flowing while the command runs:

```python
async for chunk in session.exec_stream(
    ["bash", "-lc", "pytest -q"],
    timeout_s=3600,
):
    ...
```

The Docker SDK drop-in maps docker-py exec calls onto the same
underlying container exec RPCs.

## Container acquire and lifetime

`Client.acquire_container(...)` accepts three independent bounds.
Each governs a different stage and has a system default, so you only
set the ones you need to override:

```python
session = await client.acquire_container(
    image="example/grader:1",
    acquire_timeout_s=1800,    # cold-pull budget on the chosen node
    queue_timeout_s=7200,      # how long to wait for cluster capacity
    session_deadline_s=10800,  # max lifetime before the cluster reaps
)
```

| Parameter | Governs | Default |
|---|---|---|
| `acquire_timeout_s` | The image pull / container-create round trip on the chosen node. Raise it for large images on a slow link. | 600 s |
| `queue_timeout_s` | How long the request waits in the cluster admission queue for capacity. Waiting in the queue is **not** a failure — a queued request consumes no cluster resources and does not erode any run-time deadline — so the default is a long backstop, not a deadline you are expected to hit. **Set a small value to opt into fail-fast** ("give me a slot within N seconds or fail"). | 24 h |
| `session_deadline_s` | Wall-clock cap on the container's lifetime. Once it passes, the cluster force-destroys the container — a safety net so a consumer that exits without calling destroy can't leak the container and its capacity. Raise it for genuinely long-running work. | 4 h |

### What you see while queued

A request waiting in the admission queue is not silent. The SDK polls the
cluster every 5 seconds and logs at `INFO`:

```
acquire queued — position N of M in the cluster admission queue; waiting for capacity (not an error)
```

`N` is this request's current FIFO position; `M` is the total queue depth.
A fast acquire that places immediately logs nothing — the poller fires after
the first 5-second interval and is cancelled before it fires if the acquire
returns quickly.

This log line is the expected signal for a queued job. If you see it, the
cluster is at capacity and the request is waiting its turn; it will proceed
as soon as a slot opens. It is not a sign that the acquire is hung or
failing.

### Setting these through the Docker SDK drop-in

The drop-in's `containers.create(...)` / `containers.run(...)` accept
all three — `acquire_timeout_s`, `queue_timeout_s`, `session_deadline_s`
— as keyword arguments. Because docker-py's high-level managers reject
keyword arguments they don't recognize, you can equivalently pass them
as reserved labels — the label form always works:

```python
client.containers.run(
    "example/grader:1",
    detach=True,
    labels={
        "xrlenv.acquire_timeout_s": "1800",
        "xrlenv.queue_timeout_s": "7200",
        "xrlenv.session_deadline_s": "10800",
    },
)
```

An explicit keyword argument wins over the label. All three are
cluster-mode only — in local-Docker mode they are silently ignored
(there is no admission queue, no remote lifetime reaper, and the local
daemon governs its own pulls).

Raising `acquire_timeout_s` matters most for benchmark suites that use
a distinct large image per task: every acquire is a cold pull, so on a
contended cluster the 600 s default can be too short. If acquires fail
with a pull/acquire timeout, raise it here.

### Why `session_deadline_s` exists

A container acquired this way otherwise lives until you explicitly
remove it. If the process that acquired it exits early — killed,
crashed, or interrupted — nothing else knows to clean it up: the
container keeps running, its capacity stays reserved, and it keeps
showing as `running` on the admin panel. `session_deadline_s` (and its
default cap) is the backstop that guarantees eventual cleanup. Set it
to a value comfortably above your real workload; it is a leak limit,
not a normal completion budget.

For most hard-killed harnesses, sessions are now reclaimed much sooner
than the 4 h deadline — see the consumer-liveness reaper below.

### Consumer-liveness reaper

When a harness process is hard-killed (SIGKILL, OOM, EC2 spot
preemption), it leaves its containers running and its capacity reserved
until `session_deadline_s` expires — historically up to 4 hours. The
consumer-liveness reaper closes this gap: it reclaims those sessions in
roughly **15 minutes** (the quarantine horizon), while never touching a
healthy long-running job — or a merely *stalled* one.

**Two phases, because silence is not death.** Crossing the liveness TTL
(120 s) marks a session `suspect`: a warning and a metric, nothing else.
Only continued silence through the quarantine horizon (900 s) destroys
it. The distinction matters because a consumer whose *host* stalls — a
memory-reclaim stall, a frozen VM — is alive and will come back, but at
the TTL it is indistinguishable from one that exited. Destroying it
throws away real work; waiting costs only that a genuinely dead
consumer holds its slot longer, which the 4 h deadline bounds anyway.
The liveness reaper is a reclamation *latency optimization*, not the
leak backstop, so it is tuned to favour waiting.

**How it works.** The SDK automatically beats each live raw-container
session on a background interval. The control plane tracks these
heartbeats. A session is eligible for liveness reaping only when all
four hold:

1. The consumer has sent at least one heartbeat (opt-in signal — sessions
   from older SDK versions that never heartbeat fall back to the
   deadline cap, preserving backward compatibility).
2. No session-scoped RPC is in flight — an open `exec`, `put_archive`,
   or `get_archive` means the consumer is still connected and waiting.
   A 30-minute blocking exec is never reaped mid-command.
3. The session is not already being torn down — a destroy in flight for
   any other reason (a slow consumer `destroy`, the wall-clock deadline
   sweep) takes it out of both liveness phases, so it is never marked
   `suspect` mid-teardown.
4. The liveness clock is staler than the relevant threshold — the TTL
   (default 120 s) to be marked `suspect`, the quarantine horizon
   (default 900 s) to actually be destroyed. Both are measured from the
   same clock, so a single signal retires both at once; every RPC and
   every explicit heartbeat resets it, and a suspect session that
   signals again is restored with no work lost.

The liveness clock is reset by every session-scoped RPC (implicit
heartbeat) and by every explicit keepalive beat. The raw-GC reconciler
runs the sweep once per interval, destroying at most
`XRLENV_RAW_LIVENESS_REAP_BATCH` sessions per sweep so a mass
die-off does not flood the cluster with simultaneous destroys.

**No consumer changes required.** The keepalive runs entirely inside
`Client`. It registers automatically when `acquire_container` returns
and deregisters when `destroy` completes. Existing code using
`from_env()` / `containers.run()` / `exec_run()` / `container.remove()`
gains liveness-reaping on a **package upgrade alone** — no API changes.
The keepalive is batched: one RPC per process per interval carries all
that process's live session ids, so overhead scales with the number of
consumer processes, not the number of sessions.

**Net behavior at a glance:**

| Scenario | Time to reap | Terminal status |
|---|---|---|
| Hard-killed harness (SIGKILL, OOM) | ~quarantine horizon (≈ 15 min by default; `suspect` at ≈ 2 min) | `reaped` |
| Consumer alive and mid-exec | Never — in-flight RPC blocks reap | — |
| Consumer alive and idle (heartbeats flowing) | Never | — |
| Session past wall-clock deadline | Reaps at `session_deadline_s` regardless | `reaped` |
| Reap whose teardown is not node-confirmed (raises or times out) | Re-attempted on the next sweep | Not sealed yet — the row stays `running` |

`reaped` is distinct from `failed`. It means the platform tore the session down
on purpose and teardown completed cleanly. Every teardown carrying a reason
seals `reaped` — the wall-clock deadline sweep and the liveness quarantine sweep
above, but also a group teardown (`terminate_raw_group`) and the orphan sweep
sealing a container the node reaped on its own (disk guard, OOM); the row's
`error` column carries the specific cause. It does **not** indicate an error in
the rollout's work and is excluded from the admin panel's high-failure-rate
alert.

A reap whose teardown is *not* node-confirmed — the node raised or the destroy
timed out — seals **nothing**. Capacity is released only on a node-confirmed
destroy, so the session stays charged, the row stays `running`, and the
reconciler re-attempts the teardown on its next sweep (or the orphan sweep
seals it once the container is confirmed gone).

**Env knobs (set on the control-plane host or consumer host before
process start):**

| Variable | Side | Default | Meaning |
|---|---|---|---|
| `XRLENV_RAW_LIVENESS_TTL_S` | Server | `120` | How long a raw session may go with no liveness signal (RPC or heartbeat) before it is marked `suspect`. This no longer destroys anything — it is the warning threshold. Setting it to **7200 s or more** silently disables liveness reaping entirely — see the warning below. |
| `XRLENV_RAW_LIVENESS_QUARANTINE_S` | Server | `900` | How long a suspect session may stay silent before it is actually destroyed. Must be **strictly greater** than the TTL: both phases read the same clock, so a value at or below it (including `0`) makes a session crossing the TTL a suspect *and* a reap candidate in the same sweep — i.e. destroy-on-TTL again. Such a value is not honoured, and it is **not** clamped to the TTL (equality is just as broken); the control plane logs a WARNING and uses `2 x TTL` instead — 240 s at the default 120 s TTL. Sized from measured consumer stalls. |
| `XRLENV_RAW_HEARTBEAT_INTERVAL_S` | SDK (consumer) | `30` | Keepalive beat cadence — roughly 1/4 of the TTL, so a couple of dropped beats don't flag a healthy session `suspect`. Set to `0` to disable the keepalive (sessions then rely on the wall-clock deadline only). Validated on read: unparseable, negative, non-finite, or `>= 86400` falls back to `30` with a WARNING; `>= 120` (the control plane's default liveness TTL) is honoured but warned about, since it cannot keep an idle session alive against a default-TTL server. |
| `XRLENV_RAW_LIVENESS_REAP_BATCH` | Server | `50` | Maximum liveness reaps per GC reconciler sweep, so a mass die-off does not fire thousands of destroys at once. The remainder reap on the next sweep (60 s later by default). |

The 4 h wall-clock deadline (`session_deadline_s`) is unchanged. It
remains the absolute cap for genuinely long-running jobs and for
sessions from consumers that never heartbeat.

**Tuning footgun: a quarantine horizon at or beyond the deadline turns the
feature off.** The liveness reaper only earns its keep by reclaiming *sooner*
than the wall-clock deadline. Both sweeps run in the same GC pass, deadline
first, so if the horizon lands at or past the deadline the deadline sweep
destroys the session every time and the quarantine branch is unreachable —
liveness reaping is silently disabled for every session using the default
deadline.

The easy way to hit this is **indirect**: raise `XRLENV_RAW_LIVENESS_TTL_S`
without touching `XRLENV_RAW_LIVENESS_QUARANTINE_S`. The quarantine then fails
the "strictly greater than the TTL" check, is replaced with `2 x TTL` (see the
table above), and any TTL of **7200 s or more** puts that synthesized horizon at
or beyond the 4 h default deadline. Setting `XRLENV_RAW_LIVENESS_QUARANTINE_S`
directly to `14400` or more does the same thing.

The control plane logs a WARNING at startup naming both values when this
holds. It **warns rather than clamps**, because `session_deadline_s` is
overridable per acquire — a long horizon can be deliberate for a fleet whose
consumers all raise their own deadline. If you did not intend it, lower the
quarantine (or the TTL it is derived from), or raise `session_deadline_s` on
the acquires that need it.

## System defaults

Defaults the cluster applies when you don't override them. Most are
fixed; the few an operator can tune are noted.

| Default | Governs | Value |
|---|---|---|
| Acquire round-trip | Control-plane wait for a container acquire to complete. Matches the node-side pull timeout below. | 600 s |
| Admission-queue wait | Default `queue_timeout_s` — the backstop wait for a slot in the admission queue, for both container acquires and template rollouts. Long by design: queue-wait is not a failure. A small consumer-set value opts into fail-fast. | 24 h |
| Session lifetime cap | Default `session_deadline_s` — the abandoned-container reap deadline. Operator-tunable per node-cluster build. | 4 h |
| Consumer-liveness TTL | How long a raw-container session may go with no liveness signal (RPC or heartbeat) before it is marked `suspect` (warning + metric, no teardown). Only sessions that have heartbeated at least once, have no RPC in flight, and are not already being torn down are eligible. Tunable via `XRLENV_RAW_LIVENESS_TTL_S` on the control-plane host. | 120 s |
| Consumer-liveness quarantine | How long a suspect session may stay silent before it is force-destroyed. A consumer that signals during quarantine keeps its session. Tunable via `XRLENV_RAW_LIVENESS_QUARANTINE_S`. | 900 s |
| SDK keepalive interval | Cadence at which the SDK beats live raw-container sessions to the control plane. Batched per process. Tunable via `XRLENV_RAW_HEARTBEAT_INTERVAL_S` on the consumer host; `0` disables. | 30 s |
| Destroy round-trip | Control-plane wait for a container teardown. Sized for multi-GB overlay teardown under load. | 300 s |
| Archive round-trip | Control-plane wait for a file put/get into a container. Sized so a node busy with cold image pulls can still service it. | 300 s |
| Node docker HTTP timeout | The node-agent's docker client per-request HTTP timeout. If a failure reports a timeout much smaller than this, the node-agent is running a stale binary — redeploy it. | 600 s |
| Node-health cooldown | After a node's destroy reply times out (the node is I/O-wedged — e.g. EBS IOPS ceiling hit during a large overlay teardown), the control plane seals the session as `released` (not `failed`), health-gates the node for this window so no new work is placed on it, and lets the raw-GC orphan reconciler force-destroy the container once the node recovers. Operators no longer see false destroy-path rollout failures from nodes hitting IOPS ceilings. | 120 s |
| Per-node destroy concurrency | Maximum simultaneous container teardowns per node, so the docker daemon isn't overwhelmed. Operator-tunable via the node-agent config. | 4 |
| Per-node create concurrency | Maximum simultaneous container creations (`docker run`) per node, so a burst of acquires doesn't overwhelm the daemon. Operator-tunable via the node-agent config. | 4 |
| Image pull attempts | How many times a node retries a registry pull before failing the acquire — rides out a flaky registry / auth endpoint under heavy cold-pull load. Retries stay within the acquire deadline. | 3 |
| Per-node pull concurrency | How many distinct images one node-agent pulls at once. Governed by the AIMD limiter — see below. | Floor 2 (`XRLENV_PULL_CONCURRENCY`); ceiling 64 (`XRLENV_PULL_CONCURRENCY_CEILING`); initial 16 |
| In-sandbox stub call timeout | Node-agent safety-net cap for a single in-sandbox call when no per-call cap is supplied. | 1 h |

Internal cadences (node heartbeat interval, GC reconcile interval,
admission-queue poll interval) are not workload-tunable and are
omitted here.

### Tuning per-node pull concurrency

Pull concurrency on each node is governed by a single **AIMD**
(additive-increase / multiplicative-decrease) limiter. All pulls share
the same adaptive limit — there is no separate prefetch lane. The limiter
moves between a floor and a ceiling based on the node's live container
count:

- **Busy node** (in-use containers above `pull_busy_threshold`, default 0):
  the limit halves each tick, down to the floor. Cold pulls do not starve
  live agent containers.
- **Idle node** (in-use containers at or below the threshold): the limit
  rises by `pull_aimd_additive_step` each tick (default 2), up to the
  ceiling. `xrlenv build apply` against a quiet cluster ramps up to
  saturate the registry or FSx pipe.

Key config fields and env vars (read by `xrlenv-node` at startup from
`/etc/xrlenv/node.env`):

| Config field | Env var | Meaning | Default |
|---|---|---|---|
| `pull_concurrency` | `XRLENV_PULL_CONCURRENCY` | AIMD floor (busy minimum) | 2 |
| `pull_concurrency_ceiling` | `XRLENV_PULL_CONCURRENCY_CEILING` | AIMD ceiling (idle maximum) | 64 |
| `pull_concurrency_initial` | `XRLENV_PULL_CONCURRENCY_INITIAL` | Starting limit at daemon start | 16 |

**To change floor or ceiling:** edit the relevant env var in
`/etc/xrlenv/node.env` (or the systemd drop-in) on the node and restart
the node-agent:

```bash
sudo systemctl restart xrlenv-node
sudo systemctl status xrlenv-node   # confirm it came back
```

These are **node-agent / operator** settings — they are *not* reachable
from the consumer SDK or the Docker-SDK drop-in, because they govern
node-wide pull behaviour, not any single acquire.

Guidance:

- Raise the **ceiling** for cold-pull-heavy benchmark runs where the node
  is typically idle during apply (e.g. `xrlenv build apply` on a quiet
  cluster). The AIMD loop ramps up automatically; setting a higher ceiling
  just raises the cap it converges to.
- Lower the **floor** toward `1` on nodes where even a single concurrent
  cold pull visibly affects live agents (very latency-sensitive workloads
  or a shared link).
- To check whether more concurrency would help: watch inbound bandwidth
  (`ifstat`, `nload`) during a run. If the link sits well below its
  ceiling with idle gaps, raising the ceiling will lift throughput. If
  the link is already saturated, the bottleneck is bandwidth — use a
  same-network registry instead.

A blank or non-positive `XRLENV_PULL_CONCURRENCY` falls back to the
library default of 2. A blank `XRLENV_PULL_CONCURRENCY_CEILING` falls
back to 64.

See {doc}`/technical_details/images/cache_eviction` for the full AIMD
parameter table.

## See also

- {doc}`run_config` — default deadline values for template rollouts.
- {doc}`api_reference` — SDK methods that accept timeouts.

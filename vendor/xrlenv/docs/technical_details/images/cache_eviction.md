# Image cache and eviction

A node has finite disk and benchmark images are big. The node-side
cache manager has three jobs:

1. **Know what's on disk** — track per-image state, refcounts, and
   LRU timestamps so the rest of the platform can reason about
   coverage.
2. **Reclaim disk under pressure** without re-downloading anything
   expensive when a cheap candidate exists.
3. **Stay out of the way** when there's no pressure — never touch a
   warm working set. A periodic sweep checks the same disk-pressure
   threshold as the on-pull path, so an idle, healthy cluster is a
   no-op (one `statvfs` per minute and exit).

This page is for operators tuning eviction thresholds, debugging
unexpected pulls, or deciding what to pin. The implementation is
`ImageCacheManager` in `xrlenv/node/image_cache.py`.

## What the cache tracks

Each image on a node is reported in one of four operational states:

| State | Meaning | Evictable |
|---|---|---|
| `in_use` | At least one running container currently references the image. | No |
| `pinned` | Operator pin file or runtime pin protects it. | No |
| `recently_used` | Touched within the recent-use window. | Yes, after cold images |
| `cold` | Not in use, not pinned, and outside the recent-use window. | Yes, first |

Defaults from `ImageCacheConfig`:

- Eviction **starts** when free disk drops below the adaptive START
  headroom (see next paragraph). Absolute floor: **15 GiB**; cap:
  **50 GiB**.
- Eviction **stops** when free disk reaches the adaptive STOP headroom
  — one extra slot above START, for hysteresis. Absolute floor:
  **25 GiB**; cap: **75 GiB**.
- A periodic background sweep checks free disk every **60 seconds**.
- Recent-use window is **30 minutes**.
- Pull concurrency floor defaults to **2** per node (`pull_concurrency`
  — the AIMD minimum; see below).
- Pull concurrency ceiling defaults to **64** per node
  (`pull_concurrency_ceiling` — the AIMD maximum; see below).
- Default pull timeout is **600 seconds**.

**Workload-adaptive headroom.** The eviction thresholds are not a
fixed fraction of disk. Instead, the headroom kept free scales with
the workload's image size:

```
headroom = clamp(slots × largest_cached_image × disk_safety_factor,
                 floor, cap)
```

- START uses `evict_headroom_slots` (default **4**); STOP uses
  `evict_headroom_slots + 1` (= **5** by default) — the extra slot is
  the hysteresis that prevents eviction from immediately re-triggering.
- `evict_disk_safety_factor` (default **1.5**) is a single dimensionless
  knob: higher means more free headroom and fewer concurrent pulls
  (safer); lower means a denser cache.
- Floors apply when the adaptive value is smaller than the absolute
  minimum — for example on a cold cache before any image has been
  observed, or on a small-disk/small-image node.
- Caps (`evict_threshold_cap_bytes` **50 GiB** / `evict_target_cap_bytes`
  **75 GiB**) bound a pathologically large base image from reserving
  an unreasonable fraction of the disk. Both caps are tunable per node
  via `XRLENV_EVICT_THRESHOLD_CAP_GB` / `XRLENV_EVICT_TARGET_CAP_GB`.
- Cold start / empty cache: `largest_cached_image` is unknown (0), so
  the floors apply — a fresh node still keeps a sane absolute buffer.

**Why this replaced the old fraction model.** The previous model used
20% of total disk as the START threshold and 30% as the STOP target.
On a 500 GiB node that reserved 100 GiB idle; on a 1 TiB node, 300 GiB
— for nothing. And it under-reserved on a small disk. The operator hit
this directly: a 500-image ``xrlenv build apply`` over-reserved and
mass-evicted a cache that fit comfortably, because the STOP target was
0.30 × 500 GiB = 150 GiB free rather than "a few pulls' worth of
space". Sizing the reserve to image-pulls' worth of space means a
500 GiB and a 1 TiB node keep the same modest buffer, and that buffer
grows automatically when the workload's images get bigger.

**Eviction config** (see also the AIMD config table below for
pull-concurrency fields):

| Field | Meaning | Default | Env var |
|---|---|---|---|
| `evict_headroom_slots` | image-pulls' worth of free disk kept as START headroom (STOP uses slots+1) | 4 | *(config only)* |
| `evict_disk_safety_factor` | margin on both the eviction headroom and the disk-bounded pull ceiling | 1.5 | *(config only)* |
| `evict_threshold_bytes` | absolute floor for the START headroom (empty cache / small images) | 15 GiB | *(config only)* |
| `evict_target_bytes` | absolute floor for the STOP headroom | 25 GiB | *(config only)* |
| `evict_threshold_cap_bytes` | ceiling on the START headroom | 50 GiB | `XRLENV_EVICT_THRESHOLD_CAP_GB` |
| `evict_target_cap_bytes` | ceiling on the STOP headroom | 75 GiB | `XRLENV_EVICT_TARGET_CAP_GB` |
| `sweep_interval_s` | background eviction-sweep cadence (also refreshes cached image stats) | 60 s | *(config only)* |
| `recent_window_s` | recent-use window | 30 min | *(config only)* |
| `XRLENV_CONTENT_GC_MIN_INTERVAL_S` | minimum interval between automatic Docker content-store GC sweeps; debounces the prune-lock storm at high pull concurrency | 60 s | `XRLENV_CONTENT_GC_MIN_INTERVAL_S` |

### Adaptive pull-concurrency limiter (AIMD)

Each node runs a single **AIMD** (additive-increase / multiplicative-decrease)
pull-concurrency limiter. All pulls — rollout-acquire `ensure_present` calls
and proactive `EnsurePresentCommand` prefetch dispatches — share the same
adaptive limit. There is no separate prefetch lane.

The limiter's bound moves between a floor and a ceiling based on the node's
own in-use container count:

- **Busy node** (in-use containers above `pull_busy_threshold`, default 0):
  multiplicative-decrease — `limit = max(floor, limit // 2)`. Cold pulls never
  starve time-sensitive agent containers already running on the node.
- **Calm/idle node** (in-use containers at or below `pull_busy_threshold`):
  additive-increase — `limit = min(ceiling, limit + pull_aimd_additive_step)`
  (step default 2). An idle cluster ramps up to saturate the registry or FSx
  pipe, so `xrlenv build apply` against a quiet cluster pulls at full speed.

The loop ticks every `pull_aimd_interval_s` (default 15 s). It runs
node-local — the only input is the node's own in-use container count; no new
wire protocol is involved. The implementation is `AdjustableSemaphore` +
`PullAimdController` in `xrlenv/node/adaptive_pull.py`, driven by
`ImageCacheManager.run_pull_aimd_loop()`.

The `prefetch` flag on `ensure_present` is telemetry-only — it no longer
selects a different semaphore.

**Config fields and env overrides** (all read by `xrlenv-node` at startup,
stamped into `/etc/xrlenv/node.env` by `deploy/bootstrap-common.sh`):

| Field | Meaning | Default | Env var |
|---|---|---|---|
| `pull_concurrency` | AIMD **floor** — minimum concurrent pulls on a busy node | 2 | `XRLENV_PULL_CONCURRENCY` |
| `pull_concurrency_ceiling` | AIMD **ceiling** — maximum concurrent pulls on an idle node | 64 | `XRLENV_PULL_CONCURRENCY_CEILING` |
| `pull_concurrency_initial` | Initial limit at start; clamped into [floor, ceiling] | 16 | `XRLENV_PULL_CONCURRENCY_INITIAL` |
| `pull_busy_threshold` | In-use container count at/below which the node is "idle" | 0 | *(config only)* |
| `pull_aimd_interval_s` | Tick cadence; 0 or negative disables the loop | 15.0 | *(config only)* |
| `pull_aimd_additive_step` | Slots gained per calm tick | 2 | *(config only)* |
| `pull_aimd_enabled` | Master switch; `False` pins the limit at `pull_concurrency_initial` | `True` | *(config only)* |
| `XRLENV_IO_THROTTLE` | Kill switch for the I/O-saturation input (`false`/`0`/`off` disables; default enabled) | enabled | `XRLENV_IO_THROTTLE` |
| `XRLENV_IO_UTIL_HIGH_PCT` | Disk %util upper watermark; above this the AIMD treats the node as busy (multiplicative-decrease) | `90` (0.90) | `XRLENV_IO_UTIL_HIGH_PCT` |
| `XRLENV_IO_UTIL_LOW_PCT` | Disk %util lower watermark; below this the I/O-saturated signal clears (hysteresis) | `70` (0.70) | `XRLENV_IO_UTIL_LOW_PCT` |

To raise the ceiling on a dedicated build node or change the floor, edit
`/etc/xrlenv/node.env` (or the systemd drop-in) and restart the daemon:

```bash
# Example: raise the ceiling on a node dedicated to xrlenv build apply.
# Edit /etc/xrlenv/node.env, set XRLENV_PULL_CONCURRENCY_CEILING=128, then:
sudo systemctl restart xrlenv-node
sudo systemctl status xrlenv-node   # confirm it came back
```

#### I/O-saturation input to the pull limiter

In addition to the in-use container count, the AIMD tick ORs in a second busy
signal: the disk %util of the node's data-root volume, sampled by
`DiskIoSampler` from `/sys/block/<dev>/stat`. This matters on EBS or other
provisioned-IOPS volumes where the IOPS ceiling can saturate well before
free-disk headroom runs out — without this signal, a heavy pull burst can peg
the device at 100% util and stall containerd's overlay teardown path, wedging
subsequent acquire calls even when the disk has space.

The sampler uses hysteresis to avoid rapid oscillation:

- When disk %util rises above `io_util_high` (default **90%**, env
  `XRLENV_IO_UTIL_HIGH_PCT`, expressed as an integer percentage 1–100), the
  node is treated as I/O-saturated → AIMD multiplicative-decrease fires
  (same path as "too many in-use containers").
- The saturated signal clears only once util drops back below `io_util_low`
  (default **70%**, env `XRLENV_IO_UTIL_LOW_PCT`), preventing the limit from
  thrashing on a device hovering near the high watermark.

`XRLENV_IO_THROTTLE` is the kill switch. Set it to `false`, `0`, or `off` to
disable the I/O signal entirely and revert to the pure container-count AIMD.
The fail-open rule applies regardless: if `/sys/block/<dev>/stat` is
unreadable or the sampler isn't wired, `io_saturated` is always `False` and
only the free-disk ceiling governs.

**Disk-bounded pull ceiling.** On each AIMD tick the node also caps its
pull-concurrency ceiling at what the free disk can buffer:

```
disk_ceiling     = max(floor, free_disk / (largest_cached_image × disk_safety_factor))
effective_ceiling = min(pull_concurrency_ceiling, disk_ceiling)
```

This is applied via `PullAimdController.set_ceiling()`, which also
clamps the live limit down immediately when the new ceiling is lower
than the current limit. The result: a pull burst auto-throttles as the
disk fills toward the eviction START headroom, so concurrent pulls
cannot overrun the reserve. Both controllers — eviction headroom and
pull ceiling — share the same `evict_disk_safety_factor` knob, so
raising or lowering it uniformly tightens or relaxes both.

**Daemon-free hot path.** Every eviction-trigger check and
pull-ceiling decision reads only:

1. A `statvfs` syscall for free disk (`shutil.disk_usage` — an OS
   call, not a Docker daemon round-trip).
2. An in-memory cached `_largest_image_bytes`.

The cached image stats are refreshed once per sweep tick (60 s) with a
cheap `list_images` call that skips `docker system df` (the SharedSize
layer-graph walk, which is expensive and only used by `xrlenv build
calibrate` via `include_shared_size=True`). This is why the hot path
stays sub-second even while dozens of containers run and a build burst
pulls — the previous design's per-decision daemon calls were what
wedged nodes under heavy concurrent load.

**The admin `/images` pages ride the same cache.** A `report_images`
RPC (the per-node fan-out behind `/images/cache` and `/images/catalog`)
serves its image listing from the sweep-maintained cache when it is
fresher than one sweep interval, so opening the page during a heavy
build does **not** add a live `docker images.list` that competes with
the build for the daemon lock. Refcount / pin / tier state is still
computed live; only the image *set* and sizes are up to ≈60 s stale.
The control plane adds a second, short stale-while-revalidate cache over
the whole fan-out: the page is served from the last snapshot instantly
and tagged "data as of N s ago", with a background refresh kicked off
once the snapshot passes its TTL (`XRLENV_IMAGE_SNAPSHOT_TTL_S`, default
15 s). Net effect: the image pages stay responsive under load instead of
the 30 s+ stalls a synchronous per-render fan-out produced.

### Cold-pull deadline: three layers, two knobs

A cold image pull on a fresh node has three independent deadlines stacked
on top of each other. They all default to **600 seconds** so a legitimate
slow pull surfaces at the most informative layer rather than racing the
others:

| Layer | Knob | What fires when it expires |
|---|---|---|
| Control-plane → node wire wait | `Client.acquire_container(acquire_timeout_s=...)` → control plane sets `_send_and_wait(timeout_s=...)` on the bidi RPC to the chosen node | The control plane gives up waiting for the node's `AcquireContainer` reply and surfaces a wire timeout to the consumer. |
| Node-side `ensure_present` deadline | Same `acquire_timeout_s` rides in the request body as `pull_deadline_s` and is passed to `ensure_present(deadline_s=...)`. Default lives on `ImageCacheConfig.default_pull_timeout_s` when unset. | The node aborts the pull and returns a clean `TimeoutError`. |
| docker-py HTTP socket | `DOCKER_CLIENT_HTTP_TIMEOUT_S` in `xrlenv/backends/docker.py` | urllib3 raises `ReadTimeout` mid-stream — the noisy, opaque failure. |

Note that the **consumer-to-control-plane gRPC call itself has no client-
side timeout** — `Client.acquire_container` blocks until either the
control plane returns (success or wire-timeout passthrough) or the
caller cancels the coroutine. The `acquire_timeout_s` kwarg is a
*request-body* knob; it controls the two control-plane-and-below layers
above, not the round-trip from your process.

The third layer is the easiest to overlook because it lives under the
SDK abstraction. docker-py's own default is 60 s, which used to abort
multi-GB pulls on slow links well before either upper layer noticed.
The `DockerBackend` constructor pins it to match the cache deadline,
and a unit test asserts the cross-file invariant
(`DOCKER_CLIENT_HTTP_TIMEOUT_S >= default_pull_timeout_s`).

If you're acquiring **known-huge images** (10+ GiB on a slow link),
pass `acquire_timeout_s=` to widen the upper two layers per call:

```python
async with await client.acquire_container(
    image="docker.io/jefzda/sweap-images:vuls-task-x",
    acquire_timeout_s=1800,  # 30 min for a 12 GiB cold pull
) as session:
    ...
```

If you ever need to widen the bottom layer too, bump
`DOCKER_CLIENT_HTTP_TIMEOUT_S` and `ImageCacheConfig.default_pull_timeout_s`
together — the cross-file test will fail loudly if you forget the pair.

The adaptive headroom means the reserve is always proportional to the
workload's pull burst, not the disk size. A 500 GiB and a 1 TiB node
with the same 8 GiB images reserve the same ~48 GiB START headroom —
enough for four concurrent in-flight pulls plus running containers'
overlay scratch. On a cold cache or a small-image node the absolute
floors (15/25 GiB) kick in, so a fresh node or a laptop-scale setup
never over-reserves either.

## Operator-driven eviction: `xrlenv images evict`

The automatic eviction loop handles disk pressure reactively. For the
complementary **proactive** case — you've rebuilt and re-pushed an image
under the same tag and want every node to drop its cached copy now, so
the next acquire pulls the fresh bytes — use `xrlenv images evict`.

```bash
xrlenv images evict <image_ref> \
    --connect-host <admin-host> \
    [--connect-port 8080] \
    [--operator-token <token>] \
    [--force]
```

The command fans an `EvictImageCommand` to every connected node via the
admin API. Each node matches `image_ref` **registry-agnostically**: a
bare `repo:tag` matches the registry-qualified
`host:5011/repo:tag` the node actually holds, so you can use the
same ref format your consumer config uses.

**Per-node outcomes:**

| Status | Meaning |
|---|---|
| `evicted` | Image was present and not in use (or `--force` was set); removed. Reports reclaimed bytes and the exact local tags removed. |
| `absent` | Image was not on this node. Successful no-op. |
| `in_use` | At least one running container holds the image. Skipped to avoid disrupting live rollouts. Re-run with `--force` to evict anyway. |
| `failed` | Node error or node unreachable. |

The command exits non-zero only when at least one node returned
`failed`. `absent` and `in_use` are successful outcomes — they mean
there was nothing to evict or that live rollouts are protected.

**Authentication.** `xrlenv images evict` requires an operator token
(`operator.admin` scope). Provide it via `--operator-token`, the
`$XRLENV_OPERATOR_TOKEN` env var, or `~/.xrlenv/secrets/operator.token`.

**Example — evict after a substrate rebuild:**

```bash
# The channel tag :dev was rebuilt and re-pushed.
# Evict the old digest from all nodes so the next acquire pulls fresh.
xrlenv images evict xrlenv-webarena-infinity/substrate:dev \
    --connect-host <control-plane-host>

# Example output:
# evict xrlenv-webarena-infinity/substrate:dev:
#   2 evicted / 2 node(s) queried, 1.02 GiB reclaimed
#   - node-aws-a: evicted (0.51 GiB, <registry-host>:5011/xrlenv-webarena-infinity/substrate:dev)
#   - node-aws-b: evicted (0.51 GiB, <registry-host>:5011/xrlenv-webarena-infinity/substrate:dev)
```

**Relationship to the freshness model.** The control plane's registry
tag→digest resolver ({doc}`registry_freshness`) is the *routine* path:
every new acquire for a channel tag automatically pins the current
registry digest, so a rebuilt+re-pushed tag reaches nodes on the next
acquire with no operator action. `xrlenv images evict` is the *escape
hatch*: use it when you want nodes to drop the old cached image
immediately rather than serving it for in-flight rollouts. Neither path
supersedes the other.

## When eviction runs

Eviction has two triggers, both gated by the same start-threshold
check:

1. **On-pull.** A new `ensure_present(image)` call detects free disk
   under the start threshold and runs eviction before the pull /
   build proceeds.
2. **Periodic sweep.** Every 60 s a background loop on each node
   checks free disk and runs eviction if the start threshold has
   been crossed.

The sweep exists because running containers fill disk by writing to
their overlay layers (`apt update`, `pip install`, build artifacts,
logs) — that pressure isn't visible to the on-pull trigger because no
new image is being pulled. The sweep's job is to free image-cache
disk so live containers' writable layers have room to spill into.

Both paths share a property the operator can rely on:

> If free disk is above the start threshold, **nothing evicts**.

The sweep on an idle, healthy node costs one `statvfs` syscall per
minute and exits — it never touches a warm working set. Build plans
applied to a quiet cluster stay warm; the sweep doesn't decay them.

## Worked example: sizing the reserve

**Setup:** a HyperPod worker with `/opt/sagemaker` EBS data-root of
500 GiB, largest single cached image of 8 GiB (a representative
swebench-verified task image). Defaults: `evict_headroom_slots` = 4,
`evict_disk_safety_factor` = 1.5, floors 15/25 GiB, caps 50/75 GiB.

**START headroom:**
```
clamp(4 × 8 GiB × 1.5,  floor=15, cap=50)
= clamp(48, 15, 50)
= 48 GiB
```

**STOP headroom:**
```
clamp(5 × 8 GiB × 1.5,  floor=25, cap=75)
= clamp(60, 25, 75)
= 60 GiB
```

Eviction starts only when free disk drops below 48 GiB and stops once
free climbs back to 60 GiB. The cache may grow to 500 − 48 = **452 GiB
of images before a single eviction** — roughly **56 cold 8-GiB images
held**. The old fraction model would have started evicting at 100 GiB
free (400 GiB used) and drained the cache down to 150 GiB free
(~350 GiB, ~44 images), discarding a cache that fit fine.

**Running containers (mixed-job factor).** Each live rollout writes to
its container overlay scratch (apt, pip, build artifacts, logs). The
48–60 GiB reserve is that write headroom plus room for in-flight pulls.
In-use images are never eviction candidates, so a node running 60
containers keeps all 60 live images; only cold images are eviction fuel
for container scratch.

**Build concurrency factor.** The disk-bounded pull ceiling stops a
build burst from eating the reserve. At 60 GiB free:

```
disk_ceiling = 60 / (8 × 1.5) = 5 concurrent pulls
```

As the disk fills toward 48 GiB the ceiling shrinks (48 / 12 = 4, …).
If the node is also running rollouts (busy) the AIMD halves further
toward the floor — so pulls auto-throttle exactly when scratch space is
scarce, then ramp back toward the static ceiling (64) once the disk
drains.

**Mixed cluster.** Each node sizes its own reserve from its own largest
image and its own free disk — no global tuning. A build-heavy node with
big base images reserves more (up to the 50/75 GiB caps); a rollout
node with small images reserves the floors (15/25 GiB). When the
workload's images grow, the reserve grows with them automatically.

**Large-image / cap case.** If the largest image were 20 GiB:

```
START = clamp(4 × 20 × 1.5 = 120, 15, 50) = 50 GiB  (capped)
STOP  = clamp(5 × 20 × 1.5 = 150, 25, 75) = 75 GiB  (capped)
```

The cap prevents one pathologically large base image from reserving the
whole disk. Raise it per node with `XRLENV_EVICT_THRESHOLD_CAP_GB` /
`XRLENV_EVICT_TARGET_CAP_GB`.

## What gets evicted first

Eviction only considers images that aren't `in_use` and aren't
`pinned`. Eligible images are sorted by:

1. **Rebuild-cost tier.** Cheap final task images evict before
   intermediate/stub-runtime layers, which evict before expensive
   base images.
2. **LRU timestamp within that tier.** Oldest touched image evicts
   first. Images never touched by this process sort as oldest.

This is deliberately **not** pure LRU. A base image can be older
than a final image but still be kept longer because rebuilding the
base image is more expensive.

Tier classification today reads two signals, in order:

1. **`org.xrlenv.role` Docker label** (authoritative when set):
   - `role=final` → `final` tier (cheap top layer).
   - `role=intermediate` → `stub_runtime` tier (medium, e.g. a
     benchmark addon's pip-install layer).
   - `role=base` → `base` tier (expensive bottom layer).
2. **Name pattern fallback** (for unlabeled images, typically
   upstream-base retags that can't carry labels): if the repo
   portion contains `-base/` (e.g. `swebench-verified-base/django…`),
   classify as `base`. Everything else defaults to `final`.

Plug-ins with non-standard tag conventions can pass a custom
`tier_classifier` to `ImageCacheManager`; the harbor / tb2 default
covers in-tree builders.

## Keeping things warm: pinning

Pinned images skip eviction entirely. Use pins for images that must
stay warm for the duration of a run. The admin image pages show
pinned counts and per-image pin state so operators can detect when
pins are consuming too much disk.

Operators set pins through the operator pin file or runtime pin RPC
today. The planned `pinned: true` flag on `build-plan.yaml` entries
will eventually flow straight to `ImageCacheManager.pin` after a
successful materialization, and the coordinator will run an upfront
pin-budget check at apply time — refusing the plan rather than
dispatching work that would immediately deadlock the cache. See
{doc}`build_plan` for the build-plan side of pinning.

## Pull/build idempotence

`ensure_present(image)` is safe to call repeatedly:

- On hit, it only touches the LRU timestamp.
- On miss, it serializes pull/build and eviction decisions per
  node.
- Concurrent requests for the same fresh image join the same in-
  flight work instead of starting parallel pulls.
- Re-running an applied plan is cheap when the image is already
  present.

## Defense in depth: the scheduler placement gate

The sweep covers the common case but can't reclaim disk if pins
cover most of the cache, non-image consumers (logs, build cache,
runaway crash dumps) explode, or the burst rate exceeds the sweep
cadence. As a safety net the scheduler **refuses placement** on any
node whose last-reported free disk is below 5 % of total or 5 GiB
absolute, whichever is larger. The refused rollout surfaces as
``CapacityExhausted`` rather than dying inside the container with
an opaque ``apt update`` failure.

The signal travels via the per-node heartbeat (every 5 s by
default), so the gate is at most one heartbeat stale. Freshly-
attached nodes that haven't sent their first heartbeat report
``(0, 0)`` — the gate treats that sentinel as healthy so a node
isn't blackholed before its first sample.

The same threshold drives the **disk-pressure pill** on the admin
``/capacity`` page so an operator sees the problem on the same page
they'd open when scheduling looks weird.

## Operator views

Use `/images/cache` to debug per-node disk pressure and
`/images/catalog` to inspect cluster-wide image coverage. The
`/capacity` page also flags disk-pressured nodes inline in the per-
node header.

See {doc}`/technical_details/scheduling` for how image state feeds
node selection.

## Quick notes

Nuanced or status-dependent details you only need when they come
up.

### Idle cluster, no rollouts

If a cluster goes idle and free disk stays above the start
threshold, nothing evicts even if every image is `cold`. The
periodic sweep checks the same threshold, so an idle, healthy
cluster never sees eviction churn. This is the property build
plans rely on for warm-pool semantics: an `xrlenv build apply`
followed by hours of idle leaves the placement intact.

If the operator wants to reclaim disk on an idle cluster anyway —
for example before reusing the host for an unrelated workload —
the recommended path is `xrlenv images prune` rather than relying
on eviction. Prune is explicit, scoped, and visible in the audit
log.

### Build-time grace window (planned)

Today, an image's LRU timestamp is set when something pulls or
builds it (which doubles as a "first touch") and refreshed on
every `ensure_present` hit. That works fine for registry pulls —
pull is fast, the next acquire is close behind — but it creates a
hazard for expensive `git`-source builds that the source-build
pipeline will introduce: a node could spend minutes building an
image and then evict it before any rollout has had a chance to
acquire it.

Design intent: separate `_built_at` from `_last_used` and add a
build-time grace window. An image inside its grace window will
sort as `recently_used` even if it has not yet been acquired. The
grace window will expire once the image is first touched by an
`ensure_present` hit, after which standard LRU semantics take
over. This will land alongside the source-build pipeline that
makes expensive builds reachable from a plan.

### Dynamic eviction toggle (planned)

Design intent for the operator-facing eviction toggle (not yet
implemented):

```bash
xrlenv images eviction disable                # cluster-wide
xrlenv images eviction disable --node aws-a   # per-node
xrlenv images eviction enable
xrlenv images eviction status
```

When eviction is disabled and a node hits the start threshold, the
next `ensure_present(image)` for an absent image will **fail
fast** with a structured `ImageCacheFull` error rather than
evicting any local image. Existing in-progress pulls/builds will
not be interrupted; only new work will be rejected. This is for
runs where you want a guaranteed working set on disk and prefer
scheduler-level backpressure to silent eviction.

Re-enabling brings the standard reactive flow back; no images
skipped during the disabled window will be retroactively
reclaimed.

Until this lands, the only way to suppress eviction is to keep
total working-set size under the start threshold or pin the
working set.

# Image distribution and build planning

A node can run a container only after the image is on its local
disk, and benchmark images are typically large — often a gigabyte
or more. XRLEnv treats image placement as part of scheduling rather
than a side problem: bytes are placed where they're likely to be
used, and image affinity then steers future containers toward warm
nodes. The goal is to avoid the cluster-wide "pull everything
everywhere" shape.

This page covers the **proactive path** — when you know your image
set ahead of time and want to prefetch it across the cluster. For
the runtime path, where images are pulled at acquire time, see
{doc}`on_demand`. Both paths share the same node-side cache
({doc}`cache_eviction`).

Specifically, this page answers:

- how to declare a known image set and what the schema looks like,
- how the planner decides where each image lives,
- what `xrlenv build apply` does end-to-end, and
- how planning stays coordinated with the cache evictor.

The implementation lives in `xrlenv/control/build_plan.py`,
`xrlenv/control/build_coordinator.py`, and the node-side
`ImageCacheManager` in `xrlenv/node/image_cache.py`.

---

## Quick start — operator recipes

If you have a long-running `xrlenv up` and just want to get a
benchmark's image set onto your cluster, the recipes below are the
suggested practice. Each links into the reference further down if you
want the mechanics.

If you only read one thing, it's the [flag cheat
sheet](#cheat-sheet-pick-the-right-apply-flag) at the end of this
section.

### Recipe: prefetch a benchmark's image set (the common case)

> **SWE-bench Verified:** the examples below use the **canonical** benchmark-local
> generator `xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen`, which gates
> `--all` on exact membership against the vendored 500-ID manifest. A legacy
> `xrlenv_plugins.images_build.swebench_verified.build_plan_gen` still exists (same manifest
> gate now) but is deprecated for new work.

```bash
# 1. Generate the plan from a benchmark generator (one-off).
.venv/bin/python -m xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen \
    --all --max-workers 8 --output build_plan.yaml

# 2. Apply against the live cluster.
xrlenv build apply --plan build_plan.yaml --connect-host 127.0.0.1
```

`--connect-host` dispatches to the running control plane. Without it,
the CLI tries the local-only path and refuses if it sees connected
nodes (see [Local apply vs cluster apply](#local-apply-vs-cluster-apply)).
For other benchmarks see [Per-benchmark plan
generators](#per-benchmark-plan-generators).

### Recipe: build a large plan in bulk (apply, then iterate)

When the plan is bigger than a single FFD pass can fit against the
cluster's residual disk — common for plans of hundreds of images on
small clusters — apply once, then iterate `--fill-missing` until the
deferred count converges:

```bash
xrlenv build apply --plan plan.yaml --connect-host <admin>
xrlenv build apply --plan plan.yaml --fill-missing --connect-host <admin>
xrlenv build apply --plan plan.yaml --fill-missing --connect-host <admin>
# ... repeat until deferred drops to 0 or plateaus over 2-3 passes.
```

Layer-share dedup compounds across passes, so FFD's per-image
reservation shrinks toward reality on each iteration. See [In-bulk
build recipe](#in-bulk-build-recipe-iterate---fill-missing-until-converged)
for the stop-condition heuristic and two worked examples (500-instance
SWE-bench Verified, 89-task terminal-bench-2).

### Recipe: retry just the failed/missing entries

A plan ended `partial_failure` (one image's pull was flaky), or
eviction dropped some cached images and you want them back:

```bash
xrlenv build apply --plan plan.yaml --fill-missing --connect-host <admin>
```

Works against any non-`in_flight` terminal status (`completed`,
`partial_failure`, `cancelled`, `superseded`). Only entries that no
connected node has are re-dispatched; everything else's assignment
row is re-anchored to whichever node has it. See [Targeted retry:
`--fill-missing`](#targeted-retry---fill-missing).

### Recipe: cancel an in-flight plan and retry

Discovered a real problem mid-build (bad registry credentials, wrong
Dockerfile) — cancel, fix, re-apply with the same plan file:

```bash
xrlenv build cancel --plan <id-or-prefix> --connect-host <admin>
# ... fix the underlying problem ...
xrlenv build apply --plan plan.yaml --connect-host <admin>
```

Cancel is sticky (a late-racing finalizer can't overwrite it back to
`completed`). Re-applying after cancel uses the same `plan_id` — the
plan file doesn't need to change. See [Cancelling an in-flight
plan](#cancelling-an-in-flight-plan).

### Recipe: calibrate size hints after a first build

Once the cluster has materialized a plan, replace heuristic /
registry-probed sizes with measured layer-share-aware unique-disk
values from the live cache:

```bash
xrlenv build calibrate \
    --plan plan.yaml \
    --output plan.calibrated.yaml \
    --connect-host 127.0.0.1
diff plan.yaml plan.calibrated.yaml          # eyeball the change
mv plan.calibrated.yaml plan.yaml && git commit ...
```

The calibrated YAML has a **different `plan_id`** — same image set,
different placement intent, so the next apply is a fresh dispatch
(fast on a warm cluster, but every entry runs). See [Calibrating size
hints from the cluster](#calibrating-size-hints-from-the-cluster)
for the per-image-cost arithmetic and the conditional determinism
caveats.

### Cheat sheet: pick the right `apply` flag

| Your situation | Flag | Why |
|---|---|---|
| Day-to-day re-apply on a warm cluster | *(no flag)* | Idempotent; no-op on a `completed` plan. |
| One image failed transiently; retry just it | `--fill-missing` | Re-dispatches only entries no node has. Cheapest retry. |
| Cluster eviction dropped some cached images | `--fill-missing` | Detects what's missing and refills; leaves the rest. |
| `state.db` reset but images still on disk | `--force` | Re-issues every entry as a fast cache hit. |
| Dockerfile bumped, same `image_ref`, every node should rebuild | `--force` | Bypasses idempotency, dispatches everything. |
| Warm re-dispatch + skip rebuilds for entries already tagged locally | `--skip-if-present` | Node short-circuits without cloning/building. |
| Plan is bigger than the cluster's image-cache budget | *(no flag)* | Default opportunistic FFD; overflow becomes lazy-pull-on-acquire. |
| Plan MUST fit upfront or fail | `--eager` | Strict-mode FFD; rejects with `InsufficientCapacity` if it doesn't. |
| Cancel an in-flight plan | `xrlenv build cancel --connect-host` | Sticky cancel; interrupts running builds on each node. |
| Large cluster: saturate idle nodes during apply | `--concurrency N` | Sets the coordinator fan-out for this invocation; see [Coordinator fan-out](#coordinator-fan-out---concurrency-n). |

Mutually exclusive: `--force`, `--fill-missing`, and `--eager` reject
the combination at argparse. `--skip-if-present` is compatible with
`--fill-missing`; `--force` beats `--skip-if-present` if both are
passed. `--concurrency` is independent of all of them.

---

## Known image sets

### The simple case: a flat refs file

When you just have a list of image refs, one per line, and want them
distributed across the cluster:

```bash
xrlenv images plan \
  --refs path/to/your-image-refs.txt \
  --eager-prefetch
```

`refs.txt` may include byte-size hints to drive packing:

```text
swebench/sweb.eval.x86_64.django__django-11099:latest    1430000000
swebench/sweb.eval.x86_64.sympy__sympy-13615:latest      1280000000
```

When size is unknown, the planner probes image size through node
image reports and Docker metadata as images materialize.

This is the right flow for ad-hoc sweeps. The richer flow below adds
per-image rebuild metadata, pinning, and priority.

### The structured case: `build-plan.yaml`

For benchmark catalogs and anything that needs to live in source
control, the canonical shape is `build-plan.yaml`:

```yaml
version: 1
replication: 1
budget:
  reserved_runtime_gb: 30
  buffer_gb: 10
entries:
  - image_ref: alexgshaw/fix-git:20251031
    context_source: { type: registry }
    placement:
      preferred_home_count: 1
      size_hint_bytes: 158051230
      size_hint_source: registry-probe
    pinned: false
    priority: 0

  - image_ref: xrlenv-seta-env/0:main
    context_source:
      type: git
      repo: https://github.com/camel-ai/seta-env
      ref: main
      subdir: Harbor-Dataset/0/environment
      dockerfile: Dockerfile
    placement:
      preferred_home_count: 1
      size_hint_bytes: 1500000000
      size_hint_source: heuristic

  - image_ref: my-org/private-task:v3
    context_source:
      type: tarball
      path: ./contexts/private-task.tar.gz
      dockerfile: Dockerfile
    placement:
      preferred_home_count: 2
      size_hint_bytes: 800000000
      size_hint_source: heuristic
    pinned: true
```

Each entry's fields:

| Field | Meaning |
|---|---|
| `image_ref` | The final tag the cluster materializes. Matches what consumers pass to `acquire_container(image=...)`. |
| `context_source.type` | `registry` (pull, live), `git` (clone + build from a Dockerfile), `tarball` (operator-shipped build context), or `local` (build in-place from a host-local directory on shared FSx — see below). |
| `placement.preferred_home_count` | Soft preference for how many nodes hold a copy. |
| `placement.size_hint_bytes` | Feeds the FFD bin packer. |
| `placement.size_hint_source` | `registry-probe` (accurate, from the registry manifest API), `cluster-reported` (post-build authoritative), or `heuristic` (the bin packer adds a safety margin). |
| `pinned` | Pinned entries skip eviction; see {doc}`cache_eviction`. |
| `priority` | Higher = built first when the plan exceeds the cluster budget. |
| `labels` | Extra Docker labels. The coordinator reserves `xrlenv.image.rebuild-cost` for itself. |

#### `type: local` — build from a shared-filesystem directory

`local` builds a Docker image in-place from a directory that already exists on the build host. The Dockerfile and everything it `COPY`s live under `path`; no cloning or tarball shipping happens. This is the right source type for a harbor task cache on shared FSx, where the unpacked context is already a directory tree.

Required fields:

| Field | Meaning |
|---|---|
| `path` | Absolute path to the directory that is the Docker build context. The Dockerfile lives here (or at the `dockerfile` sub-path). Must resolve identically on every build node — that guarantee is what `shared_fs` asserts. |
| `shared_fs` | REQUIRED. Names the cluster-shared filesystem topology (e.g. `hyperpod`) that guarantees `path` is mounted on every build node. The field is both a machine-readable assertion and operator-visible documentation of why a bare local path is safe. Omitting it or leaving it empty raises `ManifestInvalid` at schema-load time. |
| `dockerfile` | Dockerfile name within `path`. Defaults to `Dockerfile`. |

**`xrlenv build apply` rejects `local` entries.** The cluster apply path ships build contexts to nodes that may not share the path. Hitting this raises:

```
ManifestInvalid: build plan rejected: 'local' context sources are
build-host-only and aren't supported on the cluster build-apply path
(it ships sources to nodes that may not share the path). Build them with
scripts/build_and_push_images.py on a shared-fs build host (directly or
Slurm-sharded), then apply a registry-source plan. Offending entries: ...
```

The intended flow for `local`-source entries is:
1. Run `scripts/build_and_push_images.py` on the shared-fs build host (directly or Slurm-sharded). That script reads `local` entries from the plan, builds and pushes each image to the cluster registry.
2. Apply a registry-source plan (re-emit with `type: registry`) against the live cluster to distribute the pushed images.

For a worked example of a `local`-source plan entry see {doc}`/deploy/multi_node_deployment/private_registry` (WebArena substrate).

The schema is strict (`extra="forbid"`, frozen models): a missing
`size_hint_bytes` or unknown `context_source.type` raises
`ManifestInvalid` at load time, before the coordinator dispatches
anything.

You don't usually hand-write this file. Per-benchmark generators emit
it for you.

### Per-benchmark plan generators

Generators under `xrlenv_plugins/images_build/` translate a
benchmark's task catalog into the generic plan schema. They live
alongside the benchmark adapters but are decoupled from rollout-time
code — their only job is to translate a benchmark's task catalog
into the generic plan schema.

| Generator | Module | Default context source |
|---|---|---|
| `terminal-bench-2` | `xrlenv_plugins.images_build.terminal_bench_2.build_plan_gen` | `registry` (`alexgshaw/<task>:20251031`) |
| `swebench-verified` | `xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen` | `registry` (`swebench/sweb.eval.x86_64.<key>:latest`) |
| `seta-env` | `xrlenv_plugins.benchmarks.seta.build_plan_gen` | `git` (camel-ai/seta-env, per-task `Harbor-Dataset/<id>/environment`) |

Sample invocations:

```bash
# terminal-bench-2: 8-task smoke set, sizes probed live from Docker Hub
.venv/bin/python -m xrlenv_plugins.images_build.terminal_bench_2.build_plan_gen \
    --smoke --output build_plan.yaml

# swebench-verified: full 500-instance sweep. --max-workers 8 runs
# Docker Hub probes 8-way concurrent; cuts a 500-entry --all sweep
# from ~5 min serial to under a minute (authenticated). Output order
# is preserved regardless of pool size, so the plan_id is stable.
.venv/bin/python -m xrlenv_plugins.benchmarks.swebench_verified.build_plan_gen \
    --all --max-workers 8 --output build_plan.yaml

# seta-env: pull the canonical Harbor-Dataset task list off GitHub
.venv/bin/python -m xrlenv_plugins.benchmarks.seta.build_plan_gen \
    --remote --output build_plan.yaml
```

Each generator commits a canonical `build_plan.yaml` next to itself
— the result of running with the default selection (`--smoke` for
tb2 and swebench, `--starter` for seta-env). Operators can use the
committed file directly or regenerate.

#### Docker Hub probing and rate limits

The registry-source generators (`terminal-bench-2`, `swebench-verified`)
probe Docker Hub's v2 manifest API once per entry at generation time
to populate `placement.size_hint_bytes` with the real compressed
layer size (recorded as `size_hint_source: registry-probe`). Docker
Hub rate-limits **unauthenticated** requests at ~100 / 6 h per source
IP. A full SWE-bench Verified `--all` sweep is 500 entries, which
blows through the unauth budget around entry ~100; every subsequent
probe falls back to the generator's conservative default
(`DEFAULT_SIZE_HINT_BYTES`, currently 2.5 GiB for swebench and
1.5 GiB for tb2). That default is much larger than the real
compressed size, so the FFD bin-packer reserves disk it doesn't
need and may reject an otherwise-fittable plan at apply time with
`InsufficientCapacity`.

To lift the limit to your account's tier cap, set a Docker Hub
[Personal Access Token](https://docs.docker.com/security/for-developers/access-tokens/)
in two env vars (the generators auto-load `.env` via the
`xrlenv` import-time hook documented in
{doc}`/getting_started/installation`):

```text
DOCKERHUB_USER=your-hub-username
DOCKERHUB_TOKEN=dckr_pat_xxxxxxxxxxxxxxxxx
```

A Pro/Team/Business account on the authenticated path generally
handles a 500-entry probe burst comfortably. When the vars are
present, the generator exchanges them for a JWT once and attaches
`Authorization: Bearer <jwt>` to every probe.

**What the generator prints**

The generator surfaces probe state on stderr so an over-reserved
plan never sneaks past you:

1. **Banner before the loop.** Names whether the run is
   authenticated (and as which user) or unauthenticated (and which
   env vars to set).
2. **Loud first-failure warning.** The first probe failure prints
   the HTTP status, a response-body snippet, and — when
   unauthenticated — the env-var hint. Subsequent failures stay
   quiet to avoid flooding the log, but they're counted.
3. **End-of-run summary.** Reports `N/total succeeded`, the
   fallback count, and how much over-reservation the fallback
   created in FFD-budget terms (e.g. `300 fell back → 750 GiB of
   over-reservation`). When unauthenticated, names the env vars to
   set + re-run.

Sample output for a 500-entry unauth run that hit the rate limit
mid-way:

```text
Docker Hub probes: unauthenticated — rate-limited at ~100 / 6h per
source IP. Set $DOCKERHUB_USER + $DOCKERHUB_TOKEN (a Docker Hub
Personal Access Token) to lift the limit to your account's tier cap.

WARN: Docker Hub probe failed for swebench/sweb.eval.x86_64.django_1776_django-12345:latest:
      HTTP 429 (unauthenticated; ~100 / 6h limit. Set $DOCKERHUB_USER + $DOCKERHUB_TOKEN to lift it.)
      body: '{"detail":"too many requests"}'
      Subsequent failures will be counted silently; see the end-of-run summary for the total.

[...rest of the probe loop...]

WARN: Docker Hub probes: 102/500 succeeded, 398 fell back to the 2.5 GiB heuristic
      (995 GiB of over-reservation in FFD bin-packing).
      Cause: likely Docker Hub rate-limit. Set $DOCKERHUB_USER + $DOCKERHUB_TOKEN
      (a Docker Hub PAT) and re-run to get accurate sizes for all entries.
```

Once you see that summary, the right next step is almost always:
add the env vars, re-run the generator, and re-apply the freshly-
sized plan. The alternative is `xrlenv build calibrate` after a
first cluster build (documented in "Calibrating size hints from
the cluster" further down this page), but it only knows about
images already materialized on at least one node, so the first
apply still has to fit on heuristic sizes.

The `--no-probe` flag skips Docker Hub entirely and uses the
conservative default for every entry. Use it when you want a
plan-shape preview without spending probe budget; expect FFD to
significantly over-reserve.

**Speed: `--max-workers N`** runs `N` probes concurrently on
swebench-verified. Each probe is network-bound, so threads work
well and don't fight the GIL. A serial 500-entry `--all` run takes
~5 minutes (≈500ms/probe wall-clock); `--max-workers 8` brings it
under a minute when authenticated. Output entry order is preserved
regardless of pool size, so the content-addressed `plan_id` stays
stable across re-runs with different worker counts. Default is `1`
(serial) so existing scripts keep their previous behavior.

## How placement is decided: FFD packing

Both the flat-refs flow and the structured-plan flow share the same
packing algorithm. First-fit decreasing:

1. Read free disk budget for each node.
2. Sort images largest first by known or hinted size.
3. Place each image on the first node with enough remaining budget.
4. Persist the chosen node as the image's **preferred home**.
5. Images that can't fit immediately are recorded as deferred rows
   with a preferred home, so first use still routes toward the
   intended node when possible.

The preferred home is a soft scheduler signal, not a hard placement
lock. If the preferred node is full or violates fairness constraints,
the scheduler can choose another feasible node.

### Placement is static; dispatch is paced (disk-aware dispatch)

FFD answers a **spatial** question once, at apply time: *given a
snapshot of each node's free disk and the images' hinted sizes, which
node should hold each image?* It does **not** track disk during the
hours-long materialization that follows. Three things make "FFD already
considered disk" necessary but not sufficient:

- **It's a one-shot snapshot.** FFD reads free disk *at apply time* and
  assigns the whole plan. As images actually pull/build, the disk fills
  with no feedback loop back into FFD.
- **Size hints are soft and under-count real bytes.** FFD packs by
  `size_hint_bytes` — estimates on a `partial_calibrated` plan, and the
  *logical* image size, which on the **containerd image store** is roughly
  **half** the real on-disk cost (the store keeps the compressed content
  blobs *and* the unpacked overlay snapshots). So FFD can think a set
  "fits" while it actually needs ~2x the disk.
- **Concurrency creates a transient peak.** `--concurrency N` materializes
  up to N images at once; the peak in-flight footprint (intermediate
  layers + final images, before eviction reclaims) can exceed what FFD's
  "total fits" assumed.

So the coordinator adds a **temporal** safeguard on top of FFD: before
materializing the next image on a node, it waits until that node reports
enough free disk (`max planned image size x XRLENV_BUILD_DISK_HEADROOM_FACTOR`,
default 3x), polling the node's heartbeat `disk_state`. If a node is full,
its dispatchers park while node-side eviction reclaims, then resume — so
the build self-paces to the eviction rate instead of driving the node to
ENOSPC (or forcing eviction of freshly-built images). The wait is bounded
(`XRLENV_BUILD_DISK_WAIT_TIMEOUT_S`, default 300 s); on timeout it proceeds
and lets eviction cope rather than wedging the plan. The gate is active
only on a distributed runtime with live node heartbeats; a fully-calibrated
plan keeps it dormant because FFD's spatial decision already fits.

| Knob | Meaning | Default |
|---|---|---|
| `XRLENV_BUILD_DISK_HEADROOM_FACTOR` | Multiple of the largest planned image kept free before dispatching to a node | `3.0` |
| `XRLENV_BUILD_DISK_POLL_S` | How often a parked dispatcher re-checks the node's free disk | `5.0` |
| `XRLENV_BUILD_DISK_WAIT_TIMEOUT_S` | Max wait before proceeding anyway (eviction will reclaim) | `300.0` |

**FFD on-disk size multipliers.** FFD packs by `size_hint_bytes`, but those hints
are the *compressed* registry sizes — the actual on-disk footprint (unpacked
overlay snapshot plus the retained content blob on the containerd image store)
is roughly 2.7–4x that figure. Packing against the raw compressed size causes
FFD to think images fit when they don't, producing spurious
`InsufficientCapacity` rejections on plans that should fit, or silent ENOSPC
during the build. Two multipliers applied inside FFD correct for this:

| Knob | Applied to | Default | Operator symptom if wrong |
|---|---|---|---|
| `XRLENV_PACK_ONDISK_MULTIPLIER` | `registry-probe` and `heuristic` hints (compressed sizes) | `3.0` | Too low → ENOSPC / deferred overflow despite "fitting"; too high → FFD over-reserves and rejects a fittable plan. |
| `XRLENV_PACK_ONDISK_MULTIPLIER_CLUSTER_REPORTED` | `cluster-reported` hints (already measured on-disk after `xrlenv build calibrate`) | `1.0` | Raise above 1.0 only if calibrated values are consistently under-counting real disk usage on unusual storage backends. |

`cluster-reported` hints are measured post-materialization (the calibrate
command reads actual on-disk bytes from Docker's `GET /system/df`), so they
already reflect real storage cost. The pass-through multiplier of 1.0 avoids
double-counting. If you see unexpected FFD rejections after calibrating, check
that the plan file's `size_hint_source` fields are actually `cluster-reported`
and not still `heuristic` — the multiplier selection is source-tag-driven.

### Eager prefetch vs lazy materialization

With `--eager-prefetch`, the planner dispatches pulls/builds
immediately so nodes warm up before any rollout asks for an image.
Use this when first-use pull latency would dominate a sweep.

Without eager prefetch, the plan still matters: assignments are
registered, and future scheduling uses preferred-home routing while
no node has the image yet. Pulls happen on first acquire.

## Applying a plan

`xrlenv build apply --plan <yaml>` is what you run to push a plan
onto your cluster:

1. The control plane records the plan, FFD-bin-packs entries against
   each connected node's free disk **measured at apply time**, and
   dispatches `ensure_present(image_ref)` per chosen node.
2. Each node either pulls from the registry (cold cache) or returns
   immediately (warm cache).
3. The plan flips to `completed` once every assignment reaches
   `done`.

What surfaces:

- Admin `/builds` shows the plan with `done / total` (e.g. `8 / 8`)
  and the apply timestamp. Click through for per-assignment
  placement.
- `docker images` on each node shows that node's slice of the refs.
  With `preferred_home_count: 1` (default) every ref lands on
  exactly one node and the cluster's union covers the whole plan.

### Coordinator fan-out: `--concurrency N`

By default, the control-plane coordinator caps its cluster-wide
in-flight image dispatches at the value of the `XRLENV_BUILD_CONCURRENCY`
environment variable (default 32, read once at module import time on the
`xrlenv up` process). Historically the only way to change it was to set
the env var and restart the control plane — which confused operators who
set it on the `xrlenv build apply` client invocation instead.

`--concurrency N` sets the coordinator fan-out **for that single
invocation** with no control-plane restart. It is forwarded over
`POST /api/build/apply` (request body field `concurrency`) to
`coordinator.apply(concurrency=N)`. Omitting it falls back to the
`XRLENV_BUILD_CONCURRENCY` default. The value is validated as a positive
integer (≥ 1) and rejected client-side before any network traffic.

**How to size it.** The three concurrency knobs are orthogonal:

| Knob | Controls | Where it lives |
|---|---|---|
| `XRLENV_BUILD_CONCURRENCY` / `--concurrency N` | Cluster-wide in-flight dispatches from the coordinator | Control plane |
| Adaptive pull limiter (`pull_concurrency` floor → `pull_concurrency_ceiling` ceiling) | Concurrent pulls per node (both rollout-acquire and prefetch), governed by AIMD | Node agent |
| `pull_concurrency` / `XRLENV_PULL_CONCURRENCY` | AIMD floor — minimum pull slots even on a busy node | Node agent |

To saturate idle nodes during a build apply, set the coordinator fan-out
to roughly `num_nodes × pull_concurrency_ceiling`. The per-node AIMD
ceiling is the maximum pull concurrency when the node is idle; the floor
(`pull_concurrency`) protects live agents on a busy node; the AIMD loop
transitions between the two automatically. Note that `--concurrency` bounds
builds and pulls cluster-wide, while the per-node AIMD limiter governs
pulls only — git/tarball builds are not gated by it.

**Worked example — 3-node cluster:**

```bash
# Each node has XRLENV_PULL_CONCURRENCY_CEILING=64 (the default).
# 3 nodes × 64 = 192 → pass --concurrency 192 to keep all three nodes
# busy pulling in parallel rather than serializing behind a fan-out of 32.
xrlenv build apply \
    --plan build_plan.yaml \
    --connect-host 127.0.0.1 \
    --concurrency 192
```

Without `--concurrency 192`, the coordinator would issue at most 32
dispatches at once; most of the 192 available per-node AIMD slots would
sit idle. With it, all three nodes ramp up toward their ceiling and pull
at full capacity for as long as the plan has work queued.

```{note}
Setting `--concurrency` far above `num_nodes × pull_concurrency_ceiling`
wastes nothing (the per-node AIMD semaphore is the real ceiling) but also
provides no benefit. A value at or slightly above the product is the
practical sweet spot.
```

### Local apply vs cluster apply

`xrlenv build apply` has two execution paths and **the right one
depends on whether you have a control plane running**:

| Your situation | What to run | What it does |
|---|---|---|
| No `xrlenv up` running; you want a one-host LocalRuntime to apply against | `xrlenv build apply --plan <yaml>` | Spins up a transient in-process LocalRuntime, applies the plan, exits. Refuses to run if it sees connected nodes in `state.db` (would mark them `lost`). |
| `xrlenv up` is running with nodes attached (the typical operator setup) | `xrlenv build apply --plan <yaml> --connect-host <admin-host>` | POSTs the plan to the admin server's `/api/build/apply`, polls `/api/build/plans/<plan_id>` until terminal, prints results. The dispatch happens on the live cluster. |

If you forget `--connect-host` while `xrlenv up` is running, the
CLI fails fast with:

```
error: refusing to run local-only ``xrlenv build apply``
       while another control plane appears active.
       N node(s) heartbeated within the last 30s: ...
```

That's the guard doing its job — running the local path against
a state.db that's fronting a live cluster would corrupt the
operator's view of attached nodes. Add `--connect-host` (and
optionally `--connect-port`, default 8080; `--operator-token`,
default from `$XRLENV_OPERATOR_TOKEN` or
`~/.xrlenv/secrets/operator.token`) to dispatch to the cluster
instead.

```bash
# Cluster apply (the common case when xrlenv up is running):
xrlenv build apply \
    --plan xrlenv_plugins/benchmarks/seta/build_plan.yaml \
    --connect-host 127.0.0.1
```

For most operators with a long-running `xrlenv up`,
`--connect-host` is the path you'll always use. The local-only
form is for laptop-only experiments and the test smokes.

### Re-applying the same plan

| Command | What the cluster does | Wall-clock |
|---|---|---|
| `build apply --plan ...` (no force) on a `completed` plan | Coordinator short-circuits; returns `no_op_already_completed`. No dispatch, no state writes, no node-side work. | sub-second |
| `build apply --plan ... --force` on a `completed` plan, warm cache | Re-bin-packs, replaces assignment rows, re-issues `ensure_present` per (node, image). Each call is a **fast cache hit** — no registry pull. | seconds |
| First-ever apply, cold cache | Real registry pulls for everything missing. | dominated by pull bandwidth |

So `--force` on a warm cache is cheap. The only way re-applying
triggers real pull traffic is if the cache evicted something between
runs.

When you'd reach for `--force`:

- After resetting `state.db` while images remain cached on the
  daemons.
- After a Dockerfile bump for `git`/`tarball` plans where the
  `image_ref` didn't change but every node should re-build.
- For "make the cluster match this plan exactly" semantics in CI.

For day-to-day operation, plain `xrlenv build apply --plan ...` is
what you want — idempotent, cheap, the right thing on warm
clusters.

### Warm-cluster fast path: `--skip-if-present`

`xrlenv build apply --plan ... --skip-if-present` makes the
node-side source builder short-circuit when the image is already
tagged on the chosen node. The node returns `ok` without cloning,
untarring, or invoking `docker build`. Use this in two operator
scenarios where re-dispatching every entry from scratch is
wasteful:

| Scenario | Without `--skip-if-present` | With `--skip-if-present` |
|---|---|---|
| Calibrated plan re-apply (new plan_id, same image set) | Every entry's source builder runs end-to-end. For seta-env-style 16-image plans, ~90s wall-clock (clone + cache-validated docker build per entry). | Each entry's node confirms the image is local + returns ok. ~1-5s total wall-clock — one wire RTT per entry, no clone, no build. |
| Partial-failure retry (most entries `done`, some `failed`) | Re-apply purges old assignments + dispatches every entry; the previously-`done` ones rebuild (cached, but still slow). | Previously-`done` entries short-circuit; only the previously-`failed` ones go through the full build pipeline. |

Behavior matrix:

- **Registry-source entries**: already short-circuit via the
  node's `ensure_present` cache. The flag has no effect (no
  behavior change either way).
- **Git/tarball entries, image present locally**: builder returns
  `ok` immediately; source-spec persistence still fires so a
  later `acquire_container`-after-eviction has the recipe.
- **Git/tarball entries, image NOT present locally**: builder
  runs the full clone+build pipeline. The flag is a fast-path,
  not a no-op — missing images still get built.
- **`--force` overrides `--skip-if-present`**: if you pass both,
  `--force` wins. Forced rebuilds always dispatch, regardless of
  local presence. The intent of `--force` is "rebuild this
  even if cached" and that takes precedence over "skip if
  present."

Default is **off** (rebuild even if present) so existing operator
scripts don't silently change semantics. `--skip-if-present` is
a discoverable opt-in.

#### Why not automatic per-entry diff?

An operator could imagine a different shape: when re-applying a
changed `build-plan.yaml` (size hints edited, an entry added or
removed), the coordinator computes a per-entry diff against the
prior plan_id's assignments and only re-dispatches the actually-
changed entries. No `--skip-if-present` flag needed; the
behavior just falls out of "what changed."

This is **deferred-by-design** in favor of the explicit
`--skip-if-present` flag. The functional outcome is the same
(unchanged entries no-op fast); the design tradeoffs differ:

- "What counts as changed?" is policy: size hint? source spec?
  labels? `preferred_home_count`? Each is a different answer
  for different operator workflows. Baking one set into core
  pushes some operators toward forks.
- "What baseline to compare against?" Latest applied plan?
  Latest *completed* plan? Partial-failure plans? More policy.
- "How to handle concurrent applies racing the diff?" More
  edge cases.

`--skip-if-present` sidesteps all of this: the operator
explicitly says "skip what's present"; the cluster does one
wire round-trip per entry to check local presence; sub-second
per entry. Simpler semantics, no policy ambiguity, and the
operator stays in the driver's seat.

If your plans have hundreds of entries and the sub-second-per-
entry RTT becomes material, surface it — that's the case
that would justify revisiting automatic delta dispatch. Tracked
in `notes/phase-2-todo.md` § "Deferred-by-design".

### Targeted retry: `--fill-missing`

`xrlenv build apply --plan ... --fill-missing --connect-host <admin>`
brings the cluster into the plan's intended state by dispatching
**only** entries that aren't currently cached on any connected
node. Entries already present anywhere skip dispatch entirely;
their assignment rows get re-anchored to the node that has them.

When to reach for it (vs `--force` or a plain re-apply):

| Scenario | Best flag | Why |
|---|---|---|
| One transient pull failed (apply ended `partial_failure`); want to retry just that one | `--fill-missing` | Re-dispatches only the failed entry; the other N-1 cache-hits don't happen. |
| Cluster eviction removed some cached images; need to refill | `--fill-missing` | Detects which entries dropped out of cache and pulls them back; leaves the rest alone. |
| Images drifted to different nodes than the plan_id's assignment rows say (manual `docker pull`, prior apply's FFD picked differently) | `--fill-missing` | Re-anchors rows to current reality so the image-affinity scheduler routes rollouts to the right node. |
| Operator wants a full clean rebuild (e.g. after an upstream Dockerfile bump where image_ref is unchanged) | `--force` | Purges + re-dispatches every entry. |
| Operator wants to assert "all N images must fit upfront" | `--eager` | Strict-mode FFD; `--fill-missing`'s opportunistic FFD doesn't make that assertion. |

Implementation note for operators:

- `--fill-missing` works regardless of the plan's existing terminal
  status (`completed`, `partial_failure`, `cancelled`, `superseded`).
  Only `in_flight` rejects (concurrency control).
- Mutually exclusive with `--force` and `--eager` (the CLI rejects the
  combination at argparse).
- Requires `--connect-host` — the cluster inventory probe needs a live
  control plane with connected node-agents; the local-only LocalRuntime
  path doesn't carry an inventory provider (single-host plans don't
  need this primitive).
- An entry deferred under `--fill-missing` (missing image, no node has
  budget to host it after opportunistic FFD) is persisted as
  `status=registered`; lazy-pulled at acquire time. Same shape as the
  default opportunistic apply.

#### How "missing" is decided

The coordinator queries every connected node's `report_images()` once
to build `{image_ref: {nodes_where_present}}`. The control plane waits up to
`XRLENV_REPORT_IMAGES_TIMEOUT_S` (default **60 s**; set in
`xrlenv/control/grpc_endpoint.py`) for each node's reply. On a cluster with
a large image catalog or slow nodes, raise this value before running
`build apply --fill-missing` or `build calibrate` — a node that times out
is treated as having no images, which causes `--fill-missing` to re-dispatch
everything the timed-out node holds. For each plan entry:

- **Image is on ≥ 1 node** → mark as already-present. Assignment row
  anchored at one of those nodes (first alphabetical for
  determinism), `status=done`. No work dispatched.
- **Image is on NO node** → FFD-place against current free disk,
  emit `status=pending`, dispatch via `ensure_present_fn` (registry)
  or `build_image_fn` (git/tarball).

The "any node has it" criterion is intentional: the operator's intent
is "the cluster collectively hosts this plan." Whether the image
lives on n1 or n2 doesn't matter as long as it's somewhere. This is
what makes `--fill-missing` cheap — it sidesteps the "did FFD lay
out this image on the right node?" question entirely.

#### When the operator-mental-model and the row state disagree

`--fill-missing` always **delete-and-replaces** the prior assignment
rows for this plan_id. Stale rows pointing at nodes that don't have
the image anymore (eviction, manual `docker rmi`, FFD churn) get
cleared so the image-affinity scheduler isn't routing rollouts to
nodes that can't serve them. This is the "rows reflect reality"
contract — if you want strict adherence to the plan_id's previous
FFD layout, re-apply with `--force` instead.

### In-bulk build recipe: iterate `--fill-missing` until converged

> The command pattern is in [Recipe: build a large plan in
> bulk](#recipe-build-a-large-plan-in-bulk-apply-then-iterate). This
> section explains *why* the iteration works and how to read the
> output.

When a plan is significantly larger than a single FFD pass can fit
against the cluster's residual disk budget — common for plans on the
order of hundreds of images on small clusters — the recommended
operator workflow is to apply once, then **iterate `--fill-missing`
until the deferred count converges**. The pattern exploits two
properties of the system:

1. **Layer-share dedup compounds across iterations.** FFD reserves
   each entry's full `size_hint_bytes` as if it were brand-new disk.
   Real on-disk cost is `size_hint - shared_with_already_cached`. As
   the cluster accumulates images that share a common base, the
   marginal cost of each new entry drops well below its reservation.
   FFD's pessimism is corrected by reality on every subsequent pass.
2. **Each `--fill-missing` re-anchors what's already cached and only
   FFD-evaluates the missing subset.** The missing set shrinks each
   pass, so FFD has more budget headroom per remaining candidate.

The terminal cue is in the poller line: when two consecutive
`--fill-missing` runs report the same `deferred=N` (or `N=0`), you've
hit either full convergence or the cluster's actual capacity ceiling.

**Stop-condition heuristic**

| Pattern across 2-3 consecutive `--fill-missing` runs | Diagnosis | Action |
|---|---|---|
| `deferred` drops monotonically (e.g. 264 → 142 → 44 → 0) | Cluster still converging via layer-dedup compounding. | Run again. |
| `deferred` plateaus at the same value 2+ times | Capacity ceiling reached. | Stop. Either accept the deferred as lazy-pull-on-acquire (the design's intended steady-state for opportunistic mode), or add a node / larger disk. |
| `failed` count keeps appearing on **different** entries each run | Transient registry / network flakes. | Pause 1-2 min, run again. Usually self-heals. |
| `failed` count is **stuck on the same entries** | Genuine pull issue (image-not-in-registry, auth wrong, etc.). | Inspect the specific entry's error: `xrlenv build status --plan <id>`. |

**Watching free disk**

The admin's `/nodes` page reports `free_disk_bytes` per node. Two
thresholds:

- **Above `budget.reserved_runtime_gb + buffer_gb`** (typically 40 GB):
  healthy. FFD has room; no eviction.
- **Approaching the reservation floor**: eviction is firing during
  dispatch. The cluster is at-capacity; pulls and evictions trade off.
  Plan still completes, but each new pull may evict a different
  previously-cached image. A subsequent `--fill-missing` will identify
  the evicted entries as newly-missing and re-pull them — at-capacity
  iteration becomes a rotating working set rather than a strictly
  monotonic convergence.

When at-capacity, the canonical end-state is "the most recently
needed working set is cached, plus deferred rows for everything
else; rollouts on deferred refs lazy-pull at acquire time with LRU
rotation." That's the design intent of opportunistic mode + the
runtime path's `ensure_present`.

**Worked examples**

Two real cluster convergences observed during phase-1 validation —
both fit their entire plan onto **two nodes with ~60 GB free disk
each (~120 GB cluster cache budget after reservations)**:

- **swebench-verified, 500-instance Verified split.** Each image is
  ~500-800 MB compressed and shares a fat Python+swebench base layer
  with every sibling. Convergence trajectory across 4 successive
  applies (1 initial + 3 fill-missing): 235 done / 1 failed / 264
  deferred → 357 / 1 / 142 → 453 / 3 / 44 → 500 / 0 / 0. Each pass
  shrank the deferred set as Docker's layer dedup compounded — the
  first 235 pulls established the base layers; subsequent pulls only
  paid their unique-bytes cost (much less than the size_hint), so
  FFD's per-pass reservation arithmetic became progressively less
  pessimistic relative to reality.
- **terminal-bench-2, 89-task set.** Each task image shares a slim
  harbor-runtime base. After a first apply + one `xrlenv build
  calibrate` pass (to record unique sizes for already-cached
  entries), a single `--fill-missing` brought the cluster to
  89 / 0 / 0.

For plans of this size and shape, the convergence cost is on the
order of **a few minutes per fill-missing pass** (most entries
already cached → no work; only the missing subset's pulls run). For
smaller plans (single-digit deferreds), one pass typically suffices.

#### When iteration becomes operator pain (future)

The manual iteration loop is fine for the dozens-to-hundreds-of-
images plans we've validated. For order-of-thousands plans where
4-6 passes might be needed and the operator just wants a single
"converge" verb, a future slice can add `xrlenv build apply
--converge`: an admin-side loop that auto-runs `--fill-missing`
until the deferred count stabilizes or hits zero, returning only
when terminal. Worth shipping when the manual loop becomes a
recurring complaint, not before.

### Build-on-acquire after eviction

Source-built images (git or tarball) survive eviction without
operator action. After a successful build, the node persists the
source spec to a per-image_ref registry under
`<cache_root>/source-registry/<sha256(image_ref)>/` (the
`spec.json` plus, for tarballs, `content.bin` carrying the bytes).

When the image cache reaps a source-built image — typical case:
the LRU eviction loop fires under disk pressure and the image
hasn't been touched in a while — the next `acquire_container`
for that ref triggers `ensure_present`, which consults the
node-agent's `_lookup_image_producer`. The agent finds the spec
in the registry and re-runs the build through the same
`GitSourceBuilder` instance the original `BuildImageCommand`
used. Result: cluster operators see "image got evicted, rolled
out anyway, slight latency hiccup," not "image got evicted, my
plan_id never re-applied, every rollout fails until I notice."

The registry survives node restart: the builder lazy-loads it
from disk on first lookup, so a daemon that's been bounced for
maintenance still rebuilds source images on demand without
needing the operator to re-apply the build plan.

Tarball persistence cost: up to the per-tarball cap (default
100 MB) per image_ref, on disk under the cache root. Operators
who don't want this can lower the cap or `rm -rf
<cache_root>/source-registry/` to force re-shipping from the
operator on next eviction.

### Build-time grace window

A freshly-built image with no `acquire` touch yet sorts as
`recently_used` (not `cold`) for the duration of
`ImageCacheConfig.build_grace_window_s` (default 10 minutes).
Without this, a build that finishes seconds before the eviction
loop fires can be reaped immediately — forcing a rebuild on the
very next `acquire_container` even though it would have been
used in seconds.

The window expires the moment the image is actually acquired
(standard LRU semantics take over from there) OR after the
configured duration, whichever comes first. Tunable per-node via
`ImageCacheConfig` if 10 minutes is too short for a benchmark's
typical build-to-rollout lag.

### Calibrating size hints from the cluster

After a first cluster build, the operator can replace heuristic /
`registry-probe` size hints with measured values from the
materialized cache.

```{note}
**Registry-agnostic matching.** Calibrate normalizes the registry host
on both sides before matching. A plan entry with a bare `repo:tag`
(`image_ref: xrlenv-webarena-infinity/substrate:dev`) correctly matches
the registry-qualified tag a node holds after a pull
(`node-host:5011/xrlenv-webarena-infinity/substrate:dev`). Before
this fix, calibrate reported "0 measured" for such entries while the
admin `/images` page correctly listed the image.

**Known limitation: digest-pulled images.** When the
{doc}`registry freshness model <registry_freshness>` dispatches a
digest-pinned ref to a node (`host:5011/repo@sha256:…`), the node
holds that image without a local tag. Calibrate matches by tag, so a
digest-only image has no local tag to match against and remains
`unmeasured`. To calibrate such images, ensure the node also holds the
tag-pulled copy (e.g. via a `build apply` with a tag-shaped
`image_ref`, which results in the node holding the image under the
tag). For the channel-tag workflow, tag-pulled copies and digest-pulled
copies coexist; the tag-pulled copy is what calibrate measures.
```

```bash
xrlenv build calibrate \
    --plan build-plan.yaml \
    --output build-plan.calibrated.yaml \
    --connect-host 127.0.0.1
```

The admin walks each connected node's `report_images` snapshot
and writes the **layer-share-aware unique size** for every measured
entry — the incremental disk a node pays to cache one more image
when its base layers are already present from a sibling. The
calibrated YAML carries `placement.size_hint_bytes` =
`unique_size` and `placement.size_hint_source` = `cluster-reported`.
Unmeasured entries (no node has materialized them yet) keep their
operator-supplied hints.

**Why unique, not the full per-image size.** Docker's per-image
`Size` field sums every layer the image references — including the
shared base layer common across sibling images. For
swebench-verified that base is a multi-GB Python + dev-tools layer
that ~all 500 instance images reference. The FFD bin-packer
operates on per-image numbers; if those numbers include the shared
base layer, the cluster's projected disk consumption double-counts
the base K times for K dependent images. Net effect on a 2-node
cluster trying to host SWE-bench Verified: FFD thinks ~150-200
images saturate the disk; the actual on-disk cost is closer to
50-70% of that because shared layers dedup at the daemon level.
Calibrate's `unique = size - shared` (sourced from Docker's
`GET /system/df` API → `Images[].SharedSize`) is the metric that
matches reality once the cluster is warm.

A residual modeling error stays: the **first** image to land on a
fresh node still pays the full shared-layer cost
(`size_bytes`-equivalent). The unique-only accounting treats every
image as if its base is already there, which is true for the 2nd
through Nth image but optimistic for the 1st. A planned follow-on
(`BuildBudget.shared_base_reserve_bytes`, gated on operator
request) closes that gap by reserving one shared-base allocation
per node before FFD runs. Until then: the unique-only model is
roughly correct for plans larger than a few entries per node —
your typical sweep size.

When the node-side backend doesn't surface `SharedSize` (older
Docker daemons, the in-process LocalBackend, the in-memory test
backend), calibrate transparently falls back to the legacy
`size_bytes` value. Same behavior as the pre-2026-05 calibrate; the
hint is less accurate but still better than the operator's
heuristic.

The output goes to a separate file so the operator can `diff`
against the input + decide before promoting (typical flow:
calibrate → review → `mv plan.calibrated.yaml plan.yaml ; git
commit`). The flow is intentionally operator-driven (locked F5
in `notes/source-build-dispatch.md`): the calibrated sizes feed
FFD placement, which directly affects which nodes get which
images on the next apply, and auto-on-apply would silently
reshape the cluster between unrelated runs.

`--connect-host` is required; there's no useful local-only
fallback.

#### What "calibrated" actually means — and why the same plan can produce different numbers

The output of `xrlenv build calibrate` is a **snapshot of the per-
image incremental disk cost at the moment of probe**, against the
cluster's current full cache state. The number written into each
entry's `placement.size_hint_bytes` is `Size - SharedSize` as
reported by Docker's `GET /system/df` at probe time — where
`SharedSize` is the sum of bytes for layers that this image shares
with **any other image currently on the daemon**, not just the
plan's own siblings.

That last clause is the source of the apparent non-determinism that
operators sometimes hit: the same calibrate command, against the
same plan, on the same cluster, can write different
`size_hint_bytes` values on consecutive runs if other images came
or went on the daemons between the probes.

Concrete example: a task image whose layers include a Python base
shared with other benchmarks on the same node.

| Cluster cache state at probe time | This image's `SharedSize` | Calibrated `size_hint_bytes` (= Size − SharedSize) |
|---|---|---|
| Only this image's sibling set is cached; sibling images share only a small task-specific layer | small | close to full Size |
| Sibling set + a co-resident large benchmark whose images use the same Python base layer | large (Python base counts as shared) | much smaller |
| Same as above but the co-resident benchmark has churned (some pulled, some evicted) since the previous probe | varies | varies between probes |

**Determinism guarantee:** given a quiescent cluster — no pulls, no
evictions, no rollouts touching cache between calls — two
back-to-back `calibrate` invocations on the same plan produce
byte-identical YAML. Determinism is **conditional on cache-state
stability**, not on the plan or the calibrate code itself.

**What this means for the operator workflow:**

1. **Calibrate when the cluster is at the working set you actually
   intend to run with.** If you plan to operate with both
   benchmarks co-resident, calibrate after both are fully cached.
   If you plan to run only one, calibrate when the other isn't
   cached. The unique-size accounting only matches reality for the
   cache state captured at probe time.
2. **Re-calibrating after cluster churn is correct, not noisy.** A
   diff against the prior YAML reflects real-world changes in
   layer-share topology. If the change is small, FFD's reservation
   was already roughly right; if large, the cluster's working set
   has shifted meaningfully and the new numbers are the truth
   you'd want FFD to use going forward.
3. **The "fixed-point" verification** (re-run calibrate; if the
   diff is empty, you've converged) is only meaningful on a
   quiescent cluster — no rollouts in flight, no `xrlenv build
   apply` happening, no manual `docker pull`/`docker rmi`. Under
   those conditions, the diff *will* be empty.
4. **A stale calibrated YAML is never worse than the uncalibrated
   one** for FFD purposes. Worst case: the cache state has shifted
   such that some images' true unique cost is now higher than what
   the YAML says, FFD under-reserves, and opportunistic mode defers
   the over-reservation entries — recoverable via one more
   `--fill-missing` pass.

When you want a "team-shareable canonical reference YAML," the
recipe is:

```bash
# Step 1: bring the cluster to the cache state you want to anchor on
#         (e.g. apply + iterate --fill-missing until converged).
# Step 2: with the cluster quiescent (no rollouts, no other applies),
#         run calibrate to a separate output file.
# Step 3: re-run calibrate immediately and diff — empty diff confirms
#         the cluster is quiescent and the snapshot is stable.
# Step 4: commit the calibrated YAML with a note in the commit
#         message about which benchmarks were co-resident at
#         calibrate time.
```

If two co-resident benchmarks meaningfully overlap in their base
layers (e.g. SWE-bench Verified's ~500 Python-base instances
calibrated alongside terminal-bench-2's harbor-based task images),
expect the calibrated `size_hint_bytes` for each entry to be
**meaningfully smaller** than if either benchmark were calibrated
in isolation — the cross-benchmark layer dedup is the point of
calibrating against the joint working set.

#### Important: calibrating changes the plan_id

**The calibrated YAML has a different `plan_id` than the input.**
plan_id is the sha256 of the whole canonical plan body, including
the `placement.size_hint_bytes` and `placement.size_hint_source`
fields. When calibrate replaces those values, the canonical body
changes → the sha256 changes → fresh plan_id.

This is **by design** — same image_refs, but different placement
metadata is genuinely a different operator intent (FFD bin-packing
runs against the size hints, so different hints can produce
different node assignments). The plan_id correctly distinguishes
them.

What this means in practice:

- After promoting the calibrated YAML, the next `xrlenv build
  apply --plan <yaml>` creates a **fresh `build_plans` row** in
  state.db. The old `plan_id` row stays around as historical
  record (useful audit trail).
- The apply runs as a first-time dispatch for the new plan_id —
  the coordinator's idempotency check (`completed` → no-op
  unless `--force`) doesn't fire because it's a different
  plan_id.
- **All entries get dispatched**, even ones whose docker image
  already exists on the cluster. The per-image-ref dispatch
  path always invokes `build_image_fn` per entry; docker's own
  layer cache makes already-built images fast (~5-10s each for
  a cache-validation pass), but the build runs end-to-end. For
  16 fully-cached seta-env images this is ~90s wall-clock.
- The admin `/builds` panel shows **two rows for what feels like
  the same plan** — the old heuristic version + the new
  calibrated version. They aren't duplicates; they're different
  placement intents. The 12-char short_id makes them easy to
  distinguish at a glance.

If this re-dispatch cost is a problem for your workflow (you
have a much larger plan and most entries are already done),
there's a gap in the surface today: no `xrlenv build apply
--retry-missing-only` primitive yet. Workaround: use
`acquire_container` (or anything that drives `ensure_present`)
on the specific image_refs you need — the per-node source-spec
registry makes that idempotent + cheap when the image is already
present.

### Pin-budget enforcement at apply time

`xrlenv build apply` rejects plans whose pinned entries
collectively can't fit on each node. The check sums the
`size_hint_bytes` of every entry with `pinned: true`; if that
total exceeds any node's available image-cache budget
(`reserved_runtime_gb` + `buffer_gb` subtracted from free disk),
the apply fails fast with a `ManifestInvalid` naming the
offending node + the over-by amount.

Why hard reject: silent over-pinning bites weeks later when
unrelated work hits the threshold. The check is conservative
(every pinned entry counts toward every node's projected total,
since FFD hasn't run yet and might land them all on the same
node). If it passes, FFD can definitely find placement; if it
fails, FFD might still find one but the operator needs to
explicitly relax the budget or unpin entries before the cluster
treats this plan as safe to materialize.

Recovery options the error message lists:

- Drop `pinned: true` on lower-priority entries.
- Raise the budget (`budget.reserved_runtime_gb` or
  `budget.cap_per_node_gb`).
- Shrink size hints by recalibrating (`xrlenv build calibrate`)
  after a real cluster build — heuristic hints often over-estimate.

### Cancelling an in-flight plan

`xrlenv build cancel --plan <plan-id-or-prefix>` has two modes,
matching the two apply modes:

| Your situation | What to run | What it does |
|---|---|---|
| `xrlenv up` is running and the cluster is mid-build | `xrlenv build cancel --plan <id> --connect-host <admin-host>` | POSTs `/api/build/cancel`. The admin marks the plan `cancelled`, marks every `pending` assignment `cancelled` directly, and dispatches a `CancelBuildImageCommand` to each node currently building an assignment. Each node interrupts its in-flight `docker build` (kills the running build container labeled `xrlenv.cancel-key=<image_ref>`, cancels the asyncio task) and the `BuildImage` reply lands as `failed: cancelled by operator`. The `xrlenv build apply --connect-host` CLI you might have left polling sees the new terminal status and exits. |
| You just want to clear a stuck plan record (no live cluster, or you don't care about interrupting builds) | `xrlenv build cancel --plan <id>` | Local-only. Updates the plan row to `cancelled` in `state.db`. Running `docker build` processes on remote nodes are NOT touched. |

`<id>` accepts a full SHA-256 plan_id or a unique prefix (≥4 chars
— matches the 12-char short id the admin `/builds` panel shows).
Cancel is idempotent: against an already-`completed` /
`cancelled` / `superseded` plan it's a no-op success.

If the cancel fanout to a particular node fails (node disconnected,
docker daemon unreachable, transport raised), the per-(node,
image_ref) error appears in the response's `errors` list AND the
assignment row transitions from `building` to `failed` with an
error message naming the cancel-dispatch reason. The plan-level
status still flips to `cancelled`. State stays self-consistent
— there are never `building` rows under a `cancelled` plan — and
a subsequent `--force` re-apply will retry that image because
`failed` is not a "skip" status.

The `xrlenv.cancel-key` Docker label is reserved (set on every git
build automatically) — operators **must not** set it manually on
unrelated images, or the cancel path will kill them.

#### After cancel — what the operator should know

**Cancel is sticky.** Once a plan transitions to `cancelled`,
the coordinator's apply finalizer will not overwrite that status
back to `partial_failure` / `completed` — even if the in-flight
dispatch loop was already past its last `await` point and would
otherwise have run the terminal-status update. This is enforced
via a CAS-style state update
(`try_update_build_plan_status(plan_id, expected_current=
"in_flight", new_status=…)`): the finalizer only flips
`in_flight → terminal`; any other value (most importantly
`cancelled`) is left alone. The persisted status reflects the
operator's intent, not whichever side of the race finished last.

**Re-applying a cancelled plan starts a fresh dispatch under the
same plan_id.** The plan_id is content-addressed (sha256 of the
canonical plan body) — re-applying the same YAML produces the
same id. The coordinator's idempotency short-circuit fires only
for `in_flight` (rejected) and `completed` + no `--force` (no-op).
Every other status — including `cancelled`, `partial_failure`,
and `superseded` — falls through to the full dispatch path.
Concretely, when you re-apply after a cancel:

1. The coordinator sees `existing.status == "cancelled"`.
2. It flips the plan back to `in_flight`, purges the prior
   assignment rows (`delete_assignments`), and records a fresh
   placement.
3. Dispatch proceeds normally. Same plan_id throughout;
   `applied_at` refreshes.

So `cancelled` is not a permanent terminal state — it's "the
operator interrupted this attempt." Editing the plan YAML is
not required to retry; the same file works. See [Recipe: cancel an
in-flight plan and retry](#recipe-cancel-an-in-flight-plan-and-retry)
for the command pattern.

### Why placement can shift between runs

The bin-packer reads free disk **at apply time**. Disk drifts as
containers, logs, and OS state come and go, so the same plan can
land 8/0, 6/2, or 4/4 across two nodes on different runs. This is
not randomness — identical free-disk snapshots produce identical
placement, the cluster is just reflecting current capacity. To pin
placement deterministically, either:

- set `placement.preferred_home_count` to your cluster's node count
  (every ref lands on every node), or
- set `budget.cap_per_node_gb` so the bin-packer treats every node
  as having exactly that capacity.

### Idempotency and `plan_id`

Every plan canonicalises (sorted keys, dropped nulls, JSON-encoded)
to a stable `plan_id` (sha256). Two YAMLs with identical content
but different key order produce the same id. Dispatch uses the id
as follows:

- Re-applying a `completed` plan is a no-op (idempotency layer 2).
- Re-applying an `in_flight` plan is rejected with a clear error
  (idempotency layer 1, concurrency control).
- `--force` re-dispatches all entries regardless of prior status.

`compute_plan_id(plan)` lives in `xrlenv/control/build_plan.py`.

## How planning and eviction stay coordinated

As described in {doc}`cache_eviction`, the node-side cache evictor
is **reactive** — it only runs under disk pressure. The build
planner is its proactive counterpart, placing bytes ahead of
demand. They share state so they don't fight each other:

- The planner respects the same per-node disk budget the eviction
  loop uses (`reserved_runtime_gb` + `buffer_gb`), so an applied
  plan never plans into a region the cache manager would
  immediately evict.
- Eviction targets cold images first and considers rebuild cost
  within each LRU bucket. A `type: registry` entry is the cheapest
  to refetch; a `type: git` entry is expensive (clone + build);
  pinned entries are not eligible at all. See {doc}`cache_eviction`
  for the full ordering.
- An evicted but still-needed registry-source image is rebuilt on
  next demand: the next `acquire_container` triggers
  `ensure_present`, which re-pulls.

### Idempotence and recovery

Both planning paths are designed for safe retries:

- `ensure_present(image)` on a hit is a fast no-op plus LRU touch.
- Two acquires for the same missing image on one node share the
  same in-flight work.
- Re-applying the same build plan reuses existing assignment state
  and Docker layer cache.
- Completed assignments survive CLI reconnects because plan
  snapshots are persisted in the state store.

Use `/images/cache` to debug per-node disk pressure,
`/images/catalog` to inspect cluster-wide coverage, and `/builds`
to inspect applied plans.

## Quick notes

Nuanced or status-dependent details you only need when they come up.

### Node-side build-context cache location

For nodes running git-source builds, the per-node
`GitSourceBuilder` maintains a build-context cache (clones get
reused across builds, bounded by an LRU 5 GB total cap). The
location depends on how xrlenv was installed:

| Install shape | Default cache root | How it gets there |
|---|---|---|
| Bootstrap-managed cloud VM (`deploy/bootstrap-{gcp,aws}.sh`) | `/var/cache/xrlenv/build-context-cache/` | The bootstrap creates the dir with the right ownership and writes `XRLENV_BUILD_CONTEXT_CACHE=/var/cache/xrlenv/build-context-cache` to `/etc/xrlenv/node.env`. The systemd unit reads it from there. |
| Local-device install (running `xrlenv-node` directly, or in-process via `LocalRuntime`) | `~/.xrlenv/build-context-cache/` | The library default — matches the rest of xrlenv's per-user dotfile layout. |

The env var `XRLENV_BUILD_CONTEXT_CACHE` overrides both defaults
when set. Useful if you want to point at a faster disk
(e.g. NVMe-backed `/mnt/nvme/...`) on a node with mixed storage.

If neither the env var is set nor the default is writable (e.g.
on a bootstrap-managed node mounting `~/.xrlenv` read-only via
`ProtectHome=read-only`, **without** the bootstrap setting the
env var), `GitSourceBuilder` falls back to
`/tmp/xrlenv-build-context-cache-<uid>/` and logs a warning. The
fallback works (cache miss → re-clone is correct) but doesn't
survive reboots; this is the safety net, not the intended path.

Symptom of the read-only-default case before the bootstrap
addition or fallback (seen on early bootstrap-managed deploys):

```
XRLEnvError: remote command OSError: [Errno 30] Read-only file system: '/opt/xrlenv/.xrlenv'
```

For the operator-facing "where does disk go and how do I clear
it" picture across all of xrlenv's writable paths (runs,
harbor cache, build-context cache, Docker image cache), see
{doc}`/deploy/multi_node_deployment/runbook` § "Disk layout &
cleanup".

### What's live today vs what waits on the source-build pipeline

What ships today:

- Schema, per-benchmark generators, canonical YAMLs, and `plan_id`
  hashing.
- Coordinator dispatch for entries with
  `context_source: type: registry` (lowered to per-`(node,
  image_ref)` `ensure_present`), `type: git` (clone + `docker build`
  on the node, with persistent build-context cache), and
  `type: tarball` (operator-side bytes resolved at apply time,
  shipped over `BuildImageCommand`, untarred + built on the node).
  All three auto-label: registry images carry no rebuild-cost label
  (sort cheapest in eviction), git images carry
  `xrlenv.image.rebuild-cost=local-build-expensive`, tarball images
  carry `local-build-cheap` (re-shipping bytes is cheaper than
  re-cloning + re-fetching). Every source-built image also carries
  `xrlenv.cancel-key=<image_ref>` so operator cancel can find the
  in-flight build container.
- Operator-cancel of in-flight plans: `xrlenv build cancel
  --connect-host` dispatches `CancelBuildImageCommand` to each
  building node, kills the running build container, and marks the
  plan + assignments `cancelled`. Works for git AND tarball builds.
- Tarball size cap (default 100 MB; tunable via
  `xrlenv build apply --build-tarball-max-bytes`). Oversized
  payloads reject **operator-side** at apply time, before any wire
  traffic — the operator can `.dockerignore`-trim and retry without
  having burnt cluster cycles. Don't raise this above the gRPC
  channel cap (128 MB minus envelope headroom — the
  `BuildImageCommand` would fail to deserialize on the node).
- Opportunistic mode (default): FFD places what fits; overflow
  **registry-source** entries are recorded as `status: registered`
  against their preferred-home node and lazy-pulled at acquire time
  via the runtime's `ensure_present` path. The node's LRU image-cache
  evictor reclaims disk when needed. Use this for plans that exceed
  the cluster's current image-cache budget (e.g. a 500-instance
  SWE-bench Verified sweep on a 2-node cluster) — apply succeeds,
  rollouts pull images on demand, and the cluster rotates through
  the working set. `--eager` opts back into the legacy strict mode
  that rejects any plan that doesn't fully fit upfront.
- Overflow entries with `git`/`tarball` source: rejected upfront in
  opportunistic mode with a clear remediation message — lazy build
  for non-registry sources needs per-node source-spec broadcast that
  the per-image-ref dispatcher doesn't carry yet. Either shrink the
  plan / connect more nodes so they fit, or pass `--eager` to
  surface the rejection as `InsufficientCapacity`.
- Idempotency: the coordinator honors `in_flight` (rejects) and
  `completed` (no-op unless `--force`).

What is **not** dispatched yet:

- Entry-level delta dispatch: today a changed plan re-dispatches
  every registry entry through `ensure_present` (a fast no-op when
  the image is already on the node).

### The legacy benchmark-driven plan shape

`build-plan.yaml` has two coexisting top-level shapes:

- **Per-image-ref shape** (current): top-level `entries: [...]`
  with a discriminated `context_source`. Schema-agnostic about the
  benchmark. New work always emits this.
- **Benchmark-driven shape** (legacy): top-level
  `benchmarks: [...]`, where each entry names a manifest plus a
  selection (smoke / instances / all). Routes through registered
  `BenchmarkImageBuilder`s via `xrlenv build apply --plan ...` and
  the existing `BuildCoordinator`. Preserved only for plug-ins
  that still ship a builder.

`entries` and `benchmarks` are mutually exclusive — the schema
validator rejects a plan that sets both.

The build coordinator persists a plan snapshot in both shapes; the
admin panel shows it under `/builds` and the JSON endpoint
`GET /api/build/plans/<plan_id>` returns the same state for tools.

### Size hints and calibration

`registry-probe` sizes (terminal-bench-2, swebench-verified) come
from Docker Hub's v2 manifest API at generation time and are
accurate. `heuristic` sizes (seta-env, anything built from
source) are estimates; the bin packer adds a safety margin.

The `xrlenv build calibrate` flow (now live — see "Calibrating
size hints from the cluster" above) re-emits a plan with
`size_hint_source: cluster-reported` for every entry the cluster
has materialized at least once, sourced from each node's own
image-cache report. Re-applying the calibrated plan on a fresh
cluster reproduces the original placement deterministically.

`entries`-shaped plans persist `BuildAssignmentRecord` rows under the
synthetic `<per-image-ref>` benchmark tag, with the same status table
the legacy benchmark path uses (`pending`, `building`, `done`,
`failed`, `registered`). `registered` rows in the per-image-ref
shape today are limited to **registry-source** deferred entries
(lazy-pulled on demand at acquire time); deferred git/tarball entries
reject upfront in opportunistic mode and need `--eager` or a plan
that fits the budget.

### Common pitfall: applying with no nodes

Applying against zero connected nodes fails fast with a clear
message; wait for `xrlenv nodes` to show at least one node
`connected` before applying. A freshly-started `xrlenv up`
typically takes 5–15s before remote nodes reattach through their
systemd retry loop.

### Smoke coverage

`tests/smoke/test_build_plan_dispatch_tb2.py` exercises fresh
apply, idempotent re-apply, `--force` re-apply, operator-built
fresh plan, and size-hint calibration end-to-end against a real
cluster.

## See also

- {doc}`on_demand` — the runtime acquire path; what happens when
  an acquire arrives without a prefetched plan.
- {doc}`cache_eviction` — node cache states and eviction order.
- {doc}`/technical_details/scheduling` — placement scoring.
- {doc}`/observability/capacity` — operator views for capacity and
  image state.

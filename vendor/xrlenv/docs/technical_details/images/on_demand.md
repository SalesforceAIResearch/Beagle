# On-demand image acquire

For ad-hoc workloads — the SDK, the Docker drop-in, framework
adapters — images are pulled at acquire time rather than prefetched.
This is the default path; build planning ({doc}`build_plan`) is the
optimization for when you already know the image set ahead of time.

This page covers what happens when a workflow calls
`acquire_container(image=...)`: how the scheduler picks a node, what
`ensure_present(image)` does once the call lands on that node, and
how strict mode lets you fail fast instead of pulling.

The runtime side is `ensure_present(image)` in
`xrlenv/node/image_cache.py`. The placement side lives in
`xrlenv/control/scheduler.py` — see
{doc}`/technical_details/scheduling` for the full scoring story.

## The acquire flow

1. The workflow asks for `image=...`.
2. The scheduler checks connected nodes for backend support, CPU /
   memory / disk capacity, `task_key` fairness, and image presence.
3. Nodes that already have the image get a full image-affinity
   bonus.
4. If no node has the image but a preferred home is registered
   (from a {doc}`build plan <build_plan>`), that node gets a
   smaller preferred-home bonus.
5. The selected node runs `ensure_present(image)` before starting
   the container.

Image affinity can break ties and avoid cold pulls, but it does
not override hard scheduler constraints. Connectivity, backend
support, capacity, fairness caps, and pending reservations all
gate placement first. A warm image does not make a node eligible
by itself.

See {doc}`/technical_details/scheduling` for the scoring formula,
how image presence is gathered, and how preferred-home routing
works.

## The `ensure_present` decision tree

Once a node receives an acquire, it walks four cases in order:

1. **Hit.** The image is already on disk; touch the LRU timestamp
   and return. The fast path — sub-millisecond.
2. **Miss with a local builder.** If the node has a registered
   builder for the image (e.g. a git or tarball source declared in
   a build plan's source registry), run the builder. Source-built
   images survive eviction because the node persists the build
   recipe locally — see {doc}`build_plan` for the persistence
   model.
3. **Miss with a registry-resolvable ref.** Pull from the
   registry. Concurrency on this path is bounded by the node's adaptive
   AIMD pull limiter — see {doc}`cache_eviction` for the floor/ceiling
   config fields and `XRLENV_PULL_CONCURRENCY` /
   `XRLENV_PULL_CONCURRENCY_CEILING` tuning guidance. The per-pull deadline
   defaults to 600 s; for known-huge images (multi-GiB GPU images,
   SWE-bench Pro tags) pass `acquire_timeout_s=` on
   `Client.acquire_container` to widen it — see {doc}`cache_eviction`
   for the full three-layer deadline story.
4. **Strict acquire.** If the caller passed
   `ensure_image_present=False`, the call fails immediately on
   miss instead of pulling or building.

After a successful pull or build, the LRU timestamp is set. A
fresh build also starts the build-time grace window
({doc}`cache_eviction`) so the image isn't reaped before any
rollout has had a chance to use it.

## Idempotence

`ensure_present(image)` is safe to call repeatedly:

- On hit, it only touches the LRU timestamp.
- Two acquires for the same missing image on the same node share
  the same in-flight work. The second caller waits for the first's
  pull or build instead of starting a parallel one.
- Concurrent acquires for *different* images run in parallel up to the
  AIMD limiter's current bound (between `pull_concurrency` floor and
  `pull_concurrency_ceiling` ceiling).

This matters for raw-container workflows where the same SDK
process may issue many acquires with the same image in tight
succession — none of them duplicate work.

## Strict mode

`ensure_image_present=False` trades cold-pull latency for explicit
failure. The acquire returns a structured `ImageNotPresent` error
on miss. Two situations where this is the right default:

- **A run whose working set is supposed to be hot already.** A
  miss signals that something upstream (operator-level prune,
  unexpected eviction, missed build-plan apply) has gone wrong.
  Surface the miss instead of hiding it behind a slow pull.
- **Latency-sensitive workloads** that prefer to queue through the
  admission layer rather than block the calling coroutine on a
  multi-second pull.

The default is `ensure_image_present=True` — most workflows want
the convenient "if it's not there, get it" behavior.

## Quick notes

### Per-instance image resolution

Some templates (Pattern A adapters — typical for benchmark
adapters where each task has its own image) resolve `image=`
per-instance rather than declaring it on the manifest. The SDK
runs the resolver overlay before admission, so `manifest.image`
is the concrete per-instance ref by the time the scheduler sees
it. The on-demand flow above is unchanged; the resolver simply
fires once per acquire.

If the resolver hasn't fired yet (no `instance_id` available),
the manifest arrives at admission with a tag-only image. From the
scheduler's perspective this is the same as "no image affinity
signal" — placement falls back to resource slack alone, and the
chosen node materializes whatever the resolver eventually picks.

### What an evicted source-built image does on next acquire

If the cache evicted a source-built image but the build recipe is
still in the node's source registry, the next `acquire_container`
triggers `ensure_present`, finds the recipe, and rebuilds. From
the operator's perspective this looks like a slight latency
hiccup, not a failure. See {doc}`build_plan`'s "Build-on-acquire
after eviction" section for the persistence model.

If the recipe was also deleted (operator pruned the source
registry, or a registry-only ref was deleted upstream), strict
mode is the cleanest way to surface this — otherwise the acquire
will keep trying to pull a ref that no longer exists.

## See also

- {doc}`cache_eviction` — what happens to images on the node
  after they're acquired.
- {doc}`build_plan` — the proactive counterpart: planning image
  placement ahead of demand.
- {doc}`/technical_details/scheduling` — placement scoring,
  image affinity, preferred home, admission queueing.

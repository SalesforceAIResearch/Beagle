# Configuration & environment variables

XRLEnv is configured mostly through CLI flags and run-config; these
environment variables are the lower-level tunables, grouped by where they
take effect. Each var can also live in a `.env` file that XRLEnv loads
automatically on import (see `XRLENV_DOTENV` below).

**Scope key**

| Scope | Reads this var |
|---|---|
| control-plane | `xrlenv up` process |
| node | `xrlenv-node` daemon |
| consumer | SDK caller / benchmark script |
| benchmark-plugin | benchmark-specific plugin code |
| bootstrap | `bootstrap-*.sh` / `xrlenv bootstrap` |

---

## Connectivity & authentication

Used by consumers (SDK / benchmark scripts), nodes, and CLI tools to
reach the cluster.

| Variable | Default | What it controls | Scope |
|---|---|---|---|
| `XRLENV_GRPC_HOST` | — (required) | Hostname or IP of the control plane gRPC endpoint | consumer |
| `XRLENV_GRPC_PORT` | `50051` | gRPC port | consumer |
| `XRLENV_GRPC_SECURE` | `false` | Set `true`/`1`/`yes`/`on` to enable TLS on the gRPC channel | consumer |
| `XRLENV_CONSUMER_TOKEN` | — | Bearer token for consumer-plane auth (issued by `xrlenv tokens issue consumer`) | consumer |
| `XRLENV_OPERATOR_TOKEN` | — | Bearer token for operator-plane auth; falls back to `$XRLENV_HOME/secrets/operator.token` | consumer |
| `XRLENV_NODE_TOKEN` | — | Bearer token injected into the `xrlenv-node` systemd unit | node |
| `XRLENV_CONTROL_PLANE` | — | `host:port` of the control plane, written by bootstrap into the node's systemd environment file | node |
| `XRLENV_NODE_ID` | auto | Node identity; bootstrap auto-detects from cloud metadata; override when detection fails | node |
| `XRLENV_HOME` | `~/.xrlenv` | Root for operator state (state DB, run artifacts, secrets). Set per-checkout to isolate dev clusters | consumer / control-plane |
| `XRLENV_DOTENV` | `on` | Set `off` to suppress automatic `.env` loading on import | consumer |

---

## Consumer request tags

| Variable | Default | What it controls | Scope |
|---|---|---|---|
| `XRLENV_GROUP_ID` | — | Default `group_id` attached to raw-container acquires; overrides per-call `group_id` when set. See `Task key` in the {doc}`glossary`. | consumer |

---

## Node — resilience & concurrency

Controls per-node limits for container lifecycle operations.

| Variable | Default | What it controls | Scope |
|---|---|---|---|
| `XRLENV_MAX_GET_ARCHIVE_RELAY_BYTES` | `134217728` (128 MiB) | Maximum archive size relayed through the control plane; raises `ArchiveTooLarge` if exceeded. `0` disables the cap. | node |
| `XRLENV_RAW_ARCHIVE_CONCURRENCY` | `4` | Concurrent `get_archive` operations per node | node |
| `XRLENV_RAW_CREATE_CONCURRENCY` | `4` | Concurrent container-create operations per node | node |
| `XRLENV_RAW_SYSBOX_CREATE_CONCURRENCY` | `1` | Concurrent create operations for **sysbox** (non-`runc`) containers per node — a tighter, separate cap because sysbox-fs pre-register is much slower than a plain runc create. `0` falls back to `XRLENV_RAW_CREATE_CONCURRENCY`. | node |
| `XRLENV_RAW_DESTROY_CONCURRENCY` | `4` | Concurrent container-destroy operations per node | node |
| `XRLENV_RAW_LIVENESS_TTL_S` | `120` | Seconds before a raw session whose consumer stopped heartbeating is declared dead and reaped | control-plane |
| `XRLENV_RAW_LIVENESS_REAP_BATCH` | `50` | Maximum sessions destroyed per GC sweep | control-plane |
| `XRLENV_RAW_HEARTBEAT_INTERVAL_S` | `30` | How often the SDK sends a keepalive heartbeat for open raw sessions | consumer |
| `XRLENV_EXEC_CHUNK_TIMEOUT_S` | `3600` | Output-idle ceiling for a **streaming** `exec`: how long the control plane waits for the next stdout/stderr chunk before aborting the stream. The effective wait is `min(this, remaining exec budget)`, so the large default defers to the per-exec `timeout_s` (`timeout_s + 30` whole-stream cap) — a legitimately silent test/compile phase in a benchmark verifier is not killed at a tight idle window and spuriously retried. A dead node is still caught by the heartbeat / stream-disconnect path, not this ceiling. Must be `> 0` (a non-numeric or non-positive value falls back to the default). | control-plane |
| `XRLENV_DISK_GUARD_ENABLED` | `true` | Enable node-side disk-pressure guard | node |
| `XRLENV_DISK_GUARD_INTERVAL_S` | `15` | Seconds between disk-pressure checks | node |

---

## Node — image pull & IO throttle

Controls the AIMD pull-concurrency limiter and IO backpressure.
See {doc}`/technical_details/images/cache_eviction` for the full model.

| Variable | Default | What it controls | Scope |
|---|---|---|---|
| `XRLENV_PULL_CONCURRENCY` | `2` | AIMD floor: minimum concurrent image pulls | node |
| `XRLENV_PULL_CONCURRENCY_CEILING` | `64` | AIMD ceiling: maximum concurrent image pulls | node |
| `XRLENV_PULL_CONCURRENCY_INITIAL` | `16` | Starting concurrency before any AIMD adjustment | node |
| `XRLENV_IO_THROTTLE` | enabled | Set `off`/`false`/`0` to disable IO-utilisation-based throttling entirely | node |
| `XRLENV_IO_UTIL_HIGH_PCT` | `0.90` | IO utilisation fraction above which concurrency is reduced | node |
| `XRLENV_IO_UTIL_LOW_PCT` | `0.70` | IO utilisation fraction below which concurrency may increase | node |
| `XRLENV_CONTENT_GC_MIN_INTERVAL_S` | `60` | Minimum seconds between Docker content-store GC passes | node |

---

## Node — cache eviction

Per-node image LRU eviction thresholds.
See {doc}`/technical_details/images/cache_eviction` for the full eviction algorithm.

| Variable | Default | What it controls | Scope |
|---|---|---|---|
| `XRLENV_EVICT_THRESHOLD_CAP_GB` | `50` | Start evicting when cached images exceed this cap (GiB). Node-local floor is 15 GiB; this is the upper cap. | node |
| `XRLENV_EVICT_TARGET_CAP_GB` | `75` | Stop evicting once cached images drop to this cap (GiB). Node-local floor is 25 GiB. | node |
| `XRLENV_IMAGE_SNAPSHOT_TTL_S` | `15` | Seconds the admin panel caches the image inventory snapshot before refreshing | control-plane |

---

## Control plane — image planning & build

| Variable | Default | What it controls | Scope |
|---|---|---|---|
| `XRLENV_PACK_ONDISK_MULTIPLIER` | `3.0` | Multiplier applied to compressed image size when estimating on-disk footprint for capacity planning | control-plane |
| `XRLENV_PACK_ONDISK_MULTIPLIER_CLUSTER_REPORTED` | `1.0` | Multiplier applied to cluster-reported on-disk size (already uncompressed) | control-plane |
| `XRLENV_REPORT_IMAGES_TIMEOUT_S` | `60` | Seconds the control plane waits for a node to report its image inventory | control-plane |
| `XRLENV_BUILD_CONCURRENCY` | `32` | Maximum concurrent image builds dispatched by the build coordinator | control-plane |
| `XRLENV_BUILD_CONTEXT_CACHE` | — | Path to the git build-context cache directory on a node; set in the systemd unit by bootstrap | node |
| `XRLENV_BUILD_DISK_HEADROOM_FACTOR` | `3.0` | Minimum free-disk factor required before dispatching a build | control-plane |
| `XRLENV_BUILD_DISK_POLL_S` | `5.0` | Seconds between disk-headroom checks while waiting for space | control-plane |
| `XRLENV_BUILD_DISK_WAIT_TIMEOUT_S` | `300.0` | Seconds to wait for disk headroom before failing a build dispatch | control-plane |
| `XRLENV_BUILD_SHA` | git HEAD | Short SHA embedded in build artefacts; set by deploy scripts; falls back to `git rev-parse` in dev | control-plane |
| `XRLENV_TEMPLATE_DIRS` | — | Colon-separated list of extra directories to scan for `EnvAdapter` template packages, in addition to the `xrlenv.benchmarks` entry-point group | control-plane / node |

---

## Control plane — fleet reservations

| Variable | Default | What it controls | Scope |
|---|---|---|---|
| `XRLENV_FLEET_RESERVATION_TTL_S` | `600` | Seconds before an unreleased fleet reservation is garbage-collected by the control plane | control-plane |

---

## Registry & image freshness

Controls how the control plane resolves image tags to digests and which
registries nodes pull from. See {doc}`/technical_details/images/registry_freshness`
for the freshness model and {doc}`/technical_details/images/on_demand`
for the private-registry setup.

| Variable | Default | What it controls | Scope |
|---|---|---|---|
| `XRLENV_REGISTRY_DIGEST_RESOLVE` | `1` (enabled) | Set `0`/`false`/`off` to disable tag→digest resolution at acquire time (kill-switch for broken registries) | control-plane |
| `XRLENV_REGISTRY_SCHEME` | `http` | HTTP scheme used to probe the private registry for manifest digests (`http` or `https`) | control-plane |
| `XRLENV_REGISTRY_RESOLVE_TTL_S` | `60.0` | Seconds a resolved digest is considered fresh before re-probing | control-plane |
| `XRLENV_REGISTRY_RESOLVE_MAX_STALE_S` | `900.0` | Seconds a stale cached digest may be served during a transient registry outage | control-plane |
| `XRLENV_REGISTRY_RESOLVE_HOST_MAP` | — | Comma-separated `src=dst` host remappings applied when the CP probes the registry (e.g. `node-host:5011=127.0.0.1:5011`) | control-plane |
| `XRLENV_REGISTRY_MIRROR` | — | Docker pull-through mirror address forwarded to nodes in the node-config payload | control-plane |
| `XRLENV_PRIVATE_REGISTRY` | — | Address of the private registry forwarded to nodes (e.g. `node-host:5011`) | control-plane |
| `XRLENV_PRIVATE_REGISTRY_HOST` | — | Hostname of the private registry; used by benchmark plugins to construct image refs | benchmark-plugin |
| `XRLENV_PRIVATE_REGISTRY_PORT` | `5011` | Port of the private registry | benchmark-plugin |

---

## In-sandbox stub

Set inside the container by the Docker backend; normally not set by hand.

| Variable | Default | What it controls | Scope |
|---|---|---|---|
| `XRLENV_STUB_UDS` | `/run/xrlenv/stub.sock` | Unix-domain socket path the in-sandbox stub listens on | node (injected into container) |
| `XRLENV_STUB_LOG_LEVEL` | `INFO` | Log level for the in-sandbox stub process (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | node (injected into container) |
| `XRLENV_STUB_LOG_FORMAT` | `auto` | Log format for the stub: `auto` (JSON in TTY-less environments, human-readable otherwise), `json`, or `text` | node (injected into container) |

---

## Bootstrap

Read by `bootstrap-*.sh` shell scripts and `xrlenv bootstrap` before the
Python runtime is available.

| Variable | Default | What it controls | Scope |
|---|---|---|---|
| `XRLENV_VERSION` | `main` | Git ref or wheel version to install on the node | bootstrap |
| `XRLENV_USER` | `xrlenv` | System user under which `xrlenv-node` runs | bootstrap |
| `XRLENV_PYTHON` | auto | Path to a Python 3.12+ binary; bootstrap auto-detects if unset; pin when the system has multiple Pythons | bootstrap |
| `XRLENV_WHEEL` | — | Path to a pre-built `.whl` file; takes precedence over `XRLENV_VERSION` when set | bootstrap |
| `XRLENV_REPO` | — | Path to a local checkout to install in editable mode; takes precedence over `XRLENV_WHEEL` when set | bootstrap |

---

## Harbor plugin

Used by the Harbor framework adapter (`xrlenv_plugins.harbor`) and
terminal-bench-style benchmark scripts.

| Variable | Default | What it controls | Scope |
|---|---|---|---|
| `XRLENV_BENCHMARK_CACHE` | — | Root directory of the local Harbor task cache (populated by `build_cache.py`); required to run harbor-based benchmarks | benchmark-plugin |
| `XRLENV_HARBOR_ACQUIRE_QUEUE_TIMEOUT_S` | `240.0` | Per-acquire wait (seconds) for a capacity slot before raising `CapacityExhausted` (which the benchmark sweep retries automatically). Raise to tolerate more backpressure; keep below Harbor's ~360 s setup window so that a paced acquire fails fast and retriably before Harbor's non-retriable cancel fires. When this expires the rollout is sealed as `capacity_rejected` in the admin panel — a pacing event, not a failure. No control-plane redeploy needed; read client-side from the sweep/harness process. | benchmark-plugin |
| `XRLENV_CPU_PINNING` | — | Set to `1` in a task's `[environment.env]` block to request cpuset pinning for that task's container. Not a process-level env var — it is a per-task config key read from the task TOML. Maps to `RuntimeLimits(cpu_pinning=True)` → `cpu_isolation=best_effort` on the acquire path (pin `ceil(cpu_limit)` cores if the node has free capacity, else CFS quota). Harbor task markers cannot request `required`. See {doc}`/technical_details/resource_isolation`. | benchmark-plugin |

**`XRLENV_BENCHMARK_CACHE` default, by context.** The golden-path benchmark sweep wrappers
(`xrlenv_plugins/benchmarks/*/run_full_sweep.sh`) treat it as a deployment constant: when unset
they fall back to the shared FSx root `/path/to/benchmark-cache`. The
**phase-0 `terminal-bench-2` onboarding example** is a self-contained demo and instead defaults
to `~/.cache/harbor/tasks` (where its `scripts/populate-harbor-cache.sh` clones the catalog) —
this applies to that example's `smoke.py`, its populate script, and
`xrlenv_plugins/images_build/terminal_bench_2/build_plan_gen.py`. Either way the retired
`XRLENV_HARBOR_CACHE` variable and the `.../xrlenv_harbor_cache` path are **hard-rejected**
(fail loud with a migration hint), and `build_plan_gen --all` fails loud rather than silently
emitting the 8-task smoke plan when its cache is absent.

**Harbor image resolution** is not controlled by an environment variable. The adapter resolves the container image in this order: (1) the `xrlenv_image_template` per-run kwarg (injected by sweep drivers such as `seta/run_oracle_sweep.py` and `terminalworld/run_oracle_sweep.py` via `EnvironmentConfig(kwargs={"xrlenv_image_template": ...})`), (2) the per-task `docker_image` field in the task's `task.toml` (e.g. set by `build_cache.py --stage repin` for benchmarks like LHTB), (3) the `hb__<environment_name>` local-build convention. Sweep drivers compose the template from `--registry` / `$XRLENV_PRIVATE_REGISTRY_HOST`; operators set those, not the template itself.

---

## EvoClaw plugin

| Variable | Default | What it controls | Scope |
|---|---|---|---|
| `XRLENV_IMAGE_REGISTRY` | — | Optional Docker Hub mirror-host prefix prepended to EvoClaw image refs (e.g. `mirror.local:5000`). When unset, refs are used as-is. | benchmark-plugin |

---

## Node — CPU isolation

CPU isolation (`cpu_isolation` field on `ResourceSpec`) lets a
container request dedicated logical CPUs so unpinned neighbors cannot
trample its cores. A node must be explicitly opted in; see
{doc}`/technical_details/resource_isolation` for the full model.

| Variable / knob | Default | What it controls | Scope |
|---|---|---|---|
| `CPU_ISOLATION_POOL` | `()` (empty) | **Deploy-script array** in `slurm_scripts/deploy_dev.sh` / `deploy_prod.sh`. Lists node hostnames to enable via `scripts/enable_cpu_isolation.sh` during each deploy (idempotent). An empty array = all nodes non-capable; the committed deploy scripts populate their own pools (consult each file for the current set — both dev and prod currently list isolation-enabled nodes). On prod, enabling a node restarts docker (maintenance window). Do not overlap with `SYSBOX_POOL`. | bootstrap |
| `XRLENV_SELFTEST_IMAGE` | `xrlenv-selftest:1` | Probe image tag used by the **enable-time** container self-test in `enable_cpu_isolation.sh` (run as root), which gates whether the node sets up the delegated `xrlenv-shared` cgroup. Persisted in `/etc/xrlenv/cpu_isolation.env` (a separate optional `EnvironmentFile` that survives `node.env` rewrites). The non-root agent detects capability from the delegated cgroup (§8.13), not from this image directly — but a node whose probe cannot run (image absent) never gets delegation and so stays non-capable. | node |
| `min_shared_cores` | 25% of logical CPUs | **Internal floor (not operator-configurable via `nodes.yaml`).** The minimum number of logical CPUs kept in the shared pool; pinning that would drop below it is refused (`best_effort` degrades to CFS quota; `required` fails placement/pin). Currently a fixed default derived at node wiring, not a `nodes.yaml` field. | node |

**`cpu_isolation` field on `ResourceSpec` / `acquire_container`.** Three values:

| Value | String | Behavior |
|---|---|---|
| `CpuIsolation.OFF` | `"off"` | CFS quota only (default). |
| `CpuIsolation.BEST_EFFORT` | `"best_effort"` | Pin if free pinnable capacity exists, else CFS quota. Scheduling-neutral. |
| `CpuIsolation.REQUIRED` | `"required"` | Pin or fail; placement restricted to `isolation_capable` nodes with free pinnable cores. |

---

## `nodes.yaml` policy fields

Operator knobs that live in the `nodes.yaml` inventory file (not environment
variables) are documented in {doc}`/deploy/multi_node_deployment/inventory`.
The most security-relevant one is `policy.allowed_runtimes` — the cluster-wide
gate for OCI runtime overrides (Sysbox). It is **empty by default**: every
`container_runtime` override is rejected until an operator opts in explicitly.

See {doc}`/deploy/multi_node_deployment/sysbox_pool` for the full Sysbox node
pool operator guide.

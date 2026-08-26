# Admin panel

The admin panel is a browser view over the same state used by the
CLI: connected nodes, running containers, rollout records, image
cache state, build plans, and trajectory artifacts.

Start it with:

```bash
xrlenv up --admin-port 8080
```

Then open `http://127.0.0.1:8080/`. By default the server binds to
loopback; use an SSH tunnel for remote control-plane hosts.

## Authentication

When the panel is exposed beyond loopback, it enforces two-tier HTTP
authentication. See {doc}`admin_auth` for roles, token issuance, the
browser login flow, CLI bearer-auth, the bind-guard rules, and the
loopback dev escape hatch.

Most pages are readable by any `consumer` / `viewer` / `operator` token;
write actions need `operator`. The exception is `/users`, which exposes
**every** tenant's activity and so requires `operator` even to read — on a
public bind a `viewer`/`consumer` token gets a 403, and the nav link is
hidden for them. On a loopback bind (auth bypassed) it's open like every
other page.

## What to check first

| Page | Use it for |
|---|---|
| `/` | Cluster summary: uptime, rollout counts, node count, and utilization. Includes a **"capacity-paced in last hour"** tile for `capacity_rejected` rollouts (backpressure events, not failures) — these are excluded from the "failed in last hour" tile and per-template failure rate. When configured, a **Cluster** banner also shows the control-plane endpoint nodes dial plus the registry mirror / private build registry. |
| `/health` | **Cluster health** operator page — per-node table of docker-run p95, windowed create throughput, error/timeout counts, create-gate active/waiting, heartbeat age, and (when `--adaptive-admission` is on) the AIMD admission-limit column. A collapsible legend explains each column. |
| `/healthz` | Liveness probe for load balancers and scripts — returns `ok` with no auth required. |
| `/fairshare` | **Operator-only.** Live fair-share policy + per-owner running / effective cap / owner cap / uncapped / blocked table. See {doc}`/deploy/multi_tenancy` for fair-share configuration. |
| `/nodes` | Roster and connectivity: connection status, heartbeat age, cloud, and expected address. Includes a **rollout-distribution figure** — all-time raw rollouts per node as bar lengths, with a coefficient-of-variation (CV) spread metric, to check the scheduler is spreading work evenly (0% CV = perfectly uniform). Per-node disk / cache usage lives in `/images/cache`. |
| `/users` | **Operator-only.** Per-tenant raw-rollout scoreboard grouped by `owner_id`: total / active / released / failed / cancelled / reaped / paced (`reaped` = any platform teardown that recorded a reason — GC deadline/liveness reclaim, `terminate_raw_group`, or a node-side orphan seal — not failed; `paced` = `capacity_rejected`, excluded from the `released ÷ total` success-rate denominator so a paced-then-retried run is not scored as a partial failure). **Totals are cumulative and preserved across the retention GC**, not bounded by the retention window. When the retention janitor prunes a `raw_rollouts` row, it folds that row's (owner, status) tally into the durable `owner_rollout_lifetime` table in the same transaction, so counts are never lost to GC. Individual rollout drill-down (the per-owner detail list) is still limited to the `--raw-rollout-retention-days` window; the page header shows that date range so it is clear which rollouts are individually browsable vs. present only in the aggregate. Note: rollouts pruned *before* lifetime tracking was enabled are not retroactively counted — the tally accrues only from the point the feature was deployed, not from cluster inception. |
| `/rollouts/raw` | Managed-container sessions created by `acquire_container` or the Docker SDK drop-in. `capacity_rejected` is an accepted `raw_status` filter value; those rows render with an amber badge (like `reaped`/`cancelled`), never the red `failed` badge. |
| `/rollouts/template` | Template-driven rollouts, mainly built-in and advanced integrations. |
| `/sandboxes` | Containers backing template (case-1, gym/step) rollouts. Raw-container workloads don't create sandboxes — their runs are under `/rollouts/raw`. |
| `/images/cache` | Per-node disk pressure and cached image inventory. |
| `/images/catalog` | Distinct image refs and node coverage across the cluster. |
| `/builds` | Applied image build/distribution plans. |
| `/capacity` | Per-template capacity **planning estimate** (not live health) — how many of each template a node could hold. |

Bare `/rollouts` redirects to `/rollouts/raw` because managed
containers are the main workflow path.

## Cluster health (`/health`)

`GET /health` is an operator page — a per-node signal table fed by the
`node_health` mirror that each node pushes on every heartbeat.

**Per-node columns:**

| Column | Description | Triage threshold |
|---|---|---|
| `docker-run p95` | p95 docker-run latency (ms) over the last ~2 min window — the smooth saturation signal. | > 30 000 ms = docker daemon slow or node saturated. |
| `creates (last 2m)` | Container creates completed in the last ~2 min (throughput — a count over time, not a live value). | A drop while p95 climbs = the node is struggling. |
| `errors / timeouts` | Docker errors (and the timeout subset) in the last ~2 min window — the emergency signal. | Any non-zero value warrants investigation. |
| `create gate (active / waiting)` | Live create concurrency: acquisitions running right now vs. ones waiting for a gate slot. | "waiting" is brief, healthy backpressure — **not** an error. High + p95 climbing = the create-gate is the bottleneck. |
| `heartbeat age` | Seconds since the node last reported. | > 30 s = node may have lost the control-plane connection. |
| `adaptive limit` | Per-node AIMD admission limit (only when `--adaptive-admission` is on). | Limit contracting = AIMD is shedding load due to docker pressure; recovering = pressure eased. |

The page also shows two triage summaries:

- **Long-running & queued sessions** — sandboxes and raw rollouts alive beyond
  2 hours. This is an age heuristic, **not** a failure: long-horizon rollouts and
  persistent substrate containers legitimately run for hours, and an `acquiring`
  raw rollout is queued for capacity (admission backpressure), not hung. A
  `state` column distinguishes `running` from `queued — awaiting capacity`, and
  the section does **not** flip the health banner.
- **High failure rate** — workloads where > 25 % of the last 30 min of
  rollouts failed (minimum 4 samples).

## Raw rollout detail

The raw rollout detail page shows the XRLEnv lifecycle for one
managed container:

- displayed name and labels
- image, node, container id, and status
- created, started, stopped, and destroyed timestamps
- artifact path, when the workflow supplied one with
  `xrlenv.rollout_metadata(...)`

This is the page you use for Docker SDK drop-in and
framework/harness adapter runs. The benchmark's own logs and reports
stay in the benchmark artifact directory; XRLEnv stores a pointer so
operators can jump to it.

## Template rollout detail

Template rollout pages show the sealed trajectory: steps, actions,
observations, rewards, verifier files, `coordinator.log`, and a JSONL
download. These pages remain useful for built-in templates and
advanced `EnvAdapter` users, but they are separate from the raw
container workflow.

## Image pages

Use `/images/cache` when debugging disk pressure on a node. It groups
images by state such as in use, pinned, recently used, and cold.

Use `/images/catalog` when asking whether the cluster covers a
workload's image set. It deduplicates image refs across nodes and
shows node coverage.

## Build pages and API

`/builds` lists persisted build-plan snapshots. The detail page at
`/builds/<plan_id>` shows assignment state by node and image.

The write-capable build API is under `/api/build/*`:

- `POST /api/build/apply`
- `POST /api/build/cancel`
- `POST /api/build/calibrate`
- `GET /api/build/plans/<plan_id>`

The three POST routes require an **operator**-role credential. The GET
poll route is accessible to **viewer**-role credentials. The loopback dev
escape hatch (no TokenStore configured) allows unauthenticated access to
all routes. See {doc}`admin_auth` for the full auth model.

## Refresh behavior

Auto-refresh is **off by default** (`AdminServerConfig.refresh_interval_s`
defaults to `0`). To enable it:

- **Per page:** append `?refresh=<seconds>` to any list or overview URL,
  or use the per-page refresh selector (options: 5 s / 10 s / 30 s / 60 s).
  `?refresh=0` or `?refresh=off` disables auto-refresh for that URL.
- **Globally:** set `AdminServerConfig.refresh_interval_s > 0` when
  constructing the server to make every list/overview page auto-refresh
  by default.

Detail pages (individual rollout, build plan) never auto-refresh — losing
scroll or search position on an inspection page is disruptive.

## See also

- {doc}`metrics` — Prometheus `/metrics`.
- {doc}`logs` — structured logs.
- {doc}`capacity` — capacity and image-cache signals.
- {doc}`tracing` — distributed tracing with OpenTelemetry (Jaeger, Tempo, Honeycomb).
- {doc}`/developer_guide/cli_reference` — equivalent CLI commands.

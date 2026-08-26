# Prometheus `/metrics`

The control plane exposes `/metrics` on port 9090 (default; configurable via
`--metrics-port`).

## Two views over the same data

`/metrics` serves **the same live registry two ways**, chosen by the request's
`Accept` header (HTTP content negotiation):

- **Prometheus / curl** (`Accept: text/plain`, `*/*`, or OpenMetrics) get the
  standard Prometheus text exposition — unchanged, so existing scrape configs
  keep working. OpenMetrics negotiation and gzip are preserved.
- **A browser** (`Accept: text/html`) gets a rendered **HTML dashboard**:
  metrics grouped by the categories below, with counter totals + per-label
  breakdowns, histogram count / mean / p50–p99, and current gauge values. The
  page auto-refreshes and lists every declared series — a metric with no
  samples yet shows as "no samples yet", so the page doubles as a live
  catalogue of the contract documented here.

```bash
# Scrape (raw text) — what Prometheus does:
curl -s 127.0.0.1:9090/metrics | grep xrlenv_rollouts_finished_total

# Open http://127.0.0.1:9090/metrics in a browser for the dashboard.
```

Query-string overrides (handy for scripts and for forcing a view):

| Query | Effect |
|-------|--------|
| `?format=raw` | Force the raw text exposition regardless of `Accept`. |
| `?format=html` | Force the HTML dashboard regardless of `Accept`. |
| `?refresh=N` | Dashboard auto-refresh interval in seconds (`0` disables; default 5). |

`/` 302-redirects to `/metrics`. Any other path is `404`.

```{note}
Histogram percentiles on the dashboard are **estimated** by linear
interpolation across the configured bucket edges (the same method as
Prometheus' `histogram_quantile`), not computed from raw observations. For
exact quantiles, scrape into Prometheus and query there.
```

### Relationship to the admin panel

The `/metrics` dashboard and the {doc}`admin_panel` overlap on a few aggregate
counts (active sandboxes, rollout totals, queue depth) but are **complementary,
not redundant** — they read different stores at different altitudes:

| | `/metrics` dashboard | Admin panel |
|---|---|---|
| Source | in-memory Prometheus registry | sqlite `StateStore` + JSONL sink |
| Granularity | aggregate numbers | per-entity rows (drill-down to a rollout / sandbox / trajectory) |
| Durability | **resets on control-plane restart** | authoritative, survives restarts |
| Best at | latency percentiles, throughput/admission **rates**, trends (Prometheus) | "what is the state of *this* rollout / sandbox / node, with history" |

Because the dashboard's counters reset on restart while the admin panel keeps
the authoritative on-disk history, the two can legitimately disagree right after
a restart — the dashboard's role-clarifier banner says as much and links to the
admin panel for per-entity drill-down.

## Emitted series

**Rollout lifecycle counters:**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `xrlenv_rollouts_started_total` | counter | `template` | Rollouts admitted to RUNNING. |
| `xrlenv_rollouts_finished_total` | counter | `template`, `status` | Terminal-state rollouts by status (`finished`, `truncated`, `cancelled`, `failed`). |

**Latency histograms:**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `xrlenv_step_latency_seconds` | histogram | `template`, `backend` | Wall-clock for one `Coordinator.step` call. |
| `xrlenv_sandbox_create_seconds` | histogram | `template`, `backend` | Time from create_sandbox to first observation ready. |
| `xrlenv_sandbox_destroy_seconds` | histogram | `template`, `backend` | Time spent in destroy_sandbox. |
| `xrlenv_queue_wait_seconds` | histogram | `template` | Time a rollout spent in the admission queue. |

**Liveness gauges:**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `xrlenv_sandbox_active` | gauge | `node`, `template` | Currently-running sandboxes. |
| `xrlenv_queue_depth` | gauge | `template` | Pending rollouts in the admission queue. |
| `xrlenv_raw_sessions_suspect` | gauge | — | Raw sessions currently marked `suspect`: past the liveness TTL, consumer silent, session retained. Normally that means "inside the quarantine horizon"; during a mass die-off it also includes sessions already past the horizon but still queued behind `XRLENV_RAW_LIVENESS_REAP_BATCH`. Re-read after both liveness passes each sweep, so a session marked and reaped in the same sweep never leaves the gauge reading high. A rise-then-drain pattern is consumer stalls being ridden out; a rise that does not drain is consumers actually dying. |

**Adaptive admission (emitted only when `--adaptive-admission` is on):**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `xrlenv_node_admission_limit` | gauge | `node` | AIMD-derived concurrent-acquire limit for this node. Graphing it shows the sawtooth: the limit contracts when docker-daemon p95 or error rate rises and recovers as the node stabilises. Visible on the {doc}`admin_panel` `/health` page as the "AIMD limit" column. |

**Failure / rejection counters:**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `xrlenv_sandbox_create_failed_total` | counter | `template`, `reason` | Bootstrap-phase failures by reason. |
| `xrlenv_admission_total` | counter | `result` | Admission outcomes: `admitted`, `queued`, `queue_timeout`, `cancelled_in_queue`, `rejected_full` (the queue has stopped accepting — control-plane shutdown). |
| `xrlenv_raw_liveness_suspect_total` | counter | — | Raw sessions marked `suspect` after going silent past the liveness TTL. |
| `xrlenv_raw_liveness_recovered_total` | counter | — | Suspect sessions whose consumer signalled again before the reap fired — usually inside the quarantine horizon, but also a session already past the horizon that recovered while the sweep was destroying its siblings (the reconciler re-checks candidacy before each destroy and skips it). Work that a destroy-on-TTL reaper would have thrown away. |
| `xrlenv_raw_liveness_reaped_total` | counter | — | Raw sessions force-destroyed after staying silent for the full quarantine horizon. |

## Example Prometheus alert rules

```yaml
groups:
  - name: xrlenv
    rules:
      - alert: HighFailureRate
        expr: |
          rate(xrlenv_rollouts_finished_total{status="failed"}[5m])
          / rate(xrlenv_rollouts_finished_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "More than 10% of rollouts failing"

      - alert: AdmissionQueueDeep
        expr: xrlenv_queue_depth > 100
        for: 2m
        annotations:
          summary: "Admission queue depth > 100 — cluster may be undersized"

      - alert: AuthDenials
        expr: rate(xrlenv_audit_events_total{event="auth.denied"}[5m]) > 0
        annotations:
          summary: "Auth denials detected — check `xrlenv audit --kind auth.denied` for the per-row detail"
```

## See also

- {doc}`/developer_guide/cli_reference` — `xrlenv up --metrics-port` flag.
- {doc}`logs` — structured JSON log events.
- {doc}`admin_panel` — web-based cluster view with the same ring-buffer data.

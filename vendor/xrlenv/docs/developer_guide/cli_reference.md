# Operator CLI reference

All subcommands read `--state-db` (default `~/.xrlenv/state.db`) and
`--runs-root` (default `~/.xrlenv/runs`) as global flags.

(log-rotation)=

## `xrlenv up` logging flags

Logging is configured on every invocation, but only the long-lived
`xrlenv up` daemon needs to tune it, so the six flags below live on the
`up` subcommand (one-shot commands use the defaults: `INFO` / `auto` /
no file). They go **after** `up`, alongside the other `up` flags.

| Flag | Default | Description |
|------|---------|-------------|
| `--log-level LEVEL` | `INFO` | Minimum level admitted to all handlers (file + stdout). Valid values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `--log-format {pretty,json,auto}` | `auto` | Console (stdout) formatter. `auto` uses `pretty` on a TTY, `json` when piped. Does **not** affect the rotating file — the file is always JSON. |
| `--log-file PATH` | *(none)* | Write the full JSON firehose to PATH using size-based rotation instead of stdout. When set, stdout is floored at `WARNING` so a supervisor or Slurm `--output` capture stays small but still shows crashes. The path is stable across restarts; rotated backups get `.1` / `.2` / … suffixes. Without this flag, the full firehose goes to stdout (original behavior, unchanged). |
| `--log-max-bytes N` | `52428800` (50 MiB) | Maximum bytes per rotating file before rollover. Only meaningful with `--log-file`. |
| `--log-backup-count K` | `10` | Number of rotated backup files retained. Disk ceiling = `--log-max-bytes × (K + 1)` ≈ 500 MiB with defaults. Only meaningful with `--log-file`. |
| `--stdout-log-level LEVEL` | `--log-level` (no file) / `WARNING` (with file) | Override the minimum level echoed to stdout independently of the rotating file. Set to `INFO` to mirror the full firehose to stdout alongside `--log-file`. |

**Example — long-running deployment with bounded log file:**

```bash
xrlenv up \
    --grpc-host 0.0.0.0 \
    --grpc-port 50051 \
    --log-file ~/.xrlenv/xrlenv-up.log \
    --log-max-bytes 52428800 \
    --log-backup-count 10
```

The rotating file at `~/.xrlenv/xrlenv-up.log` is always JSON (one
structured JSON envelope per line). Tail it with:

```bash
tail -f ~/.xrlenv/xrlenv-up.log | jq -r '[.ts, .level, .event] | @tsv'
```

See {doc}`/observability/logs` for the full log format and event catalog.

---

## `xrlenv up`

Boot the control plane and block until SIGINT.

```bash
xrlenv up \
    --grpc-host 0.0.0.0 \
    --grpc-port 50051 \
    --metrics-port 9090 \
    --admin-port 8080 \
    --retention-days 14
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--grpc-host` | `127.0.0.1` | Address the gRPC endpoint binds to. Use `0.0.0.0` to accept remote nodes. |
| `--grpc-port` | `50051` | gRPC port. |
| `--metrics-host` | `127.0.0.1` | Prometheus `/metrics` bind address. |
| `--metrics-port` | `9090` | Metrics port; pass `0` to disable. |
| `--admin-host` | `127.0.0.1` | Admin panel bind address. See security notes below. |
| `--admin-port` | `8080` | Admin panel port; pass `0` to disable. |
| `--admin-allow-public` | off | Required when `--admin-host` is non-loopback. Without this flag the process refuses to start. |
| `--retention-days` | `14` | Days to keep per-rollout run directories before GC. |
| `--admin-nodes-yaml` | `None` | Path to `nodes.yaml` shown in the `/nodes` admin view. |
| `--admin-rollout-page-size` | `32` | Default rows per page in the `/rollouts` admin view (`32`, `64`, `128`, or `256`). |
| `--max-runs-per-task` | `4` | Per-node fairness cap on rollouts sharing a `task_key` (anti-affinity). Set lower (e.g. `2`) to spill same-`task_key` rollouts onto another node — useful for acceptance smokes that want deterministic per-node distribution. |
| `--adaptive-admission` | off | Enable the health-derived adaptive admission controller. Each node's concurrent-rollout limit contracts when its docker-run latency or error rate degrades, and expands when health holds. Off by default (static estimator applies). See {ref}`adaptive-admission`. |
| `--aimd-initial-limit N` | `16` | Slow-start seed: a newly-connected node's admission limit before any health data is observed. Only used with `--adaptive-admission`. |
| `--aimd-p95-threshold-s SECONDS` | `60.0` | A node whose `docker run` create p95 latency exceeds this is a "bad" tick and contracts its limit. Only used with `--adaptive-admission`. |
| `--aimd-max-limit N` | `64` | Runaway guardrail: additive-increase never grows a node's limit past this. Not a resource calculation — the real bound is node health. Only used with `--adaptive-admission`. |
| `--audit-retention-days N` | `30` | Delete `audit` table rows older than N days. Pass `0` to disable. The `audit` table is the dominant `state.db` growth source — one row per authentication event — so a tight window keeps the database small. See [State-store retention](#state-store-retention). |
| `--raw-rollout-retention-days N` | `14` | Delete terminal `raw_rollouts` rows older than N days. Pass `0` to disable. The `raw_rollouts` table is tiny (~a few MB for tens of thousands of rows). When the janitor prunes a row it folds the tally into the durable `owner_rollout_lifetime` table, so `/users` cumulative totals are preserved across GC — only per-rollout drill-down detail is bounded by this window. Note: rows pruned before lifetime tracking was enabled are not backfilled. See [State-store retention](#state-store-retention). |
| `--events-retention-days N` | `14` | Delete `events` table rows older than N days. Pass `0` to disable. |

**Security note on `--admin-host`:** When bound to localhost with no
TokenStore configured, the admin panel requires no credentials. Binding
to a non-loopback address requires both `--admin-allow-public` **and** at
least one issued viewer or operator token — a public bind with no credentials
raises `AdminBindError` at startup. The SSH-tunnel approach avoids credentials
entirely and is still recommended for most deployments:

```bash
# On your local machine:
ssh -L 8080:127.0.0.1:8080 <control-plane-vm>
# Then open http://127.0.0.1:8080 locally.
```

See {doc}`/observability/admin_auth` for the two-tier role model and the
browser login flow.

---

## `xrlenv tokens issue`

Issue a bearer token for a role. Tokens are written to
`~/.xrlenv/secrets/<role>.token` (mode 0600).

```bash
# Issue a node token (used in the systemd EnvironmentFile on each node):
xrlenv tokens issue node

# Issue a consumer token for SDK and Docker SDK drop-in workflows:
xrlenv tokens issue consumer

# Issue a viewer token (read-only admin panel access; token carries read_ prefix):
xrlenv tokens issue viewer

# Issue an operator token (full admin access; token carries write_ prefix):
xrlenv tokens issue operator
```

The bearer is printed once and never stored in plaintext. Copy it before the
command exits.

**Roles and scopes:**

| Role | Scope | Can do | Token prefix |
|------|-------|--------|--------------|
| `node` | `node.report` | Register node, send heartbeats, accept sandbox commands, write trajectory chunks | _(none)_ |
| `consumer` | `consumer.rollout` | Start, step, cancel, and replay workflow rollouts; call `/healthz` | _(none)_ |
| `viewer` | `admin.read` | Browse all admin panel GET routes; poll build plan status | `read_` |
| `operator` | `operator.admin` | All viewer access plus `POST /api/build/*` write routes and future destructive actions | `write_` |

**Per-user tokens (`--owner` / `--name`):**

In a shared cluster each tenant should have their own token so rollouts are
stamped with their `owner_id` and fair-share scheduling applies correctly.
Add `--owner <id>` to mint a per-user token (accepted for `consumer`,
`viewer`, and `operator`; rejected for `node`). `--name <label>` attaches a
human-readable label shown in `xrlenv tokens list` — operator convenience
only, no effect on privileges.

```bash
# Per-user consumer token for tenant "alice":
xrlenv tokens issue consumer --owner alice

# Per-user viewer token with an optional display name:
xrlenv tokens issue viewer --owner bob --name "Bob (read-only)"
```

Example output for a per-user issue:

```
issued consumer token for owner=alice (token_id=aaaaaaaaaaaa)
  recorded at: ~/.xrlenv/secrets/users.json (hashed; plaintext not stored)
  scope:       consumer.rollout
  raw token:   <bearer>

Copy the token now — it will not be shown again.
```

The bearer's SHA-256 is appended to `users.json` (hashed at rest — plaintext
never persisted). Many per-user tokens of the same role coexist. Omitting
`--owner` keeps the legacy single shared role-token path unchanged
(`owner_id="default"`).

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--owner ID` | _(none — shared role token)_ | Mint a per-user token for this tenant id. The control plane stamps every rollout/session this token starts with `owner_id=<id>`. Not valid with `role=node`. |
| `--name LABEL` | _(none)_ | Human label for the owner, shown in `xrlenv tokens list`. Only meaningful with `--owner`. |
| `--secrets-root PATH` | `~/.xrlenv/secrets` | Override the secrets directory. |

Tokens are never logged. Log lines carry only the first 6 characters of the
token's SHA-256 digest as an identity hint. See {doc}`/developer_guide/security` for the full
security model.

---

## `xrlenv tokens rotate`

Replace the active token for a role. The new token is written to
`<secrets-root>/<role>.token` (mode `0600`). A running `xrlenv up`
hot-reloads on the next RPC; no restart needed.

```bash
# Immediate cutover — the prior token is invalid on the next RPC:
xrlenv tokens rotate node

# Grace-window cutover — keep the prior token valid for 24 hours
# while nodes are being updated:
xrlenv tokens rotate node --grace 24h

# Use a custom secrets root:
xrlenv tokens rotate consumer --secrets-root /etc/xrlenv/secrets
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--grace DURATION` | none (immediate) | Keep the prior token valid for this window. Formats: plain seconds (`3600`), `5m`, `2h`, `1d`. Without this flag, the prior token is rejected immediately. |
| `--secrets-root PATH` | `~/.xrlenv/secrets` | Directory holding token files. |

Immediate cutover is the security default. Use `--grace` only for
deployment-rollover scenarios where nodes cannot all be updated
atomically. See {doc}`/developer_guide/tokens` for the full rotation
workflow and on-disk file layout.

---

## `xrlenv tokens revoke`

Permanently revoke a specific token by its ID or a unique prefix of its
SHA-256 digest. Appends a record to `<secrets-root>/revoked.json`;
hot-reloaded by a running `xrlenv up` on the next RPC.

```bash
# Revoke by full 12-character token_id:
xrlenv tokens revoke aaaaaaaaaaaa

# Revoke by the 6-character digest_hint from an audit log entry:
xrlenv tokens revoke aaaaaa

# Custom secrets root:
xrlenv tokens revoke aaaaaa --secrets-root /etc/xrlenv/secrets
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--secrets-root PATH` | `~/.xrlenv/secrets` | Directory holding `revoked.json`. |

Exit codes: `0` success, `1` no matching token found, `2` prefix
ambiguous or shorter than 6 characters.

The command is idempotent. `revoked.json` is not secret — it contains
only SHA-256 prefixes. See {doc}`/developer_guide/tokens` for details.

---

## `xrlenv tokens list`

Show active token state per role. Never prints raw bearer bytes — only
the `token_id` (12-char SHA-256 prefix), the `digest_hint` (6-char
prefix seen in audit logs), grace window state, and status.

```bash
xrlenv tokens list

# Custom secrets root:
xrlenv tokens list --secrets-root /etc/xrlenv/secrets
```

Example output:

```
tokens loaded from ~/.xrlenv/secrets:
  node      active   token_id=aaaaaaaaaaaa digest_hint=aaaaaa owner=default
  consumer  active   token_id=bbbbbbbbbbbb digest_hint=bbbbbb owner=default
  operator  grace    token_id=cccccccccccc digest_hint=cccccc remaining=4980s
per-user tokens (multi-user):
  consumer  user     token_id=aaaaaaaaaaaa digest_hint=aaaaaa owner=alice
  viewer    user     token_id=dddddddddddd digest_hint=dddddd owner=bob (Bob (read-only))
```

The per-user section appears only when per-user tokens have been issued via
`xrlenv tokens issue --owner`. Each row shows: role, state (`user` or
`revoked`), 12-character `token_id`, 6-character `digest_hint` (matches
audit-log entries), `owner`, and optional display name.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--secrets-root PATH` | `~/.xrlenv/secrets` | Directory to read token files from. |

---

## `xrlenv nodes`

List rostered nodes (from `nodes.yaml`) cross-referenced against active
sandboxes in `state.db`.

```bash
xrlenv nodes --nodes-yaml nodes.yaml --format text
```

Example output:

```
NODE         STATUS     SANDBOXES  LAST_SEEN
gcp-1        connected  3/8        2s ago
aws-1        connected  1/8        1s ago
local-laptop connected  0/4        0s ago
```

---

## `xrlenv rollouts`

List rollouts with optional filters.

```bash
# All running rollouts:
xrlenv rollouts --status running

# Finished rollouts from a specific template in the last 30 minutes:
xrlenv rollouts --status finished --template terminal-bench-2 --since 30m

# JSON output (for scripting):
xrlenv rollouts --format json | jq '.[] | select(.final_reward > 0.5)'
```

**Filters:**

| Flag | Example | Description |
|------|---------|-------------|
| `--status` | `running`, `finished`, `failed`, `cancelled`, `truncated` | Filter by terminal or transient state. For raw-container sessions, also accepts `acquiring`, `released`, `reaped` (see note below). |
| `--template` | `terminal-bench-2` | Exact template name match |
| `--since` | `5m`, `2h`, `1d` | Only rollouts created within this window |
| `--format` | `text` (default) or `json` | Output format |

`reaped` is a raw-container status, not a failure. It means the platform tore
the session down on purpose and recorded why, and the teardown completed
cleanly. Any teardown carrying a reason seals `reaped`: the wall-clock
`session_deadline_s` sweep, the consumer-liveness quarantine sweep, a group
teardown (`terminate_raw_group`), and the orphan sweep sealing a container the
node reaped on its own (disk guard, OOM). The `error` column carries the
specific cause. A reap whose teardown is *not* node-confirmed (it raised, or the
destroy timed out) seals nothing at all — the row stays `running` and the
reconciler re-attempts on its next sweep.

---

## `xrlenv replay`

Print the sealed trajectory for a rollout, reading from the local run directory.

```bash
xrlenv replay abc123 --format json
```

The replay command reconstructs the `Trajectory` object from `trajectory.jsonl`
and pretty-prints it. For multi-node setups where the trajectory lives on a
remote node, use the `/rollouts/<id>` admin panel view (which fetches the
trajectory over the node bidi stream) or `xrlenv attach`.

See {doc}`/observability/admin_panel` for a full description of per-rollout artifacts.

---

## `xrlenv events`

Query the rollout-lifecycle events log (`rollout.start`,
`rollout.finish`, `rollout.fail`, `sandbox.create`, ...). Auth events
live in a separate table; use `xrlenv audit` for those.

```bash
# Recent events:
xrlenv events --since 5m

# Events for a specific rollout:
xrlenv events --rollout abc123
```

---

## `xrlenv audit`

Query the audit log: `auth.denied`, `template.registered`,
`template.image_unpinned`, and (when enabled) `auth.token_used`. Separate
table from `xrlenv events` so auth records can use an independent retention
window.

**Successful-auth auditing is off by default.** `auth.token_used` rows (one
per authenticated RPC) were ~99.9% of the audit table at scale and churned the
SQLite WAL on every call. By default only `auth.denied` rows — rejected
authentication attempts — are written. To restore per-RPC success auditing set
`XRLENV_AUDIT_AUTH_SUCCESS=1` in the control-plane environment.

```bash
# All audit entries from the last 10 minutes:
xrlenv audit --since 10m

# Check for auth denials (should be 0 in a healthy cluster):
xrlenv audit --kind auth.denied --since 1h --format json

# Confirm node bidi streams attached after deploy (requires
# XRLENV_AUDIT_AUTH_SUCCESS=1 on the control plane):
xrlenv audit --kind auth.token_used --role node --since 5m
```

**Filters:** `--since DURATION` (e.g. `5m`, `2h`), `--kind <event-kind>`,
`--role <node|consumer|operator>`, `--format text|json`.

```{note}
With `XRLENV_AUDIT_AUTH_SUCCESS=1` on the control plane, node bidi-stream
attach events emit one `auth.token_used` row per VM — useful for verifying
that all nodes authenticated after a deploy. Without it, use `xrlenv nodes`
or the `/nodes` admin page to confirm node connectivity instead.
```

---

## `xrlenv tail`

Follow `trajectory.jsonl` live as steps are appended. Useful for monitoring
a long-running rollout in real time.

```bash
xrlenv tail abc123
```

Output streams one JSON line per step as the rollout progresses.
Press Ctrl+C to stop.

See {doc}`/observability/admin_panel` for the full `trajectory.jsonl` format.

---

## `xrlenv attach`

Read-only inspection of a running rollout: print the current rollout metadata
snapshot (from `state.db`) then follow `coordinator.log`.

```bash
xrlenv attach abc123
```

`coordinator.log` contains per-rollout debug lines written by the coordinator:
step timing, reward mode calls, admission queue events, and error details not
surfaced in the structured log. See {doc}`/observability/admin_panel` for details.

---

## `xrlenv images`

List Docker images cached on this host, cross-referenced with the operator pin
list.

```bash
xrlenv images --pin-file ~/.xrlenv/image-pins.yaml --format text
```

Example output:

```
IMAGE                                        SIZE     PINNED  LAST_USED
xrlenv/hello-shell:0.1                       312 MB   no      2 min ago
xrlenv/terminal-bench-2:0.1                  1.2 GB   yes     5 min ago
alexgshaw/build-cython-ext:20251031          820 MB   no      12 min ago
```

The `PINNED` column reflects the operator pin list. Pinned images are excluded
from LRU eviction even under disk pressure.

---

(xrlenv-images-evict)=

## `xrlenv images evict`

Remove an image from every connected node's cache so the next acquire
re-pulls the current registry digest. Use this after rebuilding and
re-pushing an image under the same tag; without eviction, `ensure_present`
short-circuits on local presence and nodes silently serve stale bytes.

```bash
xrlenv images evict <image_ref> \
    --connect-host <admin-host> \
    [--connect-port 8080] \
    [--operator-token <token>] \
    [--force]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--connect-host HOST` | — | **Required.** Admin host of a running `xrlenv up`. |
| `--connect-port PORT` | `8080` | Admin port. |
| `--operator-token TOKEN` | `$XRLENV_OPERATOR_TOKEN` or `~/.xrlenv/secrets/operator.token` | Bearer token with the `operator.admin` scope. |
| `--force` | off | Evict even images that are currently in use by a running container. Without this flag, in-use images are skipped to avoid disrupting live rollouts. |

**Per-node statuses:**

| Status | Meaning | Exit code contribution |
|--------|---------|----------------------|
| `evicted` | Removed successfully; reports reclaimed bytes. | 0 |
| `absent` | Not on this node — nothing to evict. | 0 |
| `in_use` | Skipped (running container holds it). Re-run with `--force` to override. | 0 |
| `failed` | Node error or unreachable. | 1 |

The command exits non-zero only when at least one node returned `failed`.
`absent` and `in_use` are successful no-ops.

The `image_ref` is matched **registry-agnostically**: a bare `repo:tag`
matches the registry-qualified `host:5011/repo:tag` the node actually
holds, so you can pass the same ref format your consumer config uses.

**Example:**

```bash
# Evict after rebuilding + re-pushing the webarena-infinity substrate image.
xrlenv images evict xrlenv-webarena-infinity/substrate:dev \
    --connect-host <control-plane-host>
```

See {doc}`/technical_details/images/cache_eviction` for the full operator
eviction guide and the relationship to the freshness model.

---

## `xrlenv warmup`

Pre-pull images onto this node before the next run, so the first
container using each image is not blocked by image pulls.

```bash
# Warm specific image refs:
xrlenv warmup xrlenv/terminal-bench-2:0.1 alexgshaw/build-cython-ext:20251031

# Set a per-image pull timeout (default 600 s):
xrlenv warmup ghcr.io/myorg/big-image:latest --deadline 900
```

For workloads with many per-task images, pre-pull by task subset:

```bash
# Pull the three scaffold task images:
xrlenv warmup \
    alexgshaw/build-cython-ext:20251031 \
    alexgshaw/extract-tarball:20251031 \
    alexgshaw/sqlite-schema:20251031
```

---

## `xrlenv build apply`

Push a build plan onto the cluster (or a local single-host runtime). The
full mechanics — plan schema, FFD bin-packing, fill-missing semantics,
etc. — are covered in {doc}`/technical_details/images/build_plan`. This
entry covers the flag reference.

```bash
# Common case: apply a plan against a running cluster.
xrlenv build apply \
    --plan build_plan.yaml \
    --connect-host 127.0.0.1

# Large cluster: saturate idle nodes by setting coordinator fan-out to
# roughly num_nodes × pull_concurrency_ceiling (3 nodes × 64 ≈ 192).
xrlenv build apply \
    --plan build_plan.yaml \
    --connect-host 127.0.0.1 \
    --concurrency 192
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--plan PATH` | — | **Required.** Path to the `build-plan.yaml` file. |
| `--connect-host HOST` | *(local-only mode)* | Admin host of a running `xrlenv up`. Required when a control plane is running; omitting it while nodes are attached raises an error. |
| `--connect-port PORT` | `8080` | Admin port. |
| `--operator-token TOKEN` | `$XRLENV_OPERATOR_TOKEN` or `~/.xrlenv/secrets/operator.token` | Bearer token with the `operator.admin` scope. |
| `--force` | off | Re-dispatch every entry regardless of prior plan status. |
| `--fill-missing` | off | Re-dispatch only entries no connected node currently has; re-anchor the rest. Mutually exclusive with `--force` and `--eager`. |
| `--eager` | off | Strict-mode FFD: reject with `InsufficientCapacity` if the full plan does not fit upfront. |
| `--skip-if-present` | off | Short-circuit the node-side builder for entries whose image is already tagged locally. |
| `--concurrency N` | `$XRLENV_BUILD_CONCURRENCY` (default 32) | Per-invocation coordinator fan-out: max in-flight image dispatches across the cluster for this apply. Overrides the process-level default with no control-plane restart. Set to roughly `num_nodes × pull_concurrency_ceiling` to saturate idle nodes (e.g. `3 × 64 = 192`). Must be a positive integer (≥ 1). |

`--force`, `--fill-missing`, and `--eager` are mutually exclusive.
`--skip-if-present` is compatible with `--fill-missing` but is overridden
by `--force`. `--concurrency` is independent of all other flags.

---

(cli-build-push)=
## `xrlenv build push`

Control-plane-orchestrated build-and-push for plans whose images must be built
from source (Dockerfile) and stored in the cluster's private registry.
`build push` is the cluster-scale, control-plane-native replacement for
the old Slurm-based distributed build workflow.

**How it differs from `build apply`.**

| | `build apply` | `build push` |
|---|---|---|
| Source types | all (`git`, `tarball`, `registry`, `local`) | `git` and `tarball` only |
| Output | image tagged locally on each node | image pushed to `--registry`; digest returned |
| Registry needed | no | yes — `--registry <host:port>` is required |
| Use case | warm nodes before a run | populate the private registry before warming |

`registry` and `local` entries are rejected by `build push` — they are already
in a registry or on-disk and do not need to be built and pushed.

**Build-once, resumable.** Before submitting any build work, the control plane
asks each node to perform a registry `HEAD` check on the manifest. If the ref is
already in the registry it is skipped — re-runs are cheap and interrupted runs
resume where they left off. Overlapping dispatches never double-push the same
ref.

**Size-aware sharding.** The coordinator distributes entries across nodes using
size-balanced (LPT-greedy) assignment: the largest images go first to the
least-loaded node, balanced by `size_hint_bytes` from the plan. Unlike
`build apply`, there is no disk-fit constraint — a plan can exceed total cluster
disk capacity because images are pushed to the shared registry and become
evictable on-node immediately. Run `xrlenv build calibrate` first to measure
accurate sizes (recommended for plans with heavy-tailed images such as CUDA
environments).

```bash
# Populate the private registry from a large build plan.
xrlenv build push \
    --plan xrlenv_plugins/benchmarks/seta/build_plan_1376_full.yaml \
    --registry <registry-host>:5011 \
    --connect-host <admin-host>

# Preview the per-node shard assignment without building anything.
xrlenv build push \
    --plan build_plan.yaml \
    --registry <registry-host>:5011 \
    --connect-host <admin-host> \
    --dry-run

# Force-rebuild all entries, even those already present in the registry.
xrlenv build push \
    --plan build_plan.yaml \
    --registry <registry-host>:5011 \
    --connect-host <admin-host> \
    --force
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--plan PATH` | — | **Required.** Path to the `build-plan.yaml`. Only `git` and `tarball` source entries are processed; `registry` and `local` entries are rejected. |
| `--registry HOST:PORT` | — | **Required.** Private registry to push built images to, e.g. `<registry-host>:5011`. Each node pushes directly to this address. |
| `--connect-host HOST` | — | **Required.** Admin host of a running `xrlenv up`. |
| `--connect-port PORT` | `8080` | Admin port. |
| `--operator-token TOKEN` | `$XRLENV_OPERATOR_TOKEN` or `~/.xrlenv/secrets/operator.token` | Bearer token with the `operator.admin` scope. |
| `--concurrency N` | `$XRLENV_BUILD_CONCURRENCY` (default 32) | Max in-flight image dispatches across the cluster. Set to roughly `num_nodes × build_concurrency_per_node` to saturate idle nodes. |
| `--dry-run` | off | Print the per-node shard assignment and exit without dispatching any builds. |
| `--force` | off | Rebuild and repush every entry, even refs already present in the registry. |
| `--build-tarball-max-bytes N` | platform default | Override the maximum tarball payload size for build context uploads. |

After `build push` completes, use `xrlenv build apply` (with a
registry-source plan) or point template `image_ref` values at
`<registry-host>:5011/<ref>` with `image_pin_mode: registry_digest` to have
the control plane pin the pushed digests at template-register time.

See {doc}`/deploy/multi_node_deployment/private_registry` for the end-to-end
private registry workflow, including how to stand up the registry server and
configure worker nodes to pull from it.

---

## `xrlenv build cancel`

Cancel an in-flight build plan.

```bash
xrlenv build cancel --plan <id-or-prefix> --connect-host <admin-host>
```

`<id-or-prefix>` accepts the full SHA-256 `plan_id` or a unique prefix
(≥ 4 chars — the 12-char short id from `/builds` works). Cancel is
idempotent and sticky: it cannot be overwritten back to `completed`.

See {doc}`/technical_details/images/build_plan` for cancel semantics and
what happens to in-flight `docker build` processes on remote nodes.

---

## `xrlenv build status`

Show the most recent build plan's status, or a specific plan's per-
assignment rollup.

```bash
# Latest applied plan:
xrlenv build status --connect-host 127.0.0.1

# Specific plan (full plan_id or unique ≥4-char prefix):
xrlenv build status --plan <plan-id-or-prefix> --connect-host 127.0.0.1
```

---

## `xrlenv build calibrate`

Probe the live cluster for layer-share-aware unique sizes and rewrite
`placement.size_hint_bytes` in the plan YAML with measured values.

```bash
xrlenv build calibrate \
    --plan build-plan.yaml \
    --output build-plan.calibrated.yaml \
    --connect-host 127.0.0.1
diff build-plan.yaml build-plan.calibrated.yaml  # review before promoting
mv build-plan.calibrated.yaml build-plan.yaml
```

Calibration changes the `plan_id` (because the size hints are part of
the content-addressed plan body). See
{doc}`/technical_details/images/build_plan` for the full calibration
workflow and conditional-determinism caveats.

---

## `xrlenv bootstrap`

Install the `xrlenv-node` daemon on a freshly provisioned VM. Intended
for operators setting up a multi-node cluster; most operators reach this
through the wrapper scripts (`deploy/bootstrap-gcp.sh` /
`deploy/bootstrap-aws.sh`) rather than invoking it directly.

```bash
# Preview the full install plan without touching the host:
sudo xrlenv bootstrap --target gcp --control-plane host:50051 --dry-run

# Full install on a GCP VM from a local repo checkout:
sudo xrlenv bootstrap --target gcp \
    --control-plane host:50051 \
    --xrlenv-repo /path/to/xrlenv
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--target {gcp,aws,linux-generic}` | — | **Required.** Install branch. `gcp` uses the upstream Docker apt repo and reads node-id from the GCP metadata service. `aws` branches on `/etc/os-release` (dnf for Amazon Linux 2023/RHEL/Fedora; apt for Ubuntu/Debian) and reads node-id from IMDSv2. `linux-generic` skips cloud-metadata auto-detect; `--node-id` must be supplied. |
| `--control-plane host:port` | `$XRLENV_CONTROL_PLANE` | Address the node will dial outbound to reach the control plane. |
| `--node-id ID` | auto | Stable node identifier. Resolved in order: this flag → `$XRLENV_NODE_ID` → cloud metadata service (gcp/aws only). |
| `--target-os {amzn,rhel,fedora,ubuntu,debian}` | auto | Override the `/etc/os-release` probe. Useful on custom AMIs. |
| `--xrlenv-wheel PATH` | — | Install xrlenv from a pre-built wheel (production path). |
| `--xrlenv-repo PATH` | — | Install xrlenv from a local checkout (non-editable). Default used by the bash wrappers. |
| `--xrlenv-version VER` | — | Install xrlenv from PyPI at the given version. |
| `--runtime-user USER` | `xrlenv` | System user that owns the node daemon process. |
| `--install-root PATH` | `/opt/xrlenv` | Root for the venv and install tree. |
| `--skip-operator-docker-group` | off | Skip adding `$SUDO_USER` to the `docker` group (for headless CI environments). |
| `--dry-run` | off | Print the full install plan — every step and every shell command — without touching the host. Run this first on a new VM. |

The bootstrap is idempotent. Each step checks a `skip_if` predicate
(package already installed, user exists, directory present) so re-running
on an already-bootstrapped host is safe.

When `$XRLENV_NODE_TOKEN` is set in the environment, the bootstrap writes
it into a mode-0600 systemd drop-in at
`/etc/systemd/system/xrlenv-node.service.d/10-token.conf`, separate from
the world-readable `/etc/xrlenv/node.env`.

See {doc}`/deploy/multi_node_deployment/runbook` for the full node-bootstrap
workflow, including the dry-run flow and install-source matrix.

---

(fairshare-cmd)=

## `xrlenv fairshare`

Inspect and tune the live multi-user fair-share policy. Changes are written
to `state.db` and picked up by the control plane on its next admission drain
(seconds) — no restart required, and running jobs are not affected.

### `xrlenv fairshare show`

Print the current fair-share policy and per-owner usage.

```bash
xrlenv fairshare show
```

Example output (fair-share enabled, two configured owners):

```
fair-share: ENABLED  default_cap=16
per-owner:
  alice                running=6  effective_cap=16
  bob                  running=2  effective_cap=4  owner_cap=4
  carol                running=12  effective_cap=uncapped  UNCAPPED
  (`--default-cap` applies to owners without an override; `--owner ... --cap`
   overrides one owner; `--uncap` bypasses fair-share for one owner. Real
   cluster resources are still enforced by the scheduler.)
```

When fair-share is disabled the output reads `fair-share: DISABLED` and all
owners run uncapped.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--state-db PATH` | `~/.xrlenv/state.db` | Path to `state.db` (use when the control plane runs with a non-default path). |

### `xrlenv fairshare set`

Tune the fair-share policy live. Lowering a cap or blocking an owner stops
only **new** admissions; rollouts already running are unaffected. Uncapping an
owner bypasses fair-share for that owner, but scheduler/node resources still
apply.

```bash
# Enable fair-share with a default cap of 16 concurrent sandboxes per owner:
xrlenv fairshare set --default-cap 16

# Override alice to allow up to 32 concurrent sandboxes:
xrlenv fairshare set --owner alice --cap 32

# Override bob down to 4 concurrent sandboxes:
xrlenv fairshare set --owner bob --cap 4

# Let carol bypass fair-share caps entirely:
xrlenv fairshare set --owner carol --uncap

# Return carol to the default cap:
xrlenv fairshare set --owner carol --recap

# Block alice (stop new admissions; running jobs continue):
xrlenv fairshare set --owner alice --block

# Unblock alice:
xrlenv fairshare set --owner alice --unblock

# Remove all overrides for bob (back to default cap, not blocked):
xrlenv fairshare set --clear-owner bob

# Turn fair-share off (all owners run uncapped):
xrlenv fairshare set --disable
```

**Global options** (affect all owners):

| Flag | Default | Description |
|------|---------|-------------|
| `--default-cap N` | _(none)_ | Default concurrent-sandbox cap for each owner. Enables fair-share. Must be ≥ 1. Mutually exclusive with `--disable`. |
| `--disable` | off | Turn fair-share off; all owners run uncapped. Mutually exclusive with `--default-cap`. |

**Per-owner options** (require `--owner`):

| Flag | Default | Description |
|------|---------|-------------|
| `--owner ID` | _(none)_ | Tenant id to configure. Required when using any of the flags below. |
| `--cap N` | _(none)_ | Set an owner-specific concurrent-sandbox cap. Must be ≥ 1. |
| `--uncap` | off | Bypass fair-share caps for this owner. Scheduler/node resources still apply. Mutually exclusive with `--cap`, `--recap`, and `--block`. |
| `--recap` | off | Return this owner to the default cap and clear uncapped/blocked state. Mutually exclusive with `--cap` and `--uncap`. |
| `--block` | off | Stop new admissions for this owner. Running jobs keep going. Mutually exclusive with `--unblock` and `--uncap`. |
| `--unblock` | off | Resume admissions for this owner. Mutually exclusive with `--block`. |
| `--clear-owner ID` | _(none)_ | Remove all overrides for the named tenant (back to default cap, not uncapped, not blocked). Accepts the tenant id as the flag's value, not `--owner`. |
| `--state-db PATH` | `~/.xrlenv/state.db` | Path to `state.db`. |

See {doc}`/deploy/multi_tenancy` for the full fair-share narrative and examples.

---

(state-store-retention)=

## State-store retention

`state.db` contains several append-only tables that grow without bound if left unchecked. A background janitor (`StateRetentionJanitor`) hard-deletes rows past their window at startup and every 24 hours. The per-table windows are set with `xrlenv up` flags (see above); run-dir artifact GC is controlled separately by `--retention-days`.

**Retention windows and rationale:**

| Table | Default window | Why |
|---|---|---|
| `audit` | 30 days | The dominant growth source: one row per authenticated RPC. With successful-auth auditing off (the default), growth is much lower, but a tight window is still recommended. |
| `raw_rollouts` | 14 days | Tiny table (~a few MB for tens of thousands of rows). The janitor folds pruned rows into `owner_rollout_lifetime` before deleting them, so `/users` cumulative totals survive GC (though rows pruned before lifetime tracking was enabled are not backfilled). Only per-rollout drill-down is bounded by this window. |
| `events` | 14 days | Rollout-lifecycle events log (`rollout.start`, `rollout.finish`, etc.). |

**Typical production tuning** — the committed prod and dev control-plane launchers (`slurm_scripts/generated/prod_xrlenv_control.sh`, `slurm_scripts/generated/dev_xrlenv_control.sh`) use:

```bash
--audit-retention-days 7 --raw-rollout-retention-days 180
```

`audit=7` keeps the audit table tight (with successful-auth auditing off, the audit table barely grows, and a 7-day window is enough to catch auth anomalies). `raw_rollouts=180` keeps six months of individually browsable rollout detail — the `/users` cumulative totals are preserved by `owner_rollout_lifetime` regardless of this window (no backfill for rows pruned before lifetime tracking was enabled), but a longer window extends how far back operators can drill into individual records.

```{note}
`DELETE` frees pages for reuse inside SQLite but does not shrink the file on disk. Run `xrlenv db vacuum` (with the control plane stopped) to return freed space to the filesystem.
```

---

## `xrlenv db prune`

Hard-delete `state.db` rows past their retention window on demand. The control plane runs this automatically every 24 hours; this command triggers it immediately — for example, before running `xrlenv db vacuum` to reclaim space.

```bash
# Prune using the same defaults as the janitor:
xrlenv db prune

# Override per-table windows for this run:
xrlenv db prune \
    --audit-retention-days 7 \
    --raw-rollout-retention-days 180 \
    --events-retention-days 14
```

The command is safe to run while the control plane is active (deletes are batched; the WAL keeps readers unblocked), though running it while quiescent avoids contending with the live writer.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--state-db PATH` | `~/.xrlenv/state.db` | Path to `state.db`. |
| `--audit-retention-days N` | `30` | Delete `audit` rows older than N days. Pass `0` to disable. |
| `--events-retention-days N` | `14` | Delete `events` rows older than N days. Pass `0` to disable. |
| `--raw-rollout-retention-days N` | `14` | Delete terminal `raw_rollouts` rows older than N days. Pass `0` to disable. |

Example output:

```
pruned 1240 expired row(s): audit=1200 events=40 raw_rollouts=0
note: DELETE frees pages for reuse but does not shrink the file;
run `xrlenv db vacuum` (control plane stopped) to reclaim disk.
```

---

## `xrlenv db vacuum`

`VACUUM` `state.db` to return freed pages to the filesystem. Run this **with the control plane stopped** — `VACUUM` requires exclusive access and fails with "database is locked" if `xrlenv up` holds the database.

The command checkpoints the WAL first, then rewrites the file. Use it after a large `xrlenv db prune` sweep to reclaim disk space:

```bash
# Stop the control plane, then:
xrlenv db vacuum

# With a non-default state.db path:
xrlenv db vacuum --state-db /opt/xrlenv/state.db
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--state-db PATH` | `~/.xrlenv/state.db` | Path to `state.db`. |

---

## See also

- {doc}`/deploy/multi_node_deployment/runbook` — step-by-step cluster deployment script.
- {doc}`/observability/admin_panel` — web-based cluster view.
- {doc}`/developer_guide/security` — token roles, scopes, and security model.
- {doc}`/deploy/multi_tenancy` — multi-user fair-share scheduling guide.

# Multi-node runbook

This runbook brings up one control-plane host and one or more
Docker-capable node VMs. Nodes connect outbound to the control plane;
your workflow connects to the same control-plane gRPC endpoint.

## Pick addresses and ports

Choose the control-plane host that nodes and workflow machines can
reach.

Default ports:

| Port | Purpose |
|---|---|
| `50051` | gRPC endpoint for SDK workflows and node streams. |
| `8080` | Admin panel. Keep loopback-only and use an SSH tunnel. |
| `9090` | Prometheus metrics. |

## Start the control plane

On the control-plane host:

```bash
xrlenv up \
  --grpc-host 0.0.0.0 \
  --grpc-port 50051 \
  --metrics-port 9090 \
  --admin-port 8080
```

:::{tip}
For long-running deployments (days or weeks) under a process supervisor
or Slurm job that captures stdout, pass `--log-file` to bound log
growth:

```bash
xrlenv up \
  --grpc-host 0.0.0.0 \
  --grpc-port 50051 \
  --metrics-port 9090 \
  --admin-port 8080 \
  --log-file ~/.xrlenv/xrlenv-up-control.log \
  --log-max-bytes 52428800 \
  --log-backup-count 10
```

This moves the full JSON firehose to a size-rotating file (≈ 500 MiB
ceiling) while the supervisor capture keeps only `WARNING`/`ERROR`. See
{ref}`log-rotation` for all four flags and their defaults.
:::

For admin access from your local host, the simplest approach is an SSH
tunnel — no token setup required:

```bash
ssh -L 8080:127.0.0.1:8080 user@control-plane
```

Then open `http://127.0.0.1:8080/`.

Alternatively, if you want teammates to reach the panel directly without
an SSH tunnel, bind publicly **after** issuing at least one viewer or
operator token (see **Issue tokens** below), then restart with:

```bash
xrlenv up \
  --grpc-host 0.0.0.0 \
  --grpc-port 50051 \
  --metrics-port 9090 \
  --admin-host 0.0.0.0 \
  --admin-port 8080 \
  --admin-allow-public
```

A public bind with no credentials raises `AdminBindError` at startup.
See {doc}`/observability/admin_auth` for the full auth model.

:::{note}
With a single shared viewer token, everyone who logs in sees the whole
cluster's jobs. If you issue **per-user** viewer tokens instead
(`xrlenv tokens issue viewer --owner <id>`), each teammate gets an admin
view **scoped to their own jobs** behind the same one admin URL — no
per-user SSH tunnel needed. An operator token (and the loopback/SSH-tunnel
flow) still sees all owners. See {doc}`/deploy/multi_tenancy`.
:::

### Adaptive admission (optional)

By default, `xrlenv up` uses a static capacity estimator to gate how many
concurrent rollouts the scheduler places on each node. For clusters where
docker-daemon health can degrade under load — large concurrent sweeps,
slow registries, resource-constrained VMs — you can enable the
**health-derived adaptive admission controller** with `--adaptive-admission`:

```bash
xrlenv up \
  --grpc-host 0.0.0.0 \
  --grpc-port 50051 \
  --metrics-port 9090 \
  --admin-port 8080 \
  --adaptive-admission \
  --aimd-initial-limit 16 \
  --aimd-p95-threshold-s 60.0 \
  --aimd-max-limit 64
```

When enabled, each node gets its own independent admission limit that
contracts automatically when the node shows signs of stress and recovers
when health returns. The four flags and their defaults:

| Flag | Default | Meaning |
|------|---------|---------|
| `--adaptive-admission` | off | Master switch. The `--aimd-*` flags below are only read when this is set. |
| `--aimd-initial-limit N` | `16` | Slow-start seed: the admission limit assigned to a node the first time it connects, before any health data is observed. |
| `--aimd-p95-threshold-s SECONDS` | `60.0` | A node whose `docker run` create p95 latency exceeds this threshold triggers a multiplicative decrease on the next controller tick. |
| `--aimd-max-limit N` | `64` | Runaway guardrail: additive-increase never grows a node's limit past this value. This is not a resource calculation — node health, not this number, is the real bound. |

**These limits are per-node, not a shared cluster budget.** The controller
keeps an independent limit for each connected node, identified by node ID.
Adding nodes while `xrlenv up` is already running is safe: each new node
independently starts at `--aimd-initial-limit` and adapts on its own health
signal. Nothing is divided across existing nodes when a new one joins.
Disconnected nodes' state is dropped.

What the limit governs: the maximum number of concurrent rollouts (running
containers) the scheduler will place on that node at once. Requests that
would exceed the limit queue in the admission queue until capacity opens up.

The AIMD rule (per node, per tick, roughly every 15 seconds):

- **Bad tick** (any of: `docker_error_count > 0`, `docker_timeout_count > 0`,
  or create p95 latency > `--aimd-p95-threshold-s`): halve the limit
  — `limit = max(1, floor(limit × 0.5))`. Floor is 1; a node never
  contracts below one in-flight acquire.
- **Good tick AND the node is exactly at its current limit**: additive
  increase — `limit = min(max_limit, limit + 1)`. An under-loaded healthy
  node holds (quiet is no evidence it can take more); a node still
  over-limit after a contraction also holds (it is draining down).
- **No health data** (node agent has not yet reported Stage-1 health):
  hold at the current limit.

The admin panel's "Cluster health" page shows each node's current
admission limit and the time of its last contraction. The Prometheus gauge
`xrlenv_node_admission_limit{node_id="..."}` graphs the sawtooth pattern.

:::{note}
There are two separate per-node AIMD controllers in XRLEnv; they are easy
to confuse. This one (admission AIMD) runs in the **control plane**, is
activated by `--adaptive-admission`, and governs the maximum number of
concurrent **rollouts placed per node**. The other (pull AIMD) runs
**node-locally** and governs the maximum number of concurrent **image
pulls per node**; it is configured via `XRLENV_PULL_CONCURRENCY` /
`XRLENV_PULL_CONCURRENCY_CEILING` on each node agent. See
{doc}`/technical_details/images/cache_eviction` for the pull AIMD.
:::

### Per-tenant fair-share (optional)

Adaptive admission above is **per-node** — it bounds concurrent rollouts
on each VM by that node's health. **Per-tenant fair-share** is a separate,
orthogonal control: it caps how many concurrent containers any one `owner_id`
may hold at a time, so one user's large sweep cannot starve another's. Real
cluster resources are still enforced by the scheduler and node capacity. It is
off by default, applies only when you mint per-user tokens, and is tuned
live from the control-plane host (no restart) with `xrlenv fairshare`:

```bash
# Let each owner reach 8 concurrent sandboxes when resources exist:
xrlenv fairshare set --default-cap 8
```

See {doc}`/deploy/multi_tenancy` for the full model, owner-specific caps,
and blocking/unblocking owners.

## Issue tokens

On the control-plane host:

```bash
xrlenv tokens issue node
xrlenv tokens issue consumer

# Optional: issue admin panel tokens if you plan to expose the panel beyond loopback.
xrlenv tokens issue viewer    # read_<...> — share with teammates for read-only panel access
xrlenv tokens issue operator  # write_<...> — keep restricted; full admin write access
```

Put the node token on each node VM. Put the consumer token in the
environment where your workflow runs:

```bash
export XRLENV_GRPC_HOST=<control-plane-host>
export XRLENV_GRPC_PORT=50051
export XRLENV_CONSUMER_TOKEN=<token>
```

To rotate a token later (e.g. after a node is decommissioned), use
`xrlenv tokens rotate <role>`. To permanently invalidate a specific
token, use `xrlenv tokens revoke <token-id>`. See
{doc}`/developer_guide/tokens` for the full workflow and
{doc}`/observability/admin_auth` for distributing viewer tokens to teammates.

:::{tip}
**Serving more than one person?** The single `consumer` token above
authenticates everyone as the same identity. To give each user their own
token — so you can revoke one person without rotating the shared token that
everyone else uses — issue per-user tokens instead with
`xrlenv tokens issue consumer --owner <id>`. Per-user tokens also stamp
ownership on each job and give each teammate's viewer token an admin view
**scoped to their own jobs**. See {doc}`/deploy/multi_tenancy` for the full
per-user workflow.
:::

## Bootstrap node VMs

The `xrlenv bootstrap` subcommand installs Docker, creates the
`xrlenv` system user, builds a venv at `/opt/xrlenv/.venv`, installs
xrlenv from the source you specify, writes `/etc/xrlenv/node.env`,
installs and enables the `xrlenv-node` systemd unit, and starts the
daemon. Each step has a skip-if predicate, so re-running on an
already-bootstrapped host is safe.

**Step 1 — preview the plan (touches nothing).** Always run a
dry-run first on a new VM. It prints the full plan — every step and
every shell command that would execute — without touching the host:

```bash
sudo -E bash deploy/bootstrap-gcp.sh <control-plane>:50051 --dry-run
# Or for AWS:
sudo -E bash deploy/bootstrap-aws.sh <control-plane>:50051 --dry-run
```

Read the output. You should see ~17 numbered steps ending with
`systemctl-restart-xrlenv-node`. No step should print `WOULD: <bash
log line>` — that signature means you're on an older branch that
predates the Python bootstrap; pull and retry.

**Step 2 — get the node token (on the control plane).** If your
`xrlenv up` has any tokens issued (the default in any non-loopback-
dev deployment), the new node needs the same node-tier bearer the
existing nodes use. On the **control-plane host**:

```bash
cat ~/.xrlenv/secrets/node.token
```

Copy the printed string — it's the value you'll paste into
`XRLENV_NODE_TOKEN` in step 3. Skip this step if your control plane
has no tokens issued (loopback-only dev flow).

**Step 3 — live run.** On the new VM, run the bootstrap with the
node token and (recommended) Docker Hub credentials inline:

```bash
sudo \
    XRLENV_NODE_TOKEN='<paste-token-here>' \
    DOCKERHUB_USER='<your-docker-hub-handle>' \
    DOCKERHUB_TOKEN='<dckr_pat_...>' \
    bash deploy/bootstrap-gcp.sh <control-plane>:50051
# Or for AWS:
sudo \
    XRLENV_NODE_TOKEN='<paste-token-here>' \
    DOCKERHUB_USER='<your-docker-hub-handle>' \
    DOCKERHUB_TOKEN='<dckr_pat_...>' \
    bash deploy/bootstrap-aws.sh <control-plane>:50051
```

The `sudo VAR=value command` form passes the env var into the
bootstrap reliably — more robust than `sudo -E`, which some
hardened sudoers configs strip. The bootstrap detects
`XRLENV_NODE_TOKEN` in its environment and writes a mode-0600
systemd drop-in at
`/etc/systemd/system/xrlenv-node.service.d/10-token.conf` so the
daemon authenticates on startup. The token never lands in the
world-readable `/etc/xrlenv/node.env`.

`DOCKERHUB_USER` + `DOCKERHUB_TOKEN` authenticate the per-node docker
daemon for image pulls. The bootstrap writes a mode-0600
`config.json` at `/opt/xrlenv/.docker/config.json` (owner: the
`xrlenv` runtime user) containing the base64-encoded credentials.
Without this, the daemon rate-limits at **~100 image pulls per 6
hours per source IP** — a 500-instance SWE-bench Verified sweep
will fail partway through with `InsufficientCapacity`-shaped errors
that end users see as failed rollouts. With auth, your Docker Hub
account-tier cap applies (Business / Pro / Team tiers are typically
unlimited). End users submitting jobs to the control plane never
touch Docker Hub directly; the operator's one-time setup here is
what insulates them from the rate limit. Use a [Personal Access
Token](https://docs.docker.com/security/for-developers/access-tokens/)
(not your password). If you forget these vars, the bootstrap prints
a loud warning at the end with the recovery one-liner; re-running
with the vars set is the simplest fix.

:::{tip}
On larger clusters you can cut upstream Docker Hub traffic further with
an optional pull-through {doc}`registry mirror <registry_mirror>`: the
first cluster-wide pull of each image is cached on a shared store and
every later pull (including after a node evicts the image) is served
LAN-local. It complements per-node Docker Hub auth — set it up after the
cluster is running and pulls become a bottleneck.
:::

Omit the `XRLENV_NODE_TOKEN=` prefix only if your control plane has
no tokens issued. **Forgetting the token** is the most common new-
node failure mode: the daemon starts, but every gRPC dial 401s with
`StatusCode.UNAUTHENTICATED, missing or unknown bearer token`. The
fix is to re-run the same command with the env var set — the
bootstrap is idempotent, so only the token-dropin + systemd-restart
steps execute. (Or hand-write the drop-in; see the troubleshooting
section at the bottom of this page.)

**Reference: install-source flags.** Each step in the bootstrap is
idempotent via skip-if predicates; re-runs are safe. The bash
wrapper passes `--xrlenv-repo $REPO_ROOT` by default. To override:

| Flag | Use case |
|------|----------|
| `--xrlenv-wheel /path/to/xrlenv-*.whl` | Production deploys with a pre-built wheel. |
| `--xrlenv-repo /path/to/xrlenv` | Operators running from a git checkout (installed non-editable). Default in the bash wrapper. |
| `--xrlenv-version 1.2.3` | PyPI fallback when a public release is available. |

**Reference: node-id resolution.** Order of precedence:
`--node-id` flag → `$XRLENV_NODE_ID` env → cloud metadata service
(GCP instance metadata / AWS IMDSv2). For `--target linux-generic`
(no cloud metadata), `--node-id` or `$XRLENV_NODE_ID` must be set
explicitly.

**Reference: bootstrap env-var equivalents.** All bootstrap CLI flags
can be supplied as environment variables in your `.env_private` (or
inline on the `sudo` invocation). Useful when you manage bootstrap
config through a dotenv file rather than flags:

| Env var | Equivalent flag | Default | Effect |
|---|---|---|---|
| `XRLENV_PYTHON` | (no flag) | _(probed at runtime)_ | Pin the Python 3.12 binary the bootstrap uses. Checked before PATH discovery (`python3.14` → `python3.13` → `python3.12`). Set when your VM has Python 3.12 at a non-standard path. |
| `XRLENV_USER` | `--runtime-user` | `xrlenv` | System user the node daemon runs as. Change only if your hardening policy requires a different service account name. |
| `XRLENV_VERSION` | `--xrlenv-version` | `main` | PyPI version pin used when neither `--xrlenv-wheel` nor `--xrlenv-repo` is set. Ignored when the bash wrapper passes `--xrlenv-repo` (the default). |

These are read by `xrlenv/cli/bootstrap.py` from the environment (or the
nearest `.env` file) at startup and merged with any explicit flags — flags
win on conflict.

**Reference: what the bash wrappers do.** They're now three-line
scripts that shift the first positional arg into
`XRLENV_CONTROL_PLANE`, the second into `XRLENV_NODE_ID`, then
`exec python3 .../xrlenv/cli/bootstrap.py --target {gcp,aws}
--xrlenv-repo <repo> "$@"`. The Python module runs stdlib-only so
it executes on a fresh VM before `xrlenv` itself is installed.

For provider-specific notes:

- {doc}`cloud_VM_providers/gcp`
- {doc}`cloud_VM_providers/aws`

## Declare inventory

Create or update `nodes.yaml` on the control-plane host so CLI and
admin views can show rostered nodes:

```yaml
nodes:
  - id: gcp-a
    host: internal-ip
    provider: gcp
    zone: us-central1-a
  - id: aws-a
    host: internal-ip
    provider: aws
    zone: us-west-2a
```

See {doc}`inventory`.

### Cluster docker-kwarg policy (optional)

The same `nodes.yaml` carries an optional `policy:` section that
controls which docker-py kwargs the cluster forwards to
`containers.run` on each node. **Defaults work for SWE-bench Verified,
terminal-bench-2, coding-bench, and SCUBA-style KVM benchmarks out of
the box** — you only need to edit this section when restricting (denying
a capability) or opting in to a Level-2 risk (host networking,
privileged mode, host-path mounts).

```yaml
policy:
  # Level 1 — allowed by default, you can clamp.
  allowed_devices:                # host devices the harness may pass
    - /dev/kvm                    #   nested-VM benchmarks (SCUBA, OSWorld)
    - /dev/net/tun                #   userland VPN / network tooling
    - /dev/fuse                   #   userspace filesystems
  denied_caps: []                 # NET_ADMIN, SYS_ADMIN, etc. are
                                  # allowed by default; list any to deny

  # Level 2 — rejected by default, opt in here.
  allow_host_network: false       # network_mode=host bypasses egress
  allow_privileged: false         # privileged=True is sandbox escape
  allowed_host_paths: []          # bind mounts from NODE VM filesystem
```

Editing the policy:

1. SSH into the control-plane host (the one running `xrlenv up`).
2. Edit `nodes.yaml`, change the `policy:` keys you need.
3. Restart the control plane (`systemctl restart xrlenv-control` or
   re-run `xrlenv up`). **Node VMs do not need a restart** — the policy
   is enforced at the control plane only.

When a kwarg is rejected the consumer sees an error of the form:

```
xrlenv: rejected docker kwarg `devices` (level 1): device '/dev/sda'
  not in cluster's allowed_devices ['/dev/fuse', '/dev/kvm', '/dev/net/tun'].
  fix: operator: add '/dev/sda' to nodes.yaml policy.allowed_devices
       and restart the control plane.
```

The four enforcement tiers, with example kwargs:

| Tier | Behavior | Examples |
| --- | --- | --- |
| 0 — always allowed | no policy option | standard caps (`SYS_PTRACE`, `NET_RAW`), `entrypoint`, `environment`, `read_only`, `shm_size`, `tmpfs` |
| 1 — allowed by default | operator can restrict | `devices` (allowlist), `cap_add` (denylist for elevated caps like `NET_ADMIN`/`SYS_ADMIN`) |
| 2 — rejected by default | operator can opt in | `privileged=True`, `network_mode="host"`, host bind mounts |
| 3 — never allowed | no policy override | `pid_mode="host"`, `ipc_mode="host"`, `cgroup_parent`, `network_mode="container:..."` |
| 4 — architectural mismatch | warn-and-drop | `platform`, `userns_mode` (set per-node at bootstrap, not from the harness) |

Enforcement is split between client and control plane:

- **Client-side drop-in** fast-fails Level 3 only (always-unsafe; no operator
  can opt these in, so it's safe to reject locally for a snappier error
  message). Level 1 and Level 2 always flow to the control plane — the
  drop-in cannot see your operator-tuned policy, so it would be wrong for it
  to pre-reject anything you might have permitted.
- **Control plane** is the sole authoritative validator for Level 1 and
  Level 2. The error message names the kwarg, the level, and the YAML
  stanza to edit.

## Verify attachment

Check from the control-plane host:

```bash
xrlenv nodes --nodes-yaml nodes.yaml
xrlenv audit --kind auth.token_used --role node --since 10m
```

The admin panel `/nodes` page should show connected nodes with recent
heartbeats.

## Run a smoke

For the direct managed-container API, run a small script from the
workflow host:

```python
import asyncio
from xrlenv import Client

async def main():
    client = Client.grpc(
        host="<control-plane-host>",
        port=50051,
        token="<consumer-token>",
    )
    async with await client.acquire_container(
        image="ubuntu:22.04",
        command=["sleep", "infinity"],
    ) as session:
        result = await session.exec(["bash", "-lc", "echo ok"])
        print(result.stdout.decode())
    await client.close()

asyncio.run(main())
```

For benchmark smokes, use:

- {doc}`/supported_benchmarks_and_harnesses/swe_bench`
- {doc}`/supported_benchmarks_and_harnesses/harbor_framework`

## Sync xrlenv updates to existing nodes

After the initial bootstrap, the fast path for pushing a new
xrlenv release (or rotating per-node config like Docker Hub auth)
onto an already-running node is `deploy/refresh.sh`. It skips the
heavyweight bootstrap steps (user creation, system package install,
Python interpreter install) and only re-does what changes on a
normal release: pip-reinstall xrlenv into `/opt/xrlenv/.venv`,
re-write `/etc/xrlenv/node.env` from the latest source's template
(picks up any new `XRLENV_*` env vars), refresh the writable
subdirectories under `/var/cache/xrlenv/`, refresh per-user Docker
Hub auth, and `systemctl restart xrlenv-node`.

```bash
# On each worker node (after `git pull` on the on-host xrlenv checkout):
sudo -E bash deploy/refresh.sh

# Or to rotate Docker Hub auth at the same time:
sudo \
    DOCKERHUB_USER='<handle>' \
    DOCKERHUB_TOKEN='<dckr_pat_...>' \
    bash deploy/refresh.sh
```

`refresh.sh` reads the same env-var contract as the bootstrap for
operator-set values. Pass `DOCKERHUB_USER` + `DOCKERHUB_TOKEN` to
rewrite the runtime user's `/opt/xrlenv/.docker/config.json`; omit
them to preserve whatever auth file is already on disk. The auth
rewrite happens **before** the daemon restart so docker-py's
`APIClient` reads the new credentials at process startup (an
auth-after-restart path leaves the running daemon on the old PAT
until the next restart cycle — a real bug we hit during phase-1
operator validation; see the order-regression test at
`tests/unit/deploy/test_bootstrap_dockerhub_auth.py`).

`XRLENV_NODE_TOKEN` doesn't need to be re-passed on a routine
refresh — `refresh.sh` preserves the existing systemd drop-in at
`/etc/systemd/system/xrlenv-node.service.d/10-token.conf`. To
rotate the node token, see "Token rotation" further down.

For the **operator-side install** (the host where you run
`xrlenv up` and the CLI), there's no `refresh.sh` equivalent —
that install is a plain editable Python venv:

```bash
# On the operator host:
cd /path/to/xrlenv && git pull --ff-only origin <branch>
# (Optional) refresh deps if the new release added any:
uv sync --extra dev --extra docs --extra observability \
        --extra terminal-bench-2 --extra swebench-verified
# Restart `xrlenv up` to pick up the new code.
```

The editable install (xrlenv shipped via `pyproject.toml`'s
`[tool.hatch.build.targets.wheel] packages` declaration) means a
plain `git pull` is enough for source changes — no reinstall
needed unless dependencies in `pyproject.toml` changed.

## Troubleshooting

| Symptom | Check |
|---|---|
| Node does not appear | `journalctl -u xrlenv-node` on the node, then `xrlenv audit --kind auth.denied`. See "Node 401s" below for the most common case. |
| Workflow cannot connect | Control-plane firewall/security group for `50051`, token env vars, and `XRLENV_GRPC_HOST`. |
| Admin page not reachable | Use an SSH tunnel (loopback default), or check that `--admin-allow-public` is set and at least one token is issued if binding publicly. |
| First acquire is slow | The selected node may be pulling the image. Check `/images/cache` and `/images/catalog`. |
| Disk pressure | See "Disk layout & cleanup" below; use `/images/cache` to identify cold images and pinned images. |
| `CancelledError` tracebacks from `grpc._cython.cygrpc` on `Ctrl-C` | Harmless shutdown noise — see "gRPC `CancelledError` on shutdown" below. |
| `version skew: node agent_version=… control plane=…` WARN | See "Version-skew warning" below; benign when the only intervening commits are docs / scripts. |
| `node_lost` for a node whose daemon is running | Likely large `get_archive` reply flooding the heartbeat stream or disk pressure — see "Node resilience / `node_lost`" below. |

### Node resilience / `node_lost`

**Symptom.** The control plane logs `node_lost` for a node that is otherwise
healthy — its `xrlenv-node` daemon is running and the underlying VM is fine.
No container or network failures preceded it.

**Dominant cause (prod-observed).** `node_lost` can be triggered by a single
benchmark task issuing a `get_archive` call that returns a very large tar —
a whole-`/testbed` archive, for example. Before the streaming fix, that
entire archive transited the shared heartbeat bidi stream between the node
agent and the control plane. A multi-hundred-megabyte transfer starved
the heartbeat, causing the control plane to declare the node lost. Simultaneously,
writing such an archive to disk contributed to disk-pressure events.

**Mitigations now in place.** No configuration is needed for typical workloads;
these knobs exist for operators running benchmarks with unusually large
`get_archive` replies:

| Env var (on the node) | Default | Effect |
|---|---|---|
| `XRLENV_MAX_GET_ARCHIVE_RELAY_BYTES` | 128 MiB | Per-call cap on bytes that a single `get_archive` may relay through the control plane. Calls that exceed this limit are refused with `ArchiveTooLarge` rather than allowed to flood the stream. |
| `XRLENV_RAW_ARCHIVE_CONCURRENCY` | (see resource_isolation.md) | Max concurrent `get_archive` / `put_archive` operations on a node. Limits how many large transfers can compete at once. |
| `XRLENV_DISK_GUARD_ENABLED` | `true` | Enables the disk-pressure guard (default on). When free disk drops below the threshold, the node raises `NodeBusyError` on new acquires rather than accepting work it cannot service. |
| `XRLENV_DISK_GUARD_INTERVAL_S` | `15.0` | How often the disk guard checks free space. |

See {doc}`/technical_details/resource_isolation` for the full knob tables,
default values, and tuning guidance.

**Diagnosis steps.**

1. On the node: `journalctl -u xrlenv-node --since "30 min ago" | grep -E "node_lost|ArchiveTooLarge|NodeBusy|disk"`.
2. Check disk pressure: `df -h /` and `docker system df` on the affected node.
3. Identify which benchmark call is generating large archives: look for
   `get_archive` calls on paths like `/testbed` or `/workspace` in the
   rollout coordinator log under `/var/lib/xrlenv/runs/<rollout-id>/coordinator.log`.
4. If the archive is genuinely needed, raise `XRLENV_MAX_GET_ARCHIVE_RELAY_BYTES`
   (set in `/etc/xrlenv/node.env`, then `sudo systemctl restart xrlenv-node`).
   If the archive is unnecessary (the benchmark fetches it as a side-effect),
   fix the benchmark — don't raise the cap.

### gRPC `CancelledError` on shutdown

Symptom: stopping the control plane with `Ctrl-C` (or `SIGTERM`) prints
one or more tracebacks like:

```text
ERROR grpc._cython.cygrpc: Exception not handled by _handle_exceptions
  in servicer method [/xrlenv.node_control.v1.NodeControl/NodeControlStream]
  File ".../_cygrpc/aio/server.pyx.pxi", line 787, in _schedule_rpc_coro
asyncio.exceptions.CancelledError
```

**This is harmless and expected — not an xrlenv error.** The node↔control
streams are long-lived bidirectional RPCs that never close on their own,
so when `grpc_server.stop(grace=…)` runs on shutdown, grpc-aio
force-cancels each still-open stream and its C extension logs the
`CancelledError` from `_schedule_rpc_coro`. There are no xrlenv frames in
the trace: it originates inside grpc, *outside* any `try/except` we
control. The servicer already catches its own cancellation and seals /
persists state **before** the server stops, so no trajectory or state is
lost — you will see one such line per node that was still attached.

We deliberately do **not** install a logging filter to suppress it: a
filter broad enough to catch this could also hide a genuine error. Treat
these lines as confirmation the streams were torn down, nothing more.

### Version-skew warning

Symptom: the control plane logs
`WARN … version skew: node agent_version=0.0.1+<sha-A>, control
plane=0.0.1+<sha-B>`. It means the node-agent and the control plane were
built from different commits. The node's build SHA is **stamped into
`/etc/xrlenv/node.env` at bootstrap time**; the control plane's is
computed **live when `xrlenv up` starts**. So committing (or `git pull`)
between bootstrapping a node and starting the control plane makes the two
SHAs differ.

It is **benign when the only intervening commits touched docs or
scripts** (no node-agent code) — the node still carries every code fix.
To clear it, re-bootstrap the nodes so `XRLENV_BUILD_SHA` is re-stamped
to the current HEAD (a plain `systemctl restart xrlenv-node` will *not*
update it — it re-reads the already-stamped `node.env`). To avoid it
entirely, commit first, then bootstrap the nodes and start the control
plane from the same HEAD.

### Node 401s with `missing or unknown bearer token`

Symptom: `journalctl -u xrlenv-node` shows the daemon retrying every
~30 seconds with
`StatusCode.UNAUTHENTICATED, missing or unknown bearer token`. Cause:
the bootstrap ran without `XRLENV_NODE_TOKEN` in its environment, so
no systemd drop-in was written.

Two recoveries:

1. **Re-run the bootstrap with the env var set** (preferred — only
   the token-dropin and systemd-restart steps actually execute,
   the rest skip via their idempotent predicates):

   ```bash
   sudo XRLENV_NODE_TOKEN='<paste-token-here>' \
       bash deploy/bootstrap-gcp.sh <control-plane>:50051
   ```

2. **Hand-write the drop-in** (when re-running the bootstrap isn't
   possible — e.g., the checkout has been removed):

   ```bash
   sudo mkdir -p /etc/systemd/system/xrlenv-node.service.d
   sudo tee /etc/systemd/system/xrlenv-node.service.d/10-token.conf <<EOF >/dev/null
   [Service]
   Environment="XRLENV_NODE_TOKEN=<paste-token-here>"
   EOF
   sudo chmod 0600 /etc/systemd/system/xrlenv-node.service.d/10-token.conf
   sudo systemctl daemon-reload && sudo systemctl restart xrlenv-node
   ```

Get the token value by running `cat ~/.xrlenv/secrets/node.token`
on the control-plane host.

## Disk layout & cleanup

The bootstrap creates four writable directories on each node, each
with different growth dynamics. The control plane has its own
analogous tree on the control-plane host. Knowing which is which
matters when a node fills up.

### On each node (created by `xrlenv bootstrap`)

| Path | What it holds | Grows because... | Safe to wipe? | How to clear |
|---|---|---|---|---|
| `/var/lib/xrlenv/runs/<rollout-id>/` | Per-rollout durable artifacts: trajectory.jsonl, coordinator.log, in-sandbox `/logs/` mirror. | Each rollout adds a directory; size depends on benchmark verbosity. | Yes for old rollouts, **no** if you still want to inspect them. | The control plane's `xrlenv up --retention-days N` GCs anything older than N days. Manual: `sudo rm -rf /var/lib/xrlenv/runs/<old-rollout-id>/`. |
| `/var/cache/xrlenv/harbor/tasks/` | Operator-populated harbor task assets (the upstream task tree the harbor adapter reads at rollout start). | Set once via `populate-harbor-cache.sh` per task subset; static after that. | Yes — re-populate by re-running the populator. | `sudo rm -rf /var/cache/xrlenv/harbor/tasks/<task-id>/` then re-run `populate-harbor-cache.sh`. |
| `/var/cache/xrlenv/build-context-cache/` | Git clones for `context_source: type: git` plan entries, plus LRU bookkeeping. | Each new `(repo, ref)` adds a checkout; bounded by `GitSourceBuilder`'s **5 GB total cap with LRU eviction**, so it self-stabilizes. | **Yes always** — the next build re-clones from upstream. | `sudo rm -rf /var/cache/xrlenv/build-context-cache/`. The next `xrlenv build apply --plan` re-clones. |
| Docker image cache (managed by Docker, not xrlenv) | Pulled / built images. Per spec-15, `ImageCacheManager` evicts cold images under disk pressure. | Each pull / build adds layers; LRU + tier-based eviction reclaims under disk pressure. | Yes for cold images. | `docker image prune` to wipe untagged layers. `docker system prune -a` for more aggressive (also wipes stopped containers). |

There is **no** `/var/cache/xrlenv/runs/` directory — run dirs are
durable state and live under `/var/lib/xrlenv/runs/` per FHS.
`/var/cache/xrlenv/` only holds regeneratable cache (harbor +
build-context).

### On the control-plane host

The control plane writes to its own paths regardless of whether
it runs on a cloud VM or a local device. Default locations
(operator can override at `xrlenv up` time):

| Path | What it holds | Grows because... | Safe to wipe? | How to clear |
|---|---|---|---|---|
| `~/.xrlenv/state.db` (or `--state-db`) | Persistent control-plane state: rollouts, sandboxes, build plans, audit log. | Every rollout, sandbox, and apply adds rows. | **No** while the control plane is running; it would lose its view of attached nodes and in-flight work. Stop `xrlenv up` first. | After stopping: `xrlenv events --since 7d --format json > backup.json` then `rm ~/.xrlenv/state.db`. Restart `xrlenv up`. |
| `~/.xrlenv/runs/` (or `--runs-root`) | Mirror of node-side `/var/lib/xrlenv/runs/<rollout-id>/` for rollouts that ran in-process under a `LocalRuntime`. | Local runs only; cloud-VM runs live on the nodes. | Yes for old rollouts. | `--retention-days N` GCs; manual `rm -rf <old-rollout-id>/`. |
| `~/.xrlenv/secrets/` | Operator + node + consumer tokens (`<role>.token`), optional grace sidecars (`<role>.token.previous.json`), revocation list (`revoked.json`). | Changes only on `tokens issue`, `tokens rotate`, or `tokens revoke`. | **No** without re-issuing. | Don't wipe casually; if you do, re-issue all roles and redistribute the node tokens. Prefer `tokens rotate` or `tokens revoke` for individual credential changes. |
| `<path from --log-file>` (optional, operator-set) | Rotating JSON log firehose when `--log-file` is passed to `xrlenv up`. Size-bounded by `--log-max-bytes × (--log-backup-count + 1)` (default ≈ 500 MiB). | Grows until first rollover, then wraps at the max-bytes boundary. Rotated files get `.1` / `.2` / … suffixes. | Yes — old rotations (`.1`, `.2`, …) are always safe to delete. The active file can be deleted only after stopping `xrlenv up`. | Delete rotated suffixes freely: `rm <path>.{1..10}`. To clear the active file, stop the control plane, then `truncate -s 0 <path>` or delete and restart. |

All three paths share a common root (`~/.xrlenv` by default). Set
`XRLENV_HOME=/some/dir` to relocate the whole tree as a group. The
canonical use case is running a **dev cluster beside prod on hosts that
share `$HOME` over FSx/Lustre or NFS**: without this, a second
`xrlenv up` on the default path would open prod's `state.db` from a
second host (two SQLite WAL writers — corruption risk) and share prod's
`secrets/` token store, so a dev token would authenticate against prod.
Giving the dev checkout's `.env` its own `XRLENV_HOME` keeps state and
tokens fully separate. `XRLENV_HOME` is read at import time (from the
shell environment or the nearest `.env` file); it must be set before
`xrlenv up` starts, not mutated mid-process. Prefer `XRLENV_HOME` over
overriding `$HOME` — the latter also relocates `~/.gitconfig`,
`~/.ssh`, and `~/.docker/config.json`. The `--state-db` and
`--runs-root` flags still override those two paths individually;
`XRLENV_HOME` is the single switch that also moves `secrets/`, which
has no `xrlenv up` flag.

### What's expected to grow unbounded

- `/var/lib/xrlenv/runs/` on every node — every rollout adds a
  dir. **Always** set `xrlenv up --retention-days N` (default 14)
  to auto-GC. Operators running long-horizon benchmarks should
  size disk for `N × max-rollouts-per-day × per-rollout-bytes`.
- **Control-plane stdout log** — when the control plane runs under a
  process supervisor (Slurm `#SBATCH --output=…`, `nohup`, `tee`) that
  captures stdout to a file, that capture file has **no rotation and no
  size cap**: it grows without bound for as long as the control plane
  runs. Fix: pass `--log-file <path>` to `xrlenv up`. This moves the
  full JSON firehose to a size-rotating file (default ceiling ≈ 500 MiB)
  while the supervisor capture retains only `WARNING` and above — just
  the boot banner and any crashes. See {ref}`log-rotation` for the full
  flag reference. The node daemon is **not** affected — it runs under
  systemd with `StandardOutput=journal`, which journald already rotates.
- `/var/cache/xrlenv/build-context-cache/` is **not** unbounded —
  the `GitSourceBuilder`'s LRU cap (default 5 GB total) holds it
  bounded. Override via `XRLENV_BUILD_CONTEXT_CACHE_TOTAL_CAP_BYTES`
  if you need a different size.
- The Docker image cache **is** unbounded by default — Docker
  itself doesn't evict. xrlenv's `ImageCacheManager` adds eviction
  on top, triggered when free disk drops below
  `image_cache.start_threshold_gb` (default 15 GB). Tune via
  `xrlenv up` flags or rely on the default.

When a node hits disk pressure unexpectedly, walk these in order:

```bash
# On the affected node:
df -h /                                    # how full?
du -sh /var/lib/xrlenv/runs/* | sort -h    # which rollouts?
du -sh /var/cache/xrlenv/*                 # harbor + build-context sizes
docker system df                            # Docker's view
docker image ls | head                      # cold images
```

`/images/cache` in the admin panel shows the same info from the
control-plane side.

## Stop and clean up

Stop node daemons with systemd on each VM:

```bash
sudo systemctl stop xrlenv-node
```

Stop the control plane with `Ctrl-C` or your process supervisor.
Destroy cloud VMs through the provider console when you are done.

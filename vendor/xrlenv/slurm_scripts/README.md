# Slurm cluster deployment

Every committed script under **`generated/`** — `deploy_<name>.sh`,
`<name>_xrlenv_{node,control}.sh`, and the runtime-written
`<name>_hyperpod_nodes.yaml` roster — is **generated**. They are a pure function
of two inputs, both at the top of this directory:

- **`clusters.yaml`** — all per-cluster *config* (one block per cluster; copy
  `clusters.example.yaml` and fill in your hosts to start);
- **`templates/`** — all script *prose* (one copy, shared by every cluster).

The split is the point: the top level of `slurm_scripts/` holds only *source*
(`clusters.yaml`, `templates/`, `generate_deployment_script.py`, and the sourced
shell library under `lib/`), while every *artifact* lives under `generated/`. So
`ls slurm_scripts/` shows inputs, never outputs, and
`bash slurm_scripts/generated/deploy_<name>.sh` is the one command an operator
runs. (Bulk image build+push is now `xrlenv build push`, native across the
fleet — the old `build_and_push_images.sh` Slurm job is gone.)

So the two routine operations cost one edit each:

| Task | What you change | Then run |
|---|---|---|
| Add a worker node | one line under `workers:` | `--cluster <name>` |
| Add a whole cluster | one `<name>:` block (~10 lines) | `--cluster <name>` |

Nothing else: no new files to create, no Python to edit, no test to update. The
generator scaffolds a new cluster's three scripts on first run and marks the
deploy script executable.

**Never hand-edit a generated script** — `--check` fails if you do, and the next
regeneration overwrites it. Change `clusters.yaml` (config) or `templates/`
(prose) instead. Keeping the prose in one template is the point: the three
scripts previously carried ~490 duplicated lines each to express ~45 lines of
real config, and their comments had already drifted apart.

## Fields

Required per cluster:

- `control_plane`: the Slurm hostname running the control-plane job.
- `registry`: image-registry endpoints — `mirror_host` + `private_host`
  (required) and `mirror_port` / `private_port` / `scratch_host` / `scratch_port`
  (optional; default `5010` / `5011` / the `private_host` / `5012`). A cluster
  may share another box's registry or run its own. The scratch (build-on-demand)
  registry is a third registry that defaults to the private host, so a cluster
  running one registry box gets all three endpoints on it.
- `workers`: the ordered worker allocation.
- `sysbox_pool`: workers on which the deploy script installs/enables Sysbox.
- `cpu_isolation_pool`: workers on which the deploy script enables CPU isolation.

Optional, each **derived from the cluster name** when omitted:

| Field | Default |
|---|---|
| `checkout` | this repo's root |
| `local_disk_root` | `/opt/sagemaker` — see below |
| `state_db` | `<local_disk_root>/xrlenv-<name>/state.db`, or `/var/lib/xrlenv-<name>/state.db` when there is no volume |
| `control_log` | `~/.xrlenv-<name>/xrlenv-up-control.log` |
| `tunnel.{admin,metrics}` | `8080` / `9090` |
| `deploy_registry` | `true` (start a registry pair on the control plane) |
| `partition`, `account`, `grpc_port`, `sysbox_max_concurrent` | the `defaults:` block |
| `allowed_host_paths` | empty — see below |
| `node_env` | empty — see below |

A top-level `defaults:` block supplies values inherited by every cluster; an
explicit per-cluster key always wins.

### `local_disk_root` — one knob, two consumers

`local_disk_root` is the mount point of the box's **dedicated local data
volume**, and it is tri-state:

| value | meaning |
|---|---|
| absent | the conventional `/opt/sagemaker` |
| a path | that mount is the local data volume |
| `null` | **this cluster's boxes have no dedicated volume** |

It is one knob because it answers one question that two things need:

- **Docker's data-root on workers.** With a volume, the node script passes
  `--hyperpod` so image layers land there instead of filling the small root disk.
  With `null` the flag is **omitted** — `deploy/node/set_docker_data_root.sh` refuses
  a target that isn't a real block device *and* refuses one resolving to the root
  device, so passing it anyway aborts the bootstrap before the node agent is ever
  installed. (cn hit exactly this on 2026-08-08: all four nodes exited 1, no
  agent, no registration — while the Slurm job still showed `RUNNING`, because
  the batch script `sleep`s after `srun`.)
- **`state_db` on the control plane.** It must be on local disk — a SQLite WAL on
  Lustre faults with `SIGBUS` — so it defaults under the volume when there is
  one, and under `/var/lib` when there isn't. The root disk is still local; it
  just isn't dedicated.

So a cluster whose nodes have only a root disk sets `local_disk_root: null` and
both behaviors follow. `cn` was that case until 2026-08-10, when devops rebuilt
its m7i fleet with a dedicated 500 GB `nvme1n1` (xfs) at `/opt/sagemaker`; it now
declares the path like dev/prod. No cluster sets `null` today — the tri-state
stays because fleets get rebuilt in both directions, and the `null` path is what
keeps a volume-less one bootable.

**Convention: declare `state_db`, `control_log` and `checkout` explicitly**, even
when the value equals what would be derived. These name durable on-disk state, so
a cluster inheriting them would silently relocate its `state.db` — losing the
control plane's registry — if the derivation ever changed. Explicit also means
`grep state_db clusters.yaml` is the complete answer. The generator prints an
advisory warning when a cluster leans on a derived value.

`allowed_host_paths` defaults to **empty** and is declared per cluster, never in
`defaults:`. It is a security allowlist (`policy.allowed_host_paths`)
authorizing real read-only host mounts into sandboxes, *and* it is cluster-
specific: different clusters mount their shared filesystem at different paths (or
not at all). Inheriting a mount permission a cluster never asked for is wrong in
both directions.

### `node_env` — the escape hatch for a fleet that isn't the current shape

A mapping of `XRLENV_*` name to value, spliced into the node bootstrap's `sudo`
line so every worker in that cluster starts its agent with those overrides.

**No cluster declares one today, and that is not a reason to delete it.** It
exists so the deploy path can carry a fleet whose boxes do *not* match the
current shape, without a code change. The worked example is `cn` before
2026-08-10: on boxes with only a ~96 GiB root disk the image cache's default
50 START / 75 STOP GiB eviction band was **unreachable** (~26 GiB of the disk is
non-evictable), so the cache evicted forever, re-pulled images every sweep, and
the disk guard began killing live containers as runaways. `node_env` carried a
hand-tuned 20/30 GiB band until that fleet got real volumes. Delete the knob and
the next such fleet needs a patch instead of a config line.

Names must match `XRLENV_[A-Z0-9_]+`; values are restricted to
`[A-Za-z0-9_.:/=+-]`. That charset is an **injection boundary, not style** — the
pairs are interpolated into a `sudo ... KEY="VALUE" \` command that runs the
bootstrap as root, so a value containing a space, quote, `$`, backtick or `;`
would be a command-injection vector rather than a syntax error. Both constraints
are enforced at parse time and covered by
`tests/unit/deploy/test_generate_deployment_script.py`.

An empty `node_env` renders to the empty string, so a cluster that declares none
produces a byte-identical script to one that never had the key — which is what
keeps the `--check` golden gate stable.

## Cluster isolation

Each cluster owns a disjoint set of runtime resources, so deploying one can
never disturb another. All of it now comes from `clusters.yaml`.

| | `dev` | `prod` | `cn` |
|---|---|---|---|
| checkout root | `…/xrlenv-dev` | `…/xrlenv` | `…/xrlenv` |
| Slurm job names | `dev-xrlenv-{nodes,control}` | `prod-xrlenv-…` | `cn-xrlenv-…` |
| `state.db` | `/opt/sagemaker/xrlenv-dev` | `/opt/sagemaker/xrlenv` | `/opt/sagemaker/xrlenv-cn` |
| control log | `~/.xrlenv-dev/` | `~/.xrlenv/` | `~/.xrlenv-cn/` |
| roster YAML | `generated/dev_hyperpod_nodes.yaml` | `generated/prod_hyperpod_nodes.yaml` | `generated/cn_hyperpod_nodes.yaml` |
| login-node tunnel (admin/metrics) | `9080` / `9190` | `8080` / `9090` | `9082` / `9192` |
| registry | shared (off-box) | own pair on CP | own pair on CP |

`state.db` and the control log are keyed to the control-plane **box**, so two
clusters sharing a control-plane host would collide on both; the tunnel ports
are keyed to the **login node**, which every cluster shares, so those must be
distinct even when the clusters are otherwise disjoint. The generator enforces
all of this — see "refuses unsafe configurations" below.

## Commands

```bash
# regenerate one cluster (after editing its block)
.venv/bin/python slurm_scripts/generate_deployment_script.py --cluster cn

# regenerate everything
.venv/bin/python slurm_scripts/generate_deployment_script.py

# CI / pre-deploy gate: assert every committed script still equals
# templates/ + clusters.yaml, byte for byte
.venv/bin/python slurm_scripts/generate_deployment_script.py --check

# additionally cross-check every host against live Slurm
.venv/bin/python slurm_scripts/generate_deployment_script.py --check --validate-slurm
```

By default the scripts are re-rendered in full from `templates/`, which is what
makes every `clusters.yaml` field take effect.

`--patch-only` is a narrow escape hatch that rewrites just the topology lines
(control plane, nodelist, pools) of existing scripts, preserving local edits. It
**cannot see** any other field — `allowed_host_paths`, `state_db`, `tunnel`,
`checkout` and friends are invisible to it, so editing one of those and running
`--patch-only` silently does nothing. Prefer the default.

`--validate-slurm` is **opt-in** because it requires the cluster to be live:
Slurm node names are IP-derived, so reprovisioning a cluster into a new subnet
renames every node at once. It catches exactly the failure that motivated it —
a hostname left behind from a previous cluster, which otherwise surfaces only as
`sbatch: Invalid node name specified` partway through a deploy. (`dev` and `prod`
currently fail it: both point at a decommissioned `10.0.*` subnet.)

The generator refuses unsafe or inconsistent configurations, including:

- a pool member that is not listed in `workers`;
- overlap between `sysbox_pool` and `cpu_isolation_pool`;
- a control-plane host also listed as a worker;
- empty workers, duplicate hosts, unknown fields, or shell-unsafe hostnames;
- two clusters sharing a job-name prefix, `state_db`, `control_log`, or tunnel
  port — each is shared infrastructure, so a collision would have one cluster
  clobber another;
- with `--validate-slurm`, any host that is not a live Slurm node.

Both pools may be empty. CPU isolation is intentionally not written to the
generated nodes YAML; node agents discover and advertise that capability.

## Consumer `.env` sync (`--env-cluster`)

The **consumer** path — `xrlenv.from_env()` and the benchmark oracle sweeps —
reads the checkout's `.env` for `XRLENV_GRPC_HOST` / `XRLENV_GRPC_PORT` and the
registry hosts/ports to dial the control plane. Those must track `clusters.yaml`
too, or a post-reboot sweep points at a dead control plane. `--env-cluster`
syncs them:

```bash
# in the dev checkout:
.venv/bin/python slurm_scripts/generate_deployment_script.py --env-cluster dev
```

It rewrites only the topology **host/port** lines
(`XRLENV_GRPC_HOST`, `XRLENV_{MIRROR,PRIVATE}_REGISTRY_HOST`/`_PORT`) in the
local `./.env`, appending any that are missing. Every **secret** line
(`*_TOKEN`, `DOCKERHUB_*`, `*_HTTP_SECRET`) and `XRLENV_GRPC_PORT` is left
byte-for-byte untouched — secrets are reboot-invariant and stay hand-maintained.
The generator only ever writes the **local** checkout's `.env`; prod's `.env`
lives in a separate checkout, so run the generator there with
`--env-cluster prod` to sync it. In `--check` mode, `.env` drift is reported
without printing the file (it contains secrets).

Use check mode in CI or before deployment to detect drift without writing:

```bash
.venv/bin/python slurm_scripts/generate_deployment_script.py --check
.venv/bin/python slurm_scripts/generate_deployment_script.py --env-cluster dev --check
```

The control job continues to run `xrlenv nodes-from-slurm` at startup, so
`generated/{dev,prod}_hyperpod_nodes.yaml` remains a runtime-generated inventory.

## Registry reconciliation across reboots

Because a node's identity (`node_id = aws-<hostname>`) is IP-derived, every
reboot mints new node_ids and orphans the previous set. The control plane
reaps them automatically: on each `xrlenv up`, after loading the roster, it
prunes `lost` node rows whose node_id is absent from `nodes.yaml` (the
projection of `clusters.yaml`). `connected` and still-rostered nodes are kept.
So the `xrlenv nodes` / admin panel shows only the current fleet — no manual
cleanup of `lost` rows from prior reboots.

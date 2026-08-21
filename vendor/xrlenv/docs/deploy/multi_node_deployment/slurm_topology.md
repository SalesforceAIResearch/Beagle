# Slurm/HyperPod topology management

This page covers the `slurm_scripts/` deploy toolchain used to keep
Slurm and HyperPod cluster deployments in sync after reboots. It is
relevant only to operators running xrlenv on a managed Slurm cluster;
generic cloud-VM operators follow {doc}`runbook` instead.

## How it fits together

The workflow for Slurm deployments uses three moving parts:

- **`slurm_scripts/clusters.yaml`** — the single source of truth for
  reboot-sensitive cluster topology (control-plane hostname, registry
  endpoints, worker list, Sysbox pool, CPU-isolation pool). Edit this
  file after any reboot that reassigns IP addresses.
- **`slurm_scripts/generate_deployment_script.py`** — reads
  `clusters.yaml` and stamps the topology into the generated SBATCH
  scripts and the local checkout's `.env`. Run this after every
  `clusters.yaml` edit.
- **`xrlenv nodes-from-slurm`** — run by the control plane at startup to
  produce the live `nodes.yaml` roster from the Slurm job allocation. This
  is what `--admin-nodes-yaml` and the startup registry reconciliation
  (see {doc}`inventory`) consume.

## `clusters.yaml` schema

```yaml
dev:
  control_plane: node-host        # Slurm hostname of the control-plane job
  registry:
    mirror_host: node-host         # pull-through mirror (:5010)
    private_host: node-host        # private registry (:5011)
    mirror_port: 5010                   # optional; defaults to 5010
    private_port: 5011                  # optional; defaults to 5011
  workers:
    - node-host
    - node-host
  sysbox_pool:
    - node-host
  cpu_isolation_pool:
    - node-host

prod:
  control_plane: node-host
  registry:
    mirror_host: node-host
    private_host: node-host
  workers:
    - node-host
    # ...
  sysbox_pool:
    - node-host
  cpu_isolation_pool:
    - node-host
```

| Field | Required | Default | Description |
|---|---|---|---|
| `control_plane` | yes | — | Slurm hostname that runs `xrlenv up`. |
| `registry.mirror_host` | yes | — | Host serving the pull-through image mirror. |
| `registry.private_host` | yes | — | Host serving the private image registry. |
| `registry.mirror_port` | no | `5010` | Mirror registry port. |
| `registry.private_port` | no | `5011` | Private registry port. |
| `workers` | yes | — | Ordered worker-node hostnames. |
| `sysbox_pool` | yes | — | Subset of `workers` that have Sysbox installed (may be empty). |
| `cpu_isolation_pool` | yes | — | Subset of `workers` that have CPU isolation enabled (may be empty). Both pools may be empty; they must not overlap with each other or with `control_plane`. |

The generator validates the config before writing anything: it rejects
pools whose members are not in `workers`, overlapping pools,
a `control_plane` host listed as a worker, empty worker lists, duplicate
hostnames, and shell-unsafe characters in hostnames.

## Regenerate after a reboot

After a reboot that changes hostnames:

1. Edit `slurm_scripts/clusters.yaml` with the new hostnames.

2. Regenerate the SBATCH scripts and the consumer `.env`:

   ```bash
   # Regenerate all clusters and sync your dev checkout's .env:
   .venv/bin/python slurm_scripts/generate_deployment_script.py --env-cluster dev

   # Update only one cluster:
   .venv/bin/python slurm_scripts/generate_deployment_script.py --cluster dev --env-cluster dev
   ```

3. If you also run sweeps from the prod checkout (`/path/to/xrlenv`),
   sync its `.env` from that checkout:

   ```bash
   cd /path/to/xrlenv
   .venv/bin/python slurm_scripts/generate_deployment_script.py --env-cluster prod
   ```

4. Redeploy the cluster (the control plane bounce triggers startup
   registry reconciliation, reaping orphaned node rows automatically).

### What the generator rewrites

The generator updates only topology-bearing lines in:

- `slurm_scripts/deploy_{dev,prod}.sh`
- `slurm_scripts/{dev,prod}_xrlenv_node.sh`
- `slurm_scripts/{dev,prod}_xrlenv_control.sh`

All non-topology script behavior is left untouched.

## Consumer `.env` sync (`--env-cluster`)

The consumer path — `xrlenv.from_env()` and benchmark oracle sweeps —
reads the checkout's `.env` for the control-plane address and registry
hosts to dial. These must stay in sync with `clusters.yaml`, or
a post-reboot sweep points at a dead control plane.

`--env-cluster` rewrites only the topology host/port lines:

- `XRLENV_GRPC_HOST`
- `XRLENV_MIRROR_REGISTRY_HOST` / `XRLENV_MIRROR_REGISTRY_PORT`
- `XRLENV_PRIVATE_REGISTRY_HOST` / `XRLENV_PRIVATE_REGISTRY_PORT`

Every secret line (`*_TOKEN`, `DOCKERHUB_*`, `*_HTTP_SECRET`) and
`XRLENV_GRPC_PORT` is left byte-for-byte untouched — those values are
reboot-invariant and stay hand-maintained.

## Check mode (CI / pre-deploy drift detection)

Run with `--check` to detect drift without writing any files:

```bash
# Check whether generated scripts are stale:
.venv/bin/python slurm_scripts/generate_deployment_script.py --check

# Also check the local .env:
.venv/bin/python slurm_scripts/generate_deployment_script.py --env-cluster dev --check
```

`--check` exits nonzero if any file is out of date. In `.env` check
mode, drift is reported without printing the file contents (the file
holds secrets).

## See also

- {doc}`inventory` — `nodes.yaml` schema and startup registry
  reconciliation, which reaps orphaned node rows after a reboot.
- {doc}`runbook` — end-to-end deployment runbook for non-Slurm deployments.
- For the full `slurm_scripts/` reference, see `slurm_scripts/README.md`
  in the repository root.

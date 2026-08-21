# Slurm cluster deployment

`clusters.yaml` is the source of truth for reboot-sensitive cluster
topology. Each cluster has these fields:

- `control_plane`: the Slurm hostname running the control-plane job.
- `registry`: the image-registry endpoints for this cluster —
  `mirror_host` + `private_host` (required) and `mirror_port` /
  `private_port` (optional; default `5010` / `5011`). Both hosts point at
  the same shared box today to save resources, but a cluster may run its
  own registry on its own ports.
- `workers`: the ordered worker allocation.
- `sysbox_pool`: workers on which the deploy script installs/enables Sysbox.
- `cpu_isolation_pool`: workers on which the deploy script enables CPU isolation.

After a cluster reboot, update the hostnames and pool assignments in
`clusters.yaml`, then synchronize the deployment scripts:

```bash
.venv/bin/python slurm_scripts/generate_deployment_script.py
```

To update only one cluster:

```bash
.venv/bin/python slurm_scripts/generate_deployment_script.py --cluster example
```

The generator updates only topology-bearing lines in:

- `deploy_example.sh`
- `example_xrlenv_node.sh`
- `example_xrlenv_control.sh`

It derives the SBATCH node count from `workers`, emits a comma-separated SBATCH
nodelist, and uses the same control-plane hostname for the deploy SSH tunnel,
worker gRPC target, control allocation, and control gRPC bind address. All
non-topology script behavior remains untouched.

The generator refuses unsafe or inconsistent configurations, including:

- a pool member that is not listed in `workers`;
- overlap between `sysbox_pool` and `cpu_isolation_pool`;
- a control-plane host also listed as a worker;
- empty workers, duplicate hosts, unknown fields, or shell-unsafe hostnames.

Both pools may be empty. CPU isolation is intentionally not written to the
generated nodes YAML; node agents discover and advertise that capability.

## Consumer `.env` sync (`--env-cluster`)

The **consumer** path — `xrlenv.from_env()` and the benchmark oracle sweeps —
reads the checkout's `.env` for `XRLENV_GRPC_HOST` / `XRLENV_GRPC_PORT` and the
registry hosts/ports to dial the control plane. Those must track `clusters.yaml`
too, or a post-reboot sweep points at a dead control plane. `--env-cluster`
syncs them:

```bash
# in the cluster checkout:
.venv/bin/python slurm_scripts/generate_deployment_script.py --env-cluster example
```

It rewrites only the topology **host/port** lines
(`XRLENV_GRPC_HOST`, `XRLENV_{MIRROR,PRIVATE}_REGISTRY_HOST`/`_PORT`) in the
local `./.env`, appending any that are missing. Every **secret** line
(`*_TOKEN`, `DOCKERHUB_*`, `*_HTTP_SECRET`) and `XRLENV_GRPC_PORT` is left
byte-for-byte untouched — secrets are reboot-invariant and stay hand-maintained.
The generator only ever writes the **local** checkout's `.env`; another cluster's `.env`
lives in a separate checkout, so run the generator there with
`--env-cluster example` to sync it. In `--check` mode, `.env` drift is reported
without printing the file (it contains secrets).

Use check mode in CI or before deployment to detect drift without writing:

```bash
.venv/bin/python slurm_scripts/generate_deployment_script.py --check
.venv/bin/python slurm_scripts/generate_deployment_script.py --env-cluster example --check
```

The control job continues to run `xrlenv nodes-from-slurm` at startup, so
`example_hyperpod_nodes.yaml` remains a runtime-generated inventory.

## Registry reconciliation across reboots

Because a node's identity (`node_id = aws-<hostname>`) is IP-derived, every
reboot mints new node_ids and orphans the previous set. The control plane
reaps them automatically: on each `xrlenv up`, after loading the roster, it
prunes `lost` node rows whose node_id is absent from `nodes.yaml` (the
projection of `clusters.yaml`). `connected` and still-rostered nodes are kept.
So the `xrlenv nodes` / admin panel shows only the current fleet — no manual
cleanup of `lost` rows from prior reboots.

# `deploy/`

Everything for standing up and operating an xrlenv fleet **by hand** on
freshly-provisioned VMs — the user has VM-only access on GCP/AWS (no Terraform,
no managed instance groups), so deployment is these shell scripts plus a static
`nodes.yaml`. For the full narrative see [`docs/deploy/`](../docs/deploy/); this
is the directory map.

## Bring-up flow (control plane + workers)

| Script | Used when |
|---|---|
| `bring-up-node.sh` | One-shot bring-up for a freshly provisioned VM. |
| `bootstrap-aws.sh` / `bootstrap-gcp.sh` | Thin wrappers around `xrlenv bootstrap --target {aws,gcp}` — install the node agent + deps on a worker. |
| `bootstrap-common.sh` | Shared install logic sourced by the per-cloud bootstraps. |
| `_preflight.sh` | Log helpers + required-env validation, sourced by the bootstraps before package installs. |
| `refresh.sh` | Fast path after `git pull` on a live VM: push the changes into the running `xrlenv-node` service (no full re-bootstrap). |
| `ship-images.sh` | Build-once-ship-many image-distribution recipe. |

## Subdirectories

- **[`registry/`](registry/README.md)** — the three registry servers (mirror
  `:5010`, private `:5011`, scratch `:5012`), their configs, and the operator
  scripts that build / warm / GC / retag their content. Its README has the full
  mirror + private + build-push story.
- **`node/`** — per-node provisioning helpers (table below).
- **`systemd/`** — the `xrlenv-node` + `xrlenv-cpu-isolation` unit files.

## `node/` — node provisioning

Run on a worker (mostly via `sudo`). The bootstrap invokes the first three; run
any by hand to fix a live node.

| Script | Used when | Cross-referenced / used in |
|---|---|---|
| `set_docker_data_root.sh` | Point Docker's data-root at the node's large EBS volume (HyperPod `/opt/sagemaker`, ~500 GB) instead of the ~97 GB root disk. | `deploy/bootstrap-aws.sh`; `deploy/registry/run-registry-mirror.sh` |
| `cleanup_cuda_cpu_node.sh` | Provisioning a CPU-only node from the Deep Learning AMI: strip ~41 GB of unused CUDA to free root disk. Run once per node. | _(standalone — no automated caller)_ |
| `enable_cpu_isolation.sh` | Opt a node into CPU isolation (cgroup-v2 cpuset delegation for the non-root agent). Bounces docker + the agent — run in a maintenance window. | `deploy/bootstrap-{aws,gcp}.sh`; `deploy/systemd/xrlenv-cpu-isolation.service`; `setup_shared_cpuset.sh`; `docs/technical_details/resource_isolation.md` |
| `setup_shared_cpuset.sh` | (Re)establish the delegated `/sys/fs/cgroup/xrlenv-shared` parent cpuset (cgroups don't survive reboot). Invoked by the cpu-isolation systemd unit / `enable_cpu_isolation.sh`. | `deploy/systemd/xrlenv-cpu-isolation.service`; `enable_cpu_isolation.sh` |

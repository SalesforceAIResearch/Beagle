# Drop-in for code that already uses docker-py

If your code already uses docker-py (`docker.from_env()` at the
top), the cleanest path onto an xrlenv cluster is a one-line swap:

```diff
-import docker
-client = docker.from_env()
+import xrlenv
+client = xrlenv.from_env()
```

Everything else flows through unchanged. The drop-in intercepts
docker-py's manager-level surface and routes each call to a
scheduler-picked node in the cluster.

## Connection (env vars)

`xrlenv.from_env()` reads connection config from the same kind of
environment-variable protocol `docker.from_env()` uses for
`DOCKER_HOST`:

| Variable | Required | Description |
|---|---|---|
| `XRLENV_GRPC_HOST` | yes | Control-plane host. |
| `XRLENV_GRPC_PORT` | no (default `50051`) | Control-plane port. |
| `XRLENV_CONSUMER_TOKEN` | when the control plane runs with auth | Bearer token from `xrlenv tokens issue consumer`. |
| `XRLENV_GRPC_SECURE` | no (default `false`) | Set to `true` / `1` / `yes` / `on` for TLS. |

Set them once in your shell (typically alongside `xrlenv up` on
the control-plane host); your code reads them implicitly.

## A worked example

```python
import io
import tarfile
import xrlenv

client = xrlenv.from_env()  # reads XRLENV_GRPC_HOST etc.

# Create + start a container.
container = client.containers.create(
    image="ubuntu:22.04",
    command=["sleep", "infinity"],
    detach=True,
)
container.start()

# Run a command inside it.
exit_code, output = container.exec_run(
    ["bash", "-c", "ls /"],
    demux=False,
)

# Copy bytes in.
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as tf:
    info = tarfile.TarInfo(name="hello.txt")
    payload = b"hello\n"
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))
container.put_archive(path="/work", data=buf.getvalue())

# Stream stdout from a long command.
for chunk in container.exec_run(
    ["bash", "-c", "for i in 1 2 3; do echo $i; sleep 1; done"],
    stream=True,
)[1]:
    print(chunk.decode(), end="")

# Tear down.
container.stop()
container.remove()
```

## What's wired

In cluster mode the drop-in covers the manager-level surface that
SWE-bench-shaped harnesses use:

- `client.containers.create(...)` / `containers.run(...)`
- `container.exec_run(...)` (batched + streaming via
  `stream=True`)
- `container.put_archive(...)` / `container.get_archive(...)`
- `container.start()` / `container.stop()` / `container.remove()`
- `client.images.get(...)` / `images.pull(...)` /
  `images.list(...)` / `images.remove(...)` / `image.history()`

Methods not yet wired raise `NotImplementedError` with a clear
message rather than failing on uninitialised state. The full
coverage list lives in
`xrlenv/compat/docker_client.py:_CLUSTER_OVERRIDES`.

## Resource limits

The resource limits docker-py callers pass through `host_config`
(`mem_limit`, `nano_cpus`, `cpu_quota`, `pids_limit`, …) are honored
in cluster mode — a harness that caps CPU/memory against a local
Docker daemon gets the same cap on the cluster.

```python
client = xrlenv.from_env()
container = client.containers.run(
    image="my-grader:1",
    command=["sleep", "infinity"],
    detach=True,
    nano_cpus=4_000_000_000,      # 4 CPU
    mem_limit="8g",
    pids_limit=4096,
)
```

- **CPU** (`nano_cpus`, or `cpu_quota`+`cpu_period`) and **memory**
  (`mem_limit`) become scheduling inputs — the cluster places the
  container on a node that can satisfy them, applies a CFS quota, and
  pins `ceil(cpu_limit)` dedicated cores.
- **`pids_limit` / `shm_size` / `tmpfs` / `read_only`** are applied at
  container creation (no scheduling effect).
- A few kwargs are **rejected with a clear error** rather than
  silently dropped — `cpu_shares` and `mem_reservation` (soft limits,
  no hard isolation), and `cpuset_cpus` / `cgroup_parent` (CPU
  placement and the cgroup hierarchy are cluster-owned).

See {doc}`/technical_details/resource_isolation` for the full table
of honored vs. rejected kwargs and the operator-side knobs.

## Worked example in tree

[`xrlenv_plugins/benchmarks/swebench_verified/run_oracle_sweep.py`](https://github.com/Yutong-Dai/XRLEnv/blob/main/xrlenv_plugins/benchmarks/swebench_verified/run_oracle_sweep.py)
drives upstream SWE-bench through the drop-in end-to-end. The
sweep driver contains literally one xrlenv-specific line:
`client = xrlenv.from_env()`. See
{doc}`/supported_benchmarks_and_harnesses/swe_bench` for the full
operator walkthrough including cache setup and the
`run_full_sweep.sh` entrypoint.

## OCI runtime selection (`runtime=`)

Same rule as standard docker-py: **omit `runtime=` for a normal container**
(docker's default `runc` — unchanged), and pass **`runtime="sysbox-runc"` to get
unprivileged Docker-in-Docker / systemd / netns**. There is no separate DinD
flag — the Sysbox runtime *is* the DinD mechanism. The drop-in honors both the
top-level `runtime=` kwarg and `HostConfig(Runtime="sysbox-runc")`, and forwards
the selector to the cluster:

```python
client = xrlenv.from_env()

# Normal container — no runtime= (unchanged behavior):
client.containers.run(image="swe-task:1", detach=True)

# Docker-in-Docker / systemd — opt into Sysbox:
container = client.containers.run(
    image="my-dind-task:1",
    command=["sleep", "infinity"],
    detach=True,
    runtime="sysbox-runc",
)
```

The same policy, placement, and egress rules apply as for the direct API (see
{doc}`direct_api`). The operator must have set up a Sysbox node pool and added
`sysbox-runc` to `allowed_runtimes` in `nodes.yaml` before these calls can
succeed.

## When to pick this over the direct API

- You have existing docker-py code you don't want to rewrite.
- Your harness's upstream library uses docker-py internally
  (e.g. `swebench.harness.run_evaluation`) and you'd rather swap
  one line than fork the library.
- Synchronous shape fits your code better than async.

If you're starting fresh and have no docker-py history to
preserve, the direct API
({doc}`direct_api`) is a smaller surface with the
same primitive underneath.

## See also

- {doc}`/supported_benchmarks_and_harnesses/swe_bench` —
  pre-wired SWE-bench operator docs (uses this same drop-in
  pattern).
- {doc}`direct_api` — the alternative shape using
  `Client.acquire_container` directly.

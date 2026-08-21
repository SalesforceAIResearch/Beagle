# Writing your own adapter

Start with the benchmark's existing execution model. A good XRLEnv
adapter should let the upstream harness keep parsing tasks, driving
agents, grading outputs, and writing reports. Replace only the part
that starts containers or runs commands inside them.

## Choose the integration shape

| Benchmark shape | Recommended XRLEnv path |
|---|---|
| It already imports `docker` and calls `docker.from_env()` | Use the Docker SDK drop-in: swap to `xrlenv.from_env()`. |
| It loads an environment/provider class from config | Write a framework/harness adapter that subclasses the framework interface. |
| You control the code and do not need docker-py compatibility | Use `Client.acquire_container(...)` directly. |
| It invokes `docker` as a raw CLI subprocess | Write an in-process subprocess interceptor. |

## Docker SDK drop-in

Use this when the upstream code is docker-py-shaped. The drop-in
covers the manager methods used by SWE-bench-style harnesses:
container create/start/stop/remove, batched and streaming exec,
archive upload/download, and common image methods.

See {doc}`swe_bench` and
{doc}`/build_with_xrlenv/work_with_xrlenv_managed_containers/docker_py_dropin`.

## Framework/harness adapter

Use this when the benchmark already has an environment abstraction.
The adapter should:

1. Subclass the framework's base environment/provider class.
2. Override only container operations such as start, stop, exec,
   upload, and download.
3. Build an `xrlenv.Client` from `XRLENV_GRPC_HOST`,
   `XRLENV_GRPC_PORT`, `XRLENV_CONSUMER_TOKEN`, and
   `XRLENV_GRPC_SECURE`.
4. Populate `xrlenv.rollout_metadata(...)` from framework concepts
   such as trial id, session id, or artifact directory so the admin
   panel shows useful names and links.

Harbor's adapter is the current worked example:
[`xrlenv_plugins/harbor/environment.py`](https://github.com/Yutong-Dai/XRLEnv/blob/main/xrlenv_plugins/harbor/environment.py).

See {doc}`harbor_framework`.

## Direct API

If there is no upstream Docker SDK or framework interface to preserve,
skip the adapter layer and call XRLEnv directly. This is usually the
smallest and clearest path for new code.

See
{doc}`/build_with_xrlenv/work_with_xrlenv_managed_containers/direct_api`.

## Raw-subprocess / CLI-shim benchmarks

Use this when the benchmark invokes the `docker` binary as a raw
subprocess — `subprocess.run(["docker", "exec", …])`,
`subprocess.Popen(["docker", "run", …])` — rather than importing
docker-py. The drop-in (`xrlenv.from_env()`) only intercepts docker-py
calls; a PATH-level shim binary would also fail because xrlenv cluster
sessions live in-process and cannot be shared across fresh child
interpreters.

The correct approach is an **in-process subprocess interceptor**: a
module that monkeypatches `subprocess.run`/`Popen` so that any argv
beginning with `docker` is routed to the cluster via the compat client,
while all other subprocess calls pass through unchanged. Because the
interceptor runs in the same process that holds the cluster sessions,
every `docker exec`, `docker cp`, and `docker rm` can look up the
container the preceding `docker run` opened.

Key design rules for the interceptor:

- **Fail loud on uncovered flags.** Enumerate only the docker
  subcommands and flags the upstream harness actually uses; raise on
  anything unrecognised rather than silently dropping a capability.
- **Namespace container names.** Benchmarks often use a fixed
  `--name` per task. Under concurrent rollouts, unprefixed names
  collide. The interceptor should prefix names automatically so
  the upstream code is untouched.
- **Handle bind mounts as `put_archive`.** The cluster node has no
  host paths, so `-v host_path:/ctr_path` binds must be resolved on
  the consumer machine and uploaded via `put_archive`.
- **Wrap `run_e2e`, not the top-level multi-repo orchestrator.** If
  the benchmark spawns per-repo workers as detached subprocesses, the
  monkeypatch is not inherited. Install the shim inside the per-repo
  entry point.

{doc}`evoclaw` is the worked example: EvoClaw drives docker entirely
through raw subprocesses, and its onboarding (`docker_shim.py` +
`run_e2e_xrlenv.py`) demonstrates the full pattern including streaming
exec, bind-mount translation, name namespacing, and image-name
reconciliation.

## What not to reimplement

Do not copy benchmark grading, report generation, or task parsing
into XRLEnv code. If upstream publishes an API or filesystem contract
for those pieces, call it or preserve it. The adapter should make the
container run somewhere else; it should not become a fork of the
benchmark.

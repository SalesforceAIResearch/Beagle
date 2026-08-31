# Work with XRLEnv-managed containers

Use this section when you want code that says: give me a container,
run commands, copy files in or out, then throw it away. XRLEnv routes
that work to a local or remote Docker host and records the lifecycle
for operators.

Pick the shape that matches your code:

| If your code… | Use this | Page |
|---|---|---|
| …already uses docker-py (`docker.from_env()` at the top), or builds on a library that does | one-line swap to `xrlenv.from_env()` | {doc}`docker_py_dropin` |
| …is fresh, no docker-py history, async-friendly | direct `Client.acquire_container` | {doc}`direct_api` |

Both ride on the same primitive: a `ClusterContainerSession` scoped to
one XRLEnv rollout id. You can mix the direct API and the drop-in in
different modules if that keeps the surrounding code simpler.

```{note}
This section is for custom workflows: your own harness, an ad-hoc
evaluation script, or a benchmark wrapper that does not already have
a supported adapter. If your benchmark is listed under
{doc}`/supported_benchmarks_and_harnesses/index`, the pre-wired path
usually saves work.
```

```{toctree}
:maxdepth: 1

docker_py_dropin
direct_api
```

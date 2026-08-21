# `xrlenv_plugins.harbor` — harbor framework adapter

Worked reference for the **`Xrlenv<Framework><BaseRole>` plug-in
pattern**: how an upstream RL framework with its own
`BaseEnvironment` / `BaseAgent` / `Provider` / etc. Protocol gets
adapted to xrlenv's primitives without putting framework-specific
maintenance into xrlenv core.

## What's here

- `environment.py` —
  - `XrlenvHarborEnvironment(harbor.environments.docker.docker.DockerEnvironment)`:
    LocalDocker shape. Subclass that satisfies harbor's
    `BaseEnvironment` Protocol while exposing xrlenv-specific kwargs
    (`xrlenv_task_key`, `xrlenv_group_id`, `xrlenv_resources`,
    `xrlenv_image_pin_mode`, …) for observability.
  - `XrlenvHarborEnvironmentCluster(XrlenvHarborEnvironment)`:
    cluster-routed shape (P1.7.C.1). Overrides
    `start`/`stop`/`exec`/`upload_*`/`download_*` to call the xrlenv
    cluster primitives instead of local `docker compose` + `docker
    cp`.

## LocalDocker mode

Pick `XrlenvHarborEnvironment` when you want harbor's stock
single-host behavior with the xrlenv-kwargs recorded on the
instance for observability. No env vars required, no control plane
required — runs against `DOCKER_HOST` like harbor's own
`DockerEnvironment`.

```yaml
# job.yaml
environment:
  import_path: xrlenv_plugins.harbor:XrlenvHarborEnvironment
```

## Cluster mode

Pick `XrlenvHarborEnvironmentCluster` when you want harbor's trial
flow to run on a remote, xrlenv-scheduled node — the same UX shape
harbor users already know from picking `e2b`, `modal`, or
`daytona`.

```yaml
# job.yaml
environment:
  import_path: xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster
```

**Required env (set on the consumer side before launching the
harness):**

| Variable | Required | Description |
|---|---|---|
| `XRLENV_GRPC_HOST` | yes | Control-plane host. Symmetric with the docker-py drop-in's `xrlenv.from_env()`. |
| `XRLENV_GRPC_PORT` | no (default `50051`) | Control-plane port. |
| `XRLENV_CONSUMER_TOKEN` | when the control plane runs with auth | Bearer token from `xrlenv tokens issue consumer`. |
| `XRLENV_GRPC_SECURE` | no (default `false`) | Set to `true` / `1` / `yes` / `on` for TLS. |

The cluster Environment lazy-constructs an `xrlenv.Client` from
those env vars on first `start()`. No new harbor-side kwargs.

**Image distribution (P1.7.C.1, staged):**

Images must be pre-built on each cluster node before consumers
acquire. The lookup tag is either `task_env_config.docker_image`
(when the upstream task ships a prebuilt) or `hb__<environment_name>`
(harbor's local-build convention).

For terminal-bench-2: pre-build via
`examples/benchmarks-onboarding/terminal-bench-2/scripts/build-task-images.sh`
on each node. Missing-image acquires fail fast with a clear
`ImageNotFound` rather than hanging.

Real build-on-acquire (`HarborImageBuilder` registered against the
control-plane build flow + acquire→build→re-acquire fallback) is
**P1.7.C.2** — out of scope for this slice. The user-facing UX gap
is one log line; the production "build if missing" comes one slice
later.

**Single-service only (P1.7.C.1, staged):**

Multi-service compose tasks (a few harbor tasks attach a `db` /
`redis` helper) are out of scope for this slice. The cluster
overrides assume a single `main` service. Multi-service support
defers to a follow-on slice.

**`is_mounted=False`:**

Cluster mode never bind-mounts host paths into the container — the
consumer's host isn't the node's host. harbor's trial driver checks
`is_mounted` and switches to the post-trial `download_dir` branch
when it's `False`, which is exactly what we want. Per-trial outputs
end up under harbor's normal `trial_paths` after the trial ends.

**Validation:**

`examples/benchmarks-onboarding/terminal-bench-2/smoke.py` drives
harbor's runner against this adapter through 8 phase-0 tasks
(`fix-git`, `build-pov-ray`, `overfull-hbox`, …). See that
directory's README for end-to-end usage.

## The pattern: writing your own framework adapter

Other RL frameworks' adapters follow the same shape.

### Naming convention

| Framework | Adapter module | Adapter class | Subclasses |
|---|---|---|---|
| harbor | `xrlenv_plugins.harbor` | `XrlenvHarborEnvironment` | `harbor.BaseEnvironment` (via `DockerEnvironment`) |
| (hypothetical) foo | `xrlenv_plugins.foo` | `XrlenvFooAgent` | `foo.BaseAgent` |
| (hypothetical) bar | `xrlenv_plugins.bar` | `XrlenvBarProvider` | `bar.Provider` |

`Xrlenv<Framework><BaseRole>`. The `<BaseRole>` reflects whatever
the upstream framework's plug-in interface is named (Environment,
Agent, Provider, Runner, …). The framework name disambiguates so
two plug-ins for two different frameworks don't collide on import.

### Where it lives

- Inside this repo: `xrlenv_plugins/<framework>/`. Reference
  implementations xrlenv ships and validates via `tests/smoke/`.
- Outside this repo: a separate pip package using B11's entry-point
  mechanism (`[project.entry-points."xrlenv.benchmarks"]`) — same
  PEP-420 namespace, third-party code, no fork required.

Pick "outside" if your adapter has framework-specific dependencies
xrlenv shouldn't carry. Pick "inside" if it's broadly useful and
you're willing to contribute a PR.

### Subclass `<framework>.<concrete>` not `<framework>.<base>` when possible

The harbor plug-in subclasses `DockerEnvironment` (concrete) rather
than `BaseEnvironment` (abstract). Abstract Protocols have lots of
abstract methods; subclassing the concrete class inherits the
heavy lifting and lets you override only the seams that need
xrlenv-specific behavior. Same trade-off as in
`xrlenv.compat.docker_client`: subclass the upstream layer, override
selectively, leave everything else inherited so the contract stays
intact for free.

### Define a routing seam

Carve out one or two methods where xrlenv-specific routing happens
(in this plug-in: `_xrlenv_route_command`). Default behavior is
pass-through (LocalDocker mode). Cluster-mode follow-on overrides
that one seam. Keeps the spike → cluster-mode evolution mechanical.

### Carry xrlenv kwargs through the constructor

Pop them before calling `super().__init__()` (harbor / docker-py /
most upstream classes reject unknown kwargs). Record them on the
instance as `self._xrlenv_kwargs` for observability. Cluster-mode
routing will read them off the instance.

The canonical xrlenv kwargs:

- `xrlenv_task_key` — anti-affinity grouping
- `xrlenv_group_id` — cancellation cohort
- `xrlenv_resources` — scheduler input (`ResourceSpec`)
- `xrlenv_image_pin_mode` — spec-19 audit input
- `xrlenv_owner_id` / `xrlenv_project_id` / `xrlenv_run_id` —
  multi-tenancy + state-store accounting

### Validate via `tests/smoke/`

Land a smoke test that runs ONE task end-to-end through the
upstream harness pointed at your adapter. Pin
`assert <upstream-success-signal>` so a future change to your
adapter can't silently break the contract. See
`tests/smoke/test_terminal_bench_2_drop_in.py` for the harbor
equivalent.

## Why not in `xrlenv/compat/`?

`xrlenv/compat/docker_client.py` adapts the *universal Python
Docker SDK* — every consumer that ever touches Docker via Python
goes through docker-py. One in-tree shim serves the whole
ecosystem.

This adapter (and any other framework adapter) is one of many. If
xrlenv core grew to hold harbor's adapter + foo's adapter + bar's
adapter + …, we'd be back to the per-benchmark integration debt
the slim pivot was supposed to escape. `xrlenv_plugins/` keeps
framework adapters out of xrlenv's core maintenance loop while
still discoverable as installable packages.

## See also

- `xrlenv/compat/docker_client.py` — the universal-substrate
  counterpart for docker-py users.
- `tests/smoke/test_terminal_bench_2_drop_in.py` — end-to-end
  validation that this adapter drives a real harbor task through.
- `xrlenv_plugins/__init__.py` (PEP-420 namespace package) — how
  third-party plug-ins coexist with in-tree ones.

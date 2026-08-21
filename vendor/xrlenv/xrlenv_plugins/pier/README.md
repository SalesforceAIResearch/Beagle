# `xrlenv_plugins.pier` — pier framework adapter

xrlenv adapter for [**pier**](https://github.com/datacurve-ai/pier) (PyPI
`datacurve-pier`), the harness for the **DeepSWE** benchmark. pier is a harbor
fork that reimplements the trial/verifier harness in-tree (it does **not** import
harbor at runtime), so this package is the direct analog of
`xrlenv_plugins.harbor` retargeted at pier's classes — it subclasses
`pier.environments.docker.docker.DockerEnvironment`, not harbor's.

See `xrlenv_plugins/harbor/README.md` for the canonical
`Xrlenv<Framework><BaseRole>` plug-in pattern; this package follows it.

## What's here

| File | Role |
|---|---|
| `environment.py` | `XrlenvPierEnvironment` (LocalDocker) + `XrlenvPierEnvironmentCluster` (cluster-routed) |
| `compose.py` | multi-service compose helpers — **verbatim copy** of `harbor/compose.py` (pure `yaml`+stdlib, framework-free) |
| `__init__.py` | lazy PEP-562 re-export of the two env classes |
| `tests/` | unit tests for the pure logic + construction/resolution seams |

## Selecting it

pier ships a first-class `import_path` escape hatch, so it's selected exactly the
way pier's built-in `docker`/`modal`/`daytona` environments are — no PR to pier:

```yaml
# pier job config
environment:
  import_path: xrlenv_plugins.pier:XrlenvPierEnvironmentCluster
```

or `pier run --environment-import-path xrlenv_plugins.pier:XrlenvPierEnvironmentCluster`,
or programmatically `EnvironmentConfig(import_path="xrlenv_plugins.pier:XrlenvPierEnvironmentCluster")`.

**Cluster opt-in env** (same protocol as the harbor adapter): `XRLENV_GRPC_HOST`
(required), `XRLENV_GRPC_PORT` (default 50051), `XRLENV_CONSUMER_TOKEN` (if the CP
has auth), `XRLENV_GRPC_SECURE`. Image ref precedence in `_resolve_image_ref()`:
`XRLENV_PIER_IMAGE_TEMPLATE` > task `docker_image` > (verifier-session fallback) >
`hb__<env>`.

## pier-specific deltas vs the harbor adapter

The cluster overrides (acquire/exec/transfer/pacing/sysbox/compose) port over
essentially verbatim. The behavioural differences are:

1. **`type()`** — pier's `BaseEnvironment` requires a `str` identifier; we return
   `"xrlenv-cluster"` (harbor keyed off an enum).
2. **`agent_install_spec` cleared in cluster mode** — the cluster builds no
   agent-preinstalled image, so we drop `agent_install_spec` (and advertise
   `capabilities.preinstall_agents=False`) so pier runs each installed agent's
   runtime `install()` via our `exec` instead of assuming a preinstalled binary.
   LocalDocker keeps it (it can genuinely build a preinstalled image). Moot for
   the OracleAgent; required for the installed-agent path.
3. **Separate-verifier seam (DeepSWE `environment_mode="separate"`)** — pier
   builds the verifier env from the task's `[verifier.environment]` (which in
   DeepSWE has **no** `docker_image`) and hardcodes `Verifier(skip_tests_upload=
   True)`. So `start()`, when it detects a verifier session
   (`"__verifier__" in session_id`):
   - resolves the base image from the verifier `tests/Dockerfile`'s `FROM` (or the
     parent task's top-level `[environment].docker_image`) — not `hb__<env>`;
   - uploads the tests build context to `/tests` (reproducing the `tests/Dockerfile`
     COPY) so the grader is present. No on-node build.
4. **`capabilities`** — `mounted=False` (round-trips the reward file through
   `download_dir`), `preinstall_agents=False`, `filtered_egress=False` (until the
   Squid egress proxy lands — see the plan §4b).

## Network / egress

For the DeepSWE **oracle** gate, offline tasks (`allow_internet=false`) acquire
`--network none`; no allowlist is needed. The pier **agent** network-allowlist
(egress proxy) is a separate capability build (reproduce pier's Squid sidecar on
the cluster compose path) — tracked in `notes/deep-swe-pier-onboarding-plan.md`
§4b; `filtered_egress` flips to `True` in that slice.

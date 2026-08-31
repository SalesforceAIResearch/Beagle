# Run-config authoring

A run-config is an optional YAML file that supplies defaults for
template rollouts. It is not needed for the direct managed-container
API, the Docker SDK drop-in, or the Harbor adapter.

Use it only when calling `Client.rollout(template=...)` and you want
to keep deadlines, idle TTLs, backend choice, network mode, or default
`init` fields outside Python code.

## Example

```yaml
manifests:
  hello-shell:
    deadlines:
      hard_s: 60
      step_timeout_s: 10
      setup_timeout_s: 30
    idle_ttl_s: 120
    init_params:
      prompt: "hello"
    backend: docker
    network: open
```

Pass it to the SDK:

```python
client = Client.grpc(
    host="127.0.0.1",
    port=50051,
    token="<token>",
    run_config="run-config.yaml",
)
```

Per-call `init` values override `init_params` from the run-config.
Per-call `deadline=...` overrides the run-config deadline block.

## Fields

| Field | Description |
|---|---|
| `manifests.<name>.deadlines.hard_s` | Required when any deadline or idle TTL value is set. It is the outer runtime budget. |
| `step_timeout_s` | Default timeout for one environment step. |
| `setup_timeout_s` | Default setup timeout. |
| `teardown_timeout_s` | Default teardown timeout. |
| `init_timeout_s` | Default timeout for initialization calls. |
| `idle_ttl_s` | How long a rollout can sit without `step()` or `heartbeat()` before XRLEnv truncates it. |
| `init_params` | Default setup parameters merged into per-call `init`. |
| `backend` | Sandbox backend. Use `docker` for the current shipped backend. |
| `network` | Network mode such as `open` or `none`. |

## Environment variables

| Variable | Description |
|---|---|
| `XRLENV_TEMPLATE_DIRS` | `os.pathsep`-separated list of filesystem directories (colon on POSIX, semicolon on Windows). The template catalog walks each directory for `manifest.yaml` files at startup. Supplements entry-point discovery; useful for development checkouts or operator-installed templates that are not pip-packaged. Missing or empty entries are skipped silently. |

Example:

```bash
export XRLENV_TEMPLATE_DIRS="/opt/my-benchmarks:/home/user/work/custom-templates"
```

## When not to use it

Do not use run-configs for Docker SDK drop-in or
`Client.acquire_container(...)` workflows. Those paths pass container
image, command, labels, env vars, and task key directly with each
request.

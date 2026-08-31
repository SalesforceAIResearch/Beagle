# Distributed tracing

XRLEnv instruments its control-plane and node hot paths with
[OpenTelemetry](https://opentelemetry.io/) traces. When enabled, every
`dispatch_rollout`, `env_step`, `create_sandbox`, and related operation
emits a span that you can explore in Jaeger, Tempo, Honeycomb, or any
OTLP-compatible backend.

Tracing is **off by default**. No env var set means a noop tracer: the
instrumentation points are present in the code, but they resolve to a
single dict lookup and a `with` block — no SDK objects are allocated on
the hot path and no memory allocations occur in the steady state.

## Installation

The OTel SDK packages are not in the default install. Pull them in with
the `observability` extra:

```bash
pip install -e '.[observability]'
```

This adds `opentelemetry-api`, `opentelemetry-sdk`, and
`opentelemetry-exporter-otlp-proto-grpc` to your environment.

If those packages are absent, `xrlenv` still imports and runs without
error. The tracing module detects the missing dependency at first use
and silently falls back to the noop tracer.

## Modes

Select a mode by setting env vars before starting `xrlenv up`. The two
vars are independent; setting both wires both processors simultaneously.

| Mode | How to activate | When to use |
|------|-----------------|-------------|
| Off (noop) | No env var set | Default; zero overhead |
| Console | `OTEL_TRACES_EXPORTER=console` | Local debugging only |
| OTLP | `OTEL_EXPORTER_OTLP_ENDPOINT=http://host:4317` | Production clusters |

### Off (default)

No configuration needed. The `get_tracer()` call returns a noop tracer
on first use and caches it; subsequent calls are a single attribute
lookup.

### Console mode

```bash
export OTEL_TRACES_EXPORTER=console
xrlenv up
```

Spans are pretty-printed to stderr as they finish. The output is human-readable
JSON, useful for confirming that instrumentation is firing and that
attribute values look correct.

```text
{
    "name": "xrlenv.coordinator.dispatch_rollout",
    "context": {
        "trace_id": "0x3e4a...",
        "span_id": "0x1b2c..."
    },
    "attributes": {
        "template": "terminal-bench-2",
        "task_key": "tb2/task-007",
        "group_id": "run-42",
        "deadline_s": 120.0
    },
    "status": "OK"
}
```

:::{note}
Console mode uses OTel's `ConsoleSpanExporter`, which blocks the exporter
thread to format and write each span. Do not run console mode in production;
the formatter serializes every span synchronously and the raw span text can
expose request shapes to anyone reading stderr.
:::

### OTLP mode

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger-host:4317
xrlenv up
```

Spans are forwarded to the endpoint over gRPC using a `BatchSpanProcessor`.
Batching is asynchronous: the hot path returns immediately and spans are
flushed in a background thread. The control plane never blocks on the
exporter.

OTLP mode works with any backend that accepts the OTLP gRPC protocol:
[Jaeger](https://www.jaegertracing.io/),
[Grafana Tempo](https://grafana.com/oss/tempo/),
[Honeycomb](https://www.honeycomb.io/), and others.

## Local Jaeger walkthrough

This is the fastest path to seeing traces in a browser on a development
machine.

**Start Jaeger:**

```bash
docker run --rm -d \
    --name jaeger \
    -p 4317:4317 \
    -p 16686:16686 \
    jaegertracing/all-in-one:latest
```

Port 4317 is the OTLP gRPC ingestion port. Port 16686 is the Jaeger UI.

**Start XRLEnv with OTLP enabled:**

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
xrlenv up
```

**Run a workload** (for example, a single managed container):

```python
import xrlenv

client = xrlenv.Client.in_process()
with client.acquire_container("my-image:latest") as ctr:
    result = ctr.exec(["echo", "hello"])
```

**Open the Jaeger UI** at `http://localhost:16686`. Select the
`xrlenv` service from the Service drop-down and click **Find Traces**.
You should see one trace per `acquire_container` call, with child spans
covering sandbox creation, image resolution, and node transport.

## Span catalog

Eight spans are emitted across the control-plane and node hot paths.

| Span name | Where it fires | Attributes |
|-----------|---------------|------------|
| `xrlenv.coordinator.dispatch_rollout` | `RolloutCoordinator.start_rollout` | `template`, `task_key`, `group_id`, `deadline_s` |
| `xrlenv.coordinator.build_apply` | `BuildCoordinator.apply` | `dry_run`, `force`, `eager`, `skip_if_present`, `applied_by` |
| `xrlenv.scheduler.place` | `Scheduler.place` | `template`, `image`, `backend`, `node_count` |
| `xrlenv.node.create_sandbox` | `NodeAgent.create_sandbox` | `rollout_id`, `backend`, `image`, `node_id` |
| `xrlenv.node.env_step` | `NodeAgent.env_step` | `sandbox_id`, `node_id` |
| `xrlenv.node.ensure_present` | `ImageCacheManager.ensure_present` | `image`, `deadline_s`, `cache_hit` |
| `xrlenv.node.source_build` | `SourceBuilder.build` | `image_ref`, `source_type`, `skip_if_present`, `timeout_s` |
| `xrlenv.transport.rpc` | `RemoteNodeTransport._send_and_wait` | `command_kind`, `node_id`, `timeout_s` |

The `cache_hit` attribute on `xrlenv.node.ensure_present` is `true` when
the image was already present on the node and `false` when it had to be
pulled or built. This is useful for spotting image-miss latency spikes.

## Adding spans to your own code

If you write an {doc}`EnvAdapter </supported_benchmarks_and_harnesses/writing_your_own_adapter>`
or a custom consumer and want to add spans that join the same trace, use
`get_tracer` directly:

```python
from xrlenv.observability.tracing import get_tracer

def my_step(obs):
    with get_tracer().start_as_current_span(
        "my_adapter.step",
        attributes={"task": obs.task_id},
    ) as span:
        result = upstream_evaluate(obs)
        span.set_attribute("score", result.score)
        return result
```

Spans created this way are automatically parented to the active
XRLEnv span in the same thread. They appear as children in the Jaeger
waterfall view.

The rule for deciding whether a new span is worth adding: if the
operation can block for more than a few milliseconds on the hot path
and its latency would otherwise be invisible in a trace, add a span.
Fine-grained spans inside tight loops are usually not worth the noise.

## Performance guarantee

When no env var is set (the default), `get_tracer()` returns a noop
tracer. The noop tracer's `start_as_current_span` is a Python context
manager that allocates no OTel SDK objects and does no I/O. The cost
is one attribute lookup on module import and one `__enter__`/`__exit__`
call per instrumented operation — negligible on any path that does real
work inside the span.

In OTLP mode, the `BatchSpanProcessor` hands spans to a background
thread. The hot path does not wait for the exporter.

## See also

- {doc}`metrics` — Prometheus `/metrics` for lifecycle counters and latency histograms.
- {doc}`logs` — structured JSON log events carrying `rollout_id` and `sandbox_id`.
- {doc}`admin_panel` — browser view of cluster state.

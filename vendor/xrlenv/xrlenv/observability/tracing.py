"""OpenTelemetry tracing surface (B7.1, P1.x slice 4).

The module is **opt-in via env var**: imports stay zero-cost on the
hot path when tracing is not configured. Three modes selected on
first ``get_tracer()`` call:

- **Off (default)** — no env var set. ``get_tracer()`` returns a
  module-level :class:`_NoopTracer` whose ``start_as_current_span``
  is a context manager that does nothing. The hot path runs at
  roughly the cost of one dict lookup + one ``with`` block; we
  measured this as < 1 µs per call on a 2024 laptop, which is
  comfortably below the noise floor of the cheapest real operation
  (a dict get on the state store). The point: leaving tracing off
  is the default and costs nothing.

- **Console (developer mode)** — ``OTEL_TRACES_EXPORTER=console``.
  Spans are pretty-printed to stderr via OTel's
  :class:`ConsoleSpanExporter`. Useful for local debugging — never
  enable in production, both because the formatter blocks the
  emitter thread and because raw span text leaks request shape.

- **OTLP (production)** — ``OTEL_EXPORTER_OTLP_ENDPOINT=http://...``
  ships spans to that OTLP gRPC endpoint (Jaeger, Tempo, Honeycomb,
  etc.) via :class:`BatchSpanProcessor` so the export is async
  and never blocks a hot path. The two env vars are not mutually
  exclusive: setting both wires both processors.

Public surface
--------------

::

    from xrlenv.observability.tracing import get_tracer

    with get_tracer().start_as_current_span(
        "xrlenv.coordinator.dispatch_rollout",
        attributes={"rollout_id": rid, "template": tpl},
    ):
        ...

The returned span object honors ``set_attribute(key, value)`` so a
call site that learns the value mid-span (e.g. ``cache_hit``) can
attach it without restructuring the surrounding code.

The module is **safe to import even when ``opentelemetry`` is
not installed.** When the SDK is missing, ``get_tracer()`` returns
the noop tracer unconditionally; the only effect is that you can't
turn tracing on. Operators wanting tracing must
``pip install -e '.[observability]'`` (or pull the three
``opentelemetry-*`` packages into their environment).

Concurrency: the lazy init is guarded by a module-level
:class:`threading.Lock` so the first concurrent ``get_tracer()``
calls under a multithreaded entrypoint don't race two tracer
providers into existence. After the first call the result is
cached and returned without re-entering the lock.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

LOGGER = logging.getLogger(__name__)

_OTLP_ENDPOINT_VAR = "OTEL_EXPORTER_OTLP_ENDPOINT"
_EXPORTER_KIND_VAR = "OTEL_TRACES_EXPORTER"
_SERVICE_NAME = "xrlenv"

_lock = threading.Lock()
_tracer: Any = None  # set on first get_tracer() call.


def get_tracer() -> Any:
    """Return the process-wide tracer, lazily configured on first call.

    The return type is intentionally ``Any``: when ``opentelemetry``
    is installed it's a real ``opentelemetry.trace.Tracer``; when
    not, it's a :class:`_NoopTracer`. Both expose the same
    ``start_as_current_span`` interface that call sites use.
    """
    global _tracer
    if _tracer is not None:
        return _tracer
    with _lock:
        if _tracer is not None:
            return _tracer
        _tracer = _build_tracer()
    return _tracer


def reset_for_tests() -> None:
    """Reset the lazy tracer cache so tests can re-configure under
    different env-var combinations. Don't call this from production
    code — the tracer is meant to be initialized exactly once per
    process."""
    global _tracer
    with _lock:
        _tracer = None


def _build_tracer() -> Any:
    """Inspect env + try to import OTel; return either a real tracer
    or :class:`_NoopTracer`.

    Failure modes that fall back to noop (without raising):
    - ``opentelemetry-api`` not installed.
    - No env var configured.
    - OTel SDK present but OTLP exporter missing while
      ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set (logged at WARNING so
      operators know why their endpoint didn't take effect).
    """
    try:
        from opentelemetry import trace
    except ImportError:
        LOGGER.debug(
            "opentelemetry-api not installed; tracing disabled. "
            "`pip install -e '.[observability]'` to enable.",
        )
        return _NoopTracer()

    otlp_endpoint = os.environ.get(_OTLP_ENDPOINT_VAR, "").strip()
    exporter_kind = os.environ.get(_EXPORTER_KIND_VAR, "").strip().lower()
    if not otlp_endpoint and exporter_kind != "console":
        LOGGER.debug(
            "no OTel env var set; tracing disabled. Set %s or %s=console.",
            _OTLP_ENDPOINT_VAR, _EXPORTER_KIND_VAR,
        )
        return _NoopTracer()

    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        LOGGER.warning(
            "opentelemetry-sdk not installed; tracing disabled even though "
            "%s/%s is set. `pip install -e '.[observability]'` to enable.",
            _OTLP_ENDPOINT_VAR, _EXPORTER_KIND_VAR,
        )
        return _NoopTracer()

    resource = Resource.create({"service.name": _SERVICE_NAME})
    provider = TracerProvider(resource=resource)

    if exporter_kind == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter()),
        )
        LOGGER.info("tracing: ConsoleSpanExporter wired (dev mode)")

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError:
            LOGGER.warning(
                "opentelemetry-exporter-otlp-proto-grpc not installed; "
                "OTLP export disabled. Console export (if configured) "
                "still active.",
            )
        else:
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)),
            )
            LOGGER.info(
                "tracing: OTLPSpanExporter wired (endpoint=%s)",
                otlp_endpoint,
            )

    trace.set_tracer_provider(provider)
    return trace.get_tracer(_SERVICE_NAME)


# ──────────────────────────────────────────────────────────────────────────────
# Noop fallback
# ──────────────────────────────────────────────────────────────────────────────


class _NoopSpan:
    """Tracer / span stand-in when OTel is unavailable or unconfigured.

    Mirrors the subset of the real ``Span`` interface that call sites
    use (``set_attribute``, the context-manager protocol). Method
    bodies are deliberately empty — the noop must not allocate, log,
    or do anything observable that could regress the hot-path cost
    we promise above.
    """

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *_a: Any) -> None:
        return None

    def set_attribute(self, _key: str, _value: Any) -> None:
        return None

    def set_attributes(self, _attrs: dict[str, Any]) -> None:
        return None

    def record_exception(self, _exc: BaseException) -> None:
        return None

    def set_status(self, _status: Any) -> None:
        return None


class _NoopTracer:
    """Tracer fallback. Returns a fresh :class:`_NoopSpan` for every
    ``start_as_current_span`` call. The span instance is cheap to
    allocate (no slots, no init body) — preferable to a singleton
    here because the OTel SDK isn't strictly singleton-safe across
    nested ``with`` blocks and we don't want to invent semantics that
    diverge from the real tracer."""

    def start_as_current_span(
        self,
        _name: str,
        *,
        attributes: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> _NoopSpan:
        _ = attributes  # silence unused — accepted for API parity.
        return _NoopSpan()


__all__ = ["get_tracer", "reset_for_tests"]

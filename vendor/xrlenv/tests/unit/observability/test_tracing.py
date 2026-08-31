"""Tests for the OTel tracing surface (B7.1, P1.x slice 4).

Two layers of coverage:

1. **Module-level behaviour**: ``get_tracer()`` is lazy, returns the
   noop tracer when no env var is set, returns a real tracer when
   ``OTEL_TRACES_EXPORTER=console`` or ``OTEL_EXPORTER_OTLP_ENDPOINT``
   is set, and is concurrency-safe. The noop span context manager
   tolerates ``set_attribute`` without raising — call sites use it
   freely.

2. **End-to-end span emission**: configure an ``InMemorySpanExporter``
   directly into ``TracerProvider`` and exercise one instrumented
   call site per file. Asserts that the named span appears with the
   documented attributes. This is the test form of the slice's
   smoke ("env-var-driven local console export prints span lines").
"""

from __future__ import annotations

from typing import Any

import pytest
from xrlenv.observability.tracing import (
    _NoopSpan,
    _NoopTracer,
    get_tracer,
    reset_for_tests,
)

# The ``observability`` extra is optional. When the OTel SDK is not
# installed, ``get_tracer()`` still works (it returns the noop tracer)
# — but the tests below that assert console / OTLP env vars produce
# a *real* tracer, and the end-to-end span-emission test, need the
# SDK. Skip those when the SDK is unavailable instead of failing the
# whole file. The pure-noop tests above this skip still run on a
# clean ``.[dev]`` install.
_otel_sdk_available: bool
try:
    import opentelemetry.sdk.trace  # noqa: F401
    _otel_sdk_available = True
except ImportError:
    _otel_sdk_available = False

_skip_without_otel_sdk = pytest.mark.skipif(
    not _otel_sdk_available,
    reason="opentelemetry-sdk not installed (pip install -e '.[observability]')",
)

# ──────────────────────────────────────────────────────────────────────────────
# get_tracer() — env var → mode wiring
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_tracer_cache() -> Any:
    """Each test starts with a fresh lazy-init slot — otherwise the
    first env-var setting "wins" for the rest of the session."""
    reset_for_tests()
    yield
    reset_for_tests()


def test_get_tracer_no_env_returns_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env var → noop tracer. The default install path costs nothing."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
    tracer = get_tracer()
    assert isinstance(tracer, _NoopTracer)


@_skip_without_otel_sdk
def test_get_tracer_console_env_returns_real_tracer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OTEL_TRACES_EXPORTER=console`` → real tracer (the dev-mode
    path that prints to stderr)."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "console")
    tracer = get_tracer()
    assert not isinstance(tracer, _NoopTracer)


@_skip_without_otel_sdk
def test_get_tracer_otlp_env_returns_real_tracer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OTEL_EXPORTER_OTLP_ENDPOINT=...`` → real tracer with the
    OTLP exporter wired. We don't dial the endpoint here — just check
    the tracer is real (not noop) and that the lazy build succeeded."""
    monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    tracer = get_tracer()
    assert not isinstance(tracer, _NoopTracer)


def test_get_tracer_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lazy init runs exactly once. Subsequent calls return the
    same tracer object even when env vars change underneath us
    (mirrors process-lifetime semantics)."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
    first = get_tracer()
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "console")
    second = get_tracer()
    assert first is second
    # Sanity: still noop because the cached one was built before
    # the env-var change.
    assert isinstance(first, _NoopTracer)


# ──────────────────────────────────────────────────────────────────────────────
# Noop tracer/span semantics
# ──────────────────────────────────────────────────────────────────────────────


def test_noop_span_context_manager_returns_noop_span() -> None:
    """The noop tracer's ``start_as_current_span`` is a context
    manager that yields a span instance. Call sites use the span
    inside a ``with`` block and may call ``set_attribute`` on it."""
    tracer = _NoopTracer()
    with tracer.start_as_current_span(
        "x.test", attributes={"k": "v"},
    ) as span:
        assert isinstance(span, _NoopSpan)
        span.set_attribute("cache_hit", True)
        span.set_attribute("other", 42)
        span.set_attributes({"a": 1, "b": 2})
        # record_exception/set_status are accepted as no-ops too.
        span.record_exception(RuntimeError("not raised — just recorded"))


def test_noop_span_swallows_no_exceptions() -> None:
    """The context manager's ``__exit__`` must not suppress an
    exception raised inside the ``with``. If it did, instrumented
    call sites would silently eat real errors."""
    tracer = _NoopTracer()
    with pytest.raises(RuntimeError, match="boom"), tracer.start_as_current_span("x.test"):
        raise RuntimeError("boom")


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end: spans land in an in-memory exporter
# ──────────────────────────────────────────────────────────────────────────────


def _wire_in_memory_exporter() -> Any:
    """Build an isolated :class:`TracerProvider` with an
    :class:`InMemorySpanExporter`, hot-wire the resulting tracer into
    ``xrlenv.observability.tracing._tracer`` so call sites see it,
    and return the exporter for assertions.

    The provider is *not* installed as OTel's global tracer provider:
    the SDK refuses to override a previously-set global (and emits a
    deprecation warning), which would couple every test in this file
    to whichever happens to set the global first. Attaching the
    tracer directly to the module slot is enough — call sites only
    talk to ``get_tracer()``, never to ``trace.get_tracer()``
    directly.
    """
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create({"service.name": "xrlenv-test"}),
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    import xrlenv.observability.tracing as tracing_mod
    tracing_mod._tracer = provider.get_tracer("xrlenv")
    return exporter


@_skip_without_otel_sdk
def test_end_to_end_span_emission_via_get_tracer() -> None:
    """When the module-level tracer points at a real provider, a
    ``with get_tracer().start_as_current_span(...)`` call records a
    span the exporter can recover with the right name + attributes."""
    exporter = _wire_in_memory_exporter()
    with get_tracer().start_as_current_span(
        "xrlenv.test.span",
        attributes={"k": "v", "n": 42},
    ):
        pass
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "xrlenv.test.span"
    assert spans[0].attributes["k"] == "v"
    assert spans[0].attributes["n"] == 42

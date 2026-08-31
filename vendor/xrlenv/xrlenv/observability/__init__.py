"""Observability — metrics + structured logs (spec 08, Slice 5a).

Phase-0 surface:

- :class:`MetricsRegistry` (xrlenv/observability/metrics.py) — typed wrapper
  around the prometheus-client primitives the coordinator and admission
  queue increment.
- :func:`configure_logging` (xrlenv/observability/logging.py) — installs
  the root-logger formatter; ``log_format="auto"`` picks colorized
  human-friendly output on a TTY and the JSON envelope when piped.
  :func:`configure_json_logging` is the legacy alias that locks the
  format to JSON regardless of TTY status.
- :func:`start_metrics_server` (xrlenv/observability/server.py) — starts
  the Prometheus text-exposition HTTP server on the requested port.

The control plane and the node agent both opt in at startup; tests can
build a private registry to avoid global state leakage between tests.

**Lazy re-exports (PEP 562).** The package-level names below are
resolved on first access via :func:`__getattr__`. Importing
``from xrlenv.observability.logging import configure_logging`` from a
lightweight context (e.g. the in-sandbox stub, which only needs
logging) does *not* drag in ``prometheus_client`` via the metrics
submodule. Callers who do ``from xrlenv.observability import
MetricsRegistry`` still get the eager-feeling behavior — the lookup
runs the metrics-import at first attribute access and caches the
result on the module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # IDE / mypy: surface the eager-style names as if they were imported
    # at module load time. ``__getattr__`` below handles the runtime
    # binding lazily.
    from xrlenv.observability.logging import configure_json_logging, configure_logging
    from xrlenv.observability.metrics import MetricsRegistry, get_default_registry
    from xrlenv.observability.server import MetricsServer, start_metrics_server

__all__ = [
    "MetricsRegistry",
    "MetricsServer",
    "configure_json_logging",
    "configure_logging",
    "get_default_registry",
    "start_metrics_server",
]


# Map each re-exported name to the submodule it lives in. Keeps the
# resolver flat (no nested ifs) and lets us import only the specific
# submodule a caller actually touched.
_SUBMODULE_BY_NAME = {
    "configure_json_logging": "xrlenv.observability.logging",
    "configure_logging":      "xrlenv.observability.logging",
    "MetricsRegistry":        "xrlenv.observability.metrics",
    "get_default_registry":   "xrlenv.observability.metrics",
    "MetricsServer":          "xrlenv.observability.server",
    "start_metrics_server":   "xrlenv.observability.server",
}


def __getattr__(name: str) -> Any:
    submodule = _SUBMODULE_BY_NAME.get(name)
    if submodule is None:
        raise AttributeError(
            f"module 'xrlenv.observability' has no attribute {name!r}"
        )
    import importlib

    mod = importlib.import_module(submodule)
    value = getattr(mod, name)
    # Cache on this module so subsequent attribute access skips the
    # __getattr__ hook (Python only calls __getattr__ for missing
    # attributes, so the second hit is a normal __dict__ lookup).
    globals()[name] = value
    return value

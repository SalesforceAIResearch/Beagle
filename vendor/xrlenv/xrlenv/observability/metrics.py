"""Prometheus metrics (spec 08 §"Metrics").

The phase-0 series the coordinator and admission queue feed are
declared once here so the names + label sets stay consistent. Spec 08
lists the full Prometheus contract including phase-1+ series like
``xrlenv_warm_pool_size`` and the per-node hardware gauges; we only
instantiate what phase 0 actually emits, so a missing series in
``/metrics`` after a deploy is a real signal rather than a permanent
zero.

A :class:`MetricsRegistry` wraps a :class:`prometheus_client.CollectorRegistry`
so that tests can build private registries (avoiding cross-test
pollution from the prometheus-client process-global default), while
the production code path uses :func:`get_default_registry` for a
single shared instance.

Multi-tenant labels (``owner_id`` / ``project_id``) are out of scope
for Slice 5a — phase 0 hard-codes ``owner_id = project_id = "default"``
so wiring them in here would just increase cardinality with no
information gain. They land alongside spec 03's tenancy hooks in a
later slice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

if TYPE_CHECKING:
    from xrlenv.types import RolloutStatus


# Phase-0 buckets. Spec 08 doesn't pin specific buckets per series, so we
# use compact, evenly log-spaced sets that cover the workloads we
# actually run (sub-millisecond container ops up to multi-minute
# deadlines for slow templates like SWE-bench).
_LATENCY_BUCKETS_FAST = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
_LATENCY_BUCKETS_SANDBOX = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
_LATENCY_BUCKETS_QUEUE = (0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 900.0)


class MetricsRegistry:
    """Owns the phase-0 Prometheus series.

    All series live on the wrapped :class:`CollectorRegistry`; the
    HTTP server (:func:`start_metrics_server`) reads from the same
    registry. Build a fresh ``MetricsRegistry()`` in tests to avoid
    polluting the process-global default.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self._registry = registry or CollectorRegistry()

        # Rollout lifecycle counters (spec 08 §"Core series")
        self.rollouts_started_total = Counter(
            "xrlenv_rollouts_started_total",
            "Rollouts admitted to RUNNING (post-bootstrap).",
            labelnames=("template",),
            registry=self._registry,
        )
        self.rollouts_finished_total = Counter(
            "xrlenv_rollouts_finished_total",
            "Rollouts that reached a terminal state, by status.",
            labelnames=("template", "status"),
            registry=self._registry,
        )

        # Latency histograms
        self.step_latency_seconds = Histogram(
            "xrlenv_step_latency_seconds",
            "Wall-clock seconds for one Coordinator.step (env_step + bookkeeping).",
            labelnames=("template", "backend"),
            buckets=_LATENCY_BUCKETS_FAST,
            registry=self._registry,
        )
        self.sandbox_create_seconds = Histogram(
            "xrlenv_sandbox_create_seconds",
            "Wall-clock seconds from create_sandbox to first observation ready.",
            labelnames=("template", "backend"),
            buckets=_LATENCY_BUCKETS_SANDBOX,
            registry=self._registry,
        )
        self.sandbox_destroy_seconds = Histogram(
            "xrlenv_sandbox_destroy_seconds",
            "Wall-clock seconds spent in destroy_sandbox.",
            labelnames=("template", "backend"),
            buckets=_LATENCY_BUCKETS_SANDBOX,
            registry=self._registry,
        )
        self.queue_wait_seconds = Histogram(
            "xrlenv_queue_wait_seconds",
            "Wall-clock seconds a rollout spent in the admission queue.",
            labelnames=("template",),
            buckets=_LATENCY_BUCKETS_QUEUE,
            registry=self._registry,
        )

        # Liveness gauges
        self.sandbox_active = Gauge(
            "xrlenv_sandbox_active",
            "Currently-running sandboxes per (node, template).",
            labelnames=("node", "template"),
            registry=self._registry,
        )
        self.queue_depth = Gauge(
            "xrlenv_queue_depth",
            "Pending rollouts waiting in admission per template.",
            labelnames=("template",),
            registry=self._registry,
        )
        # Stage-3 (P1) — the health-derived adaptive admission limit
        # per node. Graphing it shows AIMD's sawtooth: contractions
        # when a node's docker daemon saturates, recovery when it holds.
        self.node_admission_limit = Gauge(
            "xrlenv_node_admission_limit",
            "Health-derived adaptive concurrent-acquire limit per node.",
            labelnames=("node",),
            registry=self._registry,
        )

        # Failure / rejection counters
        self.sandbox_create_failed_total = Counter(
            "xrlenv_sandbox_create_failed_total",
            "Bootstrap-phase failures by classified reason.",
            labelnames=("template", "reason"),
            registry=self._registry,
        )
        self.admission_total = Counter(
            "xrlenv_admission_total",
            "Admission outcomes (admitted / queued / queue_timeout / cancelled_in_queue).",
            labelnames=("result",),
            registry=self._registry,
        )

    @property
    def collector_registry(self) -> CollectorRegistry:
        """Underlying prometheus-client registry — for the HTTP exposer."""
        return self._registry

    # ── Convenience helpers ─────────────────────────────────────────────────
    #
    # The coordinator increments these often; centralizing the .labels(...)
    # call here keeps the call sites short and prevents typos in label
    # names from going undetected (Prometheus silently swallows extra
    # labels).

    def observe_rollout_started(self, template: str) -> None:
        self.rollouts_started_total.labels(template=template).inc()

    def observe_rollout_finished(
        self, template: str, status: RolloutStatus | str
    ) -> None:
        status_value = status.value if hasattr(status, "value") else str(status)
        self.rollouts_finished_total.labels(
            template=template, status=status_value
        ).inc()

    def observe_step_latency(
        self, template: str, backend: str, seconds: float
    ) -> None:
        self.step_latency_seconds.labels(
            template=template, backend=backend
        ).observe(seconds)

    def observe_sandbox_create(
        self, template: str, backend: str, seconds: float
    ) -> None:
        self.sandbox_create_seconds.labels(
            template=template, backend=backend
        ).observe(seconds)

    def observe_sandbox_destroy(
        self, template: str, backend: str, seconds: float
    ) -> None:
        self.sandbox_destroy_seconds.labels(
            template=template, backend=backend
        ).observe(seconds)

    def observe_queue_wait(self, template: str, seconds: float) -> None:
        self.queue_wait_seconds.labels(template=template).observe(seconds)

    def observe_sandbox_create_failed(self, template: str, reason: str) -> None:
        self.sandbox_create_failed_total.labels(
            template=template, reason=reason
        ).inc()

    def observe_admission(self, result: str) -> None:
        self.admission_total.labels(result=result).inc()

    def set_sandbox_active(self, node: str, template: str, value: float) -> None:
        self.sandbox_active.labels(node=node, template=template).set(value)

    def inc_sandbox_active(self, node: str, template: str) -> None:
        self.sandbox_active.labels(node=node, template=template).inc()

    def dec_sandbox_active(self, node: str, template: str) -> None:
        self.sandbox_active.labels(node=node, template=template).dec()

    def set_queue_depth(self, template: str, value: float) -> None:
        self.queue_depth.labels(template=template).set(value)

    def set_node_admission_limit(self, node: str, value: float) -> None:
        self.node_admission_limit.labels(node=node).set(value)


_default_registry: MetricsRegistry | None = None


def get_default_registry() -> MetricsRegistry:
    """Process-wide :class:`MetricsRegistry` for production code paths.

    Tests should construct their own :class:`MetricsRegistry` instead of
    using this — sharing the global registry across tests leaks counter
    state between cases.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = MetricsRegistry()
    return _default_registry


def reset_default_registry() -> None:
    """Drop the cached default registry. Test-only escape hatch."""
    global _default_registry
    _default_registry = None


__all__ = [
    "MetricsRegistry",
    "get_default_registry",
    "reset_default_registry",
]

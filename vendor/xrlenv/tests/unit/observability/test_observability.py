"""Tests for the Slice 5a observability surface.

Covers:
- :class:`MetricsRegistry` increments + label correctness
- :class:`JsonFormatter` envelope shape + extras + exc_info handling
- :class:`MetricsServer` lifecycle + ``/metrics`` HTTP exposition
- :py:meth:`PlatformJsonlSink.record_event` writing the per-rollout
  ``coordinator.log`` (open + sealed code paths)
- End-to-end coordinator + admission integration: rollout lifecycle metrics,
  reward-failure metrics, queue admission metrics.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from prometheus_client import generate_latest
from xrlenv.backends.base import (
    ExecResult,
    ResourceSpec,
    ResourceUsage,
    SandboxHandle,
)
from xrlenv.client.client import Client
from xrlenv.control.admission import AdmissionQueue
from xrlenv.control.coordinator import RolloutCoordinator
from xrlenv.control.scheduler import Placement
from xrlenv.control.service import CoordinatorRolloutService
from xrlenv.control.state import InMemoryStateStore
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateCatalog,
    TemplateManifest,
)
from xrlenv.control.trajectory_sink import PlatformJsonlSink
from xrlenv.errors import CapacityExhausted, RolloutFailed, RolloutTruncated
from xrlenv.node.hw_probe import HardwareInfo
from xrlenv.observability.dashboard import (
    _estimate_quantile,
    prefers_html,
    render_dashboard_html,
)
from xrlenv.observability.logging import (
    JsonFormatter,
    PrettyFormatter,
    configure_json_logging,
    configure_logging,
)
from xrlenv.observability.metrics import MetricsRegistry
from xrlenv.observability.server import MetricsServer
from xrlenv.types import RolloutStatus

# ──────────────────────────────────────────────────────────────────────────────
# Test helpers (small standalone scaffolding so this file doesn't depend on
# fixtures from sibling tests)
# ──────────────────────────────────────────────────────────────────────────────


def _hw() -> HardwareInfo:
    return HardwareInfo(
        vcpus=4, mem_bytes=16 * 1024**3, disk_bytes=200 * 1024**3,
        has_kvm=False, has_gpu=False, gpu_model=None,
        kernel_version="0.0.0", platform="linux",
    )


def _manifest(name: str = "obs-t") -> TemplateManifest:
    return TemplateManifest(
        name=name, version="0.1", digest=f"sha256:{name}",
        image=f"im/{name}:1",
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )


class _FakeNode:
    """Minimal NodeTransport stand-in used by the integration tests."""

    def __init__(self, *, max_steps: int = 1, fail_create: bool = False) -> None:
        self.node_id = "fake-obs"
        self._created = 0
        self._max_steps = max_steps
        self._steps: dict[str, int] = {}
        self._fail_create = fail_create

    def supported_backends(self) -> list[str]:
        return ["docker"]

    def hardware(self) -> HardwareInfo:
        return _hw()

    async def create_sandbox(self, **_: Any) -> SandboxHandle:
        if self._fail_create:
            raise RuntimeError("create boom — image_pull_failed simulated")
        self._created += 1
        sid = f"sb-{self._created}"
        return SandboxHandle(
            id=sid, backend="docker", backend_ref=f"cid-{self._created}",
            stub_endpoint="tcp://127.0.0.1:0",
        )

    async def destroy_sandbox(self, _sb: SandboxHandle) -> None:
        return None

    async def env_setup(
        self, sb: SandboxHandle, *, adapter_module: str, adapter_class: str,
        init_params: dict[str, Any], **_kw: Any,
    ) -> dict[str, Any]:
        self._steps[sb.id] = 0
        return {"obs": {"first": True}}

    async def env_step(
        self, sb: SandboxHandle, action: Any, **_kw: Any,
    ) -> dict[str, Any]:
        self._steps[sb.id] += 1
        done = self._steps[sb.id] >= self._max_steps
        return {
            "obs": {"step": self._steps[sb.id]},
            "reward": 1.0 if done else 0.0,
            "done": done,
            "truncated": False,
            "info": {},
        }

    async def env_teardown(self, _sb: SandboxHandle, **_kw: Any) -> dict[str, Any]:
        return {"status": "ok"}

    async def run_in_sandbox(
        self, _sb: SandboxHandle, _cmd: list[str], **_: Any
    ) -> ExecResult:
        return ExecResult(exit_code=0, stdout=b"")

    async def stats(self, _sb: SandboxHandle) -> ResourceUsage:
        return ResourceUsage(cpu_seconds=0.0, rss_bytes=0, disk_bytes=0,
                             rx_bytes=0, tx_bytes=0)

    async def query_image(self, _image: str) -> Any:
        from xrlenv.node.image_cache import ImageQueryResult
        return ImageQueryResult(present=True)


def _build_runtime(
    *,
    sink: PlatformJsonlSink | None = None,
    metrics: MetricsRegistry,
    fail_create: bool = False,
    max_steps: int = 1,
) -> tuple[Client, RolloutCoordinator, InMemoryStateStore, _FakeNode]:
    node = _FakeNode(max_steps=max_steps, fail_create=fail_create)
    catalog = TemplateCatalog()
    catalog.register(_manifest())
    sched = MagicMock()
    sched.place.return_value = Placement(node=node, backend="docker", score=1)
    sched.nodes = [node]
    state = InMemoryStateStore()
    admission = AdmissionQueue(scheduler=sched, state=state, metrics=metrics)
    coord = RolloutCoordinator(
        catalog=catalog, scheduler=sched, state=state,
        admission=admission, trajectory_sink=sink, metrics=metrics,
    )
    service = CoordinatorRolloutService(coord)
    client = Client.in_process(service)
    return client, coord, state, node


async def _drain(coord: RolloutCoordinator, client: Client) -> None:
    await coord.deadline_watcher.shutdown()
    await coord.idle_ttl_watcher.shutdown()
    await client.close()


def _metric_value(reg: MetricsRegistry, name: str, **labels: str) -> float:
    """Read a single (name, labels) sample's value from the registry.

    Counter sample names end in ``_total`` while the family name strips that
    suffix; histogram sample names end in ``_sum`` / ``_count`` / ``_bucket``
    / ``_created`` while the family name strips those. We strip every known
    suffix when matching the family so callers can ask for the sample name
    they actually want to read.
    """
    suffixes = ("_total", "_sum", "_count", "_bucket", "_created")
    family_candidates = {name}
    for suffix in suffixes:
        if name.endswith(suffix):
            family_candidates.add(name.removesuffix(suffix))
    for family in reg.collector_registry.collect():
        if family.name not in family_candidates:
            continue
        for sample in family.samples:
            if sample.name != name:
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return float(sample.value)
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# MetricsRegistry unit tests
# ──────────────────────────────────────────────────────────────────────────────


def test_registry_counters_increment() -> None:
    reg = MetricsRegistry()
    reg.observe_rollout_started("hello")
    reg.observe_rollout_started("hello")
    reg.observe_rollout_finished("hello", "finished")
    reg.observe_rollout_finished("hello", RolloutStatus.FAILED)

    assert _metric_value(
        reg, "xrlenv_rollouts_started_total", template="hello"
    ) == 2.0
    assert _metric_value(
        reg, "xrlenv_rollouts_finished_total", template="hello", status="finished"
    ) == 1.0
    assert _metric_value(
        reg, "xrlenv_rollouts_finished_total", template="hello", status="failed"
    ) == 1.0


def test_registry_histograms_observe() -> None:
    reg = MetricsRegistry()
    reg.observe_step_latency("t", "docker", 0.123)
    reg.observe_sandbox_create("t", "docker", 0.5)
    reg.observe_queue_wait("t", 1.5)

    sum_step = _metric_value(
        reg, "xrlenv_step_latency_seconds_sum", template="t", backend="docker"
    )
    sum_sb = _metric_value(
        reg, "xrlenv_sandbox_create_seconds_sum", template="t", backend="docker"
    )
    sum_queue = _metric_value(reg, "xrlenv_queue_wait_seconds_sum", template="t")
    assert sum_step == pytest.approx(0.123)
    assert sum_sb == pytest.approx(0.5)
    assert sum_queue == pytest.approx(1.5)


def test_registry_active_gauge_inc_dec() -> None:
    reg = MetricsRegistry()
    reg.inc_sandbox_active("node-a", "tpl")
    reg.inc_sandbox_active("node-a", "tpl")
    reg.dec_sandbox_active("node-a", "tpl")
    assert _metric_value(
        reg, "xrlenv_sandbox_active", node="node-a", template="tpl"
    ) == 1.0


def test_registry_collector_registry_is_isolated() -> None:
    """Each MetricsRegistry owns its own CollectorRegistry; tests don't leak."""
    reg_a = MetricsRegistry()
    reg_b = MetricsRegistry()
    reg_a.observe_admission("admitted")
    assert _metric_value(reg_a, "xrlenv_admission_total", result="admitted") == 1.0
    # Fresh registry sees no data from the other one.
    assert _metric_value(reg_b, "xrlenv_admission_total", result="admitted") == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# JSON logging
# ──────────────────────────────────────────────────────────────────────────────


def test_json_formatter_envelope_shape() -> None:
    record = logging.LogRecord(
        name="xrlenv.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello %s", args=("world",), exc_info=None,
    )
    out = json.loads(JsonFormatter().format(record))
    assert out["level"] == "INFO"
    assert out["event"] == "xrlenv.test"
    assert out["message"] == "hello world"
    assert isinstance(out["ts"], float)


def test_json_formatter_carries_extras() -> None:
    record = logging.LogRecord(
        name="xrlenv.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="x", args=(), exc_info=None,
    )
    record.rollout_id = "rid-1"
    record.node_id = "node-z"
    out = json.loads(JsonFormatter().format(record))
    assert out["rollout_id"] == "rid-1"
    assert out["node_id"] == "node-z"


def test_json_formatter_handles_unserialisable_values() -> None:
    class Opaque:
        def __repr__(self) -> str:
            return "<Opaque>"

    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname=__file__, lineno=1,
        msg="m", args=(), exc_info=None,
    )
    record.opaque = Opaque()
    out = json.loads(JsonFormatter().format(record))
    assert out["opaque"] == "<Opaque>"


def test_json_formatter_includes_exc_info() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="x", level=logging.ERROR, pathname=__file__, lineno=1,
            msg="oops", args=(), exc_info=sys.exc_info(),
        )
    out = json.loads(JsonFormatter().format(record))
    assert "boom" in out["exc_info"]


def test_configure_json_logging_replaces_handlers() -> None:
    stream = io.StringIO()
    configure_json_logging(level=logging.DEBUG, stream=stream)
    logger = logging.getLogger("xrlenv.test_replace")
    logger.info("hello %s", "world")
    line = stream.getvalue().splitlines()[-1]
    payload = json.loads(line)
    assert payload["message"] == "hello world"
    assert payload["event"] == "xrlenv.test_replace"
    # Restore default config so the rest of the suite isn't affected.
    logging.getLogger().handlers.clear()


# ──────────────────────────────────────────────────────────────────────────────
# PrettyFormatter — terminal-friendly output for ``xrlenv up`` operators
# ──────────────────────────────────────────────────────────────────────────────


def test_pretty_formatter_renders_level_label_and_event() -> None:
    record = logging.LogRecord(
        name="xrlenv.control.coordinator", level=logging.INFO,
        pathname=__file__, lineno=1,
        msg="rollout %s started", args=("rid-7",), exc_info=None,
    )
    out = PrettyFormatter(colorize=False).format(record)
    assert "INFO " in out
    assert "xrlenv.control.coordinator" in out
    assert "rollout rid-7 started" in out
    # Plain mode emits no ANSI escape sequences at all.
    assert "\x1b[" not in out


def test_pretty_formatter_color_codes_level_token() -> None:
    record = logging.LogRecord(
        name="x", level=logging.WARNING,
        pathname=__file__, lineno=1,
        msg="careful", args=(), exc_info=None,
    )
    out = PrettyFormatter(colorize=True).format(record)
    # Yellow SGR around the WARN label, reset afterwards.
    assert "\x1b[33mWARN \x1b[0m" in out


def test_pretty_formatter_color_mapping_matches_spec() -> None:
    """Operator-facing color contract: red=ERROR, yellow=WARN, green=INFO, dim=DEBUG."""
    cases = [
        (logging.ERROR, "\x1b[31m"),
        (logging.WARNING, "\x1b[33m"),
        (logging.INFO, "\x1b[32m"),
        (logging.DEBUG, "\x1b[2m"),
    ]
    for level, code in cases:
        record = logging.LogRecord(
            name="x", level=level, pathname=__file__, lineno=1,
            msg="m", args=(), exc_info=None,
        )
        out = PrettyFormatter(colorize=True).format(record)
        assert code in out, f"level={level} expected {code!r} in {out!r}"


def test_pretty_formatter_honors_no_color_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    record = logging.LogRecord(
        name="x", level=logging.ERROR, pathname=__file__, lineno=1,
        msg="boom", args=(), exc_info=None,
    )
    # Even with colorize=True, NO_COLOR must suppress every ANSI code.
    out = PrettyFormatter(colorize=True).format(record)
    assert "\x1b[" not in out
    assert "ERROR" in out and "boom" in out


def test_pretty_formatter_renders_extras_at_end_of_line() -> None:
    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname=__file__, lineno=1,
        msg="step", args=(), exc_info=None,
    )
    record.rollout_id = "rid-1"
    record.node_id = "node-z"
    out = PrettyFormatter(colorize=False).format(record)
    assert "rollout_id=rid-1" in out
    assert "node_id=node-z" in out


def test_pretty_formatter_appends_traceback() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="x", level=logging.ERROR, pathname=__file__, lineno=1,
            msg="oops", args=(), exc_info=sys.exc_info(),
        )
    out = PrettyFormatter(colorize=False).format(record)
    assert "oops" in out
    assert "ValueError: boom" in out


def test_configure_logging_auto_uses_json_for_non_tty_stream() -> None:
    """``StringIO`` reports ``isatty=False`` — auto must pick the JSON formatter
    so piping ``xrlenv up | tee log.jsonl`` keeps producing parseable records."""
    stream = io.StringIO()
    try:
        configure_logging(level=logging.INFO, log_format="auto", stream=stream)
        logging.getLogger("xrlenv.test_auto_json").info("x")
        line = stream.getvalue().splitlines()[-1]
        # Round-trips through json — proves we did not pick the pretty formatter.
        payload = json.loads(line)
        assert payload["message"] == "x"
    finally:
        logging.getLogger().handlers.clear()


def test_configure_logging_auto_uses_pretty_for_tty_stream() -> None:
    class FakeTty(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = FakeTty()
    try:
        configure_logging(level=logging.INFO, log_format="auto", stream=stream)
        logging.getLogger("xrlenv.test_auto_pretty").info("hello")
        out = stream.getvalue()
        # Pretty layout starts with an HH:MM:SS clock, never the JSON ``{"ts":``.
        assert not out.lstrip().startswith("{")
        assert "hello" in out
        assert "xrlenv.test_auto_pretty" in out
    finally:
        logging.getLogger().handlers.clear()


def test_configure_logging_explicit_pretty_overrides_non_tty() -> None:
    stream = io.StringIO()
    try:
        configure_logging(level=logging.INFO, log_format="pretty", stream=stream)
        logging.getLogger("xrlenv.test_pretty_force").info("hi")
        out = stream.getvalue()
        assert not out.lstrip().startswith("{")
        # No color on a non-TTY stream even when pretty is forced.
        assert "\x1b[" not in out
        assert "INFO " in out and "hi" in out
    finally:
        logging.getLogger().handlers.clear()


def test_node_cli_main_emits_json_logs_by_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression for audit M1 against commit 1564205.

    The Slice 5a JSON logger was added but no entry point installed it; default
    operation still produced plain-text records. ``xrlenv-node serve`` must
    install :class:`JsonFormatter` on the root logger via
    :func:`configure_json_logging` before any other work runs, so the
    "connecting to …" startup record lands as a JSON envelope.
    """
    from xrlenv.node import cli as node_cli

    # Stub out asyncio.run so we don't actually start the gRPC reconnect loop
    # (it'd retry forever against the unreachable address). The CLI still
    # builds the agent + link and emits the startup log lines via _serve,
    # which is enough to prove logging is wired.
    captured_calls: list[str] = []

    async def _fake_serve(args: Any) -> None:
        node_cli.LOGGER.info(
            "xrlenv-node id=%s connecting to %s", args.node_id, args.control_plane
        )
        captured_calls.append(args.node_id)

    monkeypatch.setattr(node_cli, "_serve", _fake_serve)

    rc = node_cli.main(
        ["serve", "--control-plane", "127.0.0.1:1", "--node-id", "audit-test"]
    )
    assert rc == 0
    assert captured_calls == ["audit-test"]

    captured = capsys.readouterr().out.splitlines()
    parsed: list[dict[str, Any]] = []
    for line in captured:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    assert parsed, (
        "expected at least one JSON-formatted log line from xrlenv-node main; "
        f"saw stdout={captured!r}"
    )
    record = parsed[0]
    for required_field in ("ts", "level", "event", "message"):
        assert required_field in record, (
            f"JSON envelope missing field {required_field!r}: {record}"
        )
    assert record["event"] == "xrlenv.node"
    # Restore default config so the rest of the suite isn't affected.
    logging.getLogger().handlers.clear()


# ──────────────────────────────────────────────────────────────────────────────
# MetricsServer
# ──────────────────────────────────────────────────────────────────────────────


def test_metrics_server_serves_text_exposition() -> None:
    reg = MetricsRegistry()
    reg.observe_rollout_started("smoke")
    srv = MetricsServer(registry=reg, host="127.0.0.1", port=0)
    srv.start()
    try:
        url = f"http://{srv.host}:{srv.port}/metrics"
        body = urllib.request.urlopen(url, timeout=5.0).read().decode()
        assert 'xrlenv_rollouts_started_total{template="smoke"} 1.0' in body
    finally:
        srv.stop()


def test_metrics_server_double_start_is_noop() -> None:
    reg = MetricsRegistry()
    srv = MetricsServer(registry=reg, host="127.0.0.1", port=0)
    srv.start()
    try:
        original_port = srv.port
        srv.start()  # second call must not crash or re-bind
        assert srv.port == original_port
    finally:
        srv.stop()


# ──────────────────────────────────────────────────────────────────────────────
# /metrics content negotiation — raw exposition vs HTML dashboard
# ──────────────────────────────────────────────────────────────────────────────

# Real headers from the two clients we negotiate between.
_PROM_ACCEPT = (
    "application/openmetrics-text;version=1.0.0,"
    "text/plain;version=0.0.4;q=0.5,*/*;q=0.1"
)
_BROWSER_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"


def _http_get(srv: MetricsServer, path: str, accept: str) -> tuple[str, str]:
    """Return (content_type, body) for a GET against the running server."""
    req = urllib.request.Request(
        f"http://{srv.host}:{srv.port}{path}", headers={"Accept": accept}
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return resp.headers.get("Content-Type", ""), resp.read().decode()


def test_metrics_endpoint_serves_raw_to_prometheus_scraper() -> None:
    """A Prometheus-style Accept header must get the raw text exposition,
    never HTML — otherwise scraping silently breaks."""
    reg = MetricsRegistry()
    reg.observe_rollout_started("smoke")
    srv = MetricsServer(registry=reg, host="127.0.0.1", port=0)
    srv.start()
    try:
        ctype, body = _http_get(srv, "/metrics", _PROM_ACCEPT)
        assert "text/html" not in ctype
        assert 'xrlenv_rollouts_started_total{template="smoke"} 1.0' in body
    finally:
        srv.stop()


def test_metrics_endpoint_serves_html_to_browser() -> None:
    reg = MetricsRegistry()
    reg.observe_rollout_started("smoke")
    srv = MetricsServer(registry=reg, host="127.0.0.1", port=0)
    srv.start()
    try:
        ctype, body = _http_get(srv, "/metrics", _BROWSER_ACCEPT)
        assert "text/html" in ctype
        assert "<html" in body.lower()
        # The dashboard groups by the documented categories.
        assert "Rollout lifecycle" in body
        assert "rollouts started" in body
    finally:
        srv.stop()


def test_metrics_endpoint_format_query_overrides_accept() -> None:
    """``?format=raw`` / ``?format=html`` win over the Accept header."""
    reg = MetricsRegistry()
    reg.observe_rollout_started("smoke")
    srv = MetricsServer(registry=reg, host="127.0.0.1", port=0)
    srv.start()
    try:
        # Browser asks for HTML but forces raw.
        ctype, body = _http_get(srv, "/metrics?format=raw", _BROWSER_ACCEPT)
        assert "text/html" not in ctype
        assert "# HELP xrlenv_rollouts_started_total" in body
        # Scraper-style */* but forces HTML.
        ctype, body = _http_get(srv, "/metrics?format=html", "*/*")
        assert "text/html" in ctype
        assert "XRLEnv" in body
    finally:
        srv.stop()


def test_metrics_server_root_redirects_to_metrics() -> None:
    reg = MetricsRegistry()
    srv = MetricsServer(registry=reg, host="127.0.0.1", port=0)
    srv.start()
    try:
        # urllib transparently follows the 302 to /metrics (raw, no Accept).
        _ctype, body = _http_get(srv, "/", "*/*")
        assert "# HELP" in body
    finally:
        srv.stop()


def test_metrics_server_unknown_path_is_404() -> None:
    reg = MetricsRegistry()
    srv = MetricsServer(registry=reg, host="127.0.0.1", port=0)
    srv.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(
                f"http://{srv.host}:{srv.port}/nope", timeout=5.0
            )
        assert exc.value.code == 404
    finally:
        srv.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard rendering helpers
# ──────────────────────────────────────────────────────────────────────────────


def test_prefers_html_distinguishes_browser_from_scraper() -> None:
    assert prefers_html(_BROWSER_ACCEPT) is True
    assert prefers_html("text/html") is True
    # Prometheus / curl / empty must NOT be treated as a browser.
    assert prefers_html(_PROM_ACCEPT) is False
    assert prefers_html("*/*") is False
    assert prefers_html("") is False
    assert prefers_html(None) is False
    # An explicit q=0 on text/html is a refusal, not a request.
    assert prefers_html("text/html;q=0,*/*") is False


def test_render_dashboard_includes_live_values_and_rollups() -> None:
    reg = MetricsRegistry()
    reg.observe_rollout_started("tpl/a:main")
    for _ in range(3):
        reg.observe_rollout_finished("tpl/a:main", "finished")
    reg.observe_rollout_finished("tpl/a:main", "failed")
    reg.observe_queue_wait("tpl/a:main", 0.3)
    reg.observe_queue_wait("tpl/a:main", 2.0)

    html = render_dashboard_html(reg.collector_registry)

    # Category headers from the docs grouping are present.
    for header in ("Rollout lifecycle", "Latency", "Liveness", "Failures"):
        assert header in html
    # Counter total + status rollup rendered.
    assert "xrlenv_rollouts_finished_total" in html
    assert "By <code>status</code>" in html
    # Histogram summary columns rendered for the observed queue-wait series.
    assert "p50 (est)" in html
    # Success-rate card: 3 of 4 finished == 75.0%.
    assert "75.0%" in html


def test_render_dashboard_rolebar_clarifies_role_and_links_admin() -> None:
    """The role-clarifier banner must always state the data is ephemeral, and
    link to the admin panel when its port is known (drill-down pointer)."""
    reg = MetricsRegistry()
    html = render_dashboard_html(reg.collector_registry, admin_port=8080)
    assert "rolebar" in html
    assert "reset when the control plane" in html
    # Admin cross-link present with the configured port.
    assert ">admin panel</a>" in html
    assert "(port 8080)" in html
    # Link target is set client-side from the current host + admin port.
    assert "getElementById('adminlink')" in html
    assert "+8080+" in html


def test_render_dashboard_rolebar_without_admin_has_no_link() -> None:
    reg = MetricsRegistry()
    html = render_dashboard_html(reg.collector_registry)  # admin_port=None
    assert "rolebar" in html
    assert "the admin panel" in html
    # No dangling link / script when the admin port is unknown.
    assert "adminlink" not in html
    assert "<script>" not in html


def test_render_dashboard_shows_untouched_metrics_as_catalog() -> None:
    """A metric with no observations still renders (with its HELP text) so the
    page doubles as a live catalogue of the contract."""
    reg = MetricsRegistry()  # nothing observed
    html = render_dashboard_html(reg.collector_registry)
    assert "xrlenv_sandbox_create_seconds" in html
    assert "no samples yet" in html


def test_render_dashboard_escapes_label_values() -> None:
    reg = MetricsRegistry()
    reg.observe_rollout_started("tpl/<script>:main")
    html = render_dashboard_html(reg.collector_registry)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_estimate_quantile_interpolates_within_bucket() -> None:
    # Cumulative buckets: 5 obs in (0.1,0.5], 5 more in (0.5,1.0].
    buckets = [(0.1, 0.0), (0.5, 5.0), (1.0, 10.0), (float("inf"), 10.0)]
    p50 = _estimate_quantile(buckets, 0.50)
    p90 = _estimate_quantile(buckets, 0.90)
    assert p50 == pytest.approx(0.5, abs=1e-9)
    assert p90 == pytest.approx(0.9, abs=1e-9)


def test_estimate_quantile_open_bucket_returns_last_finite_edge() -> None:
    # Rank lands in the open +Inf bucket — fall back to the last finite le.
    buckets = [(1.0, 2.0), (float("inf"), 5.0)]
    assert _estimate_quantile(buckets, 0.90) == pytest.approx(1.0)


def test_estimate_quantile_no_observations_is_none() -> None:
    buckets = [(1.0, 0.0), (float("inf"), 0.0)]
    assert _estimate_quantile(buckets, 0.50) is None
    assert _estimate_quantile([], 0.5) is None


# ──────────────────────────────────────────────────────────────────────────────
# Sink event log
# ──────────────────────────────────────────────────────────────────────────────


def _sink_open(
    sink: PlatformJsonlSink, rollout_id: str = "rid-evt"
) -> Path:
    locator = sink.open(
        rollout_id=rollout_id, manifest=_manifest(),
        init={"max_steps": 1}, node_id="nid",
    )
    assert locator.uri is not None
    return Path(locator.uri.removeprefix("file://")).parent


def test_sink_record_event_writes_jsonl(tmp_path: Path) -> None:
    sink = PlatformJsonlSink(tmp_path)
    run_dir = _sink_open(sink)
    sink.record_event(
        "rid-evt", "rollout.start",
        {"template": "obs-t", "node_id": "nid"},
    )
    sink.record_event("rid-evt", "rollout.finish", {"status": "finished"})
    log_path = run_dir / "coordinator.log"
    assert log_path.exists()
    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [r["event"] for r in lines] == ["rollout.start", "rollout.finish"]
    assert lines[0]["payload"]["template"] == "obs-t"
    assert all("ts" in r for r in lines)


def test_sink_record_event_after_seal_reuses_disk_path(tmp_path: Path) -> None:
    """Event-log writes after seal still find the run dir by walking."""
    sink = PlatformJsonlSink(tmp_path)
    run_dir = _sink_open(sink)
    sink.seal(
        rollout_id="rid-evt", status=RolloutStatus.FINISHED,
        reason=None, final_reward=0.0, metadata={},
    )
    sink.record_event("rid-evt", "post.seal.note", {"k": "v"})
    body = (run_dir / "coordinator.log").read_text().splitlines()
    assert body, "expected at least one log entry after seal"
    assert json.loads(body[-1])["event"] == "post.seal.note"


def test_sink_record_event_unknown_rollout_is_silent(tmp_path: Path) -> None:
    """Logging an event for a never-seen rollout must not raise."""
    sink = PlatformJsonlSink(tmp_path)
    sink.record_event("never-existed", "ignored", {})  # no exception


# ──────────────────────────────────────────────────────────────────────────────
# Coordinator + admission integration
# ──────────────────────────────────────────────────────────────────────────────


async def test_successful_rollout_emits_lifecycle_metrics(tmp_path: Path) -> None:
    sink = PlatformJsonlSink(tmp_path)
    metrics = MetricsRegistry()
    client, coord, _state, _node = _build_runtime(
        sink=sink, metrics=metrics, max_steps=1,
    )
    try:
        s = await client.rollout(template="obs-t", init={"max_steps": 1})
        async with s:
            while not s.done:
                await s.step({"cmd": "noop"})
        rid = s.rollout_id

        # Counters
        assert _metric_value(
            metrics, "xrlenv_rollouts_started_total", template="obs-t"
        ) == 1.0
        assert _metric_value(
            metrics, "xrlenv_rollouts_finished_total",
            template="obs-t", status="finished",
        ) == 1.0
        # Sandbox-active gauge returns to 0 after destroy.
        assert _metric_value(
            metrics, "xrlenv_sandbox_active",
            node="fake-obs", template="obs-t",
        ) == 0.0
        # Step latency was observed at least once.
        step_count = _metric_value(
            metrics, "xrlenv_step_latency_seconds_count",
            template="obs-t", backend="docker",
        )
        assert step_count == 1.0
        # Sandbox create + destroy histograms each observed once.
        assert _metric_value(
            metrics, "xrlenv_sandbox_create_seconds_count",
            template="obs-t", backend="docker",
        ) == 1.0
        assert _metric_value(
            metrics, "xrlenv_sandbox_destroy_seconds_count",
            template="obs-t", backend="docker",
        ) == 1.0
        # Per-rollout coordinator.log captured the lifecycle inflection points.
        log_lines = [
            json.loads(line)
            for line in (
                next((p for p in tmp_path.rglob(f"{rid}/coordinator.log")), None)
                or pytest.fail("coordinator.log not found")
            ).read_text().splitlines()
        ]
        events = [r["event"] for r in log_lines]
        assert "rollout.start" in events
        assert "rollout.finish" in events
    finally:
        await _drain(coord, client)


async def test_failed_bootstrap_emits_create_failed_and_finished(
    tmp_path: Path,
) -> None:
    metrics = MetricsRegistry()
    sink = PlatformJsonlSink(tmp_path)
    client, coord, _state, _node = _build_runtime(
        sink=sink, metrics=metrics, fail_create=True,
    )
    try:
        with pytest.raises(RolloutFailed):
            await client.rollout(template="obs-t", init={})
        # Bootstrap classifier returned "sandbox_create_failed" for a generic
        # RuntimeError — confirm the create_failed counter incremented.
        assert _metric_value(
            metrics, "xrlenv_sandbox_create_failed_total",
            template="obs-t", reason="sandbox_create_failed",
        ) == 1.0
        # And the rollout was bucketed terminal.
        assert _metric_value(
            metrics, "xrlenv_rollouts_finished_total",
            template="obs-t", status="failed",
        ) == 1.0
        # No started counter — bootstrap raised before we count "started".
        assert _metric_value(
            metrics, "xrlenv_rollouts_started_total", template="obs-t"
        ) == 0.0
    finally:
        await _drain(coord, client)


async def test_admission_fast_path_increments_admitted() -> None:
    metrics = MetricsRegistry()
    client, coord, _state, _node = _build_runtime(metrics=metrics, max_steps=1)
    try:
        s = await client.rollout(template="obs-t", init={"max_steps": 1})
        async with s:
            await s.step({"cmd": "noop"})
        # Fast path: the scheduler placed immediately, so admitted=1, queued=0.
        assert _metric_value(
            metrics, "xrlenv_admission_total", result="admitted"
        ) == 1.0
        assert _metric_value(
            metrics, "xrlenv_admission_total", result="queued"
        ) == 0.0
    finally:
        await _drain(coord, client)


async def test_admission_queue_timeout_increments_counter() -> None:
    """A scheduler that always raises CapacityExhausted forces a queue
    wait → timeout — the queue_timeout counter must increment and the
    queued counter must show the request landed in the queue first.
    """
    metrics = MetricsRegistry()
    sched = MagicMock()
    sched.place.side_effect = CapacityExhausted("nope")
    sched.nodes = []
    state = InMemoryStateStore()
    queue = AdmissionQueue(scheduler=sched, state=state, metrics=metrics)
    await queue.start()
    try:
        with pytest.raises(CapacityExhausted):
            await queue.acquire(manifest=_manifest(), timeout_s=0.1)
        # Queued at least once; queue_timeout fired exactly once.
        assert _metric_value(
            metrics, "xrlenv_admission_total", result="queued"
        ) == 1.0
        assert _metric_value(
            metrics, "xrlenv_admission_total", result="queue_timeout"
        ) == 1.0
        # Depth gauge is back to 0.
        assert _metric_value(
            metrics, "xrlenv_queue_depth", template="obs-t"
        ) == 0.0
    finally:
        await queue.stop()


async def test_metrics_exposed_via_generate_latest_text_format() -> None:
    """Sanity: the registry's text exposition contains our HELP + TYPE lines."""
    metrics = MetricsRegistry()
    metrics.observe_rollout_started("smoke-t")
    text = generate_latest(metrics.collector_registry).decode()
    assert "# HELP xrlenv_rollouts_started_total" in text
    assert "# TYPE xrlenv_rollouts_started_total counter" in text
    assert 'xrlenv_rollouts_started_total{template="smoke-t"} 1.0' in text


async def test_sink_logs_terminal_event_for_truncated_rollout(
    tmp_path: Path,
) -> None:
    """When a rollout is hard-deadline truncated, the coordinator's
    terminal-status log entry uses ``rollout.truncate``, not ``rollout.finish``.
    """
    from xrlenv.types import Deadline

    sink = PlatformJsonlSink(tmp_path)
    metrics = MetricsRegistry()
    client, coord, _state, _node = _build_runtime(
        sink=sink, metrics=metrics, max_steps=10,
    )
    try:
        s = await client.rollout(
            template="obs-t",
            init={},
            deadline=Deadline(hard_s=0.05, idle_ttl_s=10.0),
        )
        rid = s.rollout_id
        # Wait for the deadline watcher to fire.
        await asyncio.sleep(0.2)
        async with s:
            with pytest.raises(RolloutTruncated):
                while not s.done:
                    await s.step({"cmd": "noop"})

        log_path = next(tmp_path.rglob(f"{rid}/coordinator.log"))
        events = [json.loads(line)["event"] for line in log_path.read_text().splitlines()]
        # truncation comes from the deadline watcher path; the per-rollout
        # log must show the truncate event (no finish) for that rollout.
        assert "rollout.truncate" in events
        assert "rollout.finish" not in events
    finally:
        await _drain(coord, client)

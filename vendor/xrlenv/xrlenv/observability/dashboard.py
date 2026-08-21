"""Human-readable HTML rendering of the Prometheus ``/metrics`` registry.

The ``/metrics`` endpoint (spec 08) must keep emitting the raw Prometheus
text exposition format so Prometheus can scrape it. But that format is
illegible to a human who opens ``:9090/metrics`` in a browser — it's a
flat wall of ``xrlenv_*_bucket{le="…"}`` lines. This module renders the
*same* live registry into a grouped operational dashboard: counters with
totals + breakdowns, histograms summarised as count / mean / p50..p99
(estimated from the bucket boundaries), and gauges as current values.

It reads only what :py:meth:`CollectorRegistry.collect` exposes, so a
newly-added series shows up automatically (categorised under "Other" if
it isn't in :data:`_CATEGORIES`). No data is dropped and no series is
hard-coded — a metric with no samples yet renders as a labelled
"no samples yet" row, so the page doubles as a live catalogue of the
contract documented in ``docs/observability/metrics.md``.

Kept deliberately dependency-light (stdlib + the registry only) — the
:mod:`xrlenv.observability.server` exposer is a small standalone WSGI app
and must not drag in the admin panel's FastAPI/Jinja stack.
"""

from __future__ import annotations

import html
import math
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from prometheus_client import CollectorRegistry
    from prometheus_client.metrics_core import Metric


# Category → ordered family names. Family names are what ``collect()``
# reports: counters strip the ``_total`` suffix, histograms keep
# ``_seconds``, gauges keep their full name. Mirrors the grouping in
# ``docs/observability/metrics.md`` so the live page and the contract doc
# line up. Anything not listed here lands under "Other".
_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Rollout lifecycle",
        ("xrlenv_rollouts_started", "xrlenv_rollouts_finished"),
    ),
    (
        "Latency",
        (
            "xrlenv_step_latency_seconds",
            "xrlenv_sandbox_create_seconds",
            "xrlenv_sandbox_destroy_seconds",
            "xrlenv_queue_wait_seconds",
        ),
    ),
    (
        "Liveness",
        (
            "xrlenv_sandbox_active",
            "xrlenv_queue_depth",
            "xrlenv_node_admission_limit",
        ),
    ),
    (
        "Failures & rejections",
        ("xrlenv_sandbox_create_failed", "xrlenv_admission"),
    ),
)

# Labels that name a low-cardinality outcome dimension. When a counter
# carries one of these we render a cross-cut "By <label>" rollup with
# percentages on top of the full per-labelset table.
_BREAKDOWN_LABELS: tuple[str, ...] = ("status", "result", "reason", "outcome")

_QUANTILES: tuple[float, ...] = (0.50, 0.90, 0.99)


# ── content negotiation ─────────────────────────────────────────────────────


def prefers_html(accept_header: str | None) -> bool:
    """Whether an HTTP ``Accept`` header asks for HTML over plain text.

    Prometheus scrapers send
    ``application/openmetrics-text;…,text/plain;…,*/*;q=0.1`` — never
    ``text/html`` — while browsers lead with ``text/html``. So presence of
    an explicit ``text/html`` (or ``application/xhtml+xml``) range that
    isn't quality-zero is a safe, scraper-proof signal that a human is
    looking. ``*/*`` (curl's default) deliberately does **not** count, so
    ``curl :9090/metrics`` keeps returning the raw exposition.
    """
    if not accept_header:
        return False
    for part in accept_header.split(","):
        media, _, params = part.strip().partition(";")
        media = media.strip().lower()
        if media not in ("text/html", "application/xhtml+xml"):
            continue
        q = 1.0
        for param in params.split(";"):
            key, _, value = param.strip().partition("=")
            if key.strip().lower() == "q":
                try:
                    q = float(value)
                except ValueError:
                    q = 0.0
        if q > 0.0:
            return True
    return False


# ── small sample-reading helpers ────────────────────────────────────────────


def _counter_rows(family: Metric) -> list[tuple[Mapping[str, str], float]]:
    """``(labels, value)`` for each ``_total`` sample of a counter family."""
    return [
        (s.labels, s.value) for s in family.samples if s.name.endswith("_total")
    ]


def _gauge_rows(family: Metric) -> list[tuple[Mapping[str, str], float]]:
    """``(labels, value)`` for each gauge sample (sample name == family name)."""
    return [(s.labels, s.value) for s in family.samples if s.name == family.name]


def _counter_total(family: Metric | None) -> float:
    return 0.0 if family is None else sum(v for _, v in _counter_rows(family))


def _counter_by_label(family: Metric | None, label: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if family is None:
        return out
    for labels, value in _counter_rows(family):
        out[labels.get(label, "?")] = out.get(labels.get(label, "?"), 0.0) + value
    return out


def _gauge_total(family: Metric | None) -> float:
    return 0.0 if family is None else sum(v for _, v in _gauge_rows(family))


def _histogram_groups(family: Metric) -> list[dict[str, object]]:
    """Collapse a histogram family into one entry per non-``le`` labelset.

    Each entry carries the sorted ``(le, cumulative_count)`` buckets plus
    the ``_count`` and ``_sum`` totals, enough to derive mean + estimated
    quantiles.
    """
    groups: dict[tuple[tuple[str, str], ...], dict[str, object]] = {}
    for s in family.samples:
        base = {k: v for k, v in s.labels.items() if k != "le"}
        key = tuple(sorted(base.items()))
        group = groups.setdefault(
            key, {"labels": base, "buckets": [], "count": 0.0, "sum": 0.0}
        )
        if s.name.endswith("_bucket"):
            le_raw = s.labels.get("le", "+Inf")
            le = math.inf if le_raw in ("+Inf", "Inf", "inf") else float(le_raw)
            buckets = group["buckets"]
            assert isinstance(buckets, list)
            buckets.append((le, s.value))
        elif s.name.endswith("_count"):
            group["count"] = s.value
        elif s.name.endswith("_sum"):
            group["sum"] = s.value
    for group in groups.values():
        buckets = group["buckets"]
        assert isinstance(buckets, list)
        buckets.sort(key=lambda b: b[0])
    return list(groups.values())


def _estimate_quantile(
    buckets: Sequence[tuple[float, float]], q: float
) -> float | None:
    """Linear-interpolated quantile estimate from cumulative buckets.

    Same shape as Prometheus' ``histogram_quantile``: locate the bucket the
    target rank lands in and interpolate within it. Returns ``None`` when
    there are no observations or the rank falls in the open ``+Inf`` bucket
    (no finite upper bound to interpolate toward). The result is an estimate
    bounded by the configured bucket edges — labelled "(est)" in the UI.
    """
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None
    rank = q * total
    prev_le = 0.0
    prev_cum = 0.0
    for le, cum in buckets:
        if cum >= rank:
            if math.isinf(le):
                return prev_le if prev_cum > 0 else None
            span = cum - prev_cum
            if span <= 0:
                return le
            return prev_le + (le - prev_le) * (rank - prev_cum) / span
        prev_le, prev_cum = le, cum
    return buckets[-1][0]


# ── formatting ───────────────────────────────────────────────────────────────


def _esc(value: object) -> str:
    return html.escape(str(value))


def _fmt_count(value: float) -> str:
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f}µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f}ms"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, rem = divmod(seconds, 60.0)
    return f"{int(minutes)}m{rem:04.1f}s"


def _labels_text(labels: Mapping[str, str]) -> str:
    if not labels:
        return "<span class='dim'>(no labels)</span>"
    return " ".join(
        f"<span class='k'>{_esc(k)}</span>=<span class='v'>{_esc(v)}</span>"
        for k, v in labels.items()
    )


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """Render a table. Header cells are escaped; row cells are inserted
    verbatim (callers pre-escape label text and format numbers)."""
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body: list[str] = []
    for row in rows:
        cells = "".join(
            f"<td class='{'lbl' if i == 0 else 'num'}'>{cell}</td>"
            for i, cell in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


# ── per-family blocks ────────────────────────────────────────────────────────


def _exposed_counter_name(family: Metric) -> str:
    return family.name if family.name.endswith("_total") else f"{family.name}_total"


def _render_counter(family: Metric) -> str:
    rows = _counter_rows(family)
    title = _exposed_counter_name(family)
    parts = [
        f"<div class='metric'><div class='mhead'>"
        f"<code>{_esc(title)}</code>"
        f"<span class='badge counter'>counter</span></div>"
        f"<p class='help'>{_esc(family.documentation)}</p>"
    ]
    if not rows:
        parts.append("<p class='empty'>no samples yet</p></div>")
        return "".join(parts)

    total = sum(v for _, v in rows)
    parts.append(f"<p class='total'>total <b>{_fmt_count(total)}</b></p>")

    label_names = [k for k in rows[0][0]]
    breakdown = next((b for b in _BREAKDOWN_LABELS if b in label_names), None)

    if breakdown is not None:
        by = _counter_by_label(family, breakdown)
        b_rows = [
            (
                _esc(value),
                _fmt_count(count),
                f"{(count / total * 100.0):.1f}%" if total else "—",
            )
            for value, count in sorted(by.items(), key=lambda kv: -kv[1])
        ]
        parts.append(f"<div class='sub'>By <code>{_esc(breakdown)}</code></div>")
        parts.append(_table((breakdown, "count", "% of total"), b_rows))

    # Full per-labelset detail — skip when the only label is the breakdown
    # one (the rollup above already shows every row).
    if not (breakdown is not None and label_names == [breakdown]):
        detail = [
            (_labels_text(labels), _fmt_count(value))
            for labels, value in sorted(rows, key=lambda r: -r[1])
        ]
        parts.append(_table(("labels", "count"), detail))

    parts.append("</div>")
    return "".join(parts)


def _render_gauge(family: Metric) -> str:
    rows = _gauge_rows(family)
    parts = [
        f"<div class='metric'><div class='mhead'>"
        f"<code>{_esc(family.name)}</code>"
        f"<span class='badge gauge'>gauge</span></div>"
        f"<p class='help'>{_esc(family.documentation)}</p>"
    ]
    if not rows:
        parts.append("<p class='empty'>no samples yet</p></div>")
        return "".join(parts)

    total = sum(v for _, v in rows)
    if len(rows) > 1:
        parts.append(f"<p class='total'>sum <b>{_fmt_count(total)}</b></p>")
    detail = [
        (_labels_text(labels), _fmt_count(value))
        for labels, value in sorted(rows, key=lambda r: -r[1])
    ]
    parts.append(_table(("labels", "current"), detail))
    parts.append("</div>")
    return "".join(parts)


def _render_histogram(family: Metric) -> str:
    groups = _histogram_groups(family)
    observed = [g for g in groups if float(g["count"]) > 0]  # type: ignore[arg-type]
    parts = [
        f"<div class='metric'><div class='mhead'>"
        f"<code>{_esc(family.name)}</code>"
        f"<span class='badge hist'>histogram</span></div>"
        f"<p class='help'>{_esc(family.documentation)}</p>"
    ]
    if not observed:
        parts.append("<p class='empty'>no samples yet</p></div>")
        return "".join(parts)

    headers = ("labels", "count", "mean", "p50 (est)", "p90 (est)", "p99 (est)")
    rows: list[tuple[str, ...]] = []
    for group in sorted(observed, key=lambda g: -float(g["count"])):  # type: ignore[arg-type]
        labels = group["labels"]
        count = float(group["count"])  # type: ignore[arg-type]
        total = float(group["sum"])  # type: ignore[arg-type]
        buckets = group["buckets"]
        assert isinstance(labels, dict) and isinstance(buckets, list)
        mean = total / count if count > 0 else None
        quants = [_estimate_quantile(buckets, q) for q in _QUANTILES]
        rows.append(
            (
                _labels_text(labels),
                _fmt_count(count),
                _fmt_seconds(mean),
                *[_fmt_seconds(v) for v in quants],
            )
        )
    parts.append(_table(headers, rows))
    parts.append("</div>")
    return "".join(parts)


def _render_family(family: Metric) -> str:
    if family.type == "histogram":
        return _render_histogram(family)
    if family.type == "gauge":
        return _render_gauge(family)
    # counter, and the "_created" summary counters fall through here too.
    return _render_counter(family)


# ── summary cards ────────────────────────────────────────────────────────────


def _stat_card(value: str, label: str, *, tone: str = "") -> str:
    cls = f"card {tone}".strip()
    return (
        f"<div class='{cls}'><div class='big'>{_esc(value)}</div>"
        f"<div class='cap'>{_esc(label)}</div></div>"
    )


def _render_rolebar(admin_port: int | None) -> str:
    """Banner clarifying what this view is — and is *not*.

    The numbers here are aggregate Prometheus counters that reset on
    control-plane restart; the admin panel holds the authoritative,
    per-entity, restart-surviving record. Spelling that out kills the
    "did rollouts disappear?" confusion when the two surfaces disagree
    after a restart. When the admin port is known we add a best-effort
    link: it points at the *current* host on the admin port, set
    client-side so it follows whatever host the operator loaded this
    page from (correct on-box; may need adjusting through a port-forward
    that remaps the port — hence the tooltip).
    """
    if admin_port is not None:
        admin = (
            "the <a id='adminlink' href='#' title='Admin panel — assumes it is "
            f"reachable on this host at port {admin_port}; adjust if you tunnel it "
            f"on a different port'>admin panel</a> (port {admin_port})"
        )
        script = (
            "<script>(function(){var a=document.getElementById('adminlink');"
            "if(a)a.href=location.protocol+'//'+location.hostname+':'+"
            + str(admin_port)
            + "+'/';})();</script>"
        )
    else:
        admin = "the admin panel"
        script = ""
    return (
        "<section class='rolebar'><span class='ico'>&#9432;</span>"
        "<span>Aggregate Prometheus counters — <b>reset when the control plane "
        "restarts</b>; meant for scraping and at-a-glance trends, not authoritative "
        "history. For per-rollout / per-sandbox / per-node detail and drill-down, use "
        f"{admin}.</span></section>{script}"
    )


def _render_summary(by_name: Mapping[str, Metric]) -> str:
    started = _counter_total(by_name.get("xrlenv_rollouts_started"))
    finished_by = _counter_by_label(by_name.get("xrlenv_rollouts_finished"), "status")
    finished = sum(finished_by.values())
    in_flight = max(0.0, started - finished)
    completed = finished_by.get("finished", 0.0)
    success = (completed / finished * 100.0) if finished else None
    active = _gauge_total(by_name.get("xrlenv_sandbox_active"))
    depth = _gauge_total(by_name.get("xrlenv_queue_depth"))
    create_failed = _counter_total(by_name.get("xrlenv_sandbox_create_failed"))

    fail_tone = "warn" if create_failed > 0 else ""
    cards = [
        _stat_card(_fmt_count(started), "rollouts started"),
        _stat_card(_fmt_count(in_flight), "in flight", tone="accent"),
        _stat_card(_fmt_count(finished), "finished"),
        _stat_card(
            "—" if success is None else f"{success:.1f}%",
            "completed (status=finished)",
        ),
        _stat_card(_fmt_count(active), "active sandboxes"),
        _stat_card(_fmt_count(depth), "queue depth", tone="warn" if depth > 0 else ""),
        _stat_card(_fmt_count(create_failed), "create failures", tone=fail_tone),
    ]
    return f"<section class='cards'>{''.join(cards)}</section>"


# ── page assembly ────────────────────────────────────────────────────────────


def render_dashboard_html(
    registry: CollectorRegistry,
    *,
    refresh_s: int = 5,
    admin_port: int | None = None,
) -> str:
    """Render the live metrics registry as a standalone HTML dashboard.

    :param registry: the prometheus-client ``CollectorRegistry`` to read.
    :param refresh_s: browser auto-refresh interval in seconds; ``0``
        disables the ``<meta refresh>`` tag.
    :param admin_port: the admin panel's port, if running. When set, the
        role-clarifier banner links to the admin panel for drill-down.
    """
    families = list(registry.collect())
    by_name = {f.name: f for f in families}
    rendered: set[str] = set()

    sections: list[str] = [_render_rolebar(admin_port), _render_summary(by_name)]

    for title, names in _CATEGORIES:
        blocks: list[str] = []
        for name in names:
            family = by_name.get(name)
            if family is None:
                continue
            blocks.append(_render_family(family))
            rendered.add(name)
        if blocks:
            sections.append(
                f"<section><h2>{_esc(title)}</h2>{''.join(blocks)}</section>"
            )

    # Anything the category map didn't claim (future series, _created
    # helpers) — surface it rather than silently dropping it.
    leftover = [
        f for f in families if f.name not in rendered and f.samples
    ]
    if leftover:
        blocks = [_render_family(f) for f in leftover]
        sections.append(f"<section><h2>Other</h2>{''.join(blocks)}</section>")

    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    refresh_meta = (
        f"<meta http-equiv='refresh' content='{refresh_s}'>" if refresh_s > 0 else ""
    )
    refresh_note = f"auto-refresh {refresh_s}s" if refresh_s > 0 else "auto-refresh off"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XRLEnv metrics</title>
{refresh_meta}
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>XRLEnv <span class="path">/metrics</span></h1>
  <div class="meta">
    rendered {ts} UTC · {refresh_note} ·
    <a href="?format=raw">raw exposition</a> ·
    <a href="?refresh=2">2s</a> · <a href="?refresh=5">5s</a> ·
    <a href="?refresh=30">30s</a> · <a href="?refresh=0">pause</a>
  </div>
</header>
{''.join(sections)}
<footer>
  Prometheus scrapes the raw text format at this same URL
  (<code>Accept: text/plain</code> or <a href="?format=raw">?format=raw</a>).
  Percentiles are estimated from histogram bucket edges, not exact.
</footer>
</body>
</html>"""


_CSS = """
:root{--bg:#0f1115;--panel:#181b22;--line:#262b35;--fg:#e6e9ef;--dim:#8b93a3;
--accent:#5b9dff;--warn:#e0a44b;--good:#5fd07a;--code:#9ad0ff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
padding:0 0 3rem}
header{padding:1.4rem 1.6rem 1rem;border-bottom:1px solid var(--line);
position:sticky;top:0;background:var(--bg);z-index:5}
h1{margin:0;font-size:1.3rem;font-weight:650;letter-spacing:.2px}
h1 .path{color:var(--accent);font-weight:500}
h2{font-size:1rem;text-transform:uppercase;letter-spacing:1.2px;color:var(--dim);
margin:2rem 1.6rem .4rem;font-weight:600}
.meta{color:var(--dim);font-size:12.5px;margin-top:.35rem}
.meta a{color:var(--accent);text-decoration:none}.meta a:hover{text-decoration:underline}
section{padding:0 .6rem}
.rolebar{display:flex;gap:.55rem;align-items:flex-start;margin:.9rem 1.6rem 0;
padding:.6rem .8rem;background:#15233a;border:1px solid #243a5e;border-radius:9px;
color:#bcd4f5;font-size:12.5px;line-height:1.45}
.rolebar .ico{color:var(--accent);font-size:15px;line-height:1.2}
.rolebar b{color:#e6e9ef}.rolebar a{color:var(--accent)}
.cards{display:flex;flex-wrap:wrap;gap:.7rem;padding:1.1rem 1.6rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:.7rem .95rem;min-width:130px;flex:1 1 130px}
.card .big{font-size:1.6rem;font-weight:680;font-variant-numeric:tabular-nums}
.card .cap{color:var(--dim);font-size:11.5px;margin-top:.15rem}
.card.accent .big{color:var(--accent)}.card.warn .big{color:var(--warn)}
.metric{background:var(--panel);border:1px solid var(--line);border-radius:10px;
margin:.7rem 1rem;padding:.8rem 1rem}
.mhead{display:flex;align-items:center;gap:.6rem}
.mhead code{color:var(--code);font-size:13.5px;font-weight:600}
.badge{font-size:10.5px;text-transform:uppercase;letter-spacing:.6px;
padding:.1rem .42rem;border-radius:5px;color:#0f1115;font-weight:700}
.badge.counter{background:#7aa2ff}.badge.gauge{background:#5fd07a}
.badge.hist{background:#c79bff}
.help{color:var(--dim);margin:.35rem 0 .5rem;font-size:12.5px}
.total{margin:.1rem 0 .55rem;color:var(--fg);font-size:12.5px}
.total b{font-variant-numeric:tabular-nums}
.sub{color:var(--dim);font-size:11.5px;text-transform:uppercase;letter-spacing:.7px;
margin:.5rem 0 .25rem}
.empty{color:var(--dim);font-style:italic;margin:.2rem 0 0}
table{border-collapse:collapse;width:100%;margin:.25rem 0 .4rem;font-size:12.5px}
th{text-align:left;color:var(--dim);font-weight:600;border-bottom:1px solid var(--line);
padding:.3rem .5rem;white-space:nowrap}
td{padding:.28rem .5rem;border-bottom:1px solid #20242d;vertical-align:top}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:none}
.k{color:var(--dim)}.v{color:var(--fg)}.dim{color:var(--dim)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
footer{color:var(--dim);font-size:12px;margin:2.4rem 1.6rem 0;
border-top:1px solid var(--line);padding-top:.9rem}
footer code{color:var(--code)}footer a{color:var(--accent)}
"""


__all__ = ["prefers_html", "render_dashboard_html"]

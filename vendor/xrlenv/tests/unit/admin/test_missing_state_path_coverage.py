"""Unit tests for audit "missing state-path coverage" — GAP 3.

GAP 3 — explicit H4 banner assertion (users.html via the /users HTTP route).

The existing test_users_span.py checks _users_blocking() internals. This file
adds an HTTP-level test that the RENDERED /users page HTML actually contains:
  - "cumulative" (the word)
  - "since" (the word, appearing as part of the inception banner)
  - the inception timestamp string (e.g. "2023-…" or "2026-…" in UTC)

These three elements together confirm the H4 "cumulative since X" banner is
rendered correctly when rollouts exist and the lifetime-inception stamp has been
written.

NOTE: At the time this test was authored the /users route handler did NOT pass
``span_start``, ``span_end``, or ``inception`` to the template context — only
``rows``, ``totals``, and ``active_page``. This means the Jinja2 template
rendered "since lifetime tracking was enabled" (the fallback branch) instead of
the actual inception timestamp. This test intentionally asserts the CORRECT
behaviour and will fail until the route handler is fixed to thread those keys.

Isolation
---------
These tests use the loopback TestClient (no authentication required for
loopback binds). The XRLENV_SQLITE_JOURNAL_MODE env is NOT set, so the
SqliteStateStore inside _users_blocking() opens in WAL mode — which is correct
for an ephemeral tmp_path DB in a test.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from xrlenv.admin.server import AdminServerConfig, build_admin_app
from xrlenv.control.state import RawRolloutRecord, SqliteStateStore

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _raw(rollout_id: str, *, owner: str = "alice", status: str = "released",
         created_at: float | None = None) -> RawRolloutRecord:
    return RawRolloutRecord(
        rollout_id=rollout_id,
        status=status,  # type: ignore[arg-type]
        image="busybox:1",
        owner_id=owner,
        created_at=created_at if created_at is not None else time.time(),
    )


@pytest.fixture
def state_db(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    (tmp_path / "runs").mkdir()
    return tmp_path / "runs"


# ──────────────────────────────────────────────────────────────────────────────
# GAP 3 — H4 cumulative-inception banner in rendered HTML
# ──────────────────────────────────────────────────────────────────────────────


def test_users_page_renders_cumulative_word_when_rollouts_exist(
    state_db: Path, runs_root: Path,
) -> None:
    """The rendered /users HTML contains the word 'cumulative' when rollouts exist.

    This confirms the H4 info banner (which labels totals as cumulative, preserved
    across GC) is rendered to the client.
    """
    s = SqliteStateStore(state_db)
    s.record_raw_rollout(_raw("r1", created_at=1_700_000_000.0))
    s.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root)
    with TestClient(build_admin_app(cfg)) as client:
        resp = client.get("/users")

    assert resp.status_code == 200
    assert "cumulative" in resp.text, (
        "Expected 'cumulative' in /users page HTML when rollouts exist; "
        "the H4 info banner may be missing from the rendered output"
    )


def test_users_page_renders_since_word_when_rollouts_exist(
    state_db: Path, runs_root: Path,
) -> None:
    """The rendered /users HTML contains the word 'since' when rollouts exist.

    The H4 banner reads: 'They cover activity since <inception-date> — rollouts
    pruned before that are not backfilled.' The word 'since' must appear.
    """
    s = SqliteStateStore(state_db)
    s.record_raw_rollout(_raw("r1", created_at=1_700_000_000.0))
    s.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root)
    with TestClient(build_admin_app(cfg)) as client:
        resp = client.get("/users")

    assert resp.status_code == 200
    assert "since" in resp.text, (
        "Expected 'since' in /users page HTML when rollouts exist"
    )


def test_users_page_renders_inception_timestamp_when_rollouts_exist(
    state_db: Path, runs_root: Path,
) -> None:
    """The rendered /users HTML contains the actual inception timestamp string.

    When the SqliteStateStore has a lifetime_inception_ts stamp and the /users
    route passes 'inception' to the template, the Jinja2 {{ inception }} block
    renders the formatted date instead of the fallback text
    'since lifetime tracking was enabled'.

    This test will FAIL until the /users route handler is fixed to thread the
    'inception' key from _users_blocking() into the template context. It is
    intentionally written to expose the current gap.

    Expected: page contains 'since <strong>20YY-' (the ISO UTC date prefix).
    Not expected: 'since lifetime tracking was enabled' (fallback, no timestamp).
    """
    s = SqliteStateStore(state_db)
    # Record rollouts with a fixed, well-known timestamp so the inception stamp is set.
    s.record_raw_rollout(_raw("r1", created_at=1_700_000_000.0))
    s.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root)
    with TestClient(build_admin_app(cfg)) as client:
        resp = client.get("/users")

    assert resp.status_code == 200

    # The inception timestamp should appear as a formatted date like "2023-11-14 22:13:20 UTC".
    # It's rendered as <strong>{{ inception }}</strong> in the template.
    import re
    has_inception_date = bool(re.search(r"since <strong>20\d+-\d\d-\d\d", resp.text))
    has_fallback = "since lifetime tracking was enabled" in resp.text

    assert has_inception_date, (
        "Expected /users HTML to contain 'since <strong>20YY-MM-DD' (the rendered "
        "inception timestamp from lifetime_inception_ts). "
        f"Got fallback text instead: 'since lifetime tracking was enabled'={has_fallback}. "
        "The /users route handler must pass 'inception' (from _users_blocking()) "
        "into the template context so the H4 banner shows the actual timestamp."
    )


def test_users_page_no_inception_banner_when_db_empty(
    state_db: Path, runs_root: Path,
) -> None:
    """With no rollouts, the H4 banner block ({% if rows %}) is suppressed.

    The page should show 'No raw rollouts recorded yet.' and NOT contain the
    cumulative banner (since the Jinja block is gated on rows existing).
    """
    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root)
    with TestClient(build_admin_app(cfg)) as client:
        resp = client.get("/users")

    assert resp.status_code == 200
    assert "No raw rollouts recorded yet" in resp.text
    # Banner is gated on {% if rows %} — must be absent when empty.
    assert "cumulative" not in resp.text, (
        "The H4 cumulative banner must not appear when there are no rollouts"
    )


def test_users_page_renders_span_window_when_live_rollouts_exist(
    state_db: Path, runs_root: Path,
) -> None:
    """When live raw_rollouts rows exist, the retention window (span_start/span_end)
    should also appear in the rendered HTML.

    The template renders: 'Individual rollout records stay browsable only for the
    retention window (<strong>{{ span_start }}</strong> - <strong>{{ span_end }}</strong>).'

    This test checks that span_start appears as a formatted date in the rendered page.
    It will FAIL until the /users route handler threads span_start/span_end into
    the template context.
    """
    import datetime

    # A single row fixes span_start == span_end == its created_at, so the
    # retention-window renders one exact, well-known UTC timestamp. Asserting the
    # SPECIFIC value (not merely "some UTC date") is what proves span_start is
    # actually threaded through: the inception stamp is "now" (a different date),
    # so it cannot vacuously satisfy this assertion.
    created = 1_700_000_000.0
    s = SqliteStateStore(state_db)
    s.record_raw_rollout(_raw("r1", created_at=created))
    s.close()

    cfg = AdminServerConfig(state_db=state_db, runs_root=runs_root)
    with TestClient(build_admin_app(cfg)) as client:
        resp = client.get("/users")

    assert resp.status_code == 200

    # Mirror admin.server._iso exactly so the expected string can't drift.
    expected_span = datetime.datetime.fromtimestamp(
        created, tz=datetime.UTC,
    ).strftime("%Y-%m-%d %H:%M:%S UTC")
    assert expected_span == "2023-11-14 22:13:20 UTC"  # sanity-pin the fixture
    assert expected_span in resp.text, (
        f"Expected the exact span_start value {expected_span!r} in the /users "
        "HTML retention-window line. The route handler must thread span_start / "
        "span_end into the template context (the inception date alone, which is "
        "'now', must not be what satisfies this)."
    )

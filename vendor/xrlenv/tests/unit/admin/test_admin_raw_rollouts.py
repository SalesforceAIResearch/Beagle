"""P1.7.B.3 — admin's /rollouts surface for raw rollouts +
artifact-path resolution.

Pins:

- ``/rollouts`` lists raw rollouts alongside case-1 rollouts.
- ``/raw-rollouts/<id>`` 404s on missing rollout.
- ``/raw-rollouts/<id>`` 200s on present rollout, renders the
  displayed_name when set, falls back to rollout_id prefix.
- Artifact-path resolution: reachable directory → rendered as
  inline listing; missing path → "missing" status; non-directory
  → "not_a_dir" status.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from xrlenv.admin.server import (
    AdminServerConfig,
    _resolve_artifact_path,
    build_admin_app,
)
from xrlenv.control.state import RawRolloutRecord, SqliteStateStore


@pytest.fixture
def cfg(tmp_path: Path) -> AdminServerConfig:
    return AdminServerConfig(
        state_db=tmp_path / "state.db",
        runs_root=tmp_path / "runs",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Artifact-path resolution
# ──────────────────────────────────────────────────────────────────────────────


def test_resolve_artifact_missing_returns_missing(tmp_path: Path) -> None:
    status, listing = _resolve_artifact_path(str(tmp_path / "no-such-dir"))
    assert status == "missing"
    assert listing == []


def test_resolve_artifact_directory_lists_contents(tmp_path: Path) -> None:
    # Build a small artifact tree mirroring what swebench writes.
    art = tmp_path / "logs" / "run_evaluation" / "run-1" / "model-1" / "instance-1"
    art.mkdir(parents=True)
    (art / "report.json").write_text('{"resolved": true}')
    (art / "run_instance.log").write_text("first line\nsecond line\n")
    (art / "patch.diff").write_text("--- a\n+++ b\n")

    status, listing = _resolve_artifact_path(str(art))
    assert status == "ok"
    names = [e["name"] for e in listing]
    assert "report.json" in names
    assert "run_instance.log" in names
    assert "patch.diff" in names
    # Sizes recorded.
    by_name = {e["name"]: e for e in listing}
    assert by_name["report.json"]["kind"] == "file"
    assert by_name["report.json"]["size_bytes"] > 0


def test_resolve_artifact_not_a_dir(tmp_path: Path) -> None:
    f = tmp_path / "some_file.txt"
    f.write_text("hi")
    status, listing = _resolve_artifact_path(str(f))
    assert status == "not_a_dir"
    assert listing == []


# ──────────────────────────────────────────────────────────────────────────────
# /rollouts surface
# ──────────────────────────────────────────────────────────────────────────────


def test_rollouts_list_omits_rollout_id_and_container_columns(
    cfg: AdminServerConfig,
) -> None:
    """The list view at /rollouts/raw deliberately drops the
    rollout_id and container_id columns — they're uuid prefixes
    that aren't useful for at-a-glance scanning. Both values stay
    visible on each rollout's detail page (covered separately).

    Operator-driven UX call (2026-05-13): the list previously
    rendered 10 columns including 12-char hex prefixes of both
    uuids; collapsing those to the detail page keeps the list
    compact at 8 columns and matches operator scan patterns
    (filter by task_key/group_id, click to drill in for ids).
    """
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStateStore(cfg.state_db)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-detail-1234567890",  # 12-char prefix would be 'r-detail-12'
        status="released", image="busybox:1",
        node_id="node-A",
        container_id="c-very-very-distinctive-id",  # prefix 'c-very-very-'
        container_name="cname-detail",
        displayed_name="instance-A",
        created_at=time.time(),
    ))

    app = build_admin_app(cfg)
    with TestClient(app) as client:
        body = client.get("/rollouts/raw").text
        # Column headers absent from the list.
        for header in ("<th>rollout</th>", "<th>container</th>"):
            assert header not in body, (
                f"{header!r} should be hidden in the list view; got: "
                f"{[line for line in body.splitlines() if '<th>' in line]}"
            )
        # The 12-char uuid prefixes don't appear as rendered cells.
        # (We can't fully assert the rollout_id is absent — it's
        # the anchor href target. But its 12-char prefix shouldn't
        # render as a visible cell, and container_id is gone
        # entirely from this page.)
        assert "<code>c-very-very-" not in body, (
            "container_id should not be rendered as a list-view cell"
        )
        # The detail page DOES render both in full — link target
        # uses the full rollout_id.
        assert "/raw-rollouts/r-detail-1234567890" in body


def test_rollouts_page_renders_raw_rollouts(
    cfg: AdminServerConfig, tmp_path: Path,
) -> None:
    """Admin's /rollouts page surfaces raw rollouts in a separate
    table below the case-1 list. Empty state shows the
    explanatory paragraph + the (no raw rollouts...) line."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStateStore(cfg.state_db)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-acq-1",
        status="released",
        image="swebench/sweb:latest",
        node_id="node-A",
        container_id="c-001",
        container_name="cname-001",
        displayed_name="astropy__astropy-7166",
        artifact_path=str(tmp_path / "logs"),
        created_at=time.time(),
        finished_at=time.time(),
    ))

    app = build_admin_app(cfg)
    with TestClient(app) as client:
        # Bare /rollouts redirects to /rollouts/raw (the case-2/3
        # default landing under the slim pivot).
        resp = client.get("/rollouts")
        assert resp.status_code == 200
        body = resp.text
        # Page header is present.
        assert "Raw container rollouts" in body
        # Displayed_name appears in the rendered row.
        assert "astropy__astropy-7166" in body
        # Image appears.
        assert "swebench/sweb:latest" in body


def test_raw_rollout_detail_404_on_missing(
    cfg: AdminServerConfig,
) -> None:
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    SqliteStateStore(cfg.state_db)  # initialize empty DB
    app = build_admin_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/raw-rollouts/r-missing")
        assert resp.status_code == 404


def test_raw_rollout_detail_renders_metadata(
    cfg: AdminServerConfig, tmp_path: Path,
) -> None:
    """Detail page shows displayed_name, status, image, container,
    artifact_path. With artifact_path that resolves on disk, the
    'Artifacts' section lists the directory contents."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    art = tmp_path / "logs"
    art.mkdir()
    (art / "report.json").write_text('{"resolved": true}')

    store = SqliteStateStore(cfg.state_db)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-acq-1",
        status="released",
        image="swebench/sweb:latest",
        node_id="node-A",
        container_id="c-001",
        container_name="cname-001",
        displayed_name="astropy__astropy-7166",
        artifact_path=str(art),
        created_at=time.time(),
        finished_at=time.time(),
    ))

    app = build_admin_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/raw-rollouts/r-acq-1")
        assert resp.status_code == 200
        body = resp.text
        assert "astropy__astropy-7166" in body
        assert "swebench/sweb:latest" in body
        assert "node-A" in body
        # Artifact directory listing rendered inline.
        assert "report.json" in body


def test_rollouts_page_raw_status_filter(
    cfg: AdminServerConfig,
) -> None:
    """``?raw_status=failed`` shows only failed raw rollouts."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStateStore(cfg.state_db)
    for i, status in enumerate(["released", "released", "failed"]):
        store.record_raw_rollout(RawRolloutRecord(
            rollout_id=f"r-{i}",
            status=status,  # type: ignore[arg-type]
            image="busybox:1",
            displayed_name=f"instance-{status}-{i}",
            created_at=time.time() + i,
        ))

    app = build_admin_app(cfg)
    with TestClient(app) as client:
        # No filter → all three appear.
        body = client.get("/rollouts").text
        assert "instance-released-0" in body
        assert "instance-failed-2" in body

        # ?raw_status=failed → only the failed row.
        body = client.get("/rollouts?raw_status=failed").text
        assert "instance-failed-2" in body
        assert "instance-released-0" not in body
        assert "instance-released-1" not in body


def test_rollouts_page_status_alias_filter(
    cfg: AdminServerConfig,
) -> None:
    """``?status=failed`` is accepted as an alias for ``raw_status``.
    Without the alias, FastAPI silently dropped the unknown param and
    returned every row mislabeled as a status-filtered listing."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStateStore(cfg.state_db)
    for i, status in enumerate(["released", "released", "failed"]):
        store.record_raw_rollout(RawRolloutRecord(
            rollout_id=f"r-{i}",
            status=status,  # type: ignore[arg-type]
            image="busybox:1",
            displayed_name=f"instance-{status}-{i}",
            created_at=time.time() + i,
        ))

    app = build_admin_app(cfg)
    with TestClient(app, follow_redirects=True) as client:
        body = client.get("/rollouts?status=failed").text
        assert "instance-failed-2" in body
        assert "instance-released-0" not in body
        assert "instance-released-1" not in body

        # raw_status takes precedence when both are supplied.
        body = client.get(
            "/rollouts?raw_status=failed&status=released",
        ).text
        assert "instance-failed-2" in body
        assert "instance-released-0" not in body


def test_rollouts_root_redirects_to_raw(
    cfg: AdminServerConfig,
) -> None:
    """Bare /rollouts redirects to /rollouts/raw (the case-2/3
    default landing under the slim pivot). Query string is
    preserved so ``/rollouts?raw_status=failed`` lands cleanly."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    SqliteStateStore(cfg.state_db)
    app = build_admin_app(cfg)
    with TestClient(app, follow_redirects=False) as client:
        resp = client.get("/rollouts")
        assert resp.status_code == 302
        assert resp.headers["location"] == "/rollouts/raw"
        # Query string preserved.
        resp = client.get("/rollouts?raw_status=failed&since=5m")
        assert resp.status_code == 302
        assert resp.headers["location"] == "/rollouts/raw?raw_status=failed&since=5m"


def test_rollouts_template_route_serves_case_1(
    cfg: AdminServerConfig,
) -> None:
    """/rollouts/template still serves the case-1 list (renamed
    from the bare /rollouts handler). The dropdown nav points
    here; bookmarks should be migrated."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    SqliteStateStore(cfg.state_db)
    app = build_admin_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/rollouts/template")
        assert resp.status_code == 200
        # Case-1 status-enum dropdown is rendered (mp/finished/etc
        # specific to case-1's RolloutStatus). Empty list shows
        # "No rollouts match..." text either way.
        assert "<title>XRLEnv admin — rollouts</title>" in resp.text
        # Case-1's filter form posts to /rollouts/template.
        assert 'action="/rollouts/template"' in resp.text


def test_rollouts_raw_route_serves_raw_list(
    cfg: AdminServerConfig,
) -> None:
    """/rollouts/raw is the new default landing for case-2/3
    evaluation harness traffic."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStateStore(cfg.state_db)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-1", status="released",
        image="busybox:1", displayed_name="instance-A",
        created_at=time.time(), finished_at=time.time(),
    ))
    app = build_admin_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/rollouts/raw")
        assert resp.status_code == 200
        body = resp.text
        assert "Raw container rollouts" in body
        assert "instance-A" in body
        # Duration column rendered.
        assert "duration" in body
        # Pagination controls rendered (matching case-1 shape).
        assert "go to page" in body


def test_rollouts_raw_since_filter(
    cfg: AdminServerConfig,
) -> None:
    """``?since=5m`` filters out rows older than 5 minutes."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStateStore(cfg.state_db)
    now = time.time()
    # Old row (10 min ago) + recent row (1 min ago).
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-old", status="released",
        image="busybox:1", displayed_name="instance-OLD",
        created_at=now - 600,
    ))
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-new", status="released",
        image="busybox:1", displayed_name="instance-NEW",
        created_at=now - 60,
    ))

    app = build_admin_app(cfg)
    with TestClient(app) as client:
        body = client.get("/rollouts/raw?since=5m").text
        assert "instance-NEW" in body
        assert "instance-OLD" not in body
        # No filter → both visible.
        body = client.get("/rollouts/raw").text
        assert "instance-NEW" in body
        assert "instance-OLD" in body


def test_rollouts_raw_filters_by_task_key(
    cfg: AdminServerConfig,
) -> None:
    """``?task_key=foo`` shows only rollouts with that scheduler
    anti-affinity key. Backs the "show me all rollouts of this
    instance" workflow (e.g. the same SWE-bench instance attempted
    in multiple runs)."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStateStore(cfg.state_db)
    now = time.time()
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-A", status="released", image="busybox:1",
        displayed_name="instance-A-attempt-1",
        task_key="astropy__astropy-7166",
        group_id="run-1",
        created_at=now,
    ))
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-B", status="released", image="busybox:1",
        displayed_name="instance-B",
        task_key="django__django-11099",
        group_id="run-1",
        created_at=now,
    ))

    app = build_admin_app(cfg)
    with TestClient(app) as client:
        body = client.get(
            "/rollouts/raw?task_key=astropy__astropy-7166",
        ).text
        assert "instance-A-attempt-1" in body
        assert "instance-B" not in body
        # Empty query value falls back to no filter (form-submit
        # idiom when an input is left blank).
        body = client.get("/rollouts/raw?task_key=").text
        assert "instance-A-attempt-1" in body
        assert "instance-B" in body


def test_rollouts_raw_filters_by_group_id(
    cfg: AdminServerConfig,
) -> None:
    """``?group_id=run-X`` shows just the rollouts of one
    harness run. The headline workflow from xrlenv's audit M1."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStateStore(cfg.state_db)
    now = time.time()
    for tag, gid in [("a", "run-X"), ("b", "run-X"), ("c", "run-Y")]:
        store.record_raw_rollout(RawRolloutRecord(
            rollout_id=f"r-{tag}", status="released", image="busybox:1",
            displayed_name=f"instance-{tag}",
            group_id=gid,
            created_at=now,
        ))

    app = build_admin_app(cfg)
    with TestClient(app) as client:
        body = client.get("/rollouts/raw?group_id=run-X").text
        assert "instance-a" in body
        assert "instance-b" in body
        assert "instance-c" not in body


def test_raw_rollout_detail_renders_task_key_and_group_id(
    cfg: AdminServerConfig,
) -> None:
    """The detail page exposes both keys as links to the filtered
    list view so operators can pivot from one rollout to its
    siblings (same task or same run) in one click."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStateStore(cfg.state_db)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-1", status="running", image="busybox:1",
        displayed_name="astropy__astropy-7166",
        task_key="astropy__astropy-7166",
        group_id="run-2026-05-12",
        created_at=time.time(),
    ))

    app = build_admin_app(cfg)
    with TestClient(app) as client:
        body = client.get("/raw-rollouts/r-1").text
        assert "task_key" in body
        assert "group_id" in body
        # Both render as links so the operator can pivot.
        assert "/rollouts/raw?task_key=astropy__astropy-7166" in body
        assert "/rollouts/raw?group_id=run-2026-05-12" in body


def test_raw_rollout_detail_pivot_links_url_encoded(
    cfg: AdminServerConfig,
) -> None:
    """Pivot link query values must be URL-encoded so task_key /
    group_id values containing query-meaningful bytes (&, ?, =, /,
    spaces) round-trip exactly. Without encoding the browser would
    split or re-parse the URL, sending a different filter than the
    operator intended."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStateStore(cfg.state_db)
    # A task_key + group_id loaded with characters that have
    # special meaning in the query string: ampersand, slash,
    # equals, space, question mark.
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-special", status="running", image="busybox:1",
        displayed_name="weird-task",
        task_key="bench/with space&extra=stuff",
        group_id="run?id=42 & cohort",
        created_at=time.time(),
    ))

    app = build_admin_app(cfg)
    with TestClient(app) as client:
        body = client.get("/raw-rollouts/r-special").text
        # Anchor text displays the raw value (HTML-escaped by Jinja
        # autoescape, but spaces/slash preserved).
        assert "bench/with space&amp;extra=stuff" in body
        # Anchor href encodes special characters per RFC 3986. We
        # don't assert the exact octets (Jinja's urlencode may use
        # %20 or +); we assert the encoded form is present and the
        # raw form is NOT, which would indicate broken pivoting.
        assert "task_key=bench/with space&extra=stuff" not in body
        assert "group_id=run?id=42 & cohort" not in body
        # Spaces become either + or %20; ampersand must be %26 so
        # the browser doesn't split on it.
        assert "%26" in body  # encoded & somewhere in pivot links
        # And the encoded link is parseable: follow it and confirm
        # the route round-trips back to the same record.
        # Extract the href via a minimal manual parse.
        import re
        match = re.search(
            r'href="(/rollouts/raw\?task_key=[^"]+)"', body,
        )
        assert match is not None, "expected task_key pivot link"
        pivot_url = match.group(1)
        resp = client.get(pivot_url)
        assert resp.status_code == 200
        # The page should list the source rollout (the filter
        # round-tripped successfully).
        assert "weird-task" in resp.text


def test_rollouts_raw_pagination(
    cfg: AdminServerConfig,
) -> None:
    """page=2 with page_size=2 returns the next slice of rows."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStateStore(cfg.state_db)
    now = time.time()
    for i in range(5):
        store.record_raw_rollout(RawRolloutRecord(
            rollout_id=f"r-{i}", status="released",
            image="busybox:1", displayed_name=f"instance-{i}",
            created_at=now + i,  # i=4 newest
        ))

    app = build_admin_app(cfg)
    with TestClient(app) as client:
        # page=1, page_size=2 → newest two (4, 3).
        body = client.get("/rollouts/raw?page=1&page_size=64").text
        # All 5 fit on one page at default size.
        for i in range(5):
            assert f"instance-{i}" in body


def test_artifact_preview_text_file(
    cfg: AdminServerConfig, tmp_path: Path,
) -> None:
    """Text files preview as text/plain inline."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    art = tmp_path / "logs"
    art.mkdir()
    (art / "report.json").write_text('{"resolved": true}\n')
    store = SqliteStateStore(cfg.state_db)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-1", status="released",
        image="busybox:1", artifact_path=str(art),
        created_at=time.time(),
    ))
    app = build_admin_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/raw-rollouts/r-1/artifact/report.json")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        assert resp.text == '{"resolved": true}\n'


def test_artifact_preview_binary_file(
    cfg: AdminServerConfig, tmp_path: Path,
) -> None:
    """Binary files serve as octet-stream download."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    art = tmp_path / "logs"
    art.mkdir()
    (art / "blob.bin").write_bytes(b"\x00\x01\x02\xff\xfe")
    store = SqliteStateStore(cfg.state_db)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-1", status="released",
        image="busybox:1", artifact_path=str(art),
        created_at=time.time(),
    ))
    app = build_admin_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/raw-rollouts/r-1/artifact/blob.bin")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"
        assert resp.content == b"\x00\x01\x02\xff\xfe"


def test_artifact_preview_path_traversal_rejected(
    cfg: AdminServerConfig, tmp_path: Path,
) -> None:
    """``../../../etc/passwd`` doesn't escape the artifact_path
    sandbox."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    art = tmp_path / "logs"
    art.mkdir()
    # A file outside art that would be readable to test the guard.
    outside = tmp_path / "secret.txt"
    outside.write_text("forbidden")
    store = SqliteStateStore(cfg.state_db)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-1", status="released",
        image="busybox:1", artifact_path=str(art),
        created_at=time.time(),
    ))
    app = build_admin_app(cfg)
    with TestClient(app) as client:
        resp = client.get(
            "/raw-rollouts/r-1/artifact/../secret.txt",
        )
        assert resp.status_code == 404


def test_artifact_preview_missing_file_404s(
    cfg: AdminServerConfig, tmp_path: Path,
) -> None:
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    art = tmp_path / "logs"
    art.mkdir()
    store = SqliteStateStore(cfg.state_db)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-1", status="released",
        image="busybox:1", artifact_path=str(art),
        created_at=time.time(),
    ))
    app = build_admin_app(cfg)
    with TestClient(app) as client:
        resp = client.get(
            "/raw-rollouts/r-1/artifact/nonexistent.txt",
        )
        assert resp.status_code == 404


def test_artifact_preview_oversize_text_truncates_with_banner(
    cfg: AdminServerConfig, tmp_path: Path,
) -> None:
    """Oversize text files (>1MiB cap) inline a head excerpt with
    a clear banner explaining the truncation."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    art = tmp_path / "logs"
    art.mkdir()
    big = "line\n" * 500_000  # ~2.5 MB of ASCII
    (art / "big.log").write_text(big)
    store = SqliteStateStore(cfg.state_db)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-1", status="released",
        image="busybox:1", artifact_path=str(art),
        created_at=time.time(),
    ))
    app = build_admin_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/raw-rollouts/r-1/artifact/big.log")
        assert resp.status_code == 200
        assert "truncated to first" in resp.text


def test_rollouts_page_raw_status_invalid_value_ignored(
    cfg: AdminServerConfig,
) -> None:
    """A bookmarked URL with a stale/typo status doesn't 500 — the
    server logs + ignores the filter, returning all rows."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStateStore(cfg.state_db)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-1", status="released",
        image="busybox:1", displayed_name="instance-A",
        created_at=time.time(),
    ))

    app = build_admin_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/rollouts?raw_status=garbage")
        assert resp.status_code == 200
        assert "instance-A" in resp.text


def test_raw_rollout_detail_missing_artifact_path(
    cfg: AdminServerConfig,
) -> None:
    """When artifact_path doesn't resolve, the page shows the
    'this path is on the consumer's machine' note rather than the
    directory listing."""
    cfg.state_db.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStateStore(cfg.state_db)
    store.record_raw_rollout(RawRolloutRecord(
        rollout_id="r-acq-1",
        status="released",
        image="swebench/sweb:latest",
        artifact_path="/no/such/path/on/this/host",
        created_at=time.time(),
        finished_at=time.time(),
    ))

    app = build_admin_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/raw-rollouts/r-acq-1")
        assert resp.status_code == 200
        body = resp.text
        # The path itself shows up.
        assert "/no/such/path/on/this/host" in body
        # And the explanatory text.
        assert "consumer" in body.lower()

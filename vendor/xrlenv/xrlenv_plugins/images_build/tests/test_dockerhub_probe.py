"""Unit tests for ``xrlenv_plugins.images_build._dockerhub_probe``.

The helper has three behavioral surfaces that all generators rely on:

1. **Auth exchange.** ``$DOCKERHUB_USER`` + ``$DOCKERHUB_TOKEN`` get
   exchanged for a Hub JWT once and attached to every probe; the
   exchange runs at most once per process; missing/invalid creds
   silently fall back to unauth.
2. **Success path.** A 200 response with a non-empty ``images``
   array and an ``int`` ``size`` field returns the size and bumps
   the ``ok`` counter.
3. **Loud-on-failure.** The first failure (4xx, 5xx, network error,
   missing-field) prints a stderr warning naming the HTTP status
   and a body snippet; subsequent failures stay quiet but increment
   the ``failed`` counter so the end-of-run summary can report.

Tests stub :func:`urllib.request.urlopen` to avoid touching the live
Hub API.
"""

from __future__ import annotations

import io
import json
import urllib.error
from collections.abc import Iterable
from typing import Any

import pytest
from xrlenv_plugins.images_build import _dockerhub_probe as probe_mod
from xrlenv_plugins.images_build._dockerhub_probe import (
    announce_auth_status,
    get_probe_stats,
    print_probe_summary,
    probe_image_size,
    reset_probe_state,
)


class _StubResp:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _StubResp:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _make_urlopen(
    *, sequence: Iterable[Any],
) -> tuple[Any, list[tuple[str, dict[str, str]]]]:
    """Return a stub ``urlopen`` that yields from ``sequence`` and
    records each call's URL + headers in the returned list."""
    seq_iter = iter(sequence)
    calls: list[tuple[str, dict[str, str]]] = []

    def _urlopen(req, timeout=10.0):  # type: ignore[no-untyped-def]
        try:
            url = req.full_url
            headers = dict(req.headers)
        except AttributeError:
            url = str(req)
            headers = {}
        calls.append((url, headers))
        nxt = next(seq_iter)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    return _urlopen, calls


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with a clean JWT cache + stats."""
    monkeypatch.delenv(probe_mod.USER_ENV, raising=False)
    monkeypatch.delenv(probe_mod.TOKEN_ENV, raising=False)
    reset_probe_state()
    yield
    reset_probe_state()


def test_probe_returns_size_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = json.dumps({
        "images": [{"size": 1_500_000_000}],
    }).encode()
    urlopen, calls = _make_urlopen(sequence=[_StubResp(200, body)])
    monkeypatch.setattr(probe_mod.urllib.request, "urlopen", urlopen)

    out = probe_image_size("alexgshaw/fix-git", "20251031")
    assert out == 1_500_000_000

    stats = get_probe_stats()
    assert stats.ok == 1
    assert stats.failed == 0
    # No DOCKERHUB_USER set → unauth path, no Authorization header.
    assert len(calls) == 1
    url, headers = calls[0]
    assert "alexgshaw/fix-git" in url
    assert "Authorization" not in headers


def test_probe_uses_jwt_when_credentials_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auth exchange runs once, JWT is cached, and gets attached to
    every subsequent probe as ``Authorization: Bearer <jwt>``."""
    monkeypatch.setenv(probe_mod.USER_ENV, "alice")
    monkeypatch.setenv(probe_mod.TOKEN_ENV, "dckr_pat_xxx")
    reset_probe_state()  # pick up the new env

    login_body = json.dumps({"token": "jwt-abc-123"}).encode()
    probe_body = json.dumps({
        "images": [{"size": 800_000_000}],
    }).encode()
    urlopen, calls = _make_urlopen(sequence=[
        _StubResp(200, login_body),    # auth exchange
        _StubResp(200, probe_body),    # first probe
        _StubResp(200, probe_body),    # second probe — should reuse JWT
    ])
    monkeypatch.setattr(probe_mod.urllib.request, "urlopen", urlopen)

    out1 = probe_image_size("alexgshaw/fix-git", "20251031")
    out2 = probe_image_size("alexgshaw/build-pov-ray", "20251031")
    assert out1 == out2 == 800_000_000

    # Three urlopen calls total: login + two probes (no second login).
    assert len(calls) == 3
    login_url, _login_headers = calls[0]
    assert "users/login" in login_url

    probe1_url, probe1_headers = calls[1]
    probe2_url, probe2_headers = calls[2]
    assert "fix-git" in probe1_url
    assert "build-pov-ray" in probe2_url
    assert probe1_headers.get("Authorization") == "Bearer jwt-abc-123"
    assert probe2_headers.get("Authorization") == "Bearer jwt-abc-123"

    stats = get_probe_stats()
    assert stats.authenticated is True
    assert stats.ok == 2


def test_first_failure_emits_loud_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """First probe failure prints HTTP status + body snippet to
    stderr; subsequent failures stay quiet but still count."""
    # Two failures in a row — only the first should print a warning.
    err1 = urllib.error.HTTPError(
        url="https://hub.docker.com/...", code=429,
        msg="Too Many Requests",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(b'{"detail":"rate-limited"}'),
    )
    err2 = urllib.error.HTTPError(
        url="https://hub.docker.com/...", code=429,
        msg="Too Many Requests",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(b'{"detail":"rate-limited"}'),
    )
    urlopen, _calls = _make_urlopen(sequence=[err1, err2])
    monkeypatch.setattr(probe_mod.urllib.request, "urlopen", urlopen)

    out1 = probe_image_size("alexgshaw/fix-git", "20251031")
    out2 = probe_image_size("alexgshaw/build-pov-ray", "20251031")
    assert out1 is None and out2 is None

    captured = capsys.readouterr()
    assert "WARN: Docker Hub probe failed" in captured.err
    assert "HTTP 429" in captured.err
    assert "fix-git" in captured.err
    # The second failure must NOT add another loud warning.
    assert captured.err.count("WARN: Docker Hub probe failed") == 1

    stats = get_probe_stats()
    assert stats.failed == 2
    assert stats.first_failure_status == 429


def test_first_failure_warning_mentions_unauth_when_no_token(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Operator must see the env-var hint when the probe fails
    without a token set — that's the actionable signal."""
    err = urllib.error.HTTPError(
        url="https://hub.docker.com/...", code=429, msg="Too Many",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(b"{}"),
    )
    urlopen, _ = _make_urlopen(sequence=[err])
    monkeypatch.setattr(probe_mod.urllib.request, "urlopen", urlopen)

    probe_image_size("alexgshaw/fix-git", "20251031")
    err_text = capsys.readouterr().err
    assert "DOCKERHUB_USER" in err_text
    assert "DOCKERHUB_TOKEN" in err_text


def test_announce_auth_status_unauthenticated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    announce_auth_status()
    out = capsys.readouterr().err
    assert "unauthenticated" in out
    assert "DOCKERHUB_USER" in out
    assert "DOCKERHUB_TOKEN" in out


def test_announce_auth_status_authenticated(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(probe_mod.USER_ENV, "alice")
    monkeypatch.setenv(probe_mod.TOKEN_ENV, "dckr_pat_xxx")
    reset_probe_state()

    login_body = json.dumps({"token": "jwt"}).encode()
    urlopen, _ = _make_urlopen(sequence=[_StubResp(200, login_body)])
    monkeypatch.setattr(probe_mod.urllib.request, "urlopen", urlopen)

    announce_auth_status()
    out = capsys.readouterr().err
    assert "authenticated" in out
    assert "alice" in out


def test_summary_reports_fallback_count(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    err = urllib.error.HTTPError(
        url="https://hub.docker.com/...", code=429, msg="Too Many",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(b"{}"),
    )
    body = json.dumps({"images": [{"size": 1_000_000_000}]}).encode()
    urlopen, _ = _make_urlopen(sequence=[
        _StubResp(200, body),  # one success
        err, err, err,         # three failures
    ])
    monkeypatch.setattr(probe_mod.urllib.request, "urlopen", urlopen)

    for ref in ["a/x", "a/y", "a/z", "a/w"]:
        probe_image_size(ref, "tag")

    capsys.readouterr()  # discard the first-failure warning
    print_probe_summary(default_size_hint_bytes=2_500_000_000)
    summary = capsys.readouterr().err

    assert "1/4 succeeded" in summary
    assert "3 fell back" in summary
    # 3 * 2.5 GiB = 7.5 GiB of over-reservation.
    assert "7 GiB" in summary or "8 GiB" in summary
    # And the actionable hint:
    assert "DOCKERHUB_USER" in summary


def test_summary_quiet_when_all_succeeded(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    body = json.dumps({"images": [{"size": 1_000_000_000}]}).encode()
    urlopen, _ = _make_urlopen(sequence=[
        _StubResp(200, body), _StubResp(200, body),
    ])
    monkeypatch.setattr(probe_mod.urllib.request, "urlopen", urlopen)

    probe_image_size("a/x", "t")
    probe_image_size("a/y", "t")
    capsys.readouterr()  # discard any prior output

    print_probe_summary(default_size_hint_bytes=2_500_000_000)
    summary = capsys.readouterr().err
    assert "2/2 succeeded" in summary
    assert "fell back" not in summary


def test_missing_images_field_counts_as_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = json.dumps({"images": []}).encode()
    urlopen, _ = _make_urlopen(sequence=[_StubResp(200, body)])
    monkeypatch.setattr(probe_mod.urllib.request, "urlopen", urlopen)

    out = probe_image_size("a/x", "t")
    assert out is None
    stats = get_probe_stats()
    assert stats.failed == 1
    assert stats.ok == 0


def test_login_failure_falls_back_to_unauth(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bad credentials → login returns non-200 → probes continue
    unauth (and the operator sees a clear warning)."""
    monkeypatch.setenv(probe_mod.USER_ENV, "alice")
    monkeypatch.setenv(probe_mod.TOKEN_ENV, "wrong-token")
    reset_probe_state()

    body = json.dumps({"images": [{"size": 1_000_000_000}]}).encode()
    urlopen, calls = _make_urlopen(sequence=[
        _StubResp(401, b'{"detail":"invalid credentials"}'),  # login
        _StubResp(200, body),                                  # probe
    ])
    monkeypatch.setattr(probe_mod.urllib.request, "urlopen", urlopen)

    out = probe_image_size("a/x", "t")
    assert out == 1_000_000_000

    err_text = capsys.readouterr().err
    assert "auth exchange returned HTTP 401" in err_text

    # The probe call must NOT carry an Authorization header.
    _probe_url, probe_headers = calls[1]
    assert "Authorization" not in probe_headers

    stats = get_probe_stats()
    assert stats.authenticated is False
    assert stats.ok == 1


def test_concurrent_probes_thread_safe_stats(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When a driver runs probes from a ThreadPoolExecutor, stat
    counters must not lose updates under contention, and only ONE
    of the concurrent first-failure threads is allowed to print the
    loud WARN banner."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    n_calls = 200
    n_workers = 16

    ok_body = json.dumps({"images": [{"size": 1_000_000}]}).encode()
    fail_status = 429

    # Half succeed, half fail — interleaved by call_id so threads
    # genuinely race on both branches of the stats path.
    counter_lock = threading.Lock()
    counter = {"i": 0}

    def _urlopen(req, timeout=10.0):  # type: ignore[no-untyped-def]
        with counter_lock:
            i = counter["i"]
            counter["i"] += 1
        if i % 2 == 0:
            return _StubResp(200, ok_body)
        raise urllib.error.HTTPError(
            url="https://hub.docker.com/...", code=fail_status,
            msg="Too Many",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"detail":"rate-limited"}'),
        )

    monkeypatch.setattr(probe_mod.urllib.request, "urlopen", _urlopen)

    refs = [f"alexgshaw/task-{i:03d}" for i in range(n_calls)]
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(lambda r: probe_image_size(r, "t"), refs))

    # Every successful call returned the size; every failing call
    # returned None. Together they account for all n_calls.
    successes = sum(1 for r in results if r == 1_000_000)
    failures = sum(1 for r in results if r is None)
    assert successes + failures == n_calls

    # Stat counters must match the actual outcomes — no lost updates.
    stats = get_probe_stats()
    assert stats.ok == successes
    assert stats.failed == failures

    # Only one thread should have emitted the loud first-failure
    # banner; the others must have seen ``_FIRST_FAILURE_REPORTED``
    # already true under the lock.
    err_text = capsys.readouterr().err
    assert err_text.count("WARN: Docker Hub probe failed") == 1

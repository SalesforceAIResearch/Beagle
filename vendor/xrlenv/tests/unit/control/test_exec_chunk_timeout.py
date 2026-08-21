"""Unit tests for the streaming-exec output-idle ceiling knob.

``_resolve_exec_chunk_timeout_s`` reads ``XRLENV_EXEC_CHUNK_TIMEOUT_S`` (seconds),
defaulting to an hour so the CP defers to the per-exec ``timeout_s`` instead of
aborting a legitimately silent test/compile phase at a tight idle window. It is
fail-soft: a non-numeric or non-positive value falls back to the default (a
``<= 0`` ceiling would abort every exec on the first idle tick).
"""

from __future__ import annotations

import pytest
from xrlenv.control.grpc_endpoint import (
    _DEFAULT_EXEC_CHUNK_TIMEOUT_S,
    _resolve_exec_chunk_timeout_s,
)

_ENV = "XRLENV_EXEC_CHUNK_TIMEOUT_S"


def test_default_is_one_hour_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    assert _resolve_exec_chunk_timeout_s() == 3600.0
    assert _DEFAULT_EXEC_CHUNK_TIMEOUT_S == 3600.0


def test_explicit_value_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "900")
    assert _resolve_exec_chunk_timeout_s() == 900.0
    monkeypatch.setenv(_ENV, "1234.5")
    assert _resolve_exec_chunk_timeout_s() == 1234.5


def test_empty_string_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "   ")
    assert _resolve_exec_chunk_timeout_s() == _DEFAULT_EXEC_CHUNK_TIMEOUT_S


def test_non_numeric_falls_back_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv(_ENV, "banana")
    with caplog.at_level("WARNING"):
        assert _resolve_exec_chunk_timeout_s() == _DEFAULT_EXEC_CHUNK_TIMEOUT_S
    assert any("non-numeric" in r.message for r in caplog.records)


@pytest.mark.parametrize("bad", ["0", "-1", "-30.5"])
def test_non_positive_falls_back_and_warns(
    bad: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """A <= 0 ceiling would make asyncio.wait_for time out immediately and abort
    every exec — reject it in favour of the default."""
    monkeypatch.setenv(_ENV, bad)
    with caplog.at_level("WARNING"):
        assert _resolve_exec_chunk_timeout_s() == _DEFAULT_EXEC_CHUNK_TIMEOUT_S
    assert any("must be > 0" in r.message for r in caplog.records)

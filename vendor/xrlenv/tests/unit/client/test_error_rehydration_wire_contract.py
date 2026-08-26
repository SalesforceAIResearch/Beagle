"""Regression guard for ``GrpcClientTransport._rehydrate_xrlenv_error``'s
generic fallback branch.

Provoked by the ``SessionReaped`` wire-crash (see
``tests/unit/control/test_grpc_rollout_endpoint.py::
test_session_reaped_round_trips_with_reason``): ``_KIND_TO_EXC`` maps a
server-stamped ``xrlenv-error-kind`` string to an exception class, and every
entry OTHER than the three carrier types (``RolloutFailed``,
``RolloutTruncated``, ``RolloutCancelled``, which get their own explicit
branches) is rehydrated via the generic ``cls(msg)`` fallback — a single
positional argument. ``SessionReaped`` added a REQUIRED second positional
arg (``reason``, no default) without adding itself to the special-cased
branches, so the generic fallback raised ``TypeError`` instead of
rehydrating it — silently, since the failure happens deep inside exception
construction on an error path that's rarely exercised in dev.

Two halves, deliberately mutually enforcing, because ``_SPECIAL_CASED``
below is a hand-maintained mirror of the branches in ``transport.py`` and
nothing in the language keeps the two in sync:

* everything NOT special-cased must survive the bare ``cls(msg)`` fallback;
* everything that IS special-cased must actually be rehydrated by a real
  branch — so deleting that branch fails here rather than leaving a stale
  set entry silently excusing the class from the first check.

Without the second half, removing the ``SessionReaped`` branch from
``_rehydrate_xrlenv_error`` leaves this file green (verified by mutation),
which is exactly the hole the first half was written to close.
"""

from __future__ import annotations

import pytest
from xrlenv.client.transport import (
    _KIND_META_KEY,
    _KIND_TO_EXC,
    _REASON_META_KEY,
    _rehydrate_xrlenv_error,
)
from xrlenv.errors import RolloutCancelled, RolloutFailed, RolloutTruncated, SessionReaped

# Exception classes ``_rehydrate_xrlenv_error`` reconstructs with their own
# explicit branch (extra required constructor args), rather than the
# generic ``cls(msg)`` fallback. Keep this in sync with the branches above
# the fallback in ``xrlenv/client/transport.py::_rehydrate_xrlenv_error``.
_SPECIAL_CASED = {RolloutFailed, RolloutTruncated, RolloutCancelled, SessionReaped}


def test_every_generically_rehydrated_kind_accepts_message_only() -> None:
    """Every ``_KIND_TO_EXC`` entry not explicitly special-cased must be
    constructible from just a message string — that's what the fallback
    branch (``return cls(msg)``) actually calls."""
    failures: list[str] = []
    for kind, cls in _KIND_TO_EXC.items():
        if cls in _SPECIAL_CASED:
            continue
        try:
            cls("some message")
        except TypeError as exc:
            failures.append(f"{kind} -> {cls.__name__}: {exc}")
    assert not failures, (
        "these _KIND_TO_EXC entries can't be rehydrated by the generic "
        "cls(msg) fallback (add an explicit branch in "
        "_rehydrate_xrlenv_error, or give the extra args defaults): "
        + "; ".join(failures)
    )


class _FakeRpcError:
    """Minimal stand-in for ``grpc.aio.AioRpcError``.

    ``_rehydrate_xrlenv_error`` only reads ``trailing_metadata()`` and
    ``details()``, so faking those exercises the real branch selection without
    standing up a server (the full-wire case is covered by
    ``test_grpc_rollout_endpoint.py::test_session_reaped_round_trips_with_reason``).
    """

    def __init__(self, kind: str, reason: str | None = "why it happened") -> None:
        self._md: list[tuple[str, str]] = [(_KIND_META_KEY, kind)]
        if reason is not None:
            self._md.append((_REASON_META_KEY, reason))

    def trailing_metadata(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._md)

    def details(self) -> str:
        return "server said no"


@pytest.mark.parametrize("cls", sorted(_SPECIAL_CASED, key=lambda c: c.__name__))
def test_special_cased_kinds_are_rehydrated_by_a_real_branch(cls: type) -> None:
    """Each ``_SPECIAL_CASED`` entry must be reconstructed as ITSELF.

    This is what makes the set above self-policing: an entry whose branch was
    deleted (or never written) either raises out of ``_rehydrate_xrlenv_error``
    or falls through to the generic fallback, and both fail here.
    """
    kind = cls.__name__
    assert kind in _KIND_TO_EXC, (
        f"{kind} is special-cased for rehydration but is not in _KIND_TO_EXC, "
        "so the server can never actually produce it — stale entry?"
    )
    exc = _rehydrate_xrlenv_error(_FakeRpcError(kind))  # type: ignore[arg-type]
    assert isinstance(exc, cls), (
        f"{kind} rehydrated as {type(exc).__name__}, not {kind} — its explicit "
        "branch in _rehydrate_xrlenv_error is missing or wrong"
    )


def test_session_reaped_rehydration_preserves_reason_and_defaults_it() -> None:
    reaped = _rehydrate_xrlenv_error(_FakeRpcError("SessionReaped", reason="quarantine expired"))
    assert isinstance(reaped, SessionReaped)
    assert reaped.reason == "quarantine expired"
    assert reaped.retryable is True
    # reaped_at has no metadata key of its own, so it cannot survive the wire.
    assert reaped.reaped_at is None

    # A server that stamped no reason must still rehydrate, not crash.
    bare = _rehydrate_xrlenv_error(_FakeRpcError("SessionReaped", reason=None))
    assert isinstance(bare, SessionReaped)
    assert bare.reason

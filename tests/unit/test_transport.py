"""Transport helpers — ``clone_with_retry`` backoff/retry policy.

The load-bearing behavior: many trials cloning the same private repo at once trip GitHub's per-token
auth throttle (a spurious 401 / RPC-failed mid ref-negotiation); a short randomized backoff clears it.
A *definitive* failure (repo/ref not found) must fail fast. These tests pin that policy with a fake
runtime + injected ``sleep``/``jitter`` (no real waiting)."""

from __future__ import annotations

from beagle.rollout.runtime.runtime import ExecResult
from beagle.rollout.runtime.transport import GitClone, clone_with_retry

_SPEC = GitClone(repo_url="https://github.com/o/r", ref="main", container_path="/agent")
_THROTTLE = "error: RPC failed; HTTP 401\nfatal: expected flush after ref listing"
_NOT_FOUND = "fatal: repository 'https://github.com/o/r' not found"


def _fail(stderr: str) -> ExecResult:
    return ExecResult(returncode=128, stdout="", stderr=stderr)


_OK = ExecResult(returncode=0, stdout="", stderr="")


class _FakeRuntime:
    """Returns scripted results for CLONE execs; the interleaved ``rm -rf`` cleanups always succeed.
    Records each call so tests can count clone attempts vs cleanups."""

    def __init__(self, clone_results: list[ExecResult]) -> None:
        self._clone_results = list(clone_results)
        self.calls: list[tuple[str, list[str]]] = []

    def exec(self, handle, command, *, timeout=None, env=None):
        is_cleanup = (
            len(command) >= 3 and command[0] == "sh" and command[1] == "-c"
            and command[2].startswith("rm -rf")
        )
        self.calls.append(("cleanup" if is_cleanup else "clone", command))
        if is_cleanup:
            return _OK
        return self._clone_results.pop(0)

    def _kind(self, kind: str) -> int:
        return sum(1 for c in self.calls if c[0] == kind)


def test_retries_transient_then_succeeds() -> None:
    rt = _FakeRuntime([_fail(_THROTTLE), _fail(_THROTTLE), _OK])
    slept: list[float] = []
    r = clone_with_retry(rt, "h", _SPEC, attempts=4, base_delay=2.0,
                         sleep=slept.append, jitter=lambda: 0.0)
    assert r.ok
    assert rt._kind("clone") == 3          # two failures + the winning attempt
    assert rt._kind("cleanup") == 2        # partial checkout wiped before each retry
    assert slept == [2.0, 4.0]             # exponential backoff (jitter pinned to 0)


def test_fails_fast_on_definitive_error() -> None:
    rt = _FakeRuntime([_fail(_NOT_FOUND)])
    slept: list[float] = []
    r = clone_with_retry(rt, "h", _SPEC, sleep=slept.append, jitter=lambda: 0.0)
    assert not r.ok
    assert rt._kind("clone") == 1          # no retry on a definitive failure
    assert rt._kind("cleanup") == 0
    assert slept == []


def test_exhausts_attempts_on_persistent_transient() -> None:
    rt = _FakeRuntime([_fail(_THROTTLE)] * 3)
    slept: list[float] = []
    r = clone_with_retry(rt, "h", _SPEC, attempts=3, base_delay=1.0,
                         sleep=slept.append, jitter=lambda: 0.0)
    assert not r.ok                        # gives up after `attempts`, returns last failure
    assert rt._kind("clone") == 3
    assert slept == [1.0, 2.0]             # 2 backoffs between 3 attempts


def test_no_retry_when_first_attempt_succeeds() -> None:
    rt = _FakeRuntime([_OK])
    slept: list[float] = []
    r = clone_with_retry(rt, "h", _SPEC, sleep=slept.append)
    assert r.ok
    assert rt._kind("clone") == 1 and rt._kind("cleanup") == 0
    assert slept == []


def test_jitter_is_added_to_backoff() -> None:
    rt = _FakeRuntime([_fail(_THROTTLE), _OK])
    slept: list[float] = []
    clone_with_retry(rt, "h", _SPEC, attempts=2, base_delay=2.0,
                     sleep=slept.append, jitter=lambda: 0.5)
    assert slept == [2.5]                  # base_delay*2**0 + jitter

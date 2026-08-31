"""Unit tests for the swebench-verified oracle sweep's pure logic (no cluster).

Network-free and cluster-free: ``_run_one_instance`` / ``_build_docker_client`` /
``main`` (which calls ``_cache_shard``) are not invoked. These tests cover
``_summarise``, ``_resolve_task_list``, and ``_run_with_retries``.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import xrlenv_plugins.benchmarks.swebench_verified.run_oracle_sweep as sweep
from xrlenv_plugins.benchmarks.swebench_verified.run_oracle_sweep import (
    _INFRA_RETRY_EXCEPTIONS,
    SMOKE_INSTANCES,
    _resolve_task_list,
    _run_with_retries,
    _summarise,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_result(instance_id: str, *, resolved: bool, completed: bool = True) -> dict[str, Any]:
    return {"instance_id": instance_id, "resolved": resolved, "completed": completed}


def _make_args(**kwargs: Any) -> SimpleNamespace:
    """Minimal argparse-shaped namespace matching _resolve_task_list's expectations."""
    defaults = {"all": False, "tasks": None, "smoke": False}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_instance(instance_id: str) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "patch": "diff --git a/x.py b/x.py\n--- a\n+++ b\n@@ -1 +1 @@\n+fix\n",
        "problem_statement": "Fix the bug.",
    }


# ── _summarise ────────────────────────────────────────────────────────────────


def test_summarise_all_resolved_returns_exit_code_0() -> None:
    results = {
        "a": _make_result("a", resolved=True),
        "b": _make_result("b", resolved=True),
    }
    exit_code, _summary = _summarise(results, expected=2)
    assert exit_code == 0


def test_summarise_all_resolved_summary_has_full_count() -> None:
    results = {
        "a": _make_result("a", resolved=True),
        "b": _make_result("b", resolved=True),
    }
    _exit_code, summary = _summarise(results, expected=2)
    assert summary["resolved"] == 2
    assert summary["failed"] == []


def test_summarise_some_unresolved_returns_exit_code_1() -> None:
    results = {
        "a": _make_result("a", resolved=True),
        "b": _make_result("b", resolved=False),
    }
    exit_code, _summary = _summarise(results, expected=2)
    assert exit_code == 1


def test_summarise_some_unresolved_failed_list_contains_unresolved_ids() -> None:
    results = {
        "a": _make_result("a", resolved=True),
        "b": _make_result("b", resolved=False),
        "c": _make_result("c", resolved=False),
    }
    _exit_code, summary = _summarise(results, expected=3)
    assert set(summary["failed"]) == {"b", "c"}


def test_summarise_none_resolved_returns_exit_code_1() -> None:
    results = {
        "x": _make_result("x", resolved=False),
        "y": _make_result("y", resolved=False),
    }
    exit_code, summary = _summarise(results, expected=2)
    assert exit_code == 1
    assert summary["resolved"] == 0
    assert len(summary["failed"]) == 2


def test_summarise_resolved_count_less_than_expected_exits_1() -> None:
    """Edge: all entries are resolved=True but expected > actual count."""
    results = {"a": _make_result("a", resolved=True)}
    exit_code, summary = _summarise(results, expected=2)
    assert exit_code == 1
    assert summary["resolved"] == 1


def test_summarise_summary_includes_per_instance_list() -> None:
    results = {"a": _make_result("a", resolved=True)}
    _exit_code, summary = _summarise(results, expected=1)
    assert "instances" in summary
    assert len(summary["instances"]) == 1
    assert summary["instances"][0]["instance_id"] == "a"
    assert summary["instances"][0]["resolved"] is True


def test_summarise_empty_results_exits_1_when_expected_nonzero() -> None:
    exit_code, summary = _summarise({}, expected=1)
    assert exit_code == 1
    assert summary["resolved"] == 0


def test_summarise_empty_results_exits_0_when_expected_zero() -> None:
    exit_code, _summary = _summarise({}, expected=0)
    assert exit_code == 0


# ── _resolve_task_list ────────────────────────────────────────────────────────


def _populate_shard(root: Path, instance_ids: list[str]) -> Path:
    shard = root / "swebench-verified"
    for iid in instance_ids:
        inst_dir = shard / iid
        inst_dir.mkdir(parents=True)
        (inst_dir / "instance.json").write_text('{"instance_id": "' + iid + '"}')
    return shard


def test_resolve_task_list_all_and_tasks_raises(tmp_path: Path) -> None:
    shard = _populate_shard(tmp_path, ["a"])
    with pytest.raises(SystemExit, match="mutually exclusive"):
        _resolve_task_list(_make_args(**{"all": True, "tasks": "a"}), shard)


def test_resolve_task_list_tasks_csv_parses_correctly(tmp_path: Path) -> None:
    shard = _populate_shard(tmp_path, ["a", "b"])
    result = _resolve_task_list(_make_args(tasks="a,b"), shard)
    assert result == ["a", "b"]


def test_resolve_task_list_tasks_csv_strips_whitespace(tmp_path: Path) -> None:
    shard = _populate_shard(tmp_path, [])
    result = _resolve_task_list(_make_args(tasks=" a , b "), shard)
    assert result == ["a", "b"]


def test_resolve_task_list_empty_selector_raises(tmp_path: Path) -> None:
    # audit M5: a present-but-empty selector (literal "" OR ",") must FAIL, not fall
    # through to the smoke/all default and report a false 0/0 (or wrong set).
    shard = _populate_shard(tmp_path, [])
    for empty in ("", ",", " , "):
        with pytest.raises(SystemExit, match="selected no instances"):
            _resolve_task_list(_make_args(tasks=empty), shard)


def test_resolve_task_list_rejects_path_traversal(tmp_path: Path) -> None:
    # audit Low: an id is joined onto the cache/artifact roots, so it must be a bare
    # component — ../.. / a/b / absolute must be rejected, not allowed to escape the tree.
    shard = _populate_shard(tmp_path, [])
    for bad in ("../../etc", "a/b", "..", "/abs"):
        with pytest.raises(SystemExit, match="unsafe instance id"):
            _resolve_task_list(_make_args(tasks=bad), shard)


def test_load_cached_instance_rejects_embedded_id_mismatch(tmp_path: Path) -> None:
    # audit Low: even with a safe DIR name, the embedded instance_id (which feeds upstream +
    # artifact paths) could be a traversal in a corrupt/custom cache — require agreement.
    shard = tmp_path / "swebench-verified"
    d = shard / "good-id"
    d.mkdir(parents=True)
    (d / "instance.json").write_text(json.dumps({"instance_id": "../evil", "patch": "p"}))
    with pytest.raises(SystemExit, match="cache corruption"):
        sweep._load_cached_instance(shard, "good-id")


def test_resolve_task_list_default_returns_smoke_set(tmp_path: Path) -> None:
    shard = _populate_shard(tmp_path, [])
    result = _resolve_task_list(_make_args(), shard)
    assert result == list(SMOKE_INSTANCES)


def test_resolve_task_list_all_enumerates_shard(tmp_path: Path) -> None:
    ids = ["astropy__astropy-7166", "django__django-11099", "sympy__sympy-18189"]
    shard = _populate_shard(tmp_path, ids)
    result = _resolve_task_list(_make_args(**{"all": True}), shard)
    assert sorted(result) == sorted(ids)


def test_resolve_task_list_all_empty_shard_raises(tmp_path: Path) -> None:
    shard = tmp_path / "swebench-verified"
    shard.mkdir(parents=True)
    with pytest.raises(SystemExit, match="empty"):
        _resolve_task_list(_make_args(**{"all": True}), shard)


def test_resolve_task_list_all_only_counts_dirs_with_instance_json(tmp_path: Path) -> None:
    """Stray directories without instance.json are not counted as instances."""
    shard = _populate_shard(tmp_path, ["valid-id"])
    # stray dir without anchor
    (shard / "stray-dir").mkdir()
    result = _resolve_task_list(_make_args(**{"all": True}), shard)
    assert result == ["valid-id"]


def test_resolve_task_list_smoke_flag_returns_smoke_set(tmp_path: Path) -> None:
    """--smoke is in the same argparse sel group; when parsed, all=False tasks=None."""
    shard = _populate_shard(tmp_path, [])
    # smoke=True means all=False, tasks=None → falls through to default (SMOKE_INSTANCES)
    result = _resolve_task_list(_make_args(smoke=True), shard)
    assert result == list(SMOKE_INSTANCES)


# ── _run_with_retries ─────────────────────────────────────────────────────────


def test_run_with_retries_succeeds_on_first_call(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _make_result("a", resolved=True)
    mock = MagicMock(return_value=expected)
    monkeypatch.setattr(sweep, "_run_one_instance", mock)

    result = _run_with_retries(
        _make_instance("a"), client=None, run_id="r", timeout=60,
        artifact_root=None, retries=3,
    )
    assert result == expected
    mock.assert_called_once()


def test_run_with_retries_does_not_retry_content_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """resolved=False returned from _run_one_instance is NOT an exception — returned as-is."""
    content_fail = _make_result("a", resolved=False)
    mock = MagicMock(return_value=content_fail)
    monkeypatch.setattr(sweep, "_run_one_instance", mock)

    result = _run_with_retries(
        _make_instance("a"), client=None, run_id="r", timeout=60,
        artifact_root=None, retries=5,
    )
    assert result == content_fail
    mock.assert_called_once()  # no retry


def test_session_reaped_is_infra_retried() -> None:
    """A platform teardown is infra, not a content result.

    ``SessionReaped`` means the control plane destroyed the session out from
    under a running trial — a stalled consumer past the quarantine horizon, a
    lost node, a deadline. The rollout's work never failed, so recording
    ``resolved=False`` would be a false negative in the eval, not a bad model.

    This was a real gap: the error is declared ``retryable = True``, but nothing
    reads that attribute — the gate a harness actually consults is this literal
    string set, matched on ``type(exc).__name__`` (see ``_infra_kind`` /
    ``_record_op_failure`` in ``xrlenv/compat/docker_client.py``). The flag was
    decorative until the name was added here, so this test guards the thing that
    actually decides.
    """
    from xrlenv_plugins.benchmarks.swebench_verified.run_oracle_sweep import (
        _INFRA_RETRY_EXCEPTIONS,
    )

    assert "SessionReaped" in _INFRA_RETRY_EXCEPTIONS


def test_run_with_retries_retries_capacity_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    """CapacityExhausted is in _INFRA_RETRY_EXCEPTIONS → retry is attempted."""
    assert "CapacityExhausted" in _INFRA_RETRY_EXCEPTIONS

    success = _make_result("a", resolved=True)

    class CapacityExhausted(Exception):
        pass

    call_count = 0

    def fake_run_one_instance(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise CapacityExhausted("no capacity")
        return success

    monkeypatch.setattr(sweep, "_run_one_instance", fake_run_one_instance)

    result = _run_with_retries(
        _make_instance("a"), client=None, run_id="r", timeout=60,
        artifact_root=None, retries=3,
    )
    assert result == success
    assert call_count == 2  # 1 failure + 1 success


def test_run_with_retries_exhausts_retries_and_returns_error_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After retries attempts, gives up and returns an error dict (not re-raises)."""
    assert "CapacityExhausted" in _INFRA_RETRY_EXCEPTIONS

    class CapacityExhausted(Exception):
        pass

    call_count = 0

    def always_fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        raise CapacityExhausted("no capacity")

    monkeypatch.setattr(sweep, "_run_one_instance", always_fail)

    result = _run_with_retries(
        _make_instance("a"), client=None, run_id="r", timeout=60,
        artifact_root=None, retries=2,
    )
    # call_count = 1 initial + 2 retries = 3 total
    assert call_count == 3
    assert result["resolved"] is False
    assert result["completed"] is False
    assert "CapacityExhausted" in result["error"]


def test_run_with_retries_does_not_retry_non_infra_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain ValueError (not in _INFRA_RETRY_EXCEPTIONS) is not retried."""

    call_count = 0

    def raises_value_error(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        raise ValueError("unexpected content error")

    monkeypatch.setattr(sweep, "_run_one_instance", raises_value_error)

    result = _run_with_retries(
        _make_instance("a"), client=None, run_id="r", timeout=60,
        artifact_root=None, retries=5,
    )
    assert call_count == 1  # no retry
    assert result["resolved"] is False
    assert "ValueError" in result["error"]


def _make_infra_raiser(
    exc_name: str, attempts: list[int],
) -> Any:
    """Return a fake _run_one_instance that raises exc_name on first call, succeeds after."""
    ExcClass = type(exc_name, (Exception,), {})

    def raiser(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        attempts.append(1)
        if len(attempts) == 1:
            raise ExcClass("transient")
        return _make_result("a", resolved=True)

    return raiser


@pytest.mark.parametrize("exc_name", sorted(_INFRA_RETRY_EXCEPTIONS))
def test_run_with_retries_retries_all_infra_exception_types(
    exc_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every name in _INFRA_RETRY_EXCEPTIONS triggers a retry."""
    attempts: list[int] = []
    monkeypatch.setattr(sweep, "_run_one_instance", _make_infra_raiser(exc_name, attempts))

    result = _run_with_retries(
        _make_instance("a"), client=None, run_id="r", timeout=60,
        artifact_root=None, retries=1,
    )
    assert len(attempts) == 2, f"{exc_name}: expected retry, got {len(attempts)} call(s)"
    assert result["resolved"] is True, f"{exc_name}: expected resolved after retry"


def test_run_with_retries_zero_retries_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With retries=0, even an infra exception is not retried."""

    class CapacityExhausted(Exception):
        pass

    call_count = 0

    def always_fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        raise CapacityExhausted("no capacity")

    monkeypatch.setattr(sweep, "_run_one_instance", always_fail)

    result = _run_with_retries(
        _make_instance("a"), client=None, run_id="r", timeout=60,
        artifact_root=None, retries=0,
    )
    assert call_count == 1
    assert result["resolved"] is False


# ── H1/M8: infra failures swebench swallowed/wrapped, via the structured record ───
#
# The infra-vs-content distinction now comes from a structured per-attempt record the xrlenv
# drop-in stamps at the acquire boundary — NOT from re-parsing run_instance.log (audit M8:
# logs are diagnostics, not retry-policy input). These tests exercise the sweep-side accessors
# against a stand-in ``client.api`` that mimics the drop-in's record dict. End-to-end recording
# by the real drop-in ``create_container`` lives in
# tests/unit/compat/test_compat_docker_cluster_mode.py.


class _RecordingApi:
    """Stand-in for the drop-in ``api``: a keyed infra-failure dict with pop/clear."""

    def __init__(self, seed: dict[str, str] | None = None) -> None:
        self._rec = dict(seed or {})

    def take_infra_failure(self, key: str) -> str | None:
        return self._rec.pop(key, None)

    def clear_infra_failure(self, key: str) -> None:
        self._rec.pop(key, None)


class _ClientWithApi:
    def __init__(self, api: _RecordingApi) -> None:
        self.api = api


def test_take_infra_failure_pops_recorded_kind() -> None:
    client = _ClientWithApi(_RecordingApi({"inst-a": "NodeLost"}))
    assert sweep._take_infra_failure(client, "inst-a") == "NodeLost"
    # popped: a second read returns None (evidence never lingers into a later attempt).
    assert sweep._take_infra_failure(client, "inst-a") is None


def test_take_infra_failure_none_when_not_recorded() -> None:
    client = _ClientWithApi(_RecordingApi())
    assert sweep._take_infra_failure(client, "inst-a") is None


def test_take_infra_failure_local_client_without_api_is_none() -> None:
    # --local uses a real docker client with no take_infra_failure -> gracefully None.
    class _Bare:
        pass
    assert sweep._take_infra_failure(_Bare(), "inst-a") is None


def test_clear_infra_failure_drops_stale_record() -> None:
    api = _RecordingApi({"inst-a": "CapacityExhausted"})
    client = _ClientWithApi(api)
    sweep._clear_infra_failure(client, "inst-a")
    assert sweep._take_infra_failure(client, "inst-a") is None


def test_clear_infra_failure_local_client_is_noop() -> None:
    class _Bare:
        pass
    sweep._clear_infra_failure(_Bare(), "inst-a")   # must not raise


def test_run_with_retries_retries_swallowed_infra_from_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # H1/M8: swebench swallows/wraps an infra blip + returns completed=false; _run_one_instance
    # recovers it from the drop-in's STRUCTURED record and raises _InfraFailure, which the
    # infra-only retry absorbs.
    success = _make_result("a", resolved=True)
    calls = 0

    def fake(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sweep._InfraFailure("NodeLost", "a")
        return success

    monkeypatch.setattr(sweep, "_run_one_instance", fake)
    result = _run_with_retries(
        _make_instance("a"), client=None, run_id="r", timeout=60,
        artifact_root=None, retries=3,
    )
    assert result == success
    assert calls == 2  # 1 swallowed-infra + 1 success


def test_run_with_retries_swallowed_infra_exhausts_to_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise sweep._InfraFailure("CapacityExhausted", "a")

    monkeypatch.setattr(sweep, "_run_one_instance", always)
    result = _run_with_retries(
        _make_instance("a"), client=None, run_id="r", timeout=60,
        artifact_root=None, retries=2,
    )
    assert result["resolved"] is False
    assert "CapacityExhausted" in result["error"]


def test_bridge_records_and_retries_swallowed_acquire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # audit M8 Low (INTEGRATED): drive an actual swallowed upstream failure through the REAL
    # drop-in recording -> sweep extraction -> _run_with_retries (not a stand-in accessor). A
    # flaky Client raises NodeLost at acquire on attempt 1 — which a swebench-like fake
    # run_instance SWALLOWS into completed=false — and succeeds on attempt 2; the infra-only
    # retry must recover the structured kind and resolve.
    import swebench.harness.run_evaluation as se
    from xrlenv.client.container_session import ClusterContainerSession
    from xrlenv.compat.docker_client import from_env
    from xrlenv.control.service import RawAcquireResult
    from xrlenv.errors import NodeLost

    class _Transport:
        async def destroy_container(self, **_k: Any) -> None:
            return None

    class _Flaky:
        def __init__(self) -> None:
            self.n = 0

        async def acquire_container(self, **_k: Any) -> Any:
            self.n += 1
            if self.n == 1:
                raise NodeLost("acquire lost")
            return ClusterContainerSession(
                _Transport(),  # type: ignore[arg-type]
                RawAcquireResult(rollout_id="r", container_id="c",
                                 container_name="n", node_id="node-A"),
            )

    dropin = from_env(client=_Flaky())  # type: ignore[arg-type]

    def fake_run_instance(*, pred: dict[str, Any], client: Any, **_k: Any) -> dict[str, Any]:
        try:
            client.api.create_container("img", command=["true"])
        except Exception:
            return {"instance_id": pred["instance_id"], "completed": False, "resolved": False}
        return {"instance_id": pred["instance_id"], "completed": True, "resolved": True}

    monkeypatch.setattr(se, "make_test_spec", lambda instance, **kw: object())
    monkeypatch.setattr(se, "run_instance", fake_run_instance)

    result = _run_with_retries(
        {"instance_id": "inst-x", "patch": "p"}, client=dropin, run_id="r",
        timeout=60, artifact_root=None, retries=2,
    )
    assert result.get("resolved") is True   # attempt 1 swallowed-infra recovered, attempt 2 ok


def test_bridge_records_and_retries_swallowed_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # audit M8 (INTEGRATED, streaming): swebench's /eval.sh runs through exec_start(stream=True).
    # A stream-side NodeLost swallowed into completed=false must be recovered as infra evidence
    # and DRIVE the infra retry — the old behavior was a silent one-attempt content result.
    import swebench.harness.run_evaluation as se
    from xrlenv.client.container_session import ClusterContainerSession
    from xrlenv.compat.docker_client import from_env
    from xrlenv.control.service import RawAcquireResult
    from xrlenv.errors import NodeLost

    class _StreamTransport:
        def container_exec_stream(self, **_k: Any) -> Any:
            async def _gen() -> Any:
                raise NodeLost("stream dropped mid-eval")
                yield None  # unreachable — makes this an async generator
            return _gen()

        async def destroy_container(self, **_k: Any) -> None:
            return None

    class _Client:
        async def acquire_container(self, **_k: Any) -> Any:
            return ClusterContainerSession(
                _StreamTransport(),  # type: ignore[arg-type]
                RawAcquireResult(rollout_id="r", container_id="c",
                                 container_name="n", node_id="node-A"),
            )

    dropin = from_env(client=_Client())  # type: ignore[arg-type]

    def fake_run_instance(*, pred: dict[str, Any], client: Any, **_k: Any) -> dict[str, Any]:
        client.api.create_container("img", command=["true"])          # acquire ok -> map built
        info = client.api.exec_create("c", ["/eval.sh"])
        try:
            list(client.api.exec_start(info["Id"], stream=True))      # streaming NodeLost
        except Exception:   # swebench swallows EVERYTHING into completed=false
            return {"instance_id": pred["instance_id"], "completed": False, "resolved": False}
        return {"instance_id": pred["instance_id"], "completed": True, "resolved": True}

    monkeypatch.setattr(se, "make_test_spec", lambda instance, **kw: object())
    monkeypatch.setattr(se, "run_instance", fake_run_instance)

    result = _run_with_retries(
        {"instance_id": "inst-y", "patch": "p"}, client=dropin, run_id="r",
        timeout=60, artifact_root=None, retries=2,
    )
    # stream always fails -> the infra retry FIRES (kind recognized) and exhausts to an infra
    # error, NOT a silent no-retry content result.
    assert result.get("resolved") is False
    assert "NodeLost" in (result.get("error") or "")


# ── M6: image-plan / execution identity (namespace + tag reach make_test_spec) ─


def test_parser_image_identity_defaults() -> None:
    args = sweep._build_parser().parse_args([])
    assert args.namespace == sweep.SWEBENCH_NAMESPACE
    assert args.instance_image_tag == "latest"


def test_parser_image_identity_overrides() -> None:
    args = sweep._build_parser().parse_args(
        ["--namespace", "mirror.internal/swebench", "--instance-image-tag", "v1"])
    assert args.namespace == "mirror.internal/swebench"
    assert args.instance_image_tag == "v1"


def test_run_with_retries_threads_image_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    # a mirrored plan's namespace/tag must reach make_test_spec via _run_one_instance,
    # else the sweep pulls the default public image the plan never warmed (audit M6).
    captured: dict[str, Any] = {}

    def fake(_inst: dict[str, Any], _client: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _make_result("a", resolved=True)

    monkeypatch.setattr(sweep, "_run_one_instance", fake)
    _run_with_retries(
        _make_instance("a"), client=None, run_id="r", timeout=60,
        artifact_root=None, retries=0,
        namespace="mirror.internal/swebench", instance_image_tag="v1",
    )
    assert captured["namespace"] == "mirror.internal/swebench"
    assert captured["instance_image_tag"] == "v1"


def test_load_cached_instance_rejects_symlinked_anchor(tmp_path: Path) -> None:
    # audit Low: direct --tasks entry bypasses the wrapper's completeness gate, so
    # _load_cached_instance must reject a symlinked anchor (an out-of-shard target).
    shard = tmp_path / "swebench-verified"
    inst = shard / "astropy__astropy-7166"
    inst.mkdir(parents=True)
    outside = tmp_path / "evil.json"
    outside.write_text('{"instance_id": "astropy__astropy-7166", "patch": "p", '
                       '"problem_statement": "s"}')
    (inst / "instance.json").symlink_to(outside)   # anchor is a symlink out of the shard
    with pytest.raises(SystemExit, match="symlinked cache entry"):
        sweep._load_cached_instance(shard, "astropy__astropy-7166")


def test_load_cached_instance_rejects_symlinked_dir(tmp_path: Path) -> None:
    # audit Low: a symlinked instance DIR is likewise untrusted.
    shard = tmp_path / "swebench-verified"
    shard.mkdir(parents=True)
    real = tmp_path / "outside-dir"
    real.mkdir()
    (real / "instance.json").write_text('{"instance_id": "x", "patch": "p", '
                                        '"problem_statement": "s"}')
    (shard / "astropy__astropy-7166").symlink_to(real)
    with pytest.raises(SystemExit, match="symlinked cache entry"):
        sweep._load_cached_instance(shard, "astropy__astropy-7166")


def test_present_instances_skips_temp_siblings(tmp_path: Path) -> None:
    # audit M7 gap 4: a crash can leave a dot-prefixed .iid.tmp-XXX/ dir carrying an
    # instance.json; enumeration must NOT count it as a real instance.
    shard = tmp_path / "swebench-verified"
    (shard / "astropy__astropy-7166").mkdir(parents=True)
    (shard / "astropy__astropy-7166" / "instance.json").write_text("{}")
    leftover = shard / ".astropy__astropy-7166.tmp-99999"
    leftover.mkdir()
    (leftover / "instance.json").write_text("{}")   # would be a false instance if enumerated
    assert sweep._present_instances(shard) == ["astropy__astropy-7166"]

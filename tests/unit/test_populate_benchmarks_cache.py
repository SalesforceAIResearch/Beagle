"""Offline tests for the registry-driven benchmark-cache bootstrap."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import pytest

import beagle.dotenv
from beagle.benchmarks.base import BenchmarkSpec
from beagle.benchmarks.swe_bench_verified import _SweBenchSource

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "populate_benchmarks_cache.py"
_SPEC = importlib.util.spec_from_file_location("populate_benchmarks_cache", _PATH)
cache_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cache_cli)  # type: ignore[union-attr]


def test_discovers_cache_builders_from_registered_benchmarks() -> None:
    expected = {
        "deep-swe": "xrlenv_plugins.benchmarks.deep_swe.build_cache",
        "swe-bench-verified": "xrlenv_plugins.benchmarks.swebench_verified.build_cache",
        "swe-rebench": "xrlenv_plugins.benchmarks.swe_rebench.build_cache",
        "terminal_bench_2_1": (
            "xrlenv_plugins.benchmarks.terminal_bench_2_1.build_cache"
        ),
    }
    assert expected.items() <= cache_cli.discover_builders().items()


def test_list_needs_no_destination(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cache_cli, "discover_builders", lambda: {"bench": "pkg.builder"})
    assert cache_cli.main(["--list"]) == 0
    assert capsys.readouterr().out == "bench\tpkg.builder\n"


def test_filtered_population_passes_common_builder_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builders = {"a": "pkg.a", "b": "pkg.b"}
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(cache_cli, "discover_builders", lambda: builders)
    monkeypatch.setattr(cache_cli, "_resolve_dest", lambda _explicit: tmp_path.resolve())
    monkeypatch.setattr(
        cache_cli,
        "run_builder",
        lambda module, dest, **_kwargs: calls.append((module, dest)) or 0,
    )

    assert cache_cli.main(["--dest", str(tmp_path), "--benchmark", "b"]) == 0
    assert calls == [("pkg.b", tmp_path.resolve())]


def test_run_builder_uses_standard_stage_and_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    builder = types.SimpleNamespace(main=lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(cache_cli.importlib, "import_module", lambda _name: builder)

    assert cache_cli.run_builder("pkg.builder", tmp_path) == 0
    assert calls == [["--stage", "all", "--dest", str(tmp_path)]]


def test_run_builder_retries_nested_transient_http_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect_error = type("ConnectError", (Exception,), {"__module__": "httpcore"})
    calls = 0

    def builder_main(_argv: list[str]) -> int:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ExceptionGroup("task downloads", [connect_error()])
        return 0

    monkeypatch.setattr(
        cache_cli.importlib, "import_module", lambda _name: types.SimpleNamespace(main=builder_main)
    )
    sleeps: list[int] = []
    monkeypatch.setattr(cache_cli.time, "sleep", sleeps.append)

    assert cache_cli.run_builder("pkg.builder", tmp_path, retries=3) == 0
    assert calls == 3
    assert sleeps == [1, 2]


def test_run_builder_does_not_retry_non_http_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def builder_main(_argv: list[str]) -> int:
        nonlocal calls
        calls += 1
        raise ValueError("bad corpus")

    monkeypatch.setattr(
        cache_cli.importlib, "import_module", lambda _name: types.SimpleNamespace(main=builder_main)
    )
    with pytest.raises(ValueError, match="bad corpus"):
        cache_cli.run_builder("pkg.builder", tmp_path)
    assert calls == 1


def test_http_502_is_classified_as_transient() -> None:
    http_status_error = type("HTTPStatusError", (Exception,), {"__module__": "httpx"})
    error = http_status_error("502 Bad Gateway")
    error.response = types.SimpleNamespace(status_code=502)

    assert cache_cli._is_transient_http_failure(ExceptionGroup("downloads", [error]))


def test_http_request_logs_are_hidden_unless_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    monkeypatch.setattr(httpx_logger, "level", logging.INFO)
    monkeypatch.setattr(httpcore_logger, "level", logging.INFO)

    cache_cli._configure_http_logging(False)

    assert httpx_logger.level == logging.WARNING
    assert httpcore_logger.level == logging.WARNING


def test_destination_rejects_unset_cache_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(beagle.dotenv, "load_project_dotenv", lambda: None)
    monkeypatch.delenv("XRLENV_BENCHMARK_CACHE", raising=False)

    with pytest.raises(SystemExit, match="no cache root"):
        cache_cli._resolve_dest(None)


@pytest.mark.parametrize("value", ["", " ", "\t"])
def test_destination_rejects_empty_cache_root(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(beagle.dotenv, "load_project_dotenv", lambda: None)
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", value)

    with pytest.raises(SystemExit, match="empty cache root"):
        cache_cli._resolve_dest(None)

    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", "/valid/fallback")
    with pytest.raises(SystemExit, match="empty cache root"):
        cache_cli._resolve_dest(value)


def test_unknown_or_failed_builder_stops_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cache_cli, "discover_builders", lambda: {"a": "pkg.a"})
    monkeypatch.setattr(cache_cli, "_resolve_dest", lambda _explicit: tmp_path.resolve())
    with pytest.raises(SystemExit, match="have no registered cache builder"):
        cache_cli.main(["--dest", str(tmp_path), "--benchmark", "missing"])

    monkeypatch.setattr(cache_cli, "run_builder", lambda _module, _dest, **_kwargs: 7)
    assert cache_cli.main(["--dest", str(tmp_path)]) == 7
    assert "exited with status 7" in capsys.readouterr().err


def _write_cached_row(root: Path, instance_id: str, problem: str) -> None:
    instance_dir = root / "swebench-verified" / instance_id
    instance_dir.mkdir(parents=True)
    (instance_dir / "instance.json").write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "problem_statement": problem,
                "repo": "example/repo",
                "base_commit": "abc123",
                "hints_text": "a hint",
                "patch": "diff --git a/x b/x",
            }
        ),
        encoding="utf-8",
    )


def test_swebench_source_reads_populated_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cached_row(tmp_path, "repo__repo-2", "second")
    _write_cached_row(tmp_path, "repo__repo-1", "first")
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", str(tmp_path))

    items = list(
        _SweBenchSource().tasks(
            BenchmarkSpec(
                name="swe-bench-verified",
                dataset="SWE-bench/SWE-bench_Verified",
                split="test",
                task_ids=["repo__repo-2"],
            )
        )
    )

    assert len(items) == 1
    task, context = items[0]
    assert task.task_id == "repo__repo-2"
    assert task.problem_statement == "second"
    assert context.image == "swebench/sweb.eval.x86_64.repo_1776_repo-2:latest"


def test_explicit_swebench_dataset_bypasses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", str(tmp_path / "missing"))
    calls: list[tuple[str, str]] = []
    datasets = types.ModuleType("datasets")

    def load_dataset(name: str, *, split: str) -> list[dict]:
        calls.append((name, split))
        return [{"instance_id": "repo__repo-1", "problem_statement": "from HF"}]

    datasets.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    items = list(
        _SweBenchSource().tasks(
            BenchmarkSpec(name="swe-bench-verified", dataset="custom/data", split="validation")
        )
    )

    assert calls == [("custom/data", "validation")]
    assert items[0][0].problem_statement == "from HF"


def test_swebench_cache_errors_are_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XRLENV_BENCHMARK_CACHE", str(tmp_path))
    source = _SweBenchSource()
    spec = BenchmarkSpec(name="swe-bench-verified")

    with pytest.raises(RuntimeError, match="cache not found"):
        list(source.tasks(spec))

    instance = tmp_path / "swebench-verified" / "repo__repo-1"
    instance.mkdir(parents=True)
    (instance / "instance.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid SWE-bench cache record"):
        list(source.tasks(spec))

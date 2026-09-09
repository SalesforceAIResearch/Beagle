#!/usr/bin/env python3
"""Populate every registered benchmark task cache through its xrlenv builder.

The benchmark registry is the source of truth: cache-capable registrations declare
their builder module on ``Benchmark.cache_builder_module``. Adding a benchmark therefore
does not require editing this script.

Examples::

    python scripts/populate_benchmarks_cache.py --dest /shared/xrlenv_benchmark_cache
    python scripts/populate_benchmarks_cache.py --benchmark terminal_bench_2_1
    python scripts/populate_benchmarks_cache.py --list
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def discover_builders() -> dict[str, str]:
    """Return registered benchmark name -> cache-builder import path."""
    from beagle import benchmarks

    discovered: dict[str, str] = {}
    for name in benchmarks.available():
        module = benchmarks.get(name).cache_builder_module
        if module:
            discovered[name] = module
    return dict(sorted(discovered.items()))


def _resolve_selection(
    requested: Sequence[str] | None, builders: dict[str, str]
) -> list[tuple[str, str]]:
    if not requested:
        return list(builders.items())

    unknown = sorted(set(requested) - builders.keys())
    if unknown:
        available = ", ".join(builders) or "<none>"
        raise SystemExit(
            f"benchmark(s) have no registered cache builder: {', '.join(unknown)}. "
            f"Cache-capable benchmarks: {available}"
        )
    requested_set = set(requested)
    return [(name, module) for name, module in builders.items() if name in requested_set]


def _exception_leaves(exc: BaseException) -> Iterator[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        for nested in exc.exceptions:
            yield from _exception_leaves(nested)
    else:
        yield exc


def _is_transient_http_failure(exc: BaseException) -> bool:
    """Whether every leaf is a retryable httpx/httpcore transport or HTTP-status error."""
    leaves = list(_exception_leaves(exc))
    if not leaves:
        return False
    for leaf in leaves:
        module = type(leaf).__module__.split(".", 1)[0]
        status = getattr(getattr(leaf, "response", None), "status_code", None)
        retryable_status = status == 429 or (isinstance(status, int) and status >= 500)
        retryable_transport = module in {"httpx", "httpcore"} and type(leaf).__name__ in {
            "ConnectError",
            "ConnectTimeout",
            "PoolTimeout",
            "ReadError",
            "ReadTimeout",
            "RemoteProtocolError",
            "WriteError",
            "WriteTimeout",
        }
        if not retryable_status and not retryable_transport:
            return False
    return True


def _failure_summary(exc: BaseException) -> str:
    leaf = next(_exception_leaves(exc))
    status = getattr(getattr(leaf, "response", None), "status_code", None)
    detail = str(leaf).strip()
    suffix = f" (HTTP {status})" if status else (f": {detail}" if detail else "")
    return f"{type(leaf).__name__}{suffix}"


def run_builder(module_name: str, dest: Path, *, retries: int = 3) -> int:
    """Run an idempotent builder, retrying only transient HTTP failures."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"cannot import cache builder {module_name!r}: {exc}") from exc

    builder_main: Callable[[list[str]], int | None] | None = getattr(module, "main", None)
    if builder_main is None or not callable(builder_main):
        raise RuntimeError(f"cache builder {module_name!r} has no callable main(argv)")

    for attempt in range(retries + 1):
        try:
            result = builder_main(["--stage", "all", "--dest", str(dest)])
            return int(result or 0)
        except Exception as exc:
            if not _is_transient_http_failure(exc):
                raise
            if attempt == retries:
                raise RuntimeError(
                    f"transient HTTP failure persisted after {retries + 1} attempts "
                    f"({_failure_summary(exc)})"
                ) from exc
            delay = 2**attempt
            print(
                f"warning: transient HTTP failure ({_failure_summary(exc)}); "
                f"retrying in {delay}s ({attempt + 2}/{retries + 1}). "
                "Completed tasks remain cached.",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _configure_http_logging(show_http_logs: bool) -> None:
    if not show_http_logs:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Populate task caches for every cache-capable benchmark registered in Beagle."
        )
    )
    parser.add_argument(
        "--dest",
        help=(
            "Shared benchmark-cache root. Defaults to $XRLENV_BENCHMARK_CACHE; "
            "evaluation must use this same root."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        dest="benchmarks",
        metavar="NAME",
        help="Populate only this registered benchmark (repeatable). Default: all.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List cache-capable registered benchmarks and exit without writing.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries after transient HTTP failures (default: 3).",
    )
    parser.add_argument(
        "--show-http-logs",
        action="store_true",
        help="Show verbose httpx/httpcore request logs (hidden by default).",
    )
    return parser


def _resolve_dest(explicit: str | None) -> Path:
    """Load project configuration and resolve the shared cache root."""
    from xrlenv_plugins.benchmarks._benchmark_cache import benchmark_cache_root

    from beagle.dotenv import load_project_dotenv

    load_project_dotenv()
    candidate = explicit if explicit is not None else os.environ.get("XRLENV_BENCHMARK_CACHE")
    if candidate is None:
        raise SystemExit(
            "no cache root: pass --dest or set XRLENV_BENCHMARK_CACHE in .env"
        )
    if not candidate.strip():
        raise SystemExit(
            "empty cache root: --dest/XRLENV_BENCHMARK_CACHE must be a non-empty path"
        )
    return Path(benchmark_cache_root(candidate.strip())).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.retries < 0:
        raise SystemExit("--retries must be non-negative")
    builders = discover_builders()

    if args.list:
        for name, module in builders.items():
            print(f"{name}\t{module}")
        return 0

    selected = _resolve_selection(args.benchmarks, builders)
    if not selected:
        print("error: no registered benchmarks declare a cache builder.", file=sys.stderr)
        return 2

    dest = _resolve_dest(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    _configure_http_logging(args.show_http_logs)

    print(f"Benchmark cache root: {dest}", file=sys.stderr)
    for index, (name, module) in enumerate(selected, 1):
        print(f"\n[{index}/{len(selected)}] Populating {name}", file=sys.stderr)
        try:
            code = run_builder(module, dest, retries=args.retries)
        except SystemExit as exc:
            if exc.code not in (None, 0):
                print(f"error: {name} cache builder failed: {exc}", file=sys.stderr)
                return int(exc.code) if isinstance(exc.code, int) else 1
            code = 0
        except RuntimeError as exc:
            print(f"error: {name} cache builder failed: {exc}", file=sys.stderr)
            return 1
        if code:
            print(f"error: {name} cache builder exited with status {code}", file=sys.stderr)
            return code
        print(f"[{index}/{len(selected)}] Complete: {name}", file=sys.stderr)

    print(
        "\nAll selected benchmark caches are ready. Set "
        f"XRLENV_BENCHMARK_CACHE={dest} for evaluation.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

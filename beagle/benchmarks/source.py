"""Reusable task sources.

``HarborCache`` is the default source for harbor-family benchmarks: it reads the
task corpus xrlenv already materialized during onboarding, so a benchmark green in
xrlenv is runnable in beagle with no task code. Non-harbor benchmarks (SWE-bench
from HuggingFace, WAI from a vendored tree) provide their own :class:`TaskSource`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from beagle.benchmarks.base import BenchmarkSpec, TaskSource
from beagle.types import Task, TaskContext

#: The unit a source yields.
TaskItem = tuple[Task, TaskContext]


def select_and_sample(items: list[TaskItem], spec: BenchmarkSpec) -> list[TaskItem]:
    """Apply ``task_ids`` selection, ``exclude_task_ids``, and ``num_samples`` expansion.

    Shared by every source so filtering semantics are identical everywhere.
    """
    if spec.task_ids is not None:
        by_id = {t.task_id: (t, c) for t, c in items}
        try:
            selected = [by_id[i] for i in spec.task_ids]  # preserves requested order
        except KeyError as e:
            raise KeyError(f"task id {e.args[0]!r} not in benchmark {spec.name!r}") from None
    else:
        selected = list(items)

    if spec.exclude_task_ids:
        ex = set(spec.exclude_task_ids)
        selected = [(t, c) for (t, c) in selected if t.task_id not in ex]

    if spec.num_samples > 1:
        expanded: list[TaskItem] = []
        for t, c in selected:
            for s in range(spec.num_samples):
                expanded.append((replace(t, task_id=f"{t.task_id}__s{s}"), c))
        selected = expanded

    return selected


class HarborCache(TaskSource):
    """Read tasks from xrlenv's benchmark cache: ``$XRLENV_BENCHMARK_CACHE/<name>/*/task.toml``.

    Each task dir holds ``task.toml`` (``[environment].docker_image`` + workdir) and
    ``instruction.md`` (the prompt). This is exactly what xrlenv's ``build_cache``
    produced, so no per-benchmark enumeration/image logic is duplicated here. (xrlenv
    renamed ``XRLENV_HARBOR_CACHE`` → ``XRLENV_BENCHMARK_CACHE`` on 2026-07-31 and now
    hard-rejects the old name, so we read only the new one.)
    """

    def __init__(
        self,
        name: str,
        *,
        cache_name: str | None = None,
        cache_root: str | Path | None = None,
    ) -> None:
        #: Canonical registry name — stamped on ``Task.benchmark`` so the Runner can
        #: resolve the benchmark back via ``benchmarks.get(task.benchmark)``.
        self.name = name
        #: Harbor cache subdirectory (filesystem only; may differ from ``name``, e.g. the
        #: hyphenated ``terminal-bench-2-1`` dir for the ``terminal_bench_2_1`` benchmark).
        #: This must NOT leak into task identity — only into where the cache is read.
        self.cache_name = cache_name or name
        self.cache_root = cache_root

    def _root(self) -> Path:
        root = self.cache_root or os.environ.get("XRLENV_BENCHMARK_CACHE")
        if not root:
            raise RuntimeError(
                "set $XRLENV_BENCHMARK_CACHE (or pass cache_root) to read benchmark tasks"
            )
        return Path(root) / self.cache_name

    def tasks(self, spec: BenchmarkSpec) -> Iterator[TaskItem]:
        root = Path(spec.dataset) if spec.dataset else self._root()
        items: list[TaskItem] = []
        for task_toml in sorted(root.glob("*/task.toml")):
            items.append(self._read_task(task_toml.parent))
        yield from select_and_sample(items, spec)

    def _read_task(self, task_dir: Path) -> TaskItem:
        try:
            import tomllib  # stdlib on 3.11+
        except ModuleNotFoundError:  # pragma: no cover - <3.11 dev boxes
            import tomli as tomllib  # type: ignore[no-redef]

        meta = tomllib.loads((task_dir / "task.toml").read_text())
        env = meta.get("environment") or {}
        instr = task_dir / "instruction.md"
        task = Task(
            task_id=task_dir.name,
            problem_statement=instr.read_text() if instr.exists() else "",
            benchmark=self.name,
            extras={"harbor_task_dir": str(task_dir)},
        )
        ctx = TaskContext(
            image=env.get("docker_image"),
            repo_path=env.get("workdir", ""),
            benchmark_name=self.name,
        )
        return task, ctx


__all__ = ["TaskItem", "select_and_sample", "HarborCache"]

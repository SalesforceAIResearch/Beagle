"""Adaptive task subset selection for full self-evolve campaigns."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

from . import eval_runner, tree


DEFAULT_CANARY_FRACTION = 0.10


@dataclass(frozen=True)
class AdaptiveSubset:
    label: str
    tasks: list[str]
    partially_solved_tasks: list[str]
    unsolved_tasks: list[str]
    canary_tasks: list[str]


def build_from_eval_result(
    result: eval_runner.EvalResult,
    *,
    campaign: str,
    root_node_id: str,
    canary_fraction: float = DEFAULT_CANARY_FRACTION,
) -> AdaptiveSubset:
    return build_from_task_lists(
        solved_tasks=result.solved_tasks,
        unsolved_tasks=result.unsolved_tasks,
        partially_solved_tasks=result.partially_solved_tasks,
        campaign=campaign,
        root_node_id=root_node_id,
        canary_fraction=canary_fraction,
    )


def build_from_node_eval(
    node_eval: tree.NodeEval,
    *,
    campaign: str,
    root_node_id: str,
    canary_fraction: float = DEFAULT_CANARY_FRACTION,
) -> AdaptiveSubset:
    return build_from_task_lists(
        solved_tasks=node_eval.solved_tasks,
        unsolved_tasks=node_eval.unsolved_tasks,
        partially_solved_tasks=node_eval.partially_solved_tasks,
        campaign=campaign,
        root_node_id=root_node_id,
        canary_fraction=canary_fraction,
    )


def build_from_task_lists(
    *,
    solved_tasks: list[str],
    unsolved_tasks: list[str],
    partially_solved_tasks: list[str],
    campaign: str,
    root_node_id: str,
    canary_fraction: float = DEFAULT_CANARY_FRACTION,
) -> AdaptiveSubset:
    partial = list(dict.fromkeys(partially_solved_tasks))
    unsolved = [t for t in dict.fromkeys(unsolved_tasks) if t not in set(partial)]
    solved = [t for t in dict.fromkeys(solved_tasks) if t not in set(partial) | set(unsolved)]
    canary_count = _canary_count(len(solved), canary_fraction)
    canaries = _sample_canaries(
        solved,
        count=canary_count,
        seed=f"{campaign}:{root_node_id}:adaptive-canaries",
    )
    tasks = list(dict.fromkeys(partial + unsolved + canaries))
    label = f"adaptive:{len(tasks)}tasks"
    return AdaptiveSubset(
        label=label,
        tasks=tasks,
        partially_solved_tasks=partial,
        unsolved_tasks=unsolved,
        canary_tasks=canaries,
    )


def _canary_count(solved_count: int, fraction: float) -> int:
    if solved_count <= 0:
        return 0
    fraction = max(0.0, fraction)
    return min(solved_count, max(1, math.ceil(solved_count * fraction)))


def _sample_canaries(tasks: list[str], *, count: int, seed: str) -> list[str]:
    if count <= 0 or not tasks:
        return []
    rng_seed = int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16)
    rng = random.Random(rng_seed)
    indexed = list(tasks)
    rng.shuffle(indexed)
    picked = set(indexed[:count])
    # Return in root-eval order for stable prompt/report readability.
    return [t for t in tasks if t in picked]


__all__ = [
    "DEFAULT_CANARY_FRACTION",
    "AdaptiveSubset",
    "build_from_eval_result",
    "build_from_node_eval",
    "build_from_task_lists",
]

"""Task subsets for the self-evolve pipeline.

Lets users iterate fast on a fixed small set of terminal-bench tasks instead
of running the full 89-task benchmark every loop.

Public API:
    SMOKE_10                 — canonical 10-task fixed list (4 passers + 6 failers)
    parse_subset_arg(arg)    — resolve a CLI value into (label, [task_names]).
    tasks_for(label)         — accessor for the named built-in subsets.
    campaign_slug_for_subset_arg(arg)
                              — short slug for generated campaign names.

Subset labels:
    "full"      — entire dataset; tasks_for() returns []. eval_runner treats
                  empty list as "no --include-task-name filter" → full run.
    "smoke-10"  — SMOKE_10.
    "@<path>"   — read newline-separated task names (lines starting with `#`
                  ignored) from a file.
    "<csv>"     — comma-separated task names inline.
"""

from __future__ import annotations

from pathlib import Path

FULL_LABEL = "full"
SMOKE_10_LABEL = "smoke-10"
SMOKE_10_CAMPAIGN_SLUG = "s10"
FULL_CAMPAIGN_SLUG = "full"
CUSTOM_CAMPAIGN_SLUG = "custom"

# 4 currently-passing canaries + 6 currently-failing improvement targets,
# drawn from jobs/full_set_gpt55_0514_02/result.json. The failers are close
# misses with concrete verifier signals, so smoke-10 stays useful for fast
# self-evolve loops.
SMOKE_10: list[str] = [
    # Passers (regression canaries):
    "regex-log",
    "prove-plus-comm",
    "kv-store-grpc",
    "crack-7z-hash",
    # Failers (improvement targets):
    "polyglot-c-py",
    "polyglot-rust-c",
    "mteb-retrieve",
    "dna-insert",
    "db-wal-recovery",
    "pypi-server",
]


_BUILTIN: dict[str, list[str]] = {
    FULL_LABEL: [],
    SMOKE_10_LABEL: SMOKE_10,
}

_CAMPAIGN_SLUGS: dict[str, str] = {
    FULL_LABEL: FULL_CAMPAIGN_SLUG,
    SMOKE_10_LABEL: SMOKE_10_CAMPAIGN_SLUG,
}


def tasks_for(label: str) -> list[str]:
    """Return the task list for a built-in subset label.

    Raises KeyError for unknown labels. `full` returns [] (no filter).
    """
    return list(_BUILTIN[label])


def campaign_slug_for_subset_arg(arg: str | None) -> str:
    """Return the compact subset slug used in generated campaign names."""
    if arg is None:
        arg = SMOKE_10_LABEL
    return _CAMPAIGN_SLUGS.get(arg, CUSTOM_CAMPAIGN_SLUG)


def infer_builtin_label_from_campaign_name(campaign: str) -> str | None:
    """Infer a built-in subset label from a generated or legacy campaign name."""
    parts = campaign.replace("-", "_").split("_")
    normalized = "_".join(parts)
    if (
        SMOKE_10_CAMPAIGN_SLUG in parts
        or "smoke_10" in normalized
    ):
        return SMOKE_10_LABEL
    if FULL_CAMPAIGN_SLUG in parts:
        return FULL_LABEL
    return None


def parse_subset_arg(arg: str) -> tuple[str, list[str]]:
    """Resolve a CLI --task-subset value.

    Returns (label, task_names). `label` is what gets stored in nodes.subset
    so the visualizer / leaderboard can group apples-to-apples.

    >>> parse_subset_arg("full")
    ('full', [])
    >>> parse_subset_arg("smoke-10")[0]
    'smoke-10'
    >>> parse_subset_arg("a,b,c")
    ('custom:a,b,c', ['a', 'b', 'c'])
    """
    if arg in _BUILTIN:
        return arg, list(_BUILTIN[arg])

    if arg.startswith("@"):
        path = Path(arg[1:]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--task-subset file not found: {path}")
        tasks: list[str] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tasks.append(line)
        if not tasks:
            raise ValueError(f"--task-subset file {path} produced 0 tasks")
        return f"file:{path.name}", tasks

    # Inline csv form. Treat lone task-names as a 1-element subset too.
    tasks = [t.strip() for t in arg.split(",") if t.strip()]
    if not tasks:
        raise ValueError(f"could not parse --task-subset value: {arg!r}")
    label = "custom:" + ",".join(tasks) if len(tasks) <= 5 else f"custom:{len(tasks)}tasks"
    return label, tasks


__all__ = [
    "CUSTOM_CAMPAIGN_SLUG",
    "FULL_CAMPAIGN_SLUG",
    "FULL_LABEL",
    "SMOKE_10",
    "SMOKE_10_CAMPAIGN_SLUG",
    "SMOKE_10_LABEL",
    "campaign_slug_for_subset_arg",
    "infer_builtin_label_from_campaign_name",
    "tasks_for",
    "parse_subset_arg",
]

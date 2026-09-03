"""Consolidate content-retry artifact dirs into ONE authoritative job dir.

Each content-retry round runs as its own pier/harbor ``Job``, so round *N* lands
in a sibling ``<job-id>-retry<N>/`` dir next to the attempt-0 ``<job-id>/`` dir.
That fragments a task's artifacts across dirs and leaves the attempt-0 dir showing
only the FIRST-pass tally — a task that failed a transient infra blip on attempt 0
and passed on retry looks non-passing in the main dir, and the operator sees stray
``-retry1``/``-retry2`` folders instead of one result set.

:func:`consolidate_retry_dirs` folds every retry round back into the attempt-0 dir
so the main dir holds exactly ONE trial per task, then removes the now-empty
``-retry<N>`` siblings. The kept trial is the one from the LATEST round that ran the
task: a task stops being retried the moment it passes (it drops out of the round's
``remaining`` set), so its highest-numbered round is its passing trial when one
exists, and the last failure otherwise — which never discards a pass for a fail.

Paths are deterministic (the driver bakes any timestamp into ``job_id`` and pier uses
``job_name`` verbatim as ``jobs_dir/job_name``), so this needs no glob heuristics and
no pier in-memory attributes: main = ``jobs_dir/job_id``, round N =
``jobs_dir/{job_id}-retry{N}``. Task identity is read from
``result.json["config"]["task"]["path"]`` (basename) — the same UNtruncated id the
coverage gate keys on — falling back to the ``<task-id>__<suffix>`` dir name only when
that is unreadable.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def _task_of(trial_dir: Path) -> str:
    """Canonical task id for a trial dir: basename of ``config.task.path`` from
    ``result.json``, falling back to the dir name with its ``__<suffix>`` stripped.

    Mirrors ``run_benchmarks._canonical_task_id`` so consolidation groups a task's
    trials the same way the coverage gate does — the dir name alone is TRUNCATED for
    long ids and would split tasks that are really the same."""
    result_json = trial_dir / "result.json"
    try:
        j = json.loads(result_json.read_text())
        path = ((j.get("config") or {}).get("task") or {}).get("path")
        if path:
            return Path(path).name
    except (OSError, ValueError, AttributeError):
        pass
    return trial_dir.name.rsplit("__", 1)[0]


def consolidate_retry_dirs(
    jobs_dir: Path, job_id: str, content_retries: int,
) -> list[str]:
    """Fold ``<job_id>-retry<N>`` dirs under ``jobs_dir`` into ``<job_id>`` and remove
    them. Returns the retry-dir names that were folded in (for logging). A no-op — and
    silent — when no retry round ran (``content_retries == 0`` or every task passed on
    attempt 0, so no sibling exists).

    Best-effort and fail-soft: a round is skipped if its dir is absent (the sweep
    converged before that round). If the attempt-0 dir is missing, nothing is done
    (there is nowhere safe to consolidate into)."""
    jobs_dir = Path(jobs_dir)
    main = jobs_dir / job_id
    if not main.is_dir():
        return []
    folded: list[str] = []
    # Ascending round order: a later round's trial supersedes an earlier one for the
    # same task, so the LAST round that ran a task is the one left in the main dir.
    for attempt in range(1, 1 + int(content_retries)):
        retry_dir = jobs_dir / f"{job_id}-retry{attempt}"
        if not retry_dir.is_dir():
            continue
        for trial in sorted(retry_dir.iterdir()):
            if not trial.is_dir():
                continue
            task = _task_of(trial)
            # Drop the superseded trial(s) for this task already sitting in the main dir.
            for existing in sorted(main.iterdir()):
                if existing.is_dir() and _task_of(existing) == task:
                    shutil.rmtree(existing)
            dest = main / trial.name
            if dest.exists():                     # paranoia: identical suffix collision
                shutil.rmtree(dest)
            shutil.move(str(trial), str(dest))
            _repoint_trials_dir(dest, retry_dir, main)
        shutil.rmtree(retry_dir, ignore_errors=True)
        folded.append(retry_dir.name)
    if folded:
        print(
            f"consolidated {len(folded)} content-retry round(s) into {main} "
            f"(removed {', '.join(folded)})",
            file=sys.stderr,
        )
    return folded


def _repoint_trials_dir(trial: Path, retry_dir: Path, main: Path) -> int:
    """Rewrite the folded trial's recorded ``trials_dir`` from the retry round to
    the main job dir. Returns the number of files rewritten.

    A trial records the dir it ran under in ``config.json`` (and, nested, in
    ``result.json``). Moving the directory without rewriting that leaves the
    trial claiming it belongs to ``<job_id>-retryN``, which breaks **harbor's
    native resume**: ``Job._init_remaining_trial_configs`` requires every
    existing trial config to equal a planned one, and ``trials_dir`` is part of
    ``TrialConfig.__eq__``. A later resume of the consolidated job then dies with
    ``ValueError: Existing trial config does not match planned job config.``
    rather than re-running just the missing trials.

    Textual substitution of the exact retry path keeps ``result.json``'s nested
    copy in sync without having to model harbor's schema. Best-effort: an
    unreadable or absent file is skipped, since consolidation must never fail a
    sweep that has already produced its results.
    """
    old, new = str(retry_dir), str(main)
    rewritten = 0
    for name in ("config.json", "result.json"):
        path = trial / name
        try:
            text = path.read_text()
        except OSError:
            continue
        if old not in text:
            continue
        try:
            path.write_text(text.replace(old, new))
        except OSError:
            continue
        rewritten += 1
    return rewritten


def consolidate_swebench_eval_dirs(artifact_root: Path, main_run_id: str) -> list[str]:
    """Fold swebench's per-attempt ``run_evaluation`` folders into the main one.

    swebench's content/infra retries rotate the UPSTREAM run_id (upstream keys
    ``report.json`` by run_id, so a reused id no-ops), producing sibling
    ``logs/run_evaluation/<main_run_id>-retryN/`` and ``-infraN/`` folders next to the
    main ``<main_run_id>/``. ``summary.json`` is the merged, authoritative result, but
    the per-instance eval logs fragment across those folders. This moves each attempt's
    ``<model>/<instance>`` dir into the main run_id folder — later attempts win per
    instance (sorted name order ``-infraN`` < ``-retry1`` < ``-retry2``, matching
    "resolved-if-any, else last try", since a resolved instance is never re-run) — then
    removes the extra folders, so one folder holds the latest eval log per instance.
    Best-effort; returns the folded folder names."""
    eval_root = Path(artifact_root) / "logs" / "run_evaluation"
    main = eval_root / main_run_id
    if not eval_root.is_dir() or not main.is_dir():
        return []
    extras = sorted(
        d for d in eval_root.iterdir()
        if d.is_dir() and d.name.startswith(f"{main_run_id}-")
    )
    folded: list[str] = []
    for extra in extras:
        for model_dir in sorted(p for p in extra.iterdir() if p.is_dir()):
            dest_model = main / model_dir.name
            dest_model.mkdir(parents=True, exist_ok=True)
            for inst_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                dest = dest_model / inst_dir.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(inst_dir), str(dest))
        shutil.rmtree(extra, ignore_errors=True)
        folded.append(extra.name)
    if folded:
        print(
            f"consolidated {len(folded)} swebench retry eval folder(s) into {main} "
            f"(removed {', '.join(folded)})",
            file=sys.stderr,
        )
    return folded

"""Shared artifact-archive helpers for the smoke scripts.

Both ``test_swebench_drop_in.py`` and
``test_terminal_bench_2_drop_in.py`` accept ``--save-artifacts PATH``
+ ``--job-id LABEL`` so a smoke run lands its native per-task
artifacts under a stable, operator-chosen layout::

    <PATH>/<job_id>/<harness-native subtree, copied verbatim>
    <PATH>/<job_id>/summary.json

Each harness's per-task artifact tree is preserved as-is — harbor's
``trials/<task_id>__<short_id>/`` and swebench's
``logs/run_evaluation/<run_id>/<model>/<instance>/`` both have
their own conventions; we don't re-layout them so operators
navigate using whichever harness's ``--help`` / docs they already
know.

Default ``--save-artifacts`` destination is ``<repo>/tmp/`` (in
``.gitignore``). Operators override with an explicit path for
out-of-tree archives like ``~/Documents/.../monet_code_eval/jobs``.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path


def default_save_artifacts_root() -> Path:
    """``<repo>/tmp/`` — default ``--save-artifacts`` destination.

    Resolved from ``__file__`` so the path is correct whether the
    smoke is invoked from the repo root or via a sub-directory.
    The dir is created lazily by ``archive_artifacts``; operators
    don't need to ``mkdir`` ahead of time. The whole ``tmp/`` tree
    is gitignored so smoke artifacts never leak into commits.
    """
    return Path(__file__).resolve().parents[2] / "tmp"


def default_job_id() -> str:
    """UTC timestamp default when ``--job-id`` is omitted.

    Format: ``smoke-YYYYMMDD-HHMMSS`` so manual sorting under
    ``<PATH>/`` lists newest last. Lexicographically sortable.
    """
    return _dt.datetime.utcnow().strftime("smoke-%Y%m%d-%H%M%S")


def archive_artifacts(
    *,
    src_dir: Path,
    save_root: Path | None,
    job_id: str,
    summary: dict,
    subtrees: list[str],
) -> Path | None:
    """Copy harness-native artifact subtrees + summary.json to a
    persistent location under ``<save_root>/<job_id>/``.

    Args:
        src_dir: tempdir the harness ran in. Harnesses typically
            write artifacts at cwd-relative paths (harbor →
            ``./trials/...``, swebench → ``./logs/...``); we
            ``os.chdir(src_dir)`` upstream so those land here.
        save_root: ``--save-artifacts`` value, or ``None`` to skip
            archiving entirely.
        job_id: ``--job-id`` value (typically a model+version label
            like ``"claude-opus-4-7-50-v1.12.0"``). Used as the
            grouping subdir under ``save_root``.
        summary: in-memory smoke summary dict — written verbatim as
            ``summary.json`` so the operator has the per-task
            pass/fail signal alongside the harness-native logs.
        subtrees: harness-native subdirectory names to copy
            (``["trials"]`` for harbor, ``["logs"]`` for swebench).
            Missing subtrees are silently skipped — harnesses
            sometimes don't create their log dir on dry runs.

    Returns:
        The destination ``<save_root>/<job_id>/`` path on success,
        or ``None`` when archiving was disabled.
    """
    if save_root is None:
        return None
    dest = save_root / job_id
    dest.mkdir(parents=True, exist_ok=True)

    for sub in subtrees:
        src = src_dir / sub
        if not src.exists():
            continue
        # ``copytree(... dirs_exist_ok=True)`` so multiple smoke
        # runs into the same job_id append rather than fail. Useful
        # when an operator iterates: same job_id, different task
        # subset; the artifact tree grows over time.
        shutil.copytree(src, dest / sub, dirs_exist_ok=True)

    # Per-run timestamped snapshot — accumulates so re-runs against
    # the same job_id leave a chronologically-sortable trail (`ls
    # summary-*.json | sort`) rather than the previous run getting
    # clobbered. Important because the harness-native subtrees above
    # use `dirs_exist_ok=True` to merge across runs — without per-
    # run snapshots there'd be no way to disentangle which task's
    # pass/fail belonged to which invocation.
    #
    # No separate ``summary.json`` mirror — earlier iterations wrote
    # both, but for the common single-run case that produced two
    # byte-identical files. Operators wanting "latest" sort the
    # snapshots: ``ls summary-*.json | sort | tail -1``.
    #
    # Microsecond precision (``%f``) so back-to-back invocations
    # within the same second don't clobber each other — rare in
    # practice but trivial to guard against.
    run_ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    (dest / f"summary-{run_ts}.json").write_text(
        json.dumps(summary, indent=2),
    )
    return dest

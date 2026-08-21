"""terminal-bench-2 (harbor-framework) end-to-end smoke against
``XrlenvHarborEnvironment``.

Mirrors the shape of ``test_swebench_drop_in.py``: a single-host
infrastructure-readiness validator that drives ONE harbor task end-
to-end through harbor's stock ``Trial`` runner with ``import_path``
pointed at our ``XrlenvHarborEnvironment`` plug-in. Confirms the
harbor adapter is wired correctly and harbor's task-format /
grading conventions reach our subclass without translation.

How "pass" gets generated — the **oracle agent**:

The smoke configures ``harbor.TrialAgentConfig(name="oracle")``,
which loads ``harbor.agents.oracle.OracleAgent``. OracleAgent does
NOT call an LLM and does NOT generate a solution. At ``run`` time
it (a) reads ``<task_dir>/solution/`` (the canonical fix the task
author shipped — files like ``solve.sh``, plus any helper
artifacts), (b) ``upload_dir``s the whole solution dir into the
container at ``EnvironmentPaths.solution_dir``, and (c) ``exec``s
``solve.sh`` in the container so the canonical fix gets applied.
After that, harbor's verifier runs and grades the post-fix state.

So the smoke is the harbor analogue of swebench's "gold patch"
flow: replay the canonical correct solution, expect every reward
to come back > 0. A pass confirms the runtime path
(harbor → XrlenvHarborEnvironment → docker container → solve.sh
→ verifier → rewards) is wired end-to-end. Pass/fail of an
agent-generated solution is a separate concern (use a different
``TrialAgentConfig.name`` for that).

Two run shapes:

pytest single-task (default ``fix-git``)::

    .venv/bin/python -m pytest tests/smoke/test_terminal_bench_2_drop_in.py -v

script — single task, no artifact archiving (default)::

    .venv/bin/python tests/smoke/test_terminal_bench_2_drop_in.py

script — single task, archive to default ``<repo>/tmp/``
(gitignored)::

    .venv/bin/python tests/smoke/test_terminal_bench_2_drop_in.py \\
        --save-artifacts

script — broader infrastructure soak (the 8-task ``SMOKE_8``
reference set), archive to a custom out-of-repo path (substitute
``$XRLENV_SMOKE_ARCHIVE_ROOT`` with whichever durable directory
you want — typically a long-lived eval-results tree outside
this repo)::

    .venv/bin/python tests/smoke/test_terminal_bench_2_drop_in.py \\
        --task-ids \\
        fix-git,build-pov-ray,overfull-hbox,cobol-modernization,prove-plus-comm,constraints-scheduling,nginx-request-logging,dna-insert \\
        --save-artifacts "$XRLENV_SMOKE_ARCHIVE_ROOT" \\
        --job-id claude-opus-4-7-50-v1.12.0

Output layout under ``<save-artifacts>/<job-id>/`` (only when the
flag is passed)::

    summary-<utc-ts>.json                                 # per-run snapshot (sort by name for chronology; latest = `tail -1`)
    trials/<task>__<short_id>/                            # harbor's per-trial tree
        config.json   result.json   trial.log
        agent/oracle.txt                                  # solve.sh stdout/stderr
        verifier/{ctrf.json, reward.txt, test-stdout.txt}

Pre-req: harbor task suite cached at
``$HARBOR_TASKS_DIR`` (default ``~/.cache/harbor/tasks/``). The
8-task reference set (``SMOKE_8``) is the phase-0 acceptance set
defined verbatim in
``examples/benchmarks-onboarding/terminal-bench-2/smoke.py::SMOKE_TASKS``.

Excluded from the default ``pytest -q`` suite via
``--ignore=tests/smoke``. Skipped automatically when no Docker
daemon is reachable, harbor isn't installed, or the harbor task
cache is empty.

Output shape: this smoke respects harbor's native ``TrialResult``
schema. The per-task report carries the full harbor result
(``model_dump()``) verbatim; the operator interprets pass/fail per
harbor's own conventions.

Note on multi-VM: this file is a SINGLE-HOST validator. Real
multi-VM smoke (xrlenv up + cluster routing of jobs across nodes)
lands when the cluster ``ContainerControl`` ships and is its own
runbook + driver — not a per-task split shoehorned into this file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# 8-task phase-0 acceptance set, mirrored from
# ``examples/benchmarks-onboarding/terminal-bench-2/smoke.py::SMOKE_TASKS``.
# Operators can pass this whole set to ``--task-ids`` for a longer
# batch run; the default is one small task for fast infrastructure
# validation.
SMOKE_8: tuple[str, ...] = (
    "fix-git",
    "build-pov-ray",
    "overfull-hbox",
    "cobol-modernization",
    "prove-plus-comm",
    "constraints-scheduling",
    "nginx-request-logging",
    "dna-insert",
)

_DEFAULT_TASK_ID = "fix-git"  # smallest task; quick smoke


# ──────────────────────────────────────────────────────────────────────────────
# Skip gates
# ──────────────────────────────────────────────────────────────────────────────


def _docker_reachable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def _harbor_available() -> bool:
    try:
        import harbor  # noqa: F401
    except ImportError:
        return False
    return True


def _harbor_tasks_dir() -> Path:
    """Locate the harbor task cache root. Default matches harbor's
    ``CACHE_DIR / "tasks"`` (``~/.cache/harbor/tasks``) — operators
    can point elsewhere via ``$HARBOR_TASKS_DIR``."""
    explicit = os.environ.get("HARBOR_TASKS_DIR")
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".cache" / "harbor" / "tasks"


def _task_dir_for(task_id: str) -> Path | None:
    """Resolve a friendly task name (e.g. ``"fix-git"``) to its on-disk
    directory in the harbor cache.

    Cache layout: ``<harbor_tasks_dir>/<content_hash>/<task_name>/{task.toml, …}``.
    The ``<content_hash>`` dir is a nanoid-style identifier harbor
    assigns at download time; we don't know it ahead of time, so
    we glob for the inner task-name subdirectory.

    Returns the first matching ``<content_hash>/<task_name>/`` path,
    or ``None`` when the task isn't cached.
    """
    base = _harbor_tasks_dir()
    if not base.is_dir():
        return None
    # The cache also has a ``packages/`` subdir (PACKAGE_CACHE_DIR
    # for the org/name/hash layout). Scan both flat and packages
    # variants — harbor's download mode varies by ``--cache``
    # vs. the package-cache flow.
    for content_dir in base.iterdir():
        if not content_dir.is_dir():
            continue
        candidate = content_dir / task_id
        if candidate.is_dir() and (candidate / "task.toml").is_file():
            return candidate
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Smoke implementation — harbor-native, no SWE-bench shape mapping
# ──────────────────────────────────────────────────────────────────────────────


async def _run_one_trial_async(*, harbor: object, task_dir: Path) -> dict:
    """Run a single harbor task through ``XrlenvHarborEnvironment``.

    Returns harbor's ``TrialResult.model_dump()`` verbatim. The
    smoke caller is responsible for inspecting harbor's own fields
    (``verifier_result.rewards``, ``exception_info``, etc.) per
    harbor's own conventions — we don't translate to a foreign
    pass/fail shape here.

    Maps to harbor 0.5's actual API: ``await Trial.create(config) →
    Trial`` (async classmethod), ``await trial.run() → TrialResult``
    (async), ``result.verifier_result.rewards`` is the per-criterion
    scoring dict. We use ``import_path`` on
    ``TrialEnvironmentConfig`` to plug in our subclass without
    touching harbor's built-in Docker registration.
    """
    from xrlenv_plugins.harbor import XrlenvHarborEnvironment

    config = harbor.TrialConfig(  # type: ignore[attr-defined]
        task=harbor.TrialTaskConfig(path=str(task_dir)),  # type: ignore[attr-defined]
        agent=harbor.TrialAgentConfig(name="oracle"),  # type: ignore[attr-defined]
        environment=harbor.TrialEnvironmentConfig(  # type: ignore[attr-defined]
            type="docker",
            # ``import_path`` overrides which class harbor instantiates
            # for the chosen ``type``. Our subclass IS a docker env (it
            # subclasses harbor.DockerEnvironment) — same wire shape,
            # different class.
            # Harbor uses ``module.path:ClassName`` (colon separator)
            # for ``import_path`` — same convention as setuptools
            # entry-points / uvicorn factory strings, NOT a plain
            # dotted attribute path.
            import_path=(
                f"{XrlenvHarborEnvironment.__module__}:"
                f"{XrlenvHarborEnvironment.__name__}"
            ),
        ),
        verifier=harbor.TrialVerifierConfig(),  # type: ignore[attr-defined]
    )
    trial = await harbor.Trial.create(config)  # type: ignore[attr-defined]
    result = await trial.run()
    # Return harbor's native shape verbatim. Pydantic's model_dump
    # gives us a fully-serializable dict.
    return result.model_dump(mode="json")


def _run_one_trial(*, harbor: object, task_dir: Path) -> dict:
    """Sync wrapper around the async harbor Trial pipeline.

    The smoke is invoked from sync code (pytest tests + the CLI
    entry point); harbor 0.5's ``Trial.create`` / ``Trial.run`` are
    coroutines. ``asyncio.run`` bridges the two for one trial at a
    time. If we ever batch trials concurrently we'd switch to
    ``asyncio.gather`` from a single ``asyncio.run`` instead.
    """
    import asyncio
    return asyncio.run(_run_one_trial_async(harbor=harbor, task_dir=task_dir))


def _run_smoke(
    *, task_ids: list[str], save_report_to: Path | None = None,
    save_artifacts: Path | None = None,
    job_id: str | None = None,
) -> dict:
    """Run harbor on each task and aggregate the native results.

    The aggregated dict carries:

    - ``per_task[<task_id>]`` — harbor's ``TrialResult.model_dump()``
      verbatim, or an ``error`` key when the trial raised before
      harbor could produce a result.
    - ``task_ids`` — the input list, in order, for downstream
      operator scripts that want to enforce expected counts.

    The smoke deliberately does NOT compute a derived ``resolved``
    flag here — harbor's own ``verifier_result`` carries the
    per-task signal in harbor's own shape, and operators interpret
    it per harbor's conventions.

    When ``save_artifacts`` is set we ``os.chdir`` into a tempdir
    so harbor's ``./trials/<task>__<short_id>/`` directories land
    there (they're cwd-relative), then archive the whole tree to
    ``<save_artifacts>/<job_id>/`` before the tempdir reaper runs.
    """
    import os
    import tempfile

    import harbor

    # Resolve task dirs first; fail fast with a clear message.
    task_dirs: dict[str, Path] = {}
    for tid in task_ids:
        d = _task_dir_for(tid)
        if d is None:
            raise RuntimeError(
                f"task {tid!r} not cached. Searched "
                f"{_harbor_tasks_dir()}/<content_hash>/{tid}/. "
                f"Populate the harbor task cache "
                f"(``harbor task download …`` or clone the "
                f"harbor-framework/terminal-bench tasks) first.",
            )
        task_dirs[tid] = d

    # Run inside a tempdir so harbor's per-trial artifacts (which
    # land at ``./trials/<task>__<short_id>/`` cwd-relative) end
    # up somewhere we can clean up + optionally archive. Mirrors
    # the SWE-bench smoke's chdir pattern.
    with tempfile.TemporaryDirectory(prefix="xrlenv-tb2-smoke-") as td:
        tmp = Path(td)
        prev_cwd = Path.cwd()
        os.chdir(tmp)
        try:
            per_task = _execute_trials(harbor, task_dirs)
        finally:
            os.chdir(prev_cwd)

        summary = {
            "task_ids": list(task_ids),
            "per_task": per_task,
        }
        if save_report_to is not None:
            save_report_to.parent.mkdir(parents=True, exist_ok=True)
            save_report_to.write_text(json.dumps(summary, indent=2))

        # Archive harbor's native ``trials/`` tree + summary while
        # the tempdir still exists.
        if save_artifacts is not None:
            from tests.smoke._artifacts import (
                archive_artifacts,
                default_job_id,
            )
            archive_artifacts(
                src_dir=tmp,
                save_root=save_artifacts,
                job_id=job_id or default_job_id(),
                summary=summary,
                # harbor writes ``./trials/<task>__<short_id>/{config.json,
                # result.json, trial.log, agent/, verifier/}`` per task.
                subtrees=["trials"],
            )
        return summary


def _execute_trials(
    harbor: object, task_dirs: dict[str, Path],
) -> dict[str, dict]:
    """Run each task's trial, capturing harbor's native result or
    the trial-level exception. Split out from ``_run_smoke`` so the
    artifact-archiving wrapper is easy to read.

    Per-task progress prints land on stderr so the operator sees
    forward motion — harbor itself runs silent during a trial (no
    progress events on its outer surface), so without these prints
    a multi-task soak looks indistinguishable from a hang. Wall-
    clock per task is captured + reported so operators can sanity-
    check whether a "slow" task is genuinely slow or hung.
    """
    import time

    per_task: dict[str, dict] = {}
    n = len(task_dirs)
    for i, (tid, task_dir) in enumerate(task_dirs.items(), start=1):
        print(
            f"[xrlenv-smoke] [{i}/{n}] task {tid}: starting "
            f"trial (harbor runs silent — wait time depends on "
            f"image build/pull + agent runtime)",
            file=sys.stderr, flush=True,
        )
        t0 = time.monotonic()
        try:
            result = _run_one_trial(harbor=harbor, task_dir=task_dir)
            per_task[tid] = {"trial_result": result}
            elapsed = time.monotonic() - t0
            rewards = (
                result.get("verifier_result", {}) or {}
            ).get("rewards") or {}
            print(
                f"[xrlenv-smoke] [{i}/{n}] task {tid}: done in "
                f"{elapsed:.1f}s rewards={rewards}",
                file=sys.stderr, flush=True,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            per_task[tid] = {
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(
                f"[xrlenv-smoke] [{i}/{n}] task {tid}: FAILED in "
                f"{elapsed:.1f}s — {type(exc).__name__}: {exc}",
                file=sys.stderr, flush=True,
            )
    return per_task


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — harbor-native pass/fail interpretation
# ──────────────────────────────────────────────────────────────────────────────


def _trial_passed(per_task_entry: dict) -> bool | None:
    """Interpret one ``per_task[<task>]`` entry per harbor's
    conventions.

    Returns ``True`` / ``False`` / ``None``:

    - ``True`` if harbor's verifier ran AND every reward is > 0.
    - ``False`` if harbor's verifier ran AND any reward is <= 0,
      OR the trial raised an exception that harbor recorded.
    - ``None`` if we can't tell (e.g. structure unfamiliar) — caller
      treats this as inconclusive rather than collapsing into pass.
    """
    if "error" in per_task_entry:
        return False
    tr = per_task_entry.get("trial_result") or {}
    exc = tr.get("exception_info")
    if exc is not None:
        return False
    vr = tr.get("verifier_result") or {}
    rewards = vr.get("rewards")
    if not isinstance(rewards, dict) or not rewards:
        return None
    return all(isinstance(v, (int, float)) and v > 0 for v in rewards.values())


# ──────────────────────────────────────────────────────────────────────────────
# pytest entry point
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _docker_reachable(), reason="docker daemon not reachable")
@pytest.mark.skipif(not _harbor_available(), reason="`harbor` not installed")
@pytest.mark.skipif(
    _task_dir_for(_DEFAULT_TASK_ID) is None,
    reason=f"harbor task suite not populated at {_harbor_tasks_dir()}",
)
def test_terminal_bench_2_drop_in_resolves_one_task() -> None:
    """harbor's stock Trial runner resolves one terminal-bench-2
    task through ``XrlenvHarborEnvironment`` end-to-end."""
    summary = _run_smoke(task_ids=[_DEFAULT_TASK_ID])
    entry = summary["per_task"][_DEFAULT_TASK_ID]
    passed = _trial_passed(entry)
    assert passed is True, (
        f"task {_DEFAULT_TASK_ID!r} did not pass harbor's verifier; "
        f"per_task[{_DEFAULT_TASK_ID}]={entry}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Standalone-script entry point — same flow, parses argv for ad-hoc use
# ──────────────────────────────────────────────────────────────────────────────


def _main_script() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--task-id", default=None,
        help=f"Single harbor task id to smoke. Default: {_DEFAULT_TASK_ID}.",
    )
    group.add_argument(
        "--task-ids", default=None,
        help="Comma-separated list of harbor task ids. Pass the "
             "whole SMOKE_8 reference list for the broader "
             "infrastructure soak.",
    )
    parser.add_argument(
        "--save-report", type=Path, default=None,
        help="Write the harbor-native per-task result JSON to this "
             "path. Each entry is harbor's TrialResult.model_dump() "
             "verbatim — no shape translation.",
    )
    from tests.smoke._artifacts import default_save_artifacts_root
    parser.add_argument(
        "--save-artifacts", nargs="?", type=Path,
        # Same shape as test_swebench_drop_in.py:
        # ``default=None`` (omitted) → archiving OFF.
        # ``--save-artifacts`` (no value) → archive to ``const`` (the
        # default path).
        # ``--save-artifacts /custom/path`` → archive to /custom/path.
        default=None, const=default_save_artifacts_root(),
        help="Persist harbor's per-task ``trials/<task>__<short_id>/`` "
             "tree (config.json, result.json, trial.log, agent/, "
             "verifier/) under <PATH>/<job-id>/ for trajectory "
             "reference. Pass ``--save-artifacts`` (no value) to "
             f"archive under ``{default_save_artifacts_root()}`` "
             "(gitignored), or ``--save-artifacts /your/path`` to "
             "override (e.g. ``~/.../monet_code_eval/jobs``). Omit "
             "the flag entirely to skip archiving.",
    )
    parser.add_argument(
        "--job-id", default=None,
        help="Subdirectory under --save-artifacts to group this run's "
             "artifacts (e.g. ``claude-opus-4-7-50-v1.12.0``). "
             "Defaults to a UTC timestamp.",
    )
    args = parser.parse_args()

    if args.task_ids is not None:
        task_ids = [s.strip() for s in args.task_ids.split(",") if s.strip()]
    elif args.task_id is not None:
        task_ids = [args.task_id]
    else:
        task_ids = [_DEFAULT_TASK_ID]

    # Resolve job_id once so the print below + the archive layout
    # see the same timestamp.
    if args.save_artifacts is not None:
        from tests.smoke._artifacts import default_job_id
        resolved_job_id = args.job_id or default_job_id()
    else:
        resolved_job_id = args.job_id

    print(f"[xrlenv-smoke] tasks={task_ids}")
    try:
        summary = _run_smoke(
            task_ids=task_ids,
            save_report_to=args.save_report,
            save_artifacts=args.save_artifacts,
            job_id=resolved_job_id,
        )
    except Exception as exc:
        print(
            f"[xrlenv-smoke] FAIL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        import traceback
        traceback.print_exc()
        return 1

    # Per-task pass/fail line, harbor-native. The full report is
    # printed below + persisted to --save-report when requested.
    passes: list[str] = []
    fails: list[str] = []
    inconclusive: list[str] = []
    print()
    print("[xrlenv-smoke] per-task harbor verdicts:")
    for tid in summary["task_ids"]:
        entry = summary["per_task"].get(tid, {})
        passed = _trial_passed(entry)
        rewards = (
            (entry.get("trial_result") or {}).get("verifier_result") or {}
        ).get("rewards")
        err = entry.get("error")
        marker = "PASS" if passed is True else (
            "FAIL" if passed is False else "?"
        )
        detail = err if err else f"rewards={rewards}"
        print(f"  [{marker}] {tid}  {detail}")
        if passed is True:
            passes.append(tid)
        elif passed is False:
            fails.append(tid)
        else:
            inconclusive.append(tid)

    print()
    print(
        f"[xrlenv-smoke] tasks pass={len(passes)} fail={len(fails)} "
        f"inconclusive={len(inconclusive)} of {len(summary['task_ids'])}",
    )
    if args.save_report is not None:
        print(f"[xrlenv-smoke] full report saved to {args.save_report.resolve()}")
    if args.save_artifacts is not None:
        archive_dest = (args.save_artifacts / resolved_job_id).resolve()
        print(f"[xrlenv-smoke] artifacts saved under {archive_dest}/")

    if fails or inconclusive:
        if fails:
            print(f"[xrlenv-smoke] failed: {fails}", file=sys.stderr)
        if inconclusive:
            print(
                f"[xrlenv-smoke] inconclusive (no rewards from harbor): "
                f"{inconclusive}",
                file=sys.stderr,
            )
        return 1
    print(
        f"\n[xrlenv-smoke] SUCCESS: {len(passes)}/{len(summary['task_ids'])} "
        f"passed harbor verifier",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main_script())

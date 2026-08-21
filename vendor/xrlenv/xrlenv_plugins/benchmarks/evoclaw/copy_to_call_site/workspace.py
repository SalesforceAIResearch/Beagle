"""Per-user EvoClaw workspace via symlinks — no dataset copy, no dataset writes.

EvoClaw couples read + write on ``--workspace-root``: it reads the repo's data
from it (``metadata.json``, ``dependencies.csv``, ``milestones.csv``, ``srs/``,
``test_results/``, ...) **and** its parent's ``config/<repo>.yaml``
(``evaluator.py:62``), and it **writes** ``e2e_trial/`` into it. For a shared
checkout that means either copying the whole dataset per user or everyone writing
into the shared dataset.

This builds a **per-user workspace** that side-steps both: it symlinks the
read-only files from the shared ``EVOCLAW_DATA_ROOT`` into
``<workspace-base>/<repo>/`` (and ``config/<repo>.yaml`` at the parent), so trials
(``e2e_trial/``) are written **locally under the workspace base**, the shared
dataset is never written, and nothing is copied.

Usage: set ``EVOCLAW_DATA_ROOT`` (shared) and pass the writable workspace base as a
flag (``--workspace-root-base`` / ``--results-root``, default ``<checkout>/results``);
the wrapper (``run_e2e_xrlenv.py`` / ``run_all_xrlenv.py``) calls
:func:`link_workspace` directly.
"""

from __future__ import annotations

from pathlib import Path

# EvoClaw reads a repo's data from <workspace-root> and WRITES trials into it.
# We symlink *every* top-level dataset entry (metadata.json, dependencies.csv,
# milestones.csv, srs/, test_results/, dockerfiles/ — which holds each
# milestone's test_config.json, ...) EXCEPT the write targets below, so we never
# silently miss a read dependency and the run's writes still land locally under
# the workspace base. The dataset dir itself often contains an `e2e_trial/`
# from a prior run — linking it would route writes straight into the shared
# dataset, so excluding it is load-bearing, not cosmetic.
_WRITE_TARGETS = frozenset({"e2e_trial", "evaluation", "mstone_trial"})


def _link(link_path: Path, target: Path) -> None:
    """Idempotently make ``link_path`` a symlink to ``target``."""
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        raise SystemExit(
            f"{link_path} exists and is not a symlink — refusing to clobber. "
            "Use a clean workspace base (--workspace-root-base / --results-root)."
        )
    link_path.symlink_to(target)


def link_workspace(
    data_root: Path, workspace_base: Path, repo: str, single_milestone: str | None = None
) -> Path:
    """Build/refresh ``<workspace_base>/<repo>`` of symlinks into the shared
    ``<data_root>/<repo>``. Returns the per-user ``--workspace-root``.

    ``single_milestone`` scopes the run to exactly one milestone: instead of
    symlinking the dataset's ``selected_milestone_ids.txt``, write a real one-line
    file containing just that id. EvoClaw copies it to ``<trial_root>`` and
    DAGManager runs only that milestone (a lone node with no in-set deps). Pass a
    per-task ``workspace_base`` so each milestone gets its own trial_root/lock; the
    repo subdir keeps the real repo name (golden extraction derives the image from
    it), so use e.g. ``<run-ws>/<repo>__<mid>`` as ``workspace_base``.
    """
    src = (data_root / repo).expanduser()
    if not (src / "metadata.json").is_file():
        raise SystemExit(
            f"EVOCLAW_DATA_ROOT has no repo data for {repo!r} at {src} "
            "(expected metadata.json). Check EVOCLAW_DATA_ROOT / --repo-name."
        )
    workspace_base = workspace_base.expanduser()
    ws = workspace_base / repo
    ws.mkdir(parents=True, exist_ok=True)

    # Repo config is read from <workspace-root>.parent/config/<repo>.yaml.
    cfg_src = data_root / "config" / f"{repo}.yaml"
    if cfg_src.is_file():
        (workspace_base / "config").mkdir(parents=True, exist_ok=True)
        _link(workspace_base / "config" / f"{repo}.yaml", cfg_src)

    for s in sorted(src.iterdir()):
        if s.name in _WRITE_TARGETS:
            continue  # write target — keep local so trials don't hit the dataset
        if single_milestone and s.name == "selected_milestone_ids.txt":
            continue  # replaced by a real one-line file below
        _link(ws / s.name, s)

    if single_milestone:
        sel = ws / "selected_milestone_ids.txt"
        if sel.is_symlink() or sel.is_file():
            sel.unlink()
        sel.write_text(f"{single_milestone}\n")
    return ws

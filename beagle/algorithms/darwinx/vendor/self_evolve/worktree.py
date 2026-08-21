"""Git worktree + monet_code submodule branch management.

Each pipeline gets its own monet_code_eval worktree under
``WORKTREE_PARENT`` (defaults to the eval repo's parent dir), with a
fresh ``evolve/<short_sha>__<pid>`` branch in the monet_code submodule.

Submodule + worktree note: ``git worktree`` shares the parent repo's
``.git/modules/<sub>`` storage across worktrees, but each worktree's
submodule subdirectory has its own checkout. So creating a different
branch in each worktree's ``monet_code/`` is safe — no two pipelines
will ever step on each other's submodule HEAD.

Path resolution
~~~~~~~~~~~~~~~
The default ``REPO_ROOT`` is derived from this file's location
(``src/monet_eval/self_evolve/worktree.py`` → ``../../../..``), so the
package works from any clone. ``WORKTREE_PARENT`` defaults to the dir
above the eval repo. Override either with environment variables:

    MONET_EVAL_REPO_ROOT       absolute path to monet_code_eval clone
    MONET_EVAL_WORKTREE_PARENT absolute dir to put per-pipeline worktrees in
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


def _default_repo_root() -> Path:
    """Discover the monet_code_eval clone this package lives in."""
    env = os.environ.get("MONET_EVAL_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # coding-bench layout: self_evolve/worktree.py → parents[1] is the repo root.
    return Path(__file__).resolve().parents[1]


def _default_worktree_parent() -> Path:
    """Where per-pipeline worktrees are created. Defaults to repo's parent."""
    env = os.environ.get("MONET_EVAL_WORKTREE_PARENT")
    if env:
        return Path(env).expanduser().resolve()
    return _default_repo_root().parent


REPO_ROOT = _default_repo_root()
WORKTREE_PARENT = _default_worktree_parent()


@dataclass(frozen=True)
class Worktree:
    eval_dir: Path        # /fsx/.../monet_code_eval__evolve_<branch>
    monet_dir: Path       # eval_dir / monet_code
    eval_branch: str      # branch in monet_code_eval
    monet_branch: str     # branch in monet_code submodule
    parent_commit: str    # commit_sha the new monet_code branch is based on


# ─── Helpers ──────────────────────────────────────────────────────────────


def _run(
    cmd: list[str],
    *,
    cwd: Path | str | None = None,
    check: bool = True,
    capture: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess:
    """Thin git-friendly wrapper. Always uses LC_ALL=C for stable output.

    On non-zero exit with `check=True`, re-raises CalledProcessError but
    surfaces stderr in the message so callers can see git's complaint
    (e.g. "fatal: could not lock the index file" under --max-parallel-workers races).
    """
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=capture,
            check=check,
            timeout=timeout,
            env={"LC_ALL": "C", **_env_passthrough()},
        )
    except subprocess.CalledProcessError as e:
        if e.stderr:
            # Append git's stderr to the exception's str() — by default
            # CalledProcessError only shows the exit code which is useless
            # for diagnosing "fatal: …" git errors.
            e.add_note(f"git stderr: {e.stderr.strip()}")
        raise


def _env_passthrough() -> dict[str, str]:
    import os
    keep = ("PATH", "HOME", "USER", "LANG", "TERM", "SSH_AUTH_SOCK", "GIT_SSH",
            "GIT_SSH_COMMAND", "GIT_CONFIG_GLOBAL", "XDG_CACHE_HOME",
            # `GIT_CONFIG_COUNT` + paired `GIT_CONFIG_KEY_<n>` /
            # `GIT_CONFIG_VALUE_<n>` are how `scripts/self_evolve.py`
            # layers a per-process git config (HTTPS-instead-of-SSH for
            # GitHub + `gh auth git-credential` helper) on top of the
            # user's `~/.gitconfig` without mutating any files on disk.
            # If we don't forward them to the `git submodule update`
            # subprocess that runs inside the per-pipeline worktree,
            # the rewrite is invisible and workers re-introduce the SSH-
            # disconnect crash class we're trying to prevent.
            "GIT_CONFIG_COUNT")
    out = {k: os.environ[k] for k in keep if k in os.environ}
    # The `GIT_CONFIG_KEY_*` / `GIT_CONFIG_VALUE_*` indices aren't a
    # fixed allowlist — they grow with whatever the parent process
    # already had set (e.g. the user pre-exported their own additions
    # before invoking `self_evolve.py`). Forward every paired key/value
    # we find so the subprocess sees the same merged view we do.
    for k, v in os.environ.items():
        if k.startswith("GIT_CONFIG_KEY_") or k.startswith("GIT_CONFIG_VALUE_"):
            out[k] = v
    return out


def short_sha(sha: str) -> str:
    return sha[:7] if len(sha) >= 7 else sha


# ─── Branch + worktree naming ─────────────────────────────────────────────


def make_branch_name(parent_commit: str, pipeline_id: str) -> str:
    """Naming scheme: evolve/<short_sha>__<pipeline_id>.

    Used for both the monet_code branch and (sanitized) the eval-side worktree
    directory. Unique per-pipeline so the shared .git/modules/monet_code
    sees no branch collisions across parallel pipelines.
    """
    return f"evolve/{short_sha(parent_commit)}__{pipeline_id}"


def make_worktree_path(branch: str) -> Path:
    """Sanitize a branch name into a worktree directory under /fsx/.../Projects."""
    safe = branch.replace("/", "_")
    return WORKTREE_PARENT / f"monet_code_eval__{safe}"


# ─── Read current state of the canonical repo ────────────────────────────


def current_monet_branch() -> str:
    """Return the branch (or 'HEAD~detached') of the canonical monet_code submodule."""
    monet_dir = REPO_ROOT / "monet_code"
    proc = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=monet_dir, check=False)
    name = (proc.stdout or "").strip()
    return name or "HEAD"


def current_monet_commit() -> str:
    monet_dir = REPO_ROOT / "monet_code"
    proc = _run(["git", "rev-parse", "HEAD"], cwd=monet_dir)
    return proc.stdout.strip()


def monet_develop_tip(*, fetch: bool = True) -> tuple[str, str]:
    """Return ``(sha, "develop")`` for the canonical seed commit of a new
    self-evolve campaign.

    Resolves ``origin/develop`` rather than whatever the developer
    happened to have checked out in ``monet_code/`` so a fresh campaign
    is reproducible regardless of local submodule state. Best-effort
    ``git fetch origin develop`` first; if the fetch fails (offline,
    auth, etc.) we fall back to the cached ``refs/remotes/origin/develop``
    ref already present on disk and log a warning. Hard-errors only when
    ``origin/develop`` doesn't exist at all.

    The branch label is always returned as ``"develop"`` (not
    ``"origin/develop"``) so it shows cleanly in the visualizer's root
    summary and in ``works.md``.
    """
    monet_dir = REPO_ROOT / "monet_code"
    if fetch:
        # Bounded timeout — a hung git-fetch must NOT wedge the
        # supervisor's bootstrap precursor. 30s is generous for a single
        # branch ref-update against any sane network; on slow links we'd
        # rather fall back to the cached remote ref than block forever.
        try:
            _run(
                ["git", "fetch", "--quiet", "origin", "develop"],
                cwd=monet_dir, check=True, timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            # Cached origin/develop may be stale; that's still better than
            # the alternative (current HEAD = whatever the developer was
            # hacking on). Surface the warning loudly.
            import warnings
            warnings.warn(
                f"git fetch origin develop failed in {monet_dir}: {e}; "
                f"falling back to cached origin/develop ref (may be stale)",
                stacklevel=2,
            )
    proc = _run(
        ["git", "rev-parse", "origin/develop"],
        cwd=monet_dir, check=False,
    )
    sha = (proc.stdout or "").strip()
    if not sha or proc.returncode != 0:
        raise RuntimeError(
            f"could not resolve origin/develop in {monet_dir} "
            f"(returncode={proc.returncode}, stderr={proc.stderr!r}); "
            f"is the remote configured?"
        )
    return sha, "develop"


def resolve_seed_commit(
    ref: str, *, fetch_ref: str | None = None, fetch: bool = True,
) -> tuple[str, str]:
    """Resolve an explicit seed commit for the bootstrap root.

    Used when ``$SELF_EVOLVE_ROOT_COMMIT`` is set to evolve on top of a
    specific agent build (e.g. a PR-head commit or a named branch) instead
    of ``origin/develop``. This is what lets a cluster campaign seed its
    root at, say, a monet PR head while still running a fresh baseline (the
    ``--baseline-logs`` path, by contrast, reuses an existing job-run's
    score). See ``docs/self_evolve/CLUSTER_LAUNCH.md``.

    ``ref`` is the sha/branch/tag to ``git rev-parse``. ``fetch_ref`` is an
    optional branch to fetch first so a *remote-only* sha becomes present in
    the canonical clone (GitHub won't always serve a bare sha to
    ``git fetch``, but fetching the containing branch makes the sha
    reachable). Both are best-effort fetches with a bounded timeout so a
    hung network can't wedge the bootstrap.

    Returns ``(sha, label)`` mirroring :func:`monet_develop_tip`. The label
    is the human-friendly ref (for ``works.md`` / the visualizer); a bare
    sha is labelled ``"root"``.
    """
    monet_dir = REPO_ROOT / "monet_code"
    if fetch:
        for fr in (fetch_ref, ref):
            if not fr:
                continue
            try:
                _run(
                    ["git", "fetch", "--quiet", "origin", fr],
                    cwd=monet_dir, check=False, timeout=120,
                )
            except subprocess.TimeoutExpired:
                import warnings
                warnings.warn(
                    f"git fetch origin {fr} timed out in {monet_dir}; "
                    f"relying on already-present objects",
                    stacklevel=2,
                )
    sha = ""
    for candidate in (ref, f"origin/{ref}"):
        proc = _run(
            ["git", "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"],
            cwd=monet_dir, check=False,
        )
        sha = (proc.stdout or "").strip()
        if sha and proc.returncode == 0:
            break
    if not sha:
        raise RuntimeError(
            f"could not resolve SELF_EVOLVE_ROOT_COMMIT={ref!r} in {monet_dir}. "
            f"Ensure the canonical monet_code clone can reach it — e.g. "
            f"`git -C {monet_dir} fetch origin <branch-containing-the-commit>` "
            f"(set SELF_EVOLVE_ROOT_FETCH_REF to that branch), then retry."
        )
    looks_like_sha = len(ref) >= 7 and all(c in "0123456789abcdefABCDEF" for c in ref)
    label = "root" if looks_like_sha else ref
    return sha, label


# ─── Worktree creation (Phase 1) ─────────────────────────────────────────


def _worktree_lock_path(repo_root: Path) -> Path:
    """Per-repo lock file under $TMPDIR. Stable across runs but unique per clone."""
    digest = hashlib.sha1(str(repo_root.resolve()).encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"monet_self_evolve.worktree-add.{digest}.lock"


@contextmanager
def _flock(lock_path: Path, *, timeout_s: float = 600.0) -> Iterator[None]:
    """Cross-process file lock used to serialize git operations that take an
    exclusive index lock on the parent repo.

    `git worktree add` and `git submodule update --init` both take exclusive
    locks on the parent repo's `.git/index.lock` and `.git/modules/<sub>/`,
    so two concurrent invocations against the same repo race and one fails
    with `fatal: Unable to create '/path/.git/index.lock': File exists`
    (exit 255).

    The fix: take an advisory file lock (fcntl.flock) before invoking git,
    release after. Linux/macOS only; on Windows this would need msvcrt.locking.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EWOULDBLOCK, errno.EACCES):
                    raise
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"could not acquire {lock_path} within {timeout_s}s"
                    )
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _worktree_add_retries() -> int:
    """How many times to (re)try a transient-failing ``git worktree add``.
    Override with ``MONET_WORKTREE_ADD_RETRIES`` (default 4)."""
    import os
    try:
        return max(1, int(os.environ.get("MONET_WORKTREE_ADD_RETRIES", "4")))
    except (TypeError, ValueError):
        return 4


def _cleanup_partial_worktree(
    canonical: Path, wt_dir: Path, monet_dir: Path, branch: str,
) -> None:
    """Best-effort teardown of a half-created worktree so a retry under the
    SAME branch name starts clean: drop the (possibly partial) worktree
    registration, prune stale admin entries, delete the branch git auto-created
    via ``-b``, then remove the eval dir. Every step is check=False — this runs
    on an already-failing path and must never raise."""
    for cmd in (
        ["git", "worktree", "remove", "--force", str(monet_dir)],
        ["git", "worktree", "prune"],
        ["git", "branch", "-D", branch],
    ):
        try:
            _run(cmd, cwd=canonical, check=False, timeout=120)
        except Exception:  # noqa: BLE001 — teardown must not mask the real error
            pass
    shutil.rmtree(wt_dir, ignore_errors=True)


def add_eval_worktree(
    *,
    pipeline_id: str,
    parent_commit: str,
    repo_root: Path = REPO_ROOT,
    worktree_parent: Path = WORKTREE_PARENT,
) -> Worktree:
    """Create a per-pipeline checkout of monet_code for the meta-agent to edit.

    coding-bench layout (differs from pristine exp_05)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    exp_05 evaluated a monet *worktree* in-place: it created a
    monet_code_eval worktree (``git worktree add`` of the OUTER eval repo)
    and ran the harbor mini-eval from that worktree's ``cwd`` with the
    monet_code submodule installed from there. coding-bench has no
    monet_code submodule and never installs monet from a worktree ``cwd``:
    the eval seam (``codingbench_eval``) drives ``python -m runner.run``
    from REPO_ROOT and the trial container ``git clone``s monet from a
    published ref (see ``codingbench_eval.DEFAULT_MONET_REF``). So here the
    worktree exists only to give the cursor-agent meta-agent an isolated
    monet_code checkout to analyze/implement/review against (and for the
    pipeline's commit/diff/revert bookkeeping); it is NOT what gets
    evaluated.

    Reconciliation (PHASE1_NOTES.md decision #2): instead of a submodule
    under the outer repo, we keep a standalone canonical monet_code clone
    at ``<repo_root>/monet_code`` and carve a private ``git worktree`` of
    *it* per pipeline. The eval-side ``eval_dir`` is a plain directory that
    holds the monet_code worktree at ``eval_dir/monet_code`` — matching the
    ``{{ wt_dir }}/monet_code/`` layout every prompt expects.

    Steps:
      1. `git -C <canonical> worktree add -b <branch> <eval_dir>/monet_code
         <parent_commit>` — a fresh branch checked out at the parent commit.

    The ``git worktree add`` takes an exclusive lock on the canonical
    clone's ``.git`` index, so we serialize it under a cross-process file
    lock to make N parallel pipelines safe.
    """
    canonical = repo_root / "monet_code"
    if not (canonical / ".git").exists():
        raise RuntimeError(
            f"canonical monet_code clone not found at {canonical}. "
            f"self-evolve needs a standalone monet_code git clone there "
            f"(the meta-agent edits a per-pipeline worktree of it). Clone "
            f"it first, e.g.\n"
            f"    git clone https://github.com/yifan-zhang_sfemu/monet_code.git "
            f"{canonical}"
        )

    monet_branch = make_branch_name(parent_commit, pipeline_id)
    eval_branch = monet_branch  # share the name; here both live in monet_code
    wt_dir = make_worktree_path(monet_branch)
    monet_dir = wt_dir / "monet_code"

    if wt_dir.exists():
        raise RuntimeError(f"worktree path already exists: {wt_dir}")
    wt_dir.mkdir(parents=True, exist_ok=False)

    # Serialize the index-lock-taking phase across workers. The lock file
    # lives under $TMPDIR (not in repo_root) so we don't depend on the
    # underlying repo filesystem having free space/inodes — flock is purely
    # an in-memory primitive between workers on the same host. The path is
    # keyed on the canonical clone's absolute path so two unrelated clones
    # don't serialize against each other.
    lock_path = _worktree_lock_path(canonical)
    # `git worktree add` on the shared Lustre FS under concurrent load
    # (multiple campaigns + confirm evals all checking out ~260 files at once)
    # intermittently fails mid-checkout with a TRANSIENT error: EINTR
    # ("Interrupted system call" → "Could not reset index file to revision"),
    # or a lock/exit-128 race. These are not real failures — a clean retry
    # almost always succeeds. Retrying here (instead of failing the whole
    # pipeline) is what stops a relaunch "stampede" or a busy cluster from
    # silently burning iterations on worktree-add. The branch/admin entry is
    # cleaned up between attempts so the same branch name can be reused.
    attempts = _worktree_add_retries()
    for attempt in range(1, attempts + 1):
        try:
            with _flock(lock_path):
                _run(
                    ["git", "worktree", "add", "-b", monet_branch,
                     str(monet_dir), parent_commit],
                    cwd=canonical,
                )
            break
        except subprocess.CalledProcessError:
            _cleanup_partial_worktree(canonical, wt_dir, monet_dir, monet_branch)
            if attempt >= attempts:
                shutil.rmtree(wt_dir, ignore_errors=True)
                raise
            time.sleep(min(30.0, 5.0 * attempt))
            wt_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Roll back the partially-created eval_dir so a retry under the same
            # branch name doesn't trip the "already exists" guard above.
            shutil.rmtree(wt_dir, ignore_errors=True)
            raise

    return Worktree(
        eval_dir=wt_dir,
        monet_dir=monet_dir,
        eval_branch=eval_branch,
        monet_branch=monet_branch,
        parent_commit=parent_commit,
    )


def link_dotenv_into_worktree(wt_dir: Path, *, repo_root: Path = REPO_ROOT) -> bool:
    """Symlink `<repo_root>/.env` → `<wt_dir>/.env` so a subprocess running
    in the worktree sees credentials at its idea of "repo root".

    Why: harbor's `MonetAgent.run` calls `core.credentials.resolve(...)`
    which calls `core.env.ensure_loaded()`. Inside the worktree's `.venv`
    (an editable install pointing at `<wt_dir>/src/`), the fallback
    `_DOTENV_PATH` resolves to `<wt_dir>/.env`. Without this symlink (and
    without `MONET_EVAL_REPO_ROOT` set) that path doesn't exist, and
    every mini-eval explodes with `MissingCredentialError`.

    Returns:
        True if a symlink (new or pre-existing) is in place after the
        call. False if there's no canonical `.env` to link to (e.g.
        first-run before `install.sh` populated it) — credentials must
        come from host env vars in that case.

    Idempotent: pre-existing symlinks pointing at the canonical path are
    left alone; broken symlinks and the (very unlikely) case of a real
    `.env` file already at `<wt_dir>/.env` are also no-ops to avoid
    surprising the user.
    """
    canonical = repo_root / ".env"
    target = wt_dir / ".env"
    if not canonical.exists():
        return False
    # is_symlink() doesn't follow the link; exists() does. So a broken
    # symlink reports is_symlink=True, exists=False — leave it alone
    # rather than silently overwriting.
    if target.is_symlink() or target.exists():
        return True
    try:
        os.symlink(canonical, target)
    except OSError:
        # Filesystem doesn't support symlinks (e.g. some network mounts);
        # fall back to a copy. Better than silently breaking credentials.
        shutil.copy2(canonical, target)
    return True


def sync_eval_worktree_deps(wt_dir: Path) -> None:
    """Install the same dep extras the canonical repo uses into wt_dir/.venv.

    Mirrors `install.sh` step 3: `uv sync --extra harbor --extra dev`. uv
    caches wheels globally, so on a warm host this is a sub-second no-op
    after the first call.

    Inherits the full parent environment (PATH, HOME, UV_CACHE_DIR, proxy
    vars, etc.) rather than the locked-down `_env_passthrough()` we use
    for git — uv needs network access and its full env to resolve deps.
    """
    proc = subprocess.run(
        ["uv", "sync", "--extra", "harbor", "--extra", "dev"],
        cwd=str(wt_dir),
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"uv sync in {wt_dir} failed (rc={proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )


# ─── Inspecting the worktree during the loop ─────────────────────────────


def commits_since(monet_dir: Path, base: str) -> list[str]:
    """Return the list of commit SHAs added on top of `base` in monet_dir.

    Newest first. Used by the orchestrator to detect "did the agent commit
    anything this iteration".
    """
    proc = _run(
        ["git", "log", "--format=%H", f"{base}..HEAD"],
        cwd=monet_dir,
        check=False,
    )
    return [s for s in (proc.stdout or "").splitlines() if s]


def head_sha(monet_dir: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=monet_dir).stdout.strip()


def diff_against(monet_dir: Path, base: str) -> str:
    """Return the unified diff `base..HEAD` for monet_dir."""
    proc = _run(
        ["git", "diff", f"{base}..HEAD", "--unified=3"],
        cwd=monet_dir,
        check=False,
        capture=True,
        timeout=120,
    )
    return proc.stdout or ""


def reset_to(monet_dir: Path, sha: str) -> None:
    """Hard-reset monet_dir to sha. Used to revert a guard-rejected iteration."""
    _run(["git", "reset", "--hard", sha], cwd=monet_dir)


def push_branch(monet_dir: Path, branch: str, remote: str = "origin") -> None:
    """Push the branch to remote (force not needed — it's a fresh branch)."""
    _run(["git", "push", "-u", remote, branch], cwd=monet_dir, timeout=300)


# ─── Removal helpers (used by both --cleanup-worktree and the cleanup CLI) ─


_WORKTREE_REMOVE_TIMEOUT_S = 180
_WORKTREE_PRUNE_TIMEOUT_S = 60
_BRANCH_DELETE_TIMEOUT_S = 60


def remove_eval_worktree(wt_dir: Path, *, repo_root: Path = REPO_ROOT) -> bool:
    """Remove a per-pipeline eval_dir + its monet_code worktree. True on success.

    coding-bench layout: ``eval_dir`` is a plain dir whose ``monet_code``
    subdir is a ``git worktree`` of the canonical ``<repo_root>/monet_code``
    clone. We ``git worktree remove`` the monet_code worktree against the
    canonical clone (NOT the outer repo — exp_05's monet_code_eval worktree
    model is gone), then rm -rf the eval_dir and prune.

    No-op-safe if eval_dir is already gone. Errors are non-fatal — the
    caller logs them and continues.
    """
    wt_dir = Path(wt_dir)
    canonical = Path(repo_root) / "monet_code"
    monet_dir = wt_dir / "monet_code"

    def _prune() -> bool:
        try:
            _run(
                ["git", "worktree", "prune"],
                cwd=canonical, check=False, timeout=_WORKTREE_PRUNE_TIMEOUT_S,
            )
            return True
        except subprocess.TimeoutExpired:
            return False

    if not wt_dir.exists():
        return _prune()

    ok = True
    if monet_dir.exists():
        try:
            _run(
                ["git", "worktree", "remove", "--force", str(monet_dir)],
                cwd=canonical, check=True, timeout=_WORKTREE_REMOVE_TIMEOUT_S,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            ok = False
    # Whether or not the git removal succeeded, drop the eval_dir tree.
    shutil.rmtree(wt_dir, ignore_errors=True)
    pruned = _prune()
    return ok and pruned


def delete_eval_branch(branch: str, *, repo_root: Path = REPO_ROOT) -> bool:
    """Force-delete the per-pipeline branch in the canonical monet_code clone.

    coding-bench layout: there is no separate monet_code_eval holding-pen
    branch; ``eval_branch`` == ``monet_branch`` and lives in the canonical
    monet_code clone. Safe no-op if absent.
    """
    proc = _run(
        ["git", "branch", "-D", branch],
        cwd=Path(repo_root) / "monet_code",
        check=False,
        timeout=_BRANCH_DELETE_TIMEOUT_S,
    )
    return proc.returncode == 0


def delete_submodule_branch(
    branch: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> bool:
    """Force-delete a monet_code submodule branch from the shared module storage."""
    if not branch.startswith("evolve/"):
        # Safety: never delete a non-evolve branch.
        raise ValueError(f"refusing to delete non-evolve branch {branch!r}")
    monet_canonical = repo_root / "monet_code"
    proc = _run(
        ["git", "branch", "-D", branch],
        cwd=monet_canonical,
        check=False,
        timeout=_BRANCH_DELETE_TIMEOUT_S,
    )
    return proc.returncode == 0


def list_evolve_branches(*, repo_root: Path = REPO_ROOT) -> list[str]:
    """Return all `evolve/*` branches in monet_code's local store."""
    monet_canonical = repo_root / "monet_code"
    proc = _run(
        ["git", "branch", "--list", "evolve/*", "--format=%(refname:short)"],
        cwd=monet_canonical,
        check=False,
    )
    return [b.strip() for b in (proc.stdout or "").splitlines() if b.strip()]


__all__ = [
    "REPO_ROOT",
    "WORKTREE_PARENT",
    "Worktree",
    "make_branch_name",
    "make_worktree_path",
    "current_monet_branch",
    "current_monet_commit",
    "add_eval_worktree",
    "link_dotenv_into_worktree",
    "sync_eval_worktree_deps",
    "commits_since",
    "head_sha",
    "diff_against",
    "reset_to",
    "push_branch",
    "remove_eval_worktree",
    "delete_eval_branch",
    "delete_submodule_branch",
    "list_evolve_branches",
    "short_sha",
]

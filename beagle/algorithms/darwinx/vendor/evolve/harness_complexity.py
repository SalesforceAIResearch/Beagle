"""Measure how big and how convoluted a harness is, at a given commit.

WHY THIS EXISTS
───────────────
The pipeline could previously only answer "did the score go up". That is enough
to justify an ADDITIVE edit, and it is exactly why an evolved agent only ever
grew: nothing in the loop could observe, let alone reward, a change that makes
the harness smaller or structurally simpler at equal capability.

The target trajectory is grow-then-consolidate: capability climbs while the
harness first accumulates, and later a rewrite folds several accumulated
special cases into one general mechanism, so capability holds or rises while
size and branching fall. To select for that, the loop needs a second axis, and
that axis has to be cheap and objective.

WHAT IS MEASURED
────────────────
Three numbers, read straight out of a git tree (no checkout, so it is safe to
call on a running campaign's bare repo):

* ``code_loc``   — non-blank, non-comment lines of agent code.
* ``prompt_loc`` — non-blank lines of the prompt/config templates. For a minimal
  agent the prompt is a first-class part of the harness, so leaving it out would
  let "simplification" mean moving code into the prompt.
* ``branches``   — count of control-flow keywords in the code. This is the one
  that distinguishes real consolidation from line-shuffling: replacing three
  special-case branches with one general mechanism lowers it even when the line
  count barely moves.

``total_loc`` is code+prompt, and is the headline size number.

DELIBERATE NON-GOALS
────────────────────
This is a *proxy*, not a complexity theory, and it is only ever used as one half
of a two-sided test (capability must not regress). Optimising it alone would be
trivially gameable — delete everything — which is why nothing in the loop is
allowed to accept on a complexity win by itself.
"""
from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from dataclasses import asdict, dataclass

# Which files constitute "the harness". Overridable so this works for an agent
# other than mini without a code change.
DEFAULT_CODE_GLOBS = ("src/minisweagent/**/*.py",)
DEFAULT_PROMPT_GLOBS = ("src/minisweagent/config/**/*.yaml", "src/minisweagent/config/**/*.yml")

# Control-flow keywords at a statement boundary. `else`/`try` are excluded:
# they do not add a decision point on their own, and counting them would reward
# rewriting `else:` into a second `if`.
_BRANCH_RE = re.compile(r"(?<![\w.])(if|elif|for|while|except|and|or)(?![\w])")
_PY_COMMENT = re.compile(r"^\s*#")


@dataclass(frozen=True)
class Complexity:
    files: int = 0
    code_loc: int = 0
    prompt_loc: int = 0
    branches: int = 0

    @property
    def total_loc(self) -> int:
        return self.code_loc + self.prompt_loc

    def as_dict(self) -> dict:
        d = asdict(self)
        d["total_loc"] = self.total_loc
        return d


def _globs(env_name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(env_name, "").strip()
    return tuple(p.strip() for p in raw.split(",") if p.strip()) if raw else default


def _matches(path: str, globs: tuple[str, ...]) -> bool:
    # fnmatch has no notion of `**`, and `*` happily crosses `/`, so `a/**/*.py`
    # already behaves recursively once the `**/` is collapsed.
    return any(fnmatch.fnmatch(path, g.replace("**/", "")) or fnmatch.fnmatch(path, g) for g in globs)


def _git(repo: str, args: list[str], *, binary: bool = False, timeout: int = 120):
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.decode('utf-8', 'replace')[:300]}")
    return p.stdout if binary else p.stdout.decode("utf-8", "replace")


def _read_tree(repo: str, sha: str, paths: list[str]) -> dict[str, str]:
    """Read many blobs from one commit in a single git process."""
    if not paths:
        return {}
    try:
        proc = subprocess.run(
            ["git", "-C", repo, "cat-file", "--batch"],
            input="\n".join(f"{sha}:{p}" for p in paths).encode(),
            capture_output=True, timeout=300,
        )
    except Exception:
        return {}
    out, res, pos = proc.stdout, {}, 0
    for path in paths:
        nl = out.find(b"\n", pos)
        if nl == -1:
            break
        header = out[pos:nl].decode("utf-8", "replace").split()
        pos = nl + 1
        if len(header) < 3:      # "<oid> missing" -- path absent at this commit
            continue
        size = int(header[2])
        res[path] = out[pos:pos + size].decode("utf-8", "replace")
        pos += size + 1          # trailing newline after each blob
    return res


_MEMO: dict[tuple[str, str], "Complexity"] = {}


def measure_commit(repo: str, sha: str) -> Complexity:
    """Complexity of the harness as it exists at ``sha``. Never raises.

    Memoised on (repo, sha): a commit's content is immutable, and on a Lustre
    filesystem the git calls dominate -- re-measuring the same commit once per
    node made whole-campaign analysis take minutes.
    """
    key = (repo, sha or "")
    if key in _MEMO:
        return _MEMO[key]
    try:
        listing = _git(repo, ["ls-tree", "-r", "--name-only", sha])
    except Exception:
        _MEMO[key] = Complexity()
        return _MEMO[key]

    code_globs = _globs("DARWINX_GATE_COMPLEXITY_CODE_GLOBS", DEFAULT_CODE_GLOBS)
    prompt_globs = _globs("DARWINX_GATE_COMPLEXITY_PROMPT_GLOBS", DEFAULT_PROMPT_GLOBS)

    # One `git cat-file --batch` for the whole tree. Reading file-by-file cost a
    # subprocess per file (~70) per node, which on Lustre made the analysis of a
    # single campaign take minutes and eventually time out.
    wanted = [
        p for p in (l.strip() for l in listing.splitlines())
        if p and (_matches(p, prompt_globs) or _matches(p, code_globs))
    ]
    blobs = _read_tree(repo, sha, wanted)

    files = code_loc = prompt_loc = branches = 0
    for path in listing.splitlines():
        path = path.strip()
        if not path:
            continue
        is_prompt = _matches(path, prompt_globs)
        # A prompt file under the code globs must count once, as a prompt.
        is_code = (not is_prompt) and _matches(path, code_globs)
        if not (is_prompt or is_code):
            continue
        blob = blobs.get(path)
        if blob is None:
            continue
        files += 1
        for line in blob.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if is_code:
                if _PY_COMMENT.match(line):
                    continue
                code_loc += 1
                branches += len(_BRANCH_RE.findall(stripped))
            else:
                prompt_loc += 1
    _MEMO[key] = Complexity(files=files, code_loc=code_loc, prompt_loc=prompt_loc, branches=branches)
    return _MEMO[key]


def delta(repo: str, parent_sha: str | None, child_sha: str | None) -> dict:
    """Parent/child complexity and the change between them. Never raises.

    Returns ``{}`` when either side cannot be measured, so callers can treat a
    missing measurement as "no complexity evidence" rather than as zero change,
    which would read as a no-op rewrite and be judged accordingly.
    """
    if not parent_sha or not child_sha:
        return {}
    before, after = measure_commit(repo, parent_sha), measure_commit(repo, child_sha)
    if before.files == 0 or after.files == 0:
        return {}
    return {
        "before": before.as_dict(),
        "after": after.as_dict(),
        "d_total_loc": after.total_loc - before.total_loc,
        "d_code_loc": after.code_loc - before.code_loc,
        "d_prompt_loc": after.prompt_loc - before.prompt_loc,
        "d_branches": after.branches - before.branches,
    }


def render_for_prompt(d: dict) -> str:
    """One-line human summary for the verdict evidence block."""
    if not d:
        return "harness complexity: unavailable"
    b, a = d["before"], d["after"]
    def sign(n: int) -> str:
        return f"+{n}" if n > 0 else str(n)
    return (
        f"harness complexity: {b['total_loc']} -> {a['total_loc']} lines "
        f"({sign(d['d_total_loc'])}; code {sign(d['d_code_loc'])}, "
        f"prompt {sign(d['d_prompt_loc'])}), "
        f"branches {b['branches']} -> {a['branches']} ({sign(d['d_branches'])})"
    )

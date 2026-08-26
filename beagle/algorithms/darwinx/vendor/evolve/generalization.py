"""Anti-overfitting guard: static diff scan + canary-task picker.

This module is the Layer-2 (static) and Layer-3 (canary) parts of the plan's
generalization guard. Layer 1 (prompt-level constraints) lives in the prompt
templates under `prompts/`.

Pure-functional so the orchestrator's behavior is unit-testable without a
real cursor-agent run.

The scanner errs on the side of false positives — it's better to make the
agent re-prompt for a generalized fix than to land an overfit one.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Violation:
    """One overfitting signal found in the diff."""

    kind: str          # 'task_name' / 'trial_suffix' / 'copied_output' / 'narrowing_conditional'
    pattern: str       # what was matched (truncated)
    file: str | None   # the file in the diff where it appeared
    line: str          # the offending added line (truncated)


# Narrowing-conditional patterns we forbid in added code. Designed to catch
# the obvious overfit patterns without nuking legitimate generic code.
# These all assume "task" is a meaningful identifier in monet_code (it is
# not — monet_code itself doesn't see task names). Any reference is suspect.
# IGNORECASE so we catch both `task_name`/`taskname`/`taskName` (Python +
# TypeScript conventions both end up in monet_code's diff).
_NARROWING_REGEX = [
    re.compile(r"\bif\b[^\n]*\btask[_\s]*name\b[^\n]*==\s*['\"]", re.IGNORECASE),
    re.compile(r"\bif\b[^\n]*\btask\b\s*==\s*['\"]", re.IGNORECASE),
    re.compile(r"\.startswith\(\s*['\"]/app/", re.IGNORECASE),
]


def scan_diff(
    diff_text: str,
    *,
    claimed_tasks: list[str],
    test_outputs: list[str] | None = None,
    min_copied_len: int = 32,
) -> list[Violation]:
    """Scan a unified-diff string for overfitting signals.

    Args:
        diff_text: output of `git diff <parent>..<HEAD>`. Only ADDED lines
            (those starting with `+` but not `+++`) are scanned.
        claimed_tasks: full task names (e.g. 'feal-linear-cryptanalysis')
            currently being fixed by this iteration. Any added line that
            contains one of these as a string literal is a violation.
        test_outputs: optional list of `verifier/test-stdout.txt` contents
            for the claimed tasks. Any string of length >= min_copied_len
            that appears verbatim in both the test output AND an added line
            is a violation (suggests the agent copied the expected output).
        min_copied_len: shortest copied substring to flag (default 32).

    Returns a list of Violation. Empty list = clean.
    """
    violations: list[Violation] = []
    test_outputs = test_outputs or []

    # Pre-compute trial suffixes from claimed_tasks. The convention is
    # `<task-name>__<6char_hash>` so we strip after `__`.
    trial_suffixes = []
    for t in claimed_tasks:
        if "__" in t:
            suffix = t.split("__", 1)[1]
            if len(suffix) >= 4:  # avoid catching short suffixes that aren't trial ids
                trial_suffixes.append(suffix)

    # Build a denylist of long substrings from test outputs.
    long_strings: list[str] = []
    for output in test_outputs:
        # Pull out all sufficiently-long quoted strings or distinctive lines.
        for line in output.splitlines():
            line = line.strip()
            if len(line) >= min_copied_len:
                long_strings.append(line)

    current_file: str | None = None
    for raw in diff_text.splitlines():
        if raw.startswith("+++ b/"):
            current_file = raw[6:]
            continue
        if raw.startswith("+++ ") or raw.startswith("--- "):
            continue
        if not raw.startswith("+"):
            continue
        # Strip the leading '+' to get the actual added content.
        added = raw[1:]

        # 1) Claimed task name as substring of added line.
        for task in claimed_tasks:
            if task and task in added:
                violations.append(Violation(
                    kind="task_name",
                    pattern=task,
                    file=current_file,
                    line=_truncate(added),
                ))

        # 2) Trial suffix as substring (catches '__abc123' tokens).
        for suffix in trial_suffixes:
            # Use word-ish boundaries so we don't match unrelated hex.
            if f"__{suffix}" in added:
                violations.append(Violation(
                    kind="trial_suffix",
                    pattern=f"__{suffix}",
                    file=current_file,
                    line=_truncate(added),
                ))

        # 3) Long verbatim copies of test output.
        for s in long_strings:
            if s in added:
                violations.append(Violation(
                    kind="copied_output",
                    pattern=_truncate(s, 60),
                    file=current_file,
                    line=_truncate(added),
                ))
                break  # one is enough per line

        # 4) Narrowing conditional patterns.
        for rx in _NARROWING_REGEX:
            if rx.search(added):
                violations.append(Violation(
                    kind="narrowing_conditional",
                    pattern=rx.pattern,
                    file=current_file,
                    line=_truncate(added),
                ))

    return violations


# --- v9: code vs skills surface separation (manager's directive) ---
# A "skill" is a reusable task-solving procedure (skills/<name>/SKILL.md or a
# bundled-skills entry). Editing/adding a skill is purely additive and touches no
# shared core, so skill-surface removals must NOT count against the code-rewrite
# budget, and the proposer/verdict track code-edits vs skill-edits separately.
_SKILL_MARKERS = ("/skills/", "skills/", "skill.md", "bundled-skills", "skill-installer")


def is_skill_path(path: str) -> bool:
    p = (path or "").lower()
    return any(m in p for m in _SKILL_MARKERS)


def classify_diff_surface(diff_text: str) -> str:
    """Classify a diff's surface: 'skill', 'code', 'mixed', or 'none'.

    Used to keep code and skill improvements separately attributable (and to let
    the gate/verdict prefer additive skill changes)."""
    files = [raw[6:] for raw in diff_text.splitlines() if raw.startswith("--- a/")]
    files += [raw[6:] for raw in diff_text.splitlines() if raw.startswith("+++ b/")]
    files = [f for f in files if f and f != "/dev/null"]
    if not files:
        return "none"
    skill = any(is_skill_path(f) for f in files)
    code = any((not is_skill_path(f)) and ("test" not in f.lower()) for f in files)
    if skill and code:
        return "mixed"
    return "skill" if skill else "code"


# Shared-core "hot" surfaces: monet's execution/dispatch engine. The dominant
# rejected-proposal failure mode is a BROAD change here (even a purely-additive
# +N/-0 one) that perturbs unrelated tasks. Touching these beyond a small churn
# budget should be bounced PRE-eval and retargeted as an additive SKILL.
_SHARED_CORE_SUBSTRINGS = (
    "src/query/loop.js", "src/query/", "src/core/agents", "src/core/agent-registry",
    "streaming-tool-executor", "src/tools/",
)
# Global skill REGISTRY: editing this makes a skill always-on for EVERY task
# (the "globally-bundled skill" failure mode) — distinct from adding a narrow,
# cue-gated standalone skills/<name>/SKILL.md. The old guard mis-exempted it as a
# safe "skill" surface; treat broad changes to it as global-scope.
_GLOBAL_BUNDLE_SUBSTRINGS = ("bundled-skills", "core/skills.js")


def _churn_budget(env_name: str, default: int) -> int:
    import os
    try:
        return max(0, int(os.environ.get(env_name, "").strip() or default))
    except (TypeError, ValueError):
        return default


def scan_diff_locality(
    diff_text: str,
    *,
    max_deletions: int = 40,
    protected_substrings: list[str] | None = None,
) -> list[Violation]:
    """Flag NON-ADDITIVE / BROAD diffs (additive-scope + locality constraint).

    Two failure modes, both bounced here PRE-eval (revert + re-prompt) so the
    final GATE is not the only line of defense and no eval is wasted:

    1. DESTRUCTIVE rewrite: removals exceed ``max_deletions`` (a modification is
       `-`+`+`; a pure addition has ~no `-`). [original behavior]
    2. BROAD shared-core / global change (NEW): churn (added+removed) to monet's
       shared execution core (``src/query/loop.js`` etc.) or to the GLOBAL skill
       registry beyond a small budget — even if additive — because the gradient
       localizes a task-specific fault, so a broad shared-core edit can't be the
       localized fix and reliably regresses unrelated tasks. Steer it to an
       additive, cue-gated SKILL or a narrowly-guarded branch.

    Budgets are env-tunable: DARWINX_GATE_SHARED_CORE_CHURN_BUDGET (default 30),
    DARWINX_GATE_GLOBAL_BUNDLE_CHURN_BUDGET (default 8).
    """
    removed = 0
    removed_by_file: dict[str, int] = {}
    churn_by_file: dict[str, int] = {}      # added + removed, per file (NEW)
    cur: str | None = None
    for raw in diff_text.splitlines():
        if raw.startswith("--- a/"):
            cur = raw[6:]
            continue
        if raw.startswith("+++ b/"):
            # prefer the b/ path when a/ is /dev/null (pure new file)
            if not cur or cur == "/dev/null":
                cur = raw[6:]
            continue
        if raw.startswith("---") or raw.startswith("+++"):
            continue
        is_add = raw.startswith("+")
        is_del = raw.startswith("-")
        if is_add or is_del:
            if cur:
                churn_by_file[cur] = churn_by_file.get(cur, 0) + 1
        if is_del:
            if cur and is_skill_path(cur):
                continue  # skill-surface removals are additive-safe (no shared core)
            removed += 1
            if cur:
                removed_by_file[cur] = removed_by_file.get(cur, 0) + 1
    violations: list[Violation] = []

    # (2) BROAD shared-core change — churn-based (catches additive-broad rewrites
    # of the execution engine the deletion-only budget missed). NOTE: we do NOT
    # block additive changes to the skill registry (bundled-skills.js) — ADDING a
    # bundled skill is the intended NARROW/additive path the guard steers toward,
    # so blocking it deadlocks the proposer (no valid surface). Skill paths are
    # exempt from the destructive-removal count already (is_skill_path), and the
    # GATE + eval judge whether a new bundled skill is too broad. Only genuine
    # execution-core (loop/dispatch) churn is bounced pre-eval here.
    core_budget = _churn_budget("DARWINX_GATE_SHARED_CORE_CHURN_BUDGET", 40)
    for f, churn in churn_by_file.items():
        fl = f.lower()
        if any(s in fl for s in _GLOBAL_BUNDLE_SUBSTRINGS):
            continue  # skill-registry edits go through the skill path (additive)
        if any(s in fl for s in _SHARED_CORE_SUBSTRINGS) and churn > core_budget:
            violations.append(Violation(
                kind="broad_shared_core_change",
                pattern=(f"{churn} lines changed in monet's shared execution core "
                         f"(budget {core_budget}) — the fault is task-specific, so a "
                         f"broad shared-core edit can't be the localized fix and "
                         f"regresses unrelated tasks. INSTEAD add a NEW, narrow, "
                         f"cue-gated skill as an entry in the BUNDLED_SKILLS array in "
                         f"src/core/bundled-skills.js (additive, ships with the agent, "
                         f"cannot regress unrelated tasks) — do NOT rewrite "
                         f"src/query/loop.js, and do NOT use .monet/skills/ (that dir "
                         f"is gitignored/runtime-only and will NOT persist)"),
                file=f, line=f"churn={churn} in {f}",
            ))

    if removed > max_deletions:
        worst = (max(removed_by_file, key=removed_by_file.get)
                 if removed_by_file else None)
        violations.append(Violation(
            kind="excessive_modification",
            pattern=(f"{removed} existing lines removed (budget {max_deletions}) "
                     f"— rewriting shared logic regresses the stable base; ADD "
                     f"isolated, conditionally-guarded paths instead"),
            file=worst,
            line=f"total removals={removed}",
        ))
    for f, n in (removed_by_file.items() if protected_substrings else []):
        if n > 0 and any(p in f for p in protected_substrings):
            violations.append(Violation(
                kind="protected_surface_modified",
                pattern=(f"modified a protected shared-core surface "
                         f"({n} removals) — extend it additively, don't rewrite"),
                file=f,
                line=f"{n} removals in {f}",
            ))
    return violations


def _truncate(s: str, n: int = 120) -> str:
    s = s.rstrip()
    return s if len(s) <= n else s[: n - 3] + "..."


# Monet-file paths whose change is BEHAVIORAL/GLOBAL — an edit here can regress
# any task, so guards should spread broadly rather than target one cluster.
_GLOBAL_EDIT_HINTS = (
    "core/context.js", "core/loop", "query/", "core/prompt", "system-prompt",
    "core/skills.js", "bundled-skills", "core/agent", "core/coordinator",
)


def _changed_files_from_diff(diff_text: str | None) -> set[str]:
    """Parse the set of changed file paths from a unified git diff."""
    files: set[str] = set()
    for line in (diff_text or "").splitlines():
        m = re.match(r"^diff --git a/\S+ b/(.+)$", line) or re.match(r"^\+\+\+ b/(.+)$", line)
        if m:
            files.add(m.group(1).strip())
    return files


def _edit_is_global(changed_files: set[str]) -> bool:
    return any(any(h in f for h in _GLOBAL_EDIT_HINTS) for f in changed_files)


def _task_domain(task: str) -> str:
    """Capability cluster of a task (ml-numerical / systems-lowlevel / ...),
    falling back to the leading task-name token. Reuses pool's cluster map so
    guard targeting and cluster-batch claiming agree on what a 'domain' is."""
    try:
        from .pool import _capability_clusters, _task_cluster
        return _task_cluster(task, _capability_clusters())
    except Exception:
        return task.split("-", 1)[0] if "-" in task else task


def _domain_stratified(pool: list[str], k: int) -> list[str]:
    """Round-robin pick across task domains (capability cluster) to span as many
    capability areas as possible — catches BROAD/global regressions."""
    if k <= 0:
        return []
    try:
        import collections
        buckets: "collections.OrderedDict[str, list[str]]" = collections.OrderedDict()
        for t in pool:  # pool is already deterministically hash-sorted
            buckets.setdefault(_task_domain(t), []).append(t)
        picked: list[str] = []
        while len(picked) < k and any(buckets.values()):
            for dom in list(buckets.keys()):
                if buckets[dom]:
                    picked.append(buckets[dom].pop(0))
                    if len(picked) >= k:
                        break
        return picked[:k]
    except Exception:
        return pool[: max(0, k)]


def pick_canary_tasks(
    *,
    passing_tasks: list[str],
    claimed_tasks: list[str],
    k: int = 1,
    rng_seed: str = "",
    changed_files: set[str] | None = None,
    targeted_fraction: float = 0.5,
) -> list[str]:
    """Deterministically pick k currently-passing tasks as regression canaries.

    HYBRID selection (smarter than a flat random sample):
      1. TARGETED (~targeted_fraction of k): passing tasks in the SAME capability
         cluster as the CLAIMED tasks. A cluster-batch edit is made to fix that
         cluster, so its sibling tasks are the most collateral-likely — guard
         them first. (Most monet edits are behavioral, so file-diff→task-domain
         is weak; the claimed cluster is the strong intent signal.)
      2. SPREAD (the remainder): domain-stratified across all other clusters, to
         catch BROAD/global regressions (the dominant rejected-proposal failure
         mode: improve a few tasks, silently regress many elsewhere).
    If ``changed_files`` shows the edit touches a GLOBAL/behavioral monet file
    (loop/prompt/context/skills core), targeting is dropped (fraction→0) and the
    whole budget goes to domain-spread, since such an edit can regress anywhere.

    Drawn from `passing_tasks` minus claimed; deterministic shuffle on rng_seed
    so parallel pipelines pick different canaries. Returns up to k tasks.
    """
    pool = [t for t in passing_tasks if t not in set(claimed_tasks)]
    pool.sort(key=lambda t: hashlib.sha256(f"{rng_seed}|{t}".encode()).hexdigest())
    if k <= 1 or len(pool) <= k:
        return pool[: max(0, k)]

    if changed_files and _edit_is_global(changed_files):
        targeted_fraction = 0.0  # behavioral/global edit → pure spread

    n_targeted = int(round(k * max(0.0, min(1.0, targeted_fraction))))
    targeted: list[str] = []
    if n_targeted > 0 and claimed_tasks:
        claimed_domains = {_task_domain(t) for t in claimed_tasks}
        targeted = [t for t in pool if _task_domain(t) in claimed_domains][:n_targeted]

    chosen = set(targeted)
    remaining = [t for t in pool if t not in chosen]
    spread = _domain_stratified(remaining, k - len(targeted))
    return (targeted + spread)[:k]


def format_violations_for_prompt(violations: list[Violation]) -> str:
    """Render a violation list as a markdown block to splice into a re-prompt."""
    if not violations:
        return ""
    lines = ["The following signals were detected in your previous diff:", ""]
    for v in violations:
        # Violations may be Violation objects (static scan) OR plain strings
        # (e.g. anti-cheat entries appended as ``f"anti-cheat: {c}"`` in the
        # pipeline). Render both robustly — a str violation has no file/pattern/line.
        if isinstance(v, str):
            lines.append(f"- {v}")
            continue
        loc = getattr(v, "file", None) or "(unknown file)"
        lines.append(f"- **{getattr(v, 'kind', 'violation')}** in `{loc}`: matched `{getattr(v, 'pattern', '')}`")
        lines.append(f"    on line `{getattr(v, 'line', '?')}`")
    lines.extend([
        "",
        "Please rewrite the fix so the change is **general** — no claimed task names "
        "as string literals, no trial suffixes, no verbatim copies of expected output, "
        "no `if task_name == X` style shortcuts. The fix should apply to other "
        "similarly-failing tasks, not just these.",
    ])
    if any(getattr(v, "kind", None) in {"excessive_modification", "protected_surface_modified"}
           for v in violations):
        lines.extend([
            "",
            "**Make the change ADDITIVE.** The failing tasks share monet's core "
            "execution/tool-dispatch path with many tasks that currently PASS "
            "(build/compile/git/systems). Rewriting that shared path is why the "
            "previous diff would regress the stable base. Instead: ADD new, "
            "conditionally-guarded code paths/helpers for the target behaviour and "
            "leave the existing shared functions intact (aim for a near-pure-"
            "addition diff, like +N/-0..1). Do not delete or rewrite logic that "
            "passing tasks already depend on.",
        ])
    return "\n".join(lines)


__all__ = [
    "Violation",
    "scan_diff",
    "scan_diff_locality",
    "classify_diff_surface",
    "is_skill_path",
    "pick_canary_tasks",
    "format_violations_for_prompt",
]

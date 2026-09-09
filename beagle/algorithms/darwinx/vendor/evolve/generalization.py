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

# A "plugin" is code the evolvee LOADS from a mounted extension directory through a fixed hook
# (opencode: ``.opencode/plugin{,s}/*.{ts,js}`` implementing ``tool.execute.before``,
# ``chat.params`` and friends). Like a skill it is purely additive and cannot restructure the
# agent loop, but unlike a skill it carries executable behaviour — the missing middle rung between
# prose guidance and a shared-core edit. Empty by default: monet has no such surface, and a
# non-empty default would silently reclassify some of its core edits as bounded-risk plugin edits,
# which is the same wrong-evolvee failure the surface knobs exist to prevent.
_PLUGIN_MARKERS: tuple[str, ...] = ()


def _surface_paths(env_name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Path substrings naming a surface of THIS evolvee, overridable because they ARE the guard.

    The defaults name monet's files. Pointed at any other evolvee they match nothing, and every
    guard keyed to them becomes a no-op that still logs as though it ran. Measured 2026-08-26 on
    opencode: an 80-line additive edit to ``packages/opencode/src/session/`` drew zero violations,
    while the identical edit to monet's ``src/query/loop.js`` was bounced pre-eval. The proposer
    was meanwhile told its only skill surface was ``src/core/bundled-skills.js``, a file opencode
    does not have, so it spent both of its iterations editing the system prompt instead.
    """
    import os
    raw = (os.environ.get(env_name) or "").strip()
    if not raw:
        return default
    return tuple(x.strip().lower() for x in raw.split(",") if x.strip())


_MONET_SKILL_DOC = (
    "add a NEW, narrow, cue-gated skill as an entry in the BUNDLED_SKILLS array in "
    "src/core/bundled-skills.js (additive, ships with the agent, cannot regress unrelated "
    "tasks) — do NOT rewrite src/query/loop.js, and do NOT use .monet/skills/ (that dir is "
    "gitignored/runtime-only and will NOT persist)"
)


def skill_target_doc() -> str:
    """How THIS evolvee wants a new skill added, in the words the proposer will read.

    Hardcoding monet's answer here is what made the guidance actively wrong for opencode:
    it names a file the evolvee lacks, and dismisses the runtime skills dir as non-persistent
    when for opencode ``.opencode/skills/`` is committed and discovered at startup.
    """
    import os
    return (os.environ.get("DARWINX_GATE_SKILL_TARGET_DOC") or "").strip() or _MONET_SKILL_DOC


def evolvee_label() -> str:
    """What to call the agent under evolution in the proposer's instructions."""
    import os
    return (os.environ.get("DARWINX_GATE_EVOLVEE_LABEL") or "").strip() or "monet"


def core_path_doc() -> str:
    """Where THIS evolvee's core lives, as the proposer should read it."""
    import os
    return (os.environ.get("DARWINX_GATE_CORE_PATH_DOC") or "").strip() or "src/"


def prompt_surface_rule() -> str:
    """The prompt-surface prohibition, or empty when this evolvee has no separate one.

    Empty for monet, whose campaigns never split the prompt out of core; set for evolvees where
    a prompt edit would otherwise be the cheapest-looking general change and is bounced.
    """
    import os
    return (os.environ.get("DARWINX_GATE_PROMPT_RULE_DOC") or "").strip()


def plugin_target_doc() -> str:
    """How THIS evolvee loads a plugin, in the words the proposer will read.

    Empty when the evolvee has no loaded-extension surface (monet), in which case the proposer is
    told about core and skills only. The hook names belong to the evolvee, not to darwinx, so they
    are declared alongside the paths: naming a hook the evolvee does not have is the same defect as
    naming a skill file it does not have, which is what sent the last two iterations at the prompt.
    """
    import os
    return (os.environ.get("DARWINX_GATE_PLUGIN_TARGET_DOC") or "").strip()


def is_skill_path(path: str) -> bool:
    p = (path or "").lower()
    return any(m in p for m in _surface_paths("DARWINX_GATE_SKILL_PATH_MARKERS", _SKILL_MARKERS))


def is_plugin_path(path: str) -> bool:
    """Whether ``path`` is one of THIS evolvee's loaded plugin/hook files.

    False for every path until the evolvee declares ``DARWINX_GATE_PLUGIN_PATHS``, and declaring it
    is only half the job: the harness must actually mount the root the paths name, or a plugin the
    eval never loads measures zero however good the hook is. That is why ``_canonical`` refuses to
    launch when a declared plugin surface is not covered by the harness's mounted extension dir.
    """
    p = (path or "").lower()
    return any(m in p for m in _surface_paths("DARWINX_GATE_PLUGIN_PATHS", _PLUGIN_MARKERS))


def classify_diff_surface(diff_text: str) -> str:
    """Classify a diff's surface: 'skill', 'plugin', 'code', 'mixed', or 'none'.

    Used to keep core and extension improvements separately attributable (and to let the
    gate/verdict prefer additive changes). ``mixed`` means the shared core is involved *too*:
    a skill-and-plugin diff stays bounded on every side, so labelling it ``mixed`` would forfeit
    the additive bias it has earned. Where the two overlap, plugin wins — it is the stricter
    reading, since a plugin ships executable code and a skill only prose."""
    files = [raw[6:] for raw in diff_text.splitlines() if raw.startswith("--- a/")]
    files += [raw[6:] for raw in diff_text.splitlines() if raw.startswith("+++ b/")]
    files = [f for f in files if f and f != "/dev/null"]
    if not files:
        return "none"
    plugin = any(is_plugin_path(f) for f in files)
    skill = any(is_skill_path(f) and not is_plugin_path(f) for f in files)
    code = any(
        (not is_skill_path(f)) and (not is_plugin_path(f)) and ("test" not in f.lower())
        for f in files
    )
    if code and (skill or plugin):
        return "mixed"
    if plugin:
        return "plugin"
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
    new_files: set[str] = set()             # created by this diff, not modified by it
    cur: str | None = None
    for raw in diff_text.splitlines():
        # Reset on the file header. Without this, a NEW file's lines are charged to
        # whichever file preceded it in the diff: git writes `--- /dev/null` for a
        # creation, which does not match `--- a/`, so `cur` kept its previous value
        # and the `+++ b/` branch below declined to correct it because `cur` was
        # neither empty nor "/dev/null". Measured on the reverted candidate
        # d99cfe39: prompt.ts was charged churn=228 (its own 34 plus the 194 lines
        # of the new test file that followed it) and tripped a budget of 40 that its
        # real 34 lines were inside.
        if raw.startswith("diff --git "):
            cur = None
            continue
        if raw.startswith("--- a/"):
            cur = raw[6:]
            continue
        if raw.startswith("--- /dev/null"):
            cur = "/dev/null"               # a creation; the real path arrives on `+++ b/`
            continue
        if raw.startswith("+++ b/"):
            # prefer the b/ path when a/ is /dev/null (pure new file)
            if not cur or cur == "/dev/null":
                cur = raw[6:]
                new_files.add(cur)
            continue
        if raw.startswith("---") or raw.startswith("+++"):
            continue
        is_add = raw.startswith("+")
        is_del = raw.startswith("-")
        if is_add or is_del:
            if cur:
                churn_by_file[cur] = churn_by_file.get(cur, 0) + 1
        if is_del:
            if cur and (is_skill_path(cur) or is_plugin_path(cur)):
                continue  # extension-surface removals are additive-safe (no shared core)
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
    # A file this diff CREATES gets its own, larger budget, for the same reason the
    # skill registry is exempted just above: charging it the modification budget
    # deadlocks the proposer, because there is then no surface on which a capability
    # can be written at all.
    #
    # This rule exists to stop a broad REWRITE of shared logic, whose danger is that
    # it changes behavior every task already depends on. A new file changes nothing
    # by itself -- it runs only if an existing file calls it, and that call site is
    # an ordinary modification counted against `core_budget` above. So the thing the
    # rule protects is still protected.
    #
    # Measured on this campaign: the first four candidates were all the same shape --
    # a new capability file in src/session/ (67, 347, 152, 185 lines), a hook into
    # prompt.ts of 19-34 lines which was INSIDE the 40-line budget every time, and
    # tests, with ZERO deletions anywhere. All four were reverted as "broad shared-core
    # change", so four nodes produced no candidate while the proposer was doing exactly
    # the additive, tested work the method asks for. A budget that no realistic
    # TypeScript capability can satisfy is not a constraint, it is a deadlock.
    new_budget = _churn_budget("DARWINX_GATE_NEW_CORE_FILE_CHURN_BUDGET", 250)
    for f, churn in churn_by_file.items():
        fl = f.lower()
        if any(s in fl for s in _surface_paths(
                "DARWINX_GATE_GLOBAL_BUNDLE_PATHS", _GLOBAL_BUNDLE_SUBSTRINGS)):
            continue  # skill-registry edits go through the skill path (additive)
        if not any(s in fl for s in _surface_paths(
                "DARWINX_GATE_SHARED_CORE_PATHS", _SHARED_CORE_SUBSTRINGS)):
            continue
        is_new = f in new_files
        budget = new_budget if is_new else core_budget
        if churn <= budget:
            continue
        what = ("added as a NEW file in" if is_new else "changed in")
        # The remedy differs by case, and naming the wrong one keeps the proposer retrying a
        # shape that cannot be accepted. Measured: candidate 2d1344c added 82 lines to the
        # EXISTING prompt.ts and was told to "add a skill" -- while the shape the method
        # wants, and the budget already permits, is the bulk in a NEW file on the same core
        # surface plus a small hook here. That is what the earlier candidates did (new file
        # plus a 19-34 line hook, inside the budget every time).
        if is_new:
            remedy = (f"This is already the larger new-file budget, so narrow the capability "
                      f"or split it across files. Alternatively {skill_target_doc()}")
        else:
            remedy = (f"INSTEAD put the bulk of the capability in a NEW file on this same "
                      f"core surface -- a new file gets a much larger budget ({new_budget} "
                      f"lines), because it changes no existing behaviour until something "
                      f"calls it -- and leave only a SMALL hook (under {core_budget} lines) "
                      f"in this existing file. A new file plus a short call site is "
                      f"accepted; a large in-place addition is not. Failing that, "
                      f"{skill_target_doc()}")
        violations.append(Violation(
            kind="broad_shared_core_change",
            pattern=(f"{churn} lines {what} the evolvee's shared execution core "
                     f"(budget {budget}) — the fault is task-specific, so a "
                     f"broad shared-core edit can't be the localized fix and "
                     f"regresses unrelated tasks. {remedy}"),
            file=f, line=f"churn={churn} in {f}",
        ))

    # (3) PROMPT-surface edit — bounced outright rather than budgeted. A prompt edit is paid for
    # by every task, can't be credited to the capability it was meant to add, and leaves nothing
    # the agent carries into a task the prompt didn't anticipate. Both of the first two opencode
    # candidates were this exact shape — a verification block appended to the system prompt for
    # all GPT models — and each measured zero gain on every claimed task. This campaign evolves
    # the agent's own capability, so the proposer is sent to code or to a skill instead. Empty by
    # default: monet's campaigns never separated this surface and must keep their behavior.
    prompt_paths = _surface_paths("DARWINX_GATE_PROMPT_PATHS", ())
    for f, churn in churn_by_file.items():
        if any(x in f.lower() for x in prompt_paths):
            violations.append(Violation(
                kind="prompt_surface_edit",
                pattern=("edited the agent's PROMPT surface, which every task pays for and no "
                         "task can be credited for — this campaign evolves the agent's own "
                         f"capability, in code or in a skill. INSTEAD {skill_target_doc()}"),
                file=f,
                line=f"churn={churn} in {f}",
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
    hints = _surface_paths("DARWINX_GATE_GLOBAL_EDIT_PATHS", _GLOBAL_EDIT_HINTS)
    return any(any(h in f.lower() for h in hints) for f in changed_files)


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
    "is_plugin_path",
    "pick_canary_tasks",
    "format_violations_for_prompt",
]

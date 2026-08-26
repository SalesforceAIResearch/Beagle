"""Anti-cheat hard constraint for the equivalence gate (SEVerA-inspired).

SEVerA (arXiv 2603.25111) observes that self-evolving agents optimized on
soft metrics alone learn to *cheat*: they delete failing tests or weaken the
verification logic so the task "passes" without being solved. The paper's fix
is a HARD behavioral constraint (e.g. AST-diff equivalence) that prunes such
candidates during synthesis — which both prevents inflated scores and steers
search toward genuine improvements.

Here the candidate is the meta-agent's patch to ``monet_code`` (parent→child
git diff). This module statically scans that diff for tampering with the
agent's own evaluation/verification surface — i.e. the evolved agent making
itself "pass" by lowering its own bar rather than getting better. It is a
PURE function (no LLM, no network), so it runs as a hard constraint even when
the gateway is unavailable, and it is cheap enough to run before the LLM gate.

This is a prototype: the heuristics are deliberately conservative (favor false
negatives over falsely rejecting genuine refactors). Tune the patterns from
observed false-positives once we have live runs.
"""
from __future__ import annotations

import re

__all__ = ["scan_diff_for_eval_tampering"]

# Paths whose net DELETION is suspicious: the agent's own correctness/
# verification surface. monet's recent lineage strengthened "task contract
# verification evidence"; weakening it is exactly the cheat to catch.
_VERIFICATION_PATH_RE = re.compile(
    r"(test|spec|verif|contract|evidence|assert|check|guard|validat)",
    re.IGNORECASE,
)
# Added lines that smell like hard-coding benchmark answers / special-casing
# task ids to fake success rather than generalizing.
_HARDCODE_RE = re.compile(
    r"(terminal[-_]?bench|tb2|task[_-]?id\s*==|EXPECTED_(OUTPUT|ANSWER)|"
    r"GROUND[_-]?TRUTH|\bcheat\b|always\s+(return\s+)?(true|pass|0))",
    re.IGNORECASE,
)


def _iter_file_hunks(diff_text: str):
    """Yield (path, deleted_file_bool, added_lines, removed_lines) per file."""
    path = None
    deleted_file = False
    added: list[str] = []
    removed: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            if path is not None:
                yield path, deleted_file, added, removed
            m = re.search(r"b/(\S+)$", line)
            path = m.group(1) if m else line
            deleted_file, added, removed = False, [], []
        elif line.startswith("deleted file mode"):
            deleted_file = True
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
    if path is not None:
        yield path, deleted_file, added, removed


def scan_diff_for_eval_tampering(diff_text: str) -> list[str]:
    """Return human-readable reasons the diff looks like eval/verification
    tampering. Empty list ⇒ the candidate is clean by these heuristics.
    """
    if not diff_text or not diff_text.strip():
        return []
    reasons: list[str] = []
    for path, deleted_file, added, removed in _iter_file_hunks(diff_text):
        is_verif = bool(_VERIFICATION_PATH_RE.search(path))
        if deleted_file and is_verif:
            reasons.append(f"deletes verification/test file: {path}")
            continue
        # Net deletion in a verification surface (removed assertions/checks)
        if is_verif and len(removed) - len(added) >= 8:
            reasons.append(
                f"weakens verification surface {path} "
                f"(-{len(removed)}/+{len(added)} lines)"
            )
        # Hard-coded benchmark answers / task-id special-casing anywhere
        for ln in added:
            if _HARDCODE_RE.search(ln):
                reasons.append(
                    f"hard-codes benchmark/answer in {path}: {ln.strip()[:80]!r}"
                )
                break
    # de-dup, stable order
    seen: set[str] = set()
    out: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out

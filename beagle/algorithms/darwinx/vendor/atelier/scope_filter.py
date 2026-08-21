"""Layer-4 scope filter: apples-to-apples path allowlist on monet_code mutations.

Self_evolve's existing guards catch *content* overfitting (Layer-2:
task-name strings, narrowing conditionals, copied verifier output) and
*behavioral* regression (Layer-3: a previously-passing task starts failing
under the new diff). They do NOT catch *structural* overfitting — a diff
that's textually general but touches monet_code's core control loop, API
client, or other machinery outside the harness layer that every production
coding agent (Claude Code, Codex, Gemini CLI) ships with their CLI.

Atelier adds Layer-4 to defend against structural overfitting: the diff
must be confined to the apples-to-apples surface within monet_code, which
is the set of files a peer agent would ship as "their harness". Anything
outside that surface (query loop, provider clients, permission classifier,
sandbox, MCP/swarm/TUI) makes the candidate not directly comparable to
those peer agents — which is the whole rationale for Atelier's existence.

Two modes:

- ``ScopeMode.SOFT_FLAG`` (default) — record violations but allow the
  commit through. Used for *measurement* (what fraction of self_evolve's
  current discoveries fit the apples-to-apples surface?).
- ``ScopeMode.STRICT_REJECT`` — return a rejection signal that the
  orchestrator turns into a revert + retry. Used in production after the
  measurement phase shows the constraint is not catastrophically
  restrictive.

Like ``self_evolve.generalization.scan_diff``, this module is pure-functional
so unit tests don't need a real cursor-agent run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ScopeMode(Enum):
    """Whether scope violations reject the candidate or just get logged."""

    SOFT_FLAG = "soft_flag"
    STRICT_REJECT = "strict_reject"


# Apples-to-apples surface within monet_code/. These are the harness slots
# every production coding agent (Claude Code, Codex, Gemini CLI) ships in
# their CLI; an Atelier candidate that confines itself to these can be
# directly compared to a peer agent shipping equivalent changes.
DEFAULT_ALLOWLIST_PATTERNS: tuple[str, ...] = (
    r"^src/core/bundled-skills\.js$",
    r"^src/core/agents\.js$",
    r"^src/core/context\.js$",
    r"^src/core/hooks\.js$",
    r"^src/tools/[a-zA-Z0-9_\-]+\.js$",
    r"^\.monet/agents/.+\.md$",
    # Tests are allowed as long as they correspond to one of the above.
    # We don't try to enforce the correspondence here; that's a softer
    # constraint we can add in a later cut.
    r"^tests/.+\.test\.js$",
    # MONET.md / docs related to skills — fine to ship.
    r"^MONET\.md$",
    r"^docs/skills/.+\.md$",
)


# Explicit deny — always violation, even if accidentally added to allowlist.
# These are the core machinery and security paths that must remain stable
# for apples-to-apples comparability.
DEFAULT_DENYLIST_PATTERNS: tuple[str, ...] = (
    r"^src/query/",
    r"^src/api/",
    r"^src/core/permissions\.js$",
    r"^src/core/sandbox\.js$",
    r"^src/core/bash-classifier\.js$",
    r"^src/mcp/",
    r"^src/swarm/",
    r"^src/tui/",
    r"^src/bridge/",
    r"^src/buddy/",
    # bin/ owns the CLI entry point and option parser; changes there are
    # not harness-shaped.
    r"^bin/",
)


# Compiled patterns built once at module load.
_DEFAULT_ALLOW = tuple(re.compile(p) for p in DEFAULT_ALLOWLIST_PATTERNS)
_DEFAULT_DENY = tuple(re.compile(p) for p in DEFAULT_DENYLIST_PATTERNS)


@dataclass(frozen=True)
class ScopeViolation:
    """One out-of-scope file in the diff."""

    file: str
    """Path as it appeared in the diff (with or without `monet_code/` prefix)."""

    reason: str
    """One of ``'denied'`` (matched the denylist) or ``'outside_allowlist'``
    (matched neither allow nor deny; soft-illegal)."""


# ─── Path classifier ──────────────────────────────────────────────────────


def _strip_monet_prefix(path: str) -> str:
    """Strip a leading ``monet_code/`` prefix so internal patterns are
    relative to monet_code's own repo root.

    Accepts both ``monet_code/src/core/...`` and ``src/core/...``.
    """
    p = path.lstrip("/")
    if p.startswith("monet_code/"):
        p = p[len("monet_code/") :]
    return p


def classify_path(
    path: str,
    *,
    allowlist: tuple[re.Pattern, ...] = _DEFAULT_ALLOW,
    denylist: tuple[re.Pattern, ...] = _DEFAULT_DENY,
) -> ScopeViolation | None:
    """Classify one changed path against the allow / deny rules.

    Returns ``None`` if the path is in-scope, otherwise the corresponding
    ``ScopeViolation`` with ``reason ∈ {'denied', 'outside_allowlist'}``.

    Denylist takes priority over allowlist — a path matching both is
    rejected.
    """
    normalized = _strip_monet_prefix(path)

    for pat in denylist:
        if pat.search(normalized):
            return ScopeViolation(file=path, reason="denied")

    for pat in allowlist:
        if pat.search(normalized):
            return None

    return ScopeViolation(file=path, reason="outside_allowlist")


def scan_paths(
    changed_files: list[str],
    *,
    allowlist: tuple[re.Pattern, ...] = _DEFAULT_ALLOW,
    denylist: tuple[re.Pattern, ...] = _DEFAULT_DENY,
) -> list[ScopeViolation]:
    """Scan a list of changed-file paths and return all scope violations.

    Empty result = the candidate is fully within the apples-to-apples
    surface. The caller decides what to do with violations based on its
    ``ScopeMode``.
    """
    violations: list[ScopeViolation] = []
    for f in changed_files:
        v = classify_path(f, allowlist=allowlist, denylist=denylist)
        if v is not None:
            violations.append(v)
    return violations


# ─── Diff parser ──────────────────────────────────────────────────────────


_DIFF_HEADER_RE = re.compile(r"^diff --git a/(?P<path>.+?) b/.+$")


def parse_git_diff_files(diff_text: str) -> list[str]:
    """Extract the changed-file paths from a unified-diff string.

    Looks for ``diff --git a/PATH b/PATH`` headers. Whether the path has
    a ``monet_code/`` prefix depends on where the diff was produced from
    (``git -C monet_code diff …`` strips it; running diff from the outer
    repo includes it). ``classify_path`` handles both.
    """
    files: list[str] = []
    for line in diff_text.splitlines():
        m = _DIFF_HEADER_RE.match(line)
        if m:
            files.append(m.group("path"))
    return files


def scan_diff(
    diff_text: str,
    *,
    allowlist: tuple[re.Pattern, ...] = _DEFAULT_ALLOW,
    denylist: tuple[re.Pattern, ...] = _DEFAULT_DENY,
) -> list[ScopeViolation]:
    """Convenience: parse a unified-diff string and scan it for violations.

    Equivalent to ``scan_paths(parse_git_diff_files(diff_text))``.
    """
    return scan_paths(
        parse_git_diff_files(diff_text), allowlist=allowlist, denylist=denylist
    )


# ─── Decision helper ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScopeDecision:
    """The result of applying a ``ScopeMode`` to a list of violations."""

    accept: bool
    """True iff the candidate may proceed. Always True under SOFT_FLAG."""

    violations: list[ScopeViolation]
    """All violations found (regardless of ``accept``). Empty list = clean."""

    mode: ScopeMode
    """The mode this decision was made under."""

    def to_summary(self) -> str:
        """Short one-line summary suitable for orchestrator logs."""
        n = len(self.violations)
        if n == 0:
            return f"scope: clean ({self.mode.value})"
        n_denied = sum(1 for v in self.violations if v.reason == "denied")
        n_outside = n - n_denied
        verdict = "accept" if self.accept else "reject"
        return (
            f"scope: {verdict} ({self.mode.value}) — "
            f"{n} violations ({n_denied} denied, {n_outside} outside-allowlist)"
        )


def decide(
    violations: list[ScopeViolation],
    *,
    mode: ScopeMode = ScopeMode.SOFT_FLAG,
) -> ScopeDecision:
    """Convert a list of violations into an accept/reject decision under
    the given mode.

    - SOFT_FLAG: always accept, just record violations.
    - STRICT_REJECT: reject if there is at least one violation.
    """
    if mode is ScopeMode.STRICT_REJECT:
        accept = len(violations) == 0
    else:
        accept = True
    return ScopeDecision(accept=accept, violations=list(violations), mode=mode)


__all__ = [
    "ScopeMode",
    "ScopeViolation",
    "ScopeDecision",
    "DEFAULT_ALLOWLIST_PATTERNS",
    "DEFAULT_DENYLIST_PATTERNS",
    "classify_path",
    "scan_paths",
    "scan_diff",
    "parse_git_diff_files",
    "decide",
]

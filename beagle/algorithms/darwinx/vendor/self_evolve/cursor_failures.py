"""Shared Cursor-agent failure classifiers for self-evolve orchestration."""

from __future__ import annotations

from pathlib import Path

from . import cursor_agent


def cursor_failure_summary(
    log_path: Path,
    result: cursor_agent.CursorResult,
) -> str:
    """Return the most useful short failure detail for a Cursor run."""
    snippets: list[str] = []
    try:
        for line in log_path.read_text(errors="replace").splitlines()[-80:]:
            stripped = line.strip()
            if not stripped:
                continue
            if "AI Model Not Found" in stripped or "Model name is not valid" in stripped:
                snippets.append(stripped)
            elif "error" in stripped.lower() or "exception" in stripped.lower():
                snippets.append(stripped)
    except OSError:
        pass
    if snippets:
        detail = snippets[-1]
        if len(detail) > 500:
            detail = detail[:497] + "..."
        return f"{result.error}: {detail}" if result.error else detail
    return result.error or f"cursor-agent exited with code {result.exit_code}"

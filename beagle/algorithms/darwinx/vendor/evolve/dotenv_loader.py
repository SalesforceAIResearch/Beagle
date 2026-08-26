"""Load credentials and other env vars from the project-root ``.env`` file.

Ported from monet_code_eval's ``monet_eval.core.env`` (the only
``monet_eval.core`` bit the self-evolve supervisor needs). The supervisor
calls :func:`ensure_loaded` once at startup so every spawned subprocess
(workers, the eval seam's ``python -m runner.run``, cursor-agent) inherits
a fully-populated ``os.environ``.

The file is loaded at most once per process. Host environment variables
take precedence over ``.env`` values (``override=False``) so shell exports
still win without editing the file — which is exactly how this campaign
threads the live Express gateway key + loopback proxy URL in (they are
exported at launch from monet_code_eval's read-only ``.env``).

Repo-root discovery
~~~~~~~~~~~~~~~~~~~
coding-bench layout: ``self_evolve/dotenv_loader.py`` → ``parents[1]`` is
the repo root, where a (typically gitignored) ``.env`` may live. The
``DARWINX_EVAL_REPO_ROOT`` env var overrides this so a subprocess launched
from inside a per-pipeline worktree still resolves the canonical ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path

# Module-level fallback used when DARWINX_EVAL_REPO_ROOT is unset.
# coding-bench layout: self_evolve/dotenv_loader.py → parents[1] is the repo root.
_DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"
_loaded = False


def dotenv_path() -> Path:
    """Return the canonical ``.env`` path, resolved at call time so
    ``DARWINX_EVAL_REPO_ROOT`` can take effect for late-binding callers."""
    env = os.environ.get("DARWINX_EVAL_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve() / ".env"
    return _DOTENV_PATH


def ensure_loaded(*, override: bool = False) -> bool:
    """Load the canonical ``.env`` once. Returns True iff the file exists.

    Idempotent: subsequent calls short-circuit. Best-effort — a missing
    ``.env`` is fine (host env vars carry the credentials in that case).
    """
    global _loaded
    path = dotenv_path()
    if _loaded:
        return path.exists()
    _loaded = True
    if not path.exists():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_dotenv_minimal(path, override=override)
        return True
    load_dotenv(path, override=override)
    return True


def _load_dotenv_minimal(path: Path, *, override: bool) -> None:
    """Tiny KEY=VALUE parser used only if python-dotenv isn't installed."""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def reset_for_tests() -> None:
    """Forget the 'already loaded' flag. Use only from tests."""
    global _loaded
    _loaded = False

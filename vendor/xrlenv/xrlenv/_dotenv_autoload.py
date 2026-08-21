"""Private auto-loader for ``.env`` files at ``xrlenv`` import time.

This module exists for a single import-cycle reason: ``xrlenv/__init__.py``
calls :func:`_maybe_auto_load_dotenv` on package import, and that
call must not trigger ``xrlenv.client.__init__.py`` (which imports
:class:`xrlenv.client.Client` — fans out to docker / prometheus /
gRPC, all of which the in-sandbox stub explicitly avoids).

Putting the auto-load helper in a private top-level module
(sibling of :mod:`xrlenv._version`) keeps the dependency chain
flat: only stdlib. The user-facing public API in
:mod:`xrlenv.client.dotenv` re-exports from here so the import
``from xrlenv.client.dotenv import parse_dotenv`` continues to
work for SDK consumers — but they pay the heavier client-package
init cost only on first use, not on package import.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_AUTO_LOADED = False
_OFF_VALUES = frozenset({"off", "false", "0", "no", "disabled"})


def parse_dotenv(source: str | Path) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` ``.env`` file into a plain dict.

    See :func:`xrlenv.client.dotenv.parse_dotenv` for the full
    contract — this is the implementation; the public name
    re-exports from here.
    """
    text = Path(source).read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not _is_valid_env_key(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        out[key] = value
    return out


def _is_valid_env_key(key: str) -> bool:
    if not key:
        return False
    if not (key[0].isalpha() or key[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in key[1:])


def load_dotenv(
    *,
    path: str | Path | None = None,
    override: bool = False,
) -> dict[str, str]:
    """Read a ``.env`` file and populate ``os.environ`` from it.

    The operator-side fix for "every xrlenv script needs the same
    ``XRLENV_GRPC_HOST`` / ``XRLENV_CONSUMER_TOKEN`` / API key
    combo, and I'm tired of ``set -a; source .env; set +a`` before
    every command." Operators almost never call this directly:
    ``import xrlenv`` already invokes it once at package import time
    via :func:`_maybe_auto_load_dotenv`. The explicit function is
    for explicit-reload scenarios (test fixtures, multi-env
    scripts) or for callers who pass a non-default ``path=``.

    Discovery: when ``path`` is given, only that file is consulted.
    Otherwise the function walks up from ``Path.cwd()`` looking for
    a ``.env`` in each directory, stopping at the first match. If
    nothing is found, returns an empty dict — never raises.

    Precedence: **existing env vars win** by default. A value set
    in the operator's shell (e.g. via ``export``) overrides the
    ``.env`` file. Pass ``override=True`` to flip that.

    Returns the dict of keys actually applied to ``os.environ``
    (post-precedence), so callers can log what changed.

    **Always re-runs.** Direct calls to ``load_dotenv()`` walk the
    filesystem and re-apply on every invocation — they do NOT
    short-circuit on the module-level ``_AUTO_LOADED`` flag (that
    flag exists only to keep the import-time hook
    :func:`_maybe_auto_load_dotenv` from re-walking on package
    re-import). Callers who want re-reads after editing ``.env``
    mid-process can simply call again, subject to the precedence
    rule above (pass ``override=True`` if the keys are already in
    ``os.environ``).
    """
    global _AUTO_LOADED

    if path is not None:
        env_path: Path | None = Path(path).expanduser()
        if env_path is not None and not env_path.is_file():
            return {}
    else:
        env_path = _find_dotenv_upward(Path.cwd())
        if env_path is None:
            _AUTO_LOADED = True
            return {}

    assert env_path is not None
    try:
        parsed = parse_dotenv(env_path)
    except (OSError, UnicodeDecodeError) as exc:
        LOGGER.debug(
            "xrlenv.load_dotenv: skipping %s (unreadable: %s)",
            env_path, exc,
        )
        _AUTO_LOADED = True
        return {}

    applied: dict[str, str] = {}
    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    _AUTO_LOADED = True
    if applied:
        LOGGER.debug(
            "xrlenv.load_dotenv: applied %d key(s) from %s",
            len(applied), env_path,
        )
    return applied


def _find_dotenv_upward(start: Path) -> Path | None:
    for parent in (start, *start.parents):
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def _maybe_auto_load_dotenv() -> None:
    """Called at :mod:`xrlenv` package import. Loads the nearest
    ``.env`` once per process unless ``XRLENV_DOTENV=off`` (or
    ``false`` / ``0`` / ``no`` / ``disabled``).

    Side-effect-only and silent on failure: the import path must
    not raise from this even if ``.env`` is malformed or absent.
    """
    if _AUTO_LOADED:
        return
    if os.environ.get("XRLENV_DOTENV", "").strip().lower() in _OFF_VALUES:
        return
    try:
        load_dotenv()
    except Exception as exc:
        LOGGER.debug("xrlenv.load_dotenv auto-load skipped: %s", exc)

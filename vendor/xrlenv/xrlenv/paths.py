"""Single source of truth for the xrlenv operator-state root.

All control-plane operator state — ``state.db``, ``secrets/``, ``runs/``,
the admin trajectory cache, and the node-side build-context cache — lives
under one root directory. By default that root is ``~/.xrlenv``. Set
``XRLENV_HOME`` to relocate the whole tree somewhere else.

Why this knob exists
--------------------
On a shared filesystem (e.g. an FSx/Lustre home mounted identically on every
node), ``~/.xrlenv`` resolves to the *same physical directory* on every box.
Running a second control plane there — a dev cluster alongside prod — would
have both ``xrlenv up`` processes open the same ``state.db`` (two SQLite WAL
writers over a network FS: corruption) and share one ``secrets/`` token store
(a dev token would authenticate against prod). ``XRLENV_HOME`` is the clean
separator: each checkout's ``.env`` names its own ``XRLENV_HOME`` and the two
clusters' state never collides.

Why not just override ``$HOME``
-------------------------------
Pointing ``$HOME`` at a per-cluster directory would relocate xrlenv's state,
but it would *also* relocate everything else a process resolves through ``~``
— ``~/.gitconfig``, ``~/.ssh``, ``~/.docker/config.json`` (registry / Docker
Hub auth), pip caches — which is a surprising side-effect class for a process
that shells out to git and docker. ``XRLENV_HOME`` scopes the relocation to
xrlenv's own state and leaves all of that untouched.

How it gets set from ``.env``
-----------------------------
``xrlenv``'s package import auto-loads the nearest ``.env`` into ``os.environ``
(``xrlenv/__init__.py`` → :func:`xrlenv._dotenv_autoload._maybe_auto_load_dotenv`)
*before* any submodule body runs, with shell-exported values winning. So
``XRLENV_HOME=/path/to/cluster-home`` in the checkout's ``.env`` is already in
``os.environ`` by the time the module-level path constants below are evaluated.
No separate loader is needed here — this module just reads ``os.environ``.

The helpers are stdlib-only (``os`` + ``pathlib``) so importing this module
never widens the dependency surface — safe to import from the node agent and
anywhere in the control plane without touching the in-sandbox stub's slim-image
contract (``tests/unit/test_import_cycles.py``).
"""

from __future__ import annotations

import os
from pathlib import Path

#: Environment variable consulted by :func:`xrlenv_home`. Populated from the
#: shell or, as a fallback layer, from the checkout's ``.env`` at import time.
ENV_VAR = "XRLENV_HOME"

#: Environment variable that relocates ONLY ``state.db`` (see :func:`state_db_path`).
#: The control plane's SQLite state store is write-latency-sensitive and, on a shared
#: network filesystem (Lustre/FSx), (a) is ~6x slower per commit and (b) can't use WAL
#: (its mmap'd ``-shm`` faults with a fatal SIGBUS on a Lustre hiccup). Pointing this at
#: CP-box-LOCAL disk lets the state store run WAL (fast + non-blocking) while the rest of
#: ``$XRLENV_HOME`` (secrets/, runs/, config) stays on the shared FS.
STATE_DB_ENV_VAR = "XRLENV_STATE_DB_PATH"


def xrlenv_home() -> Path:
    """Return the operator-state root: ``$XRLENV_HOME`` if set, else ``~/.xrlenv``.

    Read fresh from ``os.environ`` on every call so tests can monkeypatch the
    variable and so a process that sets it before constructing paths sees the
    override. An empty / whitespace-only value is treated as unset (falls back
    to ``~/.xrlenv``) — a bare ``XRLENV_HOME=`` line in ``.env`` shouldn't
    silently relocate state to the filesystem root.
    """
    raw = os.environ.get(ENV_VAR)
    if raw and raw.strip():
        return Path(raw).expanduser()
    return Path.home() / ".xrlenv"


def state_db_path() -> Path:
    """Control-plane state store path.

    ``$XRLENV_STATE_DB_PATH`` if set (relocate ONLY ``state.db`` — typically to
    CP-box-local disk to escape shared-FS write latency and enable WAL; see
    :data:`STATE_DB_ENV_VAR`), else ``$XRLENV_HOME/state.db``. Read fresh from
    ``os.environ`` each call (tests monkeypatch); an empty/whitespace value is
    treated as unset so a bare ``XRLENV_STATE_DB_PATH=`` line can't relocate state
    to the filesystem root.
    """
    raw = os.environ.get(STATE_DB_ENV_VAR)
    if raw and raw.strip():
        return Path(raw).expanduser()
    return xrlenv_home() / "state.db"


def runs_root() -> Path:
    """Per-rollout artifact root: ``$XRLENV_HOME/runs``."""
    return xrlenv_home() / "runs"


def secrets_root() -> Path:
    """Token-store / secrets root: ``$XRLENV_HOME/secrets``."""
    return xrlenv_home() / "secrets"


def admin_cache_root() -> Path:
    """Admin trajectory cache: ``$XRLENV_HOME/admin-cache/trajectories``."""
    return xrlenv_home() / "admin-cache" / "trajectories"


def build_context_cache_root() -> Path:
    """Node-side git build-context cache: ``$XRLENV_HOME/build-context-cache``."""
    return xrlenv_home() / "build-context-cache"


__all__ = [
    "ENV_VAR",
    "STATE_DB_ENV_VAR",
    "admin_cache_root",
    "build_context_cache_root",
    "runs_root",
    "secrets_root",
    "state_db_path",
    "xrlenv_home",
]

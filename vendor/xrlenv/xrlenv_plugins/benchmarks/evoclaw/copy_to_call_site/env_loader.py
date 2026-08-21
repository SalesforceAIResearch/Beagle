"""Load ``.env`` / ``.env_private`` from the callsite's project root.

So a run only needs ``python xrlenv_onboard/run_e2e_xrlenv.py ...`` with the
cluster + data vars (``XRLENV_GRPC_HOST`` / ``XRLENV_GRPC_PORT`` /
``XRLENV_CONSUMER_TOKEN`` / ``EVOCLAW_DATA_ROOT`` / ...) sitting in the EvoClaw
checkout's ``.env`` (and ``.env_private``, EvoClaw's convention) — no inlining.

Walks up from ``cwd`` to the nearest ancestor that has either file and loads
both, matching EvoClaw's own precedence (``scripts/run_all.py``): ``.env`` is the
base, ``.env_private`` **overrides** it, and an existing shell var wins over both.
Implemented with ``setdefault`` by applying ``.env_private`` first.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def load_project_dotenv(start: Path | None = None) -> Path | None:
    """Load ``.env`` + ``.env_private`` from the nearest project root. Returns it.

    Precedence: shell env > ``.env_private`` > ``.env`` (via ``setdefault``).

    A stale shell export silently shadowing ``.env_private`` is a nasty footgun
    (e.g. a leftover ``EVOCLAW_DATA_ROOT=/path/to/EvoClaw-data`` from ``source
    .env`` of the committed template). We still honor the shell (EvoClaw's own
    convention lets you do a one-off ``FOO=bar python ...``), but we **warn
    loudly** on stderr when a shell value differs from the file value, naming the
    ``unset`` that would let the file win.
    """
    here = (start or Path.cwd()).resolve()
    for root in (here, *here.parents):
        candidates = {name: root / name for name in (".env", ".env_private")}
        if not any(f.is_file() for f in candidates.values()):
            continue
        preexisting = dict(os.environ)  # the shell, before we touch anything
        # Intended file value: `.env` is the base, `.env_private` overrides it.
        intended: dict[str, str] = {}
        for name in (".env", ".env_private"):
            f = candidates[name]
            if f.is_file():
                intended.update(_parse(f))
        for key, value in intended.items():
            os.environ.setdefault(key, value)  # shell already set -> kept
        _warn_shadowed(preexisting, intended)
        return root
    return None


def _warn_shadowed(preexisting: dict[str, str], intended: dict[str, str]) -> None:
    shadowed = [k for k, v in intended.items() if k in preexisting and preexisting[k] != v]
    for key in sorted(shadowed):
        print(
            f"[env] WARNING: ${key} is set in your shell to {preexisting[key]!r}, "
            f"shadowing .env/.env_private ({intended[key]!r}). The wrapper honors "
            f"the shell (EvoClaw convention). If that shell value is stale, run: "
            f"unset {key}",
            file=sys.stderr,
        )


def _parse(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value[:1], value[-1:]) in (('"', '"'), ("'", "'")):
            value = value[1:-1]
        if key:
            out[key] = value
    return out

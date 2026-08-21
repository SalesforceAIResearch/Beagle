"""Load **bucket-1** facts/secrets from the project ``.env`` — the single source of truth for the
values that don't change run to run: xrlenv cluster topology, credentials, gateway endpoints.

Two buckets, one rule each (``notes/darwinx-env-inventory.md``):

* **Facts/secrets** (``XRLENV_*``, API keys, gateway URLs, ``GH_TOKEN``) — constant across runs,
  live in ``.env``, loaded here into ``os.environ`` once at CLI start.
* **Run knobs** (algorithm/runtime choices) — vary per run, live in the **config** (inline or
  YAML), never in ``.env``. If a knob is found in ``.env``, we load it (so nothing breaks) but
  warn that it belongs in config.

Host env wins over ``.env`` (``override=False``): a shell export or CI var is respected; ``.env``
fills only what's unset. This is beagle's own loader — no third-party dependency.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Prefixes that name run knobs (bucket 2). Present in ``.env`` → warn: they belong in config.
_KNOB_PREFIXES = ("SELF_EVOLVE_", "ATELIER_", "TRACE_ANALYZER_", "MONET_META_")


def find_dotenv(start: str | Path | None = None) -> Path | None:
    """The nearest ``.env`` at or above ``start`` (default: cwd), not searching above a ``.git``
    root (so a stray ``.env`` in a parent of the repo isn't picked up)."""
    cur = Path(start or Path.cwd()).resolve()
    for d in (cur, *cur.parents):
        cand = d / ".env"
        if cand.is_file():
            return cand
        if (d / ".git").exists():
            break   # reached the repo root without a .env
    return None


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines: skip blanks/comments, tolerate a leading ``export``, strip one
    layer of matching quotes. No interpolation — values are taken verbatim."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def load_project_dotenv(path: str | Path | None = None, *, override: bool = False,
                        verbose: bool = True) -> Path | None:
    """Load ``.env`` into ``os.environ`` and return the file loaded (or ``None`` if none found).

    ``path`` overrides discovery. Host env wins unless ``override=True``. Only KEYS are printed
    (never secret values); a one-line note lists any run-knob keys that should move to config.
    """
    p = Path(path) if path else find_dotenv()
    if p is None or not p.is_file():
        return None
    parsed = parse_dotenv(p.read_text(encoding="utf-8"))
    n = kept = 0
    for k, v in parsed.items():
        if override or k not in os.environ:
            os.environ[k] = v
            n += 1
        else:
            kept += 1  # already set in the host env → left as-is (host wins)
    if verbose:
        # Distinguish newly-loaded from already-present so "0" never reads as ".env was ignored"
        # (the common case: the user `source`d .env in their shell first, so host env already has it).
        note = f" ({kept} already set in host env, kept)" if kept else ""
        print(f"[beagle] loaded {n} env var(s) from {p}{note}")
        knobs = sorted(k for k in parsed if k.startswith(_KNOB_PREFIXES))
        if knobs:
            shown = ", ".join(knobs[:6]) + ("…" if len(knobs) > 6 else "")
            print(f"[beagle] note: {len(knobs)} run-knob(s) in {p.name} belong in config, not "
                  f".env ({shown}) — see notes/darwinx-env-inventory.md")
    return p


__all__ = ["find_dotenv", "parse_dotenv", "load_project_dotenv"]

"""Shared resolver + hard guard for the benchmark task-cache root.

Lives with the benchmarks (``xrlenv`` core is deliberately benchmark-agnostic — it manages
containers/images, not corpora). Every golden-path benchmark's ``build_cache`` /
``run_oracle_sweep`` / ``build_plan_gen`` routes its cache-root resolution through here.

**Renamed 2026-07-31.** The env var ``XRLENV_HARBOR_CACHE`` -> ``XRLENV_BENCHMARK_CACHE``, and
the shared cache path ``/path/to/data/xrlenv_harbor_cache`` ->
``/path/to/benchmark-cache`` (one shared cache serves every benchmark; the
"harbor" name was misleading now that non-harbor benchmarks share it).

The OLD var and OLD path are **hard-rejected** — a downstream user still pointing at the retired
cache would silently read stale/absent data and get **unreliable results**, so every benchmark
entrypoint fails loud and tells them to migrate rather than run against the wrong cache.
"""

from __future__ import annotations

import os

# The current env var + the retired one, and the retired path marker.
CACHE_ENV_VAR = "XRLENV_BENCHMARK_CACHE"
LEGACY_ENV_VAR = "XRLENV_HARBOR_CACHE"
LEGACY_PATH_MARKER = "xrlenv_harbor_cache"

_MIGRATION = (
    f"Export {CACHE_ENV_VAR}=/path/to/benchmark-cache (the NEW path). "
    f"The old {LEGACY_ENV_VAR} env var and the old .../{LEGACY_PATH_MARKER} path are RETIRED — "
    f"reusing them yields unreliable results; the correct, populated caches live ONLY under the "
    f"new path."
)


def guard_legacy_cache_env(explicit: str | None = None) -> None:
    """Fail loud if a caller still uses the retired cache env var or path (renamed 2026-07-31).

    Rejects, in order: the retired ``XRLENV_HARBOR_CACHE`` env var being set at all; and any
    resolved candidate (an explicit ``--dest`` or ``$XRLENV_BENCHMARK_CACHE``) that points at
    the retired ``.../xrlenv_harbor_cache`` path. Raises ``SystemExit`` with a migration hint."""
    # Membership, NOT truthiness (audit Low): the rule is "must not be SET", so an explicitly
    # empty ``XRLENV_HARBOR_CACHE=""`` is a stale-migration signal too, not a free pass.
    if LEGACY_ENV_VAR in os.environ:
        raise SystemExit(f"{LEGACY_ENV_VAR} is retired and must not be set. {_MIGRATION}")
    for candidate in (explicit, os.environ.get(CACHE_ENV_VAR)):
        if candidate and LEGACY_PATH_MARKER in candidate:
            raise SystemExit(f"the cache path {candidate!r} is retired. {_MIGRATION}")


def benchmark_cache_root(dest: str | None = None) -> str:
    """The benchmark task-cache root: ``dest`` if given, else ``$XRLENV_BENCHMARK_CACHE``.

    Fails loud on the retired var/path (:func:`guard_legacy_cache_env`) and if no root is
    resolvable at all."""
    guard_legacy_cache_env(dest)
    root = dest or os.environ.get(CACHE_ENV_VAR)
    if not root:
        raise SystemExit(f"no cache root: pass --dest or set {CACHE_ENV_VAR}")
    return root

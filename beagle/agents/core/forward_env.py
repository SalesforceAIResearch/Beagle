"""``forward_env`` normalization — shared by every agent that forwards host env into a
container (gateway creds/URL), and by ``beagle run --dry-run``'s pre-flight.

An entry is either:

* a bare string ``"VAR"`` — forward ``VAR`` from the host to ``VAR`` in the container
  (the common case: container and host names match),
* a ``[container_var, host_var]`` pair — when the two names differ, or
* a ``{container_var: host_var, …}`` mapping — same, one pair per item (the form the
  vendored driver's config emits, so a config copied from there Just Works).

Keeping the string form means a config lists ``[API_KEY, API_KEY_LIST, PROXY_URL]``
instead of the noisy ``[[API_KEY, API_KEY], …]``.
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


def normalize_forward_env(entries: Any) -> list[tuple[str, str]]:
    """Normalize ``forward_env`` entries → ``[(container_var, host_var), …]``.

    Accepts bare strings (``V`` → ``V→V``), 1- or 2-element lists/tuples, and
    ``{container: host}`` mappings. Falsy input → ``[]``. An **unrecognized** entry is
    skipped but **logged at WARNING** — a silently-dropped entry means creds that never
    reach the container (e.g. the gateway key), which surfaces only as an opaque
    in-container auth failure, so it must not pass unnoticed.
    """
    out: list[tuple[str, str]] = []
    for e in entries or ():
        if isinstance(e, str):
            out.append((e, e))
        elif isinstance(e, dict):
            out.extend((str(k), str(v)) for k, v in e.items())
        elif isinstance(e, (list, tuple)) and len(e) == 2:
            out.append((str(e[0]), str(e[1])))
        elif isinstance(e, (list, tuple)) and len(e) == 1:
            out.append((str(e[0]), str(e[0])))
        else:
            LOGGER.warning("forward_env: skipping unrecognized entry %r (expected a "
                           "string, a {container: host} dict, or a 1-2 element list)", e)
    return out


__all__ = ["normalize_forward_env"]

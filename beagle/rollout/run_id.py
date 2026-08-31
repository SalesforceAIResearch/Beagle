"""Run identity — a content hash of the config + a human-readable run id.

Two pure functions:

- :func:`compute_config_hash` hashes the **raw** config dict (what the user wrote, before
  pydantic fills defaults) so cosmetically-equal configs share a hash — the basis for
  resume's drift guard.
- :func:`build_run_id` names a run ``<utc-ts>__<model>__<agent>__<benchmark>__<short8>``.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from beagle.config import RunConfig

_SHORT_HASH_LEN = 8
_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def compute_config_hash(cfg_dict: dict[str, Any]) -> str:
    """``sha256:<hex>`` of the canonicalized raw config dict (order-independent)."""
    canonical = json.dumps(cfg_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _slug(s: str) -> str:
    return _SLUG.sub("-", s).strip("-") or "x"


def build_run_id(config: RunConfig, config_hash: str, *, timestamp: datetime | None = None) -> str:
    """``<YYYY-MM-DDTHH-MM-SSZ>__<model-slug>__<agent>__<benchmark>__<short-hash>``.

    ``timestamp`` is injectable for deterministic tests; defaults to now (UTC).
    """
    ts = (timestamp or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H-%M-%SZ")
    model = _slug(config.model.name.rsplit("/", 1)[-1])
    short = config_hash.split(":")[-1][:_SHORT_HASH_LEN]
    return f"{ts}__{model}__{_slug(config.agent.name)}__{_slug(config.benchmark.name)}__{short}"


__all__ = ["compute_config_hash", "build_run_id"]

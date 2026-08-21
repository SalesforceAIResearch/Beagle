"""Operator-managed image-pin list (spec 15 §"Operator pin list").

The pin list is a tiny YAML file the operator drops on each node:

```yaml
# /etc/xrlenv/image-pins.yaml
pins:
  - "xrlenv/hello-shell:0.1"
  - "xrlenv/terminal-bench-2:0.1"
```

Pinned images are never evicted by the LRU sweep, even under disk pressure.
Phase-0 only honors the file (and runtime
:py:meth:`xrlenv.node.image_cache.ImageCacheManager.pin`); phase-1 layers
adapter-installed pins on top.

The loader returns ``set[str]`` so the cache manager can do O(1) `in`
membership checks. Missing file → empty set (not an error).
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from xrlenv.errors import ManifestInvalid

LOGGER = logging.getLogger(__name__)

DEFAULT_PIN_FILE = Path("/etc/xrlenv/image-pins.yaml")


def load_image_pins(path: Path | None = None) -> set[str]:
    """Read ``image-pins.yaml`` and return the pinned image names as a set.

    ``path`` defaults to ``/etc/xrlenv/image-pins.yaml``; passing an explicit
    path is the recommended pattern for tests + non-default deployments.
    Missing file is OK — single-host setups don't need a pin list.
    """
    target = path or DEFAULT_PIN_FILE
    if not target.exists():
        LOGGER.debug("image-pins.yaml not present at %s; pin set is empty", target)
        return set()
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ManifestInvalid(f"{target}: top-level must be a mapping")
    pins = raw.get("pins") or []
    if not isinstance(pins, list):
        raise ManifestInvalid(f"{target}: 'pins' must be a list of image strings")
    out: set[str] = set()
    for entry in pins:
        if not isinstance(entry, str) or not entry:
            raise ManifestInvalid(
                f"{target}: every pin entry must be a non-empty string; got {entry!r}"
            )
        out.add(entry)
    return out


__all__ = ["DEFAULT_PIN_FILE", "load_image_pins"]

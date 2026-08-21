"""Unit tests for the pure mapping logic in ``scripts/retag_registry_images.py``.

The HTTP/registry I/O is exercised against a live registry, but the namespace
remap and catalog filtering must be exactly right — a bad ``dst_repo`` would push
images under the wrong name, and a bad ``repos_under`` would retag the wrong set
(or miss some). Loaded by file path (``scripts/`` is not a package).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "retag_registry_images.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("xrlenv_retag", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rt = _load()


def test_dst_repo_swaps_namespace() -> None:
    assert rt.dst_repo("xrlenv-seta-env/88", "xrlenv-seta-env", "seta-env") == "seta-env/88"
    assert rt.dst_repo("xrlenv-seta-env/0", "xrlenv-seta-env", "seta-env") == "seta-env/0"


def test_dst_repo_multi_segment_id_preserved() -> None:
    # only the leading namespace segment changes; the rest is preserved verbatim
    assert rt.dst_repo("xrlenv-seta-env/sub/9", "xrlenv-seta-env", "seta-env") == "seta-env/sub/9"


def test_dst_repo_bare_namespace() -> None:
    assert rt.dst_repo("xrlenv-seta-env", "xrlenv-seta-env", "seta-env") == "seta-env"


def test_dst_repo_unrelated_repo_unchanged() -> None:
    # a repo not under from_ns is left alone (defensive — we only ever feed it
    # repos_under(), but the prefix check must not mangle e.g. a similarly-named one)
    assert rt.dst_repo("xrlenv-seta-env-extra/1", "xrlenv-seta-env", "seta-env") == (
        "xrlenv-seta-env-extra/1"
    )


def test_repos_under_filters_and_sorts() -> None:
    catalog = [
        "seta-env/3",                 # already in dst namespace — excluded
        "xrlenv-seta-env/2",
        "xrlenv-seta-env/10",
        "xrlenv-seta-env-extra/1",    # different namespace — excluded (prefix guard)
        "other/thing",
        "xrlenv-seta-env",            # exact namespace — included
    ]
    got = rt.repos_under(catalog, "xrlenv-seta-env")
    assert got == ["xrlenv-seta-env", "xrlenv-seta-env/10", "xrlenv-seta-env/2"]


def test_repos_under_empty_when_namespace_absent() -> None:
    assert rt.repos_under(["seta-env/1", "foo/bar"], "xrlenv-seta-env") == []

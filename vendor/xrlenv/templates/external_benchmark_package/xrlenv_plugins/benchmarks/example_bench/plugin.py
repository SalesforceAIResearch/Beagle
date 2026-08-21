"""Entry-point callable for B11.2 discovery.

xrlenv's runtime walks the ``xrlenv.benchmarks`` Python entry-points
group at startup and calls each entry's resolved value. The contract:
the callable takes no arguments and returns a :class:`pathlib.Path`
(or an iterable of them) pointing at one or more ``manifest.yaml``
files on disk.

Use :func:`importlib.resources.files` to resolve paths inside the
installed wheel — that's the only form of ``__file__``-relative
path resolution that survives editable installs, namespace
packages, and zipped wheels uniformly.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def plugin_manifests() -> Path:
    """Return the path to this plug-in's ``manifest.yaml``.

    Multi-manifest plug-ins return a list (e.g.
    ``return [pkg / "a.yaml", pkg / "b.yaml"]``); single-manifest
    plug-ins return a single ``Path`` like this.
    """
    pkg = files("xrlenv_plugins.benchmarks.example_bench")
    # ``files()`` returns a Traversable; convert to Path so xrlenv's
    # discovery layer can stat / open it like any other on-disk file.
    return Path(str(pkg / "manifest.yaml"))

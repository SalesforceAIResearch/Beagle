"""Plug-in manifest discovery.

Three discovery paths feed the catalog at runtime startup:

1. **Built-in templates** at ``xrlenv/templates/`` are registered
   explicitly by :func:`build_distributed_runtime` /
   :func:`build_local_runtime`. The platform spine
   (``hello-shell``) lives here.
2. **In-tree plug-ins** under ``xrlenv_plugins/`` (PEP-420 namespace
   package — no ``__init__.py`` at the top levels). One
   ``manifest.yaml`` per plug-in at
   ``xrlenv_plugins/<category>/<name>/manifest.yaml``.
   :func:`find_plugin_manifest_files` walks this tree.
3. **External plug-ins** declare manifests via two B11 paths:

   - **Filesystem path** via :envvar:`XRLENV_TEMPLATE_DIRS` (an
     :data:`os.pathsep`-separated list of directories that the
     catalog ``register_dir``-walks for ``manifest.yaml`` files).
     :func:`extra_template_dirs_from_env` returns the parsed list.
   - **Python entry-points** under the ``xrlenv.benchmarks`` group.
     Each entry point loads to a callable that returns the plug-in's
     manifest path(s) — :func:`find_entry_point_manifest_files`
     resolves the group at runtime startup.

All three discovery paths are **soft**: missing trees / unset env
vars / no entry-points return empty results rather than raising. The
runtime imports them once at boot and feeds the union into
:py:meth:`TemplateCatalog.register_paths`.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

LOGGER = logging.getLogger(__name__)


# D22 — system paths a plug-in root must never resolve to. Narrower than
# spec-19's template-mount allowlist (which would deny ``/home`` and
# break the common dev workflow ``XRLENV_TEMPLATE_DIRS=$HOME/work/...``).
# The intent here is "if the runtime were to bind-mount this prefix into
# every sandbox, would system files leak in?". Anything matching gets
# dropped with a warning at discovery time and falls back to
# ``plugin_root=None`` (manifest registers; in-sandbox import only works
# if the adapter is image-bundled).
# Tuples of (prefix, exact_only). When ``exact_only`` is True the guard
# trips only when the resolved root EQUALS the prefix (used for "/" since
# every path is technically relative to root). When False the guard
# trips on equality OR ``is_relative_to`` (used for /etc, /proc, etc.).
_FORBIDDEN_PLUGIN_ROOT_PREFIXES: tuple[tuple[Path, bool], ...] = (
    (Path("/"), True),
    (Path("/etc"), False),
    (Path("/proc"), False),
    (Path("/sys"), False),
    (Path("/dev"), False),
    (Path("/var/run/docker.sock"), False),
)


class DiscoveredManifest(BaseModel):
    """A manifest discovered via the B11 external-plug-in paths.

    Carries the manifest path itself plus the plug-in *root directory*
    — the directory whose child is named ``xrlenv_plugins`` (i.e. what
    you'd put on ``PYTHONPATH`` to make ``xrlenv_plugins.benchmarks.<name>.adapter``
    importable). The runtime mounts each unique plug-in root into every
    sandbox so the in-sandbox stub's ``env_setup`` can resolve the
    adapter import; see ``xrlenv/backends/docker.py`` and
    ``DockerBackendConfig.extra_plugin_roots``.

    ``plugin_root`` is ``None`` when the manifest is not nested under
    an ``xrlenv_plugins/`` ancestor (e.g. a directly-pointed
    ``/some/dir/manifest.yaml`` with no canonical layout). The catalog
    still registers the manifest, but a warning is logged: the adapter
    must already be importable inside the sandbox via some other route
    (image-bundled, or a separate platform-injected mount).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    manifest_path: Path
    plugin_root: Path | None = None


def _resolve_plugin_root(manifest_path: Path) -> Path | None:
    """Ascend from ``manifest_path`` until the parent's name is
    ``xrlenv_plugins``; return that ancestor's parent (the directory you'd
    put on ``PYTHONPATH``).

    Returns ``None`` (with a warning) when:
      - no ``xrlenv_plugins`` ancestor exists; or
      - the resolved root would be a forbidden system path.
    """
    try:
        for parent in manifest_path.resolve().parents:
            if parent.name == "xrlenv_plugins":
                root = parent.parent
                resolved = root.resolve()
                for forbidden, exact_only in _FORBIDDEN_PLUGIN_ROOT_PREFIXES:
                    if exact_only:
                        match = resolved == forbidden
                    else:
                        match = (
                            resolved == forbidden
                            or resolved.is_relative_to(forbidden)
                        )
                    if match:
                        LOGGER.warning(
                            "manifest %s resolves to plug-in root %s under "
                            "forbidden system prefix %s — dropping; adapter "
                            "import inside the sandbox will only succeed if "
                            "it ships in the image",
                            manifest_path, resolved, forbidden,
                        )
                        return None
                return resolved
    except OSError as exc:
        LOGGER.warning(
            "manifest %s: failed to resolve plug-in root (%s); falling back "
            "to manifest-only registration",
            manifest_path, exc,
        )
        return None
    LOGGER.warning(
        "manifest %s is not under an xrlenv_plugins/ ancestor; adapter "
        "import inside the sandbox will only succeed if it ships in the "
        "image",
        manifest_path,
    )
    return None


#: B11.1 — operator-supplied template-dir list. Colon-separated on
#: POSIX, semicolon-separated on Windows (per :data:`os.pathsep`).
#: Each entry is a directory rooted on disk; the catalog
#: ``register_dir``-walks each for ``manifest.yaml`` files. Useful
#: for development workflows where a plug-in is checked out elsewhere
#: on disk and not yet pip-installed.
TEMPLATE_DIRS_ENV_VAR: str = "XRLENV_TEMPLATE_DIRS"

#: B11.2 — Python entry-points group used to discover external plug-ins
#: shipped as installed pip packages. Each entry point loads to a
#: callable returning :class:`Path` or :class:`Iterable[Path]` of
#: ``manifest.yaml`` files. See
#: ``docs/integration/tutorials/own_benchmark.md`` for the publishing recipe.
ENTRY_POINT_GROUP: str = "xrlenv.benchmarks"


def find_plugin_manifest_files(xrlenv_pkg_path: Path) -> list[Path]:
    """Return every in-tree ``manifest.yaml`` under ``xrlenv_plugins/``.

    Args:
        xrlenv_pkg_path: Filesystem path of the imported ``xrlenv``
            package directory (typically
            ``Path(xrlenv.__file__).resolve().parent``). The plug-ins
            tree lives at ``<xrlenv_pkg_path>/../xrlenv_plugins`` —
            sibling to the platform package, root of the repo for
            editable installs and the wheel for non-editable.

    Returns:
        Each ``manifest.yaml`` found at the canonical plug-in depth:
        ``xrlenv_plugins/<category>/<name>/manifest.yaml``. The
        two-level glob (``*/*/manifest.yaml``) covers
        ``benchmarks/<name>/`` plus near-future additions like
        ``runtimes/<name>/`` or ``integrations/<name>/`` without
        accidentally matching ``manifest.yaml`` files inside a
        plug-in's own ``tasks/`` or fixture trees. Returns ``[]``
        when the plug-ins root is absent.
    """
    plugins_root = xrlenv_pkg_path.parent / "xrlenv_plugins"
    if not plugins_root.is_dir():
        return []

    found: list[Path] = []
    for candidate in plugins_root.glob("*/*/manifest.yaml"):
        if candidate.is_file():
            found.append(candidate)
    LOGGER.debug(
        "manifest-discovery: %d plug-in manifest(s) under %s",
        len(found), plugins_root,
    )
    return found


def find_plugin_root(xrlenv_pkg_path: Path) -> Path | None:
    """Return ``<xrlenv_pkg_path>/../xrlenv_plugins`` if it exists.

    Used by :class:`xrlenv.backends.docker.DockerBackend` to bind-mount
    the plug-ins tree alongside the platform package so adapter modules
    (``xrlenv_plugins.<category>.<name>.adapter``) import natively
    inside the sandbox without a separate ``pip install``.
    """
    plugins_root = xrlenv_pkg_path.parent / "xrlenv_plugins"
    return plugins_root if plugins_root.is_dir() else None


def extra_template_dirs_from_env(
    env_var: str = TEMPLATE_DIRS_ENV_VAR,
) -> list[Path]:
    """B11.1 — return the operator-supplied template-dir list parsed
    from :envvar:`XRLENV_TEMPLATE_DIRS`.

    The env var holds an :data:`os.pathsep`-separated list of
    directories. Each entry is resolved to an absolute :class:`Path`
    but **not** required to exist — non-existent directories are
    dropped silently with a warning. The runtime appends the result
    to its base ``template_dirs`` list and walks each via
    :py:meth:`TemplateCatalog.register_dir`.

    Empty / unset env var returns ``[]``. Whitespace-only entries
    are ignored.
    """
    raw = os.environ.get(env_var, "")
    if not raw:
        return []
    out: list[Path] = []
    for chunk in raw.split(os.pathsep):
        path_str = chunk.strip()
        if not path_str:
            continue
        candidate = Path(path_str).expanduser()
        if not candidate.is_dir():
            LOGGER.warning(
                "%s: %r is not a directory — skipping",
                env_var, str(candidate),
            )
            continue
        out.append(candidate.resolve())
    LOGGER.debug(
        "template-dirs from %s: %d entry/entries (%s)",
        env_var, len(out), ", ".join(str(p) for p in out) or "(none)",
    )
    return out


def find_external_template_dir_manifests(
    env_var: str = TEMPLATE_DIRS_ENV_VAR,
) -> list[DiscoveredManifest]:
    """B11.1 + D22 — walk every directory in :envvar:`XRLENV_TEMPLATE_DIRS`
    for ``manifest.yaml`` files and return them paired with their plug-in
    root.

    Replaces the older two-step "get dirs, let the catalog rglob each"
    flow so the discovery output is uniform with
    :func:`find_entry_point_manifest_files`: both produce
    :class:`DiscoveredManifest` carrying the manifest path and the
    plug-in root needed to mount it into the sandbox. The catalog still
    registers via :py:meth:`TemplateCatalog.register_paths` — same end
    state, plus the plug-in root is now available for the docker
    backend to bind-mount.

    Manifests with no ``xrlenv_plugins/`` ancestor still register; their
    ``plugin_root`` is ``None`` and a warning is logged (manifest-only
    registration; the adapter must reach the sandbox via image-bundled
    code or a separate platform-injected mount).
    """
    out: list[DiscoveredManifest] = []
    for d in extra_template_dirs_from_env(env_var):
        for yaml_path in sorted(d.rglob("manifest.yaml")):
            if not yaml_path.is_file():
                continue
            out.append(
                DiscoveredManifest(
                    manifest_path=yaml_path,
                    plugin_root=_resolve_plugin_root(yaml_path),
                )
            )
    LOGGER.debug(
        "manifest-discovery: %d manifest(s) via %s", len(out), env_var,
    )
    return out


def find_entry_point_manifest_files(
    *, group: str = ENTRY_POINT_GROUP,
) -> list[DiscoveredManifest]:
    """B11.2 + D22 — discover ``manifest.yaml`` files exposed via Python
    entry-points, paired with each manifest's plug-in root.

    External pip packages declare:

    .. code-block:: toml

        [project.entry-points."xrlenv.benchmarks"]
        my_bench = "my_bench:plugin_manifests"

    where ``plugin_manifests`` is a callable that returns
    ``Path | Iterable[Path]`` pointing at ``manifest.yaml`` file(s).
    The callable signature is intentionally flexible so a plug-in
    package can:

      - return a single :class:`Path` (most plug-ins ship one
        manifest);
      - return a list (a multi-benchmark package can register
        several manifests at once);
      - resolve paths dynamically (e.g. read from package data via
        :func:`importlib.resources.files`).

    Errors per-entry-point are logged and skipped — one broken
    plug-in must not block the others.

    Each returned :class:`DiscoveredManifest` carries the manifest
    path plus the plug-in root inferred via :func:`_resolve_plugin_root`
    (the parent of the nearest ``xrlenv_plugins/`` ancestor). When the
    plug-in package follows the canonical layout
    ``xrlenv_plugins/<category>/<name>/manifest.yaml`` (which the B11.4
    skeleton enforces), ``plugin_root`` is the wheel's
    ``site-packages/`` directory — the right path for the docker
    backend to bind-mount under ``/opt/xrlenv-extras/<idx>``.
    """
    found: list[DiscoveredManifest] = []
    try:
        eps = importlib.metadata.entry_points(group=group)
    except Exception:
        LOGGER.exception(
            "entry-points lookup failed for group=%r; skipping", group,
        )
        return []

    for ep in eps:
        try:
            loaded = ep.load()
        except Exception:
            LOGGER.exception(
                "entry-point %r in group=%r failed to load — skipping",
                ep.name, group,
            )
            continue
        if not callable(loaded):
            LOGGER.warning(
                "entry-point %r in group=%r resolved to a non-callable "
                "(%r); plug-in must export a callable returning Path "
                "or Iterable[Path] — skipping",
                ep.name, group, type(loaded).__name__,
            )
            continue
        try:
            result = loaded()
        except Exception:
            LOGGER.exception(
                "entry-point %r in group=%r raised when called — skipping",
                ep.name, group,
            )
            continue

        for path in _coerce_paths(result, ep_name=ep.name):
            if path.is_file():
                found.append(
                    DiscoveredManifest(
                        manifest_path=path,
                        plugin_root=_resolve_plugin_root(path),
                    )
                )
            else:
                LOGGER.warning(
                    "entry-point %r returned %r which is not a file — "
                    "skipping (manifest must be on disk)",
                    ep.name, str(path),
                )
    LOGGER.debug(
        "manifest-discovery: %d external manifest(s) via entry-points group=%r",
        len(found), group,
    )
    return found


def _coerce_paths(
    result: object, *, ep_name: str,
) -> Iterable[Path]:
    """Normalise an entry-point callable's return value to an
    iterable of :class:`Path`. Accepts ``Path``, ``str``, or any
    iterable of those.
    """
    if isinstance(result, (str, Path)):
        return [Path(result)]
    if isinstance(result, Iterable):
        out: list[Path] = []
        for item in result:
            if isinstance(item, (str, Path)):
                out.append(Path(item))
            else:
                LOGGER.warning(
                    "entry-point %r yielded non-path item %r — skipping",
                    ep_name, item,
                )
        return out
    LOGGER.warning(
        "entry-point %r returned unexpected type %r — skipping",
        ep_name, type(result).__name__,
    )
    return []


__all__ = [
    "ENTRY_POINT_GROUP",
    "TEMPLATE_DIRS_ENV_VAR",
    "DiscoveredManifest",
    "extra_template_dirs_from_env",
    "find_entry_point_manifest_files",
    "find_external_template_dir_manifests",
    "find_plugin_manifest_files",
    "find_plugin_root",
]

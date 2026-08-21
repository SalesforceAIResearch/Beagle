"""Pluggable trajectory normalizers + a registry to dispatch by source.

A normalizer turns one agent's raw run file into a
:class:`~trace_analyzer.model.CanonicalTrajectory`. To support a new agent,
subclass :class:`TrajectoryNormalizer`, implement :meth:`normalize` (and,
optionally, :meth:`sniff` for ``--source auto`` detection), and register the
instance with :func:`register`. The monet subclass in
``trace_analyzer.normalizers.monet`` is the worked example: it reuses the
already-tested ``agents.monet.trajectory`` reducer wholesale.
"""

from __future__ import annotations

import abc
from pathlib import Path

from .model import CanonicalTrajectory


class NormalizeError(Exception):
    """Raised when a file can't be read as the expected source format."""


class TrajectoryNormalizer(abc.ABC):
    """Base class for all source-specific normalizers.

    Subclasses set :attr:`name` (the ``--source`` token) and implement
    :meth:`normalize`. :meth:`sniff` lets ``--source auto`` pick a normalizer by
    peeking at the file; the default returns ``False`` (never auto-selected).
    """

    name: str = ""

    @abc.abstractmethod
    def normalize(self, path: Path) -> CanonicalTrajectory:
        """Parse ``path`` into a :class:`CanonicalTrajectory`."""
        raise NotImplementedError

    @classmethod
    def sniff(cls, path: Path) -> bool:
        """Cheaply decide whether ``path`` looks like this source's format."""
        return False


_REGISTRY: dict[str, TrajectoryNormalizer] = {}


def register(normalizer: TrajectoryNormalizer) -> TrajectoryNormalizer:
    """Add a normalizer instance to the registry, keyed by its ``name``."""
    if not normalizer.name:
        raise ValueError(f"{type(normalizer).__name__} has no .name")
    _REGISTRY[normalizer.name] = normalizer
    return normalizer


def available() -> list[str]:
    """Registered source names, sorted for stable ``--help`` output."""
    return sorted(_REGISTRY)


def get(name: str) -> TrajectoryNormalizer:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise NormalizeError(
            f"unknown source {name!r}; available: {', '.join(available()) or '(none)'}"
        ) from None


def detect(path: Path) -> TrajectoryNormalizer:
    """Pick a normalizer for ``path`` by trying each one's :meth:`sniff`."""
    for name in available():
        norm = _REGISTRY[name]
        try:
            if type(norm).sniff(path):
                return norm
        except OSError as exc:
            raise NormalizeError(f"cannot read {path}: {exc}") from exc
    raise NormalizeError(
        f"could not auto-detect source for {path}; pass --source explicitly "
        f"(available: {', '.join(available()) or '(none)'})"
    )


def load(path: Path | str, *, source: str = "auto") -> CanonicalTrajectory:
    """Normalize ``path`` using ``source`` (or auto-detect when ``"auto"``)."""
    p = Path(path)
    if not p.is_file():
        raise NormalizeError(f"not a file: {p}")
    norm = detect(p) if source == "auto" else get(source)
    return norm.normalize(p)


def trace_id_from_path(path: Path) -> str:
    """Human-friendly trace id from a filename (strip trajectory suffixes)."""
    name = path.name
    for suffix in (".trajectory.jsonl", ".messages.jsonl", ".jsonl", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem

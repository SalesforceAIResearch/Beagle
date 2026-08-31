"""A tiny, typed registry used by the agent / algorithm / benchmark factories.

The factories in beagle are deliberately PyTorch-flavored: users refer to
components by short string names (``"monet"``, ``"darwinx"``,
``"terminal_bench_2_1"``) and a registry resolves the name to a builder. Each
subsystem owns its own :class:`Registry` instance; this module just provides the
mechanism.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name → builder map with decorator-style registration.

    Example
    -------
    >>> AGENTS: Registry[type] = Registry("agent")
    >>> @AGENTS.register("monet")
    ... class MonetAgent: ...
    >>> AGENTS.get("monet")  # -> MonetAgent
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str, obj: T | None = None) -> Callable[[T], T] | T:
        """Register ``obj`` under ``name``.

        Usable as a decorator (``@reg.register("x")``) or directly
        (``reg.register("x", obj)``). Re-registering an existing name raises.
        """

        def _do(target: T) -> T:
            if name in self._entries:
                raise KeyError(f"{self.kind} {name!r} is already registered")
            self._entries[name] = target
            return target

        if obj is not None:
            return _do(obj)
        return _do

    def get(self, name: str) -> T:
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(
                f"unknown {self.kind} {name!r}; registered: {sorted(self._entries)}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._entries))


__all__ = ["Registry"]

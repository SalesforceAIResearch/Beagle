"""The evolution-algorithm registry and ``build`` factory."""

from __future__ import annotations

from collections.abc import Callable

from beagle.algorithms.base import AlgorithmConfig, EvolveAlgorithm
from beagle.registry import Registry

#: name -> EvolveAlgorithm subclass. Populated by the ``@register`` decorator.
ALGORITHMS: Registry[type[EvolveAlgorithm]] = Registry("algorithm")


def register(name: str) -> Callable[[type[EvolveAlgorithm]], type[EvolveAlgorithm]]:
    """Class decorator: register an algorithm under ``name``."""

    def _decorate(cls: type[EvolveAlgorithm]) -> type[EvolveAlgorithm]:
        ALGORITHMS.register(name, cls)
        return cls

    return _decorate


def build(name: str | AlgorithmConfig, /, *, config: AlgorithmConfig | None = None,
          **kwargs) -> EvolveAlgorithm:
    """Build an algorithm from a name, or from a typed config instance.

    * ``build("darwinx", repo_root=…, max_loop_iters=1)`` — kwargs validated against the
      algorithm's ``Config`` (unknown knob fails loud).
    * ``build(DarwinXConfig(repo_root=…))`` — the declarative surface; the algorithm is inferred
      from the config type.

    ``config=`` may also be passed alongside a name to supply a ready config.
    """
    if isinstance(name, AlgorithmConfig):
        return _algorithm_for(type(name)).from_config(name)
    cls = ALGORITHMS.get(name)
    if config is None:
        config = cls.Config(**kwargs)
    return cls.from_config(config)


def _algorithm_for(config_cls: type[AlgorithmConfig]) -> type[EvolveAlgorithm]:
    """The registered algorithm whose ``Config`` is ``config_cls`` (configs are 1:1 with algos)."""
    for cls in (ALGORITHMS.get(n) for n in ALGORITHMS.names()):
        if cls.Config is config_cls:
            return cls
    raise KeyError(
        f"no registered algorithm declares {config_cls.__name__} as its Config — "
        f"pass a config whose type matches an algorithm's `Config`."
    )


def available() -> list[str]:
    return ALGORITHMS.names()


__all__ = ["ALGORITHMS", "register", "build", "available"]

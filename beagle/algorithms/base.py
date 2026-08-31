"""The evolution-algorithm base class — beagle's "optimizer".

An algorithm searches the space of harness variants and returns the best. Its **single
contract** is :meth:`evolve`: given a way to *score* a candidate (``evaluate``, built by the
Trainer from the data) plus the seed harness (``evolvee``) and the coding-agent it may drive
for edits (``evolver``), run the whole search and return the winner.

Everything past that is the algorithm's own business — it owns the entire recipe (selection,
prompts, phases, verdicts, archive, memory) and its own execution shape: a native algorithm
might run a generational loop; DarwinX launches a distributed supervisor. The framework
imposes **no** loop and no fixed cadence — that would presume a structure algorithms vary
on. If a common shape emerges once there are several native algorithms, factor it out
into an opt-in helper *then*.

The ``evolver`` is a thin coding-agent primitive: the algorithm calls
``evolver.edit(prompt, workspace, …)`` as many times as it likes, with its own prompts,
interleaving mini-evals and guards however it wants. The evolver only knows how to run one
coding instruction against one workspace — which keeps arbitrarily complex algorithms
unconstrained by the agent contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from pydantic import BaseModel, ConfigDict

from beagle.agents.core.base import AgentSource
from beagle.types import Reward, TaskId, TaskResult

if TYPE_CHECKING:
    from beagle.agents.core.base import Agent
    from beagle.config import RunConfig

#: How the algorithm scores one candidate: roll its ``source`` out on the data and set the
#: candidate's ``score``/``results`` in place. The Trainer builds this from the dataset + Runner.
Evaluate = Callable[["Candidate"], None]


class CandidateStatus(str, Enum):
    """Lifecycle of a source variant in the population."""

    PENDING = "pending"        # created, not yet evaluated
    EVALUATED = "evaluated"    # has a fitness score
    KEPT = "kept"              # accepted into the population (net improvement)
    NO_CHANGE = "no_change"    # evaluated but no net gain vs parent
    ARCHIVED = "archived"      # QD stepping stone (specialist), not on the main line
    REJECTED = "rejected"      # failed a guard / worse than parent


@dataclass
class Candidate:
    """One source variant under evaluation — a point in the search space.

    The ``source`` (an :class:`AgentSource`, i.e. a repo @ ref) is the θ; ``score`` +
    ``results`` are its measured fitness. Genealogy (``parent_id``, per-task deltas) is what
    tree-search / QD algorithms use to select and recombine.
    """

    id: str
    source: AgentSource
    parent_id: str | None = None
    status: CandidateStatus = CandidateStatus.PENDING
    score: Reward | None = None
    results: list[TaskResult] = field(default_factory=list)
    improved_tasks: list[TaskId] = field(default_factory=list)
    regressed_tasks: list[TaskId] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class AlgorithmConfig(BaseModel):
    """Typed configuration for an evolution algorithm — its knobs, validated.

    **Subclass per algorithm** (e.g. ``DarwinXConfig``) to declare typed fields; a concrete
    subclass sets ``extra='forbid'`` so an unknown/typo'd knob is a load-time error — the drift
    guard, matching :mod:`beagle.config`. This base is permissive (``extra='allow'``) so a
    trivial algorithm needs no config subclass at all.

    :meth:`to_driver_env` is the hook where an algorithm translates its knobs into the
    environment its (vendored) driver reads — empty here; overridden per algorithm.
    """

    model_config = ConfigDict(extra="allow")

    def to_driver_env(self) -> dict[str, str]:
        """Config knobs this algorithm exposes to its driver via env, translated from config
        (bucket-2 translation, kept next to the knobs). Default: nothing."""
        return {}


class EvolveAlgorithm(ABC):
    """Base class for evolution algorithms (the "Evolve Algorithm Factory").

    Subclasses implement the selection/variation policy AND their own execution — the only
    required method is :meth:`evolve`. Each declares its typed knobs via a :class:`AlgorithmConfig`
    subclass on :attr:`Config`; :func:`build` validates kwargs against it and calls
    :meth:`from_config`. ``state_dict``/``load_state_dict`` are optional, for checkpoint/resume.
    """

    #: The algorithm's typed config class. Subclasses override with their own ``AlgorithmConfig``.
    Config: ClassVar[type[AlgorithmConfig]] = AlgorithmConfig

    def __init__(self, config: AlgorithmConfig | None = None, **kwargs: Any) -> None:
        #: The validated typed config (built from kwargs against ``type(self).Config`` when not passed).
        self.config: AlgorithmConfig = config if config is not None else type(self).Config(**kwargs)

    @property
    def hparams(self) -> dict[str, Any]:
        """The set/overridden knobs as a plain dict — back-compat + generic access/display."""
        return self.config.model_dump(exclude_defaults=True)

    @classmethod
    def from_config(cls, config: AlgorithmConfig) -> EvolveAlgorithm:
        """Build the algorithm from its typed config. Override for custom construction."""
        return cls(config=config)

    @abstractmethod
    def evolve(
        self,
        *,
        evaluate: Evaluate,
        evolvee: Agent,
        evolver: Agent,
        val: Evaluate | None = None,
        config: RunConfig | None = None,
    ) -> Candidate | None:
        """Run the whole search and return the best harness (or ``None`` if none was kept).

        ``evaluate(candidate)`` scores a candidate on the training data (sets its
        ``score``/``results``); ``val`` is the optional held-out scorer. ``evolvee`` is the
        seed θ (Runnable + Evolvable); ``evolver`` is the coding agent to drive for edits
        (an Editor). The algorithm owns HOW it evolves — generational, async, launch-a-driver.
        """
        raise NotImplementedError

    # -- checkpointing (optional) --------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Serializable search state for resume. Override if the algorithm supports it."""
        raise NotImplementedError

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError


__all__ = ["CandidateStatus", "Candidate", "EvolveAlgorithm", "AlgorithmConfig", "Evaluate"]

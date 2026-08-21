"""Per-node trajectory reader (spec 17 §"Sink-aware reader").

Phase-0 ships the ``platform-jsonl`` reader: it reads the on-disk run
directory the :class:`PlatformJsonlSink` writes and slices the result
according to the spec-17 ``FetchRangeKind`` requested by the
control-plane viewer.

The reader is decoupled from the sink so phase-1 can drop in the
``slime-sample`` and ``verl-dataproto`` readers behind the same
``LocalTrajectoryReader`` Protocol without touching ``NodeAgent`` or
the spec-21 wire dispatch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from xrlenv.types import Trajectory

if TYPE_CHECKING:
    # Runtime-only import would close a cycle: this module is pulled in
    # by xrlenv.node.__init__ during xrlenv-node startup, and importing
    # PlatformJsonlSink eagerly triggers xrlenv.control.__init__, which
    # re-enters xrlenv.node.trajectory_reader before its module body has
    # finished loading. The instantiation site below imports lazily.
    from xrlenv.control.trajectory_sink import PlatformJsonlSink

LOGGER = logging.getLogger(__name__)

FetchRangeKind = Literal["whole", "summary_only", "step_range"]


class LocalTrajectoryReader(Protocol):
    """Phase-0 reader Protocol: hand back a :class:`Trajectory` for a range."""

    def read_range(
        self,
        rollout_id: str,
        *,
        range_kind: FetchRangeKind = "whole",
        step_start: int = 0,
        step_end: int | None = None,
    ) -> Trajectory: ...


class JsonlTrajectoryReader:
    """``platform-jsonl`` reader (spec 17). Wraps :class:`PlatformJsonlSink`.

    Range semantics:

    - ``whole`` — return the trajectory as-is (the spec-17 default).
    - ``summary_only`` — return the trajectory with ``steps=[]`` so the
      viewer can list / preview without paying for step bodies.
    - ``step_range`` — slice ``[step_start, step_end)``; ``step_end=None``
      means "to end of trajectory."
    """

    def __init__(self, runs_root: Path) -> None:
        # Deferred import — see TYPE_CHECKING block above for why.
        from xrlenv.control.trajectory_sink import PlatformJsonlSink

        self._sink: PlatformJsonlSink = PlatformJsonlSink(runs_root)

    @property
    def runs_root(self) -> Path:
        return self._sink._runs_root

    def read_range(
        self,
        rollout_id: str,
        *,
        range_kind: FetchRangeKind = "whole",
        step_start: int = 0,
        step_end: int | None = None,
    ) -> Trajectory:
        full = self._sink.read(rollout_id)
        if range_kind == "summary_only":
            return full.model_copy(update={"steps": []})
        if range_kind == "step_range":
            end = step_end if step_end is not None and step_end > 0 else len(full.steps)
            return full.model_copy(update={"steps": full.steps[step_start:end]})
        return full


__all__ = [
    "FetchRangeKind",
    "JsonlTrajectoryReader",
    "LocalTrajectoryReader",
]

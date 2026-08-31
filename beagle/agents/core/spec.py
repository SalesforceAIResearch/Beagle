"""Declarative specs for building agents.

A spec is the *serializable description* of an agent (what the config YAML holds);
an :class:`~beagle.agents.core.base.Agent` is the *live object* the factory builds from
it. Keeping them separate lets a whole run be reconstructed from config alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from beagle.types import AgentRole, Transparency


@dataclass
class ModelSpec:
    """The model *endpoint* metadata — ``name`` is the model the agent runs against.

    ``name`` feeds the agent's ``--model``. ``provider`` / ``api_base`` / ``params``
    are optional model-plane metadata (kept for parity with the upstream config).

    The agent's **gateway routing** — its ``--provider`` and the creds env forwarded
    into the container — is NOT declared here: it lives in ``agent.config`` (e.g.
    monet's ``monet_args`` + ``forward_env``). The harbor M+N shim serializes
    ``agent.config`` but drops model-block details, so routing must ride in the config.
    """

    name: str
    provider: str = ""
    api_base: str | None = None
    #: The model's native reasoning level (validated per-family at the config layer); how it's
    #: applied is the agent's business (codex/claude_code → ``--effort``; cursor uses the slug).
    reasoning_effort: str | None = None
    params: dict = field(default_factory=dict)


@dataclass
class AgentSource:
    """The exact version of an agent's code — a git repo at a ref. This is θ.

    ``repo`` @ ``ref`` is the source of truth (an external repo we branch for
    evolution). ``root`` is filled in once the runner *materializes* the checkout
    (host worktree, or a clone inside the task container); adapters read ``root``
    to build/run and never clone it themselves.

    A run selects a version by ``ref``: a baseline ref is un-evolved, a candidate
    ref/branch produced by an evolver is evolved. Same repo, different ref.
    """

    #: Git URL (source of truth). Defaults to the agent's ``REPO`` when unset.
    repo: str = ""
    #: Branch / commit / tag. ``None`` = the agent's baseline; a candidate ref = evolved.
    ref: str | None = None
    #: Path within the checkout used to invoke/configure (e.g. ``bin/monet.js`` or a YAML).
    entrypoint: str = ""
    #: Local materialized checkout, populated by the runner; ``None`` until materialized.
    root: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSpec:
    """Everything needed to build one agent.

    Attributes
    ----------
    name:
        Registry key (``"monet"``, ``"cursor"``, ...). Selects the class.
    role:
        Role assigned for this run (usage, not intrinsic); ``None`` until a
        Trainer assigns one, validated against the agent's capabilities.
    transparency:
        Override the class default (rare); ``None`` uses the class default.
    model:
        The LLM the agent uses. ``None`` for agents whose model is fixed elsewhere.
    config:
        Agent-specific free-form dict, validated by each concrete agent.
    source:
        The pinned version to run (repo + ref) for evolvable agents; ``None`` uses
        the agent's default repo at its baseline ref.
    preset:
        Named bundle of defaults.
    """

    name: str
    role: AgentRole | None = None
    transparency: Transparency | None = None
    model: ModelSpec | None = None
    config: dict = field(default_factory=dict)
    source: AgentSource | None = None
    preset: str | None = None


__all__ = ["ModelSpec", "AgentSource", "AgentSpec"]

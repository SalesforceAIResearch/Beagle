"""Canonical run configuration — the schema of record for a run's YAML.

A single **evaluation run** and a full **evolution run** share one explicit, stable
contract, so operators meet no surprises:

- :class:`RunConfig` — a single evaluation run:
  ``model`` / ``agent`` / ``benchmark`` / ``runtime`` / ``parallelism``.
- :class:`BeagleConfig` — the *evolution* run, composed from the **same** sub-schemas
  (``evolvee`` / ``evolver`` are :class:`AgentConfig`; ``benchmark`` / ``runtime`` /
  ``parallelism`` are shared) plus ``algorithm`` / ``data`` / ``trainer``. An evolution
  run derives a :class:`RunConfig` per candidate via :meth:`BeagleConfig.run_config`.

**The detector.** Every model is pydantic v2 with ``extra="forbid"`` — so an unknown,
renamed, or typo'd field, or a missing required one, is a hard error at load time
(the drift-guard). The only free-form escape hatches are the nested
``dict`` fields (``model.params``, ``agent.config``, ``benchmark.options``,
``algorithm.hparams``, ``data``, ``trainer``), each validated by its own consumer.

Run ``python -m beagle.config <file.yaml>`` (add ``--evolve`` for an evolution config)
to validate a file and print a pass/drift report.

The pydantic models are the *contract*; the live objects the rest of the framework
builds from are the plain-dataclass specs in :mod:`beagle.agents.core.spec` and
:mod:`beagle.benchmarks.base`. Each model's ``to_spec()`` bridges the two, so the
contract can tighten without churning runtime code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from beagle.agents.core.spec import AgentSource, AgentSpec, ModelSpec
from beagle.benchmarks.base import BenchmarkSpec
from beagle.rollout.runtime.config import RuntimeConfig as RuntimeSettings
from beagle.types import AgentRole, Transparency


class _Base(BaseModel):
    """Every config model forbids unknown keys — this is the drift detector."""

    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Base):
    """The model endpoint — ``name`` is the model the agent runs (`--model`).

    ``provider`` / ``api_base`` / ``params`` are optional model-plane metadata. The
    agent's gateway routing (its ``--provider`` + creds) is NOT here — it goes in
    ``agent.config`` (``monet_args`` + ``forward_env``), the only place the harbor shim
    preserves. Creds live in the environment, never in the config.

    ``reasoning_effort`` is the model's native reasoning level, passed by CLI backends that take
    it per-model (codex → GPT, claude_code → Claude). The **valid set differs by model family**,
    so use the family subclass (:class:`GptModelConfig` / :class:`ClaudeModelConfig`) to get it
    validated; the base accepts any string. ``cursor`` is the exception — it encodes effort in the
    model *slug* (``gpt-5.5-high``), so leave ``reasoning_effort`` unset for a cursor model.
    """

    name: str
    provider: str = ""
    api_base: str | None = None
    reasoning_effort: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    def to_spec(self) -> ModelSpec:
        return ModelSpec(**self.model_dump())


class GptModelConfig(ModelConfig):
    """A GPT / OpenAI-reasoning model (what ``codex`` runs). Narrows ``reasoning_effort`` to the
    OpenAI-style set so an invalid level fails loud."""

    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None


class ClaudeModelConfig(ModelConfig):
    """A Claude / Anthropic-reasoning model (what ``claude_code`` runs). Narrows
    ``reasoning_effort`` to Claude's set (no ``minimal``; adds ``max``)."""

    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None


class AgentSourceConfig(_Base):
    """The agent's code version — a git ``repo`` @ ``ref`` (θ). A typed form of the
    untyped inline ``agent.config.agent_source``."""

    repo: str = ""
    ref: str | None = None
    entrypoint: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_spec(self) -> AgentSource:
        return AgentSource(**self.model_dump())


class AgentConfig(_Base):
    """An agent — ``name`` / ``preset`` / ``config`` plus our typed ``source`` and
    per-agent ``model`` (the latter used by evolution's two agents; in a
    :class:`RunConfig` the top-level ``model`` applies instead).

    ``config`` is the free-form agent-specific dict; an inline ``agent_source`` may live
    here and is honored when ``source`` is unset.
    """

    name: str
    role: AgentRole | None = None
    transparency: Transparency | None = None
    model: ModelConfig | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    source: AgentSourceConfig | None = None
    preset: str | None = None

    def _resolved_source(self) -> AgentSourceConfig | None:
        if self.source is not None:
            return self.source
        inline = self.config.get("agent_source")  # untyped inline form
        if isinstance(inline, dict):
            repo = inline.get("repo") or inline.get("repo_url", "")
            return AgentSourceConfig(repo=repo, ref=inline.get("ref"),
                                     entrypoint=inline.get("entrypoint", ""))
        return None

    def to_spec(self, *, default_role: AgentRole | None = None,
                model: ModelConfig | None = None) -> AgentSpec:
        src = self._resolved_source()
        chosen_model = model or self.model
        cfg = {k: v for k, v in self.config.items() if k != "agent_source"}
        # Lift agent-adapter knobs nested in the inline ``agent_source`` (the coding-bench / DarwinX
        # driver shape) to top-level ``config`` — where the adapter reads them. Without this they're
        # silently dropped (agent_source is stripped; AgentSource carries only repo/ref/entrypoint),
        # so e.g. the clone runs unauthenticated ("could not read Username") or lands at the wrong
        # path (``cd: <container_path>: No such file or directory``) → the agent never runs.
        inline = self.config.get("agent_source")
        if isinstance(inline, dict):
            for key in ("token_env", "container_path"):
                if inline.get(key) and key not in cfg:
                    cfg[key] = inline[key]
        return AgentSpec(
            name=self.name,
            role=self.role if self.role is not None else default_role,
            transparency=self.transparency,
            model=chosen_model.to_spec() if chosen_model is not None else None,
            config=cfg,
            source=src.to_spec() if src is not None else None,
            preset=self.preset,
        )


class BenchmarkConfig(_Base):
    """Benchmark + task selection.

    Task selection is ``task_ids`` (a list restricts + orders; ``None`` = the full set)
    with ``exclude_task_ids`` applied after, and ``num_samples`` for pass@k. There is
    **no ``limit``**/first-N knob — name the tasks you want.
    """

    name: str
    dataset: str | None = None
    split: str | None = None
    task_ids: list[str] | None = None
    exclude_task_ids: list[str] | None = None
    num_samples: int = Field(default=1, ge=1)
    namespace: str | None = None
    tag: str = "main"
    registry: str | None = None
    image: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    def to_spec(self) -> BenchmarkSpec:
        return BenchmarkSpec(**self.model_dump())


class RuntimeConfig(_Base):
    """Where trials run — ``local`` or ``xrlenv-cluster``.

    Cluster connection fields are optional; unset, the runtime falls back to the
    ``XRLENV_*`` env.
    """

    kind: str = "local"
    grpc_host: str | None = None
    grpc_port: int | None = None
    token: str | None = None
    run_id: str | None = None
    artifact_root: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    def to_settings(self) -> RuntimeSettings:
        d = self.model_dump()
        if d.get("artifact_root") is not None:
            d["artifact_root"] = Path(d["artifact_root"])
        return RuntimeSettings(**d)


class RetryPolicy(_Base):
    """Task-level retry for a run (both harness shapes; the DarwinX path uses the driver's
    own retry knobs instead).

    Two independent layers, mirroring the xrlenv benchmark sweeps:

    * ``infra`` — re-run a trial ONLY on an infra-transient error (capacity / control-plane /
      node blip), up to N times. Content outcomes (agent timeout, verifier failure,
      ``resolved=False``, a rate limit) are NEVER infra-retried, so eval signal is never
      re-rolled. Rides harbor's ``RetryConfig`` on the harbor path; a per-task loop on docker.
    * ``content`` — re-run UNRESOLVED tasks up to N times; a task counts solved if ANY attempt
      passes. Absorbs flakes (e.g. a rate-limited agent). Owned by the Runner (harness-agnostic).

    ``timeout_multiplier`` scales the harness agent/verifier timeouts (harbor path).
    """

    infra: int = Field(default=0, ge=0)
    content: int = Field(default=0, ge=0)
    timeout_multiplier: float = Field(default=1.0, gt=0)


class AlgorithmConfig(_Base):
    """Evolution algorithm + hyper-parameters."""

    name: str = "darwinx"
    hparams: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_name(cls, v: Any) -> Any:
        return {"name": v} if isinstance(v, str) else v


class RunConfig(_Base):
    """One evaluation run: ``model`` / ``agent`` / ``benchmark`` / ``runtime`` /
    ``parallelism``.

    The top-level ``model`` is the agent's model; ``agent`` carries no model of its own
    here.
    """

    model: ModelConfig
    agent: AgentConfig
    benchmark: BenchmarkConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    parallelism: int = Field(default=1, ge=1)
    #: Patch-EVALUATION concurrency, separate from agent/patch-GENERATION concurrency
    #: (``parallelism``). Generation is LLM-bound; SWE-bench eval spins one test container per
    #: instance (I/O-bound), so it usually wants a HIGHER fan-out. Only a two-phase grader that
    #: batch-evaluates patches reads it (SWE-bench → swebench's ``max_workers``); other graders
    #: ignore it. ``None`` → fall back to ``parallelism``.
    parallelism_eval_patches: int | None = Field(default=None, ge=1)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        return cls.model_validate(d)

    def agent_spec(self) -> AgentSpec:
        return self.agent.to_spec(model=self.model)

    def benchmark_spec(self) -> BenchmarkSpec:
        return self.benchmark.to_spec()

    def runtime_settings(self) -> RuntimeSettings:
        return self.runtime.to_settings()


class BeagleConfig(_Base):
    """A whole evolution run — the same eval contract (``benchmark`` / ``runtime`` /
    ``parallelism``) plus ``evolvee`` + ``evolver`` (:class:`AgentConfig`), ``algorithm``,
    ``data`` (the :class:`~beagle.data.DataMixture` config), and ``trainer`` loop knobs.

    ``evolvee``/``evolver`` default to the EVOLVEE/EVOLVER roles. Derive a per-candidate
    :class:`RunConfig` with :meth:`run_config`.
    """

    evolvee: AgentConfig
    evolver: AgentConfig
    benchmark: BenchmarkConfig | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    parallelism: int = Field(default=1, ge=1)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    data: dict[str, Any] = Field(default_factory=dict)
    trainer: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _stamp_roles(self) -> BeagleConfig:
        # Default each slot to its role, then reject an explicit mismatch — a swapped
        # config (``evolvee.role: evolver``) would otherwise silently run the evolvee as
        # an evolver. The role a slot fills is fixed by the slot, not free to override.
        for slot, want in (("evolvee", AgentRole.EVOLVEE), ("evolver", AgentRole.EVOLVER)):
            cfg = getattr(self, slot)
            if cfg.role is None:
                cfg.role = want
            elif cfg.role is not want:
                raise ValueError(
                    f"{slot}.role must be {want.value!r} (or omitted), got {cfg.role.value!r}"
                )
        return self

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BeagleConfig:
        return cls.model_validate(d)

    def run_config(self, *, agent: AgentConfig | None = None) -> RunConfig:
        """The eval :class:`RunConfig` for one candidate (default: the evolvee).

        The candidate's own ``model`` becomes the run's top-level model. Requires a
        ``benchmark`` (evolution must say what candidates are scored on).
        """
        if self.benchmark is None:
            raise ValueError("BeagleConfig.run_config needs a `benchmark` to evaluate on")
        cand = agent or self.evolvee
        if cand.model is None:
            raise ValueError(f"agent {cand.name!r} has no model to evaluate with")
        # Deep-copy so a derived RunConfig never shares sub-objects with this config —
        # otherwise mutating one candidate's run (or looping over candidates) would leak
        # back into the source (the evolution trainer calls this per candidate).
        return RunConfig(
            model=cand.model.model_copy(deep=True),
            agent=cand.model_copy(deep=True),
            benchmark=self.benchmark.model_copy(deep=True),
            runtime=self.runtime.model_copy(deep=True),
            parallelism=self.parallelism,
        )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a YAML mapping, got {type(raw).__name__}")
    return raw


def load_config(path: str | Path) -> RunConfig:
    """Load + validate a single-run eval config YAML → :class:`RunConfig`."""
    return RunConfig.model_validate(_load_yaml(path))


def load_evolve_config(path: str | Path) -> BeagleConfig:
    """Load + validate an evolution-run config YAML."""
    return BeagleConfig.model_validate(_load_yaml(path))


def _main(argv: list[str] | None = None) -> int:
    """Validate a config file and report drift — the detector as a CLI."""
    import argparse

    from pydantic import ValidationError

    p = argparse.ArgumentParser(prog="python -m beagle.config",
                                description="Validate a run config against the canonical schema.")
    p.add_argument("path", help="config YAML to validate")
    p.add_argument("--evolve", action="store_true", help="validate as an evolution config")
    args = p.parse_args(argv)
    loader = load_evolve_config if args.evolve else load_config
    try:
        loader(args.path)
    except (ValidationError, ValueError, OSError) as e:
        print(f"INVALID {args.path}:\n{e}")
        return 1
    print(f"OK {args.path} — valid {'evolution' if args.evolve else 'run'} config")
    return 0


__all__ = [
    "ModelConfig", "AgentSourceConfig", "AgentConfig", "BenchmarkConfig", "RuntimeConfig",
    "RetryPolicy", "AlgorithmConfig", "RunConfig", "BeagleConfig", "load_config",
    "load_evolve_config", "GptModelConfig", "ClaudeModelConfig",
]


if __name__ == "__main__":
    raise SystemExit(_main())

"""``Trainer`` — the user entrypoint (design-plot "Trainer" box).

The Trainer wires the four pieces together and runs the evolution loop, exactly
the way ``torch``'s training loop wires model + optimizer + dataloader::

    trainer = bgl.Trainer(
        evolvee=bgl.agents.build("monet"),          # θ: the harness being evolved
        evolver=bgl.agents.build("cursor"),         # the mutation operator
        algorithm=bgl.algorithms.build("darwinx"),  # the optimizer
    )
    best = trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

**Role is assigned here, and validated against capabilities.** The same agent
(e.g. monet) can be an evolvee in one run and an evolver in another; the Trainer
just checks the agent you passed for a role actually supports it — an evolvee must
be ``Runnable + Evolvable``, an evolver must be an ``Editor``.

Responsibilities kept deliberately thin — the Trainer *orchestrates*; the real
work lives in the algorithm (selection), the evolver (variation), and the runner
(evaluation through native harnesses).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from beagle.algorithms.base import Candidate, EvolveAlgorithm
from beagle.rollout.runner import Runner
from beagle.types import AgentRole

if TYPE_CHECKING:
    from beagle.agents.core.base import Agent
    from beagle.config import RunConfig
    from beagle.data.dataset import TaskDataset


class Trainer:
    """Runs harness evolution: evolvee + evolver + algorithm over a task mixture.

    Parameters
    ----------
    evolvee:
        The agent whose source is optimized (its ``source()`` is θ). Must be
        ``Runnable`` (to be scored) and ``Evolvable`` (to be mutated).
    evolver:
        The coding agent the algorithm drives to edit source. Must be an ``Editor``.
    algorithm:
        The evolution algorithm (the optimizer).
    trainer_config:
        Loop-level knobs: ``max_generations``, ``val_every``, ``patience``, ...
    """

    @classmethod
    def from_config(cls, cfg: Any, *, trainer_config: dict[str, Any] | None = None) -> Trainer:
        """Assemble a Trainer from an :class:`~beagle.config.BeagleConfig` — build the
        evolvee/evolver/algorithm from its blocks and keep the config so ``fit`` can derive
        the per-candidate eval :class:`RunConfig`. Pure wiring; no loop logic."""
        import beagle as bgl

        evolvee = bgl.agents.build(cfg.evolvee.to_spec(default_role=AgentRole.EVOLVEE))
        evolver = bgl.agents.build(cfg.evolver.to_spec(default_role=AgentRole.EVOLVER))
        algorithm = bgl.algorithms.build(cfg.algorithm.name, **dict(cfg.algorithm.hparams))
        t = cls(evolvee=evolvee, evolver=evolver, algorithm=algorithm,
                trainer_config=trainer_config or dict(cfg.trainer))
        t._xcfg = cfg
        return t

    def __init__(
        self,
        *,
        evolvee: Agent,
        evolver: Agent,
        algorithm: EvolveAlgorithm,
        trainer_config: dict[str, Any] | None = None,
    ) -> None:
        if not evolvee.can_be_evolvee():
            raise TypeError(
                f"{evolvee.name!r} cannot be an evolvee: needs Runnable + Evolvable, "
                f"has {sorted(c.value for c in evolvee.capabilities)}"
            )
        if not evolver.can_be_evolver():
            raise TypeError(
                f"{evolver.name!r} cannot be an evolver: needs Editor, "
                f"has {sorted(c.value for c in evolver.capabilities)}"
            )
        # Stamp the assigned role (usage) on each agent's spec.
        evolvee.spec.role = AgentRole.EVOLVEE
        evolver.spec.role = AgentRole.EVOLVER

        self.evolvee = evolvee
        self.evolver = evolver
        self.algorithm = algorithm
        self.config = trainer_config or {}
        self._xcfg: Any = None   # the BeagleConfig, when built via from_config (for run_config)

    # -- internals -----------------------------------------------------------

    def _run_config(self, dataset: TaskDataset | None = None) -> RunConfig | None:
        """The per-candidate eval :class:`RunConfig` (model/benchmark/runtime).

        Two sources, in order: the evolution config if the Trainer was built via
        :meth:`from_config` and it names a benchmark; else — the **direct (PyTorch-UX) path** —
        derived from the dataset's own ``benchmark_spec`` + the evolvee's model/source + the
        trainer's runtime. ``None`` if neither names a benchmark (or the evolvee has no model);
        the algorithm surfaces that when it evaluates, so a launch-only algorithm can still run.
        """
        if self._xcfg is not None and self._xcfg.benchmark is not None:
            return self._xcfg.run_config()
        if dataset is None or self.evolvee.spec.model is None:
            return None
        spec = getattr(dataset, "benchmark_spec", None)
        specs = list(getattr(dataset, "benchmark_specs", None) or [])
        if spec is None and not specs:
            return None
        # A mixture has no single ``benchmark_spec`` (it has several), so derive from the
        # plural and let the RunConfig carry the whole list. Without this the mixture path
        # resolves to no benchmark and the algorithm refuses to score anything.
        return self._derive_run_config(spec or specs[0], specs=specs)

    def _derive_run_config(self, benchmark_spec: Any, *, specs: list[Any] | None = None) -> RunConfig:
        """Build a :class:`RunConfig` from live objects (no YAML): the evolvee's model/source, the
        dataset's benchmark, and the trainer's runtime. The specs mirror the config models
        field-for-field, so this is the inverse of ``*.to_spec()`` (dropping ``AgentSource.root``,
        which is a runtime-only field)."""
        import dataclasses as dc

        from beagle.config import (AgentConfig, AgentSourceConfig, BenchmarkConfig, ModelConfig,
                                    RunConfig, RuntimeConfig)

        ev = self.evolvee.spec
        source = None
        if ev.source is not None:
            keep = ("repo", "ref", "entrypoint", "metadata")
            source = AgentSourceConfig(**{k: v for k, v in dc.asdict(ev.source).items() if k in keep})
        primary = BenchmarkConfig(**dc.asdict(benchmark_spec))
        mixture = None
        if specs and len(specs) > 1:
            # Primary first: RunConfig requires it, so a mixture-unaware reader of
            # ``.benchmark`` still sees a benchmark that is actually being run.
            rest = [s for s in specs if s is not benchmark_spec]
            mixture = [primary] + [BenchmarkConfig(**dc.asdict(s)) for s in rest]
        return RunConfig(
            model=ModelConfig(**dc.asdict(ev.model)),
            agent=AgentConfig(name=ev.name, config=dict(ev.config or {}), source=source),
            benchmark=primary,
            benchmarks=mixture,
            runtime=RuntimeConfig(**dict(self.config.get("runtime") or {"kind": "local"})),
            parallelism=int(self.config.get("parallelism", 1)),
        )

    def _make_evaluate(self, dataset: TaskDataset, runner: Runner, config: RunConfig | None):
        """Turn the *data* into an ``evaluate(candidate)`` callable — bind the evolvee to the
        candidate's source, roll out on ``dataset`` via the Runner, record fitness in place.
        This is the Trainer's whole contribution: it hands this to ``algorithm.evolve``."""
        def evaluate(candidate: Candidate) -> None:
            if config is None:
                raise RuntimeError("cannot evaluate a candidate: the evolution config names no "
                                   "`benchmark` to score on")
            agent = self.evolvee.with_source(candidate.source)  # type: ignore[attr-defined]
            result = runner.run(agent, dataset, config=config)
            candidate.score, candidate.results = result.score, result.results
        return evaluate

    # -- entrypoint ----------------------------------------------------------

    def fit(
        self,
        *,
        train_dataset: TaskDataset,
        val_dataset: TaskDataset | None = None,
    ) -> Candidate | None:
        """Combine data + algorithm and run the search → the best harness.

        Thin by design: build ``evaluate`` (and optional ``val`` scorer) from the data via
        the Runner, then hand off to :meth:`EvolveAlgorithm.evolve` — the algorithm owns the
        loop. No generational cadence is imposed here.
        """
        cfg = self._run_config(train_dataset)
        runner = Runner(parallelism=int(self.config.get("parallelism", 1)))
        evaluate = self._make_evaluate(train_dataset, runner, cfg)
        val = self._make_evaluate(val_dataset, runner, cfg) if val_dataset is not None else None
        return self.algorithm.evolve(
            evaluate=evaluate, evolvee=self.evolvee, evolver=self.evolver, val=val, config=cfg,
        )

    def dry_run(self, *, train_dataset: TaskDataset,
                val_dataset: TaskDataset | None = None) -> RunConfig | None:
        """Resolve the run and print the plan — **no evolver, no rollouts, no spend**. The gate
        to eyeball before :meth:`fit`. Returns the derived eval :class:`RunConfig` (``None`` if
        the data names no benchmark)."""
        cfg = self._run_config(train_dataset)
        ev = self.evolvee.spec
        src = ev.source
        out = ["", "beagle dry-run — resolved plan (no spend)", "─" * 46,
               f"evolvee   : {ev.name}"
               + (f" @ {src.repo}#{src.ref}" if src and src.repo else "")
               + (f"  (model {ev.model.name})" if ev.model else "  (no model)"),
               f"evolver   : {self.evolver.spec.name}"
               + (f"  (model {self.evolver.spec.model.name})" if self.evolver.spec.model else ""),
               f"algorithm : {self.algorithm.__class__.__name__.lower()}  hparams={dict(self.algorithm.hparams)}"]
        if cfg is not None:
            b = cfg.benchmark
            tasks = b.task_ids if b.task_ids is not None else "(full set)"
            out += [f"benchmark : {b.name}  tasks={tasks}  num_samples={b.num_samples}",
                    f"runtime   : {cfg.runtime.kind}  parallelism={cfg.parallelism}"]
        else:
            out.append("benchmark : none resolved — the data names no benchmark, so candidates "
                       "can't be scored (set evolvee.model + a benchmark dataset).")
        n_val = len(val_dataset) if val_dataset is not None else 0
        out += [f"data      : {len(train_dataset)} train / {n_val} val task(s)",
                "─" * 46, "next: re-run with fit() to launch (this spends).", ""]
        print("\n".join(out))
        return cfg


__all__ = ["Trainer"]

"""DarwinX — a quality-diversity evolution algorithm with verification gates.

DarwinX is the first concrete :class:`EvolveAlgorithm`. It searches the space of
harness variants with a genealogy tree and a quality-diversity archive, and — the
part that makes evolved harnesses actually generalize — it guards every accepted
mutation behind a multi-layer verification pipeline rather than a single pass-rate
number.

Selection & variation:

* **Genealogy tree** of candidate harnesses, each scored on a task subset.
* **Quality-diversity archive** of specialist "stepping stones": a variant that
  cracks a hard task is preserved even when it regresses elsewhere, then later
  recombined into a net-positive generalist by a merge step.
* **Parent selection** as a pluggable strategy (mixed exploit/explore, or
  LLM-guided from per-node archive cards).
* **Novelty scoring** (behavioral k-NN over task-outcome vectors) to keep the
  population diverse rather than collapsing onto one lineage.

Guarded acceptance (why evolved harnesses don't overfit the eval):

* **Scope / structural filter** — reject edits that hard-code task names, narrow
  conditionals to the probe set, or otherwise special-case the benchmark.
* **Reward-hacking & tamper detection** — catch edits that game the grader or
  touch eval/test files.
* **Verifier-based fitness** — score trajectory *quality* by criteria
  decomposition and blend it with raw pass-rate, so a lucky pass doesn't win.
* **Best-of-N / consensus probes** — re-run candidate and canary tasks and decide
  by consensus before committing.
* **Cross-model / cross-benchmark transfer gates** — require an improvement to
  hold under a different model and on a held-out benchmark.
* A **reasoned ACCEPT / ARCHIVE / REJECT verdict** over all of the above decides
  whether a child is KEPT, ARCHIVED (as a stepping stone), or REJECTED.

Persistent **long-term memory** accumulates campaign-wide lessons that feed back
into the evolver's proposals.

:meth:`evolve` launches the vendored driver: it prepares the import path, emits the driver's
campaign config, injects the evolver Editor (seam B), constructs the ``PipelineConfig``, runs
the pipeline, and reads the winner back. The mechanical routing lives in :mod:`._launch`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from beagle.algorithms.base import Candidate, EvolveAlgorithm
from beagle.algorithms.darwinx.config import DarwinXConfig
from beagle.algorithms.registry import register

if TYPE_CHECKING:
    from beagle.agents.core.base import Agent
    from beagle.algorithms.base import Evaluate
    from beagle.config import RunConfig


@register("darwinx")
class DarwinX(EvolveAlgorithm):
    """Quality-diversity harness evolution with verification gates.

    DarwinX is a **launch-its-own-loop** algorithm: it drives a distributed supervisor over a
    genealogy tree + QD archive with a multi-layer verification pipeline (its own code, hosted
    — see ``notes/darwinX-migration/darwinx-dropin-contract.md``). So it implements :meth:`evolve`
    by launching that driver, NOT via a generational cadence imposed by the framework.

    Its knobs are the typed :class:`DarwinXConfig` (launch paths, driver loop/eval knobs, and the
    ``ATELIER_*`` verification-gate knobs) — ``build("darwinx", repo_root=…, max_loop_iters=1)``
    validates against it, and an unknown knob fails loud.
    """

    Config = DarwinXConfig

    config: DarwinXConfig   # narrow the base annotation for typed field access below

    def evolve(
        self,
        *,
        evaluate: Evaluate,
        evolvee: Agent,
        evolver: Agent,
        val: Evaluate | None = None,
        config: RunConfig | None = None,
    ) -> Candidate | None:
        """Launch the vendored DarwinX driver with the injected seams and return the best node.

        On launch (per the drop-in contract): put the vendored driver + seam-A ``runner.run``
        shim on the import path; emit the driver's campaign config from evolvee/evolver/benchmark
        (seam C); inject ``evolver`` into the ``meta_agent`` shim (seam B) so the proposer calls
        this Editor — and stash the evolver spec in the config so worker subprocesses rebuild it
        with no env var; build the ``PipelineConfig`` and run it; read the winner back from the
        genealogy DB. The Runner backs candidate evaluation through seam A.
        """
        from beagle.algorithms.darwinx import _launch
        from beagle.algorithms.darwinx import meta_agent as shim

        if config is None:
            raise ValueError(
                "DarwinX.evolve needs a benchmark to score candidates on — the evolution config "
                "names none. Set `benchmark` in the config."
            )
        repo_root, reports_root, campaign = self._launch_paths()
        worktree_parent = str(self.config.worktree_parent or (Path(reports_root) / "worktrees"))

        # Point the driver at the evolvee checkout + our worktree paths BEFORE importing it
        # (worktree.py reads its path overrides at import time).
        canonical = _launch.ensure_canonical_clone(repo_root, self.config.evolvee_checkout)
        # The driver pushes candidate branches to the checkout's `origin`; align it to the config's
        # repo (the manifest) so push and clone target the same fork — manifest as source of truth.
        changed = _launch.align_git_origin(canonical, evolvee.source().repo)
        if changed:
            print(f"[darwinx] aligned {canonical} origin: {changed[0]} -> {changed[1]}")
        _launch.prepare_worktree_env(repo_root, worktree_parent)
        # Non-interactive GitHub auth for the driver's host-side git (fetch θ / push candidates),
        # from the evolvee's token_env (the manifest's, e.g. GH_TOKEN).
        _launch.prepare_git_auth(evolvee.spec.config.get("token_env"))
        _launch.prepare_import_path()
        from self_evolve import pool, tree                     # noqa: E402 (path prepared above)
        from self_evolve.pipeline import PipelineConfig, SelfEvolvePipeline  # noqa: E402

        config_path = _launch.emit_campaign_config(
            evolvee=evolvee, evolver=evolver, run_config=config,
            dest=Path(reports_root) / f"{campaign}.campaign.yaml",
        )
        # Seam B: the run's evolver Editor, in-process. Worker subprocesses reconstruct it from
        # the EVOLVER_SPEC_KEY block in config_path (config-based injection — no env var).
        shim.set_editor(evolver)
        # Bucket-2 config → the driver's env channel (config wins): runtime/seed from the run,
        # and this algorithm's own gate/verifier knobs (ATELIER_*) from its typed config.
        import os

        _launch.prepare_runtime_env(config, evolvee.source())
        _launch.prepare_task_subset_env(config)          # benchmark task-subset selectors → env
        os.environ.update(self.config.to_driver_env())

        cfg = _launch.build_pipeline_config(
            PipelineConfig, campaign=campaign, reports_root=reports_root, repo_root=repo_root,
            config_path=config_path, evolver_model=evolver.spec.model.name if evolver.spec.model else "",
            subset_tasks=config.benchmark.task_ids, hparams=dict(self.hparams),
        )
        # The two-phase supervisor sequence: score the root baseline (bootstrap precursor), then
        # evolve — a single run can't pick its own unscored root.
        _launch.run_campaign(SelfEvolvePipeline, cfg, tree)
        return _launch.read_best(tree, pool, cfg, evolvee_source=evolvee.source())

    def _launch_paths(self) -> tuple[str, str, str]:
        """Resolve the launch-infra paths from the typed config (config-based, no env var).

        Only ``repo_root`` is required — a local dir whose ``monet_code`` subdir is the evolvee
        clone the driver worktrees, and under which the genealogy DB lands. ``reports_root``
        defaults to it (DB + campaign config next to the run); ``campaign`` defaults to ``"darwinx"``.
        """
        c = self.config
        if not c.repo_root:
            raise ValueError(
                "DarwinX.evolve needs launch paths in algorithm config: `repo_root` unset. It's a "
                "local dir whose `monet_code` subdir is the evolvee clone the driver evolves and "
                "under which the run's genealogy DB lands (`reports_root` defaults to it)."
            )
        return str(c.repo_root), str(c.reports_root or c.repo_root), str(c.campaign)


__all__ = ["DarwinX"]

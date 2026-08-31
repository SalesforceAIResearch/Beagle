"""``DarwinXConfig`` — DarwinX's typed knobs (the algorithm's config subclass).

This is the single home for everything DarwinX exposes: launch infra (where worktrees + the
genealogy DB live, which evolvee checkout to link in), the driver's loop/eval knobs, the runtime
/ cluster knobs, and the ~50 verification-gate knobs. ``extra='forbid'`` makes an unknown/typo'd
knob a load-time error — so the wall of ``DARWINX_GATE_*`` / ``DARWINX_EVAL_*`` env vars the vendored
driver reads becomes one validated, agent-agnostic config surface (bucket 2: config is the single
source of truth, translated to the driver's env at one boundary — see
``notes/darwinX-migration/darwinx-env-inventory.md``).

Three groups reach the vendored driver differently:

* **Loop/eval knobs** (``max_loop_iters``, ``parent_strategy``, ``mini_eval_k_samples``, …) share
  their names with the driver's ``PipelineConfig`` fields, so ``_launch.build_pipeline_config``
  picks them straight off ``hparams``. They default to ``None`` here → the driver's own default
  applies unless set. (The driver may still let a ``DARWINX_EVAL_*`` env var *override* one of these
  at its own call site; we do not also emit env for them — the config field is the source.)
* **Gate / verifier / runtime knobs** the driver reads from the environment (``DARWINX_GATE_*``,
  ``DARWINX_EVAL_*``, ``DARWINX_EVOLVE_TRACE_QC*``, ``DARWINX_TRACE_*``); :meth:`to_driver_env`
  translates the *set* fields into that env (config → env at one boundary). Only explicitly-set
  knobs are emitted, so the driver's defaults stand otherwise. Booleans emit ``"1"``/``"0"``
  (accepted by every truthy predicate the driver uses).
* **Credentials** (gateway keys/URLs for the verifier / equivalence / trace-QC models) are NOT
  here — they stay in ``.env`` (bucket 1). These fields carry only the model *name* + provider.

To type another env knob: add a field + one line to the matching ``_ENV_*`` table below. The
field name is the agent-agnostic config surface (no ``DARWINX_GATE_``/``DARWINX_EVAL_`` prefix, no
``monet`` hardcoding); the table value is the driver's actual env var.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, model_validator

from beagle.algorithms.base import AlgorithmConfig


class DarwinXConfig(AlgorithmConfig):
    """Typed configuration for :class:`DarwinX`."""

    model_config = ConfigDict(extra="forbid")   # unknown knob → load-time error (drift guard)

    # -- launch infra (consumed by _launch; never forwarded to the driver as-is) --------------
    #: Local dir where per-candidate worktrees + the genealogy DB live (``<repo_root>/monet_code``
    #: is the evolvee clone the proposer edits).
    repo_root: str | None = None
    #: Where the run's DB + emitted campaign config are written.
    reports_root: str | None = None
    #: A local checkout of the evolvee experiment copy; linked in at ``<repo_root>/monet_code``.
    evolvee_checkout: str | None = None
    #: Parent dir for per-pipeline worktrees (default: ``<reports_root>/worktrees``).
    worktree_parent: str | None = None
    #: Campaign id (namespaces the genealogy DB + reports).
    campaign: str = "darwinx"

    # -- evolvee eval knobs (agent-agnostic; the evolvee under eval, whatever agent it is) ------
    #: The evolvee's reasoning effort during eval (none|low|medium|high|max). Without it the driver
    #: runs the evolvee at its default ``none`` (minimal reasoning) — a large quality drop on hard
    #: tasks. Rides :meth:`to_driver_env` → the driver's ``DARWINX_EVAL_EFFORT`` (a run knob).
    #: (Evolvee ``max_turns`` / ``timeout`` are NOT here: the driver reads them from its own config
    #: fields, not env, so they need a one-line upstream env-read hook — a coordinate-upstream item,
    #: not a silent no-op field.)
    evolvee_effort: Literal["none", "low", "medium", "high", "max"] | None = None

    # -- driver loop / eval knobs (names mirror the driver's PipelineConfig; None = its default) --
    max_loop_iters: int | None = None
    n_failure_tasks: int | None = None
    parent_strategy: str | None = None
    guard_enabled: bool | None = None
    guard_strict: bool | None = None
    subset_label: str | None = None
    subset_eval_n_attempts: int | None = None
    fullset_eval_n_attempts: int | None = None
    mini_eval_k_samples: int | None = None

    # -- runtime / cluster fan-out (DARWINX_EVAL_* / XRLENV_ run knobs, via to_driver_env) --------
    #: Comma-separated cluster slice(s) to claim, and the claim strategy / variance band.
    clusters: str | None = None
    cluster_claim: str | None = None
    claim_variance_band: str | None = None
    #: Cluster group id for run isolation (XRLENV_-prefixed, but a per-run *choice* → config).
    xrlenv_group_id: str | None = None
    #: Absorb agent-timeout / transient-infra task failures instead of failing the eval.
    absorb_timeouts: bool | None = None
    infra_retries: int | None = None
    skip_docker_prune: bool | None = None
    #: best|mean — how a multi-sample fullset score reduces.
    fullset_metric: str | None = None
    #: Cap on the persisted final-archive size.
    final_archive_max_bytes: int | None = None

    # -- verification: gates (DARWINX_GATE_*, via to_driver_env) ------------------------------------
    gate_enabled: bool | None = None
    gate_escalate_k: int | None = None
    gate_regression_tol: int | None = None
    cross_bench_gate: bool | None = None
    cross_bench_margin: float | None = None
    # -- verification: multi-benchmark mixture (DARWINX_GATE_MIXTURE_*) -----------------------------
    #: Score candidates across a *mixture* of benchmarks rather than one, using each
    #: benchmark's own baseline and replicate sigma so the numbers are comparable. The
    #: cross_bench_* gate above holds out a whole benchmark; this one evolves on several at
    #: once and guards each of them.
    mixture_gate: bool | None = None
    #: ``@/path.json`` or inline JSON: per-benchmark weight, baseline, sd and held-out tasks.
    #: Produced by the baseline calibration run — the baselines must come from measurement,
    #: not from the campaign, or a candidate moves its own yardstick.
    mixture_spec: str | None = None
    #: How far one benchmark may fall, in its own sigma, before the floor vetoes.
    mixture_tol_sd: float | None = None
    #: Absolute drop always tolerated, for a benchmark whose measured sd came out implausibly
    #: small on few replicates.
    mixture_min_abs_drop: float | None = None
    #: Per-benchmark gains stop counting above this many sigma. Makes the aggregate concave so
    #: improving three benchmarks beats spiking one — without it the mixture selects the
    #: specialist it exists to prevent, and the floor stays silent because nothing regressed.
    mixture_gain_cap_sd: float | None = None
    #: Samples per task when the floor measures a benchmark. 1 unless a benchmark's replicate
    #: sigma is so wide that one draw cannot see the effect being gated.
    mixture_k: int | None = None
    #: Tasks per benchmark the floor scores at full size. The spec carries whole corpora because
    #: that is what the baselines measured; scoring all of them per candidate would spend the
    #: campaign on one node. A fixed seeded sample keeps parent and child comparable — on a sample
    #: this size, the variance between two task draws otherwise swamps the difference between two
    #: agents.
    mixture_gate_tasks: int | None = None
    #: Tasks per benchmark in the cheap first stage. A healthy candidate clears the screen and pays
    #: nothing more; only a benchmark that looks like it regressed is re-measured at full size.
    mixture_gate_screen_tasks: int | None = None
    #: Seed for the gate's task order, so a resumed campaign scores the same tasks.
    mixture_gate_seed: int | None = None
    #: ``{benchmark: n}`` overrides for the two sizes above. Trials do not cost the same — a
    #: Deep-SWE trial is about three SWE-V trials and runs at lower parallelism — so a uniform
    #: sample spends most of the floor's budget on its slowest member. Unequal samples are sound
    #: because each benchmark's tolerance is computed from the sample it actually got: a smaller
    #: sample widens its own tolerance and cannot veto on noise it lacks the power to see. Unequal
    #: samples sharing one tolerance would not be.
    mixture_gate_tasks_per_benchmark: dict[str, int] = {}
    mixture_gate_screen_tasks_per_benchmark: dict[str, int] = {}

    #: What a node's recorded score *is*, and therefore what parent selection ranks on.
    #:
    #: ``panel`` (the driver's default) is the pass rate on the node's eval panel — a single
    #: benchmark. ``mixture`` is the baseline-normalised, spike-capped aggregate over every member
    #: of the mixture spec, so a candidate that improves one benchmark and leaves the others flat
    #: out-ranks one that changes nothing. Under ``panel`` the other benchmarks can only veto
    #: through the regression floor; they cannot direct the search.
    #:
    #: Scores become sigmas rather than rates: the root is 0.0 by construction (fitness is defined
    #: against the baselines and the root *is* the baseline), a neutral child is ~0.0, and a
    #: regression is negative. Requires ``mixture_gate`` — see :meth:`validate`.
    node_score: Literal["panel", "mixture"] | None = None
    #: Score every node of the campaign on ONE shared panel. Not optional alongside
    #: ``defer_node_full_eval`` — see :meth:`validate`.
    fixed_eval_panel: bool | None = None
    #: Size of that shared panel, sampled deterministically from the campaign subset. 0 (the
    #: driver's default) means the whole subset.
    eval_panel_size: int | None = None
    equivalence_gate: bool | None = None
    anti_cheat: bool | None = None
    collective_knowledge: bool | None = None
    contract_guided: bool | None = None

    # -- verification: fitness / novelty / archive / LTM ---------------------------------------
    fitness_alpha: float | None = None
    novelty_m: int | None = None
    hybrid_archive: bool | None = None
    archive_max_regressions: int | None = None
    ltm_enabled: bool | None = None
    ltm_max_entries: int | None = None

    # -- verification: scope / preservation ----------------------------------------------------
    scope_mode: str | None = None
    additive_scope: bool | None = None
    require_extension: bool | None = None
    preserve_extend: bool | None = None
    max_deletions: int | None = None

    # -- verification: equivalence / probes ----------------------------------------------------
    equivalence_model: str | None = None
    equivalence_provider: str | None = None
    equivalence_n_adversarial: int | None = None
    equivalence_n_votes: int | None = None
    equivalence_probe_k: int | None = None
    equivalence_reprobe: bool | None = None
    equivalence_require_extension: bool | None = None
    probe_k_samples: int | None = None

    # -- verification: held-out gate -----------------------------------------------------------
    heldout_benchmark: str | None = None
    heldout_dataset: str | None = None
    heldout_tasks: str | None = None
    heldout_baseline: str | None = None
    heldout_k: int | None = None

    # -- verification: verifier / reasoned verdict / signals -----------------------------------
    verifier_model: str | None = None
    verifier_provider: str | None = None
    verifier_criteria_profile: str | None = None
    verifier_sgv: bool | None = None
    reasoned_verdict: bool | None = None
    reasoned_verdict_model: str | None = None
    predictions_enabled: bool | None = None
    progress_signal: bool | None = None
    sibling_pool_enabled: bool | None = None
    sibling_pool_k: int | None = None
    trace_digest_enabled: bool | None = None
    bestof2_contrast: bool | None = None
    self_contrast_sources: str | None = None
    specialist_contract: str | None = None
    teacher_timeout_s: int | None = None
    reports_subdir: str | None = None

    # -- robustness ----------------------------------------------------------------------------
    absorb_transient_infra: bool | None = None
    defer_node_full_eval: bool | None = None
    routed_code: bool | None = None

    # -- trace-QC / trace_analyzer (DARWINX_EVOLVE_TRACE_QC* + DARWINX_TRACE_*) -------------------
    #: Enable the rule-based trace-QC digest fed to the analyze prompt, and its LLM phase. (The
    #: LLM phase also needs the trace_analyzer LLM client pointed at the gateway — a separate
    #: wiring item; setting ``trace_qc_llm`` here only flips the driver's switch.)
    trace_qc: bool | None = None
    trace_qc_llm: bool | None = None
    trace_qc_config: str | None = None
    trace_analyzer_model: str | None = None
    trace_analyzer_llm_max_retries: int | None = None
    trace_analyzer_llm_backoff_s: float | None = None

    # -- config field → the driver's env var, grouped by emitted type --------------------------
    _ENV_BOOL = {
        "gate_enabled": "DARWINX_GATE_ENABLED",
        "cross_bench_gate": "DARWINX_GATE_CROSS_BENCH_GATE",
        "mixture_gate": "DARWINX_GATE_MIXTURE_GATE",
        "equivalence_gate": "DARWINX_GATE_EQUIVALENCE_GATE_ENABLED",
        "anti_cheat": "DARWINX_GATE_ANTI_CHEAT_ENABLED",
        "collective_knowledge": "DARWINX_GATE_COLLECTIVE_KNOWLEDGE",
        "contract_guided": "DARWINX_GATE_CONTRACT_GUIDED",
        "hybrid_archive": "DARWINX_GATE_HYBRID_ARCHIVE",
        "ltm_enabled": "DARWINX_GATE_LTM_ENABLED",
        "additive_scope": "DARWINX_GATE_ADDITIVE_SCOPE",
        "require_extension": "DARWINX_GATE_REQUIRE_EXTENSION",
        "preserve_extend": "DARWINX_GATE_PRESERVE_EXTEND",
        "equivalence_reprobe": "DARWINX_GATE_EQUIVALENCE_REPROBE",
        "equivalence_require_extension": "DARWINX_GATE_EQUIVALENCE_REQUIRE_EXTENSION",
        "verifier_sgv": "DARWINX_GATE_VERIFIER_SGV",
        "reasoned_verdict": "DARWINX_GATE_REASONED_VERDICT",
        "predictions_enabled": "DARWINX_GATE_PREDICTIONS_ENABLED",
        "progress_signal": "DARWINX_GATE_PROGRESS_SIGNAL",
        "sibling_pool_enabled": "DARWINX_GATE_SIBLING_POOL_ENABLED",
        "trace_digest_enabled": "DARWINX_GATE_TRACE_DIGEST_ENABLED",
        "bestof2_contrast": "DARWINX_GATE_BESTOF2_CONTRAST",
        "absorb_transient_infra": "DARWINX_GATE_ABSORB_TRANSIENT_INFRA",
        "defer_node_full_eval": "DARWINX_GATE_DEFER_NODE_FULL_EVAL",
        "fixed_eval_panel": "DARWINX_GATE_FIXED_EVAL_PANEL",
        "routed_code": "DARWINX_GATE_ROUTED_CODE",
        "absorb_timeouts": "DARWINX_EVAL_ABSORB_TIMEOUTS",
        "skip_docker_prune": "DARWINX_EVAL_SKIP_DOCKER_PRUNE",
        "trace_qc": "DARWINX_EVOLVE_TRACE_QC",
        "trace_qc_llm": "DARWINX_EVOLVE_TRACE_QC_LLM",
    }
    _ENV_INT = {
        "novelty_m": "DARWINX_GATE_NOVELTY_M",
        "gate_escalate_k": "DARWINX_GATE_ESCALATE_K",
        "mixture_k": "DARWINX_GATE_MIXTURE_K",
        "mixture_gate_tasks": "DARWINX_GATE_MIXTURE_GATE_TASKS",
        "mixture_gate_screen_tasks": "DARWINX_GATE_MIXTURE_GATE_SCREEN_TASKS",
        "mixture_gate_seed": "DARWINX_GATE_MIXTURE_GATE_SEED",
        "node_score": "DARWINX_GATE_NODE_SCORE",
        "eval_panel_size": "DARWINX_GATE_EVAL_PANEL_SIZE",
        "gate_regression_tol": "DARWINX_GATE_REGRESSION_TOL",
        "archive_max_regressions": "DARWINX_GATE_ARCHIVE_MAX_REGRESSIONS",
        "ltm_max_entries": "DARWINX_GATE_LTM_MAX_ENTRIES",
        "max_deletions": "DARWINX_GATE_MAX_DELETIONS",
        "probe_k_samples": "DARWINX_GATE_PROBE_K_SAMPLES",
        "sibling_pool_k": "DARWINX_GATE_SIBLING_POOL_K",
        "equivalence_n_adversarial": "DARWINX_GATE_EQUIVALENCE_N_ADVERSARIAL",
        "equivalence_n_votes": "DARWINX_GATE_EQUIVALENCE_N_VOTES",
        "equivalence_probe_k": "DARWINX_GATE_EQUIVALENCE_PROBE_K",
        "heldout_k": "DARWINX_GATE_HELDOUT_K",
        "teacher_timeout_s": "DARWINX_GATE_TEACHER_TIMEOUT_S",
        "infra_retries": "DARWINX_EVAL_INFRA_RETRIES",
        "final_archive_max_bytes": "DARWINX_EVAL_FINAL_ARCHIVE_MAX_BYTES",
        "trace_analyzer_llm_max_retries": "DARWINX_TRACE_LLM_MAX_RETRIES",
    }
    _ENV_FLOAT = {
        "fitness_alpha": "DARWINX_GATE_FITNESS_ALPHA",
        "cross_bench_margin": "DARWINX_GATE_CROSS_BENCH_MARGIN",
        "mixture_tol_sd": "DARWINX_GATE_MIXTURE_TOL_SD",
        "mixture_min_abs_drop": "DARWINX_GATE_MIXTURE_MIN_ABS_DROP",
        "mixture_gain_cap_sd": "DARWINX_GATE_MIXTURE_GAIN_CAP_SD",
        "trace_analyzer_llm_backoff_s": "DARWINX_TRACE_LLM_BACKOFF_S",
    }
    _ENV_STR = {
        "scope_mode": "DARWINX_GATE_SCOPE_MODE",
        "verifier_model": "DARWINX_GATE_VERIFIER_MODEL",
        "verifier_provider": "DARWINX_GATE_VERIFIER_PROVIDER",
        "verifier_criteria_profile": "DARWINX_GATE_VERIFIER_CRITERIA_PROFILE",
        "reasoned_verdict_model": "DARWINX_GATE_REASONED_VERDICT_MODEL",
        "equivalence_model": "DARWINX_GATE_EQUIVALENCE_MODEL",
        "equivalence_provider": "DARWINX_GATE_EQUIVALENCE_PROVIDER",
        "specialist_contract": "DARWINX_GATE_SPECIALIST_CONTRACT",
        "self_contrast_sources": "DARWINX_GATE_SELF_CONTRAST_SOURCES",
        "reports_subdir": "DARWINX_GATE_REPORTS_SUBDIR",
        "heldout_benchmark": "DARWINX_GATE_HELDOUT_BENCHMARK",
        "heldout_dataset": "DARWINX_GATE_HELDOUT_DATASET",
        "heldout_tasks": "DARWINX_GATE_HELDOUT_TASKS",
        "heldout_baseline": "DARWINX_GATE_HELDOUT_BASELINE",
        "mixture_spec": "DARWINX_GATE_MIXTURE_SPEC",
        "clusters": "DARWINX_EVAL_CLUSTERS",
        "cluster_claim": "DARWINX_EVAL_CLUSTER_CLAIM",
        "claim_variance_band": "DARWINX_EVAL_CLAIM_VARIANCE_BAND",
        "fullset_metric": "DARWINX_EVAL_FULLSET_METRIC",
        "xrlenv_group_id": "XRLENV_GROUP_ID",
        "trace_qc_config": "DARWINX_EVOLVE_TRACE_QC_CONFIG",
        "trace_analyzer_model": "DARWINX_TRACE_MODEL",
    }

    def to_driver_env(self) -> dict[str, str]:
        """The *set* gate/verifier/runtime knobs, translated to the driver's env. Only
        explicitly-set knobs are emitted, so the driver's own defaults stand otherwise. Booleans
        emit ``"1"``/``"0"`` (accepted by every truthy predicate the driver uses)."""
        env: dict[str, str] = {}
        for field, var in self._ENV_BOOL.items():
            val = getattr(self, field)
            if val is not None:
                env[var] = "1" if val else "0"
        for field, var in {**self._ENV_INT, **self._ENV_FLOAT, **self._ENV_STR}.items():
            val = getattr(self, field)
            if val is not None:
                env[var] = str(val)
        # The evolvee's reasoning effort during eval (the driver reads DARWINX_EVAL_EFFORT in
        # build_codingbench_config → the agent's ``--effort``); a run knob, not a gate.
        if self.evolvee_effort is not None:
            env["DARWINX_EVAL_EFFORT"] = self.evolvee_effort
        # Per-benchmark sample sizes. The driver looks for BASE_<BENCHMARK> before BASE, with the
        # benchmark name upper-cased and every non-alphanumeric character replaced by an underscore
        # (``deep-swe`` → ``DEEP_SWE``). Mirrored rather than imported so a hosted run does not
        # depend on a private helper in the vendored tree.
        for base, overrides in (
            ("DARWINX_GATE_MIXTURE_GATE_TASKS", self.mixture_gate_tasks_per_benchmark),
            ("DARWINX_GATE_MIXTURE_GATE_SCREEN_TASKS", self.mixture_gate_screen_tasks_per_benchmark),
        ):
            for benchmark, n in (overrides or {}).items():
                suffix = "".join(c if c.isalnum() else "_" for c in benchmark).upper()
                env[f"{base}_{suffix}"] = str(n)
        return env

    @model_validator(mode="after")
    def _reject_defer_without_a_shared_panel(self) -> "DarwinXConfig":
        """Reject configurations that are individually valid and jointly wrong.

        Two so far, both worth the method. ``defer_node_full_eval`` without
        ``fixed_eval_panel`` scores each node on its own claimed+guard set. The numerator barely
        moves between nodes, so a node that claims *fewer* tasks scores higher for purely arithmetic
        reasons, and parent selection ranks on exactly that score. The driver records the worked
        example: a campaign whose 11 scored nodes all solved the same two tasks, where the only score
        movement in 14 hours was the panel shrinking from 12 tasks to 6 — 0.167 to 0.333 — which the
        search read as improvement and inherited down the whole lineage.

        That is not a knob interaction anyone should have to remember, and nothing about the run
        looks wrong while it happens: scores rise. So it fails at load time instead.
        """
        if self.defer_node_full_eval and not self.fixed_eval_panel:
            raise ValueError(
                "defer_node_full_eval requires fixed_eval_panel: without a shared panel each node "
                "is scored on its own claimed+guard set, so claiming fewer tasks raises the score "
                "arithmetically and parent selection inherits the artefact. Set fixed_eval_panel=True "
                "(optionally with eval_panel_size)."
            )
        if self.node_score == "mixture" and not self.mixture_gate:
            raise ValueError(
                "node_score='mixture' requires mixture_gate=True: the multi-benchmark fitness is "
                "only computed by the gate, so with the gate off every node is scored 0.0 and "
                "parent selection ranks on ties. The campaign would run to completion and its "
                "search direction would be arbitrary."
            )
        if self.eval_panel_size is not None and self.eval_panel_size < 0:
            raise ValueError("eval_panel_size must be >= 0 (0 = use the whole campaign subset)")
        return self


__all__ = ["DarwinXConfig"]

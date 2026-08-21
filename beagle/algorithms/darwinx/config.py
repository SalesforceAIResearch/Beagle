"""``DarwinXConfig`` — DarwinX's typed knobs (the algorithm's config subclass).

This is the single home for everything DarwinX exposes: launch infra (where worktrees + the
genealogy DB live, which evolvee checkout to link in), the driver's loop/eval knobs, the runtime
/ cluster knobs, and the ~50 verification-gate knobs. ``extra='forbid'`` makes an unknown/typo'd
knob a load-time error — so the wall of ``ATELIER_*`` / ``MONET_EVAL_*`` env vars the vendored
driver reads becomes one validated, agent-agnostic config surface (bucket 2: config is the single
source of truth, translated to the driver's env at one boundary — see
``notes/darwinX-migration/darwinx-env-inventory.md``).

Three groups reach the vendored driver differently:

* **Loop/eval knobs** (``max_loop_iters``, ``parent_strategy``, ``mini_eval_k_samples``, …) share
  their names with the driver's ``PipelineConfig`` fields, so ``_launch.build_pipeline_config``
  picks them straight off ``hparams``. They default to ``None`` here → the driver's own default
  applies unless set. (The driver may still let a ``MONET_EVAL_*`` env var *override* one of these
  at its own call site; we do not also emit env for them — the config field is the source.)
* **Gate / verifier / runtime knobs** the driver reads from the environment (``ATELIER_*``,
  ``MONET_EVAL_*``, ``SELF_EVOLVE_TRACE_QC*``, ``TRACE_ANALYZER_*``); :meth:`to_driver_env`
  translates the *set* fields into that env (config → env at one boundary). Only explicitly-set
  knobs are emitted, so the driver's defaults stand otherwise. Booleans emit ``"1"``/``"0"``
  (accepted by every truthy predicate the driver uses).
* **Credentials** (gateway keys/URLs for the verifier / equivalence / trace-QC models) are NOT
  here — they stay in ``.env`` (bucket 1). These fields carry only the model *name* + provider.

To type another env knob: add a field + one line to the matching ``_ENV_*`` table below. The
field name is the agent-agnostic config surface (no ``ATELIER_``/``MONET_EVAL_`` prefix, no
``monet`` hardcoding); the table value is the driver's actual env var.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict

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
    #: tasks. Rides :meth:`to_driver_env` → the driver's ``MONET_EVAL_EFFORT`` (a run knob).
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

    # -- runtime / cluster fan-out (MONET_EVAL_* / XRLENV_ run knobs, via to_driver_env) --------
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

    # -- verification: gates (ATELIER_*, via to_driver_env) ------------------------------------
    gate_enabled: bool | None = None
    gate_escalate_k: int | None = None
    gate_regression_tol: int | None = None
    cross_bench_gate: bool | None = None
    cross_bench_margin: float | None = None
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

    # -- trace-QC / trace_analyzer (SELF_EVOLVE_TRACE_QC* + TRACE_ANALYZER_*) -------------------
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
        "gate_enabled": "ATELIER_GATE_ENABLED",
        "cross_bench_gate": "ATELIER_CROSS_BENCH_GATE",
        "equivalence_gate": "ATELIER_EQUIVALENCE_GATE_ENABLED",
        "anti_cheat": "ATELIER_ANTI_CHEAT_ENABLED",
        "collective_knowledge": "ATELIER_COLLECTIVE_KNOWLEDGE",
        "contract_guided": "ATELIER_CONTRACT_GUIDED",
        "hybrid_archive": "ATELIER_HYBRID_ARCHIVE",
        "ltm_enabled": "ATELIER_LTM_ENABLED",
        "additive_scope": "ATELIER_ADDITIVE_SCOPE",
        "require_extension": "ATELIER_REQUIRE_EXTENSION",
        "preserve_extend": "ATELIER_PRESERVE_EXTEND",
        "equivalence_reprobe": "ATELIER_EQUIVALENCE_REPROBE",
        "equivalence_require_extension": "ATELIER_EQUIVALENCE_REQUIRE_EXTENSION",
        "verifier_sgv": "ATELIER_VERIFIER_SGV",
        "reasoned_verdict": "ATELIER_REASONED_VERDICT",
        "predictions_enabled": "ATELIER_PREDICTIONS_ENABLED",
        "progress_signal": "ATELIER_PROGRESS_SIGNAL",
        "sibling_pool_enabled": "ATELIER_SIBLING_POOL_ENABLED",
        "trace_digest_enabled": "ATELIER_TRACE_DIGEST_ENABLED",
        "bestof2_contrast": "ATELIER_BESTOF2_CONTRAST",
        "absorb_transient_infra": "ATELIER_ABSORB_TRANSIENT_INFRA",
        "defer_node_full_eval": "ATELIER_DEFER_NODE_FULL_EVAL",
        "routed_code": "ATELIER_ROUTED_CODE",
        "absorb_timeouts": "MONET_EVAL_ABSORB_TIMEOUTS",
        "skip_docker_prune": "MONET_EVAL_SKIP_DOCKER_PRUNE",
        "trace_qc": "SELF_EVOLVE_TRACE_QC",
        "trace_qc_llm": "SELF_EVOLVE_TRACE_QC_LLM",
    }
    _ENV_INT = {
        "novelty_m": "ATELIER_NOVELTY_M",
        "gate_escalate_k": "ATELIER_GATE_ESCALATE_K",
        "gate_regression_tol": "ATELIER_GATE_REGRESSION_TOL",
        "archive_max_regressions": "ATELIER_ARCHIVE_MAX_REGRESSIONS",
        "ltm_max_entries": "ATELIER_LTM_MAX_ENTRIES",
        "max_deletions": "ATELIER_MAX_DELETIONS",
        "probe_k_samples": "ATELIER_PROBE_K_SAMPLES",
        "sibling_pool_k": "ATELIER_SIBLING_POOL_K",
        "equivalence_n_adversarial": "ATELIER_EQUIVALENCE_N_ADVERSARIAL",
        "equivalence_n_votes": "ATELIER_EQUIVALENCE_N_VOTES",
        "equivalence_probe_k": "ATELIER_EQUIVALENCE_PROBE_K",
        "heldout_k": "ATELIER_HELDOUT_K",
        "teacher_timeout_s": "ATELIER_TEACHER_TIMEOUT_S",
        "infra_retries": "MONET_EVAL_INFRA_RETRIES",
        "final_archive_max_bytes": "MONET_EVAL_FINAL_ARCHIVE_MAX_BYTES",
        "trace_analyzer_llm_max_retries": "TRACE_ANALYZER_LLM_MAX_RETRIES",
    }
    _ENV_FLOAT = {
        "fitness_alpha": "ATELIER_FITNESS_ALPHA",
        "cross_bench_margin": "ATELIER_CROSS_BENCH_MARGIN",
        "trace_analyzer_llm_backoff_s": "TRACE_ANALYZER_LLM_BACKOFF_S",
    }
    _ENV_STR = {
        "scope_mode": "ATELIER_SCOPE_MODE",
        "verifier_model": "ATELIER_VERIFIER_MODEL",
        "verifier_provider": "ATELIER_VERIFIER_PROVIDER",
        "verifier_criteria_profile": "ATELIER_VERIFIER_CRITERIA_PROFILE",
        "reasoned_verdict_model": "ATELIER_REASONED_VERDICT_MODEL",
        "equivalence_model": "ATELIER_EQUIVALENCE_MODEL",
        "equivalence_provider": "ATELIER_EQUIVALENCE_PROVIDER",
        "specialist_contract": "ATELIER_SPECIALIST_CONTRACT",
        "self_contrast_sources": "ATELIER_SELF_CONTRAST_SOURCES",
        "reports_subdir": "ATELIER_REPORTS_SUBDIR",
        "heldout_benchmark": "ATELIER_HELDOUT_BENCHMARK",
        "heldout_dataset": "ATELIER_HELDOUT_DATASET",
        "heldout_tasks": "ATELIER_HELDOUT_TASKS",
        "heldout_baseline": "ATELIER_HELDOUT_BASELINE",
        "clusters": "MONET_EVAL_CLUSTERS",
        "cluster_claim": "MONET_EVAL_CLUSTER_CLAIM",
        "claim_variance_band": "MONET_EVAL_CLAIM_VARIANCE_BAND",
        "fullset_metric": "MONET_EVAL_FULLSET_METRIC",
        "xrlenv_group_id": "XRLENV_GROUP_ID",
        "trace_qc_config": "SELF_EVOLVE_TRACE_QC_CONFIG",
        "trace_analyzer_model": "TRACE_ANALYZER_MODEL",
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
        # The evolvee's reasoning effort during eval (the driver reads MONET_EVAL_EFFORT in
        # build_codingbench_config → the agent's ``--effort``); a run knob, not a gate.
        if self.evolvee_effort is not None:
            env["MONET_EVAL_EFFORT"] = self.evolvee_effort
        return env


__all__ = ["DarwinXConfig"]

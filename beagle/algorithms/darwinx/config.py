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
    #: best|avg — how a full-set score reduces. ``fullset_eval_n_attempts`` controls
    #: best-of-N only; ``avg`` uses the vendored driver's fixed ``fullset_eval_k_samples``
    #: (currently 5, not yet exposed on this typed surface).
    fullset_metric: str | None = None
    #: Cap on the persisted final-archive size.
    final_archive_max_bytes: int | None = None

    #: Churn (added+removed lines, per file) allowed in a file that already exists under
    #: :data:`shared_core_paths` before the pre-eval guard reverts the commit. This is the rule
    #: that stops a broad rewrite of logic every task already depends on, so it stays tight.
    shared_core_churn_budget: int | None = None
    #: The same budget for a file the commit CREATES, which is a different risk and needs a
    #: different number: a new file changes no existing behavior on its own, and only runs
    #: because some existing file calls it -- and that call site is an ordinary modification
    #: already charged to :data:`shared_core_churn_budget`. Charging creations the modification
    #: budget left the proposer no surface to write a capability on: A1's first four nodes each
    #: produced a new capability file in ``src/session/`` (67, 347, 152 and 185 lines) with a
    #: 19-34 line hook that was inside the modification budget every time, plus tests and ZERO
    #: deletions, and all four were reverted. Still a budget, not an exemption, because a large
    #: new file does run on every task.
    new_core_file_churn_budget: int | None = None
    #: Dir holding a ``run.json`` of the baseline's per-trial outcomes, which is the ONLY input to
    #: the failure-mode theme digest (:data:`failure_theme`). Without it that flag reads as enabled
    #: and silently yields nothing: the digest reads ``$BASELINE_LOGS/run.json``, the hosted
    #: launcher never set this, so the proposer was never told which theme dominates. Note a
    #: beagle ``run.json`` is NOT directly usable -- it carries only aggregates, while the
    #: classifier iterates ``per_task_results`` -- so point this at the output of
    #: ``scripts/theme_input_from_run.py`` rather than at a run dir. Validated below, because a
    #: wrong path here is invisible at runtime.
    baseline_logs: str | None = None

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

    #: Compatibility field for the intended panel-vs-mixture score choice. The current vendored
    #: launch path records mixture-gate fitness but does not consume ``DARWINX_GATE_NODE_SCORE``
    #: when parent selection ranks nodes, so ``mixture`` is NOT a working multi-benchmark search
    #: objective yet. Keep it out of new configs; ``mixture_gate`` remains useful as a regression
    #: floor. ``mixture`` requires ``mixture_gate`` so old configs at least fail closed when the
    #: gate itself is absent.
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

    # -- per-stage proposer wall-clock caps -----------------------------------------------------
    # TWO independent caps exist and the SMALLER one binds:
    #   1. this stage cap (driver-side), and
    #   2. the evolver agent's own ``timeout:`` in its harness block.
    # Both report the same way -- "timeout after <n>s" -- so the number in the log tells you which
    # one fired. Measured 2026-08-26: the fast loop's agent cap of 1800s ate two of its three
    # iterations (implement, then analyze) while its stage caps sat far higher, and the driver's own
    # docstring records the same class of loss ("a quarter of mini_smoke_0801b's iterations were
    # lost to the analyze stage hitting the old hardcoded 1800s"). Leave the agent cap generous and
    # steer per stage from here. Note review's driver default is only 900s.
    analyze_timeout_s: int | None = None
    implement_timeout_s: int | None = None
    review_timeout_s: int | None = None

    # -- which paths are which SURFACE, for THIS evolvee --------------------------------------
    # darwinx's guards and the proposer's instructions are both keyed to path substrings, and the
    # defaults name monet's files. On any other evolvee they match nothing: measured 2026-08-26 on
    # opencode, an 80-line additive edit to packages/opencode/src/session/ drew zero violations
    # while the identical edit to monet's src/query/loop.js was bounced pre-eval. Worse, the
    # proposer was told its only skill surface was src/core/bundled-skills.js -- a file opencode
    # does not have -- so it spent both of its iterations editing the system prompt, the one shape
    # that is paid for by every task and creditable to none.
    skill_path_markers: str | None = None
    #: Paths whose files the evolvee LOADS as plugins (a hook implementation, not prose). Its own
    #: surface because it is additive like a skill but ships executable code, and it only measures
    #: anything if the harness mounts the root these paths name.
    plugin_paths: str | None = None
    shared_core_paths: str | None = None
    global_bundle_paths: str | None = None
    global_edit_paths: str | None = None
    prompt_paths: str | None = None
    skill_target_doc: str | None = None
    #: How this evolvee loads a plugin, naming the hooks it actually has. Required with
    #: plugin_paths: classifying a surface the proposer was never told about is a surface no
    #: candidate will ever target.
    plugin_target_doc: str | None = None
    evolvee_label: str | None = None
    core_path_doc: str | None = None
    prompt_rule_doc: str | None = None

    # -- campaign size: how many NODES, not how many iterations -------------------------------
    # One pipeline run produces exactly ONE node. `max_loop_iters` is how many attempts that
    # single node may make internally, so raising it does not grow the population -- it only
    # gives one node more tries. With total_steps unset the campaign is a single lineage of
    # length one, in which parent selection has nothing to choose between, lineage depth never
    # passes 1 (so PRUNE at depth>=3 and CONSOLIDATE at depth>=2 are unreachable), a scheduled
    # compaction can never fire because there is never a second accepted node, and recombination
    # is impossible because a merge needs two complementary children. Default None = 1 node,
    # which keeps an existing single-node run byte-identical.
    total_steps: int | None = None
    #: Attempt a recombination every N evolve steps (None/0 = never). Needs total_steps > N,
    #: and only fires when two complementary scored children actually exist.
    merge_every: int | None = None

    # -- node variants: what KIND of edit a pipeline is allowed to make ------------------------
    # Three classes, resolved per pipeline by independent lotteries on a hash of the pipeline id
    # (prune drawn first, then consolidate, so a node is never both; anything undrawn is ADDITIVE):
    #   ADDITIVE     -- may only add, bound by the additive + extension contracts. The default,
    #                   and the only class active when both switches below are off, which is what
    #                   keeps an additive-only control arm byte-identical.
    #   PRUNE        -- may only delete, and only what this campaign's lineage added; its verdict
    #                   rule demands a deletion-dominated diff.
    #   CONSOLIDATE  -- exempt from both contracts and may rewrite pre-evolve code, because the
    #                   move that most improves a harness (folding accumulated special cases into
    #                   one general mechanism) is often line-neutral, so additive nodes reject it
    #                   for deleting and prune nodes reject it for adding.
    prune_enabled: bool | None = None
    prune_rate: float | None = None
    prune_min_lineage: int | None = None
    consolidate_enabled: bool | None = None
    consolidate_rate: float | None = None
    consolidate_rate_late: float | None = None
    consolidate_late_depth: int | None = None
    consolidate_min_lineage: int | None = None
    #: SCHEDULED compaction, which the lotteries above cannot deliver on this search: they are
    #: gated on lineage depth, depth only grows when something extends the same line, and when
    #: improvements are rare parent selection keeps returning the same node, so the tree grows
    #: wide and the depth condition is never met (observed: "CONSOLIDATE threshold 2: NOT YET
    #: REACHABLE" for a whole run). After this many accepted ADDITIVE nodes on a lineage, the
    #: next node on it is a CONSOLIDATE. 0 = trigger off, pure lottery.
    consolidate_force_k: int | None = None
    #: Also force a compaction when the harness grew since the root while fitness stayed flat.
    consolidate_on_bloat: bool | None = None
    #: Probability of branching from the DEEPEST eligible node instead of the best-ranked one.
    #: Without this, depth-gated mechanisms can stay unreachable for an entire campaign.
    parent_deepest_p: float | None = None
    #: Require a new extension (skill/plugin) to match its cue on at least
    #: ``skill_fire_min`` tasks OUTSIDE the claim pool before it may be PROMOTEd; unproven
    #: extensions are ARCHIVEd instead. Off by default: it makes the gate strictly stricter,
    #: so it is an experiment arm rather than a default.
    skill_fire_gate: bool | None = None
    skill_fire_min: int | None = None
    #: The complexity measurement CONSOLIDATE's accept rule ("capability held, complexity down")
    #: is judged on. The driver's defaults are mini-swe-agent's tree (``src/minisweagent/**/*.py``),
    #: which matches zero files in any other evolvee -- so a consolidation would be judged against
    #: an empty measurement. Comma-separated globs, relative to the evolvee checkout.
    complexity_code_globs: str | None = None
    complexity_prompt_globs: str | None = None

    # -- gate knobs that previously had no config surface at all -------------------------------
    archive_all: bool | None = None
    #: Keep LOSSY specialists: a variant that newly solves a claimed task even if it
    #: regresses a guard is archived with its real solved/regressed sets instead of being
    #: reverted. These are the "specialist" nodes the report's node-type figure names, and
    #: they are the raw material recombination consumes -- with this off, a merge has
    #: complementary parents only by luck.
    qd_archive: bool | None = None
    #: avg@k pass-rate strictly above which a claimed task counts as cracked for the
    #: archive gate (driver default 0.0 = any seed). This is an ARCHIVE floor, not a
    #: promotion bar; archiving liberally is safe because archived nodes are excluded
    #: from best/tip.
    qd_solved_threshold: float | None = None
    #: Cross-task theme synthesis: label each trial's failure mode and inject the dominant
    #: theme into the proposer, so the search aims at a systemic bottleneck rather than
    #: per-task patches. The report describes this as a first-class component.
    failure_theme: bool | None = None
    knowledge_gate: bool | None = None
    fractional_gate: bool | None = None
    regression_margin: float | None = None
    confirm_before_parent: bool | None = None
    confirm_k_samples: int | None = None

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
        "prune_enabled": "DARWINX_GATE_PRUNE_ENABLED",
        "consolidate_enabled": "DARWINX_GATE_CONSOLIDATE_ENABLED",
        "consolidate_on_bloat": "DARWINX_GATE_CONSOLIDATE_ON_BLOAT",
        "skill_fire_gate": "DARWINX_GATE_SKILL_FIRE_GATE",
        "archive_all": "DARWINX_GATE_ARCHIVE_ALL",
        "qd_archive": "DARWINX_GATE_QD_ARCHIVE",
        "failure_theme": "DARWINX_GATE_FAILURE_THEME",
        "knowledge_gate": "DARWINX_GATE_KNOWLEDGE_GATE",
        "fractional_gate": "DARWINX_GATE_FRACTIONAL_GATE",
        "confirm_before_parent": "DARWINX_GATE_CONFIRM_BEFORE_PARENT",
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
        "shared_core_churn_budget": "DARWINX_GATE_SHARED_CORE_CHURN_BUDGET",
        "new_core_file_churn_budget": "DARWINX_GATE_NEW_CORE_FILE_CHURN_BUDGET",
        # Supervisor-level, not gate-level: read by _launch to size the campaign.
        # They travel through the env because the vendored PipelineConfig has no
        # field for them -- total_steps was a CLI arg of the un-vendored supervisor.
        "total_steps": "DARWINX_EVOLVE_TOTAL_STEPS",
        "merge_every": "DARWINX_EVOLVE_MERGE_EVERY",
        "gate_regression_tol": "DARWINX_GATE_REGRESSION_TOL",
        "archive_max_regressions": "DARWINX_GATE_ARCHIVE_MAX_REGRESSIONS",
        "ltm_max_entries": "DARWINX_GATE_LTM_MAX_ENTRIES",
        "max_deletions": "DARWINX_GATE_MAX_DELETIONS",
        "probe_k_samples": "DARWINX_GATE_PROBE_K_SAMPLES",
        "prune_min_lineage": "DARWINX_GATE_PRUNE_MIN_LINEAGE",
        "consolidate_late_depth": "DARWINX_GATE_CONSOLIDATE_LATE_DEPTH",
        "consolidate_min_lineage": "DARWINX_GATE_CONSOLIDATE_MIN_LINEAGE",
        "consolidate_force_k": "DARWINX_GATE_CONSOLIDATE_FORCE_K",
        "skill_fire_min": "DARWINX_GATE_SKILL_FIRE_MIN",
        "confirm_k_samples": "DARWINX_GATE_CONFIRM_K_SAMPLES",
        "analyze_timeout_s": "DARWINX_GATE_ANALYZE_TIMEOUT_S",
        "implement_timeout_s": "DARWINX_GATE_IMPLEMENT_TIMEOUT_S",
        "review_timeout_s": "DARWINX_GATE_REVIEW_TIMEOUT_S",
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
        "prune_rate": "DARWINX_GATE_PRUNE_RATE",
        "consolidate_rate": "DARWINX_GATE_CONSOLIDATE_RATE",
        "consolidate_rate_late": "DARWINX_GATE_CONSOLIDATE_RATE_LATE",
        "parent_deepest_p": "DARWINX_GATE_PARENT_DEEPEST_P",
        "qd_solved_threshold": "DARWINX_GATE_QD_SOLVED_THRESHOLD",
        "regression_margin": "DARWINX_GATE_REGRESSION_MARGIN",
        "cross_bench_margin": "DARWINX_GATE_CROSS_BENCH_MARGIN",
        "mixture_tol_sd": "DARWINX_GATE_MIXTURE_TOL_SD",
        "mixture_min_abs_drop": "DARWINX_GATE_MIXTURE_MIN_ABS_DROP",
        "mixture_gain_cap_sd": "DARWINX_GATE_MIXTURE_GAIN_CAP_SD",
        "trace_analyzer_llm_backoff_s": "DARWINX_TRACE_LLM_BACKOFF_S",
    }
    _ENV_STR = {
        "scope_mode": "DARWINX_GATE_SCOPE_MODE",
        "complexity_code_globs": "DARWINX_GATE_COMPLEXITY_CODE_GLOBS",
        "skill_path_markers": "DARWINX_GATE_SKILL_PATH_MARKERS",
        "plugin_paths": "DARWINX_GATE_PLUGIN_PATHS",
        "shared_core_paths": "DARWINX_GATE_SHARED_CORE_PATHS",
        "global_bundle_paths": "DARWINX_GATE_GLOBAL_BUNDLE_PATHS",
        "global_edit_paths": "DARWINX_GATE_GLOBAL_EDIT_PATHS",
        "prompt_paths": "DARWINX_GATE_PROMPT_PATHS",
        "skill_target_doc": "DARWINX_GATE_SKILL_TARGET_DOC",
        "plugin_target_doc": "DARWINX_GATE_PLUGIN_TARGET_DOC",
        "evolvee_label": "DARWINX_GATE_EVOLVEE_LABEL",
        "core_path_doc": "DARWINX_GATE_CORE_PATH_DOC",
        "prompt_rule_doc": "DARWINX_GATE_PROMPT_RULE_DOC",
        "complexity_prompt_globs": "DARWINX_GATE_COMPLEXITY_PROMPT_GLOBS",
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
        "baseline_logs": "BASELINE_LOGS",
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
                "node_score='mixture' requires mixture_gate=True. Note that node_score is a "
                "compatibility field in this release: the gate records mixture fitness, but the "
                "current parent selector does not consume it as its search objective."
            )
        for name in ("analyze_timeout_s", "implement_timeout_s", "review_timeout_s"):
            val = getattr(self, name)
            if val is not None and val <= 0:
                raise ValueError(
                    f"{name} must be > 0: the driver ignores a non-positive stage cap and silently "
                    "falls back to its default, so a typo here would look like it took effect."
                )
        if bool(self.plugin_paths) != bool(self.plugin_target_doc):
            raise ValueError(
                "plugin_paths and plugin_target_doc must be declared together: paths without the "
                "doc classify a surface the proposer was never told exists, and the doc without "
                "the paths advertises a surface whose candidates are then judged as core edits."
            )
        if self.prompt_paths and not (self.skill_target_doc and self.prompt_rule_doc):
            raise ValueError(
                "prompt_paths requires both skill_target_doc and prompt_rule_doc: bouncing the "
                "prompt surface without naming where the capability should go instead leaves the "
                "proposer with no valid surface, and reverting an edit the instructions never "
                "forbade spends an iteration teaching it a rule it was not given."
            )
        if self.consolidate_enabled and not self.complexity_code_globs:
            raise ValueError(
                "consolidate_enabled requires complexity_code_globs: a CONSOLIDATE node is accepted "
                "for holding capability while measurably lowering complexity, and that measurement "
                "defaults to mini-swe-agent's own tree (src/minisweagent/**/*.py). For any other "
                "evolvee those globs match zero files, so the accept rule would be judged against an "
                "empty measurement -- the run looks fine and the verdict rests on nothing. Point "
                "complexity_code_globs at the evolvee's source (e.g. 'packages/opencode/src/**/*.ts')."
            )
        for name in ("prune_rate", "consolidate_rate", "consolidate_rate_late"):
            val = getattr(self, name)
            if val is not None and not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be within [0, 1] (got {val})")
        for name in ("prune_min_lineage", "consolidate_min_lineage", "consolidate_late_depth",
                     "consolidate_force_k", "skill_fire_min"):
            val = getattr(self, name)
            if val is not None and val < 0:
                raise ValueError(f"{name} must be >= 0 (got {val})")
        if self.total_steps is not None and self.total_steps < 1:
            raise ValueError("total_steps must be >= 1 (it counts NODES, not iterations)")
        if self.merge_every is not None and self.merge_every < 0:
            raise ValueError("merge_every must be >= 0 (0 = never recombine)")
        # A recombination needs two complementary scored children to exist first, so a
        # merge cadence at or above the node budget can never fire. Refuse it rather than
        # let the config read as if recombination were enabled.
        if self.merge_every and (self.total_steps or 1) <= self.merge_every:
            raise ValueError(
                f"merge_every={self.merge_every} needs total_steps > {self.merge_every} "
                f"(got total_steps={self.total_steps}); otherwise no merge can ever run")
        if self.parent_deepest_p is not None and not 0.0 <= self.parent_deepest_p <= 1.0:
            raise ValueError(
                f"parent_deepest_p must be a probability in [0, 1] (got {self.parent_deepest_p})")
        # A scheduled compaction is only reachable if CONSOLIDATE is switched on at all;
        # otherwise the knob reads as configured and silently never fires, which is the exact
        # class of failure the node-variant validation exists to catch.
        if self.consolidate_force_k and not self.consolidate_enabled:
            raise ValueError(
                f"consolidate_force_k={self.consolidate_force_k} requires consolidate_enabled: "
                "true, or the trigger can never fire")
        if self.consolidate_on_bloat and not self.consolidate_enabled:
            raise ValueError(
                "consolidate_on_bloat requires consolidate_enabled: true")
        if self.skill_fire_min is not None and not self.skill_fire_gate:
            raise ValueError(
                "skill_fire_min only has an effect with skill_fire_gate: true")
        # The theme digest has exactly one input and fails safe to "" on any problem, so a missing
        # or wrong-shaped baseline_logs makes failure_theme read as enabled while feeding the
        # proposer nothing. Measured: arm A1 launched with failure_theme: true and no BASELINE_LOGS,
        # and the digest was empty for the whole run. Check the shape here, at load.
        if self.failure_theme:
            if not self.baseline_logs:
                raise ValueError(
                    "failure_theme: true requires baseline_logs (the digest reads "
                    "$BASELINE_LOGS/run.json and yields nothing without it). Build one with "
                    "scripts/theme_input_from_run.py --run-dir <baseline run> --out <dir>")
            import json as _json
            import os as _os
            rj = _os.path.join(_os.path.expanduser(self.baseline_logs), "run.json")
            if not _os.path.exists(rj):
                raise ValueError(f"baseline_logs has no run.json: {rj}")
            try:
                rows = (_json.load(open(rj)) or {}).get("per_task_results")
            except (OSError, ValueError) as exc:
                raise ValueError(f"baseline_logs run.json is unreadable: {rj} ({exc})") from exc
            if not rows:
                raise ValueError(
                    f"baseline_logs run.json has no per_task_results: {rj}. A beagle run.json "
                    "carries only aggregates; convert it with scripts/theme_input_from_run.py")
        if self.confirm_k_samples is not None and self.confirm_k_samples < 1:
            raise ValueError("confirm_k_samples must be >= 1")
        # Leave the search some additive nodes. The two lotteries are independent and prune is
        # resolved first, so the additive share is (1 - prune_rate) * (1 - consolidate_rate) at the
        # late, higher consolidate rate. Drive that to nothing and the campaign can only subtract
        # and restructure what it has already got, which no amount of monitoring makes obvious.
        if self.prune_enabled or self.consolidate_enabled:
            p_prune = (self.prune_rate if self.prune_rate is not None else 0.2) if self.prune_enabled else 0.0
            p_cons = (
                (self.consolidate_rate_late if self.consolidate_rate_late is not None else 0.5)
                if self.consolidate_enabled else 0.0
            )
            additive_share = (1.0 - p_prune) * (1.0 - p_cons)
            if additive_share < 0.10:
                raise ValueError(
                    f"prune_rate={p_prune} with consolidate_rate_late={p_cons} leaves only "
                    f"{additive_share:.1%} of nodes additive; the campaign would have almost no way "
                    "to add capability. Lower one of the rates."
                )
        if self.eval_panel_size is not None and self.eval_panel_size < 0:
            raise ValueError("eval_panel_size must be >= 0 (0 = use the whole campaign subset)")
        return self


__all__ = ["DarwinXConfig"]

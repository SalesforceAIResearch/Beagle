"""Atelier — a verification + test-time-scaling layer on top of self_evolve.

Atelier extends ``self_evolve`` (the campaign-driven evolutionary search for
monet_code improvements) with four complementary mechanisms:

1. Structural scope filter (``scope_filter``): the apples-to-apples surface
   allowlist for monet_code mutations — a Layer-4 generalization check that
   sits beside ``self_evolve.generalization``'s Layer-2 content scan. Where
   Layer-2 catches *content* overfitting (task names, copied verifier output,
   narrowing conditionals), Layer-4 catches *structural* overfitting (a fix
   that's textually general but lives in monet_code's core control loop,
   provider client, or other non-harness machinery).

2. Reward-hacking honeypot (``honeypot``): re-runs candidates against
   Terminal Wrench (a corpus of reward-hackable terminal-agent environments)
   and flags candidates whose pass rate on the honeypot rises above
   baseline — a signal that they learned to game graders rather than solve
   tasks.

3. Trajectory verifier (``verifier``, ``verifier_fitness``): an LLM-as-a-Verifier
   implementation that scores trajectories via criteria decomposition,
   repeated verification, and round-robin selection. Used in two places:
   (a) as a fine-grained fitness signal during parent selection (replacing
       binary pass/fail with continuous trajectory quality), and
   (b) as a test-time Best-of-N selector (``best_of_n``).

4. Cross-model / cross-benchmark transfer gates (``transfer``): for candidates
   that survive (1)-(3), verify they don't regress on a different model
   family or on a small SWE-bench slice before they're promoted upstream.

Each module is independently usable. ``gate.py`` is the orchestrator that
chains them into one promote-or-reject decision for any node coming out of
a self_evolve campaign.

All Atelier modules are pure additions to self_evolve. They do not modify
self_evolve's search loop. Integration happens through (a) a small
``gate_hook`` in ``evolve.pipeline`` that calls Atelier between
the final eval and the node-archive write, and (b) the post-search
test-time Best-of-N pass, invoked separately as a CLI.

See ``atelier/README.md`` for the layer matrix and roll-out schedule.
"""

from __future__ import annotations

from .best_of_n import (
    BestOfNLift,
    BestOfNResult,
    TournamentRecord,
    Trajectory,
    TrajectorySampler,
    VerifierScorer,
    round_robin_tournament,
    run_best_of_n,
)
from .gate import (
    AtelierGate,
    GateDecision,
    LayerResult,
    LayerStatus,
    TransferGate,
    VerifierGate,
)
from .novelty import (
    NoveltyScore,
    TaskVector,
    cosine_distance,
    knn_novelty,
    rank_by_pn,
    score_node_pn,
    task_vectors_from_solved_lists,
)
from .sibling_pool import (
    SiblingCard,
    build_sibling_card,
    load_node_ancestors,
    render_sibling_evidence,
    select_siblings,
)
from .llm_parent_selector import (
    ArchiveCard,
    ParentPickResult,
    build_archive_card,
    render_archive_cards_for_prompt,
    select_parent_with_llm,
)
from .coverage_sizer import (
    CoverageRecommendation,
    K_MAX as COVERAGE_K_MAX,
    K_MIN as COVERAGE_K_MIN,
    recommend_k as recommend_coverage_k,
)
from .matchfix_gate import (
    ChangeContract,
    EquivalenceVerdict,
    ExtensionResult,
    HarborRunner,
    LLMBackend,
    OpenAIChatBackend,
    OpenAIChatConfig,
    ProbeResults,
    SemanticAnalysis,
    VerificationScope,
    aggregate_verdict,
    analyze_diff,
    chat_backend_from_credentials,
    execute_probes,
    extension_check,
    select_probe_tasks,
    verdict as matchfix_verdict,
    verdict_to_dict,
    verdict_with_consensus,
    write_verdict_sidecar,
)
from .trace_analyzer import (
    FailurePattern,
    TrialDigest,
    classify_failure_pattern,
    load_trial,
    render_cross_task_overview,
    render_digest,
    scan_eval_dir,
)
from .parent_credibility import (
    DEFAULT_EMPTY_PENALTY,
    DEFAULT_FLOOR,
    credibility_weight,
    load_node_credibilities,
    weights_for_campaign,
)
from .long_term_memory import (
    DEFAULT_MAX_ENTRIES as LTM_DEFAULT_MAX_ENTRIES,
    LongTermMemoryEntry,
    append_learnings,
    load_memory,
    load_rejected_sources,
    long_term_memory_path,
    mark_source_rejected,
    parse_learnings_to_persist,
    rejected_sources_path,
    render_memory_for_prompt,
)
from .predictions import (
    FailureEvidence,
    PredictedImpact,
    PredictionCredibility,
    compare_predictions_with_actual,
    load_credibility,
    load_predictions,
    parse_predicted_impact,
    rolling_credibility_score,
    save_credibility,
    save_predictions,
)
from .honeypot import (
    HoneypotConfig,
    HoneypotCorpus,
    HoneypotDecision,
    HoneypotDelta,
    HoneypotMode,
    HoneypotResult,
    HoneypotRunner,
    TaskResult,
    calibrate_threshold,
    compute_delta,
    score_candidate,
)
from .scope_filter import (
    ScopeDecision,
    ScopeMode,
    ScopeViolation,
    scan_diff,
    scan_paths,
)
from .trajectory_loader import (
    LoadedTrajectory,
    by_task_id,
    load_job,
    load_trial,
    parse_trial_name,
    to_verifier_input,
    to_verifier_inputs_by_task,
)
from .transfer import (
    CROSS_BENCHMARK_DEFAULT,
    CROSS_MODEL_DEFAULT,
    TransferConfig,
    TransferDecision,
    TransferDelta,
    TransferEvaluator,
    TransferGateAdapter,
    TransferMode,
    TransferResult,
    TransferTaskResult,
)
from .verifier import (
    DEFAULT_CRITERIA,
    DEFAULT_SCORE_TOKENS,
    TB2_CRITERIA,
    TB2_EXTRA_CRITERIA,
    Criterion,
    CriterionScore,
    ScoreDistribution,
    TrajectoryAssessment,
    TrajectoryInput,
    Verifier,
    VerifierBackend,
    criteria_for_profile,
    logprobs_to_distribution,
)
from .verifier_backend import (
    OpenAIVerifierBackend,
    OpenAIVerifierConfig,
    backend_from_credentials,
    config_from_credentials,
)
from .verifier_fitness import (
    DEFAULT_VERIFIER_WEIGHT,
    FitnessComponents,
    FitnessRunner,
    FitnessScore,
    compute_fitness,
)

__all__ = [
    # scope_filter
    "ScopeMode",
    "ScopeViolation",
    "ScopeDecision",
    "scan_diff",
    "scan_paths",
    # honeypot
    "HoneypotMode",
    "HoneypotConfig",
    "HoneypotCorpus",
    "HoneypotRunner",
    "TaskResult",
    "HoneypotResult",
    "HoneypotDelta",
    "HoneypotDecision",
    "score_candidate",
    "compute_delta",
    "calibrate_threshold",
    # best_of_n
    "Trajectory",
    "TrajectorySampler",
    "VerifierScorer",
    "TournamentRecord",
    "BestOfNResult",
    "BestOfNLift",
    "round_robin_tournament",
    "run_best_of_n",
    # gate
    "AtelierGate",
    "GateDecision",
    "LayerResult",
    "LayerStatus",
    "TransferGate",
    "VerifierGate",
    # novelty (GEA Performance-Novelty)
    "TaskVector",
    "NoveltyScore",
    "cosine_distance",
    "knn_novelty",
    "score_node_pn",
    "rank_by_pn",
    "task_vectors_from_solved_lists",
    # sibling_pool (GEA group evidence sharing)
    "SiblingCard",
    "select_siblings",
    "build_sibling_card",
    "render_sibling_evidence",
    "load_node_ancestors",
    # llm_parent_selector (v8 LLM-first parent picker)
    "ArchiveCard",
    "ParentPickResult",
    "build_archive_card",
    "render_archive_cards_for_prompt",
    "select_parent_with_llm",
    # coverage_sizer (v8 LLM-first adaptive K)
    "CoverageRecommendation",
    "COVERAGE_K_MIN",
    "COVERAGE_K_MAX",
    "recommend_coverage_k",
    # matchfix_gate
    "LLMBackend",
    "HarborRunner",
    "SemanticAnalysis",
    "ProbeResults",
    "ExtensionResult",
    "VerificationScope",
    "ChangeContract",
    "EquivalenceVerdict",
    "analyze_diff",
    "select_probe_tasks",
    "execute_probes",
    "matchfix_verdict",
    "verdict_with_consensus",
    "extension_check",
    "aggregate_verdict",
    "verdict_to_dict",
    "write_verdict_sidecar",
    "OpenAIChatConfig",
    "OpenAIChatBackend",
    "chat_backend_from_credentials",
    # verifier
    "Criterion",
    "ScoreDistribution",
    "CriterionScore",
    "TrajectoryAssessment",
    "TrajectoryInput",
    "VerifierBackend",
    "Verifier",
    "DEFAULT_SCORE_TOKENS",
    "DEFAULT_CRITERIA",
    "TB2_EXTRA_CRITERIA",
    "TB2_CRITERIA",
    "criteria_for_profile",
    "logprobs_to_distribution",
    # verifier_backend
    "OpenAIVerifierConfig",
    "OpenAIVerifierBackend",
    "config_from_credentials",
    "backend_from_credentials",
    # trajectory_loader
    "LoadedTrajectory",
    "parse_trial_name",
    "load_trial",
    "load_job",
    "by_task_id",
    "to_verifier_input",
    "to_verifier_inputs_by_task",
    # verifier_fitness
    "DEFAULT_VERIFIER_WEIGHT",
    "FitnessComponents",
    "FitnessScore",
    "FitnessRunner",
    "compute_fitness",
    # trace_analyzer
    "FailurePattern",
    "TrialDigest",
    "classify_failure_pattern",
    "load_trial",
    "scan_eval_dir",
    "render_digest",
    "render_cross_task_overview",
    # parent_credibility
    "DEFAULT_FLOOR",
    "DEFAULT_EMPTY_PENALTY",
    "credibility_weight",
    "load_node_credibilities",
    "weights_for_campaign",
    # long_term_memory
    "LongTermMemoryEntry",
    "LTM_DEFAULT_MAX_ENTRIES",
    "long_term_memory_path",
    "rejected_sources_path",
    "load_memory",
    "load_rejected_sources",
    "mark_source_rejected",
    "render_memory_for_prompt",
    "parse_learnings_to_persist",
    "append_learnings",
    # predictions
    "FailureEvidence",
    "PredictedImpact",
    "PredictionCredibility",
    "parse_predicted_impact",
    "compare_predictions_with_actual",
    "rolling_credibility_score",
    "save_predictions",
    "load_predictions",
    "save_credibility",
    "load_credibility",
    # transfer
    "TransferMode",
    "TransferTaskResult",
    "TransferResult",
    "TransferDelta",
    "TransferConfig",
    "TransferEvaluator",
    "TransferDecision",
    "TransferGateAdapter",
    "CROSS_MODEL_DEFAULT",
    "CROSS_BENCHMARK_DEFAULT",
]

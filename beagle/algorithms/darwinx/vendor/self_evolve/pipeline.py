"""Self-evolve pipeline orchestrator.

Implements the three-phase loop described in the plan:

  Phase 1 — Preparation:    open DB, select parent, create worktree + monet_code branch.
  Phase 2 — Optional baseline: parse provided baseline logs OR run baseline eval.
  Phase 3 — Self-evolve loop: claim ≤2 failing tasks, then for each iteration:
      analyze (cursor-agent --plan) → implement (cursor-agent) → review/test/commit
      → diff scan (Layer 2 guard) → mini-eval w/ canary task (Layer 3 guard).
  After loop: full eval, optional PR + learnings.md, claim release.

Heartbeats are emitted at every stage so `self_evolve_cleanup.py` can detect
dead pipelines.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import (
    adaptive_subset,
    atelier_hook,
    cursor_agent,
    generalization,
    meta_agent,
    parent_selection,
    pool,
    regression_selection,
    run_config,
    tree,
    worktree,
)

# Proposer (meta-agent) dispatch lives in ``meta_agent``: analyze/implement/review/
# select/learnings call ``meta_agent.run`` and the active backend (cursor | monet_code |
# claude_code) is selected via the ``META_AGENT`` env var. Default ``cursor`` forwards
# verbatim to cursor_agent, so behaviour is byte-identical to the pre-dispatch pipeline.
# All backends return a ``cursor_agent.CursorResult`` and share the agent-agnostic
# ``cursor_agent.render_prompt`` / DEFAULT_TIMEOUT_S / CursorResult.

# Eval seam: route every run_full / run_subset (mini-eval) / final-eval and
# result-parse call through the coding-bench adapter instead of the pristine
# exp_05 ``eval_runner`` (which shelled out to monet_code_eval's
# ``scripts/run_harbor.sh``). ``codingbench_eval`` re-exports the SAME public
# surface — run_full / run_subset / parse_existing_result_dir /
# restrict_to_subset / task_base / task_bases / EvalResult /
# EvalInfrastructureError — so the pipeline contract is unchanged; only the
# backend that PRODUCES the EvalResult differs (``python -m runner.run``
# against terminal-bench-v2 + monet + local). See codingbench_eval.py and
# docs/self_evolve/PHASE1_NOTES.md for the mapping.
from . import codingbench_eval as eval_runner


DEFAULT_FINAL_EVAL_ARCHIVE_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_REVIEW_TIMEOUT_S = 15 * 60
DEFAULT_EVAL_HEARTBEAT_INTERVAL_S = 60.0

# Cap on how large the regression-resolver's sticky canary can grow within
# one pipeline run. Any iteration that breaks a parent-solved task adds it
# to the sticky set; without a cap a noisy task could permanently inflate
# every subsequent iteration's mini-eval cost. When the cap is hit, the
# oldest-broken entry is evicted (FIFO by `iteration first observed`).
STICKY_CANARY_MAX = 50

# Auto canary count for `regression_resolve` pipelines on full-suite runs
# when `resolver_preservation_canary_count` is left at 0 (the default).
# Picks ~min(this, len(parent_solved)/3) fragility-ranked tasks per iteration
# so a regression on any of them is caught before the iteration's commits
# get kept and propagated to phase 4.
RESOLVER_AUTO_CANARY_FULL_SUITE = 20


def _final_eval_archive_max_bytes() -> int:
    raw = os.environ.get("MONET_EVAL_FINAL_ARCHIVE_MAX_BYTES")
    if raw is None:
        return DEFAULT_FINAL_EVAL_ARCHIVE_MAX_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_FINAL_EVAL_ARCHIVE_MAX_BYTES


def _infra_eval_retries() -> int:
    """How many times to retry a full eval that died with a WHOLE-JOB infra
    failure before giving up (and marking the node failed). Default 6; set
    MONET_EVAL_INFRA_RETRIES=0 to restore the old fail-on-first-infra-error
    behaviour. Per-task infra failures are already tolerated inside the eval
    runner, so this only fires on a global cluster/tunnel/gateway outage.

    Raised from 2 -> 6 for overnight autonomy: shared upstream gateway outages
    can last tens of minutes (observed a ~53-min window). With the backoff below
    (capped at 120s) 6 retries ride out ~10 min uncontended; overnight drivers
    export a much larger value (e.g. 40 -> ~80 min) so a single backend restart
    doesn't fail a node whose evolve work already completed."""
    raw = os.environ.get("MONET_EVAL_INFRA_RETRIES", "6").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 6


def _dir_size_bytes(path: Path, *, max_bytes: int | None = None) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
            if max_bytes is not None and total > max_bytes:
                return total
    return total


def _eval_heartbeat_interval_s() -> float:
    raw = os.environ.get("MONET_EVAL_EVAL_HEARTBEAT_INTERVAL_S")
    if raw is None:
        return DEFAULT_EVAL_HEARTBEAT_INTERVAL_S
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_EVAL_HEARTBEAT_INTERVAL_S


def _progress_signal_on() -> bool:
    """MOSS-style graded progress signal (ATELIER_PROGRESS_SIGNAL=1).

    When on, the picker and the per-iteration effort summary surface the raw
    k-sample pass-rate per claimed task so partial progress on a not-yet-flipped
    task is visible and rewarded instead of binarized away. Default OFF keeps the
    control arm byte-identical.
    """
    return os.environ.get("ATELIER_PROGRESS_SIGNAL", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _additive_scope_on() -> bool:
    """Additive-scope constraint (ATELIER_ADDITIVE_SCOPE=1).

    When on, the Layer-2 guard also rejects non-additive (destructive) diffs that
    rewrite monet's shared core — the v5-campaign failure mode where fixing a
    claimed task regressed 13-24 build/git/systems tasks. Reverting + re-prompting
    steers the proposer toward additive, isolated changes. Default off = control.
    """
    return os.environ.get("ATELIER_ADDITIVE_SCOPE", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _additive_max_deletions() -> int:
    """Deletion budget for the additive-scope guard (default 40)."""
    try:
        return max(0, int(os.environ.get("ATELIER_MAX_DELETIONS", "40")))
    except ValueError:
        return 40


def _hybrid_archive_on() -> bool:
    """Hybrid quality-diversity archive (ATELIER_HYBRID_ARCHIVE=1).

    When on, an improved-but-regressed candidate (gate verdict MODIFIED, not a
    cheat, regressions bounded) is NOT reset-to-parent. It proceeds to the full
    eval and is persisted as a distinct SCORED node with its improved/regressed
    deltas — a stepping stone the (already-enabled) regression-resolver, node
    merger, and broaden parent-selection can build on. This converts the search
    from greedy hill-climbing (which discards stepping stones and gets stuck at a
    coupled local optimum) into a memetic/quality-diversity search, while
    promotion of the shippable best stays strict (clean no-regression full eval)
    and anti-cheat stays a hard reject. Default off = greedy control arm.
    """
    return os.environ.get("ATELIER_HYBRID_ARCHIVE", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _defer_node_full_eval() -> bool:
    """Defer the per-node FULL eval (ATELIER_DEFER_NODE_FULL_EVAL=1).

    Minibatch-SGD intent: a clean-improving node should be allowed to EXTEND
    (keep taking cheap gradient steps), not pay an expensive 89-task full eval the
    moment it improves. When on, ``_finalize`` scores the node on the cheap
    claimed+rotating-guard set (mini-eval shape) — enough to rank it for parent
    selection (which already prefers the search/mini-eval score) and let it be a
    parent — and the only place the full avg@k runs is the end-of-campaign
    top-N confirm. Default off = legacy per-node full eval.
    """
    return os.environ.get("ATELIER_DEFER_NODE_FULL_EVAL", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _archive_max_regressions() -> int:
    """Max gate-probe regressions a stepping stone may have and still be
    archived (vs hard-rejected). Bounds the explore eval budget. Default 25."""
    try:
        return max(0, int(os.environ.get("ATELIER_ARCHIVE_MAX_REGRESSIONS", "25")))
    except ValueError:
        return 25


def _bestof2_contrast_on() -> bool:
    """Best-of-2 contrastive self-analysis (ATELIER_BESTOF2_CONTRAST=1).

    For claimed tasks the agent solves SOMETIMES (some trials passed, some
    failed — the best-of-k>>avg headroom), surface the passing vs failing trial
    trajectories to the proposer and direct it to contrast them: find the
    decisive behavioural difference and encode it as a reliable rule/skill so
    the modal rollout matches the best rollout. Turns the agent's own variance
    into a self-bootstrapping learning signal. Default off = control arm.
    """
    return os.environ.get("ATELIER_BESTOF2_CONTRAST", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _fractional_gate_on() -> bool:
    """Score the gate on FRACTIONAL avg@k pass-rates instead of a binary
    majority vote (ATELIER_FRACTIONAL_GATE=1). The binary ``rate>=0.5 -> 1.0``
    collapse hides partial regressions (5/5 -> 3/5) and over-credits flaky wins
    (1/2 booked as solved), so the gate signal stops tracking the real avg@5
    objective. Default off = legacy binary majority."""
    return _env_on("ATELIER_FRACTIONAL_GATE")


def _regression_margin() -> float:
    """FALLBACK-ONLY noise floor (NOT the primary decision). The GATE (reasoned
    verdict) decides regressions from the fractional before/after rates + sample
    counts + trajectories + collective knowledge — the number is context, not a
    cutoff. This margin is only consulted on the legacy ``canary_passed`` path
    used when the LLM verdict is unavailable, to avoid flagging pure sampling
    jitter. Default 0.20."""
    return _env_float("ATELIER_REGRESSION_MARGIN", 0.20)


def _accept_min_gain() -> float:
    """FALLBACK-ONLY noise floor (NOT the primary decision). When the LLM verdict
    GATE is unavailable we still avoid accepting a single-sample lucky flip; the
    real accept/archive/reject judgement is the reasoned verdict reasoning over
    fractional rates + confidence + gradient + collective knowledge. Default 0.5."""
    return _env_float("ATELIER_ACCEPT_MIN_GAIN", 0.5)


def _confirm_before_parent() -> bool:
    """Require a medium confirmatory avg@k eval before a node's score is trusted
    for best-node / parent / pool steering (ATELIER_CONFIRM_BEFORE_PARENT=1), so
    a node that won by sampling luck on the tiny mini-eval set can't hijack the
    search. Default off."""
    return _env_on("ATELIER_CONFIRM_BEFORE_PARENT")


def _confirm_k_samples() -> int:
    """k for the confirm-before-parent / deferred-node avg@k eval. Default 3."""
    return max(1, _env_int("ATELIER_CONFIRM_K_SAMPLES", 3))


def _collective_knowledge_on() -> bool:
    return _env_on("ATELIER_COLLECTIVE_KNOWLEDGE")


def _knowledge_gate_on() -> bool:
    """Feed the campaign-wide collective-knowledge digest into the GATE's
    decision evidence (not just the proposer prompt), turning the verdict into a
    collective-knowledge activation function (ATELIER_KNOWLEDGE_GATE=1)."""
    return _env_on("ATELIER_KNOWLEDGE_GATE")


_FAILURE_THEME_DIGEST_CACHE: str | None = None


def _failure_theme_digest() -> str:
    """Campaign-wide FAILURE-MODE THEME digest (ATELIER_FAILURE_THEME=1).

    Classifies the baseline run's failures into themes (wrong-output / timeout-
    setup / timeout-compute / ...) via ``trace_analyzer.failure_mode`` and turns
    the dominant themes into explicit, targetable guidance for the proposer AND
    the GATE — so the search steers at the *theme* ("most gaps are wrong-output:
    get the substance right & verify"; "timeouts are dependency-setup: install
    efficiently") instead of rediscovering it one task at a time. Computed once
    from BASELINE_LOGS/run.json and cached. Fail-safe: never raises; "" on any
    problem or when the flag is off.
    """
    global _FAILURE_THEME_DIGEST_CACHE
    if not _env_on("ATELIER_FAILURE_THEME"):
        return ""
    if _FAILURE_THEME_DIGEST_CACHE is not None:
        return _FAILURE_THEME_DIGEST_CACHE
    digest = ""
    try:
        import os as _os
        base = (_os.environ.get("BASELINE_LOGS", "") or "").strip()
        run_json = _os.path.join(base, "run.json") if base else ""
        if run_json and _os.path.exists(run_json):
            from trace_analyzer import failure_mode as _fm
            s = _fm.summarize_run(run_json)
            tally = s.get("theme_tally", {})
            by_task = s.get("by_task", {})
            # tasks whose dominant trial outcome is this theme
            def _tasks_for(mode: str, limit: int = 8) -> list[str]:
                out = []
                for t, modes in by_task.items():
                    if not modes:
                        continue
                    dom = max(modes, key=modes.get)
                    if dom == mode and modes[mode] >= 2:  # a real, repeated pattern
                        out.append(t)
                return sorted(out)[:limit]
            lines = ["CAMPAIGN FAILURE THEMES (diagnosis from the baseline avg@5 run — "
                     "steer your edit at the DOMINANT theme):"]
            wo = _tasks_for("wrong-output")
            if tally.get("wrong-output"):
                lines.append(
                    f"- wrong-output ({tally['wrong-output']} trials, usually the #1 theme): the agent "
                    f"RUNS TO COMPLETION but the grader fails — a CORRECTNESS gap, not speed. "
                    f"e.g. {', '.join(wo) if wo else '(various)'}. Priority: get the SUBSTANCE right "
                    f"(correct algorithm/answer/IDs) and VERIFY against the task's own checks BEFORE "
                    f"finishing; do not stop at plausible-looking output.")
            ts = _tasks_for("timeout-setup")
            if tally.get("timeout-setup"):
                lines.append(
                    f"- timeout-setup ({tally['timeout-setup']} trials): the time budget is burned on "
                    f"DEPENDENCY INSTALL (apt-get/pip). e.g. {', '.join(ts) if ts else '(various)'}. "
                    f"Priority: install efficiently — batch installs, prefer prebuilt wheels/binaries "
                    f"over from-source compiles, check-before-install, avoid redundant `apt-get update`.")
            tc = _tasks_for("timeout-compute")
            if tally.get("timeout-compute"):
                lines.append(
                    f"- timeout-compute ({tally['timeout-compute']} trials): heavy compile/train/sim "
                    f"eats the budget. e.g. {', '.join(tc) if tc else '(various)'}. Priority: choose the "
                    f"fastest correct approach that fits the time budget.")
            if len(lines) > 1:
                digest = "\n".join(lines)
    except Exception:  # noqa: BLE001 — advisory context must never break the loop
        digest = ""
    _FAILURE_THEME_DIGEST_CACHE = digest
    return digest


def _archive_all_on() -> bool:
    """Archive every non-accepted stepping stone (diff + deltas + lesson) as a
    reusable scored node instead of resetting to parent and dropping the work
    (ATELIER_ARCHIVE_ALL=1). Realizes 'never directly REJECT, archive as needed'.
    Promotion of the shipped best stays strict (clean avg@k win only)."""
    return _env_on("ATELIER_ARCHIVE_ALL")


def _qd_archive_on() -> bool:
    """Quality-diversity archive (ATELIER_QD_ARCHIVE=1).

    Keep LOSSY *specialists* — variants that newly solve a claimed task even if
    they regress a guard task — as ARCHIVED nodes carrying their REAL
    solved/improved/regressed sets, so the merge step can recombine complementary
    specialists into a net-positive generalist. Preservation is thereby enforced
    at the MERGE (output) layer, not per-variant, which is what lets the search
    escape a preserve-and-extend plateau.

    Reuses the existing ``archived`` status (already excluded from best/tip, and
    already merge- and parent-eligible), so NO new status wiring is required.
    Default off => byte-identical legacy behavior."""
    return _env_on("ATELIER_QD_ARCHIVE")


def _qd_solved_threshold() -> float:
    """avg@k pass-rate STRICTLY ABOVE which a claimed task counts as cracked for
    the QD *archive* gate (ATELIER_QD_SOLVED_THRESHOLD, default 0.0 => any seed).

    This is an ARCHIVE floor, not a merge-trust floor — a deliberately different
    question from the win-counter's majority (>= 0.5) rule. Archiving asks only
    "is there evidence this parent-unsolved residual is crackable, and does this
    node carry the change that did it?" One clean fail->pass on a task the parent
    never solved is real signal, and archiving is safe by construction: `archived`
    nodes are excluded from best/tip and exploit-selection, so a spurious
    specialist costs only a merge attempt, while MISSING a real one forecloses the
    only path off a preserve-and-extend plateau. That asymmetry says archive
    liberally (rate > 0). The stricter confirmation belongs DOWNSTREAM at the
    merge step (re-verify the niche on a consistent task set before it composes
    into a tip) — not here. Raise this knob only if noise floods the archive."""
    return _env_float("ATELIER_QD_SOLVED_THRESHOLD", 0.0)


def _guard_count(default: int = 3) -> int:
    """Number of regression-guard tasks per node (MONET_EVAL_GUARD_COUNT).
    Larger = better regression coverage of the ~84 non-claimed tasks. The legacy
    default was 3 (~3.5% coverage)."""
    return max(0, _env_int("MONET_EVAL_GUARD_COUNT", default))


# ─── Configuration ───────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    campaign: str
    reports_root: Path
    repo_root: Path
    config_path: Path                  # configs/terminal_bench_2.yaml
    # Optional override for `gh pr create --repo`; if None, gh auto-detects
    # from the submodule's `origin` remote (the recommended path so this
    # works against any fork of monet_code).
    monet_repo_url: str | None = None

    # Subset
    subset_label: str = "full"
    subset_tasks: list[str] = field(default_factory=list)   # [] = full
    adaptive_subset_enabled: bool = True
    adaptive_canary_fraction: float = adaptive_subset.DEFAULT_CANARY_FRACTION
    subset_eval_n_attempts: int = 1
    fullset_eval_n_attempts: int = 2
    # Inner-loop mini-eval samples per task (avg@k denoise). k=1 is the legacy
    # noisy single-trial gate that whipsawed the overnight campaign; k>=3 takes
    # a majority vote so a 4/5 canary no longer false-reverts a real fix and a
    # 2/5 fluke no longer counts as a win. Override via MONET_EVAL_MINI_EVAL_K.
    mini_eval_k_samples: int = 3
    # Full-set node scoring metric. "best" = legacy best-of-N (pass@N, which
    # over-reports vs the leaderboard); "avg" = mean reward over
    # ``fullset_eval_k_samples`` samples per task = avg@k, the exact metric the
    # public TB2.1 board reports. Default "best" keeps existing campaigns
    # byte-identical; the v2 launcher sets MONET_EVAL_FULLSET_METRIC=avg.
    fullset_eval_metric: str = "best"
    fullset_eval_k_samples: int = 5

    # Selection
    parent_strategy: str = "high_score_few_children"

    # Loop knobs
    max_loop_iters: int = 4
    n_failure_tasks: int = 2

    # Cursor agent
    cursor_model: str = field(default_factory=run_config.load_cursor_model_from_config)
    cursor_timeout_s: int = cursor_agent.DEFAULT_TIMEOUT_S
    cursor_analyze_timeout_s: int | None = None
    cursor_implement_timeout_s: int | None = None
    cursor_review_timeout_s: int | None = DEFAULT_REVIEW_TIMEOUT_S
    cursor_picker_timeout_s: int | None = None

    # Meta-agent (proposer) backend — chosen by env META_AGENT (cursor |
    # monet_code | claude_code; default cursor). The proposer stages dispatch
    # through ``meta_agent.run`` using ``meta_model`` / ``meta_effort`` below; with
    # the default cursor backend those resolve to ``cursor_model`` / None so the
    # call is byte-identical to the pre-dispatch pipeline. Per-backend model +
    # reasoning effort are loaded from the campaign YAML's optional
    # ``claude_code:`` / ``monet_code:`` blocks. cursor effort lives in the model
    # slug (no flag); monet_code / claude_code translate effort to ``--effort``.
    claude_code_model: str = field(default_factory=run_config.load_claude_code_model)
    claude_code_effort: str | None = field(default_factory=run_config.load_claude_code_effort)
    monet_code_model: str | None = field(default_factory=run_config.load_monet_code_model)
    monet_code_effort: str | None = field(default_factory=run_config.load_monet_code_effort)

    # Generalization guard
    guard_enabled: bool = True
    guard_strict: bool = False
    guard_canary_count: int = 1

    # Regression resolver knobs
    # `regression_max_claim`: cap on how many of the parent node's regressed
    # tasks a single resolver pipeline claims (and therefore mini-evals each
    # iteration). 0 / None = no cap; the resolver claims the full regression
    # set so it owns "everything that broke" and the prompt sees all of it.
    # Operators tighten this when per-iteration cost is the bottleneck.
    regression_max_claim: int = 0
    # `resolver_preservation_canary_count`: per-iteration canary count for
    # regression-resolve pipelines (overrides `guard_canary_count`). 0 = use
    # the kind-aware auto default in `_pick_guard_canaries`. The resolver's
    # explicit goal is "no new regressions vs parent", so its canary set is
    # much larger than a normal evolve worker's by default.
    resolver_preservation_canary_count: int = 0
    # Deprecated compatibility knob: kept in configs/CLI output, but resolver
    # mini-eval acceptance now uses the scale-aware rate thresholds below.
    resolver_mini_eval_min_net_gain: int = 1
    # Optional hard cap on canary failures per mini-eval. 0 = no hard cap,
    # use the rate policy only.
    resolver_mini_eval_max_canary_failures: int = 0
    # Scale-aware regression-resolver mini-eval acceptance. Keep an iteration
    # when it fixes a high enough fraction of claimed regressions, breaks only
    # a small fraction of preservation canaries, and maintains a positive
    # normalized margin between the two rates.
    resolver_mini_eval_min_claimed_win_rate: float = 0.50
    resolver_mini_eval_max_canary_failure_rate: float = 0.20
    resolver_mini_eval_min_rate_margin: float = 0.25
    # Final full-eval acceptance for resolver nodes: final score must improve
    # and full-eval improved task count minus regressed task count must meet
    # this threshold.
    resolver_final_min_net_gain: int = 1

    # Misc
    cleanup_worktree: bool = True
    auto_report: bool = False
    baseline_logs: Path | None = None

    # Internal: when True, the orchestrator runs Phases 1+2 only (bootstrap
    # the root and score it) and exits before Phase 3. The supervisor uses
    # this to do a serial bootstrap before spawning N parallel workers, so
    # all N workers see a scored parent and don't race on the baseline.
    bootstrap_only: bool = False

    # Internal: when set, override the auto-generated `pipeline_id`. The
    # supervisor uses this to atomically pre-claim failing tasks under a
    # specific id before the worker starts, so the worker can skip its own
    # parent-selection + claim attempt entirely. None (the default)
    # preserves the standalone-worker behaviour of generating a fresh id
    # inside `SelfEvolvePipeline.__init__`.
    pipeline_id_override: str | None = None

    # Internal: when set, the worker uses this node id directly as its
    # parent and skips the `parent_strategy` machinery entirely. The
    # supervisor sets it to the parent it pre-claimed for this worker,
    # so two siblings spawned in the same instant can't race on the
    # strategy's tiebreak hash and land on the same parent. None (the
    # default) is the standalone-worker path: pick a parent via the
    # configured strategy.
    parent_id_override: str | None = None

    # Internal: normal workers use "evolve"; regression resolver workers reuse
    # this pipeline machinery but create a distinct typed edge and claim tasks
    # from the target node's regressed_tasks list.
    pipeline_kind: str = "evolve"

    @property
    def analyze_timeout_s(self) -> int:
        return self.cursor_analyze_timeout_s or self.cursor_timeout_s

    @property
    def implement_timeout_s(self) -> int:
        return self.cursor_implement_timeout_s or self.cursor_timeout_s

    @property
    def review_timeout_s(self) -> int:
        return self.cursor_review_timeout_s or self.cursor_timeout_s

    @property
    def picker_timeout_s(self) -> int:
        return self.cursor_picker_timeout_s or self.cursor_timeout_s

    @property
    def meta_backend(self) -> str:
        """Active proposer backend: cursor | monet_code | claude_code (env META_AGENT)."""
        return meta_agent.active_backend()

    @property
    def meta_model(self) -> str:
        """Model for the active proposer backend (passed to ``meta_agent.run``)."""
        backend = self.meta_backend
        if backend == "claude_code":
            return self.claude_code_model
        if backend == "monet_code":
            return self.monet_code_model or self.cursor_model
        return self.cursor_model

    @property
    def meta_effort(self) -> str | None:
        """Reasoning effort for the active proposer backend. None for cursor
        (its effort is encoded in the model slug, not a flag)."""
        backend = self.meta_backend
        if backend == "claude_code":
            return self.claude_code_effort
        if backend == "monet_code":
            return self.monet_code_effort
        return None


# ─── Iteration outcome ──────────────────────────────────────────────────


@dataclass
class IterationOutcome:
    iteration: int
    committed_shas: list[str] = field(default_factory=list)
    mini_eval_job_dir: Path | None = None
    rewards_per_task: dict[str, float] = field(default_factory=dict)
    # Graded k-sample pass-rate per task (0..1) BEFORE the majority-vote
    # binarization in rewards_per_task. This is the MOSS-style progress signal:
    # it lets the picker/next-iteration see a task move 0/3 -> 1/3 -> 2/3 as
    # real progress instead of a flat 0 until it fully flips. Same data, just
    # not thrown away by the >=0.5 vote. (== the best-of-k consistency signal.)
    pass_rates_per_task: dict[str, float] = field(default_factory=dict)
    canary_tasks: list[str] = field(default_factory=list)
    preservation_tasks: list[str] = field(default_factory=list)
    canary_passed: bool = True
    guard_violations: list[generalization.Violation] = field(default_factory=list)
    reverted: bool = False
    reason: str = ""
    review_duration_ms: int = 0
    review_error: str | None = None
    mini_eval_score: float | None = None
    mini_eval_n_trials: int | None = None
    mini_eval_n_errors: int | None = None
    failed_canary_tasks: list[str] = field(default_factory=list)
    mini_eval_net_gain: int | None = None

    def both_targets_pass(self, targets: list[str]) -> bool:
        return all(self.rewards_per_task.get(t, 0.0) >= 1.0 for t in targets)

    def preservation_passed(self) -> bool:
        return all(self.rewards_per_task.get(t, 0.0) >= 1.0 for t in self.preservation_tasks)

    @property
    def review_timed_out(self) -> bool:
        return bool(self.review_error and "timed out" in self.review_error.lower())


@dataclass(frozen=True)
class ResolverMiniEvalStats:
    claimed_total: int
    claimed_wins: int
    canary_total: int
    canary_failures: int
    claimed_win_rate: float
    canary_failure_rate: float
    rate_margin: float
    net_gain: int


# ─── Logger ─────────────────────────────────────────────────────────────


# Minimum length below which a plan body is treated as junk by the analyze
# step's sanity check. The analyze prompt intentionally does not prescribe a
# rigid output structure, so length is the only cheap guard here; it catches
# the failure mode where cursor-agent emitted a placeholder via CreatePlan and
# then never wrote a real final assistant text.
_MIN_PLAN_TEXT_CHARS = 200


def _invalid_plan_reasons(text: str | None) -> list[str]:
    """Return human-readable reasons an analyze plan fails the sanity check."""
    raw = text or ""
    s = raw.strip()
    reasons: list[str] = []
    if len(s) < _MIN_PLAN_TEXT_CHARS:
        reasons.append(f"too short ({len(s)} < {_MIN_PLAN_TEXT_CHARS} chars)")
    return reasons


def _looks_like_real_plan(text: str | None) -> bool:
    """True when `text` is long enough to plausibly be an analyze-step plan."""
    return not _invalid_plan_reasons(text)


def _make_logger(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Avoid double handlers if called twice for the same name.
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


# ─── Orchestrator ───────────────────────────────────────────────────────


class SelfEvolvePipeline:
    """One pipeline = one parent → one child node attempt."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        # The supervisor pre-claims tasks against a specific pipeline_id, then
        # spawns the worker with `--pipeline-id <id>`. Honor that override so
        # the worker's claim-table reads see the supervisor's pre-claimed rows
        # as belonging to "us". Standalone runs (no override) keep the
        # historical behaviour of generating a fresh id.
        self.pipeline_id = cfg.pipeline_id_override or tree.new_id()
        self.campaign_dir = tree.campaign_root(cfg.reports_root, cfg.campaign)
        self.campaign_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = tree.db_path_for(cfg.reports_root, cfg.campaign)
        self.conn = tree.connect(self.db_path)

        self.pipeline_log_dir = tree.pipeline_dir(cfg.reports_root, cfg.campaign, self.pipeline_id)
        self.pipeline_log_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path = self.pipeline_log_dir / "run.log"
        self.cursor_log_dir = self.pipeline_log_dir / "cursor"
        self.cursor_log_dir.mkdir(parents=True, exist_ok=True)

        self.prompts_dir_path = tree.prompts_dir(cfg.reports_root, cfg.campaign, self.pipeline_id)
        self.prompts_dir_path.mkdir(parents=True, exist_ok=True)

        self.log = _make_logger(f"selfevolve.{self.pipeline_id}", self.run_log_path)

        # Filled in by phases.
        self.parent_node: tree.Node | None = None
        self.child_node: tree.Node | None = None
        self.worktree: worktree.Worktree | None = None
        self.claimed_tasks: list[str] = []
        self.passing_for_canary: list[str] = []
        self.iterations: list[IterationOutcome] = []
        self.guard_violated_ever: bool = False
        # Sticky canary for `regression_resolve` pipelines: any parent-solved
        # task that fails (reward < 1.0) in any iteration's mini-eval gets
        # promoted into the canary set for every subsequent iteration of this
        # pipeline. Maps task name -> iteration first observed broken, used
        # for oldest-first eviction when the set hits `_STICKY_CANARY_MAX`.
        # Empty for non-resolver pipelines.
        self._sticky_canary: dict[str, int] = {}
        # Preflight short-circuit. Set by `_phase1_prepare` when N parallel
        # workers race for a tiny failing-task pool and this one would
        # otherwise pay ~1-2 min of worktree setup just to discover its
        # claim attempt comes up empty. When True, `run()` skips Phases
        # 2-4 and marks the pipeline `no_change` immediately.
        self._preflight_no_work: bool = False

    @contextmanager
    def _heartbeat_during_eval(self, label: str):
        """Keep this pipeline fresh while a blocking Harbor subprocess runs."""
        db_path = getattr(self, "db_path", None)
        if db_path is None:
            tree.heartbeat(self.conn, self.pipeline_id)
            try:
                yield
            finally:
                tree.heartbeat(self.conn, self.pipeline_id)
            return

        stop = threading.Event()
        interval_s = _eval_heartbeat_interval_s()

        def _beat_loop() -> None:
            conn = tree.connect(db_path)
            try:
                while not stop.wait(interval_s):
                    try:
                        tree.heartbeat(conn, self.pipeline_id)
                    except Exception:
                        self.log.debug(
                            "heartbeat failed during %s", label, exc_info=True,
                        )
            finally:
                conn.close()

        tree.heartbeat(self.conn, self.pipeline_id)
        thread = threading.Thread(
            target=_beat_loop,
            name=f"selfevolve-heartbeat-{self.pipeline_id}",
            daemon=True,
        )
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=5)
            tree.heartbeat(self.conn, self.pipeline_id)

    def _search_eval(self, node: tree.Node | None = None) -> tree.NodeEval | None:
        node = node or self.parent_node
        if node is None:
            return None
        return tree.node_search_eval(
            self.conn, campaign=self.cfg.campaign, node_id=node.id,
        )

    def _search_score(self, node: tree.Node | None = None) -> float | None:
        ev = self._search_eval(node)
        return ev.score if ev else None

    def _node_solved_tasks(self, node: tree.Node) -> list[str]:
        ev = self._search_eval(node)
        return ev.solved_tasks if ev else node.solved_tasks

    def _preservation_solved_pool(self, node: tree.Node) -> list[str]:
        """Tasks the equivalence gate must verify are preserved in the child.

        Zeyuan's adaptive-subset harness only re-measures a fixed ~10% canary
        sample of the root's solved tasks (``adaptive_subset.py``), so a
        subset-evaluated parent's recorded ``solved_tasks`` is that thin
        canary shadow (e.g. 7 of 61), NOT the agent's true capability set.
        Feeding the shadow to the gate caps ``coverage_sizer`` at ~7 probes
        and leaves the other ~54 solved tasks permanently un-probeable — the
        structural blind spot that lets a global diff regress unprobed tasks
        (see ``coverage_sizer.py`` docstring, v7 ``c22ff235``).

        Our diff-based preservation gate takes priority over the baseline
        subset sampler: probe against the lineage's FULL known-solved set =
        the parent's measured solves UNION the root's full-eval solves. The
        coverage sizer then scales K with the diff's blast radius over the
        real pool, and the LLM verdict only hard-rejects failures it can
        attribute to the diff (inherited/unrelated failures degrade to
        INCONCLUSIVE), so widening the pool cannot manufacture false rejects.

        Disable via ``ATELIER_FULL_SOLVED_PRESERVATION=0`` to restore the
        legacy parent-only (canary-shadow) pool.
        """
        pool: list[str] = [t for t in (self._node_solved_tasks(node) or []) if t]
        seen = set(pool)
        if os.environ.get(
            "ATELIER_FULL_SOLVED_PRESERVATION", "1"
        ).strip().lower() in {"0", "false", "no", "off"}:
            return pool
        try:
            root_eval = tree.root_full_eval(self.conn, campaign=self.cfg.campaign)
        except Exception as e:  # noqa: BLE001 — gate must never crash the pipeline
            self.log.warning("preservation pool: root_full_eval failed (%s)", e)
            root_eval = None
        if root_eval is not None:
            for t in (root_eval.solved_tasks or []):
                if t and t not in seen:
                    seen.add(t)
                    pool.append(t)
        return pool

    def _node_failed_tasks(self, node: tree.Node) -> list[str]:
        ev = self._search_eval(node)
        return ev.failed_tasks if ev else node.failed_tasks

    def _node_partially_solved_tasks(self, node: tree.Node) -> list[str]:
        ev = self._search_eval(node)
        return ev.partially_solved_tasks if ev else node.partially_solved_tasks

    def _node_improved_tasks(self, node: tree.Node) -> list[str]:
        ev = self._search_eval(node)
        return ev.improved_tasks if ev else node.improved_tasks

    def _node_regressed_tasks(self, node: tree.Node) -> list[str]:
        ev = self._search_eval(node)
        return ev.regressed_tasks if ev else node.regressed_tasks

    def _node_eval_job_path(self, node: tree.Node) -> str | None:
        ev = self._search_eval(node)
        return ev.job_log_path if ev else node.job_log_path

    def _adaptive_subset_for_root(
        self,
        root_eval: eval_runner.EvalResult | tree.NodeEval,
        *,
        root_node_id: str,
    ) -> adaptive_subset.AdaptiveSubset:
        if isinstance(root_eval, eval_runner.EvalResult):
            return adaptive_subset.build_from_eval_result(
                root_eval,
                campaign=self.cfg.campaign,
                root_node_id=root_node_id,
                canary_fraction=self.cfg.adaptive_canary_fraction,
            )
        return adaptive_subset.build_from_node_eval(
            root_eval,
            campaign=self.cfg.campaign,
            root_node_id=root_node_id,
            canary_fraction=self.cfg.adaptive_canary_fraction,
        )

    def _adaptive_eval_tasks(self) -> list[str]:
        root_eval = tree.root_full_eval(self.conn, campaign=self.cfg.campaign)
        if root_eval is None:
            return list(self.cfg.subset_tasks)
        subset = self._adaptive_subset_for_root(root_eval, root_node_id=root_eval.node_id)
        return subset.tasks

    def _parent_unsolved_for_deltas(
        self, *, parent_solved: list[str], parent_unsolved: list[str],
    ) -> list[str]:
        """Parent's not-solved set, widened by this pipeline's claimed tasks.

        Claimed tasks come off the failing-task pool, so the parent does not
        solve them by construction. Its own eval panel is a narrow adaptive
        slice that usually omits them entirely, so a child's real claimed-task
        win appeared in neither `parent_solved` nor `parent_unsolved` and
        `task_deltas` scored it as zero improvement — while guard-panel losses
        still counted, since those tasks *are* in `parent_solved`. That
        asymmetry (wins invisible, losses visible) left `improved_tasks` empty
        on every specialist and starved the QD archive of its whole purpose.
        """
        solved = set(parent_solved)
        return list(dict.fromkeys(
            list(parent_unsolved)
            + [t for t in (self.claimed_tasks or []) if t not in solved]
        ))

    def _persist_node_eval_result(
        self,
        *,
        node_id: str,
        eval_kind: str,
        result: eval_runner.EvalResult,
        parent_eval: tree.NodeEval | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tree.NodeEval:
        child_solved = list(result.solved_tasks)
        child_unsolved = list(result.unsolved_tasks)
        child_partial = list(result.partially_solved_tasks)
        child_failed = list(result.failed_task_names)
        if parent_eval is None:
            improved_tasks: list[str] = []
            regressed_tasks: list[str] = []
        else:
            improved_tasks, regressed_tasks = tree.task_deltas(
                parent_solved=parent_eval.solved_tasks,
                parent_unsolved=self._parent_unsolved_for_deltas(
                    parent_solved=parent_eval.solved_tasks,
                    parent_unsolved=parent_eval.failed_tasks,
                ),
                child_solved=child_solved,
                child_unsolved=child_failed,
            )
        return tree.upsert_node_eval(
            self.conn,
            campaign=self.cfg.campaign,
            node_id=node_id,
            eval_kind=eval_kind,
            subset_label=self.cfg.subset_label,
            task_names=list(result.task_names),
            n_trials=result.n_trials,
            n_errors=result.n_errors,
            score=result.score,
            job_log_path=str(result.job_dir),
            solved_tasks=child_solved,
            unsolved_tasks=child_unsolved,
            partially_solved_tasks=child_partial,
            task_rewards=dict(result.task_rewards),
            improved_tasks=improved_tasks,
            regressed_tasks=regressed_tasks,
            source_pipeline_id=self.pipeline_id,
            metadata=metadata,
        )

    def _record_experiences_from_eval(
        self,
        *,
        node: tree.Node,
        node_eval: tree.NodeEval,
        parent_eval: tree.NodeEval | None,
        worker_kind: str,
    ) -> None:
        if parent_eval is None:
            return
        affected = [
            ("improved", task)
            for task in node_eval.improved_tasks
        ] + [
            ("regressed", task)
            for task in node_eval.regressed_tasks
        ]
        if not affected:
            return
        commit_sha = node.commit_sha
        commits = node.commits
        commit_number = (
            commits.index(commit_sha) + 1
            if commit_sha and commit_sha in commits else None
        )
        for kind, task in affected:
            try:
                tree.insert_task_experience(
                    self.conn,
                    campaign=self.cfg.campaign,
                    task=task,
                    node_id=node.id,
                    pipeline_id=self.pipeline_id,
                    worker_kind=worker_kind,
                    commit_sha=commit_sha,
                    commit_number=commit_number,
                    experience_kind=kind,
                    eval_kind=node_eval.eval_kind,
                    before_reward=parent_eval.task_rewards.get(task),
                    after_reward=node_eval.task_rewards.get(task),
                    analysis=(
                        f"{kind} on {node_eval.eval_kind}; summary may need "
                        "manual confirmation against original trajectories."
                    ),
                    code_change_summary=(
                        "Automatically inferred from task reward delta after "
                        f"{worker_kind} evaluation."
                    ),
                    artifact_paths=[p for p in [node_eval.job_log_path] if p],
                    confidence=0.55 if node_eval.eval_kind == "subset_final" else 0.8,
                    metadata={"source": "orchestrator_eval_delta"},
                )
            except Exception:
                self.log.warning("could not record task experience for %s", task, exc_info=True)

    def _record_nonpromote_experiences(
        self, *, kind: str, summary: str, worker_kind: str,
    ) -> None:
        """Record a NON-promoted node's claimed-task patterns as collective
        knowledge (kind='rejected' for a regression/no-gain, 'poisoned' for a
        verifier-gaming cheat). Realizes 'never drop the lesson': a non-promoted
        edit still teaches future proposers what to avoid. Best-effort; never
        raises into the loop."""
        if not _collective_knowledge_on() and not _knowledge_gate_on():
            # Knowledge channels are off; nothing consumes these rows. Skip the
            # writes to keep the legacy path byte-identical.
            return
        try:
            node = self.child_node
            commit_sha = getattr(node, "commit_sha", None)
            for task in (self.claimed_tasks or []):
                try:
                    tree.insert_task_experience(
                        self.conn,
                        campaign=self.cfg.campaign,
                        task=task,
                        node_id=getattr(node, "id", None),
                        pipeline_id=self.pipeline_id,
                        worker_kind=worker_kind,
                        commit_sha=commit_sha,
                        experience_kind=kind,
                        analysis=summary,
                        code_change_summary=(self._child_diff_summary() or "")[:500],
                        confidence=0.6,
                        metadata={"source": "nonpromote", "decision": kind},
                    )
                except Exception:
                    self.log.debug("nonpromote experience skip for %s", task, exc_info=True)
        except Exception as exc:  # noqa: BLE001
            self.log.debug("nonpromote experience recording skipped: %s", exc)

    # ─── Entry point ──────────────────────────────────────────────────

    def run(self) -> int:
        """Execute the full pipeline. Returns shell-style exit code."""
        rc = 1
        try:
            self._register_pipeline()
            self._phase1_prepare()
            if self._preflight_no_work:
                # Sibling pipelines already hold every claim for this
                # parent's failing-task pool. We deliberately skipped
                # worktree creation in Phase 1 and have no child node;
                # finish the pipeline row cleanly and return success so
                # the supervisor's aggregate rc isn't poisoned by a
                # symptom of healthy parallel contention.
                tree.update_pipeline(
                    self.conn, self.pipeline_id,
                    status="no_change", finished_at=tree.utcnow_iso(),
                )
                return 0
            self._phase2_baseline()
            if self.cfg.bootstrap_only:
                # Supervisor-driven bootstrap precursor: stop after baseline so
                # parallel workers can fan out from a scored parent.
                self.log.info(
                    "bootstrap-only mode: parent %s scored, exiting before Phase 3",
                    self.parent_node.id if self.parent_node else "?",
                )
                tree.update_pipeline(
                    self.conn, self.pipeline_id,
                    status="done", finished_at=tree.utcnow_iso(),
                )
                rc = 0
            else:
                self._phase3_loop()
                self._phase4_pick_best_commit()
                rc = self._finalize()
            return rc
        except Exception as e:
            self.log.exception("pipeline failed: %s", e)
            self._mark_failed(str(e))
            return 1
        finally:
            # Honor --cleanup-worktree on every successful exit path
            # (including bootstrap-only and no_change). The worktree is only
            # useful for cursor-agent debugging when an iteration kept commits;
            # for everything else we save disk by removing it.
            try:
                if self.cfg.cleanup_worktree and self.worktree is not None:
                    self._cleanup_worktree()
            except Exception:
                self.log.warning("worktree cleanup failed (non-fatal)", exc_info=True)
            try:
                tree.release_claims(self.conn, pipeline_id=self.pipeline_id)
            except Exception:
                pass
            self.conn.close()

    # ─── Pipeline row ─────────────────────────────────────────────────

    def _register_pipeline(self) -> None:
        tree.insert_pipeline(
            self.conn,
            id=self.pipeline_id,
            campaign=self.cfg.campaign,
            parent_node_id=None,           # filled in after parent selection
            log_path=str(self.run_log_path),
            worktree_path=None,            # filled in after worktree creation
            pid=os.getpid(),
        )
        tree.heartbeat(self.conn, self.pipeline_id)
        self.log.info(
            "pipeline %s started (campaign=%s subset=%s parallel-friendly)",
            self.pipeline_id, self.cfg.campaign, self.cfg.subset_label,
        )

    def _mark_failed(self, msg: str) -> None:
        try:
            tree.update_pipeline(
                self.conn, self.pipeline_id,
                status="failed", finished_at=tree.utcnow_iso(),
            )
            if self.child_node:
                node = tree.get_node(self.conn, self.child_node.id)
                if node and node.status == "in_progress":
                    tree.update_node(self.conn, self.child_node.id, status="failed")
        except Exception:
            pass
        self.log.error("pipeline %s marked failed: %s", self.pipeline_id, msg)

    # ─── Phase 1: Preparation ─────────────────────────────────────────

    def _phase1_prepare(self) -> None:
        tree.update_pipeline(self.conn, self.pipeline_id, status="preparing")
        tree.heartbeat(self.conn, self.pipeline_id)
        self._maybe_bootstrap_root()
        self._select_parent()
        if self.cfg.bootstrap_only:
            # Supervisor-driven baseline scoring only — no worktree, no
            # child node. Phase 2 will write the score directly onto the
            # parent (the just-bootstrapped root) and we exit immediately
            # after, so all subsequent workers can fan out from the
            # already-scored parent.
            return
        # Supervisor pre-claim short-circuit: if the parent supervisor
        # already claimed failing tasks under our `pipeline_id`, skip the
        # "is there work left?" preflight — by construction there IS work
        # (the supervisor would not have spawned us otherwise). Worktree
        # creation proceeds.
        has_pre_claims = bool(
            tree.list_active_claims(self.conn, pipeline_id=self.pipeline_id)
        )
        # Cheap preflight before the expensive `worktree.add` (~1-2 min of
        # `git worktree add` + `uv sync`). If the parent is already scored
        # AND every one of its failing tasks is already held by a sibling
        # claim, this worker can't possibly do useful work — short-circuit
        # now and exit `no_change` instead of paying setup costs only to
        # discover the empty-claim path in Phase 3.
        if not has_pre_claims and self._preflight_pool_is_empty():
            self.log.info(
                "preflight: parent %s has no unclaimed failing tasks left "
                "for subset=%s (every candidate is already held by a "
                "sibling pipeline). Skipping worktree creation; this "
                "pipeline will exit no_change.",
                self.parent_node.id if self.parent_node else "?",
                self.cfg.subset_label,
            )
            self._preflight_no_work = True
            return
        self._create_worktree()
        self._create_child_node()

    def _preflight_pool_is_empty(self) -> bool:
        """Read-only check: would Phase 3's `pick_and_claim` come up empty?

        Returns False (i.e. "don't short-circuit") in any case where the
        check would be unreliable:
          - the campaign has no scored eligible node yet — Phase 2 will
            score the parent and populate `failed_tasks`; we can't know
            the pool here.
          - `--baseline-logs` was provided — the failing tasks aren't in
            the DB yet; Phase 2 parses them out shortly. Letting the
            normal flow run preserves the existing semantics.

        Otherwise consults `pool.unresolved_tasks` and returns True only
        when it comes back empty.
        """
        if self.cfg.baseline_logs is not None:
            return False
        if self.cfg.pipeline_kind == "regression_resolve":
            if not self.parent_node or not self.parent_node.regressed_tasks:
                return True
            active = {
                c.failure_task for c in tree.list_active_claims(
                    self.conn,
                    campaign=self.cfg.campaign,
                    subset=self.cfg.subset_label,
                    claim_kind="regression_resolve",
                    parent_id=self.parent_node.id,
                )
            }
            subset_filter = set(self.cfg.subset_tasks or [])
            candidates = [
                t for t in self.parent_node.regressed_tasks
                if (not subset_filter or t in subset_filter) and t not in active
            ]
            return not candidates
        # If no node has scored yet, `pool.best_node` returns None and the
        # pool comes back empty for the wrong reason. Don't short-circuit
        # — Phase 2 may still produce useful failing tasks.
        if pool.best_node(
            self.conn,
            campaign=self.cfg.campaign,
            subset=self.cfg.subset_label,
        ) is None:
            return False
        subset_filter = self.cfg.subset_tasks or None
        remaining = pool.unresolved_tasks(
            self.conn,
            campaign=self.cfg.campaign,
            subset=self.cfg.subset_label,
            subset_filter=subset_filter,
        )
        return not remaining

    def _existing_root_id(self) -> str | None:
        """Read-only existence check for `(campaign, subset)`'s root.

        Returns the root id if one exists, else None. Used by
        `_maybe_bootstrap_root` to short-circuit the seed-commit
        resolution (notably the `git fetch origin develop` round-trip)
        when this campaign has already been bootstrapped. The
        IMMEDIATE-transaction check inside `bootstrap_root_if_absent`
        remains the authoritative race protection.
        """
        row = self.conn.execute(
            "SELECT id FROM nodes WHERE campaign = ? AND subset = ? "
            "AND parent_id IS NULL ORDER BY created_at ASC LIMIT 1",
            (self.cfg.campaign, self.cfg.subset_label),
        ).fetchone()
        return row["id"] if row else None

    def _should_seed_from_develop(self) -> bool:
        """Return True if the bootstrap root should be seeded from
        `origin/develop` rather than from a user-supplied source.

        Policy: only "default greenfield" runs (no explicit baseline) get
        the develop-tip treatment. `--baseline-logs` short-circuits the
        develop fetch because the user has already named a job-run source.
        """
        if self.cfg.baseline_logs is not None:
            return False
        return True

    def _maybe_bootstrap_root(self) -> None:
        """Atomically ensure exactly one root exists for (campaign, subset).

        Safe under N parallel workers — `tree.bootstrap_root_if_absent` does
        the check-and-insert inside a single IMMEDIATE transaction so only
        one worker ever inserts.

        Seed-commit policy
        ~~~~~~~~~~~~~~~~~~
        - Default (no `--baseline-logs`): seed from `origin/develop` of
          the `monet_code` submodule. Fetches it fresh so a fresh campaign
          is reproducible regardless of whatever the developer happens to
          have checked out locally (e.g. a feature branch or a stale
          `develop`). This is the path the supervisor's bootstrap precursor
          takes.
        - With `--baseline-logs`: seed from `monet_code` HEAD on disk.
          The user has already produced a baseline against some
          specific commit; we trust their checkout to match.

        Performance note: we do a cheap read-only existence check
        BEFORE resolving the seed commit. `monet_develop_tip()` does
        a `git fetch origin develop` (~hundreds of ms to seconds) and
        every worker that joins an already-bootstrapped campaign
        would otherwise pay that cost for nothing. The transactional
        check inside `bootstrap_root_if_absent` remains the
        authoritative race protection — this read is just a fast
        path.
        """
        existing_id = self._existing_root_id()
        if existing_id is not None:
            # Fast path: someone (us in a previous run, or a sibling
            # worker) already bootstrapped this campaign+subset. Skip
            # the seed-commit resolution entirely (no `git fetch`) —
            # the existing row's commit/branch is what matters.
            self.log.info(
                "root for subset=%s already exists: %s",
                self.cfg.subset_label, existing_id,
            )
            self._backfill_root_works_md_if_missing(existing_id)
            return

        # Slow path: this is the first worker to see an empty tree
        # for this campaign+subset. Resolve the seed commit per policy.
        if self._should_seed_from_develop():
            seed_ref = os.environ.get("SELF_EVOLVE_ROOT_COMMIT", "").strip()
            if seed_ref:
                # Evolve on top of an explicit agent build (e.g. a PR head)
                # instead of origin/develop, while still running a fresh
                # baseline. The companion SELF_EVOLVE_ROOT_FETCH_REF names a
                # branch to fetch so a remote-only sha becomes resolvable.
                # See docs/self_evolve/CLUSTER_LAUNCH.md.
                fetch_ref = os.environ.get(
                    "SELF_EVOLVE_ROOT_FETCH_REF", "",
                ).strip() or None
                commit, branch = worktree.resolve_seed_commit(
                    seed_ref, fetch_ref=fetch_ref,
                )
                self.log.info(
                    "seeding root from SELF_EVOLVE_ROOT_COMMIT=%s -> %s "
                    "(label=%s)",
                    seed_ref, worktree.short_sha(commit), branch,
                )
            else:
                commit, branch = worktree.monet_develop_tip()
                self.log.info(
                    "seeding root from origin/develop tip: %s "
                    "(no --baseline-logs given)",
                    worktree.short_sha(commit),
                )
        else:
            commit = worktree.current_monet_commit()
            branch = worktree.current_monet_branch()
        root_id, created_now = tree.bootstrap_root_if_absent(
            self.conn,
            campaign=self.cfg.campaign,
            subset=self.cfg.subset_label,
            branch_name=branch,
            commit_sha=commit,
            pipeline_id=self.pipeline_id,
        )
        if created_now:
            self.log.info(
                "no nodes yet for subset=%s; bootstrapped root %s from %s",
                self.cfg.subset_label, root_id, branch,
            )
            # Remember we own the bootstrap so we can fall back to it in
            # _select_parent and so Phase 2 knows to score it.
            self._bootstrapped_root_id: str | None = root_id
            # Write a placeholder works.md so the visualizer shows useful
            # context for the root (commit, branch, pending baseline) even
            # before Phase 2 finishes. _phase2_baseline overwrites this with
            # the final score + failing tasks once scoring completes.
            self._write_root_works_md(tree.get_node(self.conn, root_id), pending=True)
        else:
            # Lost the race — a sibling worker inserted between our
            # pre-check and the IMMEDIATE-transaction insert. Same
            # follow-up as the fast path (idempotent works.md backfill).
            self.log.info(
                "root for subset=%s appeared mid-bootstrap: %s (lost race to sibling)",
                self.cfg.subset_label, root_id,
            )
            # Best-effort backfill for legacy campaigns whose roots were
            # bootstrapped before this code path wrote works.md. Idempotent:
            # if the file already exists with the right path recorded, we
            # rewrite it from the latest DB state (cheap, ~1KB).
            self._backfill_root_works_md_if_missing(root_id)

    def _select_parent(self) -> None:
        # Supervisor handoff: if the parent was pre-claimed for us
        # (`--_parent-id <id>`), look it up directly and skip the
        # strategy. This is what stops two siblings spawned in the
        # same instant from racing on the strategy's tiebreak hash and
        # landing on the same parent (which would in turn race on the
        # claim table). Standalone runs (no override) fall through to
        # the configured strategy.
        if self.cfg.parent_id_override is not None:
            n = tree.get_node(self.conn, self.cfg.parent_id_override)
            if n is None:
                raise RuntimeError(
                    f"--_parent-id {self.cfg.parent_id_override!r} not found "
                    f"in tree (supervisor handoff is broken or stale)"
                )
            self.parent_node = n
            self.log.info(
                "parent: %s (branch=%s commit=%s score=%s subset=%s) "
                "[supervisor pre-claim]",
                n.id, n.branch_name, worktree.short_sha(n.commit_sha or "?"),
                n.score, n.subset,
            )
            tree.update_pipeline(
                self.conn, self.pipeline_id, parent_node_id=self.parent_node.id,
            )
            return

        try:
            strategy = parent_selection.get_strategy(self.cfg.parent_strategy)
            self.parent_node = strategy.select(
                self.conn,
                campaign=self.cfg.campaign,
                subset=self.cfg.subset_label,
                pipeline_id=self.pipeline_id,
            )
            self.log.info(
                "parent: %s (branch=%s commit=%s score=%s subset=%s)",
                self.parent_node.id, self.parent_node.branch_name,
                worktree.short_sha(self.parent_node.commit_sha or "?"),
                self.parent_node.score, self.parent_node.subset,
            )
            tree.update_pipeline(
                self.conn, self.pipeline_id, parent_node_id=self.parent_node.id,
            )
        except parent_selection.NoEligibleParentError:
            # Fall back to the bootstrapped root we just created if any.
            root_id = getattr(self, "_bootstrapped_root_id", None)
            if not root_id and (self.cfg.bootstrap_only or self.cfg.baseline_logs):
                # Bootstrap precursors are allowed to select an unscored root:
                # Phase 2 immediately scores it from the provided baseline logs
                # (or by running the baseline). This also handles the narrow
                # race where bootstrap_root_if_absent reports "lost race" to a
                # just-created root before any scored parent is eligible.
                root_id = self._existing_root_id()
            if not root_id:
                raise
            n = tree.get_node(self.conn, root_id)
            assert n
            self.parent_node = n
            tree.update_pipeline(
                self.conn, self.pipeline_id, parent_node_id=self.parent_node.id,
            )
            self.log.info(
                "no eligible scored parent yet — using bootstrapped root %s "
                "(will score it in Phase 2)", n.id,
            )

    def _create_worktree(self) -> None:
        assert self.parent_node and self.parent_node.commit_sha
        self.worktree = worktree.add_eval_worktree(
            pipeline_id=self.pipeline_id,
            parent_commit=self.parent_node.commit_sha,
            repo_root=self.cfg.repo_root,
        )
        tree.update_pipeline(
            self.conn, self.pipeline_id, worktree_path=str(self.worktree.eval_dir),
        )
        self.log.info(
            "worktree: %s (branch=%s)", self.worktree.eval_dir, self.worktree.monet_branch,
        )

    def _create_child_node(self) -> None:
        assert self.parent_node and self.worktree
        child_id = tree.new_id()
        # `commit_sha` starts at the parent's HEAD (the branch base); each
        # kept iteration advances it via `tree.append_commits`. `commits=[]`
        # at insert time — children don't claim the parent's commit as
        # "theirs"; only commits added by this node's iterations are recorded.
        tree.insert_node(
            self.conn,
            id=child_id,
            campaign=self.cfg.campaign,
            branch_name=self.worktree.monet_branch,
            commit_sha=self.parent_node.commit_sha,
            commits=[],
            parent_id=self.parent_node.id,
            subset=self.cfg.subset_label,
            status="in_progress",
            pipeline_id=self.pipeline_id,
        )
        tree.insert_node_edge(
            self.conn,
            campaign=self.cfg.campaign,
            parent_id=self.parent_node.id,
            child_id=child_id,
            edge_type=(
                "regression_resolve"
                if self.cfg.pipeline_kind == "regression_resolve"
                else "evolve"
            ),
            parent_role=(
                "regression_target"
                if self.cfg.pipeline_kind == "regression_resolve"
                else "evolve"
            ),
            pipeline_id=self.pipeline_id,
        )
        tree.update_pipeline(self.conn, self.pipeline_id, child_node_id=child_id)
        # Initialize per-node markdown stubs so the visualizer shows something
        # even mid-run.
        node_dir = tree.node_dir(self.cfg.reports_root, self.cfg.campaign, child_id)
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / "works.md").write_text(
            f"# {self.cfg.pipeline_kind} node `{child_id}`\n\n"
            f"- branch: `{self.worktree.monet_branch}`\n"
            f"- parent: `{self.parent_node.id}`\n\n"
            f"_pipeline {self.pipeline_id} in progress…_\n"
        )
        (node_dir / "effort.md").write_text(
            f"# Self-evolving effort: {self.worktree.monet_branch}\n\n"
            f"Pipeline `{self.pipeline_id}` started at {tree.utcnow_iso()}.\n\n"
        )
        tree.update_node(
            self.conn, child_id,
            works_md_path=str(node_dir / "works.md"),
            effort_md_path=str(node_dir / "effort.md"),
        )
        # Refresh AFTER the path columns are populated so callers (notably
        # _effort_append in Phase 3) see the real effort.md path rather than
        # the NULL from the bare insert.
        self.child_node = tree.get_node(self.conn, child_id)
        self.log.info("child node: %s", child_id)

    # ─── Phase 2: Baseline ────────────────────────────────────────────

    def _phase2_baseline(self) -> None:
        assert self.parent_node
        tree.update_pipeline(self.conn, self.pipeline_id, status="baseline")
        tree.heartbeat(self.conn, self.pipeline_id)

        # If parent already has a search eval for our subset, we're done.
        parent_search_eval = self._search_eval(self.parent_node)
        if parent_search_eval is not None:
            self.log.info(
                "parent %s already scored (%.4f) on subset=%s — skipping baseline",
                self.parent_node.id, parent_search_eval.score, parent_search_eval.subset_label,
            )
            self._populate_passing_pool()
            return

        # Need to score the parent. Either parse existing logs or run.
        adaptive_mode = (
            self.cfg.adaptive_subset_enabled and self.cfg.subset_label == "full"
        )
        if self.cfg.baseline_logs:
            self.log.info("using --baseline-logs %s as parent score", self.cfg.baseline_logs)
            result = eval_runner.parse_existing_result_dir(self.cfg.baseline_logs)
            # If the user provided a full-benchmark baseline but is running on
            # a smaller subset (e.g. smoke-10), recompute the score over just
            # the subset's tasks — the full mean isn't apples-to-apples.
            if (
                not adaptive_mode
                and self.cfg.subset_label != "full"
                and self.cfg.subset_tasks
            ):
                result = eval_runner.restrict_to_subset(
                    result,
                    subset_label=self.cfg.subset_label,
                    task_names=self.cfg.subset_tasks,
                )
                self.log.info(
                    "restricted baseline to subset=%s: %d tasks → score %.4f",
                    self.cfg.subset_label,
                    len(self.cfg.subset_tasks), result.score,
                )
        else:
            # Bootstrap-only mode skips worktree creation, so we run the
            # baseline harbor eval in the main repo. Harbor only reads
            # monet_code/ — it doesn't mutate it — so this is safe even with
            # the user's clone in active use.
            cwd = self.worktree.eval_dir if self.worktree else self.cfg.repo_root
            # Lift n_concurrent for the bootstrap precursor only. This is the
            # blocking baseline that gates a parallel campaign, so the wall-
            # clock win is felt by every subsequent worker. Read
            # `harbor.n_concurrent_bootstrap` from the YAML and pass it via the
            # env override that run_harbor.py honors. Mini-evals (Phase 3) and
            # final evals are unaffected.
            extra_env: dict[str, str] | None = None
            if self.cfg.bootstrap_only:
                bootstrap_nc = _bootstrap_n_concurrent(self.cfg.config_path)
                if bootstrap_nc is not None:
                    self.log.info(
                        "bootstrap precursor: lifting harbor n_concurrent to %d "
                        "for one-time baseline (mini-evals keep the default)",
                        bootstrap_nc,
                    )
                    extra_env = {"MONET_EVAL_HARBOR_N_CONCURRENT": str(bootstrap_nc)}
            baseline_tasks = [] if adaptive_mode else self.cfg.subset_tasks
            self.log.info("running baseline eval on subset=%s in %s", self.cfg.subset_label, cwd)
            with self._heartbeat_during_eval("baseline eval"):
                result = self._run_full(
                    config_path=self.cfg.config_path,
                    cwd=cwd,
                    subset=self.cfg.subset_label,
                    task_names=baseline_tasks,
                    job_name=f"baseline_{self.pipeline_id}",
                    extra_env=extra_env,
                    tee_log_path=getattr(self, "run_log_path", None),
                )

        if adaptive_mode and self.parent_node.parent_id is None:
            root_full = self._persist_node_eval_result(
                node_id=self.parent_node.id,
                eval_kind="root_full",
                result=result,
                metadata={"basis": "root_full"},
            )
            subset = self._adaptive_subset_for_root(result, root_node_id=self.parent_node.id)
            subset_result = eval_runner.restrict_to_subset(
                result,
                subset_label=self.cfg.subset_label,
                task_names=subset.tasks,
            )
            self._persist_node_eval_result(
                node_id=self.parent_node.id,
                eval_kind="subset_final",
                result=subset_result,
                metadata={
                    "basis": "adaptive_root_projection",
                    "root_full_eval_id": root_full.id,
                    "adaptive_label": subset.label,
                    "partially_solved_tasks": subset.partially_solved_tasks,
                    "unsolved_tasks": subset.unsolved_tasks,
                    "canary_tasks": subset.canary_tasks,
                },
            )
            result_for_pool = subset_result
        else:
            self._persist_node_eval_result(
                node_id=self.parent_node.id,
                eval_kind="subset_final",
                result=result,
                metadata={"basis": "baseline_subset"},
            )
            result_for_pool = result

        # Promote the parent/root metadata; eval details live in node_evals.
        # Also persist the scalar score onto the node row: the bootstrap/root
        # path previously left nodes.score NULL (only node_evals was written),
        # which made the merge selector report "no scored root" and skip merges
        # entirely (the root is the floor every merge candidate is compared
        # against). Use the same subset_final score search_eval_by_node reads, so
        # nodes.score and the eval agree.
        tree.update_node(
            self.conn,
            self.parent_node.id,
            status="completed",
            score=result_for_pool.score,
        )
        self.parent_node = tree.get_node(self.conn, self.parent_node.id)
        self.log.info(
            "parent score: %.4f (%d failing tasks recorded)",
            result_for_pool.score, len(result_for_pool.failed_task_names),
        )
        # If this parent is a root (parent_id IS NULL), refresh its works.md
        # with the final baseline summary now that we have a real score +
        # failing-tasks list. Non-root parents have iteration-driven
        # works.md written by _create_child_node and aren't touched here.
        if self.parent_node.parent_id is None:
            self._write_root_works_md(self.parent_node, pending=False)
        self._populate_passing_pool(result=result_for_pool)

    def _populate_passing_pool(self, result: eval_runner.EvalResult | None = None) -> None:
        """Fill self.passing_for_canary (Layer-3 canary picker) = parent solves
        UNION root full-eval solves, so a root-solved task can't silently regress
        unprobed. Gated by ATELIER_FULL_SOLVED_PRESERVATION. (legitimacy audit 2026-07-02)"""
        base: list[str] = []
        if result is not None:
            base = list(result.solved_tasks)
        else:
            assert self.parent_node
            parent_eval = self._search_eval(self.parent_node)
            if parent_eval is not None:
                base = list(parent_eval.solved_tasks)
            elif self.parent_node.job_log_path:
                try:
                    r = eval_runner.parse_existing_result_dir(Path(self.parent_node.job_log_path))
                    base = list(r.solved_tasks)
                except Exception:
                    base = []
        enabled = os.environ.get(
            "ATELIER_FULL_SOLVED_PRESERVATION", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        root_solved: list[str] = []
        if enabled:
            try:
                root_eval = tree.root_full_eval(self.conn, campaign=self.cfg.campaign)
                root_solved = list(getattr(root_eval, "solved_tasks", None) or [])
            except Exception as e:  # noqa: BLE001
                self.log.warning("passing pool: root_full_eval failed (%s)", e)
        self.passing_for_canary = _union_solved_pools(base, root_solved, enabled=enabled)

    # ─── Phase 3: Self-evolve loop ────────────────────────────────────

    def _sort_by_repair_value(self, tasks: list[str]) -> list[str]:
        """Return `tasks` sorted descending by `_task_repair_value`.

        Used by the resolver path so the prompt's "Regressions to investigate"
        block lists the highest-impact regressions first. Repair value is a
        function of historical resolver success / failure / fragility per
        `tree.task_outcome_stats`; tasks with no history score 1.0 (neutral).

        Stable: ties break alphabetically so the prompt is deterministic.
        """
        if not tasks:
            return []
        stats = tree.task_outcome_stats(
            self.conn,
            campaign=self.cfg.campaign,
            subset=self.cfg.subset_label,
        )
        return sorted(
            tasks,
            key=lambda t: (
                -regression_selection._task_repair_value(t, stats),
                t,
            ),
        )

    def _acquire_claimed_tasks(self) -> list[str]:
        """Return the failing tasks this pipeline should attempt to resolve.

        Two-path claim flow:
          - Supervisor-driven runs: the supervisor already pre-claimed our
            failing tasks atomically before spawning us. Read them straight
            off the claims table by `pipeline_id` and skip `pick_and_claim`
            (calling it would falsely treat our own pre-claims as "taken
            by someone else" and return []).
          - Standalone runs: no pre-claim exists, so claim from the
            campaign-wide shared pool.

        For `regression_resolve` pipelines, both paths return the claimed
        tasks sorted by repair value (highest first) so the prompt's
        "Regressions to investigate" listing is stable and prioritized.

        Factored out of `_phase3_loop` so unit tests can pin down both
        branches without standing up a full worktree / child node.
        """
        assert self.parent_node
        pre_claims = [
            c.failure_task for c in tree.list_active_claims(
                self.conn, pipeline_id=self.pipeline_id,
            )
        ]
        if pre_claims:
            if self.cfg.pipeline_kind == "regression_resolve":
                pre_claims = self._sort_by_repair_value(pre_claims)
            self.log.info(
                "using %d pre-claimed task(s) from supervisor: %s",
                len(pre_claims), pre_claims,
            )
            return pre_claims
        if self.cfg.pipeline_kind == "regression_resolve":
            candidates = list(self.parent_node.regressed_tasks)
            subset_filter = set(self.cfg.subset_tasks or [])
            if subset_filter:
                candidates = [t for t in candidates if t in subset_filter]
            # Sort candidates by repair value first so `try_claim_tasks`
            # iterates highest-priority tasks first when contending with
            # sibling resolvers.
            candidates = self._sort_by_repair_value(candidates)
            # Resolver claims the full regression set by default so a single
            # worker owns the parent's whole "what got broken" list. The
            # `--regression-max-claim` knob caps this for operators worried
            # about per-iteration mini-eval cost; 0 / None means no cap.
            max_claim = self.cfg.regression_max_claim
            k = (
                len(candidates)
                if not max_claim or max_claim <= 0
                else min(max_claim, len(candidates))
            )
            claimed = tree.try_claim_tasks(
                self.conn,
                campaign=self.cfg.campaign,
                subset=self.cfg.subset_label,
                candidate_tasks=candidates,
                k=k,
                pipeline_id=self.pipeline_id,
                parent_id=self.parent_node.id,
                claim_kind="regression_resolve",
            )
            # `try_claim_tasks` preserves input order, but re-sort defensively
            # in case a future refactor changes that contract.
            return self._sort_by_repair_value(claimed)
        subset_filter = self.cfg.subset_tasks or None
        return pool.pick_and_claim(
            self.conn,
            campaign=self.cfg.campaign,
            subset=self.cfg.subset_label,
            subset_filter=subset_filter,
            k=self.cfg.n_failure_tasks,
            pipeline_id=self.pipeline_id,
            parent_id=self.parent_node.id,
        )

    def _phase3_loop(self) -> None:
        assert self.parent_node and self.child_node and self.worktree
        tree.update_pipeline(self.conn, self.pipeline_id, status="evolving")

        self.claimed_tasks = self._acquire_claimed_tasks()
        if not self.claimed_tasks:
            self.log.warning(
                "no failing tasks available to claim under parent %s "
                "(possibly all already claimed by sibling pipelines)",
                self.parent_node.id,
            )
            tree.update_pipeline(
                self.conn, self.pipeline_id, selected_tasks_json=[]
            )
            return
        tree.update_pipeline(
            self.conn, self.pipeline_id,
            selected_tasks_json=self.claimed_tasks,
        )
        self.log.info("claimed tasks: %s", self.claimed_tasks)

        # Iteration loop.
        for i in range(1, self.cfg.max_loop_iters + 1):
            tree.heartbeat(self.conn, self.pipeline_id)
            tree.update_pipeline(self.conn, self.pipeline_id, current_iteration=i)
            self.log.info("=== iteration %d/%d ===", i, self.cfg.max_loop_iters)

            outcome = self._run_iteration(i)
            self.iterations.append(outcome)

            if outcome.both_targets_pass(self.claimed_tasks) and outcome.preservation_passed():
                self.log.info(
                    "iteration %d: claimed and preservation tasks pass — early exit", i,
                )
                break
            if (
                _qd_archive_on()
                and not outcome.reverted
                and outcome.both_targets_pass(self.claimed_tasks)
            ):
                # QD: claimed target(s) solved (even if preservation regressed) —
                # lock the specialist win; further mutation risks losing it.
                self.log.info(
                    "iteration %d: QD specialist win locked (claimed solved) — "
                    "stop mutating", i,
                )
                break

    def _persist_iteration_outcome(
        self,
        outcome: IterationOutcome,
        *,
        stage: str,
        outcome_name: str,
        reverted_shas: list[str] | None = None,
    ) -> None:
        if not self.child_node:
            return
        claimed_rewards = {
            t: float(outcome.rewards_per_task.get(t, 0.0))
            for t in self.claimed_tasks
        }
        canary_rewards = {
            t: float(outcome.rewards_per_task.get(t, 0.0))
            for t in outcome.canary_tasks
        }
        tree.upsert_iteration_outcome(
            self.conn,
            campaign=self.cfg.campaign,
            pipeline_id=self.pipeline_id,
            node_id=self.child_node.id,
            iteration=outcome.iteration,
            stage=stage,
            outcome=outcome_name,
            reason=outcome.reason,
            committed_shas=outcome.committed_shas,
            reverted=outcome.reverted,
            reverted_shas=reverted_shas or ([] if not outcome.reverted else outcome.committed_shas),
            mini_eval_job_path=(
                str(outcome.mini_eval_job_dir) if outcome.mini_eval_job_dir else None
            ),
            mini_eval_score=outcome.mini_eval_score,
            mini_eval_n_trials=outcome.mini_eval_n_trials,
            mini_eval_n_errors=outcome.mini_eval_n_errors,
            claimed_rewards=claimed_rewards if outcome.rewards_per_task else None,
            canary_rewards=canary_rewards if outcome.rewards_per_task else None,
            canary_tasks=outcome.canary_tasks,
            failed_canaries=outcome.failed_canary_tasks,
            review_duration_ms=outcome.review_duration_ms or None,
            review_error=outcome.review_error,
        )

    def _run_full(self, **kwargs):
        """Full-set eval dispatcher. Default ("best") is byte-identical to a
        direct ``eval_runner.run_full(**kwargs)`` call. When
        ``fullset_eval_metric == "avg"`` (env MONET_EVAL_FULLSET_METRIC) AND the
        task list is known (non-empty), score as avg@k instead of best-of-N so
        node scores / acceptance optimise the leaderboard metric.

        A WHOLE-JOB infrastructure failure (cluster/tunnel outage that breaks
        every trial) is transient and not the candidate's fault, so retry the
        eval ``MONET_EVAL_INFRA_RETRIES`` times (default 2, exponential-ish
        backoff) before letting the :class:`EvalInfrastructureError` propagate
        — which would otherwise mark an otherwise-good node ``failed`` and throw
        away the proposer's work. After retries are exhausted, behaviour is
        identical to before (the error propagates).
        """
        metric = os.environ.get(
            "MONET_EVAL_FULLSET_METRIC",
            getattr(self.cfg, "fullset_eval_metric", "best"),
        ).strip().lower()
        task_names = kwargs.get("task_names") or []

        def _call_once():
            if metric == "avg" and not task_names:
                raise ValueError(
                    "MONET_EVAL_FULLSET_METRIC=avg requires an explicit task "
                    "list (MONET_EVAL_FULLSET_TASKS); refusing to silently fall "
                    "back to best-of-N scoring. (legitimacy audit 2026-07-02)"
                )
            if metric == "avg" and task_names:
                k = int(getattr(self.cfg, "fullset_eval_k_samples", 5))
                return eval_runner.run_full_avg_k(
                    config_path=kwargs.get("config_path"),
                    cwd=kwargs.get("cwd"),
                    task_names=task_names,
                    k_samples=k,
                    job_name=kwargs.get("job_name"),
                    extra_env=kwargs.get("extra_env"),
                    tee_log_path=kwargs.get("tee_log_path"),
                )
            return eval_runner.run_full(**kwargs)

        attempts = _infra_eval_retries() + 1
        for attempt in range(1, attempts + 1):
            try:
                return _call_once()
            except eval_runner.EvalInfrastructureError:
                if attempt >= attempts:
                    raise
                delay = min(120, 20 * attempt)
                self.log.warning(
                    "full eval whole-job INFRA failure (attempt %d/%d); "
                    "transient — retrying in %ds before marking the node failed",
                    attempt, attempts, delay,
                )
                time.sleep(delay)

    def _run_mini_eval(self, mini_tasks: list[str], *, iteration: int, _k_override: int | None = None):
        """Run the inner-loop mini-eval, denoised to an avg@k majority vote when
        ``cfg.mini_eval_k_samples > 1``.

        Returns an object exposing ``job_dir`` / ``score`` / ``n_trials`` /
        ``n_errors`` / ``task_rewards`` (the fields ``_run_iteration`` consumes),
        so the call site is identical whether we ran k=1 (legacy single trial)
        or k>=3 (majority vote per task). Reuses ``run_subset_sampled`` so the
        candidate-commit injection / Express relay / cluster path is unchanged.
        """
        env_k = os.environ.get("MONET_EVAL_MINI_EVAL_K", "").strip()
        k = _k_override if _k_override else max(1, int(env_k) if env_k.isdigit() else int(self.cfg.mini_eval_k_samples))
        job_name = f"iter_{iteration}_{self.pipeline_id}"
        tee = getattr(self, "run_log_path", None)
        if k <= 1:
            res = eval_runner.run_subset(
                config_path=self.cfg.config_path, cwd=self.worktree.eval_dir,
                task_names=mini_tasks, job_name=job_name, tee_log_path=tee,
            )
            # k=1 has no sampling spread; pass-rate degrades to the binary reward.
            try:
                res.task_pass_rates = dict(res.task_rewards)
            except Exception:  # noqa: BLE001 — never break the legacy path
                pass
            return res
        sres = eval_runner.run_subset_sampled(
            config_path=self.cfg.config_path, cwd=self.worktree.eval_dir,
            task_names=mini_tasks, k_samples=k, job_name=job_name, tee_log_path=tee,
        )
        # Majority vote over k samples -> denoised 0/1 per task. Kept as the
        # binary view for legacy win/canary counting; the fractional pass-rate
        # below is the avg@k-aligned signal the gate now prefers.
        task_rewards = {
            t: (1.0 if rate >= 0.5 else 0.0) for t, (rate, _n) in sres.rates.items()
        }
        # Preserve the raw fractional pass-rate as the graded progress signal.
        task_pass_rates = {t: float(rate) for t, (rate, _n) in sres.rates.items()}
        n_trials = sum(n for _, (_r, n) in sres.rates.items())
        # p1_metric: score on FRACTIONAL avg@k pass-rates (tracks the real
        # objective) instead of the binary majority collapse, when enabled.
        if _fractional_gate_on() and task_pass_rates:
            score = sum(task_pass_rates.values()) / len(task_pass_rates)
        else:
            score = (
                sum(task_rewards.values()) / len(task_rewards) if task_rewards else 0.0
            )
        return SimpleNamespace(
            job_dir=sres.job_dir, score=score, n_trials=n_trials,
            n_errors=0, task_rewards=task_rewards, task_pass_rates=task_pass_rates,
        )

    def _run_mini_eval_screened(self, mini_tasks: list[str], *, iteration: int):
        """k-screen with escalate-on-ambiguity. Runs the cheap k screen; if the
        verdict is BORDERLINE (coin-flip band 0.34-0.67) and ATELIER_GATE_ESCALATE_K>k
        is set, re-runs the SAME tasks at the higher k before the accept/revert
        decision. Protects the rare real improver from k-noise without paying high-k
        on clear-cut cases. Default OFF (env unset => behaviour unchanged)."""
        res = self._run_mini_eval(mini_tasks, iteration=iteration)
        env_k = os.environ.get("MONET_EVAL_MINI_EVAL_K", "").strip()
        base_k = max(1, int(env_k) if env_k.isdigit() else int(self.cfg.mini_eval_k_samples))
        try:
            esc_k = int(os.environ.get("ATELIER_GATE_ESCALATE_K", "0") or "0")
        except ValueError:
            esc_k = 0
        score = getattr(res, "score", None)
        if esc_k > base_k and score is not None and 0.34 <= float(score) <= 0.67:
            self.log.info(
                "[gate] borderline mini-eval score=%.2f at k=%d -> escalating to k=%d",
                float(score), base_k, esc_k,
            )
            return self._run_mini_eval(mini_tasks, iteration=iteration, _k_override=esc_k)
        return res

    def _mini_eval_net_gain(self, outcome: IterationOutcome) -> int:
        return self._resolver_mini_eval_stats(outcome).net_gain

    def _resolver_mini_eval_stats(self, outcome: IterationOutcome) -> ResolverMiniEvalStats:
        claimed_total = len(self.claimed_tasks)
        claimed_wins = sum(
            1 for t in self.claimed_tasks
            if outcome.rewards_per_task.get(t, 0.0) >= 1.0
        )
        canary_total = len(outcome.canary_tasks)
        canary_failures = len(outcome.failed_canary_tasks)
        claimed_win_rate = claimed_wins / claimed_total if claimed_total else 0.0
        canary_failure_rate = canary_failures / canary_total if canary_total else 0.0
        return ResolverMiniEvalStats(
            claimed_total=claimed_total,
            claimed_wins=claimed_wins,
            canary_total=canary_total,
            canary_failures=canary_failures,
            claimed_win_rate=claimed_win_rate,
            canary_failure_rate=canary_failure_rate,
            rate_margin=claimed_win_rate - canary_failure_rate,
            net_gain=claimed_wins - canary_failures,
        )

    def _resolver_mini_eval_stats_summary(self, stats: ResolverMiniEvalStats) -> str:
        return (
            f"claimed_win_rate={stats.claimed_win_rate:.2f} "
            f"({stats.claimed_wins}/{stats.claimed_total}); "
            f"canary_failure_rate={stats.canary_failure_rate:.2f} "
            f"({stats.canary_failures}/{stats.canary_total}); "
            f"rate_margin={stats.rate_margin:.2f}; net_gain={stats.net_gain}"
        )

    def _resolver_mini_eval_rejection_reason(
        self,
        outcome: IterationOutcome,
        stats: ResolverMiniEvalStats,
    ) -> str:
        failures: list[str] = []
        max_failures = self.cfg.resolver_mini_eval_max_canary_failures
        if max_failures and stats.canary_failures > max_failures:
            failures.append(
                f"canary_failures {stats.canary_failures} above cap {max_failures}"
            )
        min_claimed_rate = self.cfg.resolver_mini_eval_min_claimed_win_rate
        if stats.claimed_win_rate < min_claimed_rate:
            failures.append(
                f"claimed_win_rate {stats.claimed_win_rate:.2f} below "
                f"{min_claimed_rate:.2f}"
            )
        max_canary_rate = self.cfg.resolver_mini_eval_max_canary_failure_rate
        if stats.canary_failure_rate > max_canary_rate:
            failures.append(
                f"canary_failure_rate {stats.canary_failure_rate:.2f} above "
                f"{max_canary_rate:.2f}"
            )
        min_margin = self.cfg.resolver_mini_eval_min_rate_margin
        if stats.rate_margin < min_margin:
            failures.append(f"rate_margin {stats.rate_margin:.2f} below {min_margin:.2f}")
        failed = ", ".join(failures) if failures else "unknown predicate"
        return (
            f"guard tripped (Layer 3): resolver mini-eval policy failed "
            f"({failed}); {self._resolver_mini_eval_stats_summary(stats)}; "
            f"failed canaries: {outcome.failed_canary_tasks}"
        )

    def _reasoned_verdict_decision(self, outcome: IterationOutcome) -> bool | None:
        """v9 slot-upgrade: a reasoned verdict over structured evidence replaces
        the binary ``canary_passed`` screen. Env-gated (ATELIER_REASONED_VERDICT);
        returns None on any issue so the caller falls back to the scalar path.

        Evidence threaded in (v9): parent before-rates + after-rates per claimed
        task, the trace_qc fault-localization gradient that motivated the change,
        the diff stat, and the edited SURFACE (code vs skill). The verdict also
        emits a forward lesson + directive (gate-as-activation) recorded for the
        next proposal.
        """
        try:
            from . import reasoned_verdict as rv
        except Exception:
            return None
        if not rv.reasoned_verdict_on():
            return None
        try:
            before = self._parent_before_rates()
            claimed = set(self.claimed_tasks or [])
            guards = set(getattr(self, "passing_for_canary", []) or [])

            def _before(t):
                # Prefer the parent's recorded rate; else fall back to what the
                # pipeline KNOWS: claimed tasks are (by definition) the parent's
                # FAILURES (before ~0), guards are drawn from its PASSING set (~1.0).
                # Without this the verdict only saw after-rates and could never
                # confirm a 0->pass gain, so it rejected even genuine improvements.
                b = before.get(t)
                if b is not None:
                    return b
                if t in claimed:
                    return 0.0
                if t in guards:
                    return 1.0
                return None

            deltas: list[dict] = [
                {"task": t, "before_rate": _before(t), "after_rate": r}
                for t, r in (outcome.pass_rates_per_task or {}).items()
            ]
            for t in (outcome.failed_canary_tasks or []):
                deltas.append({"task": f"{t} (regression guard)", "before_rate": 1.0, "after_rate": 0.0})
            surface = self._child_diff_surface()
            # Effective mini-eval k (env override wins) so the GATE knows the
            # true confidence of the rates it's reading.
            _envk = os.environ.get("MONET_EVAL_MINI_EVAL_K", "").strip()
            eff_k = (int(_envk) if _envk.isdigit()
                     else int(getattr(self.cfg, "mini_eval_k_samples", 0) or 0)) or None
            # Collective-knowledge activation: feed the campaign-wide digest of how
            # prior edits fared (incl. failed/regressed) into the GATE so the
            # decision aggregates history, not just this node's noisy numbers.
            ck = self._collective_knowledge_for_gate() if _knowledge_gate_on() else None
            evidence = rv.render_evidence(
                task_deltas=deltas,
                diff_summary=self._child_diff_summary(),
                gradient_digest=getattr(self, "_last_trace_qc_digest", "") or None,
                k_samples=eff_k,
                surface=surface,
                collective_knowledge=ck,
            )
            verdict = rv.decide(evidence, surface=surface)
            if verdict is None:
                return None
            _routed = os.environ.get("ATELIER_ROUTED_CODE", "0").strip() == "1"
            _n_reg = len(outcome.failed_canary_tasks or [])
            _nd = _risk_class_downgrade(verdict.decision, surface, _n_reg, routed=_routed)
            if _nd != verdict.decision:
                from dataclasses import replace as _dc_replace
                self.log.info("risk-class guard: %s->%s (surface=%s, %d regression(s))",
                              verdict.decision, _nd, surface, _n_reg)
                verdict = _dc_replace(verdict, decision=_nd)
            self.log.info(
                "reasoned verdict: %s (conf %.2f, surface=%s) — %s",
                verdict.decision, verdict.confidence, verdict.surface or "?", verdict.rationale,
            )
            # gate-as-activation: persist the textual gradient for the next proposal
            # regardless of decision (a REJECT still propagates its lesson).
            self._record_verdict_lesson(verdict)
            # ACCEPT/ARCHIVE proceed past the screen; ARCHIVE is sorted into a
            # stepping stone downstream by the hybrid archive. REJECT stops here.
            passed = verdict.accept or verdict.archive
            # Cross-benchmark generality gate (ATELIER_CROSS_BENCH_GATE): a
            # candidate that passed the in-domain screen must also PRESERVE on a
            # HELD-OUT benchmark it was not evolved against, else it is vetoed
            # here. No-op unless enabled AND configured (held-out tasks +
            # baseline present); never blocks the loop.
            if passed:
                veto = self._cross_bench_veto()
                if veto is not None:
                    ok, reason = veto
                    self.log.info("[cross-bench] %s", reason)
                    if not ok:
                        return False
            return passed
        except Exception as exc:  # noqa: BLE001 — never block the loop
            self.log.warning("reasoned-verdict wiring failed (%s); scalar fallback", exc)
            return None

    def _cross_bench_veto(self):
        """Held-out cross-benchmark generality gate.

        Returns ``None`` when disabled / unconfigured / no signal (caller leaves
        the in-domain decision unchanged), else ``(ok, reason)`` where
        ``ok is False`` vetoes (reject) a candidate that REGRESSED on the
        held-out benchmark. The parent/root held-out baseline is supplied
        out-of-band (``ATELIER_HELDOUT_BASELINE``) so no parent re-checkout is
        needed. Fully defensive: any failure => ``None`` (never blocks the loop).
        """
        try:
            from . import cross_bench as cbx
            if not cbx.gate_enabled():
                return None
            tasks = cbx.heldout_tasks()
            baseline = cbx.heldout_baseline()
            if not tasks or not baseline:
                return None
            base_cfg = eval_runner.CodingBenchEvalConfig.from_self_evolve_config(
                self.cfg.config_path
            )
            child_rates = cbx.run_heldout_rates(
                base_cfg,
                tasks,
                config_path=self.cfg.config_path,
                cwd=self.worktree.eval_dir,
                tee_log_path=getattr(self, "run_log_path", None),
            )
            v = cbx.cross_bench_verdict(
                baseline, child_rates, margin=cbx.regression_margin()
            )
            return (v.verdict != "regress"), v.detail
        except Exception as exc:  # noqa: BLE001 — never block the loop
            self.log.warning("cross-bench gate skipped (%s)", exc)
            return None

    def _parent_before_rates(self) -> dict[str, float]:
        """Best-effort parent per-task pass-rates for the claimed tasks, so the
        verdict sees true before->after gain (not just after-rates). Returns {}
        if no recorded rates are available; never raises."""
        try:
            node = self.parent_node
            for attr in ("task_pass_rates", "pass_rates_per_task", "task_rewards"):
                rates = getattr(node, attr, None)
                if isinstance(rates, dict) and rates:
                    return {t: float(v) for t, v in rates.items()}
        except Exception:
            pass
        return {}

    def _worktree_monet_dir(self):
        """The per-node WORKTREE's monet_code dir (where the proposer's edit lives),
        not the shared repo. The reasoned verdict runs at the mini-eval screen BEFORE
        the iteration commit, so the candidate change is uncommitted in this worktree."""
        wt = getattr(self, "worktree", None)
        for attr in ("monet_dir", "eval_dir"):
            d = getattr(wt, attr, None) if wt else None
            if d:
                return d
        return None

    def _child_diff_text(self) -> str | None:
        """Full unified diff of the candidate vs the parent commit, taken from the
        per-node worktree's working tree (so it captures the proposer's change whether
        or not it has been committed yet). Used to classify the edited surface and to
        give the reasoned verdict the actual diff."""
        try:
            import subprocess
            md = self._worktree_monet_dir()
            if not (self.parent_node and self.parent_node.commit_sha and md):
                return None
            r = subprocess.run(
                ["git", "-C", str(md), "diff", str(self.parent_node.commit_sha)],
                capture_output=True, text=True, timeout=30,
            )
            return (r.stdout or "").strip() or None
        except Exception:
            return None

    def _child_diff_surface(self) -> str:
        """Classify the candidate's edited surface: code / skill / mixed / none."""
        try:
            from . import generalization
            return generalization.classify_diff_surface(self._child_diff_text() or "")
        except Exception:
            return ""

    def _record_verdict_lesson(self, verdict) -> None:
        """Append the verdict's transferable lesson + directive to a campaign-wide
        JSONL store (gate-as-activation forward-flow). Never raises."""
        if not (getattr(verdict, "lesson", "") or getattr(verdict, "next_directive", "")):
            return
        try:
            import json as _json, time as _time
            rec = {
                "ts": _time.time(),
                "node": getattr(self.child_node, "id", None),
                "surface": getattr(verdict, "surface", ""),
                "decision": verdict.decision,
                "lesson": verdict.lesson,
                "next_directive": verdict.next_directive,
            }
            with open(self.campaign_dir / "verdict_lessons.jsonl", "a") as fh:
                fh.write(_json.dumps(rec) + "\n")
        except Exception as exc:  # noqa: BLE001
            self.log.debug("verdict lesson record skipped: %s", exc)

    def _child_diff_summary(self) -> str | None:
        """Best-effort ``git diff --stat`` of the candidate (per-node worktree) vs the
        parent commit — captures uncommitted screen-time changes (see _child_diff_text)."""
        try:
            import subprocess
            md = self._worktree_monet_dir()
            if not (self.parent_node and self.parent_node.commit_sha and md):
                return None
            r = subprocess.run(
                ["git", "-C", str(md), "diff", "--stat", str(self.parent_node.commit_sha)],
                capture_output=True, text=True, timeout=30,
            )
            return (r.stdout or "").strip()[:2000] or None
        except Exception:
            return None

    def _mini_eval_accepts(self, outcome: IterationOutcome) -> bool:
        if self.cfg.pipeline_kind != "regression_resolve":
            rv = self._reasoned_verdict_decision(outcome)
            if rv is not None:
                return rv
            return outcome.canary_passed
        stats = self._resolver_mini_eval_stats(outcome)
        max_failures = self.cfg.resolver_mini_eval_max_canary_failures
        if max_failures and stats.canary_failures > max_failures:
            return False
        return (
            stats.claimed_win_rate >= self.cfg.resolver_mini_eval_min_claimed_win_rate
            and (
                stats.canary_failure_rate
                <= self.cfg.resolver_mini_eval_max_canary_failure_rate
            )
            and stats.rate_margin >= self.cfg.resolver_mini_eval_min_rate_margin
        )

    def _qd_accept_as_specialist(self, outcome: "IterationOutcome") -> bool:
        """QD (ATELIER_QD_ARCHIVE): keep a variant that CRACKS a claimed task the
        parent never solved — even if a regression-guard canary tripped — as a
        lossy specialist for the merge archive. The bar is deliberately low:
        strictly above ``_qd_solved_threshold()`` (default 0.0 => any seed passed)
        AND the parent had not already solved it. Archiving is safe (archived is
        excluded from best/tip and exploit-selection), so a spurious specialist
        costs only a merge attempt while missing a real one forecloses the only
        path off the plateau — hence archive liberally; merge-time confirmation is
        the stricter gate. Fires only when the normal gate would otherwise revert."""
        if not _qd_archive_on():
            return False
        if self.cfg.pipeline_kind == "regression_resolve":
            return False
        rates = outcome.pass_rates_per_task or {}
        claimed = set(self.claimed_tasks or [])
        thr = _qd_solved_threshold()
        parent_solved = set(self.parent_node.solved_tasks or []) if self.parent_node else set()
        newly_solved = [
            t for t in claimed
            if rates.get(t, 0.0) > thr and t not in parent_solved
        ]
        if not newly_solved:
            return False
        self.log.info(
            "QD specialist: newly solved claimed task(s) %s despite canary "
            "regression %s — keeping (not reverting) for the merge archive",
            sorted(newly_solved), outcome.failed_canary_tasks,
        )
        return True

    def _resolver_final_accepts(
        self,
        *,
        score_improved: bool,
        improved_tasks: list[str],
        regressed_tasks: list[str],
    ) -> tuple[bool, int]:
        net_gain = len(improved_tasks) - len(regressed_tasks)
        if self.cfg.pipeline_kind != "regression_resolve":
            return score_improved, net_gain
        return (
            score_improved and net_gain >= self.cfg.resolver_final_min_net_gain,
            net_gain,
        )

    def _archive_mini_eval_result(self, job_dir: Path, iteration: int) -> Path:
        """Copy mini-eval artifacts under the campaign report dir.

        Later iteration prompts point at this archive as a Harbor job dir and
        instruct the worker to read trial transcripts, verifier output, and
        trial logs. Keep the full mini-eval job tree so those paths remain
        valid after the temporary worktree is cleaned up.
        """
        if not self.child_node:
            return job_dir
        result = job_dir / "result.json"
        if not result.is_file():
            return job_dir
        dest = (
            tree.node_dir(self.cfg.reports_root, self.cfg.campaign, self.child_node.id)
            / "evals"
            / f"mini_iter_{iteration}_{self.pipeline_id}"
        )
        try:
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(job_dir, dest, symlinks=True)
            meta = {
                "source_job_dir": str(job_dir),
                "pipeline_id": self.pipeline_id,
                "iteration": iteration,
            }
            (dest / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            return dest
        except OSError:
            self.log.warning("could not archive mini-eval result %s", job_dir, exc_info=True)
            return job_dir

    def _run_iteration(self, i: int) -> IterationOutcome:
        assert self.worktree and self.child_node
        outcome = IterationOutcome(iteration=i)
        self._persist_iteration_outcome(outcome, stage="analyze", outcome_name="running")

        # Snapshot HEAD so we can revert this iteration's commits if needed.
        pre_iter_sha = worktree.head_sha(self.worktree.monet_dir)

        # Locate input logs for the analyze prompt.
        input_logs = self._iteration_input_logs(i)

        # ─── Step 1: Analyze (read-only) ────────────────────────────
        analyze_prompt = self._render_analyze_prompt(i, input_logs)
        self._save_prompt(i, "analyze", analyze_prompt)
        analyze = meta_agent.run(
            analyze_prompt,
            workspace=self.worktree.eval_dir,
            log_path=self.cursor_log_dir / f"iter_{i}_analyze.log",
            model=self.cfg.meta_model,
            plan_mode=True,
            timeout_s=self.cfg.analyze_timeout_s,
            reasoning_effort=self.cfg.meta_effort,
        )
        if analyze.error:
            self._effort_append(
                f"### Iteration {i}\n\n**analyze failed**: {analyze.error}\n"
            )
            outcome.reason = f"analyze failed: {analyze.error}"
            self._persist_iteration_outcome(
                outcome, stage="analyze", outcome_name="analyze_failed",
            )
            return outcome

        plan_path = self.prompts_dir_path / f"iter_{i}_plan.md"
        plan_text = analyze.text or ""
        plan_path.write_text(plan_text or "_(empty plan)_\n")
        self.log.info("plan written: %s (%d chars)", plan_path, len(plan_text))

        # Sanity-check the plan before handing it off to the implement step.
        # A short / header-less plan typically means cursor-agent emitted the
        # plan body via the CreatePlan tool with a placeholder, never wrote a
        # final assistant text, and the wrapper had nothing real to extract.
        # See cursor_agent._select_final_text for the channel priority.
        if not _looks_like_real_plan(plan_text):
            preview = plan_text.strip().replace("\n", " ")[:200]
            reasons = ", ".join(_invalid_plan_reasons(plan_text))
            msg = (
                f"analyze produced no usable plan ({len(plan_text)} chars, "
                f"{reasons}); preview: {preview!r}"
            )
            self.log.warning("%s", msg)
            self._effort_append(f"### Iteration {i}\n\n**analyze failed**: {msg}\n")
            outcome.reason = f"analyze failed: {msg}"
            self._persist_iteration_outcome(
                outcome, stage="analyze", outcome_name="analyze_failed",
            )
            return outcome

        # ─── Step 2: Implement ────────────────────────────────────
        self._persist_iteration_outcome(outcome, stage="implement", outcome_name="running")
        implement_prompt = self._render_implement_prompt(i, plan_path, plan_text)
        self._save_prompt(i, "implement", implement_prompt)
        implement_log_path = self.cursor_log_dir / f"iter_{i}_implement.log"
        implement = meta_agent.run(
            implement_prompt,
            workspace=self.worktree.eval_dir,
            log_path=implement_log_path,
            model=self.cfg.meta_model,
            plan_mode=False,
            timeout_s=self.cfg.implement_timeout_s,
            reasoning_effort=self.cfg.meta_effort,
        )
        if implement.error:
            self._effort_append(
                f"### Iteration {i}\n\n**implement failed**: {implement.error}\n"
            )
            outcome.reason = f"implement failed: {implement.error}"
            self._persist_iteration_outcome(
                outcome, stage="implement", outcome_name="implement_failed",
            )
            return outcome

        # ─── Step 3: Review + test + commit ─────────────────────────
        self._persist_iteration_outcome(outcome, stage="review", outcome_name="running")
        review_prompt = self._render_review_prompt(i, pre_iter_sha)
        self._save_prompt(i, "review", review_prompt)
        review_started = time.monotonic()
        review = meta_agent.run(
            review_prompt,
            workspace=self.worktree.eval_dir,
            log_path=self.cursor_log_dir / f"iter_{i}_review.log",
            model=self.cfg.meta_model,
            plan_mode=False,
            timeout_s=self.cfg.review_timeout_s,
            reasoning_effort=self.cfg.meta_effort,
        )
        outcome.review_duration_ms = int((time.monotonic() - review_started) * 1000)
        outcome.review_error = review.error
        if review.error:
            self.log.warning("review reported error: %s (continuing)", review.error)
        self.log.info(
            "review finished in %.1fs (timeout=%s, error=%s)",
            outcome.review_duration_ms / 1000,
            outcome.review_timed_out,
            review.error or "none",
        )

        # Did the agent commit anything?
        new_shas = worktree.commits_since(self.worktree.monet_dir, pre_iter_sha)
        outcome.committed_shas = new_shas
        if not new_shas:
            self._effort_append(
                f"### Iteration {i}\n\n"
                f"**no-op iteration**: no commits added.\n"
                f"{_review_summary_md(outcome, produced_commit=False)}"
            )
            outcome.reason = "no commits"
            self._persist_iteration_outcome(
                outcome, stage="review", outcome_name="no_commits",
            )
            return outcome

        # ─── Layer 2 guard: static diff scan ────────────────────────
        # Assigned unconditionally so the Layer-3 canary picker (below) works even
        # when the generalization guard is disabled (--no-generalization-guard).
        diff_text: str | None = None
        if self.cfg.guard_enabled:
            diff_text = worktree.diff_against(self.worktree.monet_dir, pre_iter_sha)
            test_outputs = self._read_test_outputs(input_logs)
            violations = generalization.scan_diff(
                diff_text,
                claimed_tasks=self.claimed_tasks,
                test_outputs=test_outputs,
            )
            # Additive-scope constraint: reject destructive (non-additive) diffs
            # that rewrite the shared core, so the iteration reverts and the next
            # one is steered toward additive, isolated changes (see
            # generalization.scan_diff_locality / v5 regression evidence).
            if _additive_scope_on():
                violations = violations + generalization.scan_diff_locality(
                    diff_text, max_deletions=_additive_max_deletions(),
                )
            # Always-on ANTI-CHEAT scan (eval/test-file tampering, weakened
            # assertions, hard-coded answers). (legitimacy audit 2026-07-02)
            try:
                from atelier.anti_cheat import scan_diff_for_eval_tampering
                cheat = scan_diff_for_eval_tampering(diff_text)
                if cheat:
                    violations = violations + [f"anti-cheat: {c}" for c in cheat]
            except Exception:  # noqa: BLE001
                pass
            outcome.guard_violations = violations
            if violations:
                self.guard_violated_ever = True
                worktree.reset_to(self.worktree.monet_dir, pre_iter_sha)
                outcome.reverted = True
                outcome.reason = f"guard tripped (Layer 2): {len(violations)} violations"
                self._effort_append(
                    f"### Iteration {i}\n\n"
                    f"{_review_summary_md(outcome, produced_commit=True)}"
                    f"**Generalization-guard rejected** (Layer 2, static scan) — "
                    f"{len(violations)} violation(s); reverted commits {new_shas}.\n\n"
                    + generalization.format_violations_for_prompt(violations) + "\n"
                )
                if self.cfg.guard_strict:
                    outcome.reason += " (--generalization-strict: fatal)"
                    raise RuntimeError(outcome.reason)
                self._persist_iteration_outcome(
                    outcome, stage="guard", outcome_name="reverted_layer2",
                    reverted_shas=new_shas,
                )
                return outcome

        # ─── Mini-eval (Layer 3 canary) ────────────────────────────
        canary = self._pick_guard_canaries(i, diff_text)
        outcome.canary_tasks = canary
        outcome.preservation_tasks = canary
        mini_tasks = list(dict.fromkeys(self.claimed_tasks + canary))
        self._persist_iteration_outcome(
            outcome, stage="mini_eval", outcome_name="running",
        )
        try:
            with self._heartbeat_during_eval(f"iteration {i} mini-eval"):
                mini = self._run_mini_eval_screened(mini_tasks, iteration=i)
            outcome.mini_eval_job_dir = self._archive_mini_eval_result(mini.job_dir, i)
            outcome.mini_eval_score = mini.score
            outcome.mini_eval_n_trials = mini.n_trials
            outcome.mini_eval_n_errors = mini.n_errors
        except eval_runner.EvalInfrastructureError as e:
            worktree.reset_to(self.worktree.monet_dir, pre_iter_sha)
            outcome.reverted = True
            outcome.reason = f"mini-eval infrastructure failure: {e}"
            outcome.mini_eval_job_dir = self._archive_mini_eval_result(e.job_dir, i)
            self.log.warning("%s; reverted commits %s", outcome.reason, new_shas)
            self._effort_append(
                f"### Iteration {i}\n\n"
                f"{_review_summary_md(outcome, produced_commit=True)}"
                f"**mini-eval infrastructure failure** — score ignored; "
                f"reverted unvalidated commits {new_shas}. Job: {e.job_dir}\n"
            )
            self._persist_iteration_outcome(
                outcome, stage="mini_eval", outcome_name="mini_eval_infra_failed",
                reverted_shas=new_shas,
            )
            return outcome
        except Exception as e:
            self.log.warning("mini-eval failed: %s", e)
            outcome.reason = f"mini-eval failed: {e}"
            self._persist_iteration_outcome(
                outcome, stage="mini_eval", outcome_name="mini_eval_failed",
            )
            return outcome

        outcome.rewards_per_task = dict(mini.task_rewards)
        outcome.pass_rates_per_task = dict(
            getattr(mini, "task_pass_rates", None) or mini.task_rewards
        )

        # Sticky canary: any parent-solved-not-claimed task that mini-eval
        # showed broken in this iteration becomes a canary for every
        # subsequent iteration of this pipeline. No-op for non-resolver
        # pipelines. Done before the Layer-3 canary check so the *next*
        # iteration's `_pick_guard_canaries` sees the new sticky entries
        # even if this iteration is about to be reverted.
        self._update_sticky_canary(i, outcome.rewards_per_task)

        # Layer 3: did the canary regress?
        # A canary counts as REGRESSED only if it actually ran and genuinely
        # failed (present in rewards_per_task with reward < 1.0). A canary that
        # is ABSENT from the rewards map was dropped by transient-infra
        # absorption (a tunnel/gateway blip → 'fetch failed'); treating that as
        # a regression would FALSE-REVERT a real improvement. So we only fault
        # present-and-failed canaries, never infra-blipped (absent) ones.
        # Fractional regression detection (when the fractional gate is on): a
        # guard regressed only if its pass-RATE dropped materially below the
        # parent baseline (~1.0, since guards are parent-passing tasks). The hard
        # margin is the FALLBACK noise-floor only — the reasoned GATE judges
        # regression from the fractional rates + sample counts + context. Legacy
        # binary path (gate off) keeps the strict rpt<1.0 behavior.
        if _fractional_gate_on():
            prt = outcome.pass_rates_per_task
            _floor = 1.0 - _regression_margin()
            outcome.failed_canary_tasks = [
                c for c in canary if c in prt and prt[c] < _floor
            ]
        else:
            rpt = outcome.rewards_per_task
            outcome.failed_canary_tasks = [
                c for c in canary if c in rpt and rpt[c] < 1.0
            ]
        canary_passed = not outcome.failed_canary_tasks
        outcome.canary_passed = canary_passed
        outcome.mini_eval_net_gain = self._mini_eval_net_gain(outcome)
        resolver_stats = self._resolver_mini_eval_stats(outcome)

        _gate_rejects = self.cfg.guard_enabled and not self._mini_eval_accepts(outcome)
        if _gate_rejects and not self._qd_accept_as_specialist(outcome):
            worktree.reset_to(self.worktree.monet_dir, pre_iter_sha)
            outcome.reverted = True
            if self.cfg.pipeline_kind == "regression_resolve":
                outcome.reason = self._resolver_mini_eval_rejection_reason(
                    outcome, resolver_stats,
                )
            else:
                outcome.reason = (
                    f"guard tripped (Layer 3): canary regressed: "
                    f"{outcome.failed_canary_tasks}"
                )
            self.guard_violated_ever = True
            self._effort_append(
                f"### Iteration {i}\n\n"
                f"{_review_summary_md(outcome, produced_commit=True)}"
                f"**Generalization-guard rejected** (Layer 3, canary regression) — "
                f"failed canaries {outcome.failed_canary_tasks} "
                f"(all canaries: {canary}); "
                f"{self._resolver_mini_eval_stats_summary(resolver_stats)}; "
                f"reverted commits {new_shas}.\n"
            )
            self._persist_iteration_outcome(
                outcome, stage="mini_eval", outcome_name="reverted_layer3",
                reverted_shas=new_shas,
            )
            return outcome

        # ─── Bookkeeping for a kept iteration ──────────────────────
        outcome.reason = "kept"
        _progress_line = ""
        if _progress_signal_on() and outcome.pass_rates_per_task:
            _progress_line = (
                f"- claimed-task pass-rates (graded, k-sample — aim to RAISE "
                f"these toward 1.0 next iteration): "
                f"{ {t: round(outcome.pass_rates_per_task.get(t, 0.0), 2) for t in self.claimed_tasks} }\n"
            )
        self._effort_append(
            f"### Iteration {i}\n\n"
            f"{_review_summary_md(outcome, produced_commit=True)}"
            f"- commits: {new_shas}\n"
            f"- mini-eval: {mini.job_dir}\n"
            f"- mini-eval score: {outcome.mini_eval_score}\n"
            f"- claimed-task rewards: "
            f"{ {t: outcome.rewards_per_task.get(t, 0.0) for t in self.claimed_tasks} }\n"
            f"{_progress_line}"
            f"- canary: {canary} (passed: {canary_passed}; "
            f"failed: {outcome.failed_canary_tasks}; "
            f"{self._resolver_mini_eval_stats_summary(resolver_stats)})\n"
        )
        # Persist the freshly-kept commits on this node. `new_shas` came
        # from `worktree.commits_since(monet_dir, pre_iter_sha)`, which lists
        # them newest → oldest; we reverse so commits_json stays chronological
        # (matches `git log <parent>..HEAD --reverse`). `append_commits`
        # also bumps `commit_sha` to the new tail in the same transaction.
        tree.append_commits(self.conn, self.child_node.id, list(reversed(new_shas)))
        # Refresh the in-memory child_node so subsequent reads of
        # `self.child_node.commits` (notably `_apply_picked_commit` after
        # the loop ends) see the freshly-appended SHAs. Without this,
        # the cached `commits=[]` from `_create_child_node` made the
        # picker's defensive `else: new_commits = current` branch fire
        # in `_apply_picked_commit`, which then wrote `commits_json=[]`
        # back to the DB and caused `_finalize` to log "no kept
        # iterations" and skip the final eval / PR / learnings even
        # when the agent had successfully resolved every claimed task.
        self.child_node = tree.get_node(self.conn, self.child_node.id)
        self._persist_iteration_outcome(
            outcome, stage="mini_eval", outcome_name="kept",
        )
        return outcome

    def _pick_guard_canaries(self, iteration: int, diff_text: str | None = None) -> list[str]:
        if not self.cfg.guard_enabled or self.cfg.guard_canary_count <= 0:
            return []
        if self.cfg.pipeline_kind == "regression_resolve" and self.parent_node:
            preserved = self._regression_preservation_tasks(iteration)
            sticky = list(self._sticky_canary.keys())
            if preserved or sticky:
                # Sticky tasks are always included so a once-broken
                # parent-solved task stays guarded for the rest of the loop.
                # Order: sticky first (most-evidence preservation risks),
                # then the freshly picked preservation set.
                return list(dict.fromkeys(sticky + preserved))
        # Evolve path: honor MONET_EVAL_GUARD_COUNT to WIDEN the regression panel
        # beyond the legacy 3 guards (~3.5% coverage of the ~84 non-claimed
        # tasks, far too thin to catch broad/partial regressions). The picker is
        # already domain-stratified for k>1.
        eff_k = _guard_count(default=self.cfg.guard_canary_count)
        # Hybrid guard selection: target the claimed cluster's siblings (most
        # collateral-likely) + domain-spread; broaden to pure spread when the
        # edit touches a global/behavioral monet file (from the iteration diff).
        changed_files = (
            generalization._changed_files_from_diff(diff_text) if diff_text else None
        )
        return generalization.pick_canary_tasks(
            passing_tasks=self.passing_for_canary,
            claimed_tasks=self.claimed_tasks,
            k=eff_k,
            rng_seed=f"{self.pipeline_id}:{iteration}",
            changed_files=changed_files,
        )

    def _resolver_canary_target_count(self, parent_solved_count: int) -> int:
        """How many preservation canaries should one resolver iteration check?

        Resolution order:
          1. `--resolver-preservation-canary-count` (>0) — explicit operator
             override.
          2. `--generalization-canary` (>1) — operator bumped the global
             canary count above the default of 1; reuse it.
          3. Auto: `min(RESOLVER_AUTO_CANARY_FULL_SUITE, ceil(parent_solved/3))`
             — much larger than the evolve default of 1 because the resolver's
             explicit goal is "no new regressions vs parent".
        """
        explicit = self.cfg.resolver_preservation_canary_count
        if explicit and explicit > 0:
            return explicit
        if self.cfg.guard_canary_count and self.cfg.guard_canary_count > 1:
            return self.cfg.guard_canary_count
        if parent_solved_count <= 0:
            return 0
        return min(
            RESOLVER_AUTO_CANARY_FULL_SUITE,
            max(1, (parent_solved_count + 2) // 3),
        )

    def _regression_preservation_tasks(self, iteration: int) -> list[str]:
        assert self.parent_node
        claimed = set(self.claimed_tasks)
        parent_solved_tasks = self._node_solved_tasks(self.parent_node)
        target_solved = [t for t in parent_solved_tasks if t not in claimed]
        if not target_solved:
            return []
        # Small subsets: covering every parent-solved task is cheap and
        # gives the strongest possible preservation guarantee.
        if 0 < len(self.cfg.subset_tasks) <= 10:
            return target_solved
        stats = tree.task_outcome_stats(
            self.conn,
            campaign=self.cfg.campaign,
            subset=self.cfg.subset_label,
        )
        fragile = tree.fragile_tasks_from_stats(stats)
        required = list(dict.fromkeys(
            [t for t in self._node_improved_tasks(self.parent_node) if t not in claimed] +
            [t for t in target_solved if t in fragile]
        ))
        target_k = max(
            self._resolver_canary_target_count(len(target_solved)),
            len(required),
        )
        if len(required) >= target_k:
            return required[:target_k] if target_k > 0 else required
        sampled = generalization.pick_canary_tasks(
            passing_tasks=target_solved,
            claimed_tasks=self.claimed_tasks + required,
            k=target_k - len(required),
            rng_seed=f"{self.pipeline_id}:regression:{iteration}",
        )
        return list(dict.fromkeys(required + sampled))

    def _update_sticky_canary(
        self, iteration: int, rewards: dict[str, float],
    ) -> None:
        """Add any newly-broken parent-solved tasks to the sticky canary.

        Called after each iteration's mini-eval. A task qualifies for sticky
        promotion when ALL of:
          - this is a `regression_resolve` pipeline
          - the task is in `parent.solved_tasks` (i.e. parent succeeded on it)
          - the task is NOT in `claimed_tasks` (we're not trying to fix it
            in this pipeline; if reward < 1.0 here that's a regression vs
            parent, not "still failing same as before")
          - this iteration's mini-eval reward is < 1.0

        Capped at `STICKY_CANARY_MAX` with FIFO eviction by `iteration first
        observed broken` so a single noisy task can't blow up cost.
        """
        if self.cfg.pipeline_kind != "regression_resolve" or not self.parent_node:
            return
        parent_solved = set(self._node_solved_tasks(self.parent_node))
        claimed = set(self.claimed_tasks)
        for task, reward in rewards.items():
            if reward >= 1.0:
                continue
            if task not in parent_solved or task in claimed:
                continue
            if task in self._sticky_canary:
                continue
            self._sticky_canary[task] = iteration
        # Evict oldest until we're at or below the cap. `dict` preserves
        # insertion order in Python 3.7+, so iterating gives us FIFO.
        while len(self._sticky_canary) > STICKY_CANARY_MAX:
            oldest = next(iter(self._sticky_canary))
            del self._sticky_canary[oldest]

    # ─── Phase 4: pick best commit ────────────────────────────────────

    def _phase4_pick_best_commit(self) -> None:
        """Ask cursor-agent to review the kept iterations and pick the best
        commit to keep before running the final eval.

        Behaviour:
          - 0 kept iterations: skip (the loop never produced a candidate;
            `_finalize` will mark the node `no_change`).
          - 1 kept iteration:  short-circuit to that commit. No agent call.
          - 2+ kept iterations: render `prompts/select.md` with each
            iteration's tip + claimed-task rewards + canary results,
            run cursor-agent in plan mode (read-only — we do the
            `git reset --hard` ourselves), parse the
            `<<<SELECTED_COMMIT>>>` block, and reset the worktree.

        If the agent picks the **parent's** commit, that's a deliberate
        "abandon this branch" signal — we keep the worktree at parent and
        `_finalize` will mark `no_change`.

        Errors (agent crash, unparseable output, invalid SHA) fall back to
        the **most recent** kept iteration's tip — the same behaviour as
        before this phase existed, so a flaky picker never makes the
        pipeline worse than the no-picker baseline.
        """
        assert self.parent_node and self.child_node and self.worktree

        kept = [o for o in self.iterations if o.committed_shas and not o.reverted]
        if not kept:
            self.log.info("phase4: no kept iterations — skipping picker")
            return

        # The TIP of each kept iteration is its newest commit. `committed_shas`
        # is the output of `worktree.commits_since(pre_iter_sha)` which is
        # newest-first → `committed_shas[0]` is the iteration's tip after
        # all its commits were applied.
        candidates = [
            {
                "iteration": out.iteration,
                "short_sha": worktree.short_sha(out.committed_shas[0]),
                "full_sha": out.committed_shas[0],
                "commits": list(reversed(out.committed_shas)),
                "claimed_rewards": {
                    t: out.rewards_per_task.get(t, 0.0) for t in self.claimed_tasks
                },
                "canary_tasks": out.canary_tasks,
                "failed_canary_tasks": out.failed_canary_tasks,
                "canary_passed": out.canary_passed,
                "canary_failures": len(out.failed_canary_tasks),
                "canary_total": len(out.canary_tasks),
                "canary_failure_rate": (
                    len(out.failed_canary_tasks) / len(out.canary_tasks)
                    if out.canary_tasks else 0.0
                ),
                "claimed_wins": sum(
                    1 for t in self.claimed_tasks
                    if out.rewards_per_task.get(t, 0.0) >= 1.0
                ),
                "claimed_total": len(self.claimed_tasks),
                "mini_eval_net_gain": self._mini_eval_net_gain(out),
                "mini_eval_job_dir": (
                    str(out.mini_eval_job_dir) if out.mini_eval_job_dir else None
                ),
                # Graded progress signal (ATELIER_PROGRESS_SIGNAL=1): the raw
                # k-sample pass-rate per claimed task (0..1) before majority-vote
                # binarization. Lets the picker keep a candidate that made real
                # progress (e.g. 0/3 -> 1/3) on a not-yet-flipped claimed task,
                # instead of abandoning every no-clean-win iteration. Empty dict
                # when the signal is off, so select.md's block stays hidden.
                "claimed_pass_rates": (
                    {t: round(out.pass_rates_per_task.get(t, 0.0), 3)
                     for t in self.claimed_tasks}
                    if _progress_signal_on() else {}
                ),
                "claimed_progress": (
                    round(sum(out.pass_rates_per_task.get(t, 0.0)
                              for t in self.claimed_tasks), 3)
                    if _progress_signal_on() else 0.0
                ),
            }
            for out in kept
        ]

        picked: str | None
        if len(candidates) == 1:
            picked = candidates[0]["full_sha"]
            self.log.info(
                "phase4: only one kept iteration; auto-picking %s",
                worktree.short_sha(picked),
            )
        else:
            picked = self._invoke_picker_agent(candidates)
            if picked is None:
                fallback = candidates[-1]["full_sha"]
                self.log.warning(
                    "phase4: agent didn't produce a valid pick; falling back "
                    "to the most recent kept iteration (%s)",
                    worktree.short_sha(fallback),
                )
                picked = fallback

        self._apply_picked_commit(picked)

    def _invoke_picker_agent(self, candidates: list[dict[str, Any]]) -> str | None:
        """Render the select.md prompt, run cursor-agent (plan mode),
        and return the picked full SHA (or None on any failure path).
        """
        assert self.parent_node and self.worktree
        parent_full = self.parent_node.commit_sha or ""
        prompt = cursor_agent.render_prompt(
            self._template_path("select.md"),
            {
                "wt_dir": str(self.worktree.eval_dir),
                "candidates": candidates,
                "kept_iterations": len(candidates),
                "claimed_tasks": self.claimed_tasks,
                "parent_short_sha": worktree.short_sha(parent_full or "?"),
                "parent_full_sha": parent_full,
                "subset_label": self.cfg.subset_label,
                "max_iters": self.cfg.max_loop_iters,
            },
        )
        self._save_prompt(0, "select", prompt)

        result = meta_agent.run(
            prompt,
            workspace=self.worktree.eval_dir,
            log_path=self.cursor_log_dir / "select.log",
            model=self.cfg.meta_model,
            plan_mode=True,                    # read-only; orchestrator does the reset
            timeout_s=self.cfg.picker_timeout_s,
            reasoning_effort=self.cfg.meta_effort,
        )
        if result.error:
            self.log.warning("phase4: cursor-agent failed: %s", result.error)
            return None

        raw_pick = _parse_picked_commit(result.text or "")
        if not raw_pick:
            self.log.warning(
                "phase4: agent reply lacked a <<<SELECTED_COMMIT>>> block",
            )
            return None

        resolved = _resolve_pick_against_candidates(
            raw_pick, candidates, parent_full,
        )
        if resolved is None:
            self.log.warning(
                "phase4: agent's pick %r is not one of the valid candidates "
                "(parent or kept iteration tip)", raw_pick,
            )
            return None
        self.log.info(
            "phase4: agent picked %s%s",
            worktree.short_sha(resolved),
            " (parent — abandon branch)" if resolved == parent_full else "",
        )
        return resolved

    def _apply_picked_commit(self, picked_sha: str) -> None:
        """Reset the worktree to `picked_sha` and reconcile node columns.

        Trims `commits_json` to the prefix up-to-and-including the pick
        (so the DB stops claiming commits that no longer exist on the
        branch), updates `commit_sha`, and records `picked_commit_sha`.
        """
        assert self.worktree and self.child_node and self.parent_node
        worktree.reset_to(self.worktree.monet_dir, picked_sha)

        # Defense-in-depth: re-read from the DB before computing the
        # trimmed commits list. `_run_iteration` already refreshes
        # `self.child_node` after each `tree.append_commits`, but the
        # cost is one cheap SELECT and the alternative — silently
        # writing `commits_json=[]` back to a node that DID stick
        # iterations — is the exact regression that left pipeline
        # 271cc003 with `score=None` and no PR after both targets
        # passed. Always read the canonical state right before we
        # overwrite it.
        self.child_node = tree.get_node(self.conn, self.child_node.id)
        current = self.child_node.commits
        if picked_sha == self.parent_node.commit_sha:
            new_commits = []
        elif picked_sha in current:
            new_commits = current[: current.index(picked_sha) + 1]
        else:
            # Picked SHA isn't in our recorded list — shouldn't happen
            # because phase 4 validated against candidates whose SHAs
            # came from `IterationOutcome.committed_shas`, which are
            # the same SHAs `tree.append_commits` persisted. But if it
            # somehow does, preserve the chain (the worktree is now at
            # `picked_sha` so leaving `commits` intact at least keeps
            # `_finalize` from short-circuiting to no_change), and log
            # loudly so the regression is debuggable.
            self.log.warning(
                "phase4: picked %s not in tracked commits %s — keeping "
                "the existing commits_json so _finalize still runs the "
                "final eval; this should never happen",
                worktree.short_sha(picked_sha),
                [worktree.short_sha(c) for c in current],
            )
            new_commits = current

        tree.update_node(
            self.conn, self.child_node.id,
            commit_sha=picked_sha,
            commits_json=new_commits,
            picked_commit_sha=picked_sha,
        )
        self.child_node = tree.get_node(self.conn, self.child_node.id)

    # ─── Phase 3 helpers ─────────────────────────────────────────────

    def _iteration_input_logs(self, i: int) -> Path:
        """Where the analyze prompt should look for failing-trial logs."""
        if i == 1:
            assert self.parent_node
            job_path = self._node_eval_job_path(self.parent_node)
            assert job_path
            return Path(job_path)
        # Use last iteration's mini-eval dir.
        for prev in reversed(self.iterations):
            if prev.mini_eval_job_dir:
                return prev.mini_eval_job_dir
        # Fallback to baseline.
        assert self.parent_node
        job_path = self._node_eval_job_path(self.parent_node)
        assert job_path
        return Path(job_path)

    def _read_test_outputs(self, job_dir: Path) -> list[str]:
        """For each claimed task, return the contents of its trial's
        verifier/test-stdout.txt (used as a denylist by the static scanner)."""
        outputs: list[str] = []
        for task in self.claimed_tasks:
            for trial_dir in job_dir.glob(f"{task}__*"):
                f = trial_dir / "verifier" / "test-stdout.txt"
                try:
                    outputs.append(f.read_text())
                except OSError:
                    pass
        return outputs

    def _trials_for(
        self,
        task: str,
        job_dir: Path,
        result: eval_runner.EvalResult | None = None,
    ) -> list[dict[str, Any]]:
        trial_names = {
            trial_dir.name for trial_dir in job_dir.glob(f"{task}__*")
            if trial_dir.is_dir()
        }
        if result is not None:
            trial_names.update(
                trial_name
                for trial_name in result.rewards_per_task
                if eval_runner.task_base(trial_name) == task
            )
        out = []
        for trial_name in sorted(trial_names):
            trial_dir = job_dir / trial_name
            reward = None
            if result is not None and trial_name in result.rewards_per_task:
                reward = float(result.rewards_per_task[trial_name])
            else:
                try:
                    rt = (trial_dir / "verifier" / "reward.txt").read_text().strip()
                    reward = float(rt)
                except (OSError, ValueError):
                    reward = 0.0
            out.append({
                "task": task,
                "name": trial_name,
                "dir": str(trial_dir),
                "reward": reward,
                "transcript": str(trial_dir / "agent" / "transcript.md"),
                "trajectory": str(trial_dir / "agent" / "trajectory.json"),
                "verifier_stdout": str(trial_dir / "verifier" / "test-stdout.txt"),
                "reward_file": str(trial_dir / "verifier" / "reward.txt"),
                "trial_log": str(trial_dir / "trial.log"),
            })
        return out

    def _task_trial_groups_for(self, tasks: list[str], job_dir: Path) -> list[dict[str, Any]]:
        try:
            result = eval_runner.parse_existing_result_dir(job_dir)
        except (OSError, ValueError, json.JSONDecodeError):
            result = None
        groups: list[dict[str, Any]] = []
        for task in tasks:
            trials = self._trials_for(task, job_dir, result)
            if result is not None:
                status = result.task_outcomes.get(task, "unknown")
            else:
                rewards = [float(t["reward"]) for t in trials]
                has_pass = any(r >= 1.0 for r in rewards)
                has_fail = any(r < 1.0 for r in rewards)
                if has_pass and has_fail:
                    status = "partially_solved"
                elif has_pass:
                    status = "solved"
                elif has_fail:
                    status = "unsolved"
                else:
                    status = "unknown"
            groups.append({
                "task": task,
                "status": status,
                "trial_count": len(trials),
                "passed_count": sum(1 for t in trials if float(t["reward"]) >= 1.0),
                "failed_count": sum(1 for t in trials if float(t["reward"]) < 1.0),
                "job_dir": str(job_dir),
                "trials": trials,
            })
        return groups

    def _render_analyze_prompt(self, i: int, input_logs: Path) -> str:
        assert self.worktree and self.parent_node
        task_trial_groups = self._task_trial_groups_for(self.claimed_tasks, input_logs)
        trials = [
            trial
            for group in task_trial_groups
            for trial in group["trials"]
        ]
        if not trials:
            # Fallback: still ask the agent to look at the job dir overall.
            trials = [{
                "task": task,
                "name": task,
                "dir": str(input_logs),
                "reward": 0.0,
                "transcript": "(see job dir)",
                "trajectory": "(see job dir)",
                "verifier_stdout": "(see job dir)",
                "reward_file": "(see job dir)",
                "trial_log": "(see job dir)",
            } for task in self.claimed_tasks]
            task_trial_groups = [{
                "task": task,
                "status": "unknown",
                "trial_count": 0,
                "passed_count": 0,
                "failed_count": 0,
                "job_dir": str(input_logs),
                "trials": [trial],
            } for task, trial in zip(self.claimed_tasks, trials)]

        template = (
            "regression_analyze.md"
            if self.cfg.pipeline_kind == "regression_resolve"
            else "analyze.md"
        )
        _analyze_prompt = cursor_agent.render_prompt(
            self._template_path(template),
            self._prompt_context(
                i,
                parent_commit=self.parent_node.commit_sha,
                trials=trials,
                task_trial_groups=task_trial_groups,
            ),
        )
        # Dual supervision INTO THE PROPOSER PROMPT (teacher SFT + self-contrast RL).
        # These were previously only reaching the GATE via analyze_digest_block; the
        # proposer never saw them. Append here so the proposer can actually use them.
        return _analyze_prompt + self._dual_supervision_block()

    def _dual_supervision_block(self) -> str:
        """Teacher (SFT) + self-contrast (RL) reference blocks for the claimed tasks,
        appended to the analyze prompt the PROPOSER reads. Env-gated: each helper is a
        strict no-op (returns "") unless its flag is set and a cached source exists."""
        out: list[str] = []
        try:
            from .teacher_supervision import maybe_teacher_block
            from .self_contrast import maybe_self_contrast_block
            for _t in (self.claimed_tasks or []):
                try:
                    out.append(maybe_teacher_block(_t) or "")
                    out.append(maybe_self_contrast_block(_t) or "")
                except Exception as _e:  # noqa: BLE001
                    self.log.debug("dual-supervision skipped for %s: %s", _t, _e)
        except Exception as _e:  # noqa: BLE001
            self.log.debug("dual-supervision import skipped: %s", _e)
        return "".join(b for b in out if b)

    def _render_implement_prompt(self, i: int, plan_path: Path, plan_text: str) -> str:
        assert self.worktree and self.parent_node
        template = (
            "regression_implement.md"
            if self.cfg.pipeline_kind == "regression_resolve"
            else "implement.md"
        )
        ctx = self._prompt_context(i, parent_commit=self.parent_node.commit_sha)
        ctx.update({"plan_path": str(plan_path), "plan_text": plan_text})
        return cursor_agent.render_prompt(self._template_path(template), ctx)

    def _render_review_prompt(self, i: int, pre_iter_sha: str) -> str:
        assert self.worktree
        template = (
            "regression_review.md"
            if self.cfg.pipeline_kind == "regression_resolve"
            else "review.md"
        )
        return cursor_agent.render_prompt(
            self._template_path(template),
            self._prompt_context(i, parent_commit=pre_iter_sha),
        )

    def _prompt_context(
        self,
        iteration: int,
        *,
        parent_commit: str | None,
        trials: list[dict] | None = None,
        task_trial_groups: list[dict] | None = None,
    ) -> dict[str, Any]:
        assert self.worktree and self.parent_node
        parent_eval = self._search_eval(self.parent_node)
        target_score = parent_eval.score if parent_eval else self.parent_node.score
        target_improved = (
            parent_eval.improved_tasks if parent_eval else self.parent_node.improved_tasks
        )
        target_regressed = (
            parent_eval.regressed_tasks if parent_eval else self.parent_node.regressed_tasks
        )
        target_solved = (
            parent_eval.solved_tasks if parent_eval else self.parent_node.solved_tasks
        )
        target_unsolved = (
            parent_eval.failed_tasks if parent_eval else self.parent_node.failed_tasks
        )
        shared_tasks = list(dict.fromkeys(self.claimed_tasks + target_regressed))
        shared_experience = tree.experience_summary_for_tasks(
            self.conn,
            campaign=self.cfg.campaign,
            tasks=shared_tasks,
        )
        return {
            "wt_dir": str(self.worktree.eval_dir),
            "campaign_dir": str(self.campaign_dir),
            "pipeline_id": self.pipeline_id,
            "iteration": iteration,
            "max_iters": self.cfg.max_loop_iters,
            "parent_commit": parent_commit,
            "monet_branch": self.worktree.monet_branch,
            "claimed_tasks": self.claimed_tasks,
            "trials": trials or [],
            "task_trial_groups": task_trial_groups or [],
            "node_ids_to_investigate": [self.parent_node.id],
            "input_eval_dir": parent_eval.job_log_path if parent_eval else self.parent_node.job_log_path,
            "eval_kind": parent_eval.eval_kind if parent_eval else "legacy_node_columns",
            "shared_experience_summary": shared_experience,
            # Phase 1 (trace_analyzer): structured, evidence-cited QC of the
            # claimed-task trajectories so analyze starts from symptoms instead
            # of cold-reading. Empty unless trials carry readable stream-json.
            "trace_qc_summary": self._trace_qc_summary(
                trials,
                parent_eval.job_log_path if parent_eval else self.parent_node.job_log_path,
            ),
            "target_node_id": self.parent_node.id,
            "target_score": target_score,
            "target_improved_tasks": target_improved,
            "target_regressed_tasks": target_regressed,
            "target_solved_tasks": target_solved,
            "target_unsolved_tasks": target_unsolved,
            # Lever 1 (active gate): a generative PRESERVE+EXTEND contract.
            # Empty string unless ATELIER_PRESERVE_EXTEND=1, so the vanilla
            # (control / Arm B) proposer is byte-identical to exp_05's.
            "preserve_extend_block": self._preserve_extend_block(),
            # Lever 1b (contract-guided proposer, SEVerA-inspired): the actual
            # per-task success criteria (verifier/contract) for claimed tasks,
            # so the proposer targets what each task CHECKS instead of guessing.
            # Empty unless ATELIER_CONTRACT_GUIDED=1. Never raises.
            "task_contract_block": self._task_contract_block(),
            # MVP Component 2 (best-of-2 contrastive self-analysis): for claimed
            # tasks with both passing and failing trials, point the proposer at
            # the pass/fail trajectory pair and have it contrast them. Empty
            # unless ATELIER_BESTOF2_CONTRAST=1. Never raises.
            "contrastive_block": self._contrastive_block(task_trial_groups or []),
            # v8 collective knowledge: campaign-wide lessons across ALL nodes
            # (not just this lineage's claimed tasks), so each node evolves with
            # the tree's shared memory. Empty unless ATELIER_COLLECTIVE_KNOWLEDGE=1.
            "collective_knowledge_block": self._collective_knowledge_block(),
            # v9 gate-as-activation: lessons (textual gradients) carried forward
            # from recent reasoned verdicts, incl. rejected attempts. Empty unless
            # ATELIER_REASONED_VERDICT=1. Never raises.
            "recent_lessons_block": self._recent_lessons_block(),
        }

    def _trace_qc_summary(self, trials: list[dict] | None, eval_dir: str | None) -> str:
        """Phase-1 trace_analyzer QC block for the analyze prompt (best-effort).

        Only meaningful for the analyze step (needs ``trials``); other steps get
        an empty string. Never raises — a digest failure must not block a run.

        Ablation switch: ``SELF_EVOLVE_TRACE_QC=0`` disables the injection so the
        analyze prompt is byte-identical to a no-trace-analyzer run (the control
        arm). Defaults on.
        """
        if os.environ.get("SELF_EVOLVE_TRACE_QC", "1").strip().lower() in ("0", "false", "no", "off"):
            return ""
        if not trials:
            return ""
        try:
            from . import trace_qc

            block = trace_qc.analyze_digest_block(trials, eval_dir=eval_dir)
            # Stash so the reasoned verdict can reuse the SAME fault-localization
            # gradient that motivated the change (evidence-threading, v9).
            self._last_trace_qc_digest = block
            return block
        except Exception as exc:  # pragma: no cover - defensive
            self.log.debug("trace_qc summary skipped: %s", exc)
            return ""

    def _collective_knowledge_block(self) -> str:
        """v8 collective-knowledge digest: aggregate the whole campaign's LTM
        (task_experiences across every node) into a what-worked / what-to-avoid
        digest fed to every proposer, so nodes share knowledge instead of each
        evolving in isolation. Empty unless ATELIER_COLLECTIVE_KNOWLEDGE=1.
        Fail-safe: never raises into the loop.
        """
        if os.environ.get("ATELIER_COLLECTIVE_KNOWLEDGE", "0").strip().lower() not in {
            "1", "true", "yes", "on",
        }:
            return ""
        try:
            digest = tree.collective_knowledge_summary(
                self.conn, campaign=self.cfg.campaign,
            )
            theme = _failure_theme_digest()
            combined = "\n".join(p for p in (theme, digest) if p)
            return ("\n" + combined + "\n") if combined else ""
        except Exception as e:  # noqa: BLE001 — advisory context must never break the loop
            self.log.warning("collective knowledge block failed (ignored): %s", e)
            return ""

    def _collective_knowledge_for_gate(self) -> str | None:
        """Campaign-wide collective-knowledge digest for the GATE's decision
        (ATELIER_KNOWLEDGE_GATE) — same aggregation as the proposer block but
        returned raw so it can be threaded into the reasoned verdict's evidence.
        Independent of the proposer-side ATELIER_COLLECTIVE_KNOWLEDGE flag so the
        GATE can aggregate history even if the proposer block is off. Never raises.
        """
        try:
            digest = tree.collective_knowledge_summary(
                self.conn, campaign=self.cfg.campaign,
            )
            theme = _failure_theme_digest()
            combined = "\n".join(p for p in (theme, digest) if p)
            return combined or None
        except Exception as e:  # noqa: BLE001
            self.log.debug("collective knowledge (gate) skipped: %s", e)
            return None

    def _recent_lessons_block(self) -> str:
        """gate-as-activation forward-flow: surface the most recent reasoned-verdict
        lessons (incl. from REJECTED attempts) to the proposer as textual gradients
        — what worked / why prior edits failed, and what to try next. Empty unless
        the reasoned verdict is on. Never raises."""
        try:
            from . import reasoned_verdict as rv
            if not rv.reasoned_verdict_on():
                return ""
            path = self.campaign_dir / "verdict_lessons.jsonl"
            if not path.exists():
                return ""
            import json as _json
            recs = []
            for line in path.read_text().splitlines()[-12:]:
                try:
                    recs.append(_json.loads(line))
                except Exception:
                    continue
            recs = [r for r in recs if r.get("lesson") or r.get("next_directive")][-6:]
            if not recs:
                return ""
            out = ["\n### Lessons from recent attempts (textual gradients — learn from these)"]
            for r in recs:
                tag = f"[{r.get('decision','?')}/{r.get('surface','?')}]"
                if r.get("lesson"):
                    out.append(f"- {tag} {r['lesson']}")
                if r.get("next_directive"):
                    out.append(f"    -> try next: {r['next_directive']}")
            return "\n".join(out) + "\n"
        except Exception as exc:  # noqa: BLE001
            self.log.debug("recent lessons block skipped: %s", exc)
            return ""

    def _contrastive_block(self, task_trial_groups: list[dict]) -> str:
        """MVP Component 2 — best-of-2 contrastive self-bootstrapping.

        For each claimed task the agent solved SOMETIMES (≥1 passing AND ≥1
        failing trial in this eval), surface the concrete passing-vs-failing
        trajectory pair and direct the proposer to diff them: identify the
        single decisive behaviour that made the passing run succeed and encode
        it as a durable rule so the modal rollout reliably reproduces it. This
        converts the agent's own best-of-k variance into a learning signal that
        closes the avg-vs-best gap. Returns "" when the lever is off or no task
        is high-variance. Fail-safe: never raises.
        """
        if not _bestof2_contrast_on():
            return ""
        try:
            pairs: list[tuple[str, dict, dict]] = []
            for g in task_trial_groups:
                if (g.get("passed_count", 0) or 0) >= 1 and (g.get("failed_count", 0) or 0) >= 1:
                    trials = g.get("trials", []) or []
                    passed = next((t for t in trials if float(t.get("reward", 0) or 0) >= 1.0), None)
                    failed = next((t for t in trials if float(t.get("reward", 0) or 0) < 1.0), None)
                    if passed and failed:
                        pairs.append((g.get("task", "?"), passed, failed))
            if not pairs:
                return ""
            lines = [
                "",
                "## Best-of-2 contrastive analysis — close the avg-vs-best gap (high-leverage)",
                "",
                "For the claimed task(s) below, the SAME agent on the SAME task "
                "PASSED on some attempts and FAILED on others — so the capability "
                "exists but is inconsistent. This is the cheapest, highest-yield "
                "win: do NOT try to add new capability, just make the agent "
                "reliably reproduce what its *successful* run already did.",
                "",
                "For each task: open BOTH trajectories, **contrast the passing run "
                "against the failing run**, and identify the single decisive "
                "difference (a step taken/skipped, a verification done/omitted, an "
                "order, a tool used). Then encode that difference as a durable, "
                "GENERAL rule in monet_code so the modal rollout reproduces the "
                "winning behaviour — additively, without disturbing other tasks. "
                "Be skeptical: confirm the difference actually caused the win "
                "(not luck) by checking it's present in the passing trace and "
                "absent in the failing one.",
                "",
            ]
            for task, p, f in pairs:
                lines.append(f"- **{task}**")
                lines.append(f"    - PASSED trajectory: `{p.get('trajectory') or p.get('trial_log') or p.get('dir')}`")
                lines.append(f"    - FAILED trajectory: `{f.get('trajectory') or f.get('trial_log') or f.get('dir')}`")
            lines.append("")
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001 — context block must never break the loop
            self.log.warning("contrastive block failed (ignored): %s", e)
            return ""

    def _task_contract_block(self) -> str:
        """Lever 1b: surface each claimed task's success *contract* (its
        instruction + verifier/test snippet from the TB2 task definition) to
        the proposer. Per SEVerA (arXiv 2603.25111), giving synthesis the
        local output contract prunes the search and steers it toward genuine
        fixes — directly aimed at the high ``no_change`` rate (the proposer
        guessing at hard tasks without knowing the pass criteria).

        Gated by ``ATELIER_CONTRACT_GUIDED`` (default on). Read-only, bounded,
        never raises (degrades to "").
        """
        try:
            if os.environ.get("ATELIER_CONTRACT_GUIDED", "1").strip().lower() in {
                "0", "false", "no", "off"
            }:
                return ""
            dataset = (
                Path(__file__).resolve().parents[1]
                / "benchmarks" / "terminal_bench" / "vendor"
            )
            return _build_task_contract(list(self.claimed_tasks or []), dataset)
        except Exception:  # noqa: BLE001 — proposer context must never crash
            return ""

    def _preserve_extend_block(self) -> str:
        """Lever 1: the dynamic-equivalence contract injected into the
        proposer prompt. Turns the gate from a post-hoc veto into the
        generative objective — the agent is told which solved structures to
        PRESERVE and which unsolved tasks to EXTEND to. Returns ``""`` when
        ATELIER_PRESERVE_EXTEND is off, keeping the control arm identical to
        plain exp_05. Never raises (degrades to "").
        """
        try:
            if os.environ.get("ATELIER_PRESERVE_EXTEND", "0").strip() != "1":
                return ""
            assert self.parent_node
            solved = list(self.parent_node.solved_tasks or [])
            unsolved = list(
                self.parent_node.unsolved_tasks
                or self.parent_node.failed_tasks or []
            )
            if not solved and not unsolved:
                return ""
            solved_md = (
                "\n".join(f"  - `{t}`" for t in solved) or "  - (none yet)"
            )
            unsolved_md = (
                "\n".join(f"  - `{t}`" for t in unsolved) or "  - (none)"
            )
            return (
                "\n## Dynamic-equivalence contract — PRESERVE + EXTEND "
                "(read first)\n\n"
                "This is a MatchFix-style evolution step, not a free-form "
                "rewrite. Your patch is judged on two things:\n\n"
                "1. **PRESERVE** — the harness already SOLVES the tasks below. "
                "Treat their passing behavior as an invariant: do NOT change, "
                "refactor, or regress the code paths they depend on. A patch "
                "that breaks any of these is rejected before it is even "
                "scored.\n"
                f"{solved_md}\n\n"
                "2. **EXTEND** — make the harness solve at least one task it "
                "currently FAILS, *without* disturbing the preserved set. "
                "Prefer additive, narrowly-scoped changes (new branches, "
                "capability checks, fallbacks) over invasive edits to shared "
                "code paths that the preserved tasks rely on.\n"
                f"{unsolved_md}\n\n"
                "If a fix for an unsolved task would require changing a code "
                "path a solved task depends on, guard it so existing behavior "
                "is unchanged on the solved inputs.\n"
            )
        except Exception:  # noqa: BLE001 — never break prompt rendering
            return ""

    def _template_path(self, name: str) -> Path:
        return Path(__file__).resolve().parent / "prompts" / name

    def _save_prompt(self, i: int, kind: str, body: str) -> None:
        (self.prompts_dir_path / f"iter_{i}_{kind}.md").write_text(body)

    def _effort_append(self, md: str) -> None:
        # Guard on the raw string, NOT on `Path(...)`. `Path("")` is
        # `PosixPath('.')` whose bool is True, so `if not Path(raw or "")`
        # would silently fall through and `open('.', 'a')` would raise
        # IsADirectoryError — the failure mode that killed bca4c905.
        assert self.child_node
        raw = self.child_node.effort_md_path
        if not raw:
            self.log.warning("effort_md_path not set on child node — skipping effort append")
            return
        with Path(raw).open("a") as f:
            f.write("\n" + md.strip() + "\n")

    # ─── Root-node summary markdown ───────────────────────────────────

    def _write_root_works_md(self, node: tree.Node, *, pending: bool) -> Path:
        """Render and persist `nodes/<root_id>/works.md` for a root node.

        Two states:
          - `pending=True` (call from `_maybe_bootstrap_root`): the baseline
            eval hasn't run yet, so we write commit/branch/subset and a
            "pending" placeholder.
          - `pending=False` (call from `_phase2_baseline` after scoring):
            we rewrite with the final score, failing-tasks list, and
            job-log path. `failed_tasks` is read from the freshly-refreshed
            `node` argument.

        Also sets `nodes.works_md_path` on the DB row so the visualizer's
        `/api/node/<id>` endpoint picks the file up.
        """
        node_dir = tree.node_dir(self.cfg.reports_root, self.cfg.campaign, node.id)
        node_dir.mkdir(parents=True, exist_ok=True)
        path = node_dir / "works.md"
        path.write_text(_render_root_works_md(node, campaign=self.cfg.campaign, pending=pending))
        if node.works_md_path != str(path):
            tree.update_node(self.conn, node.id, works_md_path=str(path))
        return path

    def _backfill_root_works_md_if_missing(self, root_id: str) -> None:
        """Idempotent backfill for campaigns whose root was bootstrapped
        before `_write_root_works_md` existed. Cheap (sub-millisecond) so
        it's safe to call on every pipeline startup; it only writes if
        the file is actually missing OR the works_md_path column is NULL.
        """
        n = tree.get_node(self.conn, root_id)
        if n is None:
            return
        if n.works_md_path and Path(n.works_md_path).is_file():
            return
        pending = n.score is None
        try:
            self._write_root_works_md(n, pending=pending)
            self.log.info(
                "backfilled works.md for root %s (pending=%s)", root_id, pending,
            )
        except OSError:
            # Don't fail the pipeline over a cosmetic backfill.
            self.log.warning("could not backfill works.md for root %s", root_id, exc_info=True)

    # ─── Dynamic-equivalence (MatchFix) gate ─────────────────────────

    def _run_pre_final_equivalence_gate(self):
        """MatchFix dynamic-equivalence gate, run BEFORE the full eval.

        Returns an equivalence verdict object (decision in
        EQUIVALENT/MODIFIED/INCONCLUSIVE) or ``None`` when the gate is
        disabled / could not run. A confirmed ``MODIFIED`` is the only
        hard-reject; everything else falls through to the full eval, which
        is the ground-truth arbiter. All failures degrade to ``None`` so a
        gate hiccup can never crash the pipeline.
        """
        if not atelier_hook.is_equivalence_gate_enabled():
            return None
        assert self.worktree and self.parent_node and self.child_node
        monet_dir = self.worktree.monet_dir
        try:
            child_commit = worktree.head_sha(monet_dir)
        except Exception as e:  # noqa: BLE001
            self.log.warning("equivalence gate: head_sha failed: %s", e)
            return None
        parent_commit = self.parent_node.commit_sha
        diff_text = ""
        if parent_commit and child_commit:
            try:
                res = subprocess.run(
                    ["git", "-C", str(monet_dir), "diff",
                     f"{parent_commit}..{child_commit}"],
                    check=False, capture_output=True, text=True, timeout=60,
                )
                diff_text = (res.stdout or "")[:20000]
            except (subprocess.TimeoutExpired, OSError) as e:
                self.log.warning("equivalence gate: git diff failed: %s", e)
        # ── Anti-cheat HARD constraint (SEVerA-inspired), checked BEFORE the
        # LLM gate. Pure diff analysis (no LLM/network), so it enforces even
        # when the gateway is down. If the candidate tampers with its own
        # verification surface / hard-codes benchmark answers, reject outright
        # (a confirmed MODIFIED) — don't spend a probe or full eval on a cheat.
        if os.environ.get("ATELIER_ANTI_CHEAT_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}:
            try:
                from atelier.anti_cheat import scan_diff_for_eval_tampering
                cheat_reasons = scan_diff_for_eval_tampering(diff_text)
            except Exception as e:  # noqa: BLE001 — never let the scan break the gate
                cheat_reasons = []
                self.log.warning("anti-cheat scan failed (ignored): %s", e)
            if cheat_reasons:
                from atelier import matchfix_gate as _mfg
                self.log.info(
                    "anti-cheat HARD-REJECT child %s: %s",
                    self.child_node.id, "; ".join(cheat_reasons),
                )
                return _mfg.EquivalenceVerdict(
                    decision="MODIFIED",
                    k_picked_tasks=(),
                    per_task_results={},
                    extension_tasks_solved=(),
                    n_regressions=len(cheat_reasons),
                    semantic_analysis=None,
                    verdict_rationale="anti-cheat: " + "; ".join(cheat_reasons),
                    accept_for_final_eval=False,
                )
        preservation_pool = self._preservation_solved_pool(self.parent_node)
        legacy_pool_n = len([t for t in (self.parent_node.solved_tasks or []) if t])
        if len(preservation_pool) != legacy_pool_n:
            self.log.info(
                "equivalence gate: preservation pool widened %d→%d tasks "
                "(parent canary-shadow ∪ root full-solved) so coverage sizer "
                "can probe beyond the adaptive-subset canaries",
                legacy_pool_n, len(preservation_pool),
            )
        eq_ctx = atelier_hook.EquivalenceGateContext(
            pipeline_id=self.pipeline_id,
            campaign=self.cfg.campaign,
            child_node_id=self.child_node.id,
            parent_node_id=self.parent_node.id,
            parent_commit=parent_commit,
            child_commit=child_commit,
            parent_solved_tasks=tuple(preservation_pool),
            parent_unsolved_tasks=tuple(
                self.parent_node.unsolved_tasks
                or self.parent_node.failed_tasks or ()),
            diff_text=diff_text,
            trial_digests="",
            eval_config_path=self.cfg.config_path,
            eval_cwd=self.worktree.eval_dir,
            reports_root=Path(self.cfg.reports_root),
            parent_score_commit_sha=parent_commit,
        )
        try:
            with self._heartbeat_during_eval("equivalence gate"):
                return atelier_hook.run_equivalence_gate(eq_ctx)
        except Exception as e:  # noqa: BLE001
            self.log.warning("equivalence gate raised (ignored): %s", e)
            return None

    def _should_archive_stepping_stone(self, eq_verdict) -> bool:
        """Hybrid archive routing: a MODIFIED candidate that is NOT a cheat and
        whose regressions are bounded is a STEPPING STONE — route it to the full
        eval (so it's persisted as a distinct scored node with its improved/
        regressed deltas for the resolver/merger/broaden to build on) rather than
        resetting it to parent and discarding it. Anti-cheat and catastrophic
        regressions still hard-reject. No-op unless ATELIER_HYBRID_ARCHIVE=1.
        """
        rationale = (getattr(eq_verdict, "verdict_rationale", "") or "").lower()
        is_cheat = (
            "anti-cheat" in rationale or "cheat" in rationale or "tamper" in rationale
        )
        if is_cheat:
            return False  # cheats are POISON — never archived as a buildable node
        # Archive-all (PROMOTE-vs-ARCHIVE): every non-cheat candidate is preserved
        # as a scored stepping stone — never reset-to-parent-and-dropped — so its
        # diff + deltas + lesson stay reusable. No regression cap.
        if _archive_all_on():
            return True
        if not _hybrid_archive_on():
            return False
        n_reg = getattr(eq_verdict, "n_regressions", 0) or 0
        return n_reg <= _archive_max_regressions()

    def _reject_via_equivalence_gate(self, eq_verdict) -> int:
        """Reset to parent + no_change: gate found a confirmed regression."""
        assert self.worktree and self.parent_node and self.child_node
        self.log.info(
            "equivalence gate REJECTED child %s (decision=%s n_regressions=%d "
            "probes=%s) - resetting to parent, skipping final eval",
            self.child_node.id, eq_verdict.decision,
            eq_verdict.n_regressions, list(eq_verdict.k_picked_tasks),
        )
        if self.parent_node.commit_sha:
            worktree.reset_to(self.worktree.monet_dir, self.parent_node.commit_sha)
        try:
            _ctx = atelier_hook.HookContext(
                pipeline_id=self.pipeline_id, campaign=self.cfg.campaign,
                child_node_id=self.child_node.id,
                parent_node_id=self.parent_node.id,
                parent_commit=self.parent_node.commit_sha,
                child_commit=self.parent_node.commit_sha, base_score=0.0,
                reports_root=Path(self.cfg.reports_root),
            )
            atelier_hook.mark_ltm_source_rejected(
                ctx=_ctx, reason=f"equivalence:{eq_verdict.decision}")
        except Exception:  # noqa: BLE001
            pass
        rationale = (eq_verdict.verdict_rationale or "")
        is_cheat = any(
            s in rationale.lower() for s in ("anti-cheat", "cheat", "tamper")
        )
        # p3_record_rejects: even a non-promoted node teaches. Record the
        # claimed-task patterns as 'poisoned' (verifier-gaming anti-pattern) or
        # 'rejected' (regression/no-gain) so collective knowledge spans these.
        self._record_nonpromote_experiences(
            kind="poisoned" if is_cheat else "rejected",
            summary=rationale[:300] or f"equivalence:{eq_verdict.decision}",
            worker_kind="equivalence_gate",
        )
        rejection = {
            "equivalence_gate_rejected": True,
            "decision": eq_verdict.decision,
            "n_regressions": eq_verdict.n_regressions,
            "probes": list(eq_verdict.k_picked_tasks),
            "rationale": rationale[:500],
            "poisoned": is_cheat,
        }
        tree.update_node(
            self.conn, self.child_node.id,
            commit_sha=self.parent_node.commit_sha,
            commits_json=[],
            picked_commit_sha=self.parent_node.commit_sha,
            score=self.parent_node.score,
            subset=self.parent_node.subset,
            job_log_path=self.parent_node.job_log_path,
            failed_tasks_json=self.parent_node.failed_tasks,
            solved_tasks_json=self.parent_node.solved_tasks,
            unsolved_tasks_json=self.parent_node.unsolved_tasks,
            improved_tasks_json=[],
            regressed_tasks_json=[],
            resolved_tasks_json=_resolved_summary(self.iterations, self.claimed_tasks) + [rejection],
        )
        self.child_node = tree.get_node(self.conn, self.child_node.id)
        self._push_child_branch()
        self._mark_done("no_change")
        return 0

    # ─── Finalize ────────────────────────────────────────────────────

    def _finalize(self) -> int:
        if not self.child_node:
            return 0
        tree.update_pipeline(self.conn, self.pipeline_id, status="eval")
        tree.heartbeat(self.conn, self.pipeline_id)

        # Decide whether to run a final eval. Skip if no commits stuck OR
        # if the phase-4 picker explicitly picked the parent's commit
        # ("abandon this branch"). In both cases the child has no commits
        # of its own and final-eval would just re-score the parent.
        any_kept = bool(self.child_node.commits)
        assert self.parent_node
        picked_parent = (
            self.child_node.picked_commit_sha is not None
            and self.child_node.picked_commit_sha == self.parent_node.commit_sha
        )
        if not any_kept or picked_parent:
            reason = "picker abandoned (picked parent)" if picked_parent else "no kept iterations"
            self.log.info(
                "%s — marking node 'no_change' and skipping final eval", reason,
            )
            tree.copy_latest_node_eval(
                self.conn,
                campaign=self.cfg.campaign,
                source_node_id=self.parent_node.id,
                target_node_id=self.child_node.id,
                eval_kind="subset_final",
                source_pipeline_id=self.pipeline_id,
                metadata={"basis": "copied_parent_no_change", "reason": reason},
            )
            tree.update_node(
                self.conn, self.child_node.id,
                status="no_change",
                score=self.parent_node.score,
                subset=self.parent_node.subset,
                job_log_path=self.parent_node.job_log_path,
                failed_tasks_json=self._node_failed_tasks(self.parent_node),
                solved_tasks_json=self._node_solved_tasks(self.parent_node),
                unsolved_tasks_json=(
                    self._search_eval(self.parent_node).unsolved_tasks
                    if self._search_eval(self.parent_node) else self.parent_node.unsolved_tasks
                ),
                partially_solved_tasks_json=self._node_partially_solved_tasks(self.parent_node),
                improved_tasks_json=[],
                regressed_tasks_json=[],
                resolved_tasks_json=_resolved_summary(self.iterations, self.claimed_tasks),
            )
            self.child_node = tree.get_node(self.conn, self.child_node.id)
            self._push_child_branch()
            self._mark_done("no_change")
            return 0

        assert self.worktree
        adaptive_mode = (
            self.cfg.adaptive_subset_enabled and self.cfg.subset_label == "full"
        )
        final_tasks = self._adaptive_eval_tasks() if adaptive_mode else self.cfg.subset_tasks
        # v9 DEFER-FULL-EVAL (minibatch-SGD): instead of an 89-task full eval the
        # moment a node improves, score it on the cheap claimed+rotating-guard set
        # so it can become a parent and EXTEND. The expensive full avg@k runs only
        # at the end-of-campaign top-N confirm. Parent selection already ranks on
        # the search/mini-eval score, so this keeps the search moving without the
        # per-node cost. Resolver pipelines keep their own (small) target eval.
        deferred_node_eval = (
            _defer_node_full_eval() and self.cfg.pipeline_kind != "regression_resolve"
        )
        if deferred_node_eval:
            guards = self._pick_guard_canaries(len(self.iterations))
            final_tasks = list(dict.fromkeys(list(self.claimed_tasks) + list(guards)))
            self.log.info(
                "v9 defer-full-eval: scoring node on %d claimed+guard task(s) "
                "(full avg@k deferred to end-of-campaign top-N confirm)",
                len(final_tasks),
            )
        # ─── Dynamic-equivalence (MatchFix) gate: preserve the parent's
        # passing structure BEFORE spending a full subset eval. HARD reject
        # ONLY on a CONFIRMED ``MODIFIED`` verdict (>=1 parent-solved probe
        # actually regressed + LLM consensus). ``INCONCLUSIVE`` and every
        # infra-degraded case fall through to the full eval (ground truth),
        # so a flaky probe can never silently discard a real improvement.
        eq_verdict = self._run_pre_final_equivalence_gate()
        if eq_verdict is not None and eq_verdict.decision == "MODIFIED":
            if self._should_archive_stepping_stone(eq_verdict):
                self.log.info(
                    "hybrid-archive: MODIFIED candidate is a non-catastrophic "
                    "stepping stone (n_regressions=%d <= %d, not a cheat) — "
                    "proceeding to full eval to ARCHIVE it as a scored node for "
                    "resolver/merge/broaden instead of resetting to parent",
                    getattr(eq_verdict, "n_regressions", 0) or 0,
                    _archive_max_regressions(),
                )
                # fall through to the full eval (do NOT reject)
            else:
                return self._reject_via_equivalence_gate(eq_verdict)
        self.log.info(
            "running subset final eval on %d task(s) (campaign subset=%s)",
            len(final_tasks) if final_tasks else 0,
            self.cfg.subset_label,
        )
        try:
            with self._heartbeat_during_eval("final eval"):
                if final_tasks:
                    if adaptive_mode or deferred_node_eval:
                        if _confirm_before_parent():
                            # Confirm-before-parent: score the deferred node on
                            # avg@k (k>=3) instead of a single k=1 sample, so the
                            # number that ranks best-node / parent / pool is a
                            # trustworthy fractional measurement — a node can no
                            # longer hijack the search on one lucky sample.
                            final = eval_runner.run_full_avg_k(
                                config_path=self.cfg.config_path,
                                cwd=self.worktree.eval_dir,
                                task_names=final_tasks,
                                k_samples=_confirm_k_samples(),
                                job_name=f"subset_final_{self.pipeline_id}",
                                tee_log_path=getattr(self, "run_log_path", None),
                            )
                        else:
                            final = eval_runner.run_subset(
                                config_path=self.cfg.config_path,
                                cwd=self.worktree.eval_dir,
                                task_names=final_tasks,
                                job_name=f"subset_final_{self.pipeline_id}",
                                tee_log_path=getattr(self, "run_log_path", None),
                            )
                    else:
                        final = self._run_full(
                            config_path=self.cfg.config_path,
                            cwd=self.worktree.eval_dir,
                            subset=self.cfg.subset_label,
                            task_names=final_tasks,
                            job_name=f"subset_final_{self.pipeline_id}",
                            tee_log_path=getattr(self, "run_log_path", None),
                        )
                else:
                    final = self._run_full(
                        config_path=self.cfg.config_path,
                        cwd=self.worktree.eval_dir,
                        subset=self.cfg.subset_label,
                        task_names=self.cfg.subset_tasks,
                        job_name=f"subset_final_{self.pipeline_id}",
                        tee_log_path=getattr(self, "run_log_path", None),
                    )
        except eval_runner.EvalInfrastructureError as e:
            self.log.error("final eval infrastructure failure: %s", e)
            archived = self._archive_final_eval_job_if_small(e.job_dir)
            tree.update_node(
                self.conn,
                self.child_node.id,
                status="failed",
                job_log_path=str(archived),
                resolved_tasks_json=[{
                    "eval_infrastructure_failure": True,
                    "reason": str(e),
                    "job_log_path": str(archived),
                    "failures": e.failures[:20],
                }],
            )
            self._mark_done("failed")
            return 1
        except Exception as e:
            self.log.exception("final eval failed: %s", e)
            tree.update_node(
                self.conn, self.child_node.id, status="failed",
            )
            self._mark_done("failed")
            return 1

        archived_job_dir = self._archive_final_eval_job_if_small(final.job_dir)
        final.job_dir = archived_job_dir
        # ── Surface genuine claimed-task wins into the finalized eval ──────
        # The deferred / subset_final panel does NOT re-run this node's claimed
        # tasks, so a real claimed-task win (recorded per-iteration in
        # iteration_outcomes.claimed_rewards) would otherwise be invisible to the
        # merge selector (reads node_eval.solved_tasks) AND to parent selection
        # (reads node.solved_tasks) -- collapsing every node's solved set to the
        # shared guard panel and starving recombination of complementarity.
        # Additive + guarded: only ADDS proven wins; a failure degrades to no-op.
        try:
            import json as _json
            _cw = {}
            for (_crj,) in self.conn.execute(
                "SELECT claimed_rewards_json FROM iteration_outcomes WHERE pipeline_id = ?",
                (self.pipeline_id,),
            ):
                for _t, _r in (_json.loads(_crj or "{}") or {}).items():
                    if _r is not None and float(_r) >= 1.0:
                        _cw[_t] = max(_cw.get(_t, 0.0), float(_r))
            if _cw:
                final.solved_tasks = sorted(set(final.solved_tasks) | set(_cw))
                final.task_rewards = {**(final.task_rewards or {}), **_cw}
                final.unsolved_tasks = [t for t in (final.unsolved_tasks or []) if t not in _cw]
                self.log.info(
                    "claimed-win propagation: surfaced %d claimed-task win(s) into "
                    "finalized solved set: %s", len(_cw), sorted(_cw),
                )
        except Exception as _e:
            self.log.warning("claimed-win propagation skipped: %s", _e)
        parent_eval = self._search_eval(self.parent_node)
        # Per-task scores restricted to the claimed tasks — the most
        # decision-relevant slice for the next iteration's picker prompt
        # and for the visualizer's Overview pane.
        per_task = {
            t: float(final.task_rewards.get(t, 0.0)) for t in self.claimed_tasks
        }
        resolved_tasks = _resolved_summary_from_rewards(per_task, self.claimed_tasks)
        child_solved = list(final.solved_tasks)
        child_unsolved = list(final.unsolved_tasks)
        child_partial = list(final.partially_solved_tasks)
        child_failed = list(final.failed_task_names)
        parent_solved = parent_eval.solved_tasks if parent_eval else self.parent_node.solved_tasks
        parent_failed = parent_eval.failed_tasks if parent_eval else self.parent_node.failed_tasks
        parent_score = parent_eval.score if parent_eval else self.parent_node.score
        improved_tasks, regressed_tasks = tree.task_deltas(
            parent_solved=parent_solved,
            parent_unsolved=self._parent_unsolved_for_deltas(
                parent_solved=parent_solved,
                parent_unsolved=parent_failed,
            ),
            child_solved=child_solved,
            child_unsolved=child_failed,
        )
        improved = (
            parent_score is not None and
            final.score is not None and
            final.score > parent_score
        )
        node_eval = self._persist_node_eval_result(
            node_id=self.child_node.id,
            eval_kind="subset_final",
            result=final,
            parent_eval=parent_eval,
            metadata={
                "basis": (
                    "deferred_mini_final" if deferred_node_eval
                    else ("adaptive_subset_final" if adaptive_mode else "subset_final")
                ),
                "adaptive_task_count": len(final_tasks) if adaptive_mode else None,
                "deferred_full_eval": deferred_node_eval,
                "deferred_task_count": len(final_tasks) if deferred_node_eval else None,
            },
        )
        # ─── Dynamic-equivalence EXTEND half of the contract ─────────────
        # A child is a real win only if it solves >=1 task the parent did
        # NOT and the net task delta is positive (extend the structure, not
        # just shuffle it). On pass-rate this is usually implied by
        # score>parent, but stating it explicitly documents the MatchFix
        # method and guards score-tie edge cases. Ablatable via
        # ATELIER_REQUIRE_EXTENSION=0 (off => pure exp_05 / DGM behavior).
        require_extension = (
            os.environ.get("ATELIER_REQUIRE_EXTENSION", "0").strip() == "1"
        )
        n_extend = len(improved_tasks)
        n_regress = len(regressed_tasks)
        if require_extension and self.cfg.pipeline_kind != "regression_resolve":
            extends = n_extend >= 1 and n_extend > n_regress
            self.log.info(
                "extension contract: improved=%d regressed=%d -> extends=%s "
                "(score_improved=%s)",
                n_extend, n_regress, extends, improved,
            )
            improved = improved and extends
        final_accepts, final_net_gain = self._resolver_final_accepts(
            score_improved=improved,
            improved_tasks=improved_tasks,
            regressed_tasks=regressed_tasks,
        )
        reject_regression_resolver = (
            self.cfg.pipeline_kind == "regression_resolve"
            and not final_accepts
        )
        if reject_regression_resolver:
            reason = (
                "final score did not improve target"
                if not improved else
                "final eval net task gain "
                f"{final_net_gain} below {self.cfg.resolver_final_min_net_gain} "
                f"(improved={improved_tasks}, regressed={regressed_tasks})"
            )
            self.log.info(
                "regression resolver rejected: %s; restoring parent-equivalent node metadata",
                reason,
            )
            if self.parent_node.commit_sha:
                worktree.reset_to(self.worktree.monet_dir, self.parent_node.commit_sha)
            tree.copy_latest_node_eval(
                self.conn,
                campaign=self.cfg.campaign,
                source_node_id=self.parent_node.id,
                target_node_id=self.child_node.id,
                eval_kind="subset_final",
                source_pipeline_id=self.pipeline_id,
                metadata={
                    "basis": "copied_parent_after_resolver_rejection",
                    "observed_eval_id": node_eval.id,
                    "reason": reason,
                },
            )
            rejection_summary = {
                "regression_resolver_rejected": True,
                "reason": reason,
                "observed_score": final.score,
                "observed_job_log_path": str(archived_job_dir),
                "observed_regressed_tasks": regressed_tasks,
                "observed_improved_tasks": improved_tasks,
                "observed_net_gain": final_net_gain,
            }
            tree.update_node(
                self.conn, self.child_node.id,
                commit_sha=self.parent_node.commit_sha,
                commits_json=[],
                picked_commit_sha=self.parent_node.commit_sha,
                score=self.parent_node.score,
                subset=self.parent_node.subset,
                job_log_path=self.parent_node.job_log_path,
                failed_tasks_json=self._node_failed_tasks(self.parent_node),
                solved_tasks_json=self._node_solved_tasks(self.parent_node),
                unsolved_tasks_json=(
                    self._search_eval(self.parent_node).unsolved_tasks
                    if self._search_eval(self.parent_node) else self.parent_node.unsolved_tasks
                ),
                partially_solved_tasks_json=self._node_partially_solved_tasks(self.parent_node),
                improved_tasks_json=[],
                regressed_tasks_json=[],
                resolved_tasks_json=resolved_tasks + [rejection_summary],
                claimed_task_scores_json=per_task,
                status="no_change",
            )
            self.child_node = tree.get_node(self.conn, self.child_node.id)
            self._push_child_branch()
            self._mark_done("no_change")
            return 0

        # ─── Phantom guard ────────────────────────────────────────────────
        # A node that solves EXACTLY the same tasks as its parent (no improved
        # AND no regressed tasks) is a no-op, regardless of its adaptive-subset
        # score. Persisting it as `completed` with a possibly-inflated subset
        # score (e.g. 1.0 on a tiny claimed subset) lets a task-equivalent
        # phantom outrank the real best node, whose `failed_tasks` then collapse
        # the failing-task pool and trigger a spurious "no work" early-stop (seen
        # in v7 once the hybrid archive started routing such candidates to the
        # full eval instead of resetting them to parent). Record it as no_change
        # with the parent's score/failed-tasks so it can never become "best" and
        # so only REAL stepping stones (improved>0 or regressed>0) are archived.
        if (
            self.cfg.pipeline_kind != "regression_resolve"
            and not improved_tasks
            and not regressed_tasks
        ):
            if _archive_all_on():
                # DGM preserve-everything: a no-measured-effect variant is NOT
                # reset to baseline — it's kept as an ARCHIVED node with its OWN
                # commit + real score, so parent-selection can still sample it
                # (diversity / future merges) while best-node excludes it (so its
                # possibly-inflated subset score can't hijack the tip/pool).
                latest = worktree.head_sha(self.worktree.monet_dir)
                self.log.info(
                    "phantom -> ARCHIVE (preserve-everything): child %s no measured "
                    "effect (improved=0 regressed=0) — kept as archived variant "
                    "(commit %s, score %.4f), NOT reset to parent",
                    self.child_node.id, (latest or "")[:12],
                    final.score if final.score is not None else -1.0,
                )
                tree.update_node(
                    self.conn, self.child_node.id,
                    commit_sha=latest,
                    score=final.score,
                    subset=self.cfg.subset_label,
                    job_log_path=str(archived_job_dir),
                    failed_tasks_json=child_failed,
                    solved_tasks_json=child_solved,
                    unsolved_tasks_json=child_unsolved,
                    partially_solved_tasks_json=child_partial,
                    improved_tasks_json=[],
                    regressed_tasks_json=[],
                    resolved_tasks_json=resolved_tasks,
                    claimed_task_scores_json=per_task,
                    status="archived",
                )
                self.child_node = tree.get_node(self.conn, self.child_node.id)
                self._push_child_branch()
                self._mark_done("no_change")
                return 0
            self.log.info(
                "phantom guard: child %s is task-equivalent to parent "
                "(improved=0 regressed=0, subset score=%.4f) — recording as "
                "no_change with parent score so it can't distort best-node/pool",
                self.child_node.id, final.score if final.score is not None else -1.0,
            )
            if self.parent_node.commit_sha:
                worktree.reset_to(self.worktree.monet_dir, self.parent_node.commit_sha)
            tree.update_node(
                self.conn, self.child_node.id,
                commit_sha=self.parent_node.commit_sha,
                commits_json=[],
                picked_commit_sha=self.parent_node.commit_sha,
                score=self.parent_node.score,
                subset=self.parent_node.subset,
                job_log_path=self.parent_node.job_log_path,
                failed_tasks_json=self._node_failed_tasks(self.parent_node),
                solved_tasks_json=self._node_solved_tasks(self.parent_node),
                unsolved_tasks_json=(
                    self._search_eval(self.parent_node).unsolved_tasks
                    if self._search_eval(self.parent_node) else self.parent_node.unsolved_tasks
                ),
                partially_solved_tasks_json=self._node_partially_solved_tasks(self.parent_node),
                improved_tasks_json=[],
                regressed_tasks_json=[],
                resolved_tasks_json=resolved_tasks,
                claimed_task_scores_json=per_task,
                status="no_change",
            )
            self.child_node = tree.get_node(self.conn, self.child_node.id)
            self._push_child_branch()
            self._mark_done("no_change")
            return 0

        # ─── QD lossy-specialist archive ─────────────────────────────────
        # A variant that adds a NEW win but is net<=0 (regresses as much or more)
        # is not a valid tip, but IS valuable merge fuel: it proves a residual
        # task is solvable and carries the harness change that solved it. Under
        # ATELIER_QD_ARCHIVE keep it as `archived` with its REAL solved/improved/
        # regressed sets (archived is excluded from best/tip but merge- and
        # parent-eligible). Net-positive nodes fall through to `completed` as usual.
        if (
            _qd_archive_on()
            and self.cfg.pipeline_kind != "regression_resolve"
            and improved_tasks
            and regressed_tasks
            and len(improved_tasks) <= len(regressed_tasks)
        ):
            latest = worktree.head_sha(self.worktree.monet_dir)
            self.log.info(
                "QD specialist -> ARCHIVE: child %s added new win(s) %s but "
                "regressed %s (net=%d) — kept as archived with REAL deltas for the "
                "merge archive; excluded from best/tip",
                self.child_node.id, improved_tasks, regressed_tasks,
                len(improved_tasks) - len(regressed_tasks),
            )
            tree.update_node(
                self.conn, self.child_node.id,
                commit_sha=latest,
                score=final.score,
                subset=self.cfg.subset_label,
                job_log_path=str(archived_job_dir),
                failed_tasks_json=child_failed,
                solved_tasks_json=child_solved,
                unsolved_tasks_json=child_unsolved,
                partially_solved_tasks_json=child_partial,
                improved_tasks_json=improved_tasks,
                regressed_tasks_json=regressed_tasks,
                resolved_tasks_json=resolved_tasks,
                claimed_task_scores_json=per_task,
                status="archived",
            )
            self.child_node = tree.get_node(self.conn, self.child_node.id)
            self._push_child_branch()
            self._mark_done("no_change")
            return 0

        latest = worktree.head_sha(self.worktree.monet_dir)
        tree.update_node(
            self.conn, self.child_node.id,
            commit_sha=latest,
            score=final.score,
            subset=self.cfg.subset_label,
            job_log_path=str(archived_job_dir),
            failed_tasks_json=child_failed,
            solved_tasks_json=child_solved,
            unsolved_tasks_json=child_unsolved,
            partially_solved_tasks_json=child_partial,
            improved_tasks_json=improved_tasks,
            regressed_tasks_json=regressed_tasks,
            resolved_tasks_json=resolved_tasks,
            claimed_task_scores_json=per_task,
            status="completed",
        )
        self.child_node = tree.get_node(self.conn, self.child_node.id)
        assert self.parent_node and self.child_node
        self._record_experiences_from_eval(
            node=self.child_node,
            node_eval=node_eval,
            parent_eval=parent_eval,
            worker_kind=self.cfg.pipeline_kind,
        )
        # `guard_violated_ever` used to be ANDed in here, but reverted
        # iterations leave no commits behind, and the Phase-4 picker only
        # selects from `kept = [o for o in iterations if o.committed_shas
        # and not o.reverted]`. So the picked commit is by construction
        # from an iteration that passed both guards — the global flag was
        # purely cosmetic and was the direct cause of `score=0.6
        # → status=no_change` when iter 1 tripped a guard but iter 2
        # cleanly resolved the claimed task.
        self.log.info(
            "final score: %.4f (parent: %.4f, delta: %+.4f) improved=%s net_task_gain=%d",
            node_eval.score,
            parent_score or 0.0,
            node_eval.score - (parent_score or 0.0),
            improved,
            final_net_gain,
        )

        if not improved:
            tree.update_node(self.conn, self.child_node.id, status="no_change")
            self.child_node = tree.get_node(self.conn, self.child_node.id)

        self._push_child_branch()
        self._mark_done("done" if improved else "no_change")
        # --cleanup-worktree is honored in `run()`'s finally block so it
        # also fires for the no_change early-exit path above.
        if self.cfg.auto_report:
            self._auto_report()
        return 0

    def _archive_final_eval_job_if_small(self, job_dir: Path) -> Path:
        """Copy final eval artifacts into reports before worktree cleanup.

        Harbor trial logs live under the per-pipeline worktree's `jobs/`
        directory. When `--cleanup-worktree` is enabled, that directory is
        removed at process exit, leaving only aggregate DB fields. Keep a
        bounded copy under `nodes/<id>/evals/` so final-eval regressions can
        be inspected later without allowing a single large run to bloat the
        reports directory.
        """
        if not self.child_node:
            return job_dir
        job_dir = Path(job_dir)
        if not job_dir.is_dir():
            return job_dir

        max_bytes = _final_eval_archive_max_bytes()
        size = _dir_size_bytes(job_dir, max_bytes=max_bytes if max_bytes > 0 else None)
        if max_bytes > 0 and size > max_bytes:
            self.log.info(
                "not archiving final eval job %s: %.1f MiB exceeds cap %.1f MiB",
                job_dir,
                size / (1024 * 1024),
                max_bytes / (1024 * 1024),
            )
            return job_dir

        node_dir = tree.node_dir(
            self.cfg.reports_root, self.cfg.campaign, self.child_node.id,
        )
        dest = node_dir / "evals" / job_dir.name
        try:
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(job_dir, dest, symlinks=True)
            self.log.info(
                "archived final eval job logs: %s -> %s (%.1f MiB)",
                job_dir,
                dest,
                size / (1024 * 1024),
            )
            return dest
        except Exception:
            self.log.warning("could not archive final eval job %s", job_dir, exc_info=True)
            return job_dir

    def _mark_done(self, status: str) -> None:
        tree.update_pipeline(
            self.conn, self.pipeline_id,
            status=status, finished_at=tree.utcnow_iso(),
        )

    def _push_child_branch(self) -> bool:
        """Best-effort remote sync for every node branch.

        Regular evolve nodes no longer open PRs themselves, but their branch
        and final commit still need to exist on the remote so merge-only runs
        from another checkout or host can materialize them later.
        """
        if not self.worktree:
            return False
        try:
            worktree.push_branch(self.worktree.monet_dir, self.worktree.monet_branch)
            self.log.info("pushed node branch: %s", self.worktree.monet_branch)
            return True
        except Exception as e:
            self.log.warning(
                "could not push node branch %s: %s",
                self.worktree.monet_branch,
                e,
            )
            return False

    def _open_pr_and_summarize(self) -> None:
        assert self.worktree and self.child_node and self.parent_node
        # 1. Push branch.
        try:
            worktree.push_branch(self.worktree.monet_dir, self.worktree.monet_branch)
        except Exception as e:
            self.log.warning("git push failed: %s — skipping PR + learnings", e)
            return
        # 2. Open PR via gh.
        body = self._pr_body()
        pr_url = None
        gh_cmd = [
            "gh", "pr", "create",
            "--base", "main",
            "--head", self.worktree.monet_branch,
            "--title", f"[self-evolve] {self.worktree.monet_branch}",
            "--body", body,
        ]
        # Only pass --repo if the caller explicitly set one; otherwise gh
        # auto-detects from the submodule's `origin` remote (works against
        # any fork).
        if self.cfg.monet_repo_url:
            gh_cmd[2:2] = ["--repo", self.cfg.monet_repo_url]

        try:
            proc = subprocess.run(
                gh_cmd,
                cwd=str(self.worktree.monet_dir),
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode == 0:
                pr_url = (proc.stdout or "").strip().splitlines()[-1]
                self.log.info("PR opened: %s", pr_url)
            else:
                self.log.warning("gh pr create failed (rc=%d): %s",
                                 proc.returncode, proc.stderr)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            self.log.warning("gh unavailable or timed out: %s", e)

        # Record PR URL in the node's resolved_tasks if we got one.
        if pr_url:
            resolved = self.child_node.resolved_tasks
            resolved.append({"pr_url": pr_url, "branch": self.worktree.monet_branch})
            tree.update_node(self.conn, self.child_node.id, resolved_tasks_json=resolved)

        # 3. Generate learnings.md via cursor-agent.
        self._write_learnings(pr_url)

    def _pr_body(self) -> str:
        assert self.child_node and self.parent_node
        kept = [o for o in self.iterations if o.committed_shas and not o.reverted]
        lines = [
            f"## Self-evolve auto-PR ({self.cfg.campaign})",
            "",
            f"- pipeline: `{self.pipeline_id}`",
            f"- parent: `{self.parent_node.branch_name}` "
            f"@ `{worktree.short_sha(self.parent_node.commit_sha or '?')}` "
            f"(score {self.parent_node.score})",
            f"- child:  `{self.child_node.branch_name}` "
            f"@ `{worktree.short_sha(self.child_node.commit_sha or '?')}` "
            f"(score {self.child_node.score})",
            f"- claimed tasks: {self.claimed_tasks}",
            f"- iterations: {len(self.iterations)} ({len(kept)} kept)",
            "",
            "See `effort.md` in the campaign reports dir for per-iteration details:",
            f"`{self.child_node.effort_md_path}`",
        ]
        return "\n".join(lines)

    def _write_learnings(self, pr_url: str | None) -> None:
        assert self.worktree and self.child_node and self.parent_node
        prompt = cursor_agent.render_prompt(
            self._template_path("learnings.md"),
            {
                "campaign": self.cfg.campaign,
                "pipeline_id": self.pipeline_id,
                "subset": self.cfg.subset_label,
                "monet_branch": self.worktree.monet_branch,
                "parent_score": self.parent_node.score,
                "child_score": self.child_node.score,
                "delta": (self.child_node.score or 0) - (self.parent_node.score or 0),
                "n_iterations": len(self.iterations),
                "pr_url": pr_url,
                "claimed_tasks": self.claimed_tasks,
                "resolved_tasks": [
                    t for t in self.claimed_tasks
                    if any(o.rewards_per_task.get(t, 0.0) >= 1.0 for o in self.iterations)
                ],
                "effort_md_path": self.child_node.effort_md_path,
            },
        )
        self._save_prompt(0, "learnings", prompt)
        result = meta_agent.run(
            prompt,
            workspace=self.worktree.eval_dir,
            log_path=self.cursor_log_dir / "learnings.log",
            model=self.cfg.meta_model,
            plan_mode=True,                # learnings is read-only synthesis
            timeout_s=self.cfg.picker_timeout_s,
            reasoning_effort=self.cfg.meta_effort,
        )
        if result.error:
            self.log.warning("learnings.md generation failed: %s", result.error)
            return
        node_dir = tree.node_dir(
            self.cfg.reports_root, self.cfg.campaign, self.child_node.id,
        )
        learnings_path = node_dir / "learnings.md"
        learnings_path.write_text(result.text or "_(empty)_\n")
        self.log.info("learnings: %s", learnings_path)

    def _cleanup_worktree(self) -> None:
        if not self.worktree:
            return
        self.log.info("cleaning up worktree %s", self.worktree.eval_dir)
        removed = worktree.remove_eval_worktree(
            self.worktree.eval_dir, repo_root=self.cfg.repo_root,
        )
        if not removed:
            self.log.warning(
                "worktree cleanup fell back or timed out for %s",
                self.worktree.eval_dir,
            )
        deleted = worktree.delete_eval_branch(
            self.worktree.eval_branch, repo_root=self.cfg.repo_root,
        )
        if not deleted:
            self.log.warning(
                "eval branch cleanup did not delete %s", self.worktree.eval_branch,
            )

    def _auto_report(self) -> None:
        # Shells out to scripts/self_evolve_report.py. Keep self-contained.
        script = self.cfg.repo_root / "scripts" / "self_evolve_report.py"
        if not script.is_file():
            self.log.warning("--auto-report on but %s not found", script)
            return
        try:
            subprocess.run(
                ["uv", "run", "python", str(script),
                 "--campaign", self.cfg.campaign,
                 "--config-path", str(self.cfg.config_path)],
                cwd=str(self.cfg.repo_root),
                check=False, timeout=30 * 60,
            )
        except subprocess.TimeoutExpired:
            self.log.warning("auto-report timed out")


class FullSetEvalPipeline:
    """Evaluate an existing node on the full benchmark without creating a node."""

    def __init__(self, cfg: PipelineConfig, *, target_node_id: str) -> None:
        self.cfg = cfg
        self.target_node_id = target_node_id
        self.pipeline_id = cfg.pipeline_id_override or tree.new_id()
        self.campaign_dir = tree.campaign_root(cfg.reports_root, cfg.campaign)
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = tree.db_path_for(cfg.reports_root, cfg.campaign)
        self.conn = tree.connect(self.db_path)
        self.pipeline_log_dir = tree.pipeline_dir(cfg.reports_root, cfg.campaign, self.pipeline_id)
        self.pipeline_log_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_path = self.pipeline_log_dir / "run.log"
        self.log = _make_logger(f"selfevolve.fullset.{self.pipeline_id}", self.run_log_path)
        self.target_node: tree.Node | None = None
        self.worktree: worktree.Worktree | None = None

    def _run_full(self, **kwargs):
        """Full-set eval for the confirm/promotion path.

        Mirrors ``SelfEvolvePipeline._run_full`` (this class doesn't inherit it):
        a direct ``eval_runner.run_full`` for the "best" metric, avg@k when the
        task list is known and ``MONET_EVAL_FULLSET_METRIC=avg``, with whole-job
        infra-failure retries so a transient cluster/tunnel outage doesn't mark
        an otherwise-good node failed. (Fixes the AttributeError that crashed the
        fullset confirm eval and let phantom subset-1.0 nodes early-stop the run.)
        """
        metric = os.environ.get(
            "MONET_EVAL_FULLSET_METRIC",
            getattr(self.cfg, "fullset_eval_metric", "best"),
        ).strip().lower()
        task_names = kwargs.get("task_names") or []

        def _call_once():
            if metric == "avg" and not task_names:
                raise ValueError(
                    "MONET_EVAL_FULLSET_METRIC=avg requires an explicit task "
                    "list (MONET_EVAL_FULLSET_TASKS); refusing to silently fall "
                    "back to best-of-N scoring. (legitimacy audit 2026-07-02)"
                )
            if metric == "avg" and task_names:
                k = int(getattr(self.cfg, "fullset_eval_k_samples", 5))
                return eval_runner.run_full_avg_k(
                    config_path=kwargs.get("config_path"),
                    cwd=kwargs.get("cwd"),
                    task_names=task_names,
                    k_samples=k,
                    job_name=kwargs.get("job_name"),
                    extra_env=kwargs.get("extra_env"),
                    tee_log_path=kwargs.get("tee_log_path"),
                )
            return eval_runner.run_full(**kwargs)

        attempts = _infra_eval_retries() + 1
        for attempt in range(1, attempts + 1):
            try:
                return _call_once()
            except eval_runner.EvalInfrastructureError:
                if attempt >= attempts:
                    raise
                delay = min(120, 20 * attempt)
                self.log.warning(
                    "fullset eval whole-job INFRA failure (attempt %d/%d); "
                    "transient — retrying in %ds before marking the node failed",
                    attempt, attempts, delay,
                )
                time.sleep(delay)

    def run(self) -> int:
        try:
            self.target_node = tree.get_node(self.conn, self.target_node_id)
            if self.target_node is None:
                raise RuntimeError(f"target node {self.target_node_id!r} not found")
            if not self.target_node.commit_sha:
                raise RuntimeError(f"target node {self.target_node_id!r} has no commit")
            tree.insert_pipeline(
                self.conn,
                id=self.pipeline_id,
                campaign=self.cfg.campaign,
                parent_node_id=self.target_node.parent_id,
                log_path=str(self.run_log_path),
                worktree_path=None,
                pid=os.getpid(),
                pipeline_kind="fullset_eval",
                target_node_id=self.target_node_id,
            )
            self.worktree = worktree.add_eval_worktree(
                pipeline_id=self.pipeline_id,
                parent_commit=self.target_node.commit_sha,
                repo_root=self.cfg.repo_root,
            )
            tree.update_pipeline(
                self.conn,
                self.pipeline_id,
                status="eval",
                worktree_path=str(self.worktree.eval_dir),
            )
            self.log.info(
                "running fullset eval for node=%s commit=%s",
                self.target_node_id,
                worktree.short_sha(self.target_node.commit_sha),
            )
            # avg@k requires an EXPLICIT task list (run_subset_sampled rejects an
            # empty list, unlike run_full's []=all). When MONET_EVAL_FULLSET_METRIC
            # =avg and MONET_EVAL_FULLSET_TASKS is provided (comma-sep), pass it so
            # _run_full routes to run_full_avg_k (leaderboard avg@k). Otherwise []
            # = full best-of-N sweep, unchanged.
            _fs_metric = os.environ.get("MONET_EVAL_FULLSET_METRIC", "").strip().lower()
            _fs_tasks: list[str] = []
            if _fs_metric == "avg":
                _raw = os.environ.get("MONET_EVAL_FULLSET_TASKS", "").strip()
                _fs_tasks = [t.strip() for t in _raw.split(",") if t.strip()]
                if _fs_tasks:
                    self.log.info("fullset eval: avg@k over %d explicit tasks", len(_fs_tasks))
            with self._heartbeat_during_eval("fullset eval"):
                result = self._run_full(
                    config_path=self.cfg.config_path,
                    cwd=self.worktree.eval_dir,
                    subset="full",
                    task_names=_fs_tasks,
                    job_name=f"fullset_{self.pipeline_id}",
                    extra_args=_fullset_eval_extra_args(self.cfg),
                    tee_log_path=self.run_log_path,
                )
            archived = self._archive_fullset_eval_job(result.job_dir)
            result.job_dir = archived
            parent_eval = None
            if self.target_node.parent_id:
                parent_eval = tree.node_search_eval(
                    self.conn,
                    campaign=self.cfg.campaign,
                    node_id=self.target_node.parent_id,
                )
            improved_tasks, regressed_tasks = tree.task_deltas(
                parent_solved=parent_eval.solved_tasks if parent_eval else [],
                parent_unsolved=parent_eval.failed_tasks if parent_eval else [],
                child_solved=result.solved_tasks,
                child_unsolved=result.failed_task_names,
            )
            full_eval = tree.upsert_node_eval(
                self.conn,
                campaign=self.cfg.campaign,
                node_id=self.target_node_id,
                eval_kind="fullset_final",
                subset_label="full",
                task_names=list(result.task_names),
                n_trials=result.n_trials,
                n_errors=result.n_errors,
                score=result.score,
                job_log_path=str(archived),
                solved_tasks=list(result.solved_tasks),
                unsolved_tasks=list(result.unsolved_tasks),
                partially_solved_tasks=list(result.partially_solved_tasks),
                task_rewards=dict(result.task_rewards),
                improved_tasks=improved_tasks,
                regressed_tasks=regressed_tasks,
                source_pipeline_id=self.pipeline_id,
                metadata={"basis": "supervisor_promoted_fullset"},
            )
            subset_eval = tree.node_search_eval(
                self.conn, campaign=self.cfg.campaign, node_id=self.target_node_id,
            )
            if subset_eval is not None:
                for kind, task in (
                    [("improved", t) for t in full_eval.improved_tasks]
                    + [("regressed", t) for t in full_eval.regressed_tasks]
                ):
                    tree.insert_task_experience(
                        self.conn,
                        campaign=self.cfg.campaign,
                        task=task,
                        node_id=self.target_node_id,
                        pipeline_id=self.pipeline_id,
                        worker_kind="fullset_eval",
                        commit_sha=self.target_node.commit_sha,
                        commit_number=None,
                        experience_kind=kind,
                        eval_kind="fullset_final",
                        before_reward=subset_eval.task_rewards.get(task),
                        after_reward=full_eval.task_rewards.get(task),
                        analysis=(
                            f"{kind} confirmed or discovered by promoted full-set eval."
                        ),
                        code_change_summary="Automatically inferred from full-set evaluation delta.",
                        artifact_paths=[str(archived)],
                        confidence=0.85,
                        metadata={"source": "fullset_promotion"},
                    )
            tree.update_pipeline(
                self.conn,
                self.pipeline_id,
                status="done",
                finished_at=tree.utcnow_iso(),
            )
            self.log.info("fullset score for %s: %.4f", self.target_node_id, result.score)
            return 0
        except Exception as exc:
            self.log.exception("fullset eval failed: %s", exc)
            try:
                tree.update_pipeline(
                    self.conn,
                    self.pipeline_id,
                    status="failed",
                    finished_at=tree.utcnow_iso(),
                )
            except Exception:
                pass
            return 1
        finally:
            try:
                tree.release_node_eval_claims(self.conn, pipeline_id=self.pipeline_id)
            except Exception:
                pass
            try:
                if self.cfg.cleanup_worktree and self.worktree is not None:
                    worktree.remove_eval_worktree(
                        self.worktree.eval_dir,
                        repo_root=self.cfg.repo_root,
                    )
                    worktree.delete_eval_branch(
                        self.worktree.eval_branch,
                        repo_root=self.cfg.repo_root,
                    )
            except Exception:
                self.log.warning("fullset worktree cleanup failed", exc_info=True)
            self.conn.close()

    @contextmanager
    def _heartbeat_during_eval(self, label: str):
        stop = threading.Event()
        interval_s = _eval_heartbeat_interval_s()

        def _beat_loop() -> None:
            conn = tree.connect(self.db_path)
            try:
                while not stop.wait(interval_s):
                    try:
                        tree.heartbeat(conn, self.pipeline_id)
                    except Exception:
                        self.log.debug("heartbeat failed during %s", label, exc_info=True)
            finally:
                conn.close()

        tree.heartbeat(self.conn, self.pipeline_id)
        thread = threading.Thread(
            target=_beat_loop,
            name=f"selfevolve-fullset-heartbeat-{self.pipeline_id}",
            daemon=True,
        )
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=5)
            tree.heartbeat(self.conn, self.pipeline_id)

    def _archive_fullset_eval_job(self, job_dir: Path) -> Path:
        job_dir = Path(job_dir)
        if self.target_node is None or not job_dir.is_dir():
            return job_dir
        node_dir = tree.node_dir(
            self.cfg.reports_root, self.cfg.campaign, self.target_node.id,
        )
        dest = node_dir / "evals" / job_dir.name
        try:
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(job_dir, dest, symlinks=True)
            return dest
        except Exception:
            self.log.warning("could not archive fullset eval job %s", job_dir, exc_info=True)
            return job_dir


# ─── Helpers ─────────────────────────────────────────────────────────


def _task_only(trial_names: list[str]) -> list[str]:
    """Strip __<hash> trial suffix from a list of trial names."""
    return eval_runner.task_bases(trial_names)


def _task_only_str(trial_name: str) -> str:
    return eval_runner.task_base(trial_name)


def _review_summary_md(outcome: IterationOutcome, *, produced_commit: bool) -> str:
    error = outcome.review_error or "none"
    return (
        f"- review: duration_s={outcome.review_duration_ms / 1000:.1f}, "
        f"timed_out={outcome.review_timed_out}, "
        f"produced_commit={produced_commit}, error={error!r}\n"
    )


def _bootstrap_n_concurrent(config_path: Path) -> int | None:
    """Read `harbor.n_concurrent_bootstrap` from the harbor YAML, if set.

    Returns None when the key is absent, the file can't be parsed, or the
    value isn't a positive int — in all of which cases the caller falls back
    to the default `harbor.n_concurrent`. Soft-fail by design: this knob is
    a wall-clock optimization for the one-time bootstrap precursor, never a
    correctness gate.
    """
    try:
        import yaml  # local import — keeps pipeline.py importable in environments
                     # without PyYAML (tests, lightweight tooling).
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, ImportError):
        return None
    try:
        v = (cfg.get("harbor") or {}).get("n_concurrent_bootstrap")
    except AttributeError:
        return None
    if not isinstance(v, int) or v <= 0:
        return None
    return v


def _final_eval_extra_args(config_path: Path) -> list[str]:
    """Return Harbor CLI extras for final full-set evals only."""
    n_attempts = _harbor_positive_int(config_path, "final_eval_n_attempts")
    if n_attempts is None:
        return []
    return ["--n-attempts", str(n_attempts)]


def _fullset_eval_extra_args(cfg: PipelineConfig) -> list[str]:
    n_attempts = cfg.fullset_eval_n_attempts or _harbor_positive_int(
        cfg.config_path, "final_eval_n_attempts",
    )
    if not n_attempts:
        n_attempts = 2
    return ["--n-attempts", str(max(1, int(n_attempts)))]


def _harbor_positive_int(config_path: Path, key: str) -> int | None:
    try:
        import yaml  # local import — keeps pipeline.py importable in lightweight tooling.
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, ImportError):
        return None
    try:
        v = (cfg.get("harbor") or {}).get(key)
    except AttributeError:
        return None
    if not isinstance(v, int) or v <= 0:
        return None
    return v


# ─── Phase-4 picker helpers (pure functions, easy to unit-test) ──────


_SELECTED_RE = re.compile(
    r"<<<SELECTED_COMMIT>>>\s*\n\s*([0-9a-fA-F]{7,40})\s*\n\s*<<<END>>>",
    flags=re.MULTILINE,
)


def _parse_picked_commit(text: str) -> str | None:
    """Extract the SHA from the agent's <<<SELECTED_COMMIT>>> block.

    Accepts any hex length between 7 and 40 (the agent might output a
    short SHA even though the template asks for 40). Returns the raw
    captured string (lowercase) or None if no block matches.
    """
    m = _SELECTED_RE.search(text)
    return m.group(1).lower() if m else None


def _resolve_pick_against_candidates(
    raw_pick: str,
    candidates: list[dict[str, Any]],
    parent_full_sha: str,
) -> str | None:
    """Validate the agent's pick against the known candidate full SHAs.

    The agent might output a short SHA; we accept any prefix-match
    against either the parent's commit or one of the candidate full
    SHAs. Returns the resolved 40-char SHA, or None if no match.
    """
    raw_pick = raw_pick.lower()
    if parent_full_sha and parent_full_sha.lower().startswith(raw_pick):
        return parent_full_sha
    for cand in candidates:
        full = cand["full_sha"]
        if full.lower().startswith(raw_pick):
            return full
    return None


def _render_root_works_md(node: tree.Node, *, campaign: str, pending: bool) -> str:
    """Format the baseline-summary markdown for a root node.

    Pure function (no I/O) so the regression test can pin the rendered
    body directly. The orchestrator's `_write_root_works_md` wraps this
    with disk + DB writes.
    """
    short = node.commit_sha[:7] if node.commit_sha else "?"
    lines = [
        f"# Root baseline — `{campaign}` · subset `{node.subset}`",
        "",
        f"- **branch:** `{node.branch_name}`",
        f"- **commit:** `{short}` (`{node.commit_sha or '?'}`)",
        f"- **subset:** `{node.subset}`",
        f"- **node id:** `{node.id}`",
    ]
    if pending:
        lines += [
            "- **status:** _pending baseline eval_",
            "",
            "The baseline harbor run is in progress. This file is overwritten "
            "with the final score, failing-task list, and eval job-log path as "
            "soon as Phase 2 finishes.",
        ]
    else:
        score_str = f"{node.score:.4f}" if node.score is not None else "?"
        failed = node.failed_tasks
        lines += [
            f"- **baseline score:** {score_str}",
            f"- **failing tasks ({len(failed)}):**",
        ]
        if failed:
            lines += [f"  - `{t}`" for t in failed]
        else:
            lines.append("  - _none — every task in this subset passed at baseline_")
        if node.job_log_path:
            lines += ["", f"- **job log:** `{node.job_log_path}`"]
        lines += [
            "",
            "> Campaign roots are created from Harbor job runs; use "
            "`--baseline-logs jobs/<job>` or `--fresh` to start a campaign.",
        ]
    return "\n".join(lines) + "\n"


def _resolved_summary(
    iterations: list[IterationOutcome],
    claimed: list[str],
) -> list[dict[str, Any]]:
    """Compact JSON-friendly summary of which tasks ended up resolved."""
    final_rewards: dict[str, float] = {}
    for it in iterations:
        for k, v in it.rewards_per_task.items():
            final_rewards[k] = max(final_rewards.get(k, -1.0), v)
    return _resolved_summary_from_rewards(final_rewards, claimed)


def _resolved_summary_from_rewards(
    final_rewards: dict[str, float],
    claimed: list[str],
) -> list[dict[str, Any]]:
    """Build the visualizer's resolved-task summary from canonical rewards."""
    return [
        {"task": t, "final_reward": final_rewards.get(t, 0.0),
         "resolved": final_rewards.get(t, 0.0) >= 1.0}
        for t in claimed
    ]


__all__ = ["PipelineConfig", "SelfEvolvePipeline", "FullSetEvalPipeline"]


def _build_task_contract(tasks, dataset, *, max_tasks=6, toml_limit=1200):
    """Proposer task-context from PUBLIC task spec ONLY (no verifier/test/solution
    files). (legitimacy audit 2026-07-02)"""
    from pathlib import Path as _P
    def _safe_text(p, limit):
        try:
            raw = p.read_bytes()[: limit * 2]
        except OSError:
            return None
        if b"\x00" in raw:
            return None
        txt = raw.decode("utf-8", errors="replace")
        txt = "".join(ch for ch in txt if ch in "\t\n" or ch >= " ")
        return txt[:limit]
    sections = []
    for task in list(tasks or [])[:max_tasks]:
        tdir = _P(dataset) / task
        if not tdir.is_dir():
            continue
        toml = tdir / "task.toml"
        if toml.is_file():
            t = _safe_text(toml, toml_limit)
            if t:
                sections.append(f"### Task spec for `{task}`\n\n```\n{t}\n```")
    if not sections:
        return ""
    block = (
        "\n## Task specifications (public instruction/spec) — target THESE\n\n"
        "For each claimed task below, this is its PUBLIC instruction/spec. Make "
        "`monet_code` genuinely solve the task; derive and verify acceptance "
        "criteria yourself — do NOT hard-code answers or weaken any verification.\n\n"
        + "\n\n".join(sections)
        + "\n"
    )
    return block.replace("\x00", "")


def _union_solved_pools(base, root_solved, *, enabled=True):
    """Order-preserving deduped union of base with root solves. (audit 2026-07-02)"""
    pool = [t for t in (base or []) if t]
    if not enabled:
        return pool
    seen = set(pool)
    for t in (root_solved or []):
        if t and t not in seen:
            seen.add(t)
            pool.append(t)
    return pool


def _risk_class_downgrade(decision, surface, n_regressions, *, routed=False):
    """PROMOTE of CODE/MIXED unrouted edit w/ >0 regressions -> ARCHIVE. (audit 2026-07-02)"""
    if (decision == "PROMOTE" and surface in ("code", "mixed")
            and not routed and (n_regressions or 0) > 0):
        return "ARCHIVE"
    return decision

"""Optional integration shim between ``self_evolve`` and ``atelier``.

This module is the **only** place inside ``self_evolve`` that imports
``atelier``. The ``pipeline.py`` orchestrator calls two functions exposed
here, both gated on env vars so the default ``self_evolve`` behavior is
unchanged:

- ``maybe_blend_score(ctx, base_score)`` — when ``DARWINX_GATE_FITNESS_ALPHA``
  is > 0, blends Atelier's verifier-augmented fitness with the natural
  pass-rate before the score is written to the tree. Otherwise returns
  ``base_score`` unchanged.
- ``on_node_finalized(ctx)`` — when ``DARWINX_GATE_ENABLED=1``, runs the
  Atelier gate (scope filter + any other layers configured via env
  vars) and writes a sidecar JSON decision to
  ``reports/<campaign>/atelier/<node_id>.json``. Otherwise no-op.

Both functions catch exceptions from the Atelier layer and log them
without re-raising. The hook is opt-in and must not be able to break
``self_evolve``'s pipeline.

Env vars:

- ``DARWINX_GATE_ENABLED`` (``0`` / ``1``, default ``0``) — record per-node
  gate decisions.
- ``DARWINX_GATE_FITNESS_ALPHA`` (float in [0, 1], default ``0``) — blending
  weight for verifier-augmented fitness. 0 = pure pass-rate (unchanged
  behavior); 0.3 is the recommended starting point per
  ``atelier.verifier_fitness.DEFAULT_VERIFIER_WEIGHT``.
- ``DARWINX_GATE_SCOPE_MODE`` (``soft_flag`` / ``strict_reject``, default
  ``soft_flag``) — Layer-4 scope filter mode.
- ``DARWINX_GATE_REPORTS_SUBDIR`` (default ``atelier``) — subdirectory under
  ``reports/<campaign>/`` to write decision sidecars into.
- ``DARWINX_GATE_PREDICTIONS_ENABLED`` (``0`` / ``1``, default ``1``) — if
  ``1``, ``capture_review_prediction`` parses the proposer's review
  text for a ``predicted_impact`` YAML block and writes a per-node
  sidecar. Independent of ``DARWINX_GATE_ENABLED`` — predictions are
  cheap to capture even when the L5 gate isn't running.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from evolve.worktree import AGENT_SUBMODULE


logger = logging.getLogger("evolve.gate_hook")


# ─── Env-var feature flags ────────────────────────────────────────────────


def is_gate_enabled() -> bool:
    """Whether ``on_node_finalized`` should run the Atelier gate."""
    return os.environ.get("DARWINX_GATE_ENABLED", "0").strip() == "1"


def fitness_alpha() -> float:
    """Atelier fitness-blend alpha, parsed from env. 0 = disabled."""
    raw = os.environ.get("DARWINX_GATE_FITNESS_ALPHA", "0").strip()
    try:
        val = float(raw)
    except ValueError:
        logger.warning(
            "DARWINX_GATE_FITNESS_ALPHA=%r is not a valid float; treating as 0", raw
        )
        return 0.0
    if not 0.0 <= val <= 1.0:
        logger.warning(
            "DARWINX_GATE_FITNESS_ALPHA=%r is outside [0, 1]; clamping to 0", raw
        )
        return 0.0
    return val


def is_fitness_enabled() -> bool:
    return fitness_alpha() > 0.0


def is_predictions_enabled() -> bool:
    """Whether ``capture_review_prediction`` should run.

    Default ``1`` — capturing predictions is cheap (no LLM call) and
    populates an analytics sidecar.

    The L5 gate that USED to consume these sidecars has been retired
    (replaced by ``run_equivalence_gate``). Predictions remain captured
    as off-line training data for future learners and for retrospective
    analysis (e.g., ``scripts/atelier_retrospective_equivalence.py``).
    """
    return os.environ.get("DARWINX_GATE_PREDICTIONS_ENABLED", "1").strip() == "1"


def is_equivalence_gate_enabled() -> bool:
    """Whether ``run_equivalence_gate`` should run before ``run_full``.

    Default ``0`` — off until calibrated. Enable with
    ``DARWINX_GATE_EQUIVALENCE_GATE_ENABLED=1`` for E4+ campaigns once the
    retrospective on E3's stored candidates confirms the gate would
    have rejected the catastrophes.
    """
    return os.environ.get("DARWINX_GATE_EQUIVALENCE_GATE_ENABLED", "0").strip() == "1"


def equivalence_probe_k() -> int:
    """How many probe tasks the equivalence gate runs on the child.

    Default 6 (per the plan). Cost ≈ K * $0.30 + ~3 LLM calls. Bump
    via ``DARWINX_GATE_EQUIVALENCE_PROBE_K``.
    """
    raw = os.environ.get("DARWINX_GATE_EQUIVALENCE_PROBE_K", "6").strip()
    try:
        val = int(raw)
    except ValueError:
        logger.warning(
            "DARWINX_GATE_EQUIVALENCE_PROBE_K=%r is not an int; using 6", raw
        )
        return 6
    return max(1, val)


def equivalence_n_adversarial() -> int:
    """How many of the K probes should come from the adversarial
    selector (vs the LLM "affected surfaces" selector).

    Default 3 (so K=6 splits 3 LLM + 3 adversarial). The adversarial
    picks come from parent.solved tasks NOT named by the analyzer —
    they catch under-claimed blast radius. Set to 0 to disable.
    """
    raw = os.environ.get("DARWINX_GATE_EQUIVALENCE_N_ADVERSARIAL", "3").strip()
    try:
        val = int(raw)
    except ValueError:
        logger.warning(
            "DARWINX_GATE_EQUIVALENCE_N_ADVERSARIAL=%r is not an int; using 3", raw
        )
        return 3
    return max(0, val)


def is_sibling_pool_enabled() -> bool:
    """Whether the analyze prompt should include GEA-style sibling
    evidence cards (arXiv:2602.04837 §3.2).

    Default ``1`` — cheap (just sidecar reads, no LLM calls) and
    high-signal. Disable with ``DARWINX_GATE_SIBLING_POOL_ENABLED=0``
    for ablation runs.
    """
    return (
        os.environ.get("DARWINX_GATE_SIBLING_POOL_ENABLED", "1").strip() == "1"
    )


def sibling_pool_k() -> int:
    """Group size K for the sibling pool. K=2 → focus + 1 sibling.

    Default 2 (matches GEA paper). K=1 disables the feature (no
    siblings picked). Bump cautiously — each sibling adds ~1-2KB
    to the analyze prompt + costs nothing in LLM calls.
    """
    raw = os.environ.get("DARWINX_GATE_SIBLING_POOL_K", "2").strip()
    try:
        val = int(raw)
    except ValueError:
        logger.warning(
            "DARWINX_GATE_SIBLING_POOL_K=%r is not int; using 2", raw
        )
        return 2
    return max(1, val)


def sibling_pool_novelty_m() -> int:
    """KNN-novelty M (number of neighbors) used when ranking
    candidate siblings. Default 4 (matches paper).
    """
    raw = os.environ.get("DARWINX_GATE_NOVELTY_M", "4").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 4


def equivalence_probe_k_fixed() -> bool:
    """If True, use the env-fixed ``DARWINX_GATE_EQUIVALENCE_PROBE_K`` as
    the probe count without invoking the LLM coverage sizer.

    Default ``0`` (sizer is on). Set to ``1`` for v5/v7-style
    fixed-K behavior (ablation).
    """
    return (
        os.environ.get(
            "DARWINX_GATE_EQUIVALENCE_PROBE_K_FIXED", "0"
        ).strip() == "1"
    )


def equivalence_n_votes() -> int:
    """How many independent verdict-stage LLM calls to aggregate
    (CANDOR-style majority vote, arXiv 2605.18747 §4.1.1).

    Default 3. Set to 1 to disable consensus (single-vote behavior,
    matches the original gate).
    """
    raw = os.environ.get("DARWINX_GATE_EQUIVALENCE_N_VOTES", "3").strip()
    try:
        val = int(raw)
    except ValueError:
        logger.warning(
            "DARWINX_GATE_EQUIVALENCE_N_VOTES=%r is not an int; using 3", raw
        )
        return 3
    return max(1, val)


def equivalence_require_extension() -> bool:
    """Whether the equivalence gate also requires
    ``|child.solved ∩ parent.unsolved| >= 1``.

    Default ``0`` (don't enforce yet, since we don't know child.solved
    until AFTER final-eval). Set ``DARWINX_GATE_EQUIVALENCE_REQUIRE_EXTENSION=1``
    when wiring the post-final-eval extension check.
    """
    return (
        os.environ.get("DARWINX_GATE_EQUIVALENCE_REQUIRE_EXTENSION", "0").strip()
        == "1"
    )


def equivalence_model() -> str:
    """Model identifier for the 3 LLM calls (analyzer / picker / verdict).

    Default ``gpt-5.4-mini`` — small + cheap is fine; the JSON we
    parse out is constrained and the verdict is heavily anchored by
    the probe results.
    """
    return os.environ.get(
        "DARWINX_GATE_EQUIVALENCE_MODEL", "gpt-5.4-mini"
    ).strip()


def equivalence_provider() -> str:
    return os.environ.get(
        "DARWINX_GATE_EQUIVALENCE_PROVIDER", "sfr_gateway"
    ).strip()


def scope_mode_str() -> str:
    return os.environ.get("DARWINX_GATE_SCOPE_MODE", "soft_flag").strip()


def reports_subdir() -> str:
    return os.environ.get("DARWINX_GATE_REPORTS_SUBDIR", "atelier").strip()


def is_ltm_enabled() -> bool:
    """Whether the LongTermMemory plumbing should run.

    Default ``1`` — LTM is cheap (just file I/O) and high-signal for
    the proposer brain. Opt out via ``DARWINX_GATE_LTM_ENABLED=0``.
    """
    return os.environ.get("DARWINX_GATE_LTM_ENABLED", "1").strip() == "1"


def ltm_max_entries() -> int:
    """Cap on LTM file size + prompt-view depth (default 30)."""
    raw = os.environ.get("DARWINX_GATE_LTM_MAX_ENTRIES", "30").strip()
    try:
        val = int(raw)
    except ValueError:
        logger.warning(
            "DARWINX_GATE_LTM_MAX_ENTRIES=%r is not an int; using 30", raw
        )
        return 30
    if val < 1:
        return 1
    return val


# ─── Equivalence gate context (separate from HookContext — extra fields) ──


@dataclass
class EquivalenceGateContext:
    """Inputs the equivalence gate needs from the pipeline.

    Distinct from ``HookContext`` because the equivalence gate needs
    more state than the post-final-eval hooks (it needs the parent's
    solved-tasks pool, the eval config, the eval cwd, the iteration's
    trial dir for digests). Built once per finalize call.
    """

    pipeline_id: str
    campaign: str
    child_node_id: str
    parent_node_id: str | None
    parent_commit: str | None
    child_commit: str
    """The child's monet_code HEAD sha at the moment the gate runs.
    Read via ``worktree.head_sha(monet_dir)`` BEFORE ``run_full``."""

    parent_solved_tasks: tuple[str, ...]
    """Pool the probe-task picker may draw from. From the parent's
    final-eval ``solved_tasks``. Empty for root or for parents that
    solved nothing → gate degrades to INCONCLUSIVE-but-accept."""

    parent_unsolved_tasks: tuple[str, ...]
    """For the optional extension_check (only used when
    DARWINX_GATE_EQUIVALENCE_REQUIRE_EXTENSION=1)."""

    diff_text: str
    """``git diff parent..child`` output. Truncated by the caller to
    keep prompt size bounded."""

    trial_digests: str
    """Pre-rendered markdown summary of the failing trials the
    proposer was reasoning about. May be empty."""

    eval_config_path: Path
    eval_cwd: Path
    """Args forwarded to ``eval_runner.run_subset`` when probing."""

    reports_root: Path = field(default=Path("reports"))

    parent_score_commit_sha: str | None = None
    """Belief-state snapshot (arXiv 2605.18747 §5.2.4 — SyncMind
    belief divergence). The parent's commit_sha at the moment the
    gate context was built. If the live DB row differs at gate-time,
    another pipeline has rewritten the parent and our cached
    ``parent_solved_tasks`` may be stale. In that case the gate logs
    a warning and degrades to accept-by-default rather than running
    against an inconsistent snapshot."""


# ─── Hook context (everything pipeline.py needs to pass in) ───────────────


@dataclass
class HookContext:
    """Snapshot of pipeline state at the moment Atelier is consulted."""

    pipeline_id: str
    campaign: str
    child_node_id: str
    parent_node_id: str | None
    parent_commit: str | None
    """The parent's HEAD commit on ``monet_code``. Used as the base for
    ``git diff``."""
    child_commit: str | None
    """The child's HEAD commit on ``monet_code``."""
    base_score: float
    """The natural pass-rate from the final eval."""
    per_task_rewards: dict[str, float] = field(default_factory=dict)
    reports_root: Path = field(default=Path("reports"))
    monet_dir: Path = field(default=Path(AGENT_SUBMODULE))
    final_eval_job_dir: Path | None = None
    """Path to the final-eval Harbor job dir (where trial transcripts
    live). Used by ``maybe_blend_score`` to load trajectories for the
    verifier. ``None`` disables verifier-fitness for this node."""


# ─── Score blending (touchpoint B) ────────────────────────────────────────


def verifier_model() -> str:
    """Model identifier for the verifier (env-overridable)."""
    return os.environ.get("DARWINX_GATE_VERIFIER_MODEL", "gpt-5.4-mini").strip()


def verifier_provider() -> str:
    """Credential provider for the verifier backend."""
    return os.environ.get("DARWINX_GATE_VERIFIER_PROVIDER", "sfr_gateway").strip()


def verifier_criteria_profile() -> str:
    """Criteria profile for the verifier.

    Profiles (see ``atelier.verifier.criteria_for_profile``):
      - ``default`` (5 generic criteria)
      - ``tb2`` (5 generic + error_recovery + task_interpretation)

    Default ``tb2`` because we're targeting TB-2; switch to ``default``
    for cross-benchmark transfer evaluations or for cost-sensitive
    runs (TB-2 profile is ~40 % more LLM calls per trajectory).
    """
    return os.environ.get("DARWINX_GATE_VERIFIER_CRITERIA_PROFILE", "tb2").strip()


def maybe_blend_score(ctx: HookContext, *, base_score: float) -> float:
    """If verifier-fitness is enabled, blend the base pass-rate with the
    Atelier verifier's per-trajectory aggregated score.

    Returns the blended score when ``DARWINX_GATE_FITNESS_ALPHA > 0`` and all
    of the following hold:
    - the ``atelier`` package is importable (with the OpenAI dep),
    - ``ctx.final_eval_job_dir`` points at a real Harbor job dir,
    - at least one trajectory loads successfully from it,
    - the verifier backend returns at least one assessment.

    Any failure (missing dep, missing job dir, verifier API error)
    logs at WARNING and returns ``base_score`` unchanged. This function
    is on the campaign hot path — it must never raise.
    """
    alpha = fitness_alpha()
    if alpha <= 0.0:
        return base_score

    if ctx.final_eval_job_dir is None or not Path(ctx.final_eval_job_dir).is_dir():
        logger.warning(
            "DARWINX_GATE_FITNESS_ALPHA=%.2f set but no final_eval_job_dir on the hook "
            "context (or dir missing) — score unchanged",
            alpha,
        )
        return base_score

    try:
        from gate import trajectory_loader, verifier as v_mod
        from gate import verifier_backend, verifier_fitness
    except ImportError as e:
        logger.warning(
            "DARWINX_GATE_FITNESS_ALPHA=%.2f set but atelier package missing a dep: %s "
            "— score unchanged (install 'atelier' extra for verifier support)",
            alpha,
            e,
        )
        return base_score

    task_ids = set(ctx.per_task_rewards.keys()) or None
    loaded = trajectory_loader.load_job(
        ctx.final_eval_job_dir, task_ids=task_ids
    )
    if not loaded:
        logger.warning(
            "atelier verifier: no trajectories loaded from %s — score unchanged",
            ctx.final_eval_job_dir,
        )
        return base_score

    try:
        backend = verifier_backend.backend_from_credentials(
            model=verifier_model(),
            provider=verifier_provider(),
        )
    except Exception as e:  # noqa: BLE001 — hook must not crash pipeline
        logger.warning(
            "atelier verifier: backend construction failed (%s) — score unchanged",
            e,
        )
        return base_score

    # MVP Component 3: Self-Grounded Verification template (priors-first,
    # adversarial, step-wise) to mitigate the verifier's agreement bias —
    # the phantom filter. Env DARWINX_GATE_VERIFIER_SGV=1; default = legacy template.
    _sgv_on = os.environ.get("DARWINX_GATE_VERIFIER_SGV", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    _verifier_kwargs = dict(
        backend=backend,
        criteria=v_mod.criteria_for_profile(verifier_criteria_profile()),
    )
    if _sgv_on and hasattr(v_mod, "SGV_PROMPT_TEMPLATE"):
        _verifier_kwargs["prompt_template"] = v_mod.SGV_PROMPT_TEMPLATE
        logger.info("atelier verifier: using Self-Grounded Verification (SGV) template")
    verifier_obj = v_mod.Verifier(**_verifier_kwargs)
    runner = verifier_fitness.FitnessRunner(verifier=verifier_obj, alpha=alpha)
    traj_by_task = trajectory_loader.to_verifier_inputs_by_task(loaded)

    try:
        fitness = runner.run_for_candidate(
            ctx.child_node_id,
            task_rewards=ctx.per_task_rewards,
            trajectories=traj_by_task,
        )
    except Exception as e:  # noqa: BLE001 — verifier API failures are expected
        logger.warning(
            "atelier verifier: FitnessRunner raised (%s) — score unchanged", e
        )
        return base_score

    logger.info(
        "atelier-fitness[%s] base=%.3f → blended=%.3f (alpha=%.2f, n_assessed=%d)",
        ctx.child_node_id,
        base_score,
        fitness.value,
        fitness.alpha,
        fitness.components.n_assessed,
    )
    return float(fitness.value)


# ─── Gate decision recording (touchpoint A) ──────────────────────────────


def _git_diff_files(
    *, monet_dir: Path, parent_commit: str | None, child_commit: str | None
) -> tuple[list[str], str | None]:
    """Return (changed_files, error). See ``atelier.cli._get_node_diff``
    for the same shape; we inline it here to keep gate_hook
    self-contained when the atelier package is unavailable."""
    if not monet_dir.is_dir():
        return ([], f"monet_dir missing: {monet_dir}")
    if not child_commit:
        return ([], "no child commit_sha")
    if not parent_commit:
        return ([], "root node (no parent)")

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(monet_dir),
                "diff",
                "--name-only",
                f"{parent_commit}..{child_commit}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return ([], f"git diff failed: {e}")
    if result.returncode != 0:
        return ([], f"git diff returncode={result.returncode}: {result.stderr.strip()}")
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return files, None


def _run_gate_and_serialize(ctx: HookContext) -> dict[str, Any] | None:
    """Try to import ``atelier`` and run the gate. Returns the JSON-safe
    decision dict, or ``None`` if ``atelier`` is unavailable. Never raises."""
    try:
        from gate import gate, scope_filter
    except ImportError as e:
        logger.warning(
            "DARWINX_GATE_ENABLED=1 but atelier package not importable: %s",
            e,
        )
        return None

    diff_files, diff_err = _git_diff_files(
        monet_dir=ctx.monet_dir,
        parent_commit=ctx.parent_commit,
        child_commit=ctx.child_commit,
    )

    try:
        scope = scope_filter.ScopeMode(scope_mode_str())
    except ValueError:
        logger.warning(
            "DARWINX_GATE_SCOPE_MODE=%r is not a valid mode; using soft_flag",
            scope_mode_str(),
        )
        scope = scope_filter.ScopeMode.SOFT_FLAG

    g = gate.AtelierGate(scope_mode=scope)
    try:
        decision = g.evaluate(
            node_id=ctx.child_node_id,
            candidate_id=ctx.child_node_id,
            diff_files=diff_files,
        )
    except Exception as e:  # noqa: BLE001 — hook must not crash pipeline
        logger.exception("Atelier gate.evaluate raised: %s", e)
        return None

    # Serialize the decision into a JSON-safe dict (asdict drills into
    # the dataclass-based payloads).
    layers_json = []
    for layer in decision.layers:
        entry: dict[str, Any] = {
            "name": layer.name,
            "status": layer.status.value,
            "summary": layer.summary,
        }
        if layer.payload is not None:
            try:
                entry["payload"] = asdict(layer.payload)
            except TypeError:
                entry["payload"] = repr(layer.payload)
        layers_json.append(entry)

    return {
        "schema_version": 1,
        "campaign": ctx.campaign,
        "pipeline_id": ctx.pipeline_id,
        "node_id": ctx.child_node_id,
        "parent_node_id": ctx.parent_node_id,
        "parent_commit": ctx.parent_commit,
        "child_commit": ctx.child_commit,
        "base_score": ctx.base_score,
        "n_diff_files": len(diff_files),
        "diff_error": diff_err,
        "accept": decision.accept,
        "reject_reasons": decision.reject_reasons,
        "layers": layers_json,
    }


def on_node_finalized(ctx: HookContext) -> None:
    """If ``DARWINX_GATE_ENABLED=1``, run the gate and write a sidecar
    decision JSON to ``reports/<campaign>/<subdir>/<node_id>.json``.

    Failures are logged + swallowed — Atelier never crashes self_evolve.
    """
    if not is_gate_enabled():
        return

    record = _run_gate_and_serialize(ctx)
    if record is None:
        return

    out_dir = ctx.reports_root / ctx.campaign / reports_subdir()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ctx.child_node_id}.json"
        out_path.write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
        logger.info(
            "atelier-gate[%s] %s — wrote %s",
            ctx.child_node_id,
            "accept" if record["accept"] else "reject",
            out_path,
        )
    except OSError as e:
        logger.warning(
            "failed to write atelier sidecar for node %s: %s",
            ctx.child_node_id,
            e,
        )


# ─── Equivalence gate (MatchFixGate, pre-final-eval) ────────────────────


_PROMPT_DIR = Path(__file__).resolve().parent.parent / "gate" / "prompts"


def _load_prompt(name: str) -> str:
    """Load a matchfix prompt template. Returns empty on missing file
    (the gate degrades to a conservative default rather than crashing).
    """
    path = _PROMPT_DIR / name
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("could not load prompt %s: %s", path, e)
        return ""


def _build_harbor_runner(
    *, eq_ctx: "EquivalenceGateContext",
):
    """Return a ``matchfix_gate.HarborRunner`` adapter that invokes
    ``eval_runner.run_subset`` on the child's worktree.

    The adapter returns a ``({task: reward}, {task: failure_digest})``
    tuple so the gate can attach first-failure evidence (MAGE-style,
    §4.2.1 of the harness survey) to the verdict + sidecar.
    """
    from pathlib import Path as _Path
    # coding-bench routes every eval through the codingbench adapter (the
    # same seam pipeline.py uses), not the legacy monet ``eval_runner``.
    from . import codingbench_eval as eval_runner

    def _run(*, child_commit_sha: str, probe_tasks):
        del child_commit_sha   # the worktree is already at this sha
        rewards: dict[str, float] = {}
        k = probe_k_samples()
        if k > 1:
            # Denoised avg@k probe: sample each probe task k times and only
            # treat a *statistically real* pass-rate drop as a regression.
            # Probes are drawn from the parent's solved set, so the parent's
            # rate is ~1.0; a child counts as regressed only if its k-sample
            # rate falls by >= eval_stats.DEFAULT_MIN_RATE_DELTA (>= 2/5 at
            # k=5), which ignores the one-off flips that produced the
            # campaign's false MODIFIED vetoes.
            from . import eval_stats
            sres = eval_runner.run_subset_sampled(
                config_path=eq_ctx.eval_config_path,
                cwd=eq_ctx.eval_cwd,
                task_names=list(probe_tasks),
                k_samples=k,
                job_name=f"equiv_{eq_ctx.child_node_id}",
            )
            rate_log: dict[str, float] = {}
            for raw_task, (rate, n) in sres.rates.items():
                base = str(raw_task).split("__", 1)[0]
                rate_log[base] = rate
                verdict = eval_stats.classify_task(
                    parent_rate=1.0, child_rate=rate, n_parent=k, n_child=n,
                )
                rewards[base] = 0.0 if verdict is eval_stats.TaskVerdict.REGRESSED else 1.0
            logger.info(
                "equivalence[%s] denoised probe avg@%d rates: %s",
                eq_ctx.child_node_id, k,
                {t: round(r, 2) for t, r in sorted(rate_log.items())},
            )
            job_dir = _Path(sres.job_dir)
        else:
            result = eval_runner.run_subset(
                config_path=eq_ctx.eval_config_path,
                cwd=eq_ctx.eval_cwd,
                task_names=list(probe_tasks),
                job_name=f"equiv_{eq_ctx.child_node_id}",
            )
            # Normalize rewards to bare task names.
            for rk, v in result.rewards_per_task.items():
                base = str(rk).split("__", 1)[0]
                try:
                    rewards[base] = float(v)
                except (TypeError, ValueError):
                    continue
            job_dir = _Path(result.job_dir)

        # First-failure digests for any regressed probe — pull verifier
        # stdout from the job dir. Best-effort: missing files just
        # contribute no digest (the verdict will still see rewards).
        digests: dict[str, str] = {}
        if job_dir.is_dir():
            for task in probe_tasks:
                if rewards.get(task, 0.0) >= 1.0:
                    continue
                # Trial dirs are named "<task>__<hash>"; pick the first.
                trial = next(
                    (d for d in job_dir.iterdir()
                     if d.is_dir() and d.name.startswith(f"{task}__")),
                    None,
                )
                if trial is None:
                    continue
                stdout_path = trial / "verifier" / "test-stdout.txt"
                if not stdout_path.is_file():
                    continue
                try:
                    text = stdout_path.read_text(
                        encoding="utf-8", errors="replace",
                    )
                except OSError:
                    continue
                if text.strip():
                    digests[task] = text[:1200]

        return (rewards, digests)

    class _Runner:
        def run(self, *, child_commit_sha: str, probe_tasks):
            return _run(child_commit_sha=child_commit_sha, probe_tasks=probe_tasks)

    return _Runner()


def equivalence_reprobe_enabled() -> bool:
    """Whether a failed probe is re-run once before it can drive a
    MODIFIED verdict. Immunizes the gate against a flaky proxy turn
    fabricating a regression on a task that should pass. Default on;
    disable with DARWINX_GATE_EQUIVALENCE_REPROBE=0.
    """
    return os.environ.get("DARWINX_GATE_EQUIVALENCE_REPROBE", "1").strip() == "1"


def probe_k_samples() -> int:
    """Independent samples per probe task for the denoised avg@k gate.

    The overnight campaign rejected every real fix because k=1 probes turn a
    4/5 task into a ~20%-of-the-time false "regression". Running each probe at
    k>=3 and only counting a *statistically real* pass-rate drop (see
    ``eval_stats.classify_task``) removes that noise veto. Default 3; set k=1 to
    restore the legacy single-sample probe. Override with DARWINX_GATE_PROBE_K_SAMPLES.
    """
    raw = os.environ.get("DARWINX_GATE_PROBE_K_SAMPLES", "3").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


def run_equivalence_gate(eq_ctx: "EquivalenceGateContext"):
    """Run the 5-stage MatchFixGate. Returns an ``EquivalenceVerdict``.

    Failure modes (any of which return a ``decision="INCONCLUSIVE",
    accept_for_final_eval=True`` verdict so a gate bug doesn't block a
    candidate that might otherwise be a legitimate improvement):

    - ``atelier`` package missing.
    - LLM credentials missing.
    - Harbor probe invocation raises.
    - Prompts could not be loaded.

    Sidecar JSON is written under
    ``reports/<campaign>/atelier/equivalence/<child>.equivalence.json``
    even on failure (with the failure mode recorded), so the
    retrospective script can audit gate behavior.
    """
    try:
        from gate import matchfix_gate
    except ImportError as e:
        logger.warning(
            "DARWINX_GATE_EQUIVALENCE_GATE_ENABLED=1 but atelier.matchfix_gate "
            "not importable (%s) — accepting candidate by default", e,
        )
        return _make_accept_default(
            reason=f"matchfix_gate import failed: {e}", eq_ctx=eq_ctx,
        )

    # ─── Belief-state freshness check (§5.2.4) ─────────────────────────
    # If parent_score_commit_sha drifted between context-build and gate-
    # run, our cached parent_solved_tasks is stale. Degrade to
    # accept-by-default rather than gating against a snapshot some
    # other pipeline has invalidated.
    if (
        eq_ctx.parent_score_commit_sha is not None
        and eq_ctx.parent_commit is not None
        and eq_ctx.parent_score_commit_sha != eq_ctx.parent_commit
    ):
        logger.warning(
            "equivalence[%s] belief-state drift detected: "
            "snapshot_sha=%s != ctx_parent_sha=%s → skipping gate "
            "(another pipeline rewrote the parent)",
            eq_ctx.child_node_id,
            eq_ctx.parent_score_commit_sha,
            eq_ctx.parent_commit,
        )
        return _make_accept_default(
            reason=(
                f"parent commit drifted from {eq_ctx.parent_score_commit_sha} "
                f"to {eq_ctx.parent_commit}; cached parent_solved_tasks may "
                f"be stale"
            ),
            eq_ctx=eq_ctx,
        )

    if not eq_ctx.parent_solved_tasks:
        # Root node or parent solved zero → no probes to run. Accept
        # by default and skip the LLM calls.
        verdict = matchfix_gate.aggregate_verdict(
            semantic_analysis=None,
            probe_results=matchfix_gate.ProbeResults(
                probe_tasks=(), rewards={},
            ),
            decision="INCONCLUSIVE",
            rationale="no parent_solved_tasks to probe — accepted by default",
            extension_required=False,
            parent_commit_sha=eq_ctx.parent_commit,
            parent_solved_count=0,
        )
        # Override to accept (the contract is trivially satisfied
        # when there's nothing to preserve).
        verdict = matchfix_gate.EquivalenceVerdict(
            decision=verdict.decision,
            k_picked_tasks=verdict.k_picked_tasks,
            per_task_results=verdict.per_task_results,
            extension_tasks_solved=verdict.extension_tasks_solved,
            n_regressions=verdict.n_regressions,
            semantic_analysis=verdict.semantic_analysis,
            verdict_rationale=verdict.verdict_rationale,
            accept_for_final_eval=True,
            extension_required=verdict.extension_required,
            verification_scope=verdict.verification_scope,
            change_contract=verdict.change_contract,
            consensus_votes=verdict.consensus_votes,
            failure_digests=verdict.failure_digests,
        )
        try:
            matchfix_gate.write_verdict_sidecar(
                reports_root=eq_ctx.reports_root,
                campaign=eq_ctx.campaign,
                child_node_id=eq_ctx.child_node_id,
                verdict=verdict,
            )
        except OSError as e:
            logger.warning("could not write equivalence sidecar: %s", e)
        return verdict

    analyze_tmpl = _load_prompt("matchfix_analyze.md")
    select_tmpl = _load_prompt("matchfix_select.md")
    verdict_tmpl = _load_prompt("matchfix_verdict.md")
    if not (analyze_tmpl and select_tmpl and verdict_tmpl):
        return _make_accept_default(
            reason="one or more matchfix prompt templates missing",
            eq_ctx=eq_ctx,
        )

    try:
        llm = matchfix_gate.chat_backend_from_credentials(
            model=equivalence_model(), provider=equivalence_provider(),
        )
    except Exception as e:  # noqa: BLE001 — credentials / import failures
        logger.warning(
            "equivalence gate: could not build LLM backend (%s) — accepting",
            e,
        )
        return _make_accept_default(
            reason=f"chat_backend_from_credentials failed: {e}", eq_ctx=eq_ctx,
        )

    # ─── Stage 1: analyze the diff ──────────────────────────────────────
    semantic = matchfix_gate.analyze_diff(
        diff_text=eq_ctx.diff_text,
        trial_digests=eq_ctx.trial_digests,
        llm=llm,
        prompt_template=analyze_tmpl,
    )
    logger.info(
        "equivalence[%s] semantic: surfaces=%s risk=%s",
        eq_ctx.child_node_id,
        list(semantic.modified_surfaces),
        semantic.risk_level,
    )

    # ─── Stage 1b: adaptive K via LLM coverage sizer (v8) ─────────────
    # Replaces the v5/v7 fixed DARWINX_GATE_EQUIVALENCE_PROBE_K=6 with an
    # LLM-recommended K based on risk_level + modified_surfaces. Opt
    # out via DARWINX_GATE_EQUIVALENCE_PROBE_K_FIXED=1.
    default_k = equivalence_probe_k()
    if equivalence_probe_k_fixed():
        k = default_k
        logger.info(
            "equivalence[%s] K=%d (fixed via env)",
            eq_ctx.child_node_id, k,
        )
    else:
        try:
            from gate import coverage_sizer
            sizer_tmpl = _load_prompt("coverage_size.md")
            if sizer_tmpl:
                rec = coverage_sizer.recommend_k(
                    semantic_analysis=semantic,
                    parent_solved_count=len(eq_ctx.parent_solved_tasks),
                    default_k=default_k,
                    llm=llm,
                    prompt_template=sizer_tmpl,
                )
                k = rec.k
                logger.info(
                    "equivalence[%s] LLM-sized K=%d (was default=%d). %s",
                    eq_ctx.child_node_id, k, default_k,
                    rec.rationale[:160],
                )
            else:
                k = default_k
                logger.warning(
                    "equivalence[%s] coverage_size.md missing — using K=%d",
                    eq_ctx.child_node_id, k,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "equivalence[%s] coverage sizer raised (%s) — fallback K=%d",
                eq_ctx.child_node_id, e, default_k,
            )
            k = default_k

    # ─── Stage 2: pick K probe tasks (LLM-affected + adversarial) ──────
    # Scale adversarial slot proportionally when K is sized adaptively.
    # The base ratio is the v5 default: 3 adversarial out of K=6 = 50%.
    n_adv_base = equivalence_n_adversarial()
    if equivalence_probe_k_fixed():
        n_adv = n_adv_base
    else:
        n_adv = max(0, min(k - 1, round(k * n_adv_base / max(1, default_k))))
    # Adversarial seed = stable hash of campaign+node so the
    # retrospective can reproduce the picks; per-candidate variability
    # diversifies coverage across the campaign.
    import hashlib as _hashlib
    seed = int(
        _hashlib.sha256(
            f"{eq_ctx.campaign}:{eq_ctx.child_node_id}".encode()
        ).hexdigest()[:8],
        16,
    )
    probes, adversarial_picks = matchfix_gate.select_probe_tasks(
        semantic_analysis=semantic,
        parent_solved_tasks=eq_ctx.parent_solved_tasks,
        k=k,
        llm=llm,
        prompt_template=select_tmpl,
        n_adversarial=n_adv,
        adversarial_seed=seed,
    )
    logger.info(
        "equivalence[%s] picked %d/%d probes (%d LLM + %d adversarial) "
        "from parent_solved (pool=%d)",
        eq_ctx.child_node_id, len(probes), k,
        len(probes) - len(adversarial_picks), len(adversarial_picks),
        len(eq_ctx.parent_solved_tasks),
    )

    if not probes:
        verdict = matchfix_gate.aggregate_verdict(
            semantic_analysis=semantic,
            probe_results=matchfix_gate.ProbeResults(probe_tasks=(), rewards={}),
            decision="INCONCLUSIVE",
            rationale="no probes selected",
            extension_required=False,
            parent_commit_sha=eq_ctx.parent_commit,
            parent_solved_count=len(eq_ctx.parent_solved_tasks),
        )
        try:
            matchfix_gate.write_verdict_sidecar(
                reports_root=eq_ctx.reports_root,
                campaign=eq_ctx.campaign,
                child_node_id=eq_ctx.child_node_id,
                verdict=verdict,
            )
        except OSError as e:
            logger.warning("could not write equivalence sidecar: %s", e)
        return verdict

    # ─── Stage 3: execute probes via Harbor ─────────────────────────────
    runner = _build_harbor_runner(eq_ctx=eq_ctx)
    try:
        probe_results = matchfix_gate.execute_probes(
            child_commit_sha=eq_ctx.child_commit,
            probe_tasks=probes,
            runner=runner,
            adversarial_picks=adversarial_picks,
        )
    except Exception as e:  # noqa: BLE001 — Harbor invocations can fail
        logger.warning(
            "equivalence[%s] probe execution failed (%s) — accepting "
            "to avoid blocking a candidate on infrastructure error",
            eq_ctx.child_node_id, e,
        )
        verdict = matchfix_gate.aggregate_verdict(
            semantic_analysis=semantic,
            probe_results=matchfix_gate.ProbeResults(probe_tasks=probes, rewards={}),
            decision="INCONCLUSIVE",
            rationale=f"probe execution raised: {e}",
            extension_required=False,
        )
        # Override accept_for_final_eval=True since we can't blame the
        # candidate for our infrastructure failure.
        verdict_dict = matchfix_gate.verdict_to_dict(verdict)
        verdict_dict["accept_for_final_eval"] = True
        try:
            sidecar_dir = (
                Path(eq_ctx.reports_root) / eq_ctx.campaign
                / "atelier" / "equivalence"
            )
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            (sidecar_dir / f"{eq_ctx.child_node_id}.equivalence.json").write_text(
                json.dumps(verdict_dict, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError:
            pass
        return matchfix_gate.EquivalenceVerdict(
            decision="INCONCLUSIVE",
            k_picked_tasks=probes,
            per_task_results={},
            extension_tasks_solved=(),
            n_regressions=0,
            semantic_analysis=semantic,
            verdict_rationale=f"probe execution failed: {e}",
            accept_for_final_eval=True,
        )

    logger.info(
        "equivalence[%s] probes: pass=%d fail=%d (regressed=%s; "
        "adversarial_regressed=%d/%d)",
        eq_ctx.child_node_id,
        probe_results.n_pass,
        probe_results.n_fail,
        list(probe_results.regressed_tasks),
        probe_results.n_adversarial_regressed,
        len(probe_results.adversarial_picks),
    )

    # ─── Stage 3b: re-probe failed probes once (proxy-noise immunity) ──
    # A single flaky proxy turn can make a probe that should PASS score
    # 0.0, fabricating a regression that would trigger a false MODIFIED
    # reject. Re-run just the regressed probes once; a task only counts as
    # regressed if it fails BOTH the original probe and the re-probe.
    if probe_results.regressed_tasks and equivalence_reprobe_enabled():
        regressed = list(probe_results.regressed_tasks)
        logger.info(
            "equivalence[%s] re-probing %d failed probe(s) to rule out "
            "proxy noise: %s",
            eq_ctx.child_node_id, len(regressed), regressed,
        )
        try:
            reout = runner.run(
                child_commit_sha=eq_ctx.child_commit, probe_tasks=regressed,
            )
            re_rewards, _re_digests = (
                reout if isinstance(reout, tuple) else (reout, {})
            )
            merged = dict(probe_results.rewards)
            recovered = []
            for t in regressed:
                if float(re_rewards.get(t, 0.0)) >= 1.0:
                    merged[t] = 1.0
                    recovered.append(t)
            if recovered:
                logger.info(
                    "equivalence[%s] re-probe RECOVERED %d false "
                    "regression(s) (proxy noise): %s",
                    eq_ctx.child_node_id, len(recovered), recovered,
                )
                probe_results = matchfix_gate.ProbeResults(
                    probe_tasks=probe_results.probe_tasks,
                    rewards=merged,
                    failure_digests={
                        k: v for k, v in probe_results.failure_digests.items()
                        if merged.get(k, 0.0) < 1.0
                    },
                    adversarial_picks=probe_results.adversarial_picks,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "equivalence[%s] re-probe failed (%s) — using original "
                "probe results", eq_ctx.child_node_id, e,
            )

    # ─── Stage 4: verdict (multi-vote consensus, CANDOR-style) ──────────
    n_votes = equivalence_n_votes()
    decision, rationale, votes = matchfix_gate.verdict_with_consensus(
        semantic_analysis=semantic,
        probe_results=probe_results,
        llm=llm,
        prompt_template=verdict_tmpl,
        n_votes=n_votes,
    )

    # ─── Stage 5: extension check (when configured) ─────────────────────
    extension = None
    if equivalence_require_extension():
        # We don't know child_solved yet (that's what final-eval would
        # produce). The post-final-eval pass will enforce this; here we
        # leave extension=None and accept on decision alone.
        pass

    verdict = matchfix_gate.aggregate_verdict(
        semantic_analysis=semantic,
        probe_results=probe_results,
        decision=decision,
        rationale=rationale,
        extension=extension,
        extension_required=False,   # always False at this point; the
                                    # post-final-eval pass handles it.
        parent_commit_sha=eq_ctx.parent_commit,
        parent_solved_count=len(eq_ctx.parent_solved_tasks),
        consensus_votes=votes,
    )

    try:
        matchfix_gate.write_verdict_sidecar(
            reports_root=eq_ctx.reports_root,
            campaign=eq_ctx.campaign,
            child_node_id=eq_ctx.child_node_id,
            verdict=verdict,
        )
    except OSError as e:
        logger.warning("could not write equivalence sidecar: %s", e)

    logger.info(
        "equivalence[%s] %s — accept=%s (rationale=%r)",
        eq_ctx.child_node_id,
        verdict.decision,
        verdict.accept_for_final_eval,
        (verdict.verdict_rationale or "")[:200],
    )
    return verdict


def _make_accept_default(*, reason: str, eq_ctx: "EquivalenceGateContext"):
    """Build an INCONCLUSIVE-but-accept verdict (used when the gate
    cannot run for infrastructure reasons)."""
    from gate import matchfix_gate

    verdict = matchfix_gate.EquivalenceVerdict(
        decision="INCONCLUSIVE",
        k_picked_tasks=(),
        per_task_results={},
        extension_tasks_solved=(),
        n_regressions=0,
        semantic_analysis=None,
        verdict_rationale=f"gate skipped: {reason}",
        accept_for_final_eval=True,
        extension_required=False,
    )
    try:
        matchfix_gate.write_verdict_sidecar(
            reports_root=eq_ctx.reports_root,
            campaign=eq_ctx.campaign,
            child_node_id=eq_ctx.child_node_id,
            verdict=verdict,
        )
    except OSError as e:
        logger.warning("could not write equivalence sidecar: %s", e)
    return verdict


# ─── Predicted-impact capture (touchpoint C, AHE-style falsifiability) ───


def capture_review_prediction(
    *,
    ctx: HookContext,
    review_text: str,
    iteration: int | None = None,
) -> None:
    """Parse the cursor-agent's review-step text for a
    ``predicted_impact`` YAML block and persist the result as a
    sidecar JSON.

    Called by ``pipeline._run_iteration`` immediately after the review
    step completes (whether or not the iteration ultimately commits —
    if the agent did emit a prediction we want to record it for
    debugging even if downstream layers later reject the candidate).

    If ``iteration`` is provided, also writes a per-iteration sidecar
    at ``<node>.iter_NN.predicted.json``. This preserves the full
    prediction history across iters so credibility analysis can find
    the BEST per-iter prediction, not just the latest one (which
    today's `ab_overnight_3` showed is often degraded from iter 1).

    Default-on (``DARWINX_GATE_PREDICTIONS_ENABLED=1``). Disabled-off via
    env. Never raises.
    """
    if not is_predictions_enabled():
        return
    if not review_text:
        return

    try:
        from gate import predictions
    except ImportError as e:
        logger.warning(
            "DARWINX_GATE_PREDICTIONS_ENABLED=1 but atelier.predictions not "
            "importable (%s) — skipping",
            e,
        )
        return

    try:
        parsed = predictions.parse_predicted_impact(review_text)
        path = predictions.save_predictions(
            reports_root=ctx.reports_root,
            campaign=ctx.campaign,
            child_node_id=ctx.child_node_id,
            predicted=parsed,
            iteration=iteration,
        )
    except Exception as e:  # noqa: BLE001 — hook must not crash pipeline
        logger.warning(
            "capture_review_prediction[%s] failed: %s", ctx.child_node_id, e
        )
        return

    if parsed.is_empty:
        logger.info(
            "atelier-predictions[%s] proposer emitted EMPTY prediction (L5 "
            "penalty applies) → %s",
            ctx.child_node_id,
            path,
        )
    else:
        logger.info(
            "atelier-predictions[%s] captured: should_pass=%d, at_risk=%d, "
            "root_cause=%dch → %s",
            ctx.child_node_id,
            len(parsed.should_pass),
            len(parsed.at_risk),
            len(parsed.root_cause),
            path,
        )


def compute_credibility(
    *,
    ctx: HookContext,
    parent_solved: set[str] | list[str] | tuple[str, ...],
    child_solved: set[str] | list[str] | tuple[str, ...],
) -> None:
    """Compare the child's predicted_impact against the next-iter actual
    flips and persist a ``<child>.credibility.json`` sidecar.

    Called from ``_finalize`` after final-eval rewards are recorded.
    Looks up the child's saved prediction; if absent, records an
    empty-prediction credibility (penalty in L5). Never raises.

    ``parent_solved`` is the set of tasks the parent node solved in
    its OWN final-eval (so this is a per-node sequential
    comparison: parent_node → child_node).
    """
    if not is_predictions_enabled():
        return

    try:
        from gate import predictions
    except ImportError as e:
        logger.warning(
            "compute_credibility[%s]: atelier.predictions not importable "
            "(%s) — skipping",
            ctx.child_node_id,
            e,
        )
        return

    if not ctx.parent_node_id:
        # Root node — no parent to compare against.
        return

    try:
        # 1) Latest prediction (backward-compat — what the gate uses today).
        predicted = predictions.load_predictions(
            reports_root=ctx.reports_root,
            campaign=ctx.campaign,
            child_node_id=ctx.child_node_id,
        )
        if predicted is None:
            predicted = predictions.PredictedImpact()

        credibility = predictions.compare_predictions_with_actual(
            parent_node_id=ctx.parent_node_id,
            child_node_id=ctx.child_node_id,
            predicted=predicted,
            parent_solved=parent_solved,
            child_solved=child_solved,
        )
        path = predictions.save_credibility(
            reports_root=ctx.reports_root,
            campaign=ctx.campaign,
            child_node_id=ctx.child_node_id,
            credibility=credibility,
        )

        # 2) Best-of-iters prediction (the proposer's max-honesty moment).
        # Often the iter-1 prediction is the most accurate one and gets
        # degraded by later iters' overwrites. We surface this for the
        # report tool — it doesn't (yet) feed back into the gate so the
        # latest-prediction discipline still applies.
        best = predictions.load_best_prediction(
            reports_root=ctx.reports_root,
            campaign=ctx.campaign,
            child_node_id=ctx.child_node_id,
            parent_solved=parent_solved,
            child_solved=child_solved,
        )
        if best is not None:
            best_iter, _best_pred, best_cred = best
            logger.info(
                "atelier-credibility[%s] best-of-iters: iter=%d jaccard=%.2f "
                "(vs latest=%.2f)",
                ctx.child_node_id, best_iter,
                best_cred.jaccard_accuracy,
                credibility.jaccard_accuracy,
            )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "compute_credibility[%s] failed: %s", ctx.child_node_id, e
        )
        return

    logger.info(
        "atelier-credibility[%s] parent=%s jaccard=%.2f over=%.2f under=%.2f "
        "regression_surprise=%.2f empty=%s → %s",
        ctx.child_node_id,
        ctx.parent_node_id,
        credibility.jaccard_accuracy,
        credibility.over_prediction_rate,
        credibility.under_prediction_rate,
        credibility.regression_surprise_rate,
        credibility.is_empty_prediction,
        path,
    )


# ─── Long-term memory (touchpoint D, AHE-style persistent wisdom) ────────


def is_trace_digest_enabled() -> bool:
    """Whether _render_analyze_prompt should include trace_analyzer
    digests for each failing trial.

    Default ``1`` — trace_analyzer is read-only and the digest is
    cheap. Opt out via ``DARWINX_GATE_TRACE_DIGEST_ENABLED=0``.
    """
    return os.environ.get("DARWINX_GATE_TRACE_DIGEST_ENABLED", "1").strip() == "1"


def render_trial_digests(
    *,
    trials: list[dict],
    verbose: bool = False,
) -> str:
    """Build a markdown section with one compact digest per trial.

    ``trials`` is the list of trial dicts that ``pipeline._trials_for``
    already constructs (each has a ``dir`` key). For each trial we
    load the digest via trace_analyzer and emit a compact summary
    (events, tool calls, pattern classification, final text, verifier
    excerpt) — the proposer should consume THIS before falling back
    to the raw transcript.

    Returns the empty string if disabled or no digests are available.
    Never raises.
    """
    if not is_trace_digest_enabled():
        return ""
    if not trials:
        return ""

    try:
        from gate import trace_analyzer
    except ImportError as e:
        logger.warning(
            "DARWINX_GATE_TRACE_DIGEST_ENABLED=1 but trace_analyzer not "
            "importable (%s) — skipping",
            e,
        )
        return ""

    digests: list[str] = []
    for i, trial in enumerate(trials, start=1):
        trial_dir_str = trial.get("dir")
        if not trial_dir_str:
            continue
        trial_dir = Path(trial_dir_str)
        try:
            digest = trace_analyzer.load_trial(trial_dir)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "trace_analyzer.load_trial(%s) failed: %s", trial_dir, e
            )
            continue
        if digest is None:
            continue

        # Compact rendering (use to_markdown() for full).
        verdict = "PASSED" if digest.passed else "FAILED"
        ft = (digest.final_text or "").strip().splitlines()
        # Keep first + last line of final text — enough to spot
        # "Done." vs an actual answer. Full text is in transcript.
        ft_snip = ft[0] if ft else "(empty)"
        if len(ft) > 1:
            ft_snip = ft_snip + "  …  " + ft[-1]
        ft_snip = ft_snip[:240]

        ve_lines = (digest.verifier_excerpt or "").strip().splitlines()
        ve_snip = "\n".join(ve_lines[:8])[:500]

        digests.append(
            "\n".join([
                f"### Trial {i}: `{digest.task}` — {verdict} (reward={digest.reward:.2f})",
                "",
                f"- pattern: `{digest.pattern.primary}`"
                + (f" — {digest.pattern.indicators[0]}" if digest.pattern.indicators else ""),
                f"- events: {digest.n_events}, turns: {digest.n_turns}, "
                f"tool_calls: {digest.tool_call_counts}, "
                f"tool_errors: {digest.tool_error_count}",
                f"- final_text: _{ft_snip}_",
                f"- verifier_output (first 500 chars):",
                "  ```",
                "  " + (ve_snip.replace("\n", "\n  ") or "(empty)"),
                "  ```",
                f"- raw transcript (fallback): `{trial.get('transcript', '?')}`",
                "",
            ])
        )

    if not digests:
        return ""

    return "\n".join([
        "## Per-trial structured digests (Atelier trace_analyzer)",
        "",
        "Read these BEFORE the raw transcripts. Each digest distills",
        "the trial's tool-call shape, failure-mode classification, and",
        "the verifier's complaint into a compact summary. Refer back",
        "to the raw transcript path only if the digest is unclear.",
        "",
        "",
        *digests,
    ])


def render_long_term_memory(
    *, reports_root: Path, campaign: str, max_entries: int | None = None
) -> str:
    """Return a markdown LongTermMemory section ready to inline into
    the analyze prompt. Empty string if LTM is disabled / no file.

    Called from ``pipeline._render_analyze_prompt`` so the proposer
    sees accumulated wisdom on every iteration.
    """
    if not is_ltm_enabled():
        return ""
    try:
        from gate import long_term_memory
    except ImportError as e:
        logger.warning(
            "DARWINX_GATE_LTM_ENABLED=1 but atelier.long_term_memory not "
            "importable (%s) — skipping",
            e,
        )
        return ""
    try:
        return long_term_memory.render_memory_for_prompt(
            reports_root=reports_root,
            campaign=campaign,
            max_entries=max_entries or ltm_max_entries(),
        )
    except Exception as e:  # noqa: BLE001 — hook must not crash pipeline
        logger.warning(
            "atelier-LTM render failed (%s) — proceeding without LTM", e
        )
        return ""


def mark_ltm_source_rejected(
    *, ctx: HookContext, reason: str = "",
) -> bool:
    """Filter out the child node's LTM contributions from future
    proposers' prompts when the equivalence gate (or any downstream
    consumer) rejects the candidate. See
    ``atelier.long_term_memory.mark_source_rejected``.

    Returns True iff the node id was newly added to the rejected set.
    Never raises.
    """
    if not is_ltm_enabled():
        return False
    try:
        from gate import long_term_memory
    except ImportError as e:
        logger.warning(
            "DARWINX_GATE_LTM_ENABLED=1 but long_term_memory not "
            "importable (%s) — skipping rejection mark", e,
        )
        return False
    try:
        return long_term_memory.mark_source_rejected(
            reports_root=ctx.reports_root,
            campaign=ctx.campaign,
            node_id=ctx.child_node_id,
            reason=reason,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "mark_ltm_source_rejected[%s] failed: %s", ctx.child_node_id, e,
        )
        return False


def persist_review_learnings(
    *,
    ctx: HookContext,
    review_text: str,
) -> int:
    """Parse the proposer's review text for a ``learnings_to_persist:``
    block and append new lessons to the campaign's LongTermMemory file.

    Returns the number of NEW entries appended (0 on dedupe or
    missing block). Never raises.
    """
    if not is_ltm_enabled():
        return 0
    if not review_text:
        return 0

    try:
        from gate import long_term_memory
    except ImportError as e:
        logger.warning(
            "DARWINX_GATE_LTM_ENABLED=1 but atelier.long_term_memory not "
            "importable (%s) — skipping",
            e,
        )
        return 0

    try:
        learnings = long_term_memory.parse_learnings_to_persist(review_text)
        if not learnings:
            return 0
        n = long_term_memory.append_learnings(
            reports_root=ctx.reports_root,
            campaign=ctx.campaign,
            source_node=ctx.child_node_id,
            learnings=learnings,
            max_entries=ltm_max_entries(),
        )
        return n
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "atelier-LTM persist failed (%s) for node %s", e, ctx.child_node_id
        )
        return 0


# ─── GEA patch archive (v7) ──────────────────────────────────────────────


def persist_patch_record(
    *,
    ctx: HookContext,
    parent_commit: str | None,
    modified_files: list[str] | tuple[str, ...],
    lines_added: int = 0,
    lines_removed: int = 0,
    verdict_modified_surfaces: tuple[str, ...] = (),
    score_delta: float | None = None,
) -> Path | None:
    """Write a structured patch record so the GEA sibling-pool can
    cite concrete diffs (v7) and v8 can implement explicit cross-
    branch patch composition.

    Sidecar layout:
        reports/<campaign>/atelier/patches/<node>.patch.json
        {
          "node_id": "...",
          "parent_node_id": "...",
          "parent_commit": "...",
          "child_commit": "...",
          "files_changed": [...],
          "lines_added": int,
          "lines_removed": int,
          "verdict_modified_surfaces": [...],
          "score_delta": float | null,
        }

    Returns the sidecar path on success, None when path can't be
    written (logs + swallows). Never raises.
    """
    try:
        out_dir = Path(ctx.reports_root) / ctx.campaign / "atelier" / "patches"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ctx.child_node_id}.patch.json"
        record = {
            "schema_version": 1,
            "node_id": ctx.child_node_id,
            "parent_node_id": ctx.parent_node_id,
            "parent_commit": parent_commit,
            "child_commit": ctx.child_commit,
            "files_changed": list(modified_files),
            "lines_added": int(lines_added),
            "lines_removed": int(lines_removed),
            "verdict_modified_surfaces": list(verdict_modified_surfaces),
            "score_delta": score_delta,
        }
        out_path.write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8",
        )
        logger.info(
            "atelier-patches[%s] recorded %d file(s), Δ=%s → %s",
            ctx.child_node_id, len(modified_files),
            f"{score_delta:+.3f}" if score_delta is not None else "—",
            out_path,
        )
        return out_path
    except OSError as e:
        logger.warning(
            "persist_patch_record[%s] failed: %s", ctx.child_node_id, e,
        )
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "persist_patch_record[%s] raised (ignored): %s",
            ctx.child_node_id, e,
        )
        return None


# ─── GEA sibling-pool integration (v7) ───────────────────────────────────


def render_sibling_evidence(
    *,
    conn,
    campaign: str,
    subset: str,
    focus_node,
    reports_root: Path,
    iteration: int,
    pipeline_id: str,
) -> str:
    """Render the GEA sibling-evidence markdown block for the analyze
    prompt. Returns empty string when:
    - sibling-pool is disabled (env opt-out)
    - K <= 1 (no siblings requested)
    - archive too small (< 2 scored siblings available)
    - any sidecar / DB read raises (graceful degradation)

    Called from ``pipeline._render_analyze_prompt`` on every iter; on
    iter 1 the proposer sees siblings, on iter 2+ they get the same
    pool (we don't re-pick siblings mid-iteration to avoid trace
    instability).
    """
    if not is_sibling_pool_enabled():
        return ""
    k = sibling_pool_k()
    if k <= 1:
        return ""

    try:
        from gate import sibling_pool, novelty as nov_mod
        from . import tree
    except ImportError as e:
        logger.warning(
            "DARWINX_GATE_SIBLING_POOL_ENABLED=1 but sibling_pool not "
            "importable (%s) — skipping", e,
        )
        return ""

    try:
        all_nodes = tree.list_nodes(conn, campaign=campaign, subset=subset)
    except Exception as e:  # noqa: BLE001
        logger.warning("sibling_pool: tree.list_nodes failed: %s", e)
        return ""

    eligible = [
        n for n in all_nodes
        if n.status in {"completed", "no_change"} and n.score is not None
    ]
    if len(eligible) < 2:
        return ""

    # Rank by PN score to seed the candidate list.
    try:
        tvs = nov_mod.task_vectors_from_solved_lists(
            solved_by_node={
                n.id: (n.solved_tasks or ()) for n in eligible
            },
        )
        ranked = nov_mod.rank_by_pn(
            nodes=tvs,
            scores={n.id: n.score for n in eligible},
            m=sibling_pool_novelty_m(),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("sibling_pool: rank_by_pn failed: %s", e)
        return ""

    parent_of = {n.id: n.parent_id for n in all_nodes}
    sibling_ids = sibling_pool.select_siblings(
        focus_node_id=focus_node.id,
        candidates=[(nid, pn) for nid, pn, _ in ranked],
        parent_of=parent_of,
        k=k,
        exclude_lineage=True,
    )
    if not sibling_ids:
        return ""

    by_id = {n.id: n for n in all_nodes}
    cards: list = []
    for sid in sibling_ids:
        s = by_id.get(sid)
        if s is None:
            continue
        try:
            card = sibling_pool.build_sibling_card(
                sibling_node_id=s.id,
                sibling_score=s.score or 0.0,
                sibling_parent_id=s.parent_id,
                sibling_improved_tasks=s.improved_tasks or (),
                sibling_regressed_tasks=s.regressed_tasks or (),
                reports_root=reports_root,
                campaign=campaign,
            )
            cards.append(card)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "sibling_pool: build_sibling_card[%s] failed: %s",
                s.id, e,
            )
            continue

    if not cards:
        return ""

    # Log + emit a sidecar so we can audit which siblings were picked.
    try:
        sidecar_dir = (
            Path(reports_root) / campaign / "atelier" / "sibling_pool"
        )
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "pipeline_id": pipeline_id,
            "iteration": iteration,
            "focus_node_id": focus_node.id,
            "focus_score": focus_node.score,
            "sibling_ids": [c.node_id for c in cards],
            "sibling_scores": [c.score for c in cards],
            "k": k,
        }
        import json as _json
        (sidecar_dir / f"{pipeline_id}_iter{iteration}.json").write_text(
            _json.dumps(rec, indent=2, default=str), encoding="utf-8",
        )
    except OSError as e:
        logger.warning("sibling_pool: could not write sidecar: %s", e)

    logger.info(
        "sibling_pool[pipeline=%s iter=%d]: focus=%s, siblings=%s",
        pipeline_id, iteration, focus_node.id,
        [c.node_id for c in cards],
    )

    return sibling_pool.render_sibling_evidence(
        focus_node_id=focus_node.id,
        focus_score=focus_node.score,
        siblings=cards,
    )


__all__ = [
    "HookContext",
    "EquivalenceGateContext",
    "is_gate_enabled",
    "is_fitness_enabled",
    "is_predictions_enabled",
    "is_ltm_enabled",
    "is_equivalence_gate_enabled",
    "equivalence_probe_k",
    "equivalence_probe_k_fixed",
    "equivalence_n_adversarial",
    "equivalence_n_votes",
    "equivalence_require_extension",
    "equivalence_model",
    "equivalence_provider",
    "fitness_alpha",
    "maybe_blend_score",
    "on_node_finalized",
    "run_equivalence_gate",
    "capture_review_prediction",
    "compute_credibility",
    "render_long_term_memory",
    "persist_review_learnings",
    "mark_ltm_source_rejected",
    "is_trace_digest_enabled",
    "render_trial_digests",
    "is_sibling_pool_enabled",
    "sibling_pool_k",
    "sibling_pool_novelty_m",
    "render_sibling_evidence",
    "persist_patch_record",
]

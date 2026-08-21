"""Evidence-based failure digest for the analyze step.

Turns the claimed-task trajectories of an iteration into a structured QC block
(via ``trace_analyzer``) that the cursor meta-agent reads instead of cold-reading
raw transcripts. Deterministic and offline (rule proposers) — zero API cost, safe
to run every iteration. Best-effort throughout: any problem yields an empty block
so the analyze prompt simply omits the section rather than failing.

See ``trace_analyzer/docs/evolve_integration.md`` (Phase 1).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("self_evolve.trace_qc")

_SUFFIX = ".trajectory.jsonl"


def _jsonl_for_trial(trial: dict[str, Any]) -> Path | None:
    """The monet stream-json sidecar for a trial, if retained.

    Monet writes ``trajectory.jsonl`` (stream-json) beside the ATIF
    ``trajectory.json`` the prompt otherwise references; trace_analyzer reads
    the former. Falls back to ``<dir>/agent/trajectory.jsonl``.
    """
    traj_json = trial.get("trajectory")
    if traj_json:
        cand = Path(traj_json).with_name("trajectory.jsonl")
        if cand.is_file():
            return cand
    d = trial.get("dir")
    if d:
        cand = Path(d) / "agent" / "trajectory.jsonl"
        if cand.is_file():
            return cand
    return None


def _find_in_eval_dir(eval_dir: Path, task: str) -> Path | None:
    if not task or not eval_dir.exists():
        return None
    try:
        hits = list(eval_dir.rglob(f"{task}{_SUFFIX}"))
    except OSError:
        return None
    return hits[0] if hits else None


def analyze_digest_block(
    trials: list[dict[str, Any]],
    *,
    eval_dir: str | Path | None = None,
    config: str = "default",
) -> str:
    """Render a markdown QC block for the analyze prompt, or ``""``.

    ``trials`` is the pipeline's per-trial list (each has ``task``/``dir``/
    ``trajectory``). ``eval_dir`` is an optional root to search when a trial's
    sidecar isn't beside its ATIF file.
    """
    try:
        from trace_analyzer.digest import digest_paths
    except Exception as exc:  # trace_analyzer not importable — degrade silently
        log.debug("trace_analyzer unavailable: %s", exc)
        return ""

    eval_root = Path(eval_dir) if eval_dir else None
    items: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for trial in trials:
        task = str(trial.get("task") or trial.get("name") or "")
        path = _jsonl_for_trial(trial)
        if path is None and eval_root is not None:
            path = _find_in_eval_dir(eval_root, task)
        if path is not None and str(path) not in seen:
            seen.add(str(path))
            items.append((task or path.name[: -len(_SUFFIX)], path))

    if not items:
        log.info("trace_qc: no stream-json trajectories found for %d trials", len(trials))
        return ""

    # Q3: the digest cites issues by `#message_index` into the *normalized*
    # turn-level view — which the agent can't navigate in the raw SSE stream.
    # Write each one's <task>.messages.jsonl beside the raw trajectory so the
    # cited indices are openable, and point the agent at them.
    msg_paths: list[tuple[str, Path]] = []
    for task, path in items:
        mp = _write_messages_sidecar(path)
        if mp is not None:
            msg_paths.append((task, mp))

    # Phase 2 (env-gated): add the SEMANTIC LLM proposers (instruction_not_followed,
    # premature/wrong-but-clean, ...) — the dense fault-localization gradient our hard
    # tail (wrong-but-confident failures) needs. Phase-1 rule proposers alone find
    # nothing on semantic failures. SELF_EVOLVE_TRACE_QC_LLM=1 turns it on; the config
    # defaults to the monet-tuned profile (rubrics for overconfident-stopping / shallow
    # verification / thrash); the LLM client auto-resolves the campaign's Express
    # gateway. Best-effort: missing creds or any error degrades to rule-only.
    llm_client = None
    qc_config = config
    if os.environ.get("SELF_EVOLVE_TRACE_QC_LLM", "0").strip().lower() in ("1", "true", "yes", "on"):
        qc_config = os.environ.get("SELF_EVOLVE_TRACE_QC_CONFIG", "monet")
        try:
            from trace_analyzer.llm import OpenAIClient

            llm_client = OpenAIClient()
        except Exception as exc:  # no creds / unavailable — degrade to rule-only
            log.warning("trace_qc: LLM proposers requested but client unavailable (%s); rule-only", exc)
            llm_client = None
    try:
        block = digest_paths(items, config=qc_config, llm=llm_client).render_markdown()
    except Exception as exc:  # never let analyze fail on the digest
        log.warning("trace_qc digest failed: %s", exc)
        return ""

    if msg_paths:
        lines = [
            "",
            "Each issue's `#N` indexes the **normalized turn-level view** written "
            "beside the raw trajectory — open these to drill into a finding "
            "(the raw SSE stream is the `.trajectory.jsonl` next to each):",
        ]
        lines += [f"- `{task}`: `{mp}`" for task, mp in msg_paths]
        block += "\n" + "\n".join(lines) + "\n"

    # OPTIONAL contrastive teacher supervision (env-gated; cache-only no-op by default).
    # Strict no-op unless ATELIER_TEACHER_SUPERVISION=1 AND a cached teacher trace
    # exists; wrapped so any failure degrades to the failure-only baseline.
    try:
        from .teacher_supervision import maybe_teacher_block
        for _t, _ in items:
            block += maybe_teacher_block(_t, trials, eval_dir)
    except Exception as _e:
        log.debug("teacher supervision hook skipped: %s", _e)
    # OPTIONAL GRPO-style self-rollout contrast (env-gated ATELIER_SELF_CONTRAST=1;
    # cache-only no-op by default). For VARIANCE-band tasks it contrasts monet's own
    # passing vs failing rollout (group-relative advantage). Complements the teacher
    # (which covers 0/k walls). Any failure degrades to the failure-only baseline.
    try:
        from .self_contrast import maybe_self_contrast_block
        for _t, _ in items:
            block += maybe_self_contrast_block(_t, trials, eval_dir)
    except Exception as _e:
        log.debug("self-contrast hook skipped: %s", _e)
    return block


def _write_messages_sidecar(jsonl_path: Path) -> Path | None:
    """Write the normalized ``<task>.messages.jsonl`` beside the raw trajectory.

    Best-effort: returns the path on success, ``None`` on any failure (so a
    read-only dir or a parse error can't break analyze).
    """
    try:
        import json

        from trace_analyzer.normalizer import load

        traj = load(jsonl_path)
        out = Path(jsonl_path).with_name(f"{traj.trace_id}.messages.jsonl")
        out.write_text(
            "\n".join(json.dumps(m, ensure_ascii=False) for m in traj.messages()) + "\n",
            encoding="utf-8",
        )
        return out
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("trace_qc: messages sidecar failed for %s: %s", jsonl_path, exc)
        return None

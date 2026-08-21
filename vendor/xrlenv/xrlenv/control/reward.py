"""In-sandbox reward computation (spec 02 RewardContract, ``in_sandbox_final``).

Runs each grader in the contract's ``effective_graders()`` via
``NodeTransport.run_in_sandbox``, parses the per-grader score per its
``output_format``, and aggregates them into a single ``final_reward``
per the contract's ``aggregator``. Per-grader scores survive in the
returned :class:`RewardComputation` so the coordinator can stash them in
``trajectory.metadata.rewards`` for the consumer / admin to inspect.

Output formats handled:

- ``exit_code``      — reward = ``1.0 if exit_code == 0 else 0.0``
- ``stdout_float``   — last non-empty stdout line parsed as ``float``
- ``json_stdout``    — ``json.loads(stdout)[score_key]``
- ``json_file``      — fetch ``output_path`` from inside the sandbox (via
                        ``run_in_sandbox(["cat", path])``), parse JSON,
                        extract ``score_key``

``on_error`` (``fail_rollout`` | ``zero_reward`` | ``partial``) determines
what happens when a grader's command fails (non-zero exit, timeout, or
parse error). Aggregation only sees graders that produced a numeric
result — failures either propagate, contribute 0, or are skipped per the
contract's mode.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from xrlenv.backends.base import ExecResult, SandboxHandle
from xrlenv.control.node_transport import NodeTransport
from xrlenv.control.template_catalog import (
    GraderSpec,
    RewardAggregator,
    RewardContract,
)

if TYPE_CHECKING:
    from xrlenv.control.instance_resolver import VerifierUpload

LOGGER = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result shapes
# ──────────────────────────────────────────────────────────────────────────────


#: Cap on the per-grader stdout/stderr we ship into trajectory
#: metadata. Trajectories are read by the admin viewer + ``xrlenv
#: replay``; an unbounded grader (e.g. one that prints a 50 MB pytest
#: log) would balloon the JSONL bodies. 16 KiB is enough to capture
#: typical wrapper output (``test.log``, harbor verifier traces) and
#: the last error trail without bloating storage.
GRADER_OUTPUT_BYTES_CAP: int = 16 * 1024


@dataclass(frozen=True, slots=True)
class GraderResult:
    """One grader's outcome — score on success, error on failure."""

    name: str
    score: float | None
    weight: float
    error: str | None
    """``None`` on success; otherwise a short label like ``exit_5``,
    ``timeout``, ``parse_error``, ``missing_score_key``."""
    stdout: str | None = None
    """Last :data:`GRADER_OUTPUT_BYTES_CAP` bytes of the grader command's
    stdout, decoded as UTF-8 with replacement. Surfaced in
    ``trajectory.metadata.rewards[<name>].stdout`` so a "score=0.0
    with no error" rollout can be diagnosed without a manual
    ``docker exec`` reproduction."""
    stderr: str | None = None
    """Last :data:`GRADER_OUTPUT_BYTES_CAP` bytes of the grader command's
    stderr, decoded UTF-8 + replacement. Especially useful for
    ``stdout_float`` graders whose stdout is just the score; the
    diagnostic content (``test.log``, error tracebacks) lands here."""


@dataclass(frozen=True, slots=True)
class RewardComputation:
    """Result of evaluating a contract against a sandbox.

    ``final_reward`` is the aggregator output. ``per_grader`` carries every
    grader's score (or error) so the coordinator can write
    ``trajectory.metadata.rewards = {name: score}`` for inspection. ``failed``
    is True if any grader failed AND ``on_error`` would block sealing
    (``fail_rollout``); the coordinator then routes through the failed-rollout
    path instead of sealing finished.
    """

    final_reward: float
    per_grader: tuple[GraderResult, ...]
    failed: bool
    """True only when ``on_error == "fail_rollout"`` and at least one grader
    errored. ``zero_reward`` and ``partial`` modes never set this."""
    error_message: str | None = None


class RewardComputationError(Exception):
    """Raised when ``on_error == fail_rollout`` and any grader fails.

    Carries the partial :class:`RewardComputation` so the coordinator can
    surface the per-grader scores even on failure.
    """

    def __init__(self, message: str, computation: RewardComputation) -> None:
        super().__init__(message)
        self.computation = computation


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


async def compute_in_sandbox_final_reward(
    *,
    node: NodeTransport,
    sandbox: SandboxHandle,
    contract: RewardContract,
    verifier_uploads: tuple[VerifierUpload, ...] = (),
) -> RewardComputation:
    """Run every grader in ``contract`` and return the aggregated reward.

    Raises :class:`RewardComputationError` if ``on_error == fail_rollout`` and
    any grader fails. Otherwise the failure mode is folded into the
    aggregation per ``zero_reward`` / ``partial`` semantics and the call
    returns normally.

    ``verifier_uploads`` (D12 stage 1) — when non-empty, each entry's
    tarball is extracted into its ``target_dir`` inside the sandbox
    immediately before the first grader runs. This is the
    timing-isolation primitive that closes audit H1: the grader files
    do not exist in the sandbox during the agent's step() loop, so the
    agent cannot read or modify them.
    """
    graders = contract.effective_graders()
    if not graders:
        # Defensive — RewardContract validation already forbids this for
        # mode=in_sandbox_final, but keep the guard so a programmatic
        # caller can't surprise us.
        return RewardComputation(final_reward=0.0, per_grader=(), failed=False)

    # D12 stage 1: inject grader assets BEFORE running graders. Each
    # upload is dispatched with ``clean_target=True`` so the backend
    # runs ``rm -rf <target>`` as root via ``docker exec --user root``
    # before extracting — agent-created residue cannot survive the
    # wipe even when the in-sandbox stub runs as a non-root user
    # (audit H1 follow-up: the wipe must be backend/root-backed and
    # fail closed on non-zero exit, not best-effort via the stub).
    for upload in verifier_uploads:
        await node.put_archive(
            sandbox, upload.target_dir, upload.tarball,
            clean_target=True,
        )

    results: list[GraderResult] = []
    for grader in graders:
        timeout_s = grader.timeout_s if grader.timeout_s is not None else contract.timeout_s
        result = await _run_grader_internal(
            node=node,
            sandbox=sandbox,
            grader=grader,
            timeout_s=timeout_s,
        )
        results.append(result)

    return _aggregate(tuple(results), contract)


# ──────────────────────────────────────────────────────────────────────────────
# Per-grader execution + parsing
# ──────────────────────────────────────────────────────────────────────────────


def _parse_exec_result(
    exec_result: ExecResult, grader: GraderSpec
) -> GraderResult:
    """Parse exit_code / stdout_float / json_stdout formats inline.

    json_file is handled separately in :func:`_run_grader_internal` because
    it needs an extra in-sandbox call to fetch the file body.
    """
    fmt = grader.output_format
    if fmt == "exit_code":
        score = 1.0 if exec_result.exit_code == 0 else 0.0
        return GraderResult(name=grader.name, score=score, weight=grader.weight, error=None)

    if exec_result.exit_code != 0:
        # Non-zero exit on stdout-bearing formats is a grader error; the
        # spec puts this under on_error semantics rather than letting the
        # parser try to extract a score from a failed run.
        return GraderResult(
            name=grader.name,
            score=None,
            weight=grader.weight,
            error=f"exit_{exec_result.exit_code}",
        )

    if fmt == "stdout_float":
        return _parse_stdout_float(exec_result.stdout, grader)
    if fmt == "json_stdout":
        return _parse_json_payload(exec_result.stdout, grader, source="stdout")
    # json_file is handled in _run_grader_internal; reaching here means a
    # caller misused the parser.
    return GraderResult(
        name=grader.name, score=None, weight=grader.weight, error=f"unhandled_format:{fmt}"
    )


def _parse_stdout_float(stdout: bytes, grader: GraderSpec) -> GraderResult:
    text = stdout.decode("utf-8", errors="replace").strip()
    if not text:
        return GraderResult(
            name=grader.name, score=None, weight=grader.weight, error="empty_stdout"
        )
    last = text.splitlines()[-1].strip()
    try:
        score = float(last)
    except ValueError:
        return GraderResult(
            name=grader.name, score=None, weight=grader.weight, error="parse_error"
        )
    return GraderResult(name=grader.name, score=score, weight=grader.weight, error=None)


def _parse_json_payload(
    blob: bytes, grader: GraderSpec, *, source: str
) -> GraderResult:
    text = blob.decode("utf-8", errors="replace").strip()
    if not text:
        return GraderResult(
            name=grader.name, score=None, weight=grader.weight, error=f"empty_{source}"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return GraderResult(
            name=grader.name, score=None, weight=grader.weight, error=f"json_decode_error:{source}"
        )
    if not isinstance(payload, dict):
        return GraderResult(
            name=grader.name,
            score=None,
            weight=grader.weight,
            error=f"json_not_object:{source}",
        )
    if grader.score_key not in payload:
        return GraderResult(
            name=grader.name,
            score=None,
            weight=grader.weight,
            error="missing_score_key",
        )
    raw = payload[grader.score_key]
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return GraderResult(
            name=grader.name,
            score=None,
            weight=grader.weight,
            error="score_not_float",
        )
    return GraderResult(name=grader.name, score=score, weight=grader.weight, error=None)


# ──────────────────────────────────────────────────────────────────────────────
# json_file path needs an extra in-sandbox call — keep _run_grader async
# coherent by handling it inline.
# ──────────────────────────────────────────────────────────────────────────────


def _capture_grader_output(payload: bytes) -> str | None:
    """Decode + tail-truncate a grader's stdout/stderr for surfacing in
    ``trajectory.metadata.rewards[<name>].{stdout,stderr}``.

    Returns ``None`` for empty payloads so the JSON shape stays small
    when the grader didn't emit anything. Truncates from the front
    (keeping the trailing :data:`GRADER_OUTPUT_BYTES_CAP` bytes)
    because the tail of a long log usually carries the actual error
    message; the head is typically setup chatter.
    """
    if not payload:
        return None
    if len(payload) > GRADER_OUTPUT_BYTES_CAP:
        payload = payload[-GRADER_OUTPUT_BYTES_CAP:]
    return payload.decode("utf-8", errors="replace")


async def _run_grader_internal(
    *,
    node: NodeTransport,
    sandbox: SandboxHandle,
    grader: GraderSpec,
    timeout_s: float,
) -> GraderResult:
    """Run one grader and parse its result.

    For ``output_format=json_file`` we follow the grader's command with a
    ``run_in_sandbox(["cat", output_path])`` to fetch the score file; the
    coordinator never adds a dedicated FetchFile proto command for this
    in phase 0 (a single extra exec call is cheaper than the proto +
    converter surface).

    Every successful or score-bearing return path attaches the
    grader's captured stdout/stderr (front-truncated) so a
    "score=0.0 with no error" rollout can be diagnosed without a
    manual ``docker exec`` reproduction.
    """
    try:
        exec_result = await node.run_in_sandbox(
            sandbox, list(grader.cmd), timeout_s=timeout_s
        )
    except Exception as exc:
        LOGGER.warning(
            "grader=%s transport-level failure: %s", grader.name, exc
        )
        # No exec_result available → no stdout/stderr to attach.
        return GraderResult(
            name=grader.name,
            score=None,
            weight=grader.weight,
            error=f"transport_error:{type(exc).__name__}",
        )

    captured_stdout = _capture_grader_output(exec_result.stdout)
    captured_stderr = _capture_grader_output(exec_result.stderr)

    def _attach(result: GraderResult) -> GraderResult:
        return dataclasses.replace(
            result, stdout=captured_stdout, stderr=captured_stderr,
        )

    if exec_result.timed_out:
        return _attach(GraderResult(
            name=grader.name, score=None, weight=grader.weight, error="timeout",
        ))
    if exec_result.exit_code != 0 and grader.output_format != "exit_code":
        return _attach(GraderResult(
            name=grader.name,
            score=None,
            weight=grader.weight,
            error=f"exit_{exec_result.exit_code}",
        ))

    if grader.output_format == "json_file":
        assert grader.output_path is not None  # validated at manifest load
        try:
            cat_result = await node.run_in_sandbox(
                sandbox, ["cat", grader.output_path], timeout_s=timeout_s
            )
        except Exception as exc:
            return _attach(GraderResult(
                name=grader.name,
                score=None,
                weight=grader.weight,
                error=f"file_fetch_error:{type(exc).__name__}",
            ))
        if cat_result.exit_code != 0 or cat_result.timed_out:
            return _attach(GraderResult(
                name=grader.name,
                score=None,
                weight=grader.weight,
                error=f"file_missing:{grader.output_path}",
            ))
        return _attach(_parse_json_payload(cat_result.stdout, grader, source="file"))

    return _attach(_parse_exec_result(exec_result, grader))


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation + on_error
# ──────────────────────────────────────────────────────────────────────────────


def _aggregate(
    results: tuple[GraderResult, ...],
    contract: RewardContract,
) -> RewardComputation:
    failures = [r for r in results if r.error is not None]
    successes = [r for r in results if r.error is None and r.score is not None]

    if failures:
        if contract.on_error == "fail_rollout":
            comp = RewardComputation(
                final_reward=0.0,
                per_grader=results,
                failed=True,
                error_message=_summarize_failures(failures),
            )
            raise RewardComputationError(comp.error_message or "grader failed", comp)
        if contract.on_error == "zero_reward":
            # Treat failed graders as score=0; carry on with aggregation.
            patched = tuple(
                GraderResult(
                    name=r.name,
                    score=0.0 if r.error is not None else r.score,
                    weight=r.weight,
                    error=r.error,
                )
                for r in results
            )
            return RewardComputation(
                final_reward=_apply_aggregator(
                    [(r.score or 0.0, r.weight) for r in patched],
                    contract.aggregator,
                ),
                per_grader=patched,
                failed=False,
            )
        # on_error == "partial" — drop failed graders, aggregate the rest.
        if not successes:
            # Nothing to aggregate; surface 0 with the failure context.
            return RewardComputation(
                final_reward=0.0,
                per_grader=results,
                failed=False,
                error_message=_summarize_failures(failures),
            )

    aggregated = _apply_aggregator(
        [(r.score or 0.0, r.weight) for r in successes],
        contract.aggregator,
    )
    return RewardComputation(
        final_reward=aggregated,
        per_grader=results,
        failed=False,
    )


def _apply_aggregator(
    scored: list[tuple[float, float]],
    aggregator: RewardAggregator,
) -> float:
    if not scored:
        return 0.0
    if aggregator == "first":
        return scored[0][0]
    scores = [s for s, _ in scored]
    if aggregator == "mean":
        return sum(scores) / len(scores)
    if aggregator == "sum":
        return sum(scores)
    if aggregator == "weighted_sum":
        return sum(s * w for s, w in scored)
    if aggregator == "max":
        return max(scores)
    if aggregator == "min":
        return min(scores)
    # Unreachable per Literal type.
    return 0.0


def _summarize_failures(failures: list[GraderResult]) -> str:
    return ", ".join(f"{r.name}={r.error}" for r in failures)


__all__ = [
    "GraderResult",
    "RewardComputation",
    "RewardComputationError",
    "compute_in_sandbox_final_reward",
]

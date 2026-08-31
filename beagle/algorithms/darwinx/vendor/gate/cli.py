"""Command-line interface for ``atelier``.

Subcommands:

- ``verify-campaign`` — walk a completed self_evolve campaign, run the
  Atelier gate over each viable node, emit a JSON report.

Usage::

    python -m atelier verify-campaign \\
        --campaign se_s10_260514_081752 \\
        --reports-root reports/ \\
        --repo-root .

Only layers whose runtime dependencies are present run "for real". Layers
that aren't wired (honeypot without a Terminal Wrench runner, verifier
without an OpenAI API key, transfer without harbor configs) are recorded
as ``SKIPPED`` or ``NOT_YET_IMPLEMENTED`` in the report — useful for the
initial "measure scope-filter violations across self_evolve's archive"
audit which only needs Layer 4.

This is intentionally a *thin* CLI. The actual layer wiring (Terminal
Wrench → Harbor, verifier → OpenAI, cross-model → Sonnet config) is
parameterized through `AtelierGate` construction; the CLI just walks the
campaign and invokes the gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import scope_filter
from .gate import AtelierGate, GateDecision, LayerStatus

from ._evolvee import agent_submodule


logger = logging.getLogger("atelier.cli")


# ─── Argparse ────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="atelier",
        description=(
            "Atelier — verification + test-time-scaling layer over "
            "self_evolve campaigns."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    vp = sub.add_parser(
        "verify-campaign",
        help="Run Atelier verification gates over every node in a campaign.",
    )
    vp.add_argument(
        "--campaign",
        required=True,
        help="Campaign identifier (matches self_evolve's --campaign).",
    )
    vp.add_argument(
        "--reports-root",
        type=Path,
        default=Path("reports"),
        help="Path to self_evolve reports root.",
    )
    vp.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Path to monet_code_eval repo root (for diff extraction).",
    )
    vp.add_argument(
        "--subset",
        default=None,
        help="Restrict to a specific subset (default: all subsets).",
    )
    vp.add_argument(
        "--scope-mode",
        choices=[m.value for m in scope_filter.ScopeMode],
        default=scope_filter.ScopeMode.SOFT_FLAG.value,
        help=(
            "Scope filter mode: 'soft_flag' (default; record but don't reject) "
            "or 'strict_reject'."
        ),
    )
    vp.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Default: "
            "<reports-root>/<campaign>/atelier_decisions.json."
        ),
    )
    vp.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: limit the number of nodes processed (for testing).",
    )
    vp.add_argument(
        "--include-failed",
        action="store_true",
        help=(
            "Also evaluate nodes with status='failed'. Default: only "
            "completed/no_change nodes (the same set self_evolve uses for "
            "parent selection)."
        ),
    )
    return p


# ─── Diff extraction ─────────────────────────────────────────────────────


def _get_node_diff(
    *,
    repo_root: Path,
    node_branch: str,
    node_commit: str | None,
    parent_commit: str | None,
) -> tuple[list[str], str | None]:
    """Return (changed_files, error).

    Runs ``git -C <agent-submodule> diff parent..commit --name-only`` from the
    repo root's monet_code submodule. Returns an empty list on error
    (and the error string in the second slot for the report).
    """
    submodule = agent_submodule()
    monet_dir = repo_root / submodule
    if not monet_dir.is_dir():
        return ([], f"{submodule} submodule missing at {monet_dir}")

    if not node_commit:
        return ([], "node has no commit_sha")

    if not parent_commit:
        # Root node — no diff to compare against (skip cleanly).
        return ([], "root node (no parent)")

    cmd = ["git", "-C", str(monet_dir), "diff", "--name-only", f"{parent_commit}..{node_commit}"]
    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=30
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return ([], f"git diff failed: {e}")

    if result.returncode != 0:
        return ([], f"git diff returncode={result.returncode}: {result.stderr.strip()}")

    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return (files, None)


# ─── Report ──────────────────────────────────────────────────────────────


def _decision_to_json(decision: GateDecision, *, extra: dict[str, Any]) -> dict:
    """Serialize a GateDecision into a JSON-safe dict.

    The verifier / honeypot / transfer payloads contain nested dataclasses
    that are JSON-encodable through asdict; for scope it's a ScopeDecision.
    """
    layers = []
    for layer in decision.layers:
        layer_dict = {
            "name": layer.name,
            "status": layer.status.value,
            "summary": layer.summary,
        }
        if layer.payload is not None:
            try:
                layer_dict["payload"] = asdict(layer.payload)
            except TypeError:
                # Payload isn't a dataclass — fall back to string repr.
                layer_dict["payload"] = repr(layer.payload)
        layers.append(layer_dict)

    return {
        "node_id": decision.node_id,
        "accept": decision.accept,
        "reject_reasons": decision.reject_reasons,
        "layers": layers,
        **extra,
    }


# ─── verify-campaign ─────────────────────────────────────────────────────


def cmd_verify_campaign(args: argparse.Namespace) -> int:
    """Walk a self_evolve campaign and apply the Atelier gate to each node."""

    # Lazy import to keep `atelier` usable without `self_evolve` machinery
    # always available (e.g., during unit tests of just the layers).
    try:
        from evolve import tree
    except ImportError as e:
        logger.error(
            "verify-campaign requires self_evolve to be importable: %s", e
        )
        return 2

    reports_root: Path = args.reports_root
    db_path = tree.db_path_for(reports_root, args.campaign)
    if not Path(db_path).is_file():
        logger.error("campaign DB missing: %s", db_path)
        return 2

    conn = tree.connect(db_path)

    if args.subset:
        nodes = tree.list_nodes(conn, campaign=args.campaign, subset=args.subset)
    else:
        # Aggregate across all subsets — go through all rows.
        nodes = tree.list_nodes(conn, campaign=args.campaign, subset=None)  # type: ignore[arg-type]

    if not args.include_failed:
        nodes = [n for n in nodes if n.status in ("completed", "no_change")]

    if args.limit is not None:
        nodes = nodes[: args.limit]

    if not nodes:
        logger.warning("no nodes matched filters; nothing to verify")
        return 0

    # Build a parent-lookup so we can compute diffs.
    nodes_by_id = {n.id: n for n in nodes}

    # Construct the gate. Week-2 scope: only the scope filter is wired
    # without external infrastructure. Other layers need configs the
    # caller plugs in later (see atelier/README.md).
    gate = AtelierGate(
        scope_mode=scope_filter.ScopeMode(args.scope_mode),
    )

    decisions: list[dict] = []
    for node in nodes:
        parent_commit: str | None = None
        if node.parent_id is not None:
            parent_node = nodes_by_id.get(node.parent_id)
            if parent_node is None:
                # Parent isn't in the filtered set; fetch directly.
                parent_node = tree.get_node(conn, node.parent_id)
            parent_commit = parent_node.commit_sha if parent_node else None

        diff_files, diff_err = _get_node_diff(
            repo_root=args.repo_root,
            node_branch=node.branch_name,
            node_commit=node.commit_sha,
            parent_commit=parent_commit,
        )

        decision = gate.evaluate(
            node_id=node.id,
            candidate_id=node.id,  # 1:1 mapping for self_evolve
            diff_files=diff_files,
        )

        record = _decision_to_json(
            decision,
            extra={
                "campaign": args.campaign,
                "subset": node.subset,
                "branch_name": node.branch_name,
                "commit_sha": node.commit_sha,
                "parent_id": node.parent_id,
                "parent_commit": parent_commit,
                "status": node.status,
                "score": node.score,
                "n_diff_files": len(diff_files),
                "diff_error": diff_err,
            },
        )
        decisions.append(record)
        logger.info("%s", decision.to_summary())

    # Aggregate stats for quick observation.
    n_accept = sum(1 for d in decisions if d["accept"])
    n_reject = len(decisions) - n_accept
    n_scope_violations = sum(
        1
        for d in decisions
        for layer in d["layers"]
        if layer["name"] == "scope" and layer["payload"]
        and len(layer["payload"].get("violations", [])) > 0
    )

    out_path = args.output or (
        reports_root / args.campaign / "atelier_decisions.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "campaign": args.campaign,
                "n_nodes": len(decisions),
                "n_accept": n_accept,
                "n_reject": n_reject,
                "n_scope_violations": n_scope_violations,
                "decisions": decisions,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info(
        "wrote %s — %d nodes (%d accept, %d reject, %d scope-violations)",
        out_path,
        len(decisions),
        n_accept,
        n_reject,
        n_scope_violations,
    )
    return 0


# ─── main ────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.command == "verify-campaign":
        return cmd_verify_campaign(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

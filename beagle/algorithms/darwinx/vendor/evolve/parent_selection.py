"""Parent-node selection strategies for the self-evolve campaign.

Each strategy picks one node from the campaign's tree to serve as the parent
that the new pipeline will branch off. Strategies are pluggable so different
exploration policies can be tried without touching the orchestrator.

Default strategy: MixedHighScore — for each spawning pipeline, flips a
deterministic-on-pipeline_id coin. Both halves rank primarily by CUMULATIVE
net task-gain from root (the compounding signal; see ``_rank_key``), with the
subset_final score as a tiebreak:

  - 50% **exploit**: pick the highest cumulative-net-gain eligible node (greedy
    on compounding improvement).
  - 50% **broaden**: pick by `(cum_net_gain DESC, score DESC, child_count ASC)`;
    prefers a same-tier node with fewer children when the top tier is over-fanned.

The 50/50 split trades exploitation for tree breadth: half of all spawned
workers chase the current best, the other half deliberately avoid bottle-
necking the densest branch. Both halves agree under typical conditions
(top node has few children), so the practical exploration cost is small.

Subset-aware: a node is only eligible if its `subset` matches the pipeline's
configured subset (apples-to-apples scores) AND its status is in
{completed, no_change} with a non-NULL score.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from abc import ABC, abstractmethod

from . import tree


class NoEligibleParentError(RuntimeError):
    """Raised when no node matches the strategy's eligibility criteria."""


class ParentSelectionStrategy(ABC):
    """Pick a parent node id from the campaign tree.

    Subclasses receive the open `state.db` connection plus the pipeline_id
    (used as a deterministic tiebreaker — same pipeline_id always picks the
    same parent given identical tree state).
    """

    name: str = "abstract"

    @abstractmethod
    def select(
        self,
        conn: sqlite3.Connection,
        *,
        campaign: str,
        subset: str,
        pipeline_id: str,
    ) -> tree.Node: ...


def _eligible_nodes(
    conn: sqlite3.Connection, *, campaign: str, subset: str,
) -> list[tree.Node]:
    """All nodes a strategy may legally pick from."""
    all_nodes = tree.list_nodes(conn, campaign=campaign, subset=subset)
    search_evals = tree.search_eval_by_node(conn, campaign=campaign, subset=subset)
    merge_children = {
        edge.child_id
        for edge in tree.list_node_edges(conn, campaign=campaign, edge_type="merge")
    }
    # 'archived' nodes ARE sampleable as parents (DGM preserve-everything: a
    # preserved stepping stone can seed a future win / merge) even though they're
    # excluded from best-node. This is what stops the search from "resetting to
    # baseline" — non-improving variants stay in the gene pool.
    eligible = [
        n for n in all_nodes
        if n.status in {"completed", "no_change", "archived"}
        and (n.id in search_evals or n.score is not None)
        and not (n.status in {"no_change", "archived"} and n.id in merge_children)
    ]
    if not eligible:
        raise NoEligibleParentError(
            f"no eligible parent nodes for campaign={campaign!r} subset={subset!r}; "
            f"bootstrap a root node first (with --baseline-logs or by running the "
            f"baseline)."
        )
    return eligible


def _net_gain(node: tree.Node) -> int:
    """Tasks this node FIXED minus tasks it BROKE, relative to its parent (the
    count-based extension-contract signal). Denominator-free and comparable
    across nodes — unlike an absolute solved-count, which is on each node's own
    (different) claimed subset."""
    def _c(raw: str | None) -> int:
        try:
            return len(json.loads(raw or "[]"))
        except (TypeError, ValueError):
            return 0
    return _c(node.improved_tasks_json) - _c(node.regressed_tasks_json)


def _cumulative_net_gain(node: tree.Node, by_id: dict[str, tree.Node]) -> int:
    """Sum of per-node net gains along the lineage back to the root — the
    COMPOUNDING signal. A child that improves on top of an already-improved
    parent outranks both, so the search extends the best cumulative line
    instead of perpetually restarting from the baseline (net 0)."""
    total = 0
    seen: set[str] = set()
    cur: tree.Node | None = node
    while cur is not None and cur.id not in seen:
        seen.add(cur.id)
        total += _net_gain(cur)
        cur = by_id.get(cur.parent_id) if cur.parent_id else None
    return total


def _rank_key(
    conn: sqlite3.Connection, campaign: str, node: tree.Node,
    by_id: dict[str, tree.Node],
) -> tuple[int, float]:
    """Rank parents by CUMULATIVE net task-gain from root (compounding), with
    the subset_final score as a secondary tiebreak.

    Why NOT absolute solved-count (the previous key): each node's subset_final
    is on a DIFFERENT claimed+guard subset, so the count isn't comparable — the
    full-eval'd root (solved ~70) or a root-equivalent ``no_change`` node
    (solved ~6 on the easy baseline subset) always outranks a real edit that
    netted +1 on a harder flaky subset (solved ~4). That made the baseline an
    UNBEATABLE parent: every pipeline re-branched from root and the search could
    never compound. Net-gain-vs-parent is a denominator-free delta that IS
    comparable; cumulating it along the lineage rewards genuine multi-generation
    improvement and lets a strong child/stone become the tip.
    """
    ev = tree.node_search_eval(conn, campaign=campaign, node_id=node.id)
    score = (ev.score if ev is not None else node.score) or 0.0
    return (_cumulative_net_gain(node, by_id), score)


class MixedHighScoreStrategy(ParentSelectionStrategy):
    """Deterministic 50/50 mix of "pure highest score" and "high score +
    few children" parent selection.

    The coin is the low bit of `sha256(pipeline_id || ":coin")`, so:
      - the same `pipeline_id` always lands on the same side of the coin
        given identical tree state (reproducible re-runs), and
      - two siblings spawned together with different `pipeline_id`s tend
        to disagree on which half they take (exploration).

    Both halves break ties with `_tiebreak_hash(pipeline_id, node_id)`,
    matching the retired strategy's tiebreak rule so pipelines with the
    same pipeline_id+tree pick the same node on each half.
    """

    name = "mixed_high_score"

    def select(
        self,
        conn: sqlite3.Connection,
        *,
        campaign: str,
        subset: str,
        pipeline_id: str,
    ) -> tree.Node:
        eligible = _eligible_nodes(conn, campaign=campaign, subset=subset)
        # Full node map (incl. non-eligible ancestors) so cumulative net-gain can
        # walk each candidate's lineage back to the root.
        by_id = {
            n.id: n
            for n in tree.list_nodes(conn, campaign=campaign, subset=subset)
        }

        if _coin_explore(pipeline_id):
            # Broaden half: `(cum_net_gain DESC, score DESC, child_count ASC, tiebreak)`.
            scored = []
            for n in eligible:
                net, score = _rank_key(conn, campaign, n, by_id)
                cc = tree.child_count(conn, n.id)
                tb = _tiebreak_hash(pipeline_id, n.id)
                scored.append((-net, -score, cc, tb, n))
            scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            return scored[0][4]

        # Exploit half: `(cum_net_gain DESC, score DESC, tiebreak)` — greedy on
        # compounding improvement, score only as a tiebreak.
        scored = []
        for n in eligible:
            net, score = _rank_key(conn, campaign, n, by_id)
            tb = _tiebreak_hash(pipeline_id, n.id)
            scored.append((-net, -score, tb, n))
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        return scored[0][3]


class LlmFirstStrategy(ParentSelectionStrategy):
    """LLM-guided parent picker (ported from the monet_code_eval v8/v9
    ``llm_first`` lineage).

    Presents the LLM a compact "archive card" per eligible node (score,
    child fan-out, solved/unsolved counts, recent improved/regressed tasks)
    and asks it to pick the node most likely to yield a *new* improvement
    when evolved — balancing exploit (high score) against explore (under-
    developed branches / unsolved coverage). Rationale: a greedy score pick
    over-fans the densest branch and ignores which unsolved tasks are still
    reachable; the LLM can reason about that frontier.

    SAFETY: if no LLM backend is reachable (e.g. gateway down) or anything
    in the LLM path fails/returns an unparseable id, it falls back to
    :class:`MixedHighScoreStrategy` deterministically. So selecting
    ``llm_first`` never hard-fails a campaign — it degrades to the proven
    default. (This matters right now: with the gateway unavailable it will
    behave exactly like ``mixed_high_score`` until an LLM is reachable.)
    """

    name = "llm_first"

    def select(
        self,
        conn: sqlite3.Connection,
        *,
        campaign: str,
        subset: str,
        pipeline_id: str,
    ) -> tree.Node:
        eligible = _eligible_nodes(conn, campaign=campaign, subset=subset)
        if len(eligible) == 1:
            return eligible[0]
        fallback = MixedHighScoreStrategy()
        try:
            picked = self._llm_pick(conn, eligible, campaign=campaign, subset=subset)
            if picked is not None:
                return picked
        except Exception:  # noqa: BLE001 — never let the picker break a run
            pass
        return fallback.select(
            conn, campaign=campaign, subset=subset, pipeline_id=pipeline_id,
        )

    def _llm_pick(
        self,
        conn: sqlite3.Connection,
        eligible: list[tree.Node],
        *,
        campaign: str,
        subset: str,
    ) -> "tree.Node | None":
        import os

        # Lazy import: the LLM backend lives in the atelier package and is only
        # needed on the LLM path (keeps parent_selection import-light + avoids
        # a hard dependency when the picker falls back).
        from gate.matchfix_gate import chat_backend_from_credentials

        cards = [self._archive_card(conn, n, campaign=campaign) for n in eligible]
        sys_prompt = (
            "You are the parent selector for a self-evolving coding-agent search. "
            "Pick the ONE parent node most likely to produce a NEW improvement when "
            "mutated next: prefer high score, but favor nodes with unsolved tasks "
            "still reachable and avoid over-fanning an already-dense branch. "
            "Reply with ONLY the node id."
        )
        user = "Eligible parents:\n" + "\n".join(
            f"- id={c['id']} score={c['score']} children={c['children']} "
            f"solved={c['solved']} unsolved={c['unsolved']} "
            f"recent_improved={c['improved']} recent_regressed={c['regressed']}"
            for c in cards
        )
        backend = chat_backend_from_credentials(
            model=os.environ.get("DARWINX_GATE_EQUIVALENCE_MODEL", "gpt-5.5"),
            provider=os.environ.get("DARWINX_GATE_EQUIVALENCE_PROVIDER", "sfr_gateway"),
            max_tokens=64,
        )
        reply = (backend.complete(prompt=user, system=sys_prompt) or "").strip()
        by_id = {n.id: n for n in eligible}
        # exact id, else first eligible id that is a substring of the reply
        if reply in by_id:
            return by_id[reply]
        for nid, node in by_id.items():
            if nid[:8] in reply or nid in reply:
                return node
        return None

    @staticmethod
    def _archive_card(
        conn: sqlite3.Connection, n: tree.Node, *, campaign: str,
    ) -> dict:
        import json

        def _count(row_attr: str) -> int:
            raw = getattr(n, row_attr, None)
            try:
                return len(json.loads(raw)) if raw else 0
            except (TypeError, ValueError):
                return 0

        ev = tree.node_search_eval(conn, campaign=campaign, node_id=n.id)
        score = ev.score if ev else (n.score or 0.0)
        return {
            "id": n.id[:8],
            "score": round(score, 4),
            "children": tree.child_count(conn, n.id),
            "solved": _count("solved_tasks_json"),
            "unsolved": _count("unsolved_tasks_json"),
            "improved": _count("improved_tasks_json"),
            "regressed": _count("regressed_tasks_json"),
        }


DEFAULT_STRATEGY_NAME = MixedHighScoreStrategy.name


_REGISTRY: dict[str, type[ParentSelectionStrategy]] = {
    MixedHighScoreStrategy.name: MixedHighScoreStrategy,
    LlmFirstStrategy.name: LlmFirstStrategy,
}


def get_strategy(name: str, **kwargs) -> ParentSelectionStrategy:
    """Look up a strategy by name.

    The retired strategies (`high_score_few_children`, `fixed`) are no
    longer registered. `mixed_high_score` is currently the only entry —
    pinning every pipeline in a run to the same parent defeats the point
    of evolving a tree, and the pure greedy half of `mixed_high_score`
    already covers the "always exploit best" use case.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown --parent-strategy {name!r}; available: "
            f"{sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def _coin_explore(pipeline_id: str) -> bool:
    """Deterministic 50/50 coin on `pipeline_id`.

    True → take the "broaden" (high_score + few_children) half.
    False → take the "exploit" (pure highest score) half.
    """
    h = hashlib.sha256(f"{pipeline_id}:coin".encode()).hexdigest()
    return (int(h[0], 16) & 1) == 0


def _tiebreak_hash(pipeline_id: str, node_id: str) -> str:
    """Deterministic ordering key used to break score+child ties between
    parallel pipelines. Same `pipeline_id` always picks the same parent
    given identical tree state (important for reproducibility)."""
    return hashlib.sha256(f"{pipeline_id}:{node_id}".encode()).hexdigest()


__all__ = [
    "ParentSelectionStrategy",
    "MixedHighScoreStrategy",
    "LlmFirstStrategy",
    "NoEligibleParentError",
    "DEFAULT_STRATEGY_NAME",
    "get_strategy",
]

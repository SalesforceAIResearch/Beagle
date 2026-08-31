"""The claim pool must stay as wide as the subset when nodes are scored on a shared panel.

`unresolved_tasks` reads the best node's failures. Under DARWINX_GATE_FIXED_EVAL_PANEL a node's eval covers
the panel (40 tasks), not the subset (500), so its failure list names the panel's failures and the pool
would narrow to those the moment the first child outscored the root -- silently, since claims still
succeed. These tests pin the widening and, just as importantly, pin that it does nothing at all when
the best node has a full-width eval.
"""
from __future__ import annotations

import json
import uuid

import pytest

# self_evolve is vendored under the plugin and is only placed on sys.path by _launch.py
# during a DarwinX launch; outside a launch (e.g. the plain test env) it is absent. Skip the
# whole module cleanly then rather than aborting collection for the entire suite.
pool = pytest.importorskip("evolve.pool")
tree = pytest.importorskip("evolve.tree")

CAMPAIGN = "pool-width-test"
SUBSET = "swev-full"

# 10 stands in for the subset; 4 of them stand in for the shared panel.
SUBSET_TASKS = [f"task-{i:02d}" for i in range(10)]
PANEL = SUBSET_TASKS[:4]


@pytest.fixture()
def conn(tmp_path):
    return tree.connect(tmp_path / "state.db")  # applies the schema itself


def _node(conn, *, node_id: str, parent_id: str | None, score: float,
          failed: list[str] | None = None, solved: list[str] | None = None) -> None:
    conn.execute(
        "INSERT INTO nodes (id, campaign, branch_name, subset, parent_id, status, score, "
        "failed_tasks_json, solved_tasks_json, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))",
        (node_id, CAMPAIGN, f"evolve/{node_id}", SUBSET, parent_id, "completed", score,
         json.dumps(failed or []), json.dumps(solved or [])),
    )
    conn.commit()


def _eval(conn, *, node_id: str, kind: str, solved: list[str], failed: list[str]) -> None:
    """Record an eval covering exactly solved+failed -- i.e. its width is what it measured."""
    conn.execute(
        "INSERT INTO node_evals (id, campaign, node_id, eval_kind, subset_label, task_names_json, "
        "n_trials, n_errors, score, solved_tasks_json, unsolved_tasks_json, "
        "partially_solved_tasks_json, task_rewards_json, improved_tasks_json, "
        "regressed_tasks_json, created_at, metadata_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'),'{}')",
        (
            str(uuid.uuid4()), CAMPAIGN, node_id, kind, SUBSET,
            json.dumps(solved + failed), len(solved) + len(failed), 0,
            len(solved) / max(1, len(solved) + len(failed)),
            json.dumps(solved), json.dumps(failed), "[]",
            json.dumps({t: 1.0 for t in solved} | {t: 0.0 for t in failed}), "[]", "[]",
        ),
    )
    conn.commit()


def _root_with_full_eval(conn) -> None:
    """The root, measured on all 10: solves 6, fails 4."""
    _node(conn, node_id="root", parent_id=None, score=0.6)
    _eval(conn, node_id="root", kind="subset_final",
          solved=SUBSET_TASKS[:6], failed=SUBSET_TASKS[6:])


def test_a_panel_scored_child_does_not_narrow_the_pool(conn):
    """The regression this file exists for.

    The child is best, and its eval covers only the 4-task panel. Its own failure list holds one task.
    Before the fix the pool was that one task; the other three the root never solved would be
    unreachable for the rest of the campaign.
    """
    _root_with_full_eval(conn)
    _node(conn, node_id="child", parent_id="root", score=0.75)
    _eval(conn, node_id="child", kind="subset_final",
          solved=PANEL[:3], failed=PANEL[3:])

    got = pool.unresolved_tasks(conn, campaign=CAMPAIGN, subset=SUBSET,
                                subset_filter=SUBSET_TASKS)

    # The root's four failures are all still claimable, plus the panel task the child failed.
    assert set(SUBSET_TASKS[6:]) <= set(got), (
        f"root's failures dropped out of the pool: {sorted(got)}"
    )
    assert len(got) >= 4


def test_a_task_the_best_node_fixed_is_not_re_proposed(conn):
    """The subtraction. task-06 is a root failure the child now solves, so it must not come back."""
    _root_with_full_eval(conn)
    _node(conn, node_id="child", parent_id="root", score=0.75)
    # The panel here includes task-06, one of the root's failures, and the child solves it.
    _eval(conn, node_id="child", kind="subset_final",
          solved=["task-00", "task-06"], failed=["task-01"])

    got = pool.unresolved_tasks(conn, campaign=CAMPAIGN, subset=SUBSET,
                                subset_filter=SUBSET_TASKS)

    assert "task-06" not in got, "a task the best node solves was offered for claiming again"
    assert {"task-07", "task-08", "task-09"} <= set(got)


def test_a_full_width_best_node_changes_nothing(conn):
    """Self-limiting: with a full-width eval the union is empty by construction.

    This is what makes the fix safe to apply unconditionally -- a campaign that scores every node on
    the whole subset sees exactly the pool it saw before.
    """
    _root_with_full_eval(conn)
    _node(conn, node_id="child", parent_id="root", score=0.8)
    # Measured on all 10: solves 8 (including three of the root's four failures), fails 2.
    _eval(conn, node_id="child", kind="subset_final",
          solved=SUBSET_TASKS[:6] + ["task-06", "task-07"], failed=["task-08", "task-09"])

    got = pool.unresolved_tasks(conn, campaign=CAMPAIGN, subset=SUBSET,
                                subset_filter=SUBSET_TASKS)

    assert sorted(got) == ["task-08", "task-09"], (
        f"widening leaked into a full-width campaign: {sorted(got)}"
    )


def test_the_root_being_best_is_not_double_counted(conn):
    """Generation 0: best IS the root, so the union must be a no-op rather than a self-join."""
    _root_with_full_eval(conn)

    got = pool.unresolved_tasks(conn, campaign=CAMPAIGN, subset=SUBSET,
                                subset_filter=SUBSET_TASKS)

    assert sorted(got) == SUBSET_TASKS[6:]
    assert len(got) == len(set(got)), "duplicates from unioning the root with itself"


def test_the_subset_filter_still_bounds_the_pool(conn):
    """Widening must not reach outside the configured subset."""
    _root_with_full_eval(conn)
    _node(conn, node_id="child", parent_id="root", score=0.75)
    _eval(conn, node_id="child", kind="subset_final", solved=PANEL[:3], failed=PANEL[3:])

    got = pool.unresolved_tasks(conn, campaign=CAMPAIGN, subset=SUBSET,
                                subset_filter=["task-07"])

    assert got == ["task-07"]


def test_active_claims_are_still_excluded_from_the_widened_pool(conn):
    """A sibling's claim has to hold against the widened list too, or two workers take one task."""
    _root_with_full_eval(conn)
    _node(conn, node_id="child", parent_id="root", score=0.75)
    _eval(conn, node_id="child", kind="subset_final", solved=PANEL[:3], failed=PANEL[3:])

    claimed = pool.pick_and_claim(
        conn, campaign=CAMPAIGN, subset=SUBSET, k=1,
        pipeline_id="worker-a", subset_filter=SUBSET_TASKS, parent_id="child",
    )
    assert len(claimed) == 1

    got = pool.unresolved_tasks(conn, campaign=CAMPAIGN, subset=SUBSET,
                                subset_filter=SUBSET_TASKS)
    assert claimed[0] not in got, "a held task reappeared in the pool"


class TestRootSearchEval:
    """tree.root_search_eval: the lookup the widening depends on."""

    def test_finds_a_non_adaptive_root(self, conn):
        """root_full_eval misses this one -- 'root_full' is only written by the adaptive path."""
        _root_with_full_eval(conn)
        assert tree.root_full_eval(conn, campaign=CAMPAIGN) is None
        got = tree.root_search_eval(conn, campaign=CAMPAIGN)
        assert got is not None and got.node_id == "root"

    def test_prefers_root_full_when_both_exist(self, conn):
        """The adaptive path writes both; root_full is the wider measurement."""
        _node(conn, node_id="root", parent_id=None, score=0.6)
        _eval(conn, node_id="root", kind="subset_final", solved=PANEL[:2], failed=PANEL[2:])
        _eval(conn, node_id="root", kind="root_full",
              solved=SUBSET_TASKS[:6], failed=SUBSET_TASKS[6:])

        got = tree.root_search_eval(conn, campaign=CAMPAIGN)
        assert got is not None and got.eval_kind == "root_full"
        assert len(got.task_names) == 10

    def test_ignores_evals_on_non_root_nodes(self, conn):
        _node(conn, node_id="root", parent_id=None, score=0.6)
        _node(conn, node_id="child", parent_id="root", score=0.9)
        _eval(conn, node_id="child", kind="subset_final", solved=PANEL, failed=[])

        assert tree.root_search_eval(conn, campaign=CAMPAIGN) is None

    def test_empty_campaign_is_none_not_an_error(self, conn):
        assert tree.root_search_eval(conn, campaign="nothing-here") is None

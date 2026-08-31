"""Stage 3 — tests for the health-derived AIMD admission controller
(``HealthAimdController`` in ``xrlenv/control/capacity.py``)."""

from __future__ import annotations

from xrlenv.control.capacity import (
    AimdConfig,
    HealthAimdController,
    NodeHealthInput,
)


def _good() -> NodeHealthInput:
    """A healthy snapshot — low p95, no errors."""
    return NodeHealthInput(
        create_p95_ms=1_000.0, docker_error_count=0, docker_timeout_count=0,
    )


def _slow() -> NodeHealthInput:
    """p95 above the 60s default bad-threshold."""
    return NodeHealthInput(
        create_p95_ms=90_000.0, docker_error_count=0, docker_timeout_count=0,
    )


def _erroring() -> NodeHealthInput:
    return NodeHealthInput(
        create_p95_ms=1_000.0, docker_error_count=2, docker_timeout_count=1,
    )


def test_unseen_node_gets_the_slow_start_seed() -> None:
    c = HealthAimdController(AimdConfig(initial_limit=16))
    assert c.limit_for("node-A") == 16


def test_bad_tick_high_p95_halves_the_limit() -> None:
    c = HealthAimdController(AimdConfig(initial_limit=16))
    c.step(health={"n": _slow()}, load={"n": 16})
    assert c.limit_for("n") == 8
    assert c.last_contraction_at("n") is not None


def test_bad_tick_docker_errors_halves_the_limit() -> None:
    """Any docker error/timeout in the window is the emergency signal."""
    c = HealthAimdController(AimdConfig(initial_limit=16))
    c.step(health={"n": _erroring()}, load={"n": 4})
    assert c.limit_for("n") == 8  # halved regardless of load


def test_good_tick_at_limit_grows_by_one() -> None:
    c = HealthAimdController(AimdConfig(initial_limit=16))
    c.step(health={"n": _good()}, load={"n": 16})  # load == limit
    assert c.limit_for("n") == 17


def test_good_tick_under_limit_holds() -> None:
    """A healthy but under-loaded node is no evidence it could take
    more — the limit holds, it does not drift up."""
    c = HealthAimdController(AimdConfig(initial_limit=16))
    c.step(health={"n": _good()}, load={"n": 3})
    assert c.limit_for("n") == 16


def test_good_tick_over_limit_holds_post_contraction() -> None:
    """Right after a contraction the node is over its new limit;
    a good tick must NOT grow it — it is being drained down."""
    c = HealthAimdController(AimdConfig(initial_limit=16))
    c.step(health={"n": _slow()}, load={"n": 16})   # contract 16 -> 8
    assert c.limit_for("n") == 8
    c.step(health={"n": _good()}, load={"n": 16})   # load 16 > limit 8
    assert c.limit_for("n") == 8                    # held, not grown


def test_repeated_bad_ticks_floor_at_one() -> None:
    c = HealthAimdController(AimdConfig(initial_limit=16, floor=1))
    for _ in range(10):
        c.step(health={"n": _slow()}, load={"n": 16})
    assert c.limit_for("n") == 1


def test_repeated_good_ticks_cap_at_max_limit() -> None:
    c = HealthAimdController(AimdConfig(initial_limit=16, max_limit=20))
    for _ in range(20):
        # Keep load at the limit each round so every good tick grows it.
        c.step(health={"n": _good()}, load={"n": c.limit_for("n")})
    assert c.limit_for("n") == 20


def test_unknown_health_holds_the_limit() -> None:
    """A pre-Stage-1 node-agent reports no health → neither grow nor
    contract; the limit holds at the seed."""
    c = HealthAimdController(AimdConfig(initial_limit=16))
    c.step(health={"n": None}, load={"n": 16})
    assert c.limit_for("n") == 16
    assert c.last_contraction_at("n") is None


def test_disconnected_node_state_is_pruned() -> None:
    c = HealthAimdController(AimdConfig(initial_limit=16))
    c.step(health={"n": _slow()}, load={"n": 16})  # n now tracked
    assert c.last_contraction_at("n") is not None
    c.step(health={}, load={})  # n no longer connected
    assert c.last_contraction_at("n") is None
    # back to the seed — no stale limit retained
    assert c.limit_for("n") == 16


def test_p95_bad_threshold_is_tunable() -> None:
    """A node at 40s p95 is healthy under the 60s default but bad under
    a tightened 30s threshold."""
    healthy_default = HealthAimdController(AimdConfig(initial_limit=16))
    mid = NodeHealthInput(
        create_p95_ms=40_000.0, docker_error_count=0, docker_timeout_count=0,
    )
    healthy_default.step(health={"n": mid}, load={"n": 16})
    assert healthy_default.limit_for("n") == 17  # 40s < 60s default → good

    strict = HealthAimdController(
        AimdConfig(initial_limit=16, p95_bad_threshold_ms=30_000.0),
    )
    strict.step(health={"n": mid}, load={"n": 16})
    assert strict.limit_for("n") == 8  # 40s > 30s → bad


# ──────────────────────────────────────────────────────────────────────────────
# Edge-triggered error faults + recovery decay (prod "limit pinned at 1" fix).
#
# ``docker_error_count`` is a WINDOWED count (errors in the last ~120 s) but the
# controller ticks far more often (~15 s), so one error sits in the window for
# ~8 ticks. The pre-fix controller halved on every tick where the window was
# non-empty → one error collapsed 16→1 and a steady trickle pinned the node at
# the floor forever, because regrowth needed ``node_load == limit`` exactly —
# never met once admission was choked and the node drained to idle.
# ──────────────────────────────────────────────────────────────────────────────


def _err(count: int, *, timeouts: int = 0, p95_ms: float = 1_000.0) -> NodeHealthInput:
    """Health snapshot carrying a specific *cumulative windowed* error count."""
    return NodeHealthInput(
        create_p95_ms=p95_ms,
        docker_error_count=count,
        docker_timeout_count=timeouts,
    )


def test_single_error_lingering_in_window_halves_only_once() -> None:
    """One error visible across many ticks (the 120 s window) is a single
    fault — it must halve the limit exactly once, not once per tick."""
    c = HealthAimdController(AimdConfig(initial_limit=16, floor=1))
    # The same windowed count=1 reported on 6 consecutive ticks. Keep the
    # node saturated so the only variable under test is the error edge.
    for _ in range(6):
        c.step(health={"n": _err(1)}, load={"n": c.limit_for("n")})
    # First tick: 0→1 is a NEW error → halve 16→8. The following 5 ticks
    # see count==1 (no new edge) and good p95 at load==limit → +1 each.
    # Pre-fix this would have been pinned at 1.
    assert c.limit_for("n") == 13


def test_each_genuinely_new_error_contracts() -> None:
    """A *growing* count (1,2,3,…) means fresh faults keep landing — each
    new edge is a bad tick."""
    c = HealthAimdController(AimdConfig(initial_limit=16, floor=1))
    for i in range(1, 6):
        c.step(health={"n": _err(i)}, load={"n": 16})
    # 16→8→4→2→1→1 across five new-error ticks.
    assert c.limit_for("n") == 1


def test_window_eviction_is_not_a_new_error() -> None:
    """When old errors age out of the window the count SHRINKS; a smaller
    count is never a new fault."""
    c = HealthAimdController(AimdConfig(initial_limit=16))
    c.step(health={"n": _err(3)}, load={"n": 16})  # 0→3 new → halve 16→8
    assert c.limit_for("n") == 8
    # Two errors age out: count 3→1. No new edge → not bad. Under-loaded so
    # recovery applies (limit 8 < seed 16) → grow toward seed.
    c.step(health={"n": _err(1)}, load={"n": 0})
    assert c.limit_for("n") == 9


def test_floored_idle_node_recovers_to_the_seed() -> None:
    """The prod scenario: a node halved to the floor by a transient fault,
    now idle and healthy, climbs back to the slow-start seed — it does NOT
    stay pinned at 1 waiting for ``node_load == limit``."""
    c = HealthAimdController(AimdConfig(initial_limit=16, floor=1))
    # Drive it to the floor with a single new error per tick.
    for i in range(1, 9):
        c.step(health={"n": _err(i)}, load={"n": 16})
    assert c.limit_for("n") == 1
    # Now clean + idle (load 0). The error count holds (no new edge).
    for _ in range(20):
        c.step(health={"n": _err(8)}, load={"n": 0})
    assert c.limit_for("n") == 16  # recovered to the seed, capped there


def test_recovery_stops_at_the_seed_not_max_limit() -> None:
    """Recovery decay only undoes a contraction back to the seed; growing
    ABOVE the seed still requires demonstrated saturation."""
    c = HealthAimdController(AimdConfig(initial_limit=16, max_limit=64, floor=1))
    c.step(health={"n": _err(1)}, load={"n": 16})  # halve 16→8
    for _ in range(40):
        c.step(health={"n": _err(1)}, load={"n": 0})  # idle + clean
    assert c.limit_for("n") == 16  # climbed to seed, then HOLDS (under-loaded)


def test_overloaded_node_below_seed_drains_does_not_recover() -> None:
    """A node still OVER its (contracted) limit is draining down; recovery
    must not fight the drain by growing it, even though it is below seed."""
    c = HealthAimdController(AimdConfig(initial_limit=16, floor=1))
    c.step(health={"n": _err(1)}, load={"n": 16})  # halve 16→8
    assert c.limit_for("n") == 8
    # Still 16 running, limit 8 → over-loaded. Clean health, but draining.
    c.step(health={"n": _err(1)}, load={"n": 16})
    assert c.limit_for("n") == 8  # held, not recovered (node_load > limit)


# ──────────────────────────────────────────────────────────────────────────────
# Idle convergence to the seed (the "idle node stuck at 23" report): a healthy
# node whose live load is at/below the seed converges its limit to the default
# from EITHER side — decay down from above, recover up from below.
# ──────────────────────────────────────────────────────────────────────────────


def _grow_above_seed(c: HealthAimdController, node: str = "n") -> int:
    """Drive a node above the slow-start seed via demonstrated load."""
    for _ in range(12):
        c.step(health={node: _good()}, load={node: c.limit_for(node)})
    assert c.limit_for(node) > 16
    return c.limit_for(node)


def test_idle_node_above_seed_decays_to_the_seed() -> None:
    c = HealthAimdController(AimdConfig(initial_limit=16, max_limit=64))
    _grow_above_seed(c)
    # Now idle (load 0) + healthy → decays back down to the default and
    # holds there (does not keep falling below the seed).
    for _ in range(60):
        c.step(health={"n": _good()}, load={"n": 0})
    assert c.limit_for("n") == 16


def test_lightly_loaded_node_above_seed_decays_to_seed() -> None:
    """Convergence is driven by load being at/below the seed, not strictly
    idle: a node at limit 30 running only 5 has no need for 30."""
    c = HealthAimdController(AimdConfig(initial_limit=16, max_limit=64))
    _grow_above_seed(c)
    for _ in range(60):
        c.step(health={"n": _good()}, load={"n": 5})  # 5 < seed 16
    assert c.limit_for("n") == 16  # decays to seed, never below (16 > 5)


def test_loaded_above_seed_node_keeps_its_headroom() -> None:
    """A node genuinely carrying load above the seed keeps its elevated
    limit — that load justifies it; we don't decay into its working set."""
    c = HealthAimdController(AimdConfig(initial_limit=16, max_limit=64))
    high = _grow_above_seed(c)
    # Carrying 20 (> seed 16) but under the limit → justified → holds.
    c.step(health={"n": _good()}, load={"n": 20})
    assert c.limit_for("n") == high


def test_decay_down_stops_exactly_at_the_seed() -> None:
    c = HealthAimdController(AimdConfig(initial_limit=16, max_limit=64))
    _grow_above_seed(c)
    # One long idle stretch: it must land on the seed, not overshoot below.
    seen_below = False
    for _ in range(100):
        c.step(health={"n": _good()}, load={"n": 0})
        if c.limit_for("n") < 16:
            seen_below = True
    assert not seen_below
    assert c.limit_for("n") == 16


def test_idle_at_seed_holds() -> None:
    """A node already at the default, idle and healthy, holds — no drift."""
    c = HealthAimdController(AimdConfig(initial_limit=16))
    for _ in range(10):
        c.step(health={"n": _good()}, load={"n": 0})
    assert c.limit_for("n") == 16

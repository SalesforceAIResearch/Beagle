"""Tests for xrlenv/control/coordinator.py.

Uses a FakeNodeAgent so no Docker or real backend is needed.
Covers: start_rollout success, idempotency, template-unknown error,
bootstrap-failure cleanup obligation, step on non-RUNNING rollout, finish,
cancel, double-finish idempotency.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from xrlenv.backends.base import ResourceSpec, SandboxHandle
from xrlenv.control.coordinator import RolloutCoordinator, _classify_startup_error
from xrlenv.control.state import InMemoryStateStore
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    RewardContract,
    TemplateCatalog,
    TemplateManifest,
)
from xrlenv.errors import RolloutFailed, TemplateUnknown, XRLEnvError
from xrlenv.types import RolloutStatus

# ── Helpers & fakes ──────────────────────────────────────────────────────────


def _make_manifest(name: str = "t", backend: str = "docker") -> TemplateManifest:
    return TemplateManifest(
        name=name,
        version="0.1",
        digest="sha256:abc",
        image="im:1",

        resources=ResourceSpec(
            cpu_request=0.25,
            cpu_limit=1.0,
            mem_request_bytes=64_000_000,
            mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )


class FakeNodeAgent:
    """Implements the NodeAgent surface without a real backend."""

    def __init__(
        self,
        node_id: str = "fake-node",
        *,
        setup_obs: Any = None,
        raise_on_setup: Exception | None = None,
        raise_on_create: Exception | None = None,
    ) -> None:
        self.node_id = node_id
        self._setup_obs = setup_obs if setup_obs is not None else {"obs": "first"}
        self._raise_on_setup = raise_on_setup
        self._raise_on_create = raise_on_create
        self.destroyed: list[str] = []
        self.setup_calls = 0
        self.teardown_calls = 0
        self._step_results: list[dict[str, Any]] = []
        self._sandboxes: dict[str, Any] = {}

    def supported_backends(self) -> list[str]:
        return ["docker"]

    async def create_sandbox(self, **_kwargs: Any) -> SandboxHandle:
        if self._raise_on_create is not None:
            raise self._raise_on_create
        return SandboxHandle(
            id="sb-1",
            backend="docker",
            backend_ref="container-abc",
            stub_endpoint="tcp://127.0.0.1:9999",
        )

    async def destroy_sandbox(self, sb: SandboxHandle) -> None:
        self.destroyed.append(sb.id)

    async def env_setup(self, _sb: SandboxHandle, **_kwargs: Any) -> dict[str, Any]:
        self.setup_calls += 1
        if self._raise_on_setup is not None:
            raise self._raise_on_setup
        return self._setup_obs

    async def env_step(
        self, _sb: SandboxHandle, action: Any, **_kw: Any,
    ) -> dict[str, Any]:
        if self._step_results:
            return self._step_results.pop(0)
        return {"obs": action, "reward": 1.0, "done": False, "truncated": False, "info": {}}

    async def env_teardown(
        self, _sb: SandboxHandle, **_kw: Any,
    ) -> dict[str, Any]:
        self.teardown_calls += 1
        return {"status": "ok"}

    async def _stub_for(self, _sb: SandboxHandle) -> Any:
        stub = MagicMock()
        stub.commands = MagicMock(return_value={"exit_code": 0})
        return stub

    async def query_image(self, _image: str) -> Any:
        # A1 / D19 (P1.2) — coordinator pre-flight check expects every
        # NodeTransport to answer ``query_image``. Default fake
        # covers the happy path; tests asserting the image_missing
        # branch construct a fake that overrides this method.
        from xrlenv.node.image_cache import ImageQueryResult
        return ImageQueryResult(present=True)


def _make_coordinator(
    agent: FakeNodeAgent | None = None,
    manifest: TemplateManifest | None = None,
    *,
    scratch_registry_host: str | None = None,
) -> tuple[RolloutCoordinator, InMemoryStateStore]:
    if agent is None:
        agent = FakeNodeAgent()
    if manifest is None:
        manifest = _make_manifest()

    catalog = TemplateCatalog()
    catalog.register(manifest)

    # Patch Scheduler.nodes to return our fake agent.
    from xrlenv.control.scheduler import Placement

    sched = MagicMock()
    sched.place.return_value = Placement(node=agent, backend="docker", score=1)
    sched.nodes = [agent]

    state = InMemoryStateStore()
    coord = RolloutCoordinator(
        catalog=catalog, scheduler=sched, state=state,
        scratch_registry_host=scratch_registry_host,
    )
    return coord, state


# ── start_rollout success ─────────────────────────────────────────────────────


async def test_start_rollout_returns_id_and_obs() -> None:
    coord, state = _make_coordinator()
    rid, obs = await coord.start_rollout(template_name="t", init={})
    assert isinstance(rid, str) and len(rid) > 0
    # FakeNodeAgent.env_setup returns {"obs": "first"}, so coordinator extracts
    # setup_reply.get("obs") which is the string "first".
    assert obs == "first"

    record = state.get_rollout(rid)
    assert record.status == RolloutStatus.RUNNING
    assert record.sandbox_id == "sb-1"


async def test_start_rollout_emits_starting_and_running_events() -> None:
    coord, state = _make_coordinator()
    _rid, _ = await coord.start_rollout(template_name="t", init={})
    events = list(state.events_since(0))
    kinds = [e.kind for e in events]
    assert "rollout.starting" in kinds
    assert "rollout.running" in kinds


# ── Idempotency ───────────────────────────────────────────────────────────────


async def test_start_rollout_idempotent_by_request_id() -> None:
    coord, state = _make_coordinator()
    rid1, _obs1 = await coord.start_rollout(
        template_name="t", init={}, request_id="req-abc"
    )
    # Second call with same request_id should return the same rollout_id.
    rid2, _obs2 = await coord.start_rollout(
        template_name="t", init={}, request_id="req-abc"
    )
    assert rid1 == rid2
    # Should not have created a second sandbox.
    assert len(state.list_sandboxes()) == 1


# ── Owner-namespaced idempotency (audit M1) ────────────────────────────────────


async def test_start_rollout_idempotent_per_owner_namespace() -> None:
    """A repeat (owner, request_id) returns the SAME rollout for that owner —
    the idempotency key is namespaced by the server-stamped owner_id, so the
    namespace is per-tenant."""
    coord, state = _make_coordinator()
    rid1, _ = await coord.start_rollout(
        template_name="t", init={}, request_id="r1", owner_id="alice",
    )
    rid_again, _ = await coord.start_rollout(
        template_name="t", init={}, request_id="r1", owner_id="alice",
    )
    assert rid_again == rid1
    assert state.get_rollout(rid1).owner_id == "alice"
    # No second sandbox for the idempotent replay.
    assert len(state.list_sandboxes()) == 1


class _UniqueSandboxAgent(FakeNodeAgent):
    """A FakeNodeAgent that hands out a fresh sandbox id per ``create_sandbox``.

    The base fake returns a constant ``sb-1``; the M1 cross-owner test starts
    two *real* rollouts (alice + bob), and the StateStore rejects a duplicate
    sandbox id — so we need distinct ids to exercise two concurrent placements.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self._sandbox_seq = 0

    async def create_sandbox(self, **_kwargs: Any) -> SandboxHandle:
        self._sandbox_seq += 1
        return SandboxHandle(
            id=f"sb-{self._sandbox_seq}",
            backend="docker",
            backend_ref=f"container-{self._sandbox_seq}",
            stub_endpoint="tcp://127.0.0.1:9999",
        )


async def test_same_request_id_different_owner_is_not_a_collision() -> None:
    """Two tenants may legitimately reuse the same request_id string. Because
    the idempotency key is ``owner\\x00request_id``, bob sending alice's
    request_id gets a fresh rollout stamped with *his* owner — never alice's
    rollout_id or observation (the cross-tenant leak audit M1 closed)."""
    coord, state = _make_coordinator(agent=_UniqueSandboxAgent())
    rid_alice, _ = await coord.start_rollout(
        template_name="t", init={}, request_id="r1", owner_id="alice",
    )
    rid_bob, _ = await coord.start_rollout(
        template_name="t", init={}, request_id="r1", owner_id="bob",
    )

    # Distinct rollouts, each stamped with the right owner.
    assert rid_alice != rid_bob
    assert state.get_rollout(rid_alice).owner_id == "alice"
    assert state.get_rollout(rid_bob).owner_id == "bob"
    # Two real placements happened — bob's request_id was not a cache hit.
    assert len(state.list_sandboxes()) == 2

    # Alice re-sending her request_id still resolves to HER original rollout,
    # not bob's — the namespaces stayed independent after bob's insert.
    rid_alice_again, _ = await coord.start_rollout(
        template_name="t", init={}, request_id="r1", owner_id="alice",
    )
    assert rid_alice_again == rid_alice
    # And bob's repeat resolves to bob's.
    rid_bob_again, _ = await coord.start_rollout(
        template_name="t", init={}, request_id="r1", owner_id="bob",
    )
    assert rid_bob_again == rid_bob
    # Still only the two sandboxes — both repeats were idempotent hits.
    assert len(state.list_sandboxes()) == 2


# ── Template-unknown error ─────────────────────────────────────────────────────


async def test_start_rollout_unknown_template_raises() -> None:
    coord, _ = _make_coordinator()
    with pytest.raises(TemplateUnknown):
        await coord.start_rollout(template_name="nonexistent", init={})


# ── Bootstrap-failure cleanup obligation ──────────────────────────────────────


async def test_bootstrap_failure_destroys_sandbox() -> None:
    """If env_setup fails, the sandbox must still be destroyed (spec-02 cleanup)."""
    agent = FakeNodeAgent(raise_on_setup=RuntimeError("setup boom"))
    coord, state = _make_coordinator(agent=agent)

    with pytest.raises(RolloutFailed):
        await coord.start_rollout(template_name="t", init={})

    # Sandbox was created then destroyed.
    assert "sb-1" in agent.destroyed
    # State store has no live sandbox.
    assert len(state.list_sandboxes()) == 0


async def test_bootstrap_failure_marks_rollout_failed() -> None:
    agent = FakeNodeAgent(raise_on_setup=RuntimeError("boom"))
    coord, state = _make_coordinator(agent=agent)

    with pytest.raises(RolloutFailed):
        await coord.start_rollout(template_name="t", init={})

    rollouts = state.list_rollouts()
    assert len(rollouts) == 1
    assert rollouts[0].status == RolloutStatus.FAILED


async def test_create_failure_marks_rollout_failed_no_destroy() -> None:
    """If create_sandbox itself fails there is nothing to destroy."""
    agent = FakeNodeAgent(raise_on_create=RuntimeError("docker is down"))
    coord, state = _make_coordinator(agent=agent)

    with pytest.raises(RolloutFailed):
        await coord.start_rollout(template_name="t", init={})

    assert agent.destroyed == []
    rollouts = state.list_rollouts()
    assert rollouts[0].status == RolloutStatus.FAILED


# ── step on non-RUNNING rollout ───────────────────────────────────────────────


async def test_step_on_non_running_rollout_raises() -> None:
    coord, state = _make_coordinator()
    rid, _ = await coord.start_rollout(template_name="t", init={})
    # Manually force the record into a terminal state.
    state.update_rollout(rid, status=RolloutStatus.FAILED, reason="forced")

    with pytest.raises(RolloutFailed, match="not running"):
        await coord.step(rid, {"cmd": "noop"})


async def test_step_on_cancelled_rollout_raises_rollout_cancelled() -> None:
    """M1 fix: step() on a CANCELLED rollout must raise RolloutCancelled, not RolloutFailed."""
    from xrlenv.errors import RolloutCancelled

    coord, _ = _make_coordinator()
    rid, _ = await coord.start_rollout(template_name="t", init={})
    await coord.cancel(rid, reason="consumer_cancelled")

    with pytest.raises(RolloutCancelled):
        await coord.step(rid, {"cmd": "noop"})


# ── step accumulates reward ──────────────────────────────────────────────────


async def test_step_accumulates_reward_and_appends_step() -> None:
    agent = FakeNodeAgent()
    agent._step_results = [
        {"obs": "o1", "reward": 0.3, "done": False, "truncated": False, "info": {}},
        {"obs": "o2", "reward": 0.5, "done": False, "truncated": False, "info": {}},
    ]
    coord, state = _make_coordinator(agent=agent)
    rid, _ = await coord.start_rollout(template_name="t", init={})

    await coord.step(rid, "a1")
    await coord.step(rid, "a2")

    record = state.get_rollout(rid)
    assert len(record.steps) == 2
    assert record.final_reward == pytest.approx(0.8)


async def test_step_truncated_seals_as_truncated_with_step_timeout() -> None:
    """Audit M1 (2026-04-29): when the EnvAdapter returns
    ``StepResult.truncated=True`` (e.g. terminal-bench-2 step
    timeout), the coordinator must seal the rollout as TRUNCATED
    with reason=step_timeout, destroy the sandbox, and raise
    RolloutTruncated to the SDK. Pre-fix the SDK marked the
    session done on truncated but the coordinator never
    transitioned the rollout state, so a hung command surfaced
    as a successful rollout with no final reward."""
    from xrlenv.errors import RolloutTruncated

    agent = FakeNodeAgent()
    agent._step_results = [
        {"obs": "o1", "reward": 0.0, "done": False, "truncated": True, "info": {"timed_out": True}},
    ]
    coord, state = _make_coordinator(agent=agent)
    rid, _ = await coord.start_rollout(template_name="t", init={})

    with pytest.raises(RolloutTruncated, match="step timeout"):
        await coord.step(rid, "sleep 999")

    record = state.get_rollout(rid)
    assert record.status == RolloutStatus.TRUNCATED
    assert record.reason == "step_timeout"
    # Sandbox was torn down as part of the seal — the agent's
    # destroy_sandbox was called.
    assert "sb-1" in agent.destroyed


async def test_step_truncated_skips_in_sandbox_final_reward() -> None:
    """Truncation must NOT run ``in_sandbox_final`` reward — the
    sandbox is being torn down as truncated, and running the
    grader on a truncated state would produce nonsense scores
    (or in tb2's case, the verifier wrapper running on an
    incomplete task)."""
    from xrlenv.errors import RolloutTruncated

    agent = FakeNodeAgent()
    agent._step_results = [
        {"obs": "o1", "reward": 0.0, "done": False, "truncated": True, "info": {}},
    ]
    in_sandbox_final_manifest = TemplateManifest(
        name="t",
        version="0.1",
        digest="sha256:abc",
        image="im:1",
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(
            mode="in_sandbox_final",
            cmd=("/opt/xrlenv/run-task-tests.sh",),
            output_format="stdout_float",
        ),
    )
    coord, _state = _make_coordinator(
        agent=agent, manifest=in_sandbox_final_manifest,
    )
    rid, _ = await coord.start_rollout(template_name="t", init={})

    # Track that run_in_sandbox (which the in_sandbox_final reward
    # path calls) is NOT invoked on a truncated step. The fake
    # node's ``_stub_for`` returns a mock; we'll patch the agent
    # to track whether the reward command path was attempted.
    agent.commands_called = False

    async def _track_commands(*_a: Any, **_kw: Any) -> Any:
        agent.commands_called = True
        return MagicMock(exit_code=0, stdout=b"0.5\n", stderr=b"")

    agent.run_in_sandbox = _track_commands  # type: ignore[attr-defined]

    with pytest.raises(RolloutTruncated):
        await coord.step(rid, "sleep 999")

    assert agent.commands_called is False


# ── finish ────────────────────────────────────────────────────────────────────


async def test_finish_seals_trajectory_with_finished_status() -> None:
    coord, _ = _make_coordinator()
    rid, _ = await coord.start_rollout(template_name="t", init={})
    traj = await coord.finish(rid)
    assert traj.status == RolloutStatus.FINISHED
    assert traj.rollout_id == rid


async def test_finish_calls_teardown_and_destroy() -> None:
    agent = FakeNodeAgent()
    coord, _ = _make_coordinator(agent=agent)
    rid, _ = await coord.start_rollout(template_name="t", init={})
    await coord.finish(rid)
    assert agent.teardown_calls == 1
    assert "sb-1" in agent.destroyed


async def test_double_finish_is_idempotent() -> None:
    coord, _ = _make_coordinator()
    rid, _ = await coord.start_rollout(template_name="t", init={})
    traj1 = await coord.finish(rid)
    traj2 = await coord.finish(rid)
    assert traj1.rollout_id == traj2.rollout_id
    assert traj2.status == RolloutStatus.FINISHED


# ── cancel ────────────────────────────────────────────────────────────────────


async def test_cancel_seals_trajectory_with_cancelled_status() -> None:
    coord, _ = _make_coordinator()
    rid, _ = await coord.start_rollout(template_name="t", init={})
    traj = await coord.cancel(rid, reason="consumer_cancelled")
    assert traj.status == RolloutStatus.CANCELLED
    assert traj.reason == "consumer_cancelled"


async def test_cancel_on_already_terminal_rollout_is_idempotent() -> None:
    coord, _ = _make_coordinator()
    rid, _ = await coord.start_rollout(template_name="t", init={})
    await coord.finish(rid)
    traj = await coord.cancel(rid, reason="late_cancel")
    # Already finished — cancel returns the sealed FINISHED trajectory.
    assert traj.status == RolloutStatus.FINISHED


async def test_cancel_calls_teardown_even_when_teardown_raises() -> None:
    """Teardown failure must not block sandbox destroy (best-effort)."""
    agent = FakeNodeAgent()
    call_log: list[str] = []

    async def teardown_raises(sb: Any, **_kw: Any) -> dict[str, Any]:
        call_log.append("teardown_raised")
        raise RuntimeError("teardown failed")

    async def destroy(sb: Any) -> None:
        call_log.append("destroyed")
        agent.destroyed.append(sb.id)

    agent.env_teardown = teardown_raises  # type: ignore[method-assign]
    agent.destroy_sandbox = destroy  # type: ignore[method-assign]

    coord, _ = _make_coordinator(agent=agent)
    rid, _ = await coord.start_rollout(template_name="t", init={})
    # Should not raise even though teardown fails.
    traj = await coord.cancel(rid, reason="test")
    assert traj.status == RolloutStatus.CANCELLED
    assert "teardown_raised" in call_log
    assert "destroyed" in call_log


# ── _classify_startup_error ───────────────────────────────────────────────────


def test_classify_startup_error_image() -> None:
    # _classify_startup_error inspects exc.__class__.__name__ for substrings.
    class ImageNotFound(Exception):
        pass

    assert _classify_startup_error(ImageNotFound("pull failed")) == "image_pull_failed"


def test_classify_startup_error_capacity() -> None:
    class CapacityError(Exception):
        pass

    assert _classify_startup_error(CapacityError("no slots")) == "over_capacity"


def test_classify_startup_error_generic() -> None:
    assert _classify_startup_error(Exception("docker daemon is dead")) == "sandbox_create_failed"


# ── _terminate per-RPC timeouts ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_terminate_destroy_timeout_force_seals_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``destroy_sandbox`` hangs past ``_DESTROY_TIMEOUT_S``, the
    rollout MUST still seal as terminal — pre-fix it stayed in
    ``cancelling`` forever and only direct SQL surgery cleared it.
    The sandbox row gets marked ``destroy_pending`` so D15's
    reconciler / node-startup sweep can reap the orphan later.
    """
    import asyncio as _asyncio

    from xrlenv.control import coordinator as coord_mod

    # Tighten the destroy timeout so the test doesn't take 3 minutes.
    monkeypatch.setattr(coord_mod, "_DESTROY_TIMEOUT_S", 0.05)
    monkeypatch.setattr(coord_mod, "_TEARDOWN_TIMEOUT_S", 0.05)
    monkeypatch.setattr(coord_mod, "_VERIFIER_FETCH_TIMEOUT_S", 0.05)

    class HangingAgent(FakeNodeAgent):
        async def destroy_sandbox(self, sb: SandboxHandle) -> None:
            await _asyncio.sleep(60)  # would block forever pre-fix

    agent = HangingAgent()
    coord, state = _make_coordinator(agent=agent)
    rid, _obs = await coord.start_rollout(template_name="t", init={})

    traj = await coord.cancel(rid, reason="user_cancel")

    # The terminal state IS reached — the rollout doesn't get stuck.
    assert traj.status == RolloutStatus.CANCELLED
    rec = state.get_rollout(rid)
    assert rec.status == RolloutStatus.CANCELLED
    # Sandbox row carries the orphan marker for the GC reconciler.
    assert rec.sandbox_id is not None
    sandbox_rows = [
        s for s in state.list_sandboxes() if s.sandbox_id == rec.sandbox_id
    ]
    if sandbox_rows:  # post-fix the row is preserved as destroy_pending
        assert sandbox_rows[0].status == "destroy_pending"


# ── sweep_stuck_transients ────────────────────────────────────────────────────


def test_sweep_stuck_transients_promotes_old_cancelling_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restart-time sweep: any rollout in a transient state
    (``cancelling`` / ``finishing`` / ``starting``) older than the
    grace window gets force-sealed. Mirrors the pre-fix manual-SQL
    cleanup the operator had to run after a process crash mid-cancel.
    """
    import time as _time

    from xrlenv.control.state import RolloutRecord

    coord, state = _make_coordinator()
    now = _time.time()

    # Three stuck transient rows + one fresh transient + one running
    # (control) + one already terminal (also control). Only the three
    # stale transients should get swept.
    state.insert_rollout(RolloutRecord(
        rollout_id="r-stale-cancel", template="t",
        status=RolloutStatus.CANCELLING, reason="user_cancel",
        node_id="n", sandbox_id=None, final_reward=0.0,
        created_at=now - 1000, last_touched_at=now - 1000,
    ))
    state.insert_rollout(RolloutRecord(
        rollout_id="r-stale-finish", template="t",
        status=RolloutStatus.FINISHING, reason=None,
        node_id="n", sandbox_id=None, final_reward=0.0,
        created_at=now - 1000, last_touched_at=now - 1000,
    ))
    state.insert_rollout(RolloutRecord(
        rollout_id="r-stale-start", template="t",
        status=RolloutStatus.STARTING, reason=None,
        node_id="n", sandbox_id=None, final_reward=0.0,
        created_at=now - 1000, last_touched_at=now - 1000,
    ))
    state.insert_rollout(RolloutRecord(
        rollout_id="r-fresh-cancel", template="t",
        status=RolloutStatus.CANCELLING, reason=None,
        node_id="n", sandbox_id=None, final_reward=0.0,
        created_at=now - 5, last_touched_at=now - 5,
    ))
    state.insert_rollout(RolloutRecord(
        rollout_id="r-running", template="t",
        status=RolloutStatus.RUNNING, reason=None,
        node_id="n", sandbox_id=None, final_reward=0.0,
        created_at=now - 1000, last_touched_at=now - 1000,
    ))
    state.insert_rollout(RolloutRecord(
        rollout_id="r-already-cancelled", template="t",
        status=RolloutStatus.CANCELLED, reason="prior",
        node_id="n", sandbox_id=None, final_reward=0.0,
        created_at=now - 1000, last_touched_at=now - 1000,
    ))

    swept = coord.sweep_stuck_transients(grace_s=300.0)
    assert swept == 3

    # Stale transients promoted to terminal.
    assert state.get_rollout("r-stale-cancel").status == RolloutStatus.CANCELLED
    assert state.get_rollout("r-stale-finish").status == RolloutStatus.FAILED
    assert state.get_rollout("r-stale-start").status == RolloutStatus.FAILED
    # Reason gets a ``/swept_at_startup`` suffix; existing reason is
    # preserved (the sweep doesn't clobber the cancel context).
    assert "swept_at_startup" in (
        state.get_rollout("r-stale-cancel").reason or ""
    )
    assert "user_cancel" in (state.get_rollout("r-stale-cancel").reason or "")
    # Fresh transient inside the grace window is left alone.
    assert state.get_rollout("r-fresh-cancel").status == RolloutStatus.CANCELLING
    # Running rollouts and already-terminal rollouts are untouched.
    assert state.get_rollout("r-running").status == RolloutStatus.RUNNING
    assert state.get_rollout("r-already-cancelled").status == RolloutStatus.CANCELLED


def test_sweep_stuck_transients_no_op_on_clean_state() -> None:
    """A fresh state-store (no transient rows) makes the sweep a
    no-op; returns 0 swept rows. Important: the sweep runs at every
    coordinator init, so it must be cheap when there's nothing to do.
    """
    coord, _ = _make_coordinator()
    assert coord.sweep_stuck_transients() == 0


# ── D17 stage 1: HTTP-cap derivation in env_setup's init_params ──────────────


def test_http_cap_helper_picks_max_phase_plus_buffer() -> None:
    """The pure helper picks the max of the four phase timeouts and
    adds the documented 60 s headroom. Pin both halves of the formula
    so a future tweak to either half (different aggregator, different
    buffer) surfaces here.
    """
    from xrlenv.control.coordinator import (
        _HTTP_TIMEOUT_BUFFER_S,
        _http_cap_from_manifest,
    )

    manifest = _make_manifest()
    # Default manifest values: init=120, setup=60, step=30, teardown=30.
    # Max is 120 → cap = 120 + 60 = 180.
    assert _http_cap_from_manifest(manifest) == 120.0 + _HTTP_TIMEOUT_BUFFER_S

    # A tb2-shaped manifest with a long step_timeout dominates.
    manifest_long_step = manifest.model_copy(update={"step_timeout_s": 1800.0})
    assert _http_cap_from_manifest(manifest_long_step) == (
        1800.0 + _HTTP_TIMEOUT_BUFFER_S
    )


async def test_coordinator_passes_http_cap_to_create_sandbox() -> None:
    """Audit response (H2): the coordinator must stage the per-sandbox
    HTTP cap at ``create_sandbox`` time so ``init_cmd`` /
    ``run_in_sandbox`` calls see the manifest-derived cap rather than
    the 1 h default. The earlier path threaded the cap through
    ``env_setup``'s ``init_params``, which got bypassed by manifests
    with ``init_cmd`` (run_in_sandbox built the StubClient before
    env_setup could stage the cap).
    """
    from xrlenv.control.coordinator import _HTTP_TIMEOUT_BUFFER_S

    captured: dict[str, Any] = {}

    class _CapturingFakeNode(FakeNodeAgent):
        async def create_sandbox(self, **kwargs: Any) -> SandboxHandle:
            captured["create_kwargs"] = kwargs
            return await super().create_sandbox(**{
                k: v for k, v in kwargs.items()
                if k != "stub_request_timeout_s"
            })

    base = _make_manifest()
    manifest = base.model_copy(update={
        "init_timeout_s": 90.0,
        "setup_timeout_s": 45.0,
        "step_timeout_s": 600.0,
        "teardown_timeout_s": 20.0,
    })
    coord, _state = _make_coordinator(
        agent=_CapturingFakeNode(), manifest=manifest,
    )

    await coord.start_rollout(template_name="t", init={})

    assert "stub_request_timeout_s" in captured["create_kwargs"], (
        "coordinator must pass the HTTP cap to create_sandbox"
    )
    assert captured["create_kwargs"]["stub_request_timeout_s"] == (
        600.0 + _HTTP_TIMEOUT_BUFFER_S
    ), (
        f"cap should be max(init,setup,step,teardown) + buffer; got "
        f"{captured['create_kwargs']['stub_request_timeout_s']}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# D17 stage 2 — per-call HTTP cap from manifest's per-phase budget
# ──────────────────────────────────────────────────────────────────────────────


def test_per_phase_http_cap_helper_adds_buffer() -> None:
    """``_per_phase_http_cap`` is the stage-2 analogue of stage 1's
    ``_http_cap_from_manifest`` but takes a single phase budget
    instead of the max across all four. The buffer constant is
    shared so the two formulas stay calibrated together.
    """
    from xrlenv.control.coordinator import (
        _HTTP_TIMEOUT_BUFFER_S,
        _per_phase_http_cap,
    )
    assert _per_phase_http_cap(0.0) == _HTTP_TIMEOUT_BUFFER_S
    assert _per_phase_http_cap(30.0) == 30.0 + _HTTP_TIMEOUT_BUFFER_S
    assert _per_phase_http_cap(1800.0) == 1800.0 + _HTTP_TIMEOUT_BUFFER_S


async def test_step_passes_per_call_cap_from_manifest_step_timeout() -> None:
    """A5 / D17 stage 2: ``RolloutCoordinator.step`` must derive the
    per-call cap from ``manifest.step_timeout_s + buffer`` and pass it
    to :py:meth:`NodeAgent.env_step` as ``request_timeout_s``. Tighter
    than the per-sandbox stage-1 cap (which is sized for the *widest*
    phase) so a hung env_step surfaces in roughly step_timeout +
    buffer rather than max-phase + buffer.
    """
    from xrlenv.control.coordinator import _HTTP_TIMEOUT_BUFFER_S

    captured: dict[str, Any] = {}

    class _CapturingFakeNode(FakeNodeAgent):
        async def env_step(
            self, _sb: SandboxHandle, action: Any, **kw: Any,
        ) -> dict[str, Any]:
            captured["step_request_timeout_s"] = kw.get("request_timeout_s")
            return await super().env_step(_sb, action)

    base = _make_manifest()
    # Long step (1800 s) + tighter setup (45 s) — the per-call cap
    # for env_step must use the 1800 s budget, not the 45 s.
    manifest = base.model_copy(update={
        "init_timeout_s": 90.0,
        "setup_timeout_s": 45.0,
        "step_timeout_s": 1800.0,
        "teardown_timeout_s": 20.0,
    })
    coord, _state = _make_coordinator(
        agent=_CapturingFakeNode(), manifest=manifest,
    )
    rid, _ = await coord.start_rollout(template_name="t", init={})
    await coord.step(rid, action={"cmd": "noop"})

    assert captured["step_request_timeout_s"] == 1800.0 + _HTTP_TIMEOUT_BUFFER_S


async def test_setup_passes_per_call_cap_from_manifest_setup_timeout() -> None:
    """A5 / D17 stage 2: ``env_setup`` carries its own per-call cap
    derived from ``manifest.setup_timeout_s``. Smaller than the
    per-sandbox stage-1 cap when the manifest's longest phase is
    elsewhere (e.g. step_timeout_s ≫ setup_timeout_s).
    """
    from xrlenv.control.coordinator import _HTTP_TIMEOUT_BUFFER_S

    captured: dict[str, Any] = {}

    class _CapturingFakeNode(FakeNodeAgent):
        async def env_setup(
            self, _sb: SandboxHandle, **kw: Any,
        ) -> dict[str, Any]:
            captured["setup_request_timeout_s"] = kw.get("request_timeout_s")
            return await super().env_setup(_sb, **kw)

    base = _make_manifest()
    manifest = base.model_copy(update={
        "init_timeout_s": 90.0,
        "setup_timeout_s": 45.0,
        "step_timeout_s": 1800.0,
        "teardown_timeout_s": 20.0,
    })
    coord, _state = _make_coordinator(
        agent=_CapturingFakeNode(), manifest=manifest,
    )
    await coord.start_rollout(template_name="t", init={})

    assert captured["setup_request_timeout_s"] == 45.0 + _HTTP_TIMEOUT_BUFFER_S


async def test_teardown_passes_per_call_cap_from_manifest_teardown_timeout() -> None:
    """A5 / D17 stage 2: env_teardown's per-call cap derives from
    ``manifest.teardown_timeout_s``. Pinned alongside the env_step
    and env_setup tests so all three phase-specific caps remain wired
    if the call sites are refactored.
    """
    from xrlenv.control.coordinator import _HTTP_TIMEOUT_BUFFER_S

    captured: dict[str, Any] = {}

    class _CapturingFakeNode(FakeNodeAgent):
        async def env_teardown(
            self, _sb: SandboxHandle, **kw: Any,
        ) -> dict[str, Any]:
            captured["teardown_request_timeout_s"] = kw.get("request_timeout_s")
            return await super().env_teardown(_sb)

    base = _make_manifest()
    manifest = base.model_copy(update={
        "init_timeout_s": 90.0,
        "setup_timeout_s": 45.0,
        "step_timeout_s": 1800.0,
        "teardown_timeout_s": 20.0,
    })
    coord, _state = _make_coordinator(
        agent=_CapturingFakeNode(), manifest=manifest,
    )
    rid, _ = await coord.start_rollout(template_name="t", init={})
    await coord.finish(rid)

    assert captured["teardown_request_timeout_s"] == 20.0 + _HTTP_TIMEOUT_BUFFER_S


# ──────────────────────────────────────────────────────────────────────────────
# H4 audit follow-up — per-rollout effective phase budget snapshot +
# TimeoutError lifecycle handling
# ──────────────────────────────────────────────────────────────────────────────


async def test_start_rollout_persists_effective_phase_budgets_in_metadata() -> None:
    """H4 follow-up: the merged per-phase budgets (manifest seed ⊕
    resolver init_params ⊕ user init) must end up on
    ``record.metadata['effective_*_timeout_s']`` so later step() /
    _terminate() calls can read the *effective* budget instead of
    the outer manifest's default. Without this snapshot, Pattern-A
    benchmarks like terminal-bench-2 — whose resolver writes per-task
    ``step_timeout_s = 1800`` for tasks like ``crack-7z-hash`` — would
    have step()'s HTTP cap calibrated against the 30s outer default.
    """
    coord, state = _make_coordinator()
    # User init overrides step_timeout_s; resolver init_params is empty
    # in this test. Outer manifest default is 30s; user override = 600s.
    rid, _ = await coord.start_rollout(
        template_name="t", init={"step_timeout_s": 600.0},
    )
    record = state.get_rollout(rid)
    assert record.metadata["effective_step_timeout_s"] == 600.0
    # Setup + teardown weren't overridden — should reflect manifest default.
    assert record.metadata["effective_setup_timeout_s"] == 60.0
    assert record.metadata["effective_teardown_timeout_s"] == 30.0


async def test_step_per_call_cap_uses_effective_snapshot() -> None:
    """H4 follow-up: when ``record.metadata['effective_step_timeout_s']``
    is present, ``step()`` derives the per-call HTTP cap from it
    instead of from the outer catalog manifest. This is the primary
    fix for the audit's "Pattern-A per-task overrides ignored" finding.
    """
    from xrlenv.control.coordinator import _HTTP_TIMEOUT_BUFFER_S

    captured: dict[str, Any] = {}

    class _CapturingFakeNode(FakeNodeAgent):
        async def env_step(
            self, _sb: SandboxHandle, action: Any, **kw: Any,
        ) -> dict[str, Any]:
            captured["step_request_timeout_s"] = kw.get("request_timeout_s")
            return await super().env_step(_sb, action)

    coord, _state = _make_coordinator(agent=_CapturingFakeNode())
    # Outer manifest default step_timeout_s = 30s. User init override
    # raises the workload budget to 1800s — the snapshot path must
    # propagate that into the per-call HTTP cap so we don't trip
    # ``aiohttp.ClientTimeout`` after just 90s on a legitimate long
    # step.
    rid, _ = await coord.start_rollout(
        template_name="t", init={"step_timeout_s": 1800.0},
    )
    await coord.step(rid, action={"cmd": "noop"})
    assert captured["step_request_timeout_s"] == 1800.0 + _HTTP_TIMEOUT_BUFFER_S


async def test_terminate_per_call_cap_uses_effective_teardown_snapshot() -> None:
    """H4 follow-up: ``_terminate()``'s env_teardown call also reads
    from the effective-budget snapshot. Symmetric with the env_step
    fix; matters when the resolver gives a Pattern-A task a longer
    teardown budget (e.g. flushing a large per-task DB).
    """
    from xrlenv.control.coordinator import _HTTP_TIMEOUT_BUFFER_S

    captured: dict[str, Any] = {}

    class _CapturingFakeNode(FakeNodeAgent):
        async def env_teardown(
            self, _sb: SandboxHandle, **kw: Any,
        ) -> dict[str, Any]:
            captured["teardown_request_timeout_s"] = kw.get("request_timeout_s")
            return await super().env_teardown(_sb)

    coord, _state = _make_coordinator(agent=_CapturingFakeNode())
    # User init override pushes teardown budget from 30s to 300s.
    rid, _ = await coord.start_rollout(
        template_name="t", init={"teardown_timeout_s": 300.0},
    )
    await coord.finish(rid)
    assert captured["teardown_request_timeout_s"] == 300.0 + _HTTP_TIMEOUT_BUFFER_S


async def test_step_terminalizes_cleanly_on_per_call_timeout() -> None:
    """H4 follow-up second half: when ``node.env_step`` raises
    ``TimeoutError`` (the per-call HTTP cap fired — stub is
    unresponsive), ``step()`` must:

    1. Seal the rollout FAILED with reason="transport_timeout"
    2. Skip env_teardown (calling it would hang on the same stub)
    3. Destroy the sandbox via _terminate so capacity is released
    4. Raise RolloutFailed carrying the partial trajectory so
       batch_rollout / Session.__aexit__ bucket the rollout correctly

    Pre-fix the bare TimeoutError propagated and the rollout's row
    stayed RUNNING until something else (hard deadline, operator
    action) cleaned it up.
    """
    class _TimingOutNode(FakeNodeAgent):
        async def env_step(
            self, _sb: SandboxHandle, action: Any, **kw: Any,
        ) -> dict[str, Any]:
            raise TimeoutError("simulated aiohttp ClientTimeout")

    agent = _TimingOutNode()
    coord, state = _make_coordinator(agent=agent)
    rid, _ = await coord.start_rollout(template_name="t", init={})

    # Capture the failure carrier + its partial trajectory.
    with pytest.raises(RolloutFailed) as exc_info:
        await coord.step(rid, action={"cmd": "noop"})

    assert exc_info.value.reason == "transport_timeout"
    assert exc_info.value.partial is not None
    assert exc_info.value.partial.status == RolloutStatus.FAILED

    # Rollout state should be terminalized.
    record = state.get_rollout(rid)
    assert record.status == RolloutStatus.FAILED
    assert record.reason == "transport_timeout"

    # Sandbox should have been destroyed (env_teardown skipped, so
    # destroy_sandbox is the only cleanup hook that runs).
    assert "sb-1" in agent.destroyed

    # And env_teardown must NOT have been called — calling it would
    # have hung on the same wedged stub.
    assert agent.teardown_calls == 0


async def test_step_cap_falls_back_to_catalog_when_snapshot_absent() -> None:
    """Backward-compat / defensive: rollouts started under an older
    code version (before the snapshot was added) won't have
    ``effective_step_timeout_s`` in metadata. The ``_effective_phase_timeout``
    helper must fall back to the catalog manifest in that case so
    those rollouts continue to work.
    """
    from xrlenv.control.coordinator import _HTTP_TIMEOUT_BUFFER_S

    captured: dict[str, Any] = {}

    class _CapturingFakeNode(FakeNodeAgent):
        async def env_step(
            self, _sb: SandboxHandle, action: Any, **kw: Any,
        ) -> dict[str, Any]:
            captured["step_request_timeout_s"] = kw.get("request_timeout_s")
            return await super().env_step(_sb, action)

    coord, state = _make_coordinator(agent=_CapturingFakeNode())
    rid, _ = await coord.start_rollout(template_name="t", init={})

    # Simulate an older-code rollout: drop the snapshot keys from
    # metadata after start_rollout populated them.
    record = state.get_rollout(rid)
    cleaned_metadata = {
        k: v for k, v in record.metadata.items()
        if not k.startswith("effective_")
    }
    state.update_rollout(rid, metadata=cleaned_metadata)

    await coord.step(rid, action={"cmd": "noop"})
    # Manifest default step_timeout_s = 30s → cap = 30 + buffer.
    assert captured["step_request_timeout_s"] == 30.0 + _HTTP_TIMEOUT_BUFFER_S


async def test_coordinator_no_longer_writes_http_cap_into_init_params() -> None:
    """Audit response: confirm the cap is NOT in ``init_params``
    anymore (it's now on ``create_sandbox`` instead). A user-supplied
    ``init`` field that happens to look like the old private key
    flows through verbatim — we no longer overwrite it.
    """
    captured: dict[str, Any] = {}

    class _CapturingFakeNode(FakeNodeAgent):
        async def env_setup(
            self,
            _sb: SandboxHandle,
            *,
            adapter_module: str,
            adapter_class: str,
            init_params: dict[str, Any],
            **_kw: Any,
        ) -> dict[str, Any]:
            captured["init_params"] = init_params
            return await super().env_setup(_sb)

    coord, _state = _make_coordinator(agent=_CapturingFakeNode())
    await coord.start_rollout(
        template_name="t", init={"user_data": "kept"},
    )
    # No HTTP-cap key should appear in init_params anymore.
    assert "_xrlenv_http_timeout_s" not in captured["init_params"]
    # Other user fields still flow through.
    assert captured["init_params"]["user_data"] == "kept"


async def test_init_cmd_does_not_bypass_http_cap_regression() -> None:
    """Audit H2 regression: a manifest with ``init_cmd`` triggers
    ``run_in_sandbox`` BEFORE ``env_setup`` runs. With the prior
    init_params-injection path, the StubClient got built by
    ``run_in_sandbox`` with the 1 h default; ``env_setup`` later
    staged the cap on the record but ``_stub_for`` returned the
    cached stub. With the create_sandbox-side path, the cap is
    staged at handle-creation time and the very first stub-touching
    call sees it. Pin the contract: after ``start_rollout`` on a
    manifest with ``init_cmd``, the per-sandbox record carries the
    derived cap (not None / not 1 h).
    """
    from xrlenv.control.coordinator import _HTTP_TIMEOUT_BUFFER_S
    from xrlenv.node.agent import _SandboxRecord

    # Custom fake node that records the kwargs *and* exposes a per-
    # sandbox record table mirroring what NodeAgent does in real life.
    class _MirroringFakeNode(FakeNodeAgent):
        def __init__(self) -> None:
            super().__init__()
            self.records: dict[str, _SandboxRecord] = {}
            self.init_cmd_called_before_env_setup = False
            self._env_setup_called = False

        async def create_sandbox(self, **kwargs: Any) -> SandboxHandle:
            handle = await super().create_sandbox(**{
                k: v for k, v in kwargs.items()
                if k != "stub_request_timeout_s"
            })
            self.records[handle.id] = _SandboxRecord(
                handle=handle, template="t", backend="docker",
                stub_request_timeout_s_override=kwargs.get(
                    "stub_request_timeout_s",
                ),
            )
            return handle

        async def run_in_sandbox(
            self, sb: SandboxHandle, cmd: list[str], **_kw: Any,
        ) -> Any:
            from xrlenv.backends.base import ExecResult
            if not self._env_setup_called:
                self.init_cmd_called_before_env_setup = True
            return ExecResult(exit_code=0, stdout=b"", stderr=b"", timed_out=False)

        async def env_setup(self, _sb: SandboxHandle, **_kw: Any) -> dict[str, Any]:
            self._env_setup_called = True
            return await super().env_setup(_sb)

    base = _make_manifest()
    manifest_with_init_cmd = base.model_copy(update={
        "init_cmd": ("/bin/true",),
        "init_timeout_s": 5.0,
        "setup_timeout_s": 10.0,
        "step_timeout_s": 90.0,
        "teardown_timeout_s": 7.0,
    })
    fake = _MirroringFakeNode()
    coord, _state = _make_coordinator(
        agent=fake, manifest=manifest_with_init_cmd,
    )

    await coord.start_rollout(template_name="t", init={})

    # Sanity-check: init_cmd really did run before env_setup (the path
    # the audit identified as the bypass).
    assert fake.init_cmd_called_before_env_setup, (
        "test scaffolding regression: init_cmd should run before env_setup"
    )
    # The per-sandbox record carries the derived cap, set at create
    # time. ``stub_request_timeout_s_override`` would be ``None`` if the
    # cap had been injected only through env_setup's init_params.
    assert len(fake.records) == 1
    record = next(iter(fake.records.values()))
    expected = 90.0 + _HTTP_TIMEOUT_BUFFER_S  # max phase = step_timeout_s
    assert record.stub_request_timeout_s_override == expected, (
        f"create_sandbox should stage the cap before init_cmd runs; "
        f"got {record.stub_request_timeout_s_override}, expected {expected}"
    )


# ── A1 / D19 (P1.2) — pre-flight image check ─────────────────────────────────


async def test_preflight_per_node_local_image_missing_fails_fast() -> None:
    """A1 / D19 (P1.2) — when a manifest declares
    ``image_pin_mode='per_node_local'`` AND the chosen node says it
    doesn't have the image, ``_bootstrap_sandbox`` raises
    ``ImageMissingOnNode`` BEFORE sending ``CreateSandboxCommand``.
    The coordinator classifier maps that to
    ``reason="image_missing"``. Per-node-local images have no
    registry authority for the node to pull from, so a missing
    image is a hard error.
    """
    from xrlenv.node.image_cache import ImageQueryResult

    class _AbsentNode(FakeNodeAgent):
        async def query_image(self, _image: str) -> Any:
            return ImageQueryResult(present=False)

        async def create_sandbox(self, **_: Any) -> SandboxHandle:
            raise AssertionError(
                "create_sandbox must NOT be called after per_node_local "
                "pre-flight miss; the rollout should fail with "
                "reason='image_missing' first"
            )

    base = _make_manifest()
    manifest = base.model_copy(update={"image_pin_mode": "per_node_local"})
    coord, state = _make_coordinator(
        agent=_AbsentNode(), manifest=manifest,
    )
    with pytest.raises(RolloutFailed) as exc_info:
        await coord.start_rollout(template_name="t", init={})

    rec = next(r for r in state.list_rollouts())
    assert rec.status == RolloutStatus.FAILED
    assert rec.reason == "image_missing", (
        f"reason should be 'image_missing'; got {rec.reason!r}"
    )
    assert exc_info.value.reason == "image_missing"


async def test_preflight_registry_digest_cold_cache_falls_through_to_create() -> None:
    """Audit response (H3, 2026-05-02): for the default
    ``image_pin_mode='registry_digest'``, a cold-cache miss is NOT
    an error — the node's ``ImageCacheManager.ensure_present()``
    inside ``create_sandbox`` will pull the image from the registry.
    The pre-flight check must fall through (only audit the miss),
    not seal the rollout. Pre-fix, every cold-cache miss became
    ``image_missing``, breaking registry-backed deployments.
    """
    from xrlenv.node.image_cache import ImageQueryResult

    class _ColdNode(FakeNodeAgent):
        async def query_image(self, _image: str) -> Any:
            # Image absent locally — but the registry will pull
            # successfully (modeled by FakeNodeAgent's default
            # create_sandbox just returning sb-1).
            return ImageQueryResult(present=False)

    # Default _make_manifest() has image_pin_mode="registry_digest".
    coord, state = _make_coordinator(agent=_ColdNode())
    rid, _obs = await coord.start_rollout(template_name="t", init={})

    rec = state.get_rollout(rid)
    assert rec.status == RolloutStatus.RUNNING, (
        f"registry_digest cold-cache must fall through to create; "
        f"got status={rec.status!r} reason={rec.reason!r}"
    )


async def test_preflight_emits_per_placement_audit_event() -> None:
    """Audit M3 closure: the pre-flight check emits a
    ``placement.image_check`` audit row for spec-19 transparency.
    Pin both the kind and the payload shape (node_id, image,
    image_pin_mode, present, digest_source, digest) so future
    audit-trail consumers see a stable shape.
    """
    from xrlenv.node.image_cache import ImageQueryResult

    class _PresentNode(FakeNodeAgent):
        async def query_image(self, _image: str) -> Any:
            return ImageQueryResult(
                present=True, digest="sha256:" + "ab" * 32, last_used_at=42.0,
            )

    base = _make_manifest()
    manifest = base.model_copy(update={"image_pin_mode": "per_node_local"})
    coord, state = _make_coordinator(
        agent=_PresentNode(), manifest=manifest,
    )
    await coord.start_rollout(template_name="t", init={})

    audit_rows = [
        row for row in state.audit_since(0)
        if row.kind == "placement.image_check"
    ]
    assert len(audit_rows) == 1
    payload = audit_rows[0].payload
    assert payload["node_id"] == "fake-node"
    assert payload["image"] == "im:1"
    assert payload["image_pin_mode"] == "per_node_local"
    assert payload["present"] is True
    assert payload["digest_source"] == "per_node"
    assert payload["digest"] == "sha256:" + "ab" * 32


async def test_preflight_passes_when_node_has_image() -> None:
    """Sanity: when ``query_image`` returns ``present=True``, the
    pre-flight check is a no-op (other than the audit event) and
    the rollout proceeds normally."""
    coord, state = _make_coordinator()
    rid, _obs = await coord.start_rollout(template_name="t", init={})
    rec = state.get_rollout(rid)
    assert rec.status == RolloutStatus.RUNNING


async def test_preflight_scratch_build_cold_cache_falls_through() -> None:
    """D19 / scratch-registry: a manifest with ``image_pin_mode='scratch_build'``
    and a node that says the image is absent (cold cache) must NOT raise
    ``ImageMissingOnNode``.  A cold scratch miss is the *normal* first-use
    state — the node builds on demand.  This mirrors the registry_digest
    cold-cache test and pins the exclusion list in the fail-fast guard."""
    from xrlenv.image_build import ImageBuildSpec
    from xrlenv.node.image_cache import ImageQueryResult

    class _AbsentNode(FakeNodeAgent):
        async def query_image(self, _image: str) -> Any:
            return ImageQueryResult(present=False)

    base = _make_manifest()
    # scratch_build with an image already set (e.g. a known scratch ref from
    # a prior build). The node says it's absent — normal cold-cache state.
    manifest = base.model_copy(
        update={
            "image_pin_mode": "scratch_build",
            "image_build": ImageBuildSpec(context="./environment"),
        },
    )
    coord, state = _make_coordinator(agent=_AbsentNode(), manifest=manifest)
    # Must NOT raise ImageMissingOnNode — scratch cold-miss falls through.
    rid, _obs = await coord.start_rollout(template_name="t", init={})
    rec = state.get_rollout(rid)
    assert rec.status == RolloutStatus.RUNNING, (
        f"scratch_build cold-cache miss must not fail; "
        f"got status={rec.status!r} reason={rec.reason!r}"
    )


async def test_preflight_scratch_build_emits_digest_source_scratch() -> None:
    """D19 audit event shape for scratch_build: ``digest_source`` must be
    ``'scratch'``.  This pins the audit-trail shape so consumers that
    distinguish build-on-demand from registry-pull see the correct label."""
    from xrlenv.image_build import ImageBuildSpec
    from xrlenv.node.image_cache import ImageQueryResult

    class _PresentNode(FakeNodeAgent):
        async def query_image(self, _image: str) -> Any:
            return ImageQueryResult(
                present=True, digest="sha256:" + "cc" * 32, last_used_at=1.0,
            )

    base = _make_manifest()
    manifest = base.model_copy(
        update={
            "image_pin_mode": "scratch_build",
            "image_build": ImageBuildSpec(context="./environment"),
        },
    )
    coord, state = _make_coordinator(agent=_PresentNode(), manifest=manifest)
    await coord.start_rollout(template_name="t", init={})

    audit_rows = [
        row for row in state.audit_since(0)
        if row.kind == "placement.image_check"
    ]
    assert len(audit_rows) == 1
    payload = audit_rows[0].payload
    assert payload["image_pin_mode"] == "scratch_build"
    assert payload["digest_source"] == "scratch", (
        f"scratch_build must emit digest_source='scratch'; got {payload['digest_source']!r}"
    )


# ── scratch build-on-demand coordinator wiring (slice 2c-ii) ──────────────────


class _ScratchRecordingNode(FakeNodeAgent):
    """FakeNodeAgent that records register_scratch_source calls."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.scratch_registrations: list[tuple[str, Any]] = []
        self.durable_registrations: list[str | None] = []

    def register_scratch_source(
        self, image_ref: str, source: Any, *, durable_to: str | None = None,
    ) -> None:
        self.scratch_registrations.append((image_ref, source))
        self.durable_registrations.append(durable_to)


def _scratch_manifest() -> TemplateManifest:
    from xrlenv.image_build import GitContext, ImageBuildSpec
    spec = ImageBuildSpec(git=GitContext(repo="https://x/y", ref="abc123", subdir="env"))
    return _make_manifest().model_copy(update={
        "image": None,
        "image_pin_mode": "scratch_build",
        "image_build": spec,
    })


async def test_scratch_build_rollout_computes_ref_and_registers_source() -> None:
    """A scratch_build rollout computes the content-addressed ref, sets it as
    the image, and registers the build source on the chosen node."""
    from xrlenv.control.build_plan import GitSource
    from xrlenv.control.scratch_build import resolve_scratch_image

    manifest = _scratch_manifest()
    node = _ScratchRecordingNode()
    coord, state = _make_coordinator(
        agent=node, manifest=manifest, scratch_registry_host="cp-box:5012",
    )
    rid, _obs = await coord.start_rollout(template_name="t", init={})
    assert state.get_rollout(rid).status == RolloutStatus.RUNNING

    assert manifest.image_build is not None
    expected_ref, _ = resolve_scratch_image(
        manifest.image_build, scratch_host="cp-box:5012",
    )
    assert len(node.scratch_registrations) == 1
    reg_ref, reg_source = node.scratch_registrations[0]
    assert reg_ref == expected_ref
    assert isinstance(reg_source, GitSource)
    assert reg_source.repo == "https://x/y"


async def test_scratch_build_rollout_unconfigured_host_fails_clearly() -> None:
    """No scratch registry configured → scratch_build rollout fails fast with
    a clear XRLENV_SCRATCH_REGISTRY_HOST pointer."""
    coord, _state = _make_coordinator(
        agent=_ScratchRecordingNode(), manifest=_scratch_manifest(),
        scratch_registry_host=None,
    )
    with pytest.raises(XRLEnvError, match="XRLENV_SCRATCH_REGISTRY_HOST"):
        await coord.start_rollout(template_name="t", init={})


async def test_scratch_build_rollout_transport_without_register_fails() -> None:
    """A node transport lacking register_scratch_source (e.g. the not-yet-wired
    distributed path) fails fast with a clear pointer rather than a confusing
    missing-image error."""
    coord, _state = _make_coordinator(
        agent=FakeNodeAgent(), manifest=_scratch_manifest(),
        scratch_registry_host="cp-box:5012",
    )
    with pytest.raises(XRLEnvError, match="no register_scratch_source"):
        await coord.start_rollout(template_name="t", init={})


async def test_scratch_build_noop_when_image_already_set() -> None:
    """_maybe_prepare_scratch_image is a no-op when the manifest already
    carries a concrete image (idempotent re-entry / resolver-supplied path).
    No scratch source should be registered on the node."""
    from xrlenv.image_build import GitContext, ImageBuildSpec

    spec = ImageBuildSpec(git=GitContext(repo="https://x/y", ref="abc123"))
    manifest = _make_manifest().model_copy(update={
        "image": "cp-box:5012/scratch/alreadyset",  # image already populated
        "image_pin_mode": "scratch_build",
        "image_build": spec,
    })
    node = _ScratchRecordingNode()
    coord, state = _make_coordinator(
        agent=node, manifest=manifest, scratch_registry_host="cp-box:5012",
    )
    rid, _obs = await coord.start_rollout(template_name="t", init={})
    assert state.get_rollout(rid).status == RolloutStatus.RUNNING
    # No scratch source registration — image was already set
    assert node.scratch_registrations == [], (
        "_maybe_prepare_scratch_image must be a no-op when image is not None"
    )


async def test_scratch_build_idempotent_replay_does_not_double_register() -> None:
    """A replayed scratch_build request_id short-circuits at the idempotency
    check — _maybe_prepare_scratch_image and register_scratch_source must NOT
    be called a second time for the same rollout."""
    manifest = _scratch_manifest()
    node = _ScratchRecordingNode()
    coord, _state = _make_coordinator(
        agent=node, manifest=manifest, scratch_registry_host="cp-box:5012",
    )
    # First call — creates the rollout and registers the source
    rid1, _obs = await coord.start_rollout(
        template_name="t", init={}, request_id="req-scratch-1",
    )
    assert len(node.scratch_registrations) == 1
    # Second call with the same request_id — must return the existing rollout_id
    rid2, _obs2 = await coord.start_rollout(
        template_name="t", init={}, request_id="req-scratch-1",
    )
    assert rid2 == rid1, "idempotent replay must return the same rollout_id"
    # Still exactly one registration — no second call to register_scratch_source
    assert len(node.scratch_registrations) == 1, (
        "idempotency replay must not call register_scratch_source a second time"
    )


async def test_scratch_build_context_dir_uses_tarball_source() -> None:
    """A scratch_build manifest with a context dir (not git) wires a
    TarballSource through to register_scratch_source."""
    import tempfile
    from pathlib import Path

    from xrlenv.control.build_plan import TarballSource
    from xrlenv.image_build import ImageBuildSpec

    with tempfile.TemporaryDirectory() as td:
        ctx = Path(td) / "env"
        ctx.mkdir()
        (ctx / "Dockerfile").write_text("FROM busybox\n")
        spec = ImageBuildSpec(context=str(ctx))
        manifest = _make_manifest().model_copy(update={
            "image": None,
            "image_pin_mode": "scratch_build",
            "image_build": spec,
        })
        node = _ScratchRecordingNode()
        coord, state = _make_coordinator(
            agent=node, manifest=manifest, scratch_registry_host="cp-box:5012",
        )
        rid, _obs = await coord.start_rollout(template_name="t", init={})
        assert state.get_rollout(rid).status == RolloutStatus.RUNNING
        assert len(node.scratch_registrations) == 1
        _ref, source = node.scratch_registrations[0]
        assert isinstance(source, TarballSource), (
            "context-dir scratch_build must register a TarballSource on the node"
        )
        assert source.content_b64 is not None, "TarballSource must carry inline bytes"


async def test_scratch_build_async_register_is_awaited() -> None:
    """When a node transport's register_scratch_source is async, the coordinator
    must await it (via inspect.isawaitable)."""
    from xrlenv.image_build import GitContext, ImageBuildSpec

    class _AsyncRecordingNode(FakeNodeAgent):
        """Node whose register_scratch_source is a coroutine."""

        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            self.scratch_registrations: list[tuple[str, Any]] = []

        async def register_scratch_source(  # type: ignore[override]
            self, image_ref: str, source: Any, *, durable_to: str | None = None,
        ) -> None:
            self.scratch_registrations.append((image_ref, source))

    spec = ImageBuildSpec(git=GitContext(repo="https://x/y", ref="abc123"))
    manifest = _make_manifest().model_copy(update={
        "image": None,
        "image_pin_mode": "scratch_build",
        "image_build": spec,
    })
    node = _AsyncRecordingNode()
    coord, state = _make_coordinator(
        agent=node, manifest=manifest, scratch_registry_host="cp-box:5012",
    )
    rid, _obs = await coord.start_rollout(template_name="t", init={})
    assert state.get_rollout(rid).status == RolloutStatus.RUNNING
    # If the coordinator did NOT await, the coroutine would be garbage-collected
    # without running, so scratch_registrations would remain empty.
    assert len(node.scratch_registrations) == 1, (
        "coordinator must await an async register_scratch_source"
    )


async def test_scratch_build_durable_to_is_threaded_to_node() -> None:
    """image_build.durable_to flows to register_scratch_source; a durable_to
    without a tag gets the content-addressed input_digest appended."""
    from xrlenv.image_build import GitContext, ImageBuildSpec
    spec = ImageBuildSpec(
        git=GitContext(repo="https://x/y", ref="abc123"),
        durable_to="reg.internal:5000/team/env",
    )
    manifest = _make_manifest().model_copy(update={
        "image": None, "image_pin_mode": "scratch_build", "image_build": spec,
    })
    node = _ScratchRecordingNode()
    coord, state = _make_coordinator(
        agent=node, manifest=manifest, scratch_registry_host="cp-box:5012",
    )
    rid, _obs = await coord.start_rollout(template_name="t", init={})
    assert state.get_rollout(rid).status == RolloutStatus.RUNNING
    assert len(node.durable_registrations) == 1
    durable = node.durable_registrations[0]
    assert durable is not None
    assert durable.startswith("reg.internal:5000/team/env:")


async def test_scratch_build_without_durable_registers_none() -> None:
    node = _ScratchRecordingNode()
    coord, _state = _make_coordinator(
        agent=node, manifest=_scratch_manifest(), scratch_registry_host="cp-box:5012",
    )
    await coord.start_rollout(template_name="t", init={})
    assert node.durable_registrations == [None]

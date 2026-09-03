"""Task-level retry — the two layers (infra + content) on the evaluate/run path.

Mirrors the xrlenv benchmark sweeps: infra retry re-runs a trial ONLY on an infra-transient
error (harbor RetryConfig / a per-task loop); content retry re-runs UNRESOLVED tasks (Runner-level,
best-of-attempts). The DarwinX path uses the driver's own retry knobs instead — not tested here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from beagle import benchmarks
from beagle.benchmarks.base import BenchmarkHarness, GradeReport
from beagle.config import RetryPolicy, RunConfig
from beagle.rollout.retry import (
    INFRA_RETRY_EXCEPTIONS,
    better_attempt,
    is_infra_error,
    run_with_infra_retry,
)
from beagle.rollout.runner import Runner
from beagle.types import RolloutStatus, Task, TaskContext, TaskResult


class CapacityExhausted(Exception):
    pass


class AgentTimeoutError(Exception):
    pass


# --- the retry primitives ----------------------------------------------------

def test_is_infra_error_gates_on_type_name() -> None:
    assert is_infra_error(CapacityExhausted("x")) and is_infra_error("NodeLost")
    assert not is_infra_error(AgentTimeoutError("x")) and not is_infra_error("ApiUsageLimitError")


def test_is_infra_error_sees_through_the_wrap_chain() -> None:
    # XrlenvDockerRuntime wraps xrlenv errors: `raise RuntimeError(...) from e`. The matcher must
    # follow __cause__/__context__, else run_with_infra_retry never fires on the docker path.
    try:
        try:
            raise CapacityExhausted("no slot")
        except CapacityExhausted as e:
            raise RuntimeError("xrlenv containers.run failed: …") from e   # explicit __cause__
    except RuntimeError as wrapped:
        assert is_infra_error(wrapped)
    try:
        try:
            raise CapacityExhausted("no slot")
        except CapacityExhausted:
            raise RuntimeError("boom")                                     # implicit __context__
    except RuntimeError as ctx_wrapped:
        assert is_infra_error(ctx_wrapped)
    # a wrapped CONTENT error stays content — never re-rolled
    try:
        try:
            raise AgentTimeoutError("slow")
        except AgentTimeoutError as e:
            raise RuntimeError("wrapped") from e
    except RuntimeError as wrapped:
        assert not is_infra_error(wrapped)


def test_run_with_infra_retry_retries_a_wrapped_infra_error() -> None:
    # Regression for the docker-path gap: a runtime-wrapped infra error is now retried.
    calls = {"n": 0}

    def f() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            try:
                raise CapacityExhausted("blip")
            except CapacityExhausted as e:
                raise RuntimeError("xrlenv exec_run failed: …") from e
        return "ok"

    assert run_with_infra_retry(f, attempts=3) == "ok" and calls["n"] == 3


def test_run_with_infra_retry_retries_infra_then_returns() -> None:
    calls = {"n": 0}

    def f() -> str:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise CapacityExhausted("blip")
        return "ok"

    assert run_with_infra_retry(f, attempts=3) == "ok" and calls["n"] == 3


def test_run_with_infra_retry_exhausts_and_reraises() -> None:
    calls = {"n": 0}

    def f() -> str:
        calls["n"] += 1
        raise CapacityExhausted("blip")

    with pytest.raises(CapacityExhausted):
        run_with_infra_retry(f, attempts=3)
    assert calls["n"] == 3      # tried exactly `attempts` times


def test_run_with_infra_retry_never_retries_content() -> None:
    calls = {"n": 0}

    def f() -> str:
        calls["n"] += 1
        raise AgentTimeoutError("agent gave up")

    with pytest.raises(AgentTimeoutError):
        run_with_infra_retry(f, attempts=5)
    assert calls["n"] == 1      # a content failure is not re-rolled, even with attempts left


def test_better_attempt_pick_rule() -> None:
    unres = TaskResult(task_id="t", resolved=False, reward=0.0)
    res = TaskResult(task_id="t", resolved=True, reward=1.0)
    assert better_attempt(None, unres)                 # first result always wins
    assert better_attempt(unres, res)                  # fail -> pass is an upgrade
    assert not better_attempt(res, unres)              # never downgrade a resolved attempt
    lo = TaskResult(task_id="t", resolved=True, reward=0.5)
    hi = TaskResult(task_id="t", resolved=True, reward=0.9)
    assert better_attempt(lo, hi) and not better_attempt(hi, lo)  # tie on resolved -> higher reward


# --- config surface ----------------------------------------------------------

def test_retry_policy_defaults_and_validation() -> None:
    assert RetryPolicy().model_dump() == {"infra": 0, "content": 0}
    with pytest.raises(ValidationError):
        RetryPolicy(infra=-1)                          # ge=0


def test_timeout_multiplier_is_a_run_knob_not_a_retry_knob() -> None:
    """It scales each task's declared phase budget on the FIRST attempt as much as on a re-run, so
    it never belonged under `retry`. An old config says where it went instead of failing with
    pydantic's generic 'extra fields not permitted'."""
    from beagle.config import RunConfig

    with pytest.raises(ValidationError, match="moved to run.timeout_multiplier"):
        RetryPolicy(infra=1, timeout_multiplier=1.5)
    fields = RunConfig.model_fields
    assert "timeout_multiplier" in fields and fields["timeout_multiplier"].default == 1.0


# --- infra retry on the per-task (docker) harness path -----------------------

class _Agent:
    def rollout_binding(self, ctx):
        return None


class _FailNTimes(BenchmarkHarness):
    """A per-task harness whose ``run`` raises ``exc`` the first ``fails`` calls, then resolves."""

    def __init__(self, fails: int, exc: type[Exception]) -> None:
        self.calls = 0
        self.fails = fails
        self.exc = exc

    def run(self, binding, task, task_ctx, *, runtime):
        self.calls += 1
        if self.calls <= self.fails:
            raise self.exc(f"boom {self.calls}")
        return TaskResult(task_id=task.task_id, resolved=True, reward=1.0)


def _one_item(tid: str = "t"):
    return [(Task(task_id=tid, benchmark="b"), TaskContext(image=None))]


def test_base_rollout_retries_infra_then_succeeds(tmp_path) -> None:
    h = _FailNTimes(2, CapacityExhausted)
    out = list(h.rollout(_Agent(), _one_item(), runtime=None, run_dir=tmp_path,
                         retry=RetryPolicy(infra=2)))
    assert h.calls == 3 and out[0].resolved      # 2 infra failures re-run, 3rd resolves


def test_base_rollout_does_not_retry_content_failure(tmp_path) -> None:
    h = _FailNTimes(1, AgentTimeoutError)
    out = list(h.rollout(_Agent(), _one_item(), runtime=None, run_dir=tmp_path,
                         retry=RetryPolicy(infra=5)))
    assert h.calls == 1                              # content failure is never infra-retried (ran once)
    assert out[0].status is RolloutStatus.FAILED     # ...and is captured, not raised (batch survives)
    assert "AgentTimeoutError" in (out[0].error or "")


# --- content retry at the Runner (harness-agnostic, best-of-attempts) --------

def test_runner_content_retry_reruns_only_unresolved(tmp_path, monkeypatch) -> None:
    rounds: list[tuple[int, list[str]]] = []

    class _H:
        def rollout(self, agent, items, *, runtime, run_dir, parallelism, retry=None,
                timeout_multiplier=1.0, attempt=0, resuming=False):
            rounds.append((attempt, [t.task_id for t, _ in items]))
            out = []
            for t, _ in items:
                # "solid" passes immediately; "flaky" only from round 1 (a rate-limit-style flake).
                resolved = t.task_id == "solid" or (t.task_id == "flaky" and attempt >= 1)
                out.append(TaskResult(task_id=t.task_id, resolved=resolved,
                                      reward=1.0 if resolved else 0.0,
                                      tokens={"prompt": 1, "completion": 1}))
            return out

        def completed(self, items, *, run_dir):
            return []

    class _G:
        def grade(self, results, *, runtime, run_dir, parallelism=1):
            res = sum(1 for r in results if r.resolved)
            return GradeReport(num_tasks=len(results), num_resolved=res,
                               score=res / len(results) if results else 0.0)

    class _B:
        def harness(self):
            return _H()

        def grader(self):
            return _G()

    monkeypatch.setattr(benchmarks, "get", lambda name: _B())
    cfg = RunConfig.from_dict({
        "model": {"name": "gpt-5.5"}, "agent": {"name": "monet", "config": {}},
        "benchmark": {"name": "b", "task_ids": ["solid", "flaky"]},
        "retry": {"content": 1},
    })
    ds = [(Task(task_id="solid", benchmark="b"), TaskContext(image=None)),
          (Task(task_id="flaky", benchmark="b"), TaskContext(image=None))]
    res = Runner(parallelism=1, results_root=tmp_path).run(
        agent=object(), dataset=ds, config=cfg, run_id="R")

    assert res.metrics["num_resolved"] == 2               # flaky resolved on the retry round
    assert rounds[0] == (0, ["solid", "flaky"])           # round 0 runs everything
    assert rounds[1] == (1, ["flaky"])                    # round 1 re-runs ONLY the unresolved task


# --- group_id tagging: every rollout runs inside rollout_metadata(group_id=run_id) -----------

def test_runner_tags_every_rollout_with_group_id(tmp_path, monkeypatch) -> None:
    """The Runner wraps each group's rollout in ``xrlenv.rollout_metadata(group_id=run_id)`` so a
    Ctrl-C teardown (``terminate_raw_group``) can find this run's containers. Regression for the
    evaluate-path bug: harbor/pier acquire through the drop-in and are tagged ONLY by that
    contextvar — previously set only by the evolve caller, so ``beagle evaluate`` left them
    untagged (``group_id`` unset) and Ctrl-C teardown was a no-op."""
    import contextlib as _cl

    import xrlenv

    events: list[tuple[str, str | None]] = []

    @_cl.contextmanager
    def _spy(*, group_id=None, **_kw):
        events.append(("enter", group_id))
        try:
            yield
        finally:
            events.append(("exit", group_id))

    monkeypatch.setattr(xrlenv, "rollout_metadata", _spy)   # _tag_group imports it at call time

    class _H:
        def rollout(self, agent, items, *, runtime, run_dir, parallelism, retry=None,
                timeout_multiplier=1.0, attempt=0, resuming=False):
            events.append(("rollout", None))               # must fall BETWEEN enter/exit
            return [TaskResult(task_id=t.task_id, resolved=True, reward=1.0) for t, _ in items]

        def completed(self, items, *, run_dir):
            return []

    class _G:
        def grade(self, results, *, runtime, run_dir, parallelism=1):
            return GradeReport(num_tasks=len(results),
                               num_resolved=sum(r.resolved for r in results), score=1.0)

    class _B:
        def harness(self):
            return _H()

        def grader(self):
            return _G()

    monkeypatch.setattr(benchmarks, "get", lambda name: _B())
    cfg = RunConfig.from_dict({
        "model": {"name": "gpt-5.5"}, "agent": {"name": "monet", "config": {}},
        "benchmark": {"name": "b", "task_ids": ["t1"]},
    })
    ds = [(Task(task_id="t1", benchmark="b"), TaskContext(image=None))]
    Runner(parallelism=1, results_root=tmp_path).run(
        agent=object(), dataset=ds, config=cfg, run_id="RID-xyz")

    assert ("enter", "RID-xyz") in events                  # tagged with the run's id
    assert events.index(("enter", "RID-xyz")) < events.index(("rollout", None)) \
        < events.index(("exit", "RID-xyz"))                # rollout ran INSIDE the group_id scope


# --- harbor path: the RetryConfig wiring -------------------------------------

def test_harbor_run_job_wires_infra_retry_config(monkeypatch, tmp_path) -> None:
    pytest.importorskip("harbor")
    import harbor
    from harbor.models.trial.config import AgentConfig

    from beagle.benchmarks.harness import HarborHarness

    task_dir = tmp_path / "b" / "t"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('[environment]\ndocker_image = "img:1"\n')

    cap: dict = {}

    class _FakeJob:
        @classmethod
        async def create(cls, config):
            cap["config"] = config
            return cls()

        async def run(self):
            return SimpleNamespace(trial_results=[])

    monkeypatch.setattr(harbor, "Job", _FakeJob)
    items = [(Task(task_id="t", benchmark="b", extras={"harbor_task_dir": str(task_dir)}),
              TaskContext(image=None))]
    HarborHarness()._run_job(items, AgentConfig(), run_dir=tmp_path / "out", parallelism=2,
                             retry=RetryPolicy(infra=3), timeout_multiplier=1.5)

    c = cap["config"]
    assert c.retry.max_retries == 3
    assert c.retry.include_exceptions == set(INFRA_RETRY_EXCEPTIONS)   # infra whitelist only
    assert c.timeout_multiplier == 1.5


def test_harbor_run_job_no_retry_leaves_defaults(monkeypatch, tmp_path) -> None:
    pytest.importorskip("harbor")
    import harbor
    from harbor.models.trial.config import AgentConfig

    from beagle.benchmarks.harness import HarborHarness

    task_dir = tmp_path / "b" / "t"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('[environment]\ndocker_image = "img:1"\n')
    cap: dict = {}

    class _FakeJob:
        @classmethod
        async def create(cls, config):
            cap["config"] = config
            return cls()

        async def run(self):
            return SimpleNamespace(trial_results=[])

    monkeypatch.setattr(harbor, "Job", _FakeJob)
    items = [(Task(task_id="t", benchmark="b", extras={"harbor_task_dir": str(task_dir)}),
              TaskContext(image=None))]
    # retry=None → harbor's own default RetryConfig (max_retries=0) stands.
    HarborHarness()._run_job(items, AgentConfig(), run_dir=tmp_path / "out", parallelism=1)
    assert cap["config"].retry.max_retries == 0


# --- result mapping: an agent-process crash is errored, not a false reward=0 -----

def test_agent_crash_maps_to_error_not_reward_zero() -> None:
    """A monet/agent crash lives at ``agent_result.metadata.error`` with ``exception_info`` unset,
    and the verifier may still score the partial container state ``reward=0``. That must map to an
    ERRORED task (``num_errored``), NOT a legitimate ``reward=0`` — else infra crashes (e.g. a
    ``fetch failed`` network death) silently depress the pass-rate."""
    from pathlib import Path

    from beagle.benchmarks.harness.drivers import _agent_error, _result_from_harbor_json

    crash = {"exception_info": None,
             "verifier_result": {"rewards": {"reward": 0}},
             "agent_result": {"metadata": {"error": "monet exited rc=1: Error: fetch failed"}}}
    r = _result_from_harbor_json(crash, task_id="t", trial_dir=Path("/tmp/x"))
    assert r.status is RolloutStatus.FAILED and r.resolved is False
    assert r.error and "fetch failed" in r.error                 # crash captured as the task error

    # a clean reward=0 with NO agent crash stays a CONTENT outcome (completed, not errored) — the
    # agent ran (produced tokens), it just didn't pass
    clean = {"exception_info": None, "verifier_result": {"rewards": {"reward": 0}},
             "agent_result": {"n_input_tokens": 120, "n_output_tokens": 15, "metadata": {}}}
    rc = _result_from_harbor_json(clean, task_id="t", trial_dir=Path("/tmp/x"))
    assert rc.error is None and rc.status is RolloutStatus.COMPLETED and rc.resolved is False

    # _agent_error reads both the on-disk dict and the live-object (``_trial_to_result``) forms
    class _M:
        error = "boom rc=1"

    class _AR:
        metadata = _M()

    assert _agent_error(_AR()) == "boom rc=1"                    # live object
    assert _agent_error({"metadata": {"error": "x"}}) == "x"     # on-disk dict
    assert _agent_error(None) is None and _agent_error({"metadata": {}}) is None


# --- canonical config → RunConfig.retry --------------------------------------

def test_canonical_build_evaluation_maps_run_retry() -> None:
    from beagle.cli._canonical import build_evaluation

    raw = {
        "run": {"dir": "./tmp", "name": "e", "runtime": "local", "timeout_multiplier": 2.0,
                "retry": {"infra": 4, "content": 2}},
        "agent": {"harness": {"name": "monet"}, "model": {"name": "gpt-5.5"}},
        "data": [{"benchmark": "terminal_bench_2_1", "tasks": ["t1"]}],
    }
    cfg, _ = build_evaluation(raw)
    assert cfg.retry.infra == 4 and cfg.retry.content == 2 and cfg.timeout_multiplier == 2.0


# --- timeout_multiplier vs. harnesses written before it existed ---------------

def test_old_harness_signature_still_runs_at_the_default(tmp_path) -> None:
    """`rollout` is a documented extension point, so a harness predating this knob (or living
    outside the repo) has the old signature. The runner must not kill it with an unexpected
    keyword for a value it doesn't need — and must NOT silently drop a non-default multiplier."""
    from beagle.benchmarks.base import BenchmarkHarness
    from beagle.rollout.runner import _timeout_multiplier_kwarg

    class _PreChangeHarness(BenchmarkHarness):
        def rollout(self, agent, items, *, runtime, run_dir, parallelism=1, retry=None,
                    attempt=0, resuming=False):
            return []

    class _CurrentHarness(BenchmarkHarness):
        def rollout(self, agent, items, *, runtime, run_dir, parallelism=1, retry=None,
                    timeout_multiplier=1.0, attempt=0, resuming=False):
            return []

    old, new = _PreChangeHarness(), _CurrentHarness()
    assert _timeout_multiplier_kwarg(old, 1.0) == {}                    # nothing lost → just run
    assert _timeout_multiplier_kwarg(new, 1.0) == {"timeout_multiplier": 1.0}
    assert _timeout_multiplier_kwarg(new, 1.5) == {"timeout_multiplier": 1.5}
    # asking for scaled budgets from a harness that can't scale them would misreport the run
    with pytest.raises(TypeError, match="does not accept `timeout_multiplier`"):
        _timeout_multiplier_kwarg(old, 1.5)


def test_every_in_repo_harness_accepts_the_knob() -> None:
    # NativeRunnerHarness was missed when the parameter was added, so WAI would have died on the
    # unconditional keyword. Assert the whole family, not the two that were remembered.
    import inspect

    from beagle.benchmarks.base import BenchmarkHarness
    from beagle.benchmarks.harness import (
        DockerHarness,
        HarborHarness,
        NativeRunnerHarness,
        PierHarness,
    )

    for cls in (BenchmarkHarness, HarborHarness, PierHarness, DockerHarness, NativeRunnerHarness):
        params = inspect.signature(cls.rollout).parameters
        assert "timeout_multiplier" in params, f"{cls.__name__}.rollout() is missing the knob"

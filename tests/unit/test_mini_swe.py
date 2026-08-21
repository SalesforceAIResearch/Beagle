"""mini-swe evolvee — drives upstream's own ``mini`` CLI (the documented single-task invocation)
inside the beagle-provisioned container, and parses the trajectory it writes."""

from __future__ import annotations

import json
import pytest

import beagle as bgl
from beagle.agents.mini_swe import _parse_trajectory
from beagle.config import AgentConfig, AgentSourceConfig, ModelConfig
from beagle.types import RolloutStatus, Task, TaskContext


def test_parse_trajectory_sums_tokens_and_counts_turns() -> None:
    raw = json.dumps({"messages": [
        {"role": "system"},
        {"role": "assistant", "extra": {"response": {"usage": {"prompt_tokens": 10, "completion_tokens": 3}}}},
        {"role": "user"},
        {"role": "assistant", "extra": {"response": {"usage": {"prompt_tokens": 5, "completion_tokens": 2}}}},
    ]})
    assert _parse_trajectory(raw) == (
        {"prompt": 15, "completion": 5, "total": 20,
         "input_uncached": 15, "cache_read": 0, "cache_write": 0}, 2)


def test_parse_trajectory_tolerates_garbage() -> None:
    assert _parse_trajectory("not json at all") == ({}, 0)
    assert _parse_trajectory(json.dumps({"nope": 1})) == ({}, 0)   # no messages → no tokens/turns


class _Exec:
    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "") -> None:
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class _FakeRuntime:
    """Captures exec command strings (full argv joined) + their env; returns the fake
    diff/trajectory for the read-backs. A command containing ``fail`` comes back non-zero
    (rc=1, stderr) — models a swallowed failure."""

    def __init__(self, *, diff: str = "", traj: str = "", fail: str | None = None,
                 base: str | None = None) -> None:
        self.cmds: list[str] = []
        self.envs: list[dict | None] = []
        self._diff, self._traj, self._fail, self._base = diff, traj, fail, base
        self.destroyed = False

    def acquire(self, *, image, command, **kw):
        return "H"

    def exec(self, handle, cmd, *, env=None, timeout=None):
        joined = " ".join(str(x) for x in cmd) if isinstance(cmd, list) else str(cmd)
        self.cmds.append(joined)
        self.envs.append(env)
        if self._fail and self._fail in joined:
            return _Exec(returncode=1, stderr="boom-from-container")
        if "git rev-parse HEAD" in joined:                      # base commit (empty → no git repo)
            return _Exec(self._base or "")
        if "git diff" in joined and ("..HEAD" in joined or "--cached" in joined):
            return _Exec(self._diff)                            # base..HEAD or working-tree fallback
        if "cat /logs/agent/mini.traj.json" in joined:
            return _Exec(self._traj)
        return _Exec("")

    def destroy(self, handle):
        self.destroyed = True

    def env_for(self, needle: str) -> dict | None:
        """The env passed to the first exec whose command contains ``needle``."""
        return next((e for c, e in zip(self.cmds, self.envs) if needle in c), None)


def test_run_invokes_the_upstream_mini_cli() -> None:
    # mini-swe as an EVOLVEE: install the evolved repo@ref, drive upstream's `mini` CLI against the
    # provisioned container, capture the patch + trajectory. Asserts the DOCUMENTED flags (not the
    # old wrong `-e local`).
    agent = bgl.agents.build(AgentConfig(
        name="mini-swe", model=ModelConfig(name="gpt-5.5"),
        source=AgentSourceConfig(repo="https://x/mini-fork", ref="abc123"),
        config={"config_path": "src/minisweagent/config/benchmarks/swebench.yaml"}))
    traj = json.dumps({"messages": [
        {"role": "assistant", "extra": {"response": {"usage": {"prompt_tokens": 7, "completion_tokens": 4}}}}]})
    rt = _FakeRuntime(diff="THE-PATCH", traj=traj)

    res = agent.run(Task(task_id="t", problem_statement="fix the bug"),
                    TaskContext(image="img:1", repo_path="/testbed"), runtime=rt)

    mini = next(c for c in rt.cmds if "mini -t" in c)
    # the documented non-interactive invocation, with the corrected flags
    assert "mini -t 'fix the bug' -m gpt-5.5" in mini
    assert "-y --exit-immediately --environment-class local --agent-class default" in mini
    assert "-c /agent/src/minisweagent/config/benchmarks/swebench.yaml" in mini
    assert "-o /logs/agent/mini.traj.json -l 0" in mini
    assert "-e local" not in mini                       # the old (wrong) flag is gone
    assert rt.env_for("mini -t")["MSWEA_CONFIGURED"] == "1"   # skips mini's interactive setup wizard
    # installed the exact repo@ref (its config YAML is the evolvable surface) via the shared
    # GitClone helper; branch/tag ref → shallow `--branch` clone, then a uv-managed 3.11 install
    # (the container's own Python is too old) — the `mini` entrypoint runs from /agent/.venv.
    assert any("git clone --depth 1 --branch abc123 https://x/mini-fork /agent" in c for c in rt.cmds)
    assert any("uv venv --python 3.11 /agent/.venv" in c and "uv pip install -q -e /agent" in c
               for c in rt.cmds)
    assert "/agent/.venv/bin/mini -t 'fix the bug'" in mini
    # patch from git diff (authoritative); tokens + turns parsed; status marked done
    assert res.patch == "THE-PATCH" and res.status is RolloutStatus.COMPLETED
    assert res.tokens == {"prompt": 7, "completion": 4, "total": 11,
                          "input_uncached": 7, "cache_read": 0, "cache_write": 0}
    assert res.num_turns == 1
    assert rt.destroyed


def _agent():
    return bgl.agents.build(AgentConfig(
        name="mini-swe", model=ModelConfig(name="gpt-5.5"),
        source=AgentSourceConfig(repo="https://x/mini-fork", ref="abc123")))


def test_clone_injects_token_for_a_private_fork(monkeypatch) -> None:
    # The experiment copy is private → the clone must authenticate. Token VALUE is forwarded as an
    # exec env var; the URL is rewritten via shell expansion of $GH_TOKEN (not baked into argv).
    monkeypatch.setenv("GH_TOKEN", "secret-tok")
    agent = bgl.agents.build(AgentConfig(
        name="mini-swe", model=ModelConfig(name="gpt-5.5"),
        source=AgentSourceConfig(repo="https://github.com/org/fork", ref="a" * 40),
        config={"token_env": "GH_TOKEN"}))
    rt = _FakeRuntime(diff="P")
    agent.run(Task(task_id="t", problem_statement="x"),
              TaskContext(image="i", repo_path="/testbed"), runtime=rt)
    clone = next(c for c in rt.cmds if "x-access-token" in c)
    assert "x-access-token:$GH_TOKEN@" in clone            # URL rewritten via shell expansion
    assert rt.env_for("x-access-token") == {"GH_TOKEN": "secret-tok"}   # value forwarded, not in argv
    # full-SHA ref → shallow fetch-by-commit (`--branch` rejects a SHA)
    assert "git fetch -q --depth 1 origin" in clone and "FETCH_HEAD" in clone


def test_clone_fails_loud_when_token_env_named_but_unset(monkeypatch) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)
    agent = bgl.agents.build(AgentConfig(
        name="mini-swe", model=ModelConfig(name="gpt-5.5"),
        source=AgentSourceConfig(repo="https://github.com/org/fork", ref="a" * 40),
        config={"token_env": "GH_TOKEN"}))
    rt = _FakeRuntime(diff="P")
    res = agent.run(Task(task_id="t", problem_statement="x"),
                    TaskContext(image="i", repo_path="/testbed"), runtime=rt)
    assert res.status is RolloutStatus.FAILED and res.error and "GH_TOKEN" in res.error
    assert not any("git" in c or "mini -t" in c for c in rt.cmds)   # short-circuit before any exec


def test_run_surfaces_a_failed_install() -> None:
    # A failed install must FAIL LOUD (rc + stderr), not silently yield an empty patch that
    # the grader would score 0 (the bug behind the PENDING/reward=0 smoke).
    rt = _FakeRuntime(fail="pip install")
    res = _agent().run(Task(task_id="t", problem_statement="x"),
                       TaskContext(image="i", repo_path="/testbed"), runtime=rt)
    assert res.status is RolloutStatus.FAILED
    assert res.error and "install failed" in res.error and "boom-from-container" in res.error
    assert not any("mini -t" in c for c in rt.cmds)      # short-circuited before running mini
    assert rt.destroyed


def test_run_surfaces_a_failed_mini_with_no_patch() -> None:
    # mini itself failing (e.g. gateway/creds) with no produced patch → FAILED + surfaced error.
    rt = _FakeRuntime(fail="mini -t", diff="")
    res = _agent().run(Task(task_id="t", problem_statement="x"),
                       TaskContext(image="i", repo_path="/testbed"), runtime=rt)
    assert res.status is RolloutStatus.FAILED
    assert res.error and "mini run failed" in res.error and res.patch is None


def _gw_agent(**config):
    return bgl.agents.build(AgentConfig(
        name="mini-swe", model=ModelConfig(name="gpt-5.5"),
        source=AgentSourceConfig(repo="https://x/mini-fork", ref="abc123"), config=config))


def test_run_routes_litellm_at_the_gateway(monkeypatch) -> None:
    # Model-agnostic: with `provider` set (the gateway selector) the adapter points litellm at the
    # gateway via mini `-c` overrides — api_base (provider-neutral) + a key + force the OpenAI wire
    # shape (what the unified proxy speaks). The config's model name stays bare (gpt-5.5/claude ok).
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", "http://node:18088/")
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_API_KEY", "sk-real")
    rt = _FakeRuntime(diff="P")
    _gw_agent(provider="llm-gateway-express-local-proxy").run(
        Task(task_id="t", problem_statement="x"),
        TaskContext(image="i", repo_path="/testbed"), runtime=rt)
    mini = next(c for c in rt.cmds if "mini -t" in c)
    assert "-c model.model_kwargs.api_base=http://node:18088/" in mini
    assert "-c model.model_kwargs.api_key=sk-real" in mini
    assert "-c model.model_kwargs.custom_llm_provider=openai" in mini
    assert "-m gpt-5.5" in mini and "openai/gpt-5.5" not in mini   # model name stays bare


def test_run_provider_gates_the_gateway(monkeypatch) -> None:
    # `provider` gates gateway routing (mirrors monet): no provider → mini uses its own litellm
    # defaults even if the gateway env happens to be present.
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", "http://node:18088/")
    rt = _FakeRuntime(diff="P")
    _gw_agent().run(Task(task_id="t", problem_statement="x"),   # no provider in config
                    TaskContext(image="i", repo_path="/testbed"), runtime=rt)
    mini = next(c for c in rt.cmds if "mini -t" in c)
    assert "model.model_kwargs" not in mini


def test_run_captures_base_to_head_diff_when_agent_self_commits() -> None:
    # deep-swe/pier: the agent commits its own work, so a post-run working-tree diff is empty. We
    # record the base commit up front and capture `git diff base..HEAD` (after committing leftovers)
    # so the submission reflects the agent's commits, not just the working tree.
    rt = _FakeRuntime(diff="COMMITTED-PATCH", base="basesha0")
    res = _gw_agent().run(Task(task_id="t", problem_statement="x"),
                          TaskContext(image="i", repo_path="/app"), runtime=rt)
    assert res.patch == "COMMITTED-PATCH"
    assert any("git rev-parse HEAD" in c for c in rt.cmds)        # recorded the base
    assert any("git diff basesha0..HEAD" in c for c in rt.cmds)   # diffed base..HEAD, not the tree
    assert any("commit -q -m" in c for c in rt.cmds)              # committed leftovers first


def test_run_wires_effort_and_max_turns() -> None:
    # The shared first-level vocabulary configures mini-swe: effort → the Responses-API model class
    # + reasoning_effort (reasoning models return reasoning + the tool call as separate Responses
    # items, which mini's chat class — reading choices[0] — misses); max_turns → agent.step_limit.
    rt = _FakeRuntime(diff="P")
    _gw_agent(effort="high", max_turns=150).run(
        Task(task_id="t", problem_statement="x"),
        TaskContext(image="i", repo_path="/testbed"), runtime=rt)
    mini = next(c for c in rt.cmds if "mini -t" in c)
    assert "-c model.model_class=litellm_response" in mini
    assert "-c model.model_kwargs.reasoning_effort=high" in mini
    assert "-c agent.step_limit=150" in mini


def test_run_no_effort_stays_on_chat_default() -> None:
    # No effort → no Responses class / reasoning override (non-reasoning runs keep the chat default).
    rt = _FakeRuntime(diff="P")
    _gw_agent(max_turns=150).run(Task(task_id="t", problem_statement="x"),
                                 TaskContext(image="i", repo_path="/testbed"), runtime=rt)
    mini = next(c for c in rt.cmds if "mini -t" in c)
    assert "model.model_class" not in mini and "reasoning_effort" not in mini
    assert "-c agent.step_limit=150" in mini


def test_run_applies_layer_1_2_prompt_override() -> None:
    # Escape hatch: config `prompt_override` {system, instruction} → mini `-c` overrides that
    # REPLACE the agent's own framing. system → agent.system_template, instruction →
    # agent.instance_template. Multi-line bodies survive (whole key=value is shlex-quoted; mini
    # splits on the first `=`).
    agent = bgl.agents.build(AgentConfig(
        name="mini-swe", model=ModelConfig(name="gpt-5.5"),
        source=AgentSourceConfig(repo="https://x/mini-fork", ref="abc123"),
        config={"prompt_override": {"system": "You are careful.\nBe terse.",
                                    "instruction": "Solve: {{task}}"}}))
    rt = _FakeRuntime(diff="P")
    agent.run(Task(task_id="t", problem_statement="x"),
              TaskContext(image="i", repo_path="/testbed"), runtime=rt)
    mini = next(c for c in rt.cmds if "mini -t" in c)
    assert "-c" in mini and "agent.system_template=You are careful.\nBe terse." in mini
    assert "agent.instance_template=Solve: {{task}}" in mini      # keeps the {{task}} placeholder


def test_run_without_prompt_override_touches_no_agent_templates() -> None:
    rt = _FakeRuntime(diff="P")
    _agent().run(Task(task_id="t", problem_statement="x"),
                 TaskContext(image="i", repo_path="/testbed"), runtime=rt)
    mini = next(c for c in rt.cmds if "mini -t" in c)
    assert "agent.system_template" not in mini and "agent.instance_template" not in mini


def test_install_run_split_and_commit_for_base_head_graders() -> None:
    # install/run split: install() clones+builds (INSTALL phase, network open on pier/harbor);
    # run_in() runs mini, captures the working-tree patch, THEN commits so a base..HEAD grader
    # (deep-swe/pier's verifier.collect) sees the work too. Both share one container handle.
    agent = _agent()   # branch ref, no token
    rt = _FakeRuntime(diff="THE-PATCH", traj='{"messages": []}')
    handle = rt.acquire(image="i", command=["sleep", "infinity"])
    agent.install(handle, TaskContext(image="i", repo_path="/testbed"), runtime=rt)
    res = agent.run_in(handle, Task(task_id="t", problem_statement="x"),
                       TaskContext(image="i", repo_path="/testbed"), runtime=rt)
    assert res.patch == "THE-PATCH" and res.status is RolloutStatus.COMPLETED
    assert any("git clone --depth 1 --branch abc123" in c for c in rt.cmds)      # install: clone
    assert any("uv pip install -q -e /agent" in c for c in rt.cmds)              # install: build
    assert any("/agent/.venv/bin/mini -t" in c for c in rt.cmds)                 # run_in: run
    # capture (git diff --cached) precedes the commit; commit makes base..HEAD reflect the work
    assert any("git add -A && git diff --cached" in c for c in rt.cmds)
    assert any('commit -q -m "beagle agent changes"' in c for c in rt.cmds)


def test_failed_install_raises_agent_install_error() -> None:
    from beagle.agents.core.base import AgentInstallError

    rt = _FakeRuntime(fail="pip install")   # the uv install step fails
    handle = rt.acquire(image="i", command=["sleep", "infinity"])
    with pytest.raises(AgentInstallError, match="install failed"):
        _agent().install(handle, TaskContext(image="i", repo_path="/testbed"), runtime=rt)


def test_network_hosts_is_the_gateway(monkeypatch) -> None:
    # the run phase only needs the LLM gateway → allowlisted on a network-restricted benchmark
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", "http://node:18088/")
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_API_KEY", "sk-real")
    assert _agent().network_hosts() == ["http://node:18088/"]


def test_network_hosts_empty_without_gateway(monkeypatch) -> None:
    monkeypatch.delenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", raising=False)
    assert _agent().network_hosts() == []

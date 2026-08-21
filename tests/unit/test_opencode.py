"""opencode evolvee — clones + builds the experiment-copy Bun source in the provisioned
container, drives opencode's own ``run --format json`` headless CLI, and parses the event
stream it writes. Mirrors the monet/mini-swe adapter tests (a fake runtime + scripted output)."""

from __future__ import annotations

import json

import pytest

import beagle as bgl
from beagle.agents.core.spec import AgentSource, AgentSpec, ModelSpec
from beagle.agents.core.usage import Usage
from beagle.agents.opencode import _helpers as _oc
from beagle.types import RolloutStatus, Task, TaskContext


# --- pure helpers ------------------------------------------------------------

def _stream(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events)


def _combined(rc: int, stream: str) -> str:
    """The inner script's fenced stdout: rc line + sentinel-fenced event stream."""
    return (f"OPENCODE_RC={rc}\n{_oc._STREAM_SENTINEL_START}\n{stream}\n{_oc._STREAM_SENTINEL_END}\n")


def test_parse_combined_output_splits_rc_and_stream() -> None:
    stream = _stream({"type": "text", "part": {"id": "t", "text": "hi"}})
    rc, out = _oc.parse_combined_output(_combined(0, stream))
    assert rc == 0
    assert json.loads(out)["part"]["text"] == "hi"


def test_parse_combined_output_missing_sentinels_is_tolerant() -> None:
    rc, out = _oc.parse_combined_output("garbage with no rc line")
    assert rc is None and out == ""


def test_usage_and_turns_summed_across_step_finish() -> None:
    stream = _stream(
        {"type": "step_finish", "part": {"tokens": {"input": 100, "output": 20, "reasoning": 5,
                                                     "cache": {"read": 10, "write": 0}}}},
        {"type": "step_finish", "part": {"tokens": {"input": 50, "output": 8}}},
    )
    # input=fresh, cache.read/write split out, reasoning folds into output; prompt total = 160.
    assert _oc.parse_opencode_usage(stream) == Usage(input_uncached=150, cache_read=10, cache_write=0, output=33)
    assert _oc.parse_opencode_usage(stream).to_token_counts()["prompt"] == 100 + 10 + 50
    assert _oc.count_opencode_turns(stream) == 2


def test_last_stream_error_reads_error_events() -> None:
    stream = _stream(
        {"type": "text", "part": {"id": "t", "text": "x"}},
        {"type": "error", "error": {"name": "ProviderError", "message": "429 rate limited"}},
    )
    assert _oc.last_stream_error(stream) == "429 rate limited"
    assert _oc.last_stream_error(_stream({"type": "error", "error": "boom"})) == "boom"


def test_build_provider_config_is_openai_compatible() -> None:
    cfg = _oc.OpenCodeConfig(model="gpt-5.5", provider_id="gw")
    doc = json.loads(_oc.build_provider_config(cfg, {"api_base": "http://gw/v1", "api_key": "sk-1"}))
    prov = doc["provider"]["gw"]
    assert prov["npm"] == "@ai-sdk/openai-compatible"
    assert prov["options"] == {"baseURL": "http://gw/v1", "apiKey": "sk-1"}
    assert "gpt-5.5" in prov["models"]


def test_build_inner_script_pipes_prompt_and_composes_flags() -> None:
    cfg = _oc.OpenCodeConfig(model="gpt-5.5", provider_id="gw", variant="high")
    s = _oc.build_inner_script(cfg, repo_path="/testbed")
    assert 'printf \'%s\' "$OPENCODE_PROMPT" | bun ' in s   # prompt via stdin (dodges argv limits)
    assert "index.ts run" in s and "--format json" in s
    assert "--model gw/gpt-5.5" in s
    assert "--variant high" in s
    assert "--dir /testbed" in s
    assert "--auto" in s                                    # default behavior flag


def test_build_inner_script_omits_variant_when_unset() -> None:
    cfg = _oc.OpenCodeConfig(model="gpt-5.5", provider_id="gw", variant="")
    assert "--variant" not in _oc.build_inner_script(cfg, repo_path="/t")


def test_install_script_pins_bun_version() -> None:
    cfg = _oc.OpenCodeConfig(model="m", bun_version="1.3.14")
    assert "bun-v1.3.14" in _oc.build_install_script(cfg)


# --- agent over a fake runtime ----------------------------------------------

class _Exec:
    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "") -> None:
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class _FakeRuntime:
    """Captures (cmd, env) per exec; returns the invoke's combined stdout when the exec carries
    the OPENCODE_PROMPT env, the base sha for ``git rev-parse``, the diff for ``git diff …HEAD``.
    A command containing ``fail`` comes back non-zero (models a swallowed clone/install failure)."""

    def __init__(self, *, invoke_out: str = "", diff: str = "", base: str | None = None,
                 fail: str | None = None) -> None:
        self.calls: list[tuple[str, dict | None]] = []
        self._invoke_out, self._diff, self._base, self._fail = invoke_out, diff, base, fail
        self.destroyed = False

    def acquire(self, **kw):
        self.calls.append(("acquire", kw))
        return "H"

    def exec(self, handle, cmd, *, env=None, timeout=None, workdir=None):
        joined = " ".join(str(x) for x in cmd) if isinstance(cmd, list) else str(cmd)
        self.calls.append((joined, env))
        if self._fail and self._fail in joined:
            return _Exec(returncode=1, stderr="boom-from-container")
        if env and "OPENCODE_PROMPT" in env:                       # the opencode invoke
            return _Exec(self._invoke_out)
        if "git rev-parse HEAD" in joined:                         # base commit
            return _Exec(self._base or "")
        if "git diff" in joined and "HEAD" in joined:              # base..HEAD submission
            return _Exec(self._diff)
        return _Exec("")

    def destroy(self, handle):
        self.destroyed = True

    def invoke_env(self) -> dict | None:
        return next((e for c, e in self.calls if e and "OPENCODE_PROMPT" in e), None)

    def cmds(self) -> list[str]:
        return [c for c, _ in self.calls if c != "acquire"]


def _agent(config: dict | None = None):
    return bgl.agents.build(AgentSpec(
        name="opencode",
        source=AgentSource(repo="https://github.com/o/opencode-beagle", ref="deadbeef"),
        model=ModelSpec(name="gpt-5.5"),
        config=config or {"provider": "gw", "forward_env": ["LLM_GATEWAY_EXPRESS_API_KEY"]},
    ))


def test_run_happy_path_clones_invokes_and_captures_base_head(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", "http://gw/v1")
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_API_KEY", "sk-key")
    stream = _stream(
        {"type": "step_start", "part": {"id": "p0"}},
        {"type": "text", "part": {"id": "t1", "text": "editing"}},
        {"type": "tool_use", "part": {"callID": "c1", "tool": "edit",
                                      "state": {"status": "completed", "input": {"file": "a.py"},
                                                "output": "ok"}}},
        {"type": "step_finish", "part": {"tokens": {"input": 30, "output": 12}}},
    )
    rt = _FakeRuntime(invoke_out=_combined(0, stream), diff="diff --git a/a.py b/a.py\n", base="basecommit")
    res = _agent().run(Task(task_id="t1", problem_statement="fix a.py"),
                       TaskContext(image="img", repo_path="/testbed"), runtime=rt)

    assert res.status is RolloutStatus.COMPLETED and res.resolved
    assert res.patch == "diff --git a/a.py b/a.py\n"              # base..HEAD, not the empty stream patch
    assert res.tokens == {"prompt": 30, "completion": 12, "total": 42,
                          "input_uncached": 30, "cache_read": 0, "cache_write": 0}
    assert res.num_turns == 1
    assert res.trajectory.format == "opencode-json"
    # native stream surfaced so the docker-drop-in path persists agent/opencode.stream.jsonl + ATIF
    assert res.trajectory_text is not None and '"type": "tool_use"' in res.trajectory_text
    assert rt.destroyed
    cmds = rt.cmds()
    assert any("https://github.com/o/opencode-beagle" in c and "deadbeef" in c for c in cmds)  # clone
    assert any("git add -A" in c and "commit" in c for c in cmds)                                # committed
    env = rt.invoke_env()
    assert env["OPENCODE_PROMPT"] == "fix a.py"
    # gateway wired via opencode's native config-content env → an openai-compatible provider block
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"])["provider"]["gw"]["options"]["baseURL"] == "http://gw/v1"
    assert env["LLM_GATEWAY_EXPRESS_API_KEY"] == "sk-key"        # forward_env carried the creds
    # The inline provider declares the model, so opencode must NOT fetch its remote catalog: from source
    # that fetch is `Effect.orDie`, so on a sealed-egress trial (DeepSWE) it would crash before the first
    # LLM call. This native flag skips it — set whenever the gateway (self-contained provider) is wired.
    assert env["OPENCODE_DISABLE_MODELS_FETCH"] == "1"


def test_run_records_harbor_shaped_phase_timing(monkeypatch) -> None:
    # The shared run() seam times acquire/install/run_in as harbor-shaped spans + duration_sec, so a
    # docker-drop-in task's result.json timing is comparable to a harbor/pier trial's native breakdown.
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", "http://gw/v1")
    stream = _stream({"type": "step_finish", "part": {"tokens": {"input": 5, "output": 1}}})
    rt = _FakeRuntime(invoke_out=_combined(0, stream), base="b")
    res = _agent().run(Task(task_id="t", problem_statement="x"),
                       TaskContext(image="i", repo_path="/w"), runtime=rt)
    assert set(res.timing) == {"environment_setup", "agent_setup", "agent_execution"}
    for span in res.timing.values():
        assert span["started_at"].endswith("Z") and span["finished_at"].endswith("Z")
        assert span["finished_at"] >= span["started_at"]
    assert res.duration_sec >= 0.0


def test_install_failure_result_still_carries_timing() -> None:
    # An install failure has environment_setup + agent_setup but no agent_execution (never ran).
    rt = _FakeRuntime(fail="bun install")
    res = _agent().run(Task(task_id="t", problem_statement="x"),
                       TaskContext(image="i", repo_path="/w"), runtime=rt)
    assert res.status is RolloutStatus.FAILED and "install failed" in (res.error or "")
    assert set(res.timing) == {"environment_setup", "agent_setup"}
    assert res.trajectory.format == "opencode-json"   # _install_error_result attaches the stream ref


def test_run_reports_opencode_nonzero_exit_as_failed() -> None:
    rt = _FakeRuntime(invoke_out=_combined(1, _stream({"type": "text", "part": {"id": "t", "text": "x"}})),
                      base="b")
    res = _agent().run(Task(task_id="t", problem_statement="x"),
                       TaskContext(image="i", repo_path="/w"), runtime=rt)
    assert res.status is RolloutStatus.FAILED and not res.resolved
    assert res.error and "opencode exited rc=1" in res.error


def test_run_surfaces_stream_error(monkeypatch) -> None:
    stream = _stream({"type": "error", "error": {"message": "no provider configured"}})
    rt = _FakeRuntime(invoke_out=_combined(0, stream), base="b")
    res = _agent().run(Task(task_id="t", problem_statement="x"),
                       TaskContext(image="i", repo_path="/w"), runtime=rt)
    assert res.status is RolloutStatus.FAILED
    assert res.error == "stream_error: no provider configured"


def test_install_raises_on_clone_failure() -> None:
    rt = _FakeRuntime(fail="git fetch")   # the clone argv's fetch step (git_clone_argv)
    res = _agent().run(Task(task_id="t", problem_statement="x"),
                       TaskContext(image="i", repo_path="/w"), runtime=rt)
    assert res.status is RolloutStatus.FAILED and "git clone failed" in (res.error or "")
    assert rt.destroyed


def test_install_raises_on_build_failure() -> None:
    rt = _FakeRuntime(fail="bun install")
    res = _agent().run(Task(task_id="t", problem_statement="x"),
                       TaskContext(image="i", repo_path="/w"), runtime=rt)
    assert res.status is RolloutStatus.FAILED and "install failed" in (res.error or "")


def test_install_run_in_split_for_pier(monkeypatch) -> None:
    # The pier/DeepSWE path calls install() then run_in() directly (not run()). install clones+builds;
    # run_in invokes opencode then commits so `git diff base..HEAD` (the DeepSWE submission) sees edits.
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", "http://gw/v1")
    stream = _stream({"type": "step_finish", "part": {"tokens": {"input": 5, "output": 1}}})
    rt = _FakeRuntime(invoke_out=_combined(0, stream), diff="diff --git a/y b/y\n", base="b")
    agent = _agent()
    ctx = TaskContext(image="i", repo_path="/work")
    agent.install("H", ctx, runtime=rt)
    res = agent.run_in("H", Task(task_id="t", problem_statement="x"), ctx, runtime=rt)
    assert res.status is RolloutStatus.COMPLETED and res.patch == "diff --git a/y b/y\n"
    assert any("git add -A" in c and "commit" in c for c in rt.cmds())


def test_variant_wired_from_effort_config() -> None:
    cfg = _agent({"provider": "gw", "effort": "high"})._cfg()
    assert cfg.variant == "high"
    assert "--variant high" in _oc.build_inner_script(cfg, repo_path="/w")


def test_network_and_install_hosts() -> None:
    a = _agent()
    assert a.install_hosts()[:2] == ["github.com", "codeload.github.com"]
    assert "bun.sh" in a.install_hosts()
    assert "registry.npmjs.org" in a.install_hosts()
    # node-gyp downloads Node headers from nodejs.org to build native modules (tree-sitter,
    # node-pty) during `bun install`; missing it 403s on a filtered-egress trial and fails install.
    assert "nodejs.org" in a.install_hosts()
    assert a.source().repo.split("//")[1].split("/")[0] in a.install_hosts()  # source host appended


def test_network_hosts_is_gateway(monkeypatch) -> None:
    # Ensure a clean gateway env first — `import xrlenv` (any prior test) autoloads .env, which
    # may set the real proxy URL; the "empty" assertion must not depend on ambient env.
    monkeypatch.delenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", raising=False)
    assert _agent().network_hosts() == []                       # no gateway env → empty
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", "http://gw/v1")
    assert _agent().network_hosts() == ["http://gw/v1"]


def test_default_source_requires_repo() -> None:
    a = bgl.agents.build(AgentSpec(name="opencode", model=ModelSpec(name="gpt-5.5")))
    with pytest.raises(ValueError, match="experiment-copy repo"):
        a._cfg()

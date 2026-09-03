"""Unit tests for the monet agent + the M+N harbor integration seam.

All hermetic: no harbor, no cluster, no Docker. The load-bearing test drives
``MonetAgent.run`` against a *fake* ``ContainerRuntime`` — proving the single
``run()`` works on any runtime, which is the whole point of the capability seam.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import beagle as bgl
from beagle.agents.monet import _helpers as _monet
from beagle.agents.core.usage import Usage
from beagle.agents.core.spec import AgentSource, AgentSpec, ModelSpec
from beagle.benchmarks.harness.drivers import _agent_identity
from beagle.rollout.runtime.harbor_env import HarborEnvRuntime
from beagle.rollout.runtime.runtime import ExecResult
from beagle.rollout.runtime.transport import BindMount
from beagle.types import RolloutStatus, Task, TaskContext


# --- pure _monet helpers -----------------------------------------------------


def test_parse_monet_usage_sums_per_turn() -> None:
    stdout = "\n".join(
        [
            '{"type":"turn_complete","usage":{"inputTokens":10,"outputTokens":5}}',
            '{"type":"turn_complete","usage":{"inputTokens":20,"cacheReadTokens":3,"outputTokens":7}}',
            "not json — ignored",
        ]
    )
    # inputTokens=fresh, cacheReadTokens=cache hit; prompt total stays 33 (10+20+3).
    assert _monet.parse_monet_usage(stdout) == Usage(input_uncached=30, cache_read=3, cache_write=0, output=12)
    assert _monet.count_monet_turns(stdout) == 2
    assert _monet.parse_monet_usage("") == Usage()


def test_parse_monet_usage_openai_cache_is_subset_not_additive() -> None:
    # OpenAI-compatible gateway path: `cacheTokens` (= prompt_tokens_details.cached_tokens) is a
    # cached SUBSET of `inputTokens` (the TOTAL prompt). prompt must equal inputTokens, NOT be
    # inflated by adding the cache on top (the old bug: 100 + 90 = 190).
    line = '{"usage":{"inputTokens":100,"cacheTokens":90,"outputTokens":10}}'
    u = _monet.parse_monet_usage(line)
    assert u == Usage(input_uncached=10, cache_read=90, cache_write=0, output=10)
    assert u.to_token_counts()["prompt"] == 100          # NOT 190
    # a cache that (bogusly) exceeds the total is clamped — fresh never goes negative
    clamped = _monet.parse_monet_usage('{"usage":{"inputTokens":100,"cacheTokens":150}}')
    assert clamped == Usage(input_uncached=0, cache_read=100, cache_write=0, output=0)


def test_parse_monet_usage_anthropic_cache_is_additive() -> None:
    # Anthropic-shaped: `inputTokens` is FRESH; cacheReadTokens / cacheCreationTokens are billed IN
    # ADDITION → prompt = 30 + 70 + 5 = 105.
    line = '{"usage":{"inputTokens":30,"cacheReadTokens":70,"cacheCreationTokens":5,"outputTokens":12}}'
    u = _monet.parse_monet_usage(line)
    assert u == Usage(input_uncached=30, cache_read=70, cache_write=5, output=12)
    assert u.to_token_counts()["prompt"] == 105


def test_hit_max_turns_and_stream_error() -> None:
    assert _monet.hit_max_turns('{"type":"max_turns_reached"}') is True
    assert _monet.hit_max_turns('{"type":"turn_complete"}') is False
    s = '{"type":"stream_error","error":"Throttling"}\n{"type":"stream_error","error":"5xx"}'
    assert _monet.last_stream_error(s) == "5xx"  # last one wins
    assert _monet.last_stream_error("{}") is None


def test_summarize_monet_failure_skips_node_warnings() -> None:
    # Node prints undici's EnvHttpProxyAgent ExperimentalWarning (+ the trace-warnings hint) BEFORE
    # any real error, so the summary must skip them and report the actual cause — not the warning.
    stderr = (
        "(node:732) [UNDICI-EHPA] Warning: EnvHttpProxyAgent is experimental, expect them to change at any time.\n"
        "(Use `node --trace-warnings ...` to show where the warning was created)\n"
        "/app/src/query/loop.js:3205\n"
        "      throw err;\n"
        "Error: gateway 429 after 6 retries: rate limited\n"
        "    at streamWithProviderRetry (/app/src/api/retry.js:215:22)\n"
    )
    summary = _monet.summarize_monet_failure(1, stderr)
    assert "UNDICI-EHPA" not in summary and "EnvHttpProxyAgent" not in summary
    assert summary == "monet exited rc=1: Error: gateway 429 after 6 retries: rate limited"

    # All-warnings stderr → still summarize (fall back to the first line) rather than claim "no stderr".
    warn_only = "(node:9) [DEP0040] DeprecationWarning: punycode is deprecated"
    assert _monet.summarize_monet_failure(1, warn_only) == f"monet exited rc=1: {warn_only}"
    assert _monet.summarize_monet_failure(3, "  \n \n") == "monet exited rc=3 (no stderr)"


def test_parse_combined_output_roundtrip() -> None:
    # Emulate the inner script's fenced stdout.
    combined = "\n".join(
        [
            "MONET_RC=0",
            _monet._PATCH_SENTINEL_START,
            "diff --git a/f b/f",
            _monet._PATCH_SENTINEL_END,
            _monet._MONET_OUT_SENTINEL_START,
            '{"type":"turn_complete"}',
            _monet._MONET_OUT_SENTINEL_END,
        ]
    )
    rc, patch, monet_stdout = _monet.parse_combined_output(combined)
    assert rc == 0
    assert patch == "diff --git a/f b/f"
    assert monet_stdout == '{"type":"turn_complete"}'


def test_config_paths_and_inner_script() -> None:
    cfg = _monet.MonetConfig(model="claude-x", container_path="/opt/a", entrypoint="bin/monet.js")
    assert cfg.monet_bin == "/opt/a/bin/monet.js"
    assert cfg.stream_path == "/logs/agent/monet.stream.jsonl"
    script = _monet.build_inner_script(cfg, repo_path="/work")
    assert "node /opt/a/bin/monet.js" in script
    assert '--print="$MONET_PROMPT"' in script  # prompt via env, no template
    assert "--strict-max-turns" in script  # hard turn cap
    assert "cd /work" in script


# --- MonetAgent.run against a fake runtime (the M+N proof) --------------------


class _FakeRuntime:
    """Records calls; returns canned ExecResults in order. Substitutable for any
    ContainerRuntime — a real container is never touched."""

    def __init__(self, outputs: list[ExecResult]) -> None:
        self.calls: list = []
        self._outputs = list(outputs)

    def acquire(self, **kw):  # noqa: ANN003
        self.calls.append(("acquire", kw))
        return "H"

    def exec(self, handle, command, *, timeout=None, env=None, workdir=None):  # noqa: ANN001
        joined = " ".join(str(x) for x in command) if isinstance(command, list) else str(command)
        self.calls.append(("exec", command, env))
        # Plumbing execs (the proxy-preload install/write in install(); run_in's commit) don't consume
        # the scripted clone/build/invoke outputs — the tests only script those three.
        if any(p in joined for p in (
            "npm install undici", "beagle-proxy.cjs", "git add -A", "git rev-parse", "git diff")):
            return ExecResult(returncode=0, stdout="", stderr="")
        return self._outputs.pop(0) if self._outputs else ExecResult(returncode=0, stdout="", stderr="")

    def destroy(self, handle) -> None:  # noqa: ANN001
        self.calls.append(("destroy", handle))


def _monet_agent(config: dict | None = None):
    return bgl.agents.build(
        AgentSpec(
            name="monet",
            source=AgentSource(repo="https://github.com/o/monet_code-beagle", ref="deadbeef"),
            model=ModelSpec(name="claude-x"),
            config=config or {},
        )
    )


def _invoke_call(rt):
    """(command, env) of the monet-invoke exec — the one carrying MONET_PROMPT — regardless of how
    many install-plumbing execs (clone/build/proxy-preload) precede it."""
    for kind, cmd, env in ((c[0], c[1], c[2]) for c in rt.calls if c[0] == "exec"):
        if env and "MONET_PROMPT" in env:
            return cmd, env
    raise AssertionError("no monet invoke exec found")


def _invoke_stdout() -> str:
    return "\n".join(
        [
            "MONET_RC=0",
            _monet._PATCH_SENTINEL_START,
            "diff --git a/x b/x",
            _monet._PATCH_SENTINEL_END,
            _monet._MONET_OUT_SENTINEL_START,
            '{"type":"turn_complete","usage":{"inputTokens":100,"outputTokens":40}}',
            _monet._MONET_OUT_SENTINEL_END,
        ]
    )


def test_monet_run_happy_path_on_fake_runtime() -> None:
    rt = _FakeRuntime(
        [
            ExecResult(returncode=0, stdout="", stderr=""),  # clone
            ExecResult(returncode=0, stdout="", stderr=""),  # install
            ExecResult(returncode=0, stdout=_invoke_stdout(), stderr=""),  # invoke
        ]
    )
    agent = _monet_agent()
    task = Task(task_id="t1", problem_statement="fix the bug")
    res = agent.run(task, TaskContext(image="img", repo_path="/work", agent_timeout_s=1800), runtime=rt)

    assert res.status is RolloutStatus.COMPLETED and res.resolved
    assert res.patch == "diff --git a/x b/x"
    assert res.tokens == {"prompt": 100, "completion": 40, "total": 140,
                          "input_uncached": 100, "cache_read": 0, "cache_write": 0}
    assert res.num_turns == 1
    assert res.trajectory.format == "monet-stream-json"
    # the raw stream is handed back as trajectory_text so the docker/swe-bench path can persist
    # agent/monet.stream.jsonl (+ ATIF); harbor bind-mounts it instead, but setting it is harmless there.
    assert res.trajectory_text is not None and '"usage"' in res.trajectory_text
    assert res.trajectory.path.name == "monet.stream.jsonl"   # → agent/<name> written by the docker path

    kinds = [c[0] for c in rt.calls]
    assert kinds[0] == "acquire" and kinds[-1] == "destroy"   # acquire → …execs… → destroy
    execs = [" ".join(c[1]) if isinstance(c[1], list) else str(c[1]) for c in rt.calls if c[0] == "exec"]
    # clone argv carries the experiment-copy repo + ref; a commit follows the invoke (base..HEAD).
    assert any("https://github.com/o/monet_code-beagle" in c and "deadbeef" in c for c in execs)
    assert any("git add -A" in c and "commit" in c for c in execs)
    # the invoke is the exec whose env carries MONET_PROMPT (+ the proxy NODE_OPTIONS).
    _invoke_cmd, invoke_env = _invoke_call(rt)
    assert invoke_env["MONET_PROMPT"] == "fix the bug"
    assert "beagle-proxy.cjs" in invoke_env["NODE_OPTIONS"]


def test_monet_install_run_in_split_for_pier() -> None:
    # The pier/DeepSWE path calls install() then run_in() directly (not run()). install clones+builds;
    # run_in invokes monet then commits so `git diff base..HEAD` (the DeepSWE submission) sees edits.
    rt = _FakeRuntime([
        ExecResult(returncode=0, stdout="", stderr=""),               # clone
        ExecResult(returncode=0, stdout="", stderr=""),               # install
        ExecResult(returncode=0, stdout=_invoke_stdout(), stderr=""),  # invoke
    ])
    agent = _monet_agent()
    ctx = TaskContext(image="i", repo_path="/work", agent_timeout_s=1800)
    agent.install("H", ctx, runtime=rt)
    res = agent.run_in("H", Task(task_id="t", problem_statement="x"), ctx, runtime=rt)
    assert res.status is RolloutStatus.COMPLETED and res.patch == "diff --git a/x b/x"
    cmds = [" ".join(c[1]) if isinstance(c[1], list) else str(c[1]) for c in rt.calls if c[0] == "exec"]
    assert any("git add -A" in c and "commit" in c for c in cmds)   # committed for base..HEAD


def test_monet_run_in_prefers_base_head_diff() -> None:
    # monet self-commits, so its stream patch is empty on a solved task; run_in recovers the work as
    # `git diff base..HEAD` and that committed diff becomes the submission patch (like mini-swe).
    stream = "\n".join([
        "MONET_RC=0",
        _monet._MONET_OUT_SENTINEL_START,
        '{"type":"turn_complete","usage":{"inputTokens":100,"outputTokens":40}}',
        _monet._MONET_OUT_SENTINEL_END,
    ])  # no PATCH sentinels → empty stream patch

    class _GitFake(_FakeRuntime):
        def exec(self, handle, command, *, timeout=None, env=None, workdir=None):  # noqa: ANN001
            joined = " ".join(str(x) for x in command) if isinstance(command, list) else str(command)
            self.calls.append(("exec", command, env))
            if "git rev-parse" in joined:
                return ExecResult(returncode=0, stdout="basecommit\n", stderr="")
            if "git diff" in joined:
                return ExecResult(returncode=0, stdout="diff --git a/y b/y\n", stderr="")
            if any(p in joined for p in ("npm install undici", "beagle-proxy.cjs", "git add -A")):
                return ExecResult(returncode=0, stdout="", stderr="")
            return self._outputs.pop(0) if self._outputs else ExecResult(returncode=0, stdout="", stderr="")

    rt = _GitFake([
        ExecResult(returncode=0, stdout="", stderr=""),      # clone
        ExecResult(returncode=0, stdout="", stderr=""),      # install
        ExecResult(returncode=0, stdout=stream, stderr=""),  # invoke (empty stream patch)
    ])
    agent = _monet_agent()
    ctx = TaskContext(image="i", repo_path="/work", agent_timeout_s=1800)
    agent.install("H", ctx, runtime=rt)
    res = agent.run_in("H", Task(task_id="t", problem_statement="x"), ctx, runtime=rt)
    assert res.patch == "diff --git a/y b/y\n"      # base..HEAD, not the empty stream patch
    assert res.status is RolloutStatus.COMPLETED


def test_monet_install_raises_on_clone_failure() -> None:
    import pytest

    from beagle.agents.core.base import AgentInstallError
    rt = _FakeRuntime([ExecResult(returncode=128, stdout="", stderr="fatal: repo not found")])
    with pytest.raises(AgentInstallError, match="clone failed"):
        _monet_agent().install("H", TaskContext(image="i", repo_path="/w", agent_timeout_s=1800), runtime=rt)


def test_monet_pier_network_and_install_hosts(monkeypatch) -> None:
    # A filtered-egress trial allowlists network_hosts() (run: the gateway) + install_hosts() (clone
    # + node/npm). monet's source host folds into install_hosts (deduped).
    monkeypatch.setenv("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL", "http://gw:18088/")
    agent = _monet_agent()
    assert agent.network_hosts() == ["http://gw:18088/"]
    ih = agent.install_hosts()
    assert {"github.com", "registry.npmjs.org", "deb.nodesource.com"} <= set(ih)
    assert ih.count("github.com") == 1                              # source host deduped


def test_monet_run_clone_failure_is_reported_and_container_destroyed() -> None:
    rt = _FakeRuntime([ExecResult(returncode=128, stdout="", stderr="fatal: repo not found")])
    res = _monet_agent().run(Task(task_id="t"), TaskContext(image="img", agent_timeout_s=1800), runtime=rt)
    assert res.status is RolloutStatus.FAILED and not res.resolved
    assert "git clone failed" in res.error
    assert ("destroy", "H") in rt.calls  # cleanup ran despite the failure


def test_monet_default_source_requires_repo() -> None:
    bare = bgl.agents.build("monet")  # no source in spec
    with pytest.raises(ValueError, match="experiment-copy repo"):
        bare.run(Task(task_id="t"), TaskContext(image="img", agent_timeout_s=1800), runtime=_FakeRuntime([]))


# --- _agent_identity (serialized descriptor the harbor shim rebuilds from) ----


def test_agent_identity_captures_source_for_evolvable() -> None:
    ident = _agent_identity(_monet_agent())
    assert ident["agent"] == "monet"
    assert ident["source"] == {
        "repo": "https://github.com/o/monet_code-beagle",
        "ref": "deadbeef",
        "entrypoint": "bin/monet.js",  # intrinsic default filled in by _default_source
    }
    assert ident["model"] == "claude-x"


def test_agent_identity_source_none_for_non_evolvable() -> None:
    # cursor is EDIT-only (not Evolvable) — no source descriptor.
    ident = _agent_identity(bgl.agents.build("cursor"))
    assert ident["agent"] == "cursor" and ident["source"] is None


# --- HarborEnvRuntime async bridge -------------------------------------------


class _FakeEnv:
    """Duck-typed harbor BaseEnvironment: async exec returning .return_code/.stdout."""

    def __init__(
        self, *, raise_exc: bool = False, exc: BaseException | None = None, stdout: str | None = "hi\n"
    ) -> None:
        self.calls: list = []
        self._raise = raise_exc
        self._exc = exc
        self._stdout = stdout

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):  # noqa: ANN001
        self.calls.append((command, cwd, env, timeout_sec, user))
        if self._exc is not None:
            raise self._exc
        if self._raise:
            raise TimeoutError("deadline")
        return SimpleNamespace(stdout=self._stdout, stderr=None, return_code=0)


def _loop_in_thread():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    return loop, t


def test_harbor_env_runtime_bridges_exec() -> None:
    loop, t = _loop_in_thread()
    try:
        env = _FakeEnv()
        rt = HarborEnvRuntime(env, loop)
        res = rt.exec("H", ["echo", "hi"], workdir="/w")
        assert res.returncode == 0 and res.stdout == "hi\n"
        # argv joined into the single shell string harbor's exec wants; cwd forwarded.
        assert env.calls[0][0] == "echo hi" and env.calls[0][1] == "/w"
        assert rt.destroy("H") is None  # harbor owns lifecycle
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join()


def test_harbor_env_runtime_env_hook_transforms_agent_command_env() -> None:
    # A filtered-egress harness (pier) supplies env_hook=environment.agent_process_env so the agent's
    # commands (clone/run) get the Squid proxy vars. The generic runtime just applies the hook; the
    # harbor path (no hook) leaves env untouched.
    loop, t = _loop_in_thread()
    try:
        # pier-like: the hook injects HTTP(S)_PROXY into whatever env the agent passes
        proxy = {"HTTP_PROXY": "http://squid:3128", "HTTPS_PROXY": "http://squid:3128"}
        hook = lambda e: {**(e or {}), **proxy}  # noqa: E731 — mirrors agent_process_env
        env = _FakeEnv()
        HarborEnvRuntime(env, loop, env_hook=hook).exec("H", ["git", "clone"], env={"GH_TOKEN": "t"})
        passed_env = env.calls[0][2]
        assert passed_env["GH_TOKEN"] == "t" and passed_env["HTTP_PROXY"] == "http://squid:3128"

        # harbor path: no hook → env passed through verbatim (proxy vars never injected)
        env2 = _FakeEnv()
        HarborEnvRuntime(env2, loop).exec("H", ["git", "clone"], env={"GH_TOKEN": "t"})
        assert env2.calls[0][2] == {"GH_TOKEN": "t"}
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join()


def test_harbor_env_runtime_timeout_maps_to_124_and_rejects_mounts() -> None:
    loop, t = _loop_in_thread()
    try:
        rt = HarborEnvRuntime(_FakeEnv(raise_exc=True), loop)
        res = rt.exec("H", ["sleep", "999"])
        assert res.returncode == 124
        with pytest.raises(RuntimeError, match="bind mounts"):
            rt.acquire(mounts=[BindMount(host_path="/h", container_path="/c")])
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join()


def test_harbor_env_runtime_sub_second_timeout_not_dropped() -> None:
    # A <1s timeout must round up to >=1, not collapse to 0→None (unbounded).
    loop, t = _loop_in_thread()
    try:
        env = _FakeEnv()
        HarborEnvRuntime(env, loop).exec("H", ["x"], timeout=0.4)
        assert env.calls[0][3] == 1  # timeout_sec passed to harbor's exec
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join()


def test_harbor_env_runtime_transport_error_maps_to_125() -> None:
    # A non-timeout backend error must map to rc125 (distinct from the rc124 timeout
    # path) so the timeout-vs-crash signal survives at the layer that produces it.
    loop, t = _loop_in_thread()
    try:
        rt = HarborEnvRuntime(_FakeEnv(exc=RuntimeError("grpc down")), loop)
        res = rt.exec("H", ["x"], timeout=5)
        assert res.returncode == 125 and "grpc down" in res.stderr
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join()


def test_harbor_env_runtime_coalesces_none_stdout() -> None:
    # Real harbor ExecResult.stdout/stderr are Optional[str]; must not leak None.
    loop, t = _loop_in_thread()
    try:
        res = HarborEnvRuntime(_FakeEnv(stdout=None), loop).exec("H", ["true"])
        assert res.stdout == "" and res.stderr == "" and res.returncode == 0
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join()


# --- MonetAgent.run: error/branch coverage -----------------------------------


def _ok() -> ExecResult:
    return ExecResult(returncode=0, stdout="", stderr="")


def test_monet_run_install_failure_reported() -> None:
    rt = _FakeRuntime([_ok(), ExecResult(returncode=1, stdout="", stderr="npm ERR!")])
    res = _monet_agent().run(Task(task_id="t"), TaskContext(image="img", agent_timeout_s=1800), runtime=rt)
    assert res.status is RolloutStatus.FAILED and "install failed" in res.error
    assert ("destroy", "H") in rt.calls


def test_monet_run_timeout_maps_to_error() -> None:
    rt = _FakeRuntime([_ok(), _ok(), ExecResult(returncode=124, stdout="", stderr="")])
    res = _monet_agent().run(Task(task_id="t"), TaskContext(image="img", agent_timeout_s=1800), runtime=rt)
    assert not res.resolved and res.error == "timeout after 1800.0s"


def test_monet_run_runtime_error_rc125() -> None:
    rt = _FakeRuntime([_ok(), _ok(), ExecResult(returncode=125, stdout="", stderr="grpc down")])
    res = _monet_agent().run(Task(task_id="t"), TaskContext(image="img", agent_timeout_s=1800), runtime=rt)
    assert not res.resolved and "runtime error" in res.error


def test_monet_run_token_env_missing_returns_error(monkeypatch) -> None:
    monkeypatch.delenv("NOPE_TOKEN", raising=False)
    rt = _FakeRuntime([])  # no exec should run — fails before clone
    res = _monet_agent({"token_env": "NOPE_TOKEN"}).run(
        Task(task_id="t"), TaskContext(image="img", agent_timeout_s=1800), runtime=rt
    )
    assert res.status is RolloutStatus.FAILED and "not set" in res.error
    assert [c[0] for c in rt.calls] == ["acquire", "destroy"]  # returned, not raised


def test_monet_run_forwards_env(monkeypatch) -> None:
    monkeypatch.setenv("HOST_KEY", "secretval")
    rt = _FakeRuntime([_ok(), _ok(), ExecResult(returncode=0, stdout=_invoke_stdout(), stderr="")])
    agent = _monet_agent({"forward_env": [["CONTAINER_KEY", "HOST_KEY"]]})
    agent.run(Task(task_id="t", problem_statement="go"), TaskContext(image="img", agent_timeout_s=1800), runtime=rt)
    invoke_env = _invoke_call(rt)[1]
    assert invoke_env["CONTAINER_KEY"] == "secretval" and invoke_env["MONET_PROMPT"] == "go"


# --- gateway routing rides in agent.config (monet_args + forward_env) ----------


def test_monet_gateway_via_config_not_model_block(monkeypatch) -> None:
    # The gateway (--provider) + creds come from agent.config (monet_args + forward_env),
    # NOT the model block — the harbor shim drops model-block details. `model.name`
    # only supplies --model, and it must NOT emit a second --provider.
    monkeypatch.setenv("GW_KEY", "secret")
    agent = bgl.agents.build(
        AgentSpec(
            name="monet",
            source=AgentSource(repo="https://x/r", ref="deadbeef"),
            model=ModelSpec(name="gpt-5.5"),
            config={"monet_args": ["--provider", "llm-gateway-express-local-proxy",
                                   "--output-format", "stream-json"],
                    "forward_env": [["GW_KEY", "GW_KEY"]]},
        )
    )
    rt = _FakeRuntime([_ok(), _ok(), ExecResult(returncode=0, stdout=_invoke_stdout(), stderr="")])
    agent.run(Task(task_id="t", problem_statement="go"), TaskContext(image="img", repo_path="/w", agent_timeout_s=1800), runtime=rt)
    invoke_cmd, invoke_env = _invoke_call(rt)
    assert invoke_env["GW_KEY"] == "secret"                              # forward_env forwarded
    assert "--provider llm-gateway-express-local-proxy" in invoke_cmd[2]  # from monet_args
    assert "--model gpt-5.5" in invoke_cmd[2]                            # from model.name
    assert invoke_cmd[2].count("--provider") == 1                        # no duplicate


def test_effort_config_prepends_effort_flag() -> None:
    # monet's reasoning effort is inert unless wired: the `effort` config key → --effort, and it
    # must survive as a config value (the harbor shim drops the model block, so the model's typed
    # reasoning_effort is lost on that path). Absent → no --effort (monet's default `none`).
    with_e = _monet_agent({"effort": "high"})._config("m", "bin/monet.js")
    assert with_e.monet_args[:2] == ("--effort", "high")
    assert "--effort" not in _monet_agent({})._config("m", "bin/monet.js").monet_args


def test_provider_config_prepends_flag_without_restating_defaults() -> None:
    # `provider` is the clean way to select the gateway: it prepends --provider to monet's
    # DEFAULT behavior flags, so a caller need not restate them (setting monet_args would
    # replace the defaults wholesale). Absent → no --provider (monet's direct-key default,
    # which dies with "No Anthropic API key configured" in a sealed container).
    from beagle.agents.monet._helpers import DEFAULT_MONET_ARGS

    with_p = _monet_agent({"provider": "llm-gateway-express-local-proxy"})._config("m", "bin/monet.js")
    assert with_p.monet_args == ("--provider", "llm-gateway-express-local-proxy", *DEFAULT_MONET_ARGS)
    without = _monet_agent({})._config("m", "bin/monet.js")
    assert "--provider" not in without.monet_args


def _combined(*, rc: str, patch: str, extra_lines: list[str]) -> str:
    return "\n".join(
        [f"MONET_RC={rc}", _monet._PATCH_SENTINEL_START, patch, _monet._PATCH_SENTINEL_END,
         _monet._MONET_OUT_SENTINEL_START, *extra_lines, _monet._MONET_OUT_SENTINEL_END]
    )


def test_monet_result_max_turns_with_patch_is_resolved() -> None:
    # Budget exhausted but a usable diff survives ⇒ resolved (documented rule).
    agent = _monet_agent()
    cfg = agent._config("m", "bin/monet.js")
    stdout = _combined(rc="1", patch="diff --git a/x b/x", extra_lines=['{"type":"max_turns_reached"}'])
    res = agent._result(Task(task_id="t"), cfg, 0, stdout, "")
    assert res.resolved and res.patch == "diff --git a/x b/x"


def test_monet_result_stream_error_when_no_patch() -> None:
    agent = _monet_agent()
    cfg = agent._config("m", "bin/monet.js")
    stdout = _combined(rc="0", patch="", extra_lines=['{"type":"stream_error","error":"Throttling"}'])
    res = agent._result(Task(task_id="t"), cfg, 0, stdout, "")
    assert not res.resolved and res.error == "stream_error: Throttling"


# --- identity config round-trip (forward_env/token_env flow through it) -------


def test_agent_identity_forwards_config() -> None:
    cfg = {"forward_env": [["A", "B"]], "token_env": "TOK", "max_turns": 3}
    ident = _agent_identity(_monet_agent(cfg))
    assert ident["config"] == cfg


def test_smoke_run_config_passthrough() -> None:
    # The canonical RunConfig's agent.config must reach the monet agent so an operator
    # can set the gateway provider (monet_args), install_cmd, etc.; the top-level model
    # becomes the agent's model.
    from beagle.config import RunConfig

    rc = RunConfig.from_dict({
        "model": {"name": "gpt-5.5", "provider": "llm-gateway-express-local-proxy"},
        "agent": {"name": "monet",
                  "source": {"repo": "https://x/r", "ref": "deadbeef"},
                  "config": {"monet_args": ["--provider", "llm-gateway-express-local-proxy"],
                             "container_path": "/opt/agent", "token_env": "GH_TOKEN"}},
        "benchmark": {"name": "terminal_bench_2_1", "task_ids": ["t1"]},
    })
    spec = rc.agent_spec()
    assert spec.source and spec.source.repo == "https://x/r" and spec.source.ref == "deadbeef"
    assert spec.model and spec.model.name == "gpt-5.5"  # top-level model applied to the agent
    ident = _agent_identity(bgl.agents.build(spec))
    assert ident["config"]["monet_args"] == ["--provider", "llm-gateway-express-local-proxy"]
    assert ident["config"]["container_path"] == "/opt/agent"
    assert ident["config"]["token_env"] == "GH_TOKEN"


# --- HarborHarness.rollout branch selection ----------------------------------


def test_rollout_empty_items_returns_empty() -> None:
    # Early return before any harbor import.
    from beagle.benchmarks.harness import HarborHarness

    assert list(HarborHarness().rollout(_monet_agent(), [], runtime=None, run_dir=Path("/j"))) == []


def test_rollout_generic_wraps_agent_in_shim(monkeypatch) -> None:
    pytest.importorskip("harbor")
    from beagle.benchmarks.harness import HarborHarness

    captured: dict = {}

    def _fake_run_job(self, items, agent_config, *, run_dir, parallelism, job_name=None,
                      retry=None, timeout_multiplier=1.0, resuming=False):
        captured["cfg"] = agent_config
        return []

    monkeypatch.setattr(HarborHarness, "_run_job", _fake_run_job)
    items = [(Task(task_id="t", extras={"harbor_task_dir": "/x"}), TaskContext(image=None))]
    list(HarborHarness().rollout(_monet_agent(), items, runtime=None, run_dir=Path("/j")))
    assert captured["cfg"].import_path == HarborHarness.SHIM_IMPORT_PATH
    assert captured["cfg"].kwargs["identity"]["agent"] == "monet"


def test_harbor_job_dir_named_after_benchmark(monkeypatch, tmp_path) -> None:
    """The harbor job dir is named after the benchmark (deterministic, resume-friendly),
    NOT a random beagle-<uuid> — so the native tree is <run_dir>/<benchmark>/<trial>/…."""
    pytest.importorskip("harbor")
    import harbor
    from harbor.models.trial.config import AgentConfig

    from beagle.benchmarks.harness import HarborHarness

    task_dir = tmp_path / "terminal-bench-2-1" / "adaptive-rejection-sampler"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('[environment]\ndocker_image = "img:1"\n')

    captured: dict = {}

    class _FakeJob:
        @classmethod
        async def create(cls, config):  # noqa: ANN001
            captured["config"] = config
            return cls()

        async def run(self):
            return SimpleNamespace(trial_results=[])

    monkeypatch.setattr(harbor, "Job", _FakeJob)

    items = [(Task(task_id="adaptive-rejection-sampler", benchmark="terminal_bench_2_1",
                   extras={"harbor_task_dir": str(task_dir)}), TaskContext(image=None))]
    HarborHarness()._run_job(items, AgentConfig(), run_dir=tmp_path / "out", parallelism=2)

    cfg = captured["config"]
    assert cfg.job_name == "terminal_bench_2_1"             # benchmark, not beagle-<uuid>
    assert Path(cfg.jobs_dir) == tmp_path / "out"
    # ⇒ harbor writes <run_dir>/terminal_bench_2_1/<trial>/… (native <job>/<trial>)


def test_harbor_completed_reads_native_tree(tmp_path) -> None:
    """Resume seam: HarborHarness.completed reads back finished trials from harbor's own
    result.json (the source of truth) — no house ledger. Reward/tokens/error map across."""
    import json

    from beagle.benchmarks.harness import HarborHarness

    td = tmp_path / "terminal_bench_2_1" / "bn-fit-modify__XYZ"
    td.mkdir(parents=True)
    (td / "result.json").write_text(json.dumps({
        "verifier_result": {"rewards": {"reward": 1.0}},
        "exception_info": None,
        "agent_result": {"n_input_tokens": 100, "n_output_tokens": 20},
    }))
    items = [
        (Task(task_id="bn-fit-modify", benchmark="terminal_bench_2_1"), TaskContext(image=None)),
        (Task(task_id="never-ran", benchmark="terminal_bench_2_1"), TaskContext(image=None)),
    ]
    done = HarborHarness().completed(items, run_dir=tmp_path)

    assert [r.task_id for r in done] == ["bn-fit-modify"]     # only the one with a result.json
    r = done[0]
    assert r.resolved is True and r.reward == 1.0
    assert r.tokens == {"prompt": 100, "completion": 20,       # from agent_result n_input/cache/output
                        "input_uncached": 100, "cache_read": 0, "cache_write": 0}
    assert r.artifact_dir == td


def test_harness_emits_atif_trajectory_post_job(tmp_path) -> None:
    """Trajectory emission happens POST-JOB in the harness (not in the shim, which runs
    before the cluster syncs the stream). Given a synced trial dir, the harness writes a
    valid ATIF trajectory.json and maps the instruction by (suffix-stripped) task id."""
    pytest.importorskip("harbor")
    import json
    from types import SimpleNamespace

    from harbor.models.trial.config import AgentConfig
    from harbor.utils.trajectory_validator import TrajectoryValidator

    from beagle.benchmarks.harness.drivers import _emit_trajectories

    agent_dir = tmp_path / "terminal_bench_2_1" / "bn-fit-modify__ABC" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "monet.stream.jsonl").write_text("\n".join([
        '{"type":"session_meta","session_id":"S"}',
        '{"type":"text_delta","text":"Fixing."}',
        '{"type":"turn_complete","turn":0,"stopReason":"end_turn"}',
        '{"type":"turn_done"}',
    ]))
    items = [(Task(task_id="bn-fit-modify", problem_statement="Fix the bug.",
                   benchmark="terminal_bench_2_1"), TaskContext(image=None))]
    ac = AgentConfig(import_path="x", kwargs={"identity": {
        "agent": "monet", "source": {"ref": "abc123"}, "model": "gpt-5.5"}})
    jr = SimpleNamespace(trial_results=[SimpleNamespace(trial_name="bn-fit-modify__ABC")])

    _emit_trajectories(jr, items, ac, run_dir=tmp_path, job_name="terminal_bench_2_1")

    tj = agent_dir / "trajectory.json"
    assert tj.exists() and TrajectoryValidator().validate(str(tj))
    d = json.loads(tj.read_text())
    assert d["agent"] == {"name": "monet", "version": "abc123", "model_name": "gpt-5.5"}
    assert d["steps"][0]["message"] == "Fix the bug."  # instruction mapped via stripped task_id


# --- the generic shim (harbor-gated: skips where harbor isn't installed) ------


def test_shim_run_bridges_to_agent_and_writes_context() -> None:
    pytest.importorskip("harbor")
    from beagle.benchmarks.harness._harbor_agent import BeagleInstalledAgent
    from beagle.types import TaskResult

    class _StubAgent:
        seen: dict = {}

        def run(self, task, task_ctx, *, runtime):  # noqa: ANN001
            _StubAgent.seen = {"repo_path": task_ctx.repo_path, "prompt": task.problem_statement}
            return TaskResult(task_id="x", tokens={"prompt": 7, "completion": 3}, patch="d")

    class _Env:
        default_user = None

        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):  # noqa: ANN001
            return SimpleNamespace(stdout="/work\n", stderr="", return_code=0)

    ident = {"agent": "monet", "source": {"repo": "https://x/r", "ref": "deadbeef",
             "entrypoint": "bin/monet.js"}, "config": {}, "model": "m"}
    shim = BeagleInstalledAgent(Path("/tmp/xa"), identity=ident)
    shim._agent = _StubAgent()

    from harbor.models.agent.context import AgentContext

    ctx = AgentContext()
    asyncio.run(shim.run("do it", _Env(), ctx))
    assert ctx.n_input_tokens == 7 and ctx.n_output_tokens == 3 and ctx.metadata["patch"] == "d"
    assert _StubAgent.seen == {"repo_path": "/work", "prompt": "do it"}


def test_shim_install_runs_git_bootstrap_as_root() -> None:
    pytest.importorskip("harbor")
    from beagle.benchmarks.harness import _harbor_agent

    ident = {"agent": "monet", "source": {"repo": "https://x/r", "ref": "deadbeef",
             "entrypoint": "bin/monet.js"}, "config": {}, "model": "m"}
    shim = _harbor_agent.BeagleInstalledAgent(Path("/tmp/xa"), identity=ident)
    calls: list = []

    async def _fake_root(environment, command=None, timeout_sec=None):  # noqa: ANN001
        calls.append((command, timeout_sec))

    shim.exec_as_root = _fake_root  # type: ignore[method-assign]
    asyncio.run(shim.install(object()))
    assert calls == [(_harbor_agent._GIT_BOOTSTRAP, 300)]

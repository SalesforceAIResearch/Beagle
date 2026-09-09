"""DeepSWE onboarding — pier-family benchmark (pier is a harbor fork).

The pier Job execution can't run without ``datacurve-pier`` installed, so these tests cover the
parts that don't need it: registration + wiring (PierHarness inherits the harbor driver, retargeted
at pier), the task source reading a pier ``task.toml`` + ``instruction.md``, and the framework
parametrization resolving pier's classes (with a fake pier package)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

import beagle as bgl
from beagle.benchmarks.base import BenchmarkSpec
from beagle.benchmarks.grader import InBandGrader
from beagle.benchmarks.harness import HarborHarness, PierHarness
from beagle.benchmarks.source import HarborCache


def test_deepswe_registered_with_pier_harness() -> None:
    b = bgl.benchmarks.get("deep-swe")
    assert b.name == "deep-swe"
    h = b.harness()
    # pier reuses the harbor Job driver, retargeted by class attrs — no duplicated harness code
    assert isinstance(h, PierHarness) and isinstance(h, HarborHarness)
    assert h.FRAMEWORK == "pier"
    assert h.ENV_IMPORT_PATH == "xrlenv_plugins.pier:XrlenvPierEnvironmentCluster"
    assert h.SHIM_IMPORT_PATH == "beagle.benchmarks.harness._pier_agent:BeaglePierAgent"
    local = b.harness_for_runtime("local")
    assert isinstance(local, PierHarness)
    assert local.ENV_IMPORT_PATH == "xrlenv_plugins.pier:XrlenvPierEnvironment"
    assert isinstance(b.grader(), InBandGrader)


def test_deepswe_source_reads_a_pier_task_dir(tmp_path) -> None:
    # a pier task dir has the same shape HarborCache reads: task.toml [environment].docker_image
    # + instruction.md — so the harbor cache source is reused unchanged.
    td = tmp_path / "deep-swe" / "abs-module-cache-flags"
    td.mkdir(parents=True)
    (td / "task.toml").write_text('[environment]\ndocker_image = "public.ecr.aws/x/img:v1"\n')
    (td / "instruction.md").write_text("Fix the module loader.")
    src = HarborCache("deep-swe", cache_name="deep-swe", cache_root=tmp_path)

    (t, c), = list(src.tasks(BenchmarkSpec(name="deep-swe")))
    assert t.task_id == "abs-module-cache-flags" and t.benchmark == "deep-swe"
    assert t.problem_statement == "Fix the module loader."
    assert c.image == "public.ecr.aws/x/img:v1"
    assert t.extras["harbor_task_dir"] == str(td)   # PierHarness reads this to build TaskConfig(path=)


def test_pier_harness_api_resolves_pier_classes(monkeypatch) -> None:
    # Framework parametrization: PierHarness pulls pier's Job/config from `pier` — and pier exposes
    # Job at `pier.job.Job` (not top-level), which _harness_api's fallback must handle.
    def _mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    _mod("pier")                                     # top-level: NO `Job` attr → forces the fallback
    job_mod = _mod("pier.job"); job_mod.Job = type("Job", (), {})          # type: ignore[attr-defined]
    _mod("pier.models"); _mod("pier.models.job"); _mod("pier.models.trial")
    jc = _mod("pier.models.job.config")
    jc.JobConfig = type("JobConfig", (), {}); jc.RetryConfig = type("RetryConfig", (), {})  # type: ignore[attr-defined]
    tc = _mod("pier.models.trial.config")
    for n in ("AgentConfig", "EnvironmentConfig", "TaskConfig"):
        setattr(tc, n, type(n, (), {}))

    api = PierHarness()._harness_api()
    assert api["Job"] is job_mod.Job                 # resolved via pier.job.Job fallback
    assert api["JobConfig"] is jc.JobConfig and api["TaskConfig"] is tc.TaskConfig
    assert api["EnvironmentConfig"] is tc.EnvironmentConfig


def test_harbor_harness_api_still_resolves_harbor() -> None:
    # the same parametrization must leave the harbor path intact. This one genuinely needs the
    # real package — it asserts the import resolves — so it SKIPS without the optional extra
    # rather than failing, which is what it did before.
    pytest.importorskip("harbor")
    api = HarborHarness()._harness_api()
    assert api["Job"].__module__.startswith("harbor")
    assert all(k in api for k in ("JobConfig", "RetryConfig", "AgentConfig", "EnvironmentConfig", "TaskConfig"))


def _import_pier_shim(monkeypatch):
    import importlib

    def _mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    _mod("pier"); _mod("pier.agents"); _mod("pier.agents.installed")
    _mod("pier.agents.installed.base").BaseInstalledAgent = type(  # type: ignore[attr-defined]
        "BaseInstalledAgent", (), {"__init__": lambda self, *a, **k: None})
    _mod("pier.agents.network").allowlist_from_urls = list  # type: ignore[attr-defined]
    _mod("pier.environments")
    _mod("pier.environments.base").BaseEnvironment = type("BaseEnvironment", (), {})  # type: ignore[attr-defined]
    _mod("pier.models"); _mod("pier.models.agent")
    _mod("pier.models.agent.context").AgentContext = type("AgentContext", (), {})  # type: ignore[attr-defined]
    monkeypatch.delitem(sys.modules, "beagle.benchmarks.harness._pier_agent", raising=False)
    return importlib.import_module("beagle.benchmarks.harness._pier_agent")


def test_pier_open_install_egress_helpers(monkeypatch) -> None:
    """Legacy configs retain their historical IP-vs-hostname routing decision."""
    m = _import_pier_shim(monkeypatch)

    # deep-swe's LLM-gateway local proxy is a bare IPv4 → open-install path, sealed to /32
    assert m._all_ipv4(["http://192.0.2.20:18088"]) is True
    assert m._run_egress_cidrs(["http://192.0.2.20:18088"]) == ["192.0.2.20/32"]
    # a hostname run host → stays on the Squid path (no cidrs, not all-ipv4)
    assert m._all_ipv4(["https://gateway.example.com/v1"]) is False
    assert m._run_egress_cidrs(["https://gateway.example.com/v1"]) == []
    # mixed or empty → not open-install
    assert m._all_ipv4(["http://192.0.2.20", "https://x.com"]) is False
    assert m._all_ipv4([]) is False


def test_pier_phase_split_allowlist_is_empty_but_legacy_is_unchanged(monkeypatch) -> None:
    m = _import_pier_shim(monkeypatch)
    fake_agent = type("Agent", (), {
        "network_hosts": lambda self: ["https://api.openai.com"],
        "install_hosts": lambda self: ["https://github.com", "https://registry.npmjs.org"],
    })()
    shim = m.BeaglePierAgent.__new__(m.BeaglePierAgent)
    shim._agent = fake_agent

    shim._phase_split_egress = True
    assert shim.network_allowlist() == []

    shim._phase_split_egress = False
    assert shim.network_allowlist() == [
        "https://api.openai.com", "https://github.com", "https://registry.npmjs.org",
    ]


@pytest.mark.asyncio
async def test_pier_resolves_pins_and_scopes_hostname_egress(monkeypatch) -> None:
    m = _import_pier_shim(monkeypatch)

    class Env:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def exec(self, command: str, **kwargs):
            self.calls.append((command, kwargs))
            if command.startswith("getent "):
                return SimpleNamespace(
                    return_code=0,
                    stdout="203.0.113.8 STREAM api.openai.com\n203.0.113.9 STREAM api.openai.com\n",
                    stderr=None,
                )
            return SimpleNamespace(return_code=0, stdout=None, stderr=None)

    env = Env()
    cidrs, ports = await m._resolve_and_pin_run_egress(
        env, ["https://api.openai.com/v1"],
    )

    assert cidrs == ["203.0.113.8/32", "203.0.113.9/32"]
    assert ports == (443,)
    assert env.calls[0][0] == "getent ahostsv4 api.openai.com"
    assert env.calls[1][1]["user"] == "root"
    assert "203.0.113.8 api.openai.com" in env.calls[1][0]


@pytest.mark.asyncio
async def test_pier_hostname_resolution_failure_aborts_before_seal(monkeypatch) -> None:
    m = _import_pier_shim(monkeypatch)

    class Env:
        async def exec(self, command: str, **kwargs):
            return SimpleNamespace(return_code=2, stdout="", stderr="not found")

    with pytest.raises(RuntimeError, match="could not resolve"):
        await m._resolve_and_pin_run_egress(Env(), ["https://missing.example"])


@pytest.mark.asyncio
async def test_pier_ip_and_empty_targets_need_no_dns_or_hosts_write(monkeypatch) -> None:
    m = _import_pier_shim(monkeypatch)

    class Env:
        async def exec(self, command: str, **kwargs):
            raise AssertionError(f"unexpected exec: {command}")

    assert await m._resolve_and_pin_run_egress(
        Env(), ["http://192.0.2.20:18088/v1"],
    ) == (["192.0.2.20/32"], (18088,))
    assert await m._resolve_and_pin_run_egress(Env(), []) == ([], None)


@pytest.mark.asyncio
async def test_pier_rejects_heterogeneous_ports_instead_of_broadening(monkeypatch) -> None:
    m = _import_pier_shim(monkeypatch)

    class Env:
        async def exec(self, command: str, **kwargs):
            raise AssertionError(f"unexpected exec: {command}")

    with pytest.raises(ValueError, match="heterogeneous endpoint ports"):
        await m._resolve_and_pin_run_egress(
            Env(), ["https://192.0.2.10", "http://192.0.2.20:8080"],
        )


@pytest.mark.parametrize(
    "host", ["https://bad host/v1", "https://x.test:99999", "ftp://x.test/file"],
)
def test_pier_rejects_malformed_run_targets(monkeypatch, host: str) -> None:
    m = _import_pier_shim(monkeypatch)
    with pytest.raises(ValueError, match="invalid RUN egress"):
        m._run_egress_targets([host])


@pytest.mark.asyncio
async def test_pier_phase_split_seals_before_agent_run(monkeypatch, tmp_path) -> None:
    m = _import_pier_shim(monkeypatch)
    state = {"sealed": False, "ran": False}

    class Agent:
        def network_hosts(self):
            return ["https://api.openai.com"]

        def run_in(self, handle, task, task_ctx, *, runtime):
            assert state["sealed"] is True
            state["ran"] = True
            return m.TaskResult(task_id=task.task_id, status=m.RolloutStatus.COMPLETED)

    class Env:
        def task_internet_disabled(self):
            return True

        async def exec(self, command: str, **kwargs):
            if command.startswith("getent "):
                return SimpleNamespace(
                    return_code=0, stdout="203.0.113.8 STREAM api.openai.com\n", stderr=None,
                )
            return SimpleNamespace(return_code=0, stdout=None, stderr=None)

        async def apply_egress(self, cidrs, *, ports=None):
            assert cidrs == ["203.0.113.8/32"]
            assert ports == (443,)
            state["sealed"] = True

    shim = m.BeaglePierAgent.__new__(m.BeaglePierAgent)
    shim._agent = Agent()
    shim._phase_split_egress = True
    shim._install_error = None
    shim._task_ctx = None
    shim._logs_dir = tmp_path / "agent"
    shim._handle = object()
    shim._runtime = object()

    await shim.run("fix it", Env(), object())

    assert state == {"sealed": True, "ran": True}


@pytest.mark.asyncio
async def test_pier_phase_split_leaves_online_task_open(monkeypatch, tmp_path) -> None:
    m = _import_pier_shim(monkeypatch)
    state = {"ran": False}

    class Agent:
        def network_hosts(self):
            return ["https://api.openai.com"]

        def run_in(self, handle, task, task_ctx, *, runtime):
            state["ran"] = True
            return m.TaskResult(task_id=task.task_id, status=m.RolloutStatus.COMPLETED)

    class Env:
        def task_internet_disabled(self):
            return False

        async def apply_egress(self, cidrs, *, ports=None):
            raise AssertionError("online task must not be sealed")

    shim = m.BeaglePierAgent.__new__(m.BeaglePierAgent)
    shim._agent = Agent()
    shim._phase_split_egress = True
    shim._install_error = None
    shim._task_ctx = None
    shim._logs_dir = tmp_path / "agent"
    shim._handle = object()
    shim._runtime = object()

    await shim.run("fix it", Env(), object())

    assert state["ran"] is True

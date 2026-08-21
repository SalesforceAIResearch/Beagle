"""DeepSWE onboarding — pier-family benchmark (pier is a harbor fork).

The pier Job execution can't run without ``datacurve-pier`` installed, so these tests cover the
parts that don't need it: registration + wiring (PierHarness inherits the harbor driver, retargeted
at pier), the task source reading a pier ``task.toml`` + ``instruction.md``, and the framework
parametrization resolving pier's classes (with a fake pier package)."""

from __future__ import annotations

import sys
import types

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
    # the same parametrization must leave the harbor path intact (harbor IS installed here)
    api = HarborHarness()._harness_api()
    assert api["Job"].__module__.startswith("harbor")
    assert all(k in api for k in ("JobConfig", "RetryConfig", "AgentConfig", "EnvironmentConfig", "TaskConfig"))


def test_pier_open_install_egress_helpers(monkeypatch) -> None:
    """The IP-vs-hostname decision that routes a trial onto pier's open-install path (direct install
    + iptables run-seal) vs. the Squid domain-filter path. Imports the shim behind a fake `pier`."""
    import importlib

    def _mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    _mod("pier"); _mod("pier.agents"); _mod("pier.agents.installed")
    _mod("pier.agents.installed.base").BaseInstalledAgent = type(  # type: ignore[attr-defined]
        "BaseInstalledAgent", (), {"__init__": lambda self, *a, **k: None})
    _mod("pier.environments")
    _mod("pier.environments.base").BaseEnvironment = type("BaseEnvironment", (), {})  # type: ignore[attr-defined]
    _mod("pier.models"); _mod("pier.models.agent")
    _mod("pier.models.agent.context").AgentContext = type("AgentContext", (), {})  # type: ignore[attr-defined]
    monkeypatch.delitem(sys.modules, "beagle.benchmarks.harness._pier_agent", raising=False)
    m = importlib.import_module("beagle.benchmarks.harness._pier_agent")

    # deep-swe's LLM-gateway local proxy is a bare IPv4 → open-install path, sealed to /32
    assert m._all_ipv4(["http://10.0.173.227:18088"]) is True
    assert m._run_egress_cidrs(["http://10.0.173.227:18088"]) == ["10.0.173.227/32"]
    # a hostname run host → stays on the Squid path (no cidrs, not all-ipv4)
    assert m._all_ipv4(["https://gateway.example.com/v1"]) is False
    assert m._run_egress_cidrs(["https://gateway.example.com/v1"]) == []
    # mixed or empty → not open-install
    assert m._all_ipv4(["http://10.0.173.227", "https://x.com"]) is False
    assert m._all_ipv4([]) is False

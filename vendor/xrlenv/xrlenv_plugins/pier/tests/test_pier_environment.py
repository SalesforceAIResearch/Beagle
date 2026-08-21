"""Unit tests for the pier framework adapter (``xrlenv_plugins.pier``).

Covers the pure logic + the construction-time / resolution seams that don't need
a live cluster or docker daemon:

- ``_parse_dockerfile_from_ref`` (the verifier base-image ``FROM`` parser);
- ``type()`` (pier's abstract str identifier);
- ``capabilities`` (the cluster overrides: ``mounted=False``,
  ``preinstall_agents=False``, ``filtered_egress=False`` — §0.5/§2/§4b);
- ``agent_install_spec`` clearing in cluster mode (§0.5) vs kept in LocalDocker;
- the separate-verifier image resolution (``_is_verifier_session`` /
  ``_verifier_base_image`` / ``_resolve_image_ref`` — §2.5).

The cluster acquire/exec/transfer paths need a control plane + docker and are
exercised by the live G1 gate, not here.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.task.config import EnvironmentConfig as TaskEnvCfg
from pier.models.trial.paths import TrialPaths
from xrlenv_plugins.pier.environment import (
    XrlenvPierEnvironment,
    XrlenvPierEnvironmentCluster,
    _parse_dockerfile_from_ref,
)

# ── Pure: Dockerfile FROM parser ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # plain ref
        ("FROM public.ecr.aws/d3j8x8q7/swe:kh-v1.1", "public.ecr.aws/d3j8x8q7/swe:kh-v1.1"),
        # --platform flag + AS stage dropped
        ("FROM --platform=linux/amd64 ubuntu:24.04 AS base", "ubuntu:24.04"),
        # quoted
        ('FROM "quoted/ref:tag"', "quoted/ref:tag"),
        ("FROM 'single/quoted:tag'", "single/quoted:tag"),
        # lowercase keyword + leading whitespace
        ("   from  busybox:1.36  ", "busybox:1.36"),
        # first FROM wins (multi-stage)
        ("FROM base:1 AS a\nRUN x\nFROM final:2 AS b", "base:1"),
        # unresolved ARG / ${..} -> None (caller falls back to parent task.toml)
        ("ARG BASE\nFROM $BASE", None),
        ("FROM ${REGISTRY}/img:tag", None),
        # comments / no FROM
        ("# just a comment\nRUN echo hi", None),
        ("", None),
        # a commented-out FROM is not matched (regex requires FROM at line start-ish)
        ("#FROM ignored:tag\nFROM real:tag", "real:tag"),
    ],
)
def test_parse_dockerfile_from_ref(text: str, expected: str | None) -> None:
    assert _parse_dockerfile_from_ref(text) == expected


# ── Instance construction helpers ─────────────────────────────────────────────


def _make_env_dir(tmp: Path, *, dockerfile: str = "FROM scratch\n") -> Path:
    """A minimal valid pier environment dir (pier's ``_validate_definition``
    requires a ``Dockerfile`` or ``docker-compose.yaml``)."""
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "Dockerfile").write_text(dockerfile)
    return tmp


def _cluster(
    env_dir: Path,
    *,
    session_id: str = "trial-abc",
    docker_image: str | None = None,
    **kwargs: object,
) -> XrlenvPierEnvironmentCluster:
    return XrlenvPierEnvironmentCluster(
        environment_dir=env_dir,
        environment_name="deepswe-fastapi",
        session_id=session_id,
        trial_paths=TrialPaths(trial_dir=Path(tempfile.mkdtemp())),
        task_env_config=TaskEnvCfg(docker_image=docker_image),
        **kwargs,
    )


# ── type() + capabilities ─────────────────────────────────────────────────────


def test_type_is_stable_str() -> None:
    assert XrlenvPierEnvironmentCluster.type() == "xrlenv-cluster"
    assert XrlenvPierEnvironment.type() == "xrlenv-cluster"


def test_capabilities_cluster_overrides(tmp_path: Path) -> None:
    env = _cluster(_make_env_dir(tmp_path / "environment"), docker_image="i:t")
    caps = env.capabilities
    # mounted=False forces pier's Verifier to round-trip the reward file through
    # download_dir (the whole reason the separate-verifier reward reaches the host).
    assert caps.mounted is False
    # No preinstalled-agent image is built on the cluster -> pier must install at
    # runtime via exec (paired with clearing agent_install_spec).
    assert caps.preinstall_agents is False
    # Squid egress proxy IS implemented (S6) -> an offline task + agent allowlist
    # runs behind it; construction of such a task must not be rejected.
    assert caps.filtered_egress is True
    assert caps.disable_internet is True
    assert caps.gpus is False
    assert caps.windows is False


# ── agent_install_spec: cleared on cluster, kept on LocalDocker ────────────────


def _spec() -> AgentInstallSpec:
    return AgentInstallSpec(
        agent_name="claude-code",
        steps=[InstallStep(run="echo install")],
    )


def test_cluster_clears_agent_install_spec(tmp_path: Path) -> None:
    env = _cluster(
        _make_env_dir(tmp_path / "environment"),
        docker_image="i:t",
        agent_install_spec=_spec(),
    )
    # Cluster must drop it so pier runs the agent's runtime install() via exec
    # instead of assuming a (nonexistent) preinstalled image (§0.5).
    assert env.agent_install_spec is None


def test_localdocker_keeps_agent_install_spec(tmp_path: Path) -> None:
    spec = _spec()
    env = XrlenvPierEnvironment(
        environment_dir=_make_env_dir(tmp_path / "environment"),
        environment_name="deepswe-fastapi",
        session_id="trial-abc",
        trial_paths=TrialPaths(trial_dir=Path(tempfile.mkdtemp())),
        task_env_config=TaskEnvCfg(docker_image="i:t"),
        agent_install_spec=spec,
    )
    # LocalDocker CAN build a preinstalled image -> it keeps the spec.
    assert env.agent_install_spec is spec


# ── separate-verifier detection + image resolution (§2.5) ─────────────────────


def test_is_verifier_session(tmp_path: Path) -> None:
    env_dir = _make_env_dir(tmp_path / "environment")
    assert _cluster(env_dir, session_id="t__verifier__trial")._is_verifier_session()
    assert not _cluster(env_dir, session_id="t-agent-run")._is_verifier_session()


def test_verifier_base_image_from_dockerfile(tmp_path: Path) -> None:
    # verifier build context == the task's tests/ dir; its Dockerfile FROM is the base.
    tests = _make_env_dir(
        tmp_path / "tests",
        dockerfile="FROM public.ecr.aws/x/base:v1.1\nCOPY test.sh /tests/test.sh\n",
    )
    env = _cluster(tests, session_id="t__verifier__trial")
    assert env._verifier_base_image() == "public.ecr.aws/x/base:v1.1"


def test_verifier_base_image_falls_back_to_parent_task_toml(tmp_path: Path) -> None:
    # ARG-templated FROM -> unresolved -> fall back to parent task.toml docker_image.
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        '[environment]\ndocker_image = "public.ecr.aws/x/base:from-toml"\n',
    )
    tests = _make_env_dir(task_dir / "tests", dockerfile="ARG B\nFROM $B\n")
    env = _cluster(tests, session_id="t__verifier__trial")
    assert env._verifier_base_image() == "public.ecr.aws/x/base:from-toml"


def test_verifier_base_image_none_when_unresolvable(tmp_path: Path) -> None:
    tests = _make_env_dir(tmp_path / "tests", dockerfile="ARG B\nFROM $B\n")
    env = _cluster(tests, session_id="t__verifier__trial")
    assert env._verifier_base_image() is None


def test_resolve_image_ref_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_dir = _make_env_dir(tmp_path / "environment")

    # 1. docker_image on the (agent) env wins.
    agent = _cluster(env_dir, session_id="trial-abc", docker_image="ecr/agent:tag")
    assert agent._resolve_image_ref() == "ecr/agent:tag"

    # 2. verifier session, no docker_image -> resolve from tests/Dockerfile FROM.
    tests = _make_env_dir(
        tmp_path / "tests", dockerfile="FROM public.ecr.aws/x/base:v1.1\n",
    )
    ver = _cluster(tests, session_id="t__verifier__trial")
    assert ver._resolve_image_ref() == "public.ecr.aws/x/base:v1.1"

    # 3. non-verifier, no docker_image, nothing resolvable -> hb__<env> fallback.
    plain = _cluster(env_dir, session_id="trial-plain")
    assert plain._resolve_image_ref() == "hb__deepswe-fastapi"

    # 4. XRLENV_PIER_IMAGE_TEMPLATE overrides everything.
    monkeypatch.setenv("XRLENV_PIER_IMAGE_TEMPLATE", "reg:5011/deep-swe/{task_id}:main")
    # task_id == the environment_dir's PARENT name.
    templated = _cluster(env_dir, session_id="trial-abc", docker_image="ecr/ignored:tag")
    assert templated._resolve_image_ref() == f"reg:5011/deep-swe/{env_dir.parent.name}:main"


# ── _maybe_upload_verifier_tests is a no-op for a non-verifier session ─────────


def test_maybe_upload_verifier_tests_noop_for_agent(tmp_path: Path) -> None:
    env = _cluster(_make_env_dir(tmp_path / "environment"), session_id="trial-agent")

    async def _boom(*_a: object, **_k: object) -> None:  # pragma: no cover - must not run
        raise AssertionError("exec must not be called for a non-verifier session")

    env.exec = _boom  # type: ignore[method-assign]
    # Should return immediately without touching exec/upload.
    asyncio.run(env._maybe_upload_verifier_tests())


# ── egress-proxy capability (§4b / S6) ────────────────────────────────────────

from pier.models.agent.network import NetworkAllowlist  # noqa: E402
from xrlenv_plugins.pier.environment import (  # noqa: E402
    _EGRESS_PROXY_IMAGE,
    _egress_proxy_runtime_command,
    build_egress_proxy_compose,
)


def test_egress_proxy_runtime_command_installs_squid_no_shebang() -> None:
    cmd = _egress_proxy_runtime_command()
    assert not cmd.startswith("#!")  # shebang stripped (we run under bash -lc)
    assert "apt-get install" in cmd and "squid" in cmd  # runtime install (no build)
    # pier's squid bootstrap body is reused verbatim -> the allowlist policy is there
    assert "allowed_domains" in cmd and "dstdomain" in cmd


def test_build_egress_proxy_compose_shape() -> None:
    compose, images = build_egress_proxy_compose(
        main_ref="public.ecr.aws/x/base:v1",
        main_command=["sh", "-c", "sleep infinity"],
        domains=["api.anthropic.com", ".openai.com"],
        token="TOK",
    )
    svcs = compose["services"]
    assert set(svcs) == {"main", "pier-egress-proxy"}
    # main: internal-only (no direct egress), waits for the proxy to be healthy
    assert svcs["main"]["networks"] == ["pier-egress-internal"]
    assert svcs["main"]["image"] == "public.ecr.aws/x/base:v1"
    assert svcs["main"]["depends_on"]["pier-egress-proxy"]["condition"] == "service_healthy"
    # proxy: mirror-pullable base (no build/push), on both nets, allowlist via env
    px = svcs["pier-egress-proxy"]
    assert px["image"] == _EGRESS_PROXY_IMAGE == "ubuntu:24.04"
    assert px["networks"] == ["pier-egress-internal", "default"]
    assert px["environment"]["PROXY_TOKEN"] == "TOK"
    assert px["environment"]["ALLOWLIST_DOMAINS"] == ".openai.com,api.anthropic.com"
    assert px["command"][0:2] == ["bash", "-lc"]
    # internal network flag + ensure-present images
    assert compose["networks"]["pier-egress-internal"]["internal"] is True
    assert images == ["public.ecr.aws/x/base:v1", "ubuntu:24.04"]


def _cluster_net(
    env_dir: Path, *, allow_internet: bool, domains: list[str] | None,
) -> XrlenvPierEnvironmentCluster:
    kwargs: dict[str, object] = {}
    if domains is not None:
        kwargs["network_allowlist"] = NetworkAllowlist(domains=domains)
    return XrlenvPierEnvironmentCluster(
        environment_dir=env_dir,
        environment_name="deepswe-fastapi",
        session_id="trial-abc",
        trial_paths=TrialPaths(trial_dir=Path(tempfile.mkdtemp())),
        task_env_config=TaskEnvCfg(docker_image="i:t", allow_internet=allow_internet),
        **kwargs,
    )


def test_egress_domains_gate(tmp_path: Path) -> None:
    env_dir = _make_env_dir(tmp_path / "environment")
    # offline + allowlist -> proxy domains
    e1 = _cluster_net(env_dir, allow_internet=False, domains=["api.anthropic.com"])
    assert e1._egress_domains() == ["api.anthropic.com"]
    # online + allowlist -> no proxy (task can reach the internet directly)
    e2 = _cluster_net(env_dir, allow_internet=True, domains=["api.anthropic.com"])
    assert e2._egress_domains() is None
    # offline + no allowlist (the OracleAgent path) -> no proxy
    e3 = _cluster_net(env_dir, allow_internet=False, domains=None)
    assert e3._egress_domains() is None
    e4 = _cluster_net(env_dir, allow_internet=False, domains=[])
    assert e4._egress_domains() is None


def test_agent_process_env_injects_only_when_proxy_set(tmp_path: Path) -> None:
    # pier's inherited agent_process_env injects self._egress_proxy_env; default {} = no-op.
    env = _cluster(_make_env_dir(tmp_path / "environment"), docker_image="i:t")
    assert env._egress_proxy_env == {}  # pier default
    assert env.agent_process_env({"A": "1"}) == {"A": "1"}  # passthrough
    # simulate the egress path having set it
    env._egress_proxy_env = {"HTTP_PROXY": "http://agent:t@pier-egress-proxy:8080"}
    out = env.agent_process_env({"A": "1"})
    assert out["HTTP_PROXY"] == "http://agent:t@pier-egress-proxy:8080"
    assert out["A"] == "1"

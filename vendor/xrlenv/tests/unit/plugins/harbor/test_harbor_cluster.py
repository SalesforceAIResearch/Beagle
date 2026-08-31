"""Daemon-free unit tests for ``XrlenvHarborEnvironmentCluster``
(P1.7.C.1).

Lives under ``tests/unit/`` for the same sys.path-collision reason
as ``test_harbor_plugin_shape.py`` — see that file's docstring.

Strategy: stub harbor's ``DockerEnvironment.__init__`` chain via
``__new__`` + manual attribute set, so we don't need a real
docker-compose project on disk. The cluster overrides we exercise
only touch a small attribute set (``_xrlenv_session``,
``_xrlenv_client``, ``task_env_config``, ``environment_name``,
``session_id``, ``_persistent_env``, ``default_user``, ``logger``,
``_keep_containers``).
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import io
import logging
import os
import tarfile
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml


def _harbor_available() -> bool:
    try:
        import harbor  # noqa: F401
        from harbor.environments.docker.docker import DockerEnvironment  # noqa: F401
    except ImportError:
        return False
    return True


def _harbor_has_network_mode() -> bool:
    """Newer harbor models the internet contract via
    ``EnvironmentConfig.network_mode`` + a ``NetworkMode`` enum; harbor
    0.8.x (the version pinned by ``xrlenv[terminal-bench-2]``) has neither
    and expresses it through ``allow_internet`` instead. Assertions on the
    newer field only apply where it's actually present — the 0.8.x path is
    covered by ``..._falls_back_to_allow_internet_on_old_harbor``.
    """
    try:
        from harbor.models.task.config import EnvironmentConfig, NetworkMode  # noqa: F401
    except ImportError:
        return False
    return "network_mode" in getattr(EnvironmentConfig, "model_fields", {})


pytestmark = pytest.mark.skipif(
    not _harbor_available(),
    reason="harbor not installed (pip install 'xrlenv[terminal-bench-2]')",
)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level: subclass + property contract.
# ──────────────────────────────────────────────────────────────────────────────


def test_cluster_subclasses_local_subclasses_docker() -> None:
    import harbor
    from harbor.environments.docker.docker import DockerEnvironment
    from xrlenv_plugins.harbor import (
        XrlenvHarborEnvironment,
        XrlenvHarborEnvironmentCluster,
    )

    assert issubclass(XrlenvHarborEnvironmentCluster, XrlenvHarborEnvironment)
    assert issubclass(XrlenvHarborEnvironmentCluster, DockerEnvironment)
    assert issubclass(XrlenvHarborEnvironmentCluster, harbor.BaseEnvironment)


def test_cluster_is_mounted_is_false() -> None:
    """Pinning ``is_mounted=False`` is load-bearing — harbor's trial
    driver branches on this property to switch from bind-mount reads
    to ``download_dir`` calls. If a refactor flips this to True,
    harbor would silently try to read host paths that don't exist on
    the consumer (the bind would be on the node)."""
    from xrlenv_plugins.harbor import XrlenvHarborEnvironmentCluster

    inst = XrlenvHarborEnvironmentCluster.__new__(XrlenvHarborEnvironmentCluster)
    # capabilities now consults _can_enforce_egress (task_env_config.env +
    # _xrlenv_kwargs) for the network flags — supply the minimal stub attrs.
    inst.task_env_config = SimpleNamespace(env={})
    inst._xrlenv_kwargs = {}
    # harbor 0.13 replaced the legacy ``is_mounted`` property with
    # ``capabilities.mounted`` (which the cluster class overrides to False).
    assert inst.capabilities.mounted is False


def test_cluster_can_disable_internet_is_true() -> None:
    """Cluster mode CAN disable internet, so it truthfully advertises
    ``capabilities.disable_internet=True``. Otherwise harbor's
    ``_validate_internet_config`` refuses every ``allow_internet=False`` task
    with ``allow_internet=False is not supported by ... DOCKER environment``.

    The container is now acquired OPEN (install/bootstrap needs network); an
    offline task's egress is restricted AFTER install by the consumer's Trial
    via ``apply_egress`` (open-setup→tighten). The capability stays truthful —
    the env CAN disable, now post-install rather than at acquire."""
    from xrlenv_plugins.harbor import XrlenvHarborEnvironmentCluster

    inst = XrlenvHarborEnvironmentCluster.__new__(XrlenvHarborEnvironmentCluster)
    inst.task_env_config = SimpleNamespace(env={})
    inst._xrlenv_kwargs = {}
    assert inst.capabilities.disable_internet is True


def test_cluster_task_internet_disabled_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``task_internet_disabled`` is the offline decision the consumer Trial
    reads (post-install) to decide whether to restrict egress — derived from
    the same version-tolerant ``_network_mode_for_task`` logic (``"none"`` ⇔
    offline) so the allow_internet reading lives in one place.

    Driven via the harbor-0.8.x ``allow_internet`` path (the installed harbor),
    matching ``..._falls_back_to_allow_internet_on_old_harbor``: patch
    ``NetworkMode`` to ``None`` and feed an ``allow_internet``-only config."""
    from types import SimpleNamespace

    import xrlenv_plugins.harbor.environment as env_mod
    from xrlenv_plugins.harbor import XrlenvHarborEnvironmentCluster

    monkeypatch.setattr(env_mod, "NetworkMode", None)
    inst = XrlenvHarborEnvironmentCluster.__new__(XrlenvHarborEnvironmentCluster)

    # allow_internet=True (default) → internet on → not restricted.
    inst.task_env_config = SimpleNamespace(allow_internet=True)  # type: ignore[attr-defined]
    assert inst.task_internet_disabled() is False
    # allow_internet=False → offline → restrict.
    inst.task_env_config = SimpleNamespace(allow_internet=False)  # type: ignore[attr-defined]
    assert inst.task_internet_disabled() is True


@pytest.mark.skipif(
    not _harbor_has_network_mode(),
    reason="installed harbor (0.8.x) has no EnvironmentConfig.network_mode; "
    "the 0.8.x contract is covered by "
    "test_cluster_network_mode_falls_back_to_allow_internet_on_old_harbor",
)
def test_cluster_network_mode_follows_network_mode() -> None:
    """``_network_mode_for_task`` keys off the resolved ``network_mode``,
    not the deprecated ``allow_internet`` flag.

    Driven off a *real* ``EnvironmentConfig`` (not a stub) so the test
    exercises the same field upstream actually populates — the regression in
    #31 keyed off ``allow_internet`` (``None`` on every modern task) and ran
    every internet-on task with ``--network none``. The default-config case
    (``network_mode`` unset → ``public``) is the load-bearing one that was
    missing; it MUST leave internet on.
    """
    from harbor.models.task.config import EnvironmentConfig
    from xrlenv_plugins.harbor import XrlenvHarborEnvironmentCluster

    inst = XrlenvHarborEnvironmentCluster.__new__(XrlenvHarborEnvironmentCluster)

    # Default config: no network_mode, no allow_internet → public → internet on.
    inst.task_env_config = EnvironmentConfig()  # type: ignore[attr-defined]
    assert inst.task_env_config.network_mode.value == "public"
    assert inst.task_env_config.allow_internet is None  # the deprecated trap
    assert inst._network_mode_for_task() is None

    # Explicit public → bridge (internet on).
    inst.task_env_config = EnvironmentConfig(network_mode="public")  # type: ignore[attr-defined]
    assert inst._network_mode_for_task() is None

    # no-network → "none" (loopback-only, external internet blocked).
    inst.task_env_config = EnvironmentConfig(network_mode="no-network")  # type: ignore[attr-defined]
    assert inst._network_mode_for_task() == "none"

    # allowlist errs open (no raw-docker allowlist primitive yet) — bridge, not none.
    inst.task_env_config = EnvironmentConfig(  # type: ignore[attr-defined]
        network_mode="allowlist", allowed_hosts=["example.com"]
    )
    assert inst._network_mode_for_task() is None


def test_cluster_network_mode_falls_back_to_allow_internet_on_old_harbor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """harbor 0.8.x has no ``network_mode``/``NetworkMode``; the plug-in must
    still load and key off ``allow_internet`` (a plain bool, default True).

    Simulate that env: patch the module-level ``NetworkMode`` to ``None`` (as
    the guarded import sets it when the symbol is absent) and feed a config that
    exposes only ``allow_internet`` — no ``network_mode`` attribute. Regression
    guard for the import-time crash where keying a hard ``import NetworkMode``
    off newer harbor broke every coding-bench/turing task on harbor 0.8.0.
    """
    from types import SimpleNamespace

    import xrlenv_plugins.harbor.environment as env_mod
    from xrlenv_plugins.harbor import XrlenvHarborEnvironmentCluster

    monkeypatch.setattr(env_mod, "NetworkMode", None)
    inst = XrlenvHarborEnvironmentCluster.__new__(XrlenvHarborEnvironmentCluster)

    # allow_internet=True (0.8.x default) → bridge (internet on).
    inst.task_env_config = SimpleNamespace(allow_internet=True)  # type: ignore[attr-defined]
    assert inst._network_mode_for_task() is None
    # allow_internet=False → "none" (loopback-only).
    inst.task_env_config = SimpleNamespace(allow_internet=False)  # type: ignore[attr-defined]
    assert inst._network_mode_for_task() == "none"


# ──────────────────────────────────────────────────────────────────────────────
# Pure helpers.
# ──────────────────────────────────────────────────────────────────────────────


def test_sanitize_image_tag_lowercases_and_replaces_invalid_chars() -> None:
    from xrlenv_plugins.harbor.environment import _sanitize_image_tag

    assert _sanitize_image_tag("hb__Fix-Git") == "hb__fix-git"
    assert _sanitize_image_tag("hb__build POV ray") == "hb__build-pov-ray"
    assert _sanitize_image_tag("HB__DNA/INSERT") == "hb__dna-insert"


def test_sanitize_container_name_strips_leading_punctuation() -> None:
    from xrlenv_plugins.harbor.environment import _sanitize_container_name

    assert _sanitize_container_name("trial.fix-git__abc") == "trial.fix-git__abc"
    assert _sanitize_container_name("-leading-dash") == "leading-dash"
    assert _sanitize_container_name("___") == "harbor-cluster-session"
    assert _sanitize_container_name("foo/bar:baz") == "foo-bar-baz"


def test_tar_one_file_round_trips(tmp_path: Path) -> None:
    from xrlenv_plugins.harbor.environment import _tar_one_file, _untar_one_file

    src = tmp_path / "input.txt"
    src.write_bytes(b"hello cluster")
    tarball = _tar_one_file(src, arcname="renamed.txt")
    dst = tmp_path / "out" / "landed.txt"
    _untar_one_file(tarball, dst)
    assert dst.read_bytes() == b"hello cluster"


def test_tar_dir_contents_and_untar_round_trip(tmp_path: Path) -> None:
    from xrlenv_plugins.harbor.environment import (
        _tar_dir_contents,
        _untar_dir_contents,
    )

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_bytes(b"alpha")
    (src / "nested").mkdir()
    (src / "nested" / "b.txt").write_bytes(b"beta")

    tarball = _tar_dir_contents(src)

    # Wrap the children-only tarball into a docker-style get_archive
    # tarball (rooted at ``src/...``) so _untar_dir_contents can be
    # exercised on the real shape it sees from the wire.
    wire_buf = io.BytesIO()
    with (
        tarfile.open(fileobj=wire_buf, mode="w") as out,
        tarfile.open(fileobj=io.BytesIO(tarball), mode="r") as inp,
    ):
        for member in inp.getmembers():
            f = inp.extractfile(member) if member.isfile() else None
            member.name = f"src/{member.name}"
            out.addfile(member, f)

    dst = tmp_path / "dst"
    _untar_dir_contents(wire_buf.getvalue(), dst)
    assert (dst / "a.txt").read_bytes() == b"alpha"
    assert (dst / "nested" / "b.txt").read_bytes() == b"beta"


def test_tar_relaxes_modes_for_sysbox_uploads(tmp_path: Path) -> None:
    """``world_accessible=True`` (the sysbox path) makes uploaded files world
    read/exec and dirs world rwx, root-owned. Load-bearing: under sysbox,
    put_archive can't id-shift, so files land owned by an unmapped uid (65534)
    that container-root can't chmod — the exec bit must be set in the tar or the
    oracle's solve.sh dies with 126. See _relax_modes."""
    from xrlenv_plugins.harbor.environment import (
        _tar_dir_contents,
        _tar_one_file,
    )

    src = tmp_path / "solution"
    src.mkdir()
    solve = src / "solve.sh"
    solve.write_bytes(b"#!/bin/bash\necho hi\n")
    solve.chmod(0o644)  # non-executable source, like the real corpus

    # Single-file upload: exec bit is added.
    one = _tar_one_file(solve, arcname="solve.sh", world_accessible=True)
    with tarfile.open(fileobj=io.BytesIO(one), mode="r") as tf:
        m = tf.getmember("solve.sh")
        assert m.mode & 0o111, "expected exec bits on the sysbox-uploaded file"
        assert m.mode & 0o444, "expected read bits"
        assert m.uid == 0 and m.gid == 0

    # Dir upload: files get o+rx, dirs get 0o777.
    (src / "sub").mkdir()
    (src / "sub" / "data.txt").write_bytes(b"x")
    (src / "sub" / "data.txt").chmod(0o600)
    dir_tar = _tar_dir_contents(src, world_accessible=True)
    with tarfile.open(fileobj=io.BytesIO(dir_tar), mode="r") as tf:
        for m in tf.getmembers():
            if m.isdir():
                assert m.mode == 0o777, f"{m.name}: dirs should be 0o777"
            else:
                assert m.mode & 0o555 == 0o555, f"{m.name}: files need world r-x"
            assert m.uid == 0 and m.gid == 0


def test_tar_preserves_modes_without_sysbox_flag(tmp_path: Path) -> None:
    """Default (runc path): modes/ownership are preserved verbatim — the working
    non-sysbox upload path is byte-for-byte unchanged."""
    from xrlenv_plugins.harbor.environment import _tar_one_file

    src = tmp_path / "solve.sh"
    src.write_bytes(b"#!/bin/bash\n")
    src.chmod(0o640)
    tar = _tar_one_file(src, arcname="solve.sh")  # world_accessible defaults False
    with tarfile.open(fileobj=io.BytesIO(tar), mode="r") as tf:
        m = tf.getmember("solve.sh")
        assert m.mode == 0o640
        assert not (m.mode & 0o001)  # no world-exec added


def test_untar_one_file_rejects_multi_entry_tarball(tmp_path: Path) -> None:
    from xrlenv_plugins.harbor.environment import _untar_one_file

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, payload in (("a.txt", b"a"), ("b.txt", b"b")):
            data = io.BytesIO(payload)
            ti = tarfile.TarInfo(name=name)
            ti.size = len(payload)
            tf.addfile(ti, data)

    with pytest.raises(RuntimeError, match="expected one tar entry"):
        _untar_one_file(buf.getvalue(), tmp_path / "out.txt")


def test_untar_one_file_rejects_empty_tarball(tmp_path: Path) -> None:
    from xrlenv_plugins.harbor.environment import _untar_one_file

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w"):
        pass

    with pytest.raises(RuntimeError, match="no file entries"):
        _untar_one_file(buf.getvalue(), tmp_path / "out.txt")


# ──────────────────────────────────────────────────────────────────────────────
# Lazy client construction from env.
# ──────────────────────────────────────────────────────────────────────────────


def test_client_from_env_raises_with_hint_when_host_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xrlenv_plugins.harbor.environment import _client_from_env

    monkeypatch.delenv("XRLENV_GRPC_HOST", raising=False)
    with pytest.raises(RuntimeError, match="XRLENV_GRPC_HOST"):
        _client_from_env()


def test_client_from_env_constructs_grpc_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xrlenv_plugins.harbor import environment as env_mod

    monkeypatch.setenv("XRLENV_GRPC_HOST", "10.0.0.5")
    monkeypatch.setenv("XRLENV_GRPC_PORT", "50061")
    monkeypatch.setenv("XRLENV_CONSUMER_TOKEN", "tok-123")
    monkeypatch.setenv("XRLENV_GRPC_SECURE", "true")

    captured: dict[str, Any] = {}

    class _StubClient:
        @classmethod
        def grpc(cls, *, host: str, port: int, token: str | None,
                 secure: bool) -> Any:
            captured["host"] = host
            captured["port"] = port
            captured["token"] = token
            captured["secure"] = secure
            return "stub-client"

    import xrlenv.client.client as client_mod
    monkeypatch.setattr(client_mod, "Client", _StubClient)

    result = env_mod._client_from_env()
    assert result == "stub-client"
    assert captured == {
        "host": "10.0.0.5", "port": 50061,
        "token": "tok-123", "secure": True,
    }


def test_client_from_env_falls_back_to_default_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xrlenv_plugins.harbor import environment as env_mod

    monkeypatch.setenv("XRLENV_GRPC_HOST", "host.local")
    monkeypatch.delenv("XRLENV_GRPC_PORT", raising=False)
    monkeypatch.delenv("XRLENV_GRPC_SECURE", raising=False)
    monkeypatch.delenv("XRLENV_CONSUMER_TOKEN", raising=False)

    captured: dict[str, Any] = {}

    class _StubClient:
        @classmethod
        def grpc(cls, *, host: str, port: int, token: str | None,
                 secure: bool) -> Any:
            captured.update(host=host, port=port, token=token, secure=secure)
            return "ok"

    import xrlenv.client.client as client_mod
    monkeypatch.setattr(client_mod, "Client", _StubClient)

    env_mod._client_from_env()
    assert captured["port"] == 50061 or captured["port"] == 50051
    # Specifically pin default = 50051 (drop-in default).
    assert captured["port"] == 50051
    assert captured["token"] is None
    assert captured["secure"] is False


# ──────────────────────────────────────────────────────────────────────────────
# Lifecycle: start / stop / image-ref.
# ──────────────────────────────────────────────────────────────────────────────


def _make_inst(
    *,
    docker_image: str | None = None,
    environment_name: str = "fix-git",
    session_id: str = "trial.fix-git__abc",
    workdir: str | None = None,
    keep_containers: bool = False,
    trial_dir: Path | None = None,
    environment_dir: Path | None = None,
    xrlenv_kwargs: dict[str, Any] | None = None,
    cpus: int | None = None,
    memory_mb: int | None = None,
) -> Any:
    from harbor.models.trial.config import ResourceMode
    from xrlenv_plugins.harbor import XrlenvHarborEnvironmentCluster

    inst = XrlenvHarborEnvironmentCluster.__new__(XrlenvHarborEnvironmentCluster)
    inst._xrlenv_session = None
    inst._xrlenv_client = None
    inst._sysbox_upload = False  # set by start(); default matches __init__
    inst._xfer_seq = 0
    inst._xrlenv_kwargs = dict(xrlenv_kwargs or {})
    inst._keep_containers = keep_containers
    inst._persistent_env = {}
    # harbor 0.20 merges per-exec env overlays via a ContextVar that
    # BaseEnvironment.__init__ sets; the __new__ stub must supply it (empty
    # default) so the cluster exec path's _merge_env doesn't AttributeError.
    inst._exec_env_overlays = contextvars.ContextVar(
        "exec_env_overlays", default=(),
    )
    # harbor 0.20 DockerEnvironment.validate_network_policy_support reads this
    # __init__-set flag (Windows containers can't switch policy).
    inst._is_windows_container = False
    inst.default_user = None
    inst.environment_name = environment_name
    # harbor sets environment_dir = <task_dir>/environment, so parent.name is the
    # task id (the directory name). Default mirrors that.
    inst.environment_dir = environment_dir or Path(f"/cache/{environment_name}/environment")
    inst.session_id = session_id
    inst.logger = logging.getLogger(f"test.cluster.{environment_name}")

    cfg = MagicMock()
    cfg.docker_image = docker_image
    cfg.workdir = workdir
    # harbor's _effective_cpus/_effective_memory_mb read these + the enforcement
    # mode (AUTO ≠ IGNORE → the declared value flows through). __new__ bypasses
    # __init__, so set the attributes those accessors need explicitly. Default
    # cpus/memory_mb=None → _effective_* return None → start() omits the limit.
    cfg.cpus = cpus
    cfg.memory_mb = memory_mb
    # Default to an empty env dict so the per-task env-marker reads (cpu-pinning,
    # container-runtime, inner-dockerd) see a real mapping, not a MagicMock.
    # Tests that exercise a marker set cfg.env explicitly.
    cfg.env = {}
    inst.task_env_config = cfg
    inst._cpu_resource_mode = ResourceMode.AUTO
    inst._memory_resource_mode = ResourceMode.AUTO

    trial_paths = MagicMock()
    trial_paths.trial_dir = trial_dir or Path("/tmp/trials/fix-git__abc")
    inst.trial_paths = trial_paths
    return inst


def test_resolve_image_ref_prefers_prebuilt() -> None:
    inst = _make_inst(docker_image="ghcr.io/upstream/fix-git:1.0")
    assert inst._resolve_image_ref() == "ghcr.io/upstream/fix-git:1.0"


def test_resolve_image_ref_expands_registry_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host-agnostic private-registry placeholder in docker_image (lhtb's repin output)
    is expanded from .env at acquire — so the cache never bakes a host (GUIDELINE §5.3.1)."""
    monkeypatch.setenv("XRLENV_PRIVATE_REGISTRY_HOST", "ip-10-0-9-10")
    monkeypatch.setenv("XRLENV_PRIVATE_REGISTRY_PORT", "5011")
    inst = _make_inst(
        docker_image="${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}"
        "/lhtb/chess-mate:main",
    )
    assert inst._resolve_image_ref() == "ip-10-0-9-10:5011/lhtb/chess-mate:main"


def test_resolve_image_ref_fails_loud_on_unresolved_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the registry host env is unset, fail loud rather than pull a literal
    ``${...}`` ref (which would be a confusing docker error)."""
    from xrlenv.errors import XRLEnvError

    monkeypatch.delenv("XRLENV_PRIVATE_REGISTRY_HOST", raising=False)
    monkeypatch.delenv("XRLENV_PRIVATE_REGISTRY_PORT", raising=False)
    inst = _make_inst(docker_image="${XRLENV_PRIVATE_REGISTRY_HOST}:5011/lhtb/x:main")
    with pytest.raises(XRLEnvError, match="unresolved registry placeholder"):
        inst._resolve_image_ref()


def test_resolve_image_ref_falls_back_to_hb_tag() -> None:
    inst = _make_inst(docker_image=None, environment_name="Fix Git")
    # Sanitize: "hb__Fix Git" → "hb__fix-git"
    assert inst._resolve_image_ref() == "hb__fix-git"


def test_resolve_image_ref_template_uses_task_id() -> None:
    # The image-template kwarg (seta-env's path to a private-registry build) derives
    # the task id from the task directory name (environment_dir.parent.name).
    inst = _make_inst(
        docker_image=None, environment_dir=Path("/cache/88/environment"),
        xrlenv_kwargs={"xrlenv_image_template": "10.0.0.1:5011/seta-env/{task_id}:main"},
    )
    assert inst._resolve_image_ref() == "10.0.0.1:5011/seta-env/88:main"


def test_resolve_image_ref_template_takes_precedence_over_docker_image() -> None:
    # An explicit sweep-injected template kwarg wins over a task-declared docker_image.
    inst = _make_inst(
        docker_image="ghcr.io/upstream/x:1.0",
        environment_dir=Path("/cache/0/environment"),
        xrlenv_kwargs={"xrlenv_image_template": "reg:5011/seta-env/{task_id}:main"},
    )
    assert inst._resolve_image_ref() == "reg:5011/seta-env/0:main"


# ── _image_namespace_tag: sidecar namespace, template kwarg OR derived-from-main-ref ──
#
# The decoupling: a multi-service compose task's sidecars need a private-registry
# namespace. It can come from the sweep-injected ``xrlenv_image_template`` kwarg, but
# that ALSO overrides every task's main image, so it can't safely be set for a whole
# sweep. Now the namespace is derived from the already-repinned main ref, so
# chess-mate resolves in the ordinary sweep with no template set at all.

def test_image_namespace_tag_derives_from_repinned_main_ref() -> None:
    # No template set: the namespace falls out of the repinned main ref, so the
    # sidecar (chess-mate-game) resolves under the same private registry/namespace.
    inst = _make_inst(environment_dir=Path("/cache/chess-mate/environment"))
    ns, tag = inst._image_namespace_tag("ip-10-0-5-6:5011/lhtb/chess-mate:main")
    assert ns == "ip-10-0-5-6:5011/lhtb"
    assert tag == "main"


def test_image_namespace_tag_none_when_main_ref_is_docker_io() -> None:
    # A docker.io-relative main ref (the 45 prebuilt LHTB tasks) yields no
    # private-registry namespace — assemble_project then fails loud only if the task
    # actually has sub-dir builds (single-service tasks never reach that path).
    inst = _make_inst()
    ns, _tag = inst._image_namespace_tag("zli12321/lhtb-2048:20260615")
    assert ns is None


def test_image_namespace_tag_template_kwarg_wins_over_main_ref() -> None:
    # An explicitly-injected template kwarg is honored (seta-env / opt-in overrides),
    # taking precedence over the derived-from-main-ref path.
    inst = _make_inst(
        xrlenv_kwargs={"xrlenv_image_template": "reg:5011/seta-env/{task_id}:main"},
    )
    ns, tag = inst._image_namespace_tag("ip-10-0-5-6:5011/lhtb/chess-mate:main")
    # template namespace (split on {task_id}), NOT the main-ref-derived one
    assert ns == "reg:5011/seta-env"
    assert tag == "main"


@pytest.mark.asyncio
async def test_start_acquires_and_chmods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inst = _make_inst()

    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(
        return_value=_async_chunks([_chunk(stdout=b"", done=True, exit_code=0)]),
    )
    fake_session.destroy = AsyncMock()

    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()

    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)

    fake_client.acquire_container.assert_awaited_once()
    call_kwargs = fake_client.acquire_container.await_args.kwargs
    assert call_kwargs["image"] == "hb__fix-git"
    assert call_kwargs["command"] == ["sleep", "infinity"]
    assert call_kwargs["task_key"] == "fix-git"
    labels = call_kwargs["labels"]
    assert labels["harbor.environment_name"] == "fix-git"
    # Default xrlenv labels: artifact_path = trial_paths.trial_dir,
    # displayed_name = harbor session_id (trial-level disambig in
    # admin UI). These should appear without the operator wrapping
    # the trial in ``with xrlenv.rollout_metadata(...):``.
    assert labels["xrlenv.rollout.artifact_path"] == "/tmp/trials/fix-git__abc"
    assert labels["xrlenv.rollout.displayed_name"] == "trial.fix-git__abc"

    # exec_stream was called for the mkdir + chmod 777 setup step.
    fake_session.exec_stream.assert_called_once()
    setup_cmd = fake_session.exec_stream.call_args.args[0]
    assert setup_cmd[0] == "bash"
    assert "mkdir -p" in setup_cmd[2]
    assert "chmod 777" in setup_cmd[2]
    assert "/logs/agent" in setup_cmd[2]
    assert "/logs/verifier" in setup_cmd[2]
    assert "/logs/artifacts" in setup_cmd[2]
    assert fake_session.exec_stream.call_args.kwargs.get("user") == "root"


async def _start_and_capture_acquire(
    inst: Any, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Drive ``start()`` with a mocked client and return the kwargs passed
    to ``acquire_container`` — shared by the cpu-pinning opt-in tests."""
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(
        return_value=_async_chunks([_chunk(stdout=b"", done=True, exit_code=0)]),
    )
    fake_session.destroy = AsyncMock()
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()

    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)
    fake_client.acquire_container.assert_awaited_once()
    return fake_client.acquire_container.await_args.kwargs


@pytest.mark.asyncio
async def test_start_env_marker_enables_cpu_pinning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-task ``[environment.env] XRLENV_CPU_PINNING = "1"`` marker (the
    patched-cache channel for nproc-scaling oracles) makes the cluster env
    acquire with ``RuntimeLimits(cpu_pinning=True)`` — while cpu/memory caps
    are still forwarded (limits respected)."""
    inst = _make_inst(cpus=2, memory_mb=4096)
    inst.task_env_config.env = {"XRLENV_CPU_PINNING": "1"}

    call_kwargs = await _start_and_capture_acquire(inst, monkeypatch)

    rl = call_kwargs["runtime_limits"]
    assert rl is not None and rl.cpu_pinning is True
    # Limits still enforced alongside pinning.
    assert call_kwargs["cpu_limit"] == 2.0
    assert call_kwargs["mem_limit_bytes"] == 4096 * 1024 * 1024


@pytest.mark.asyncio
@pytest.mark.parametrize("truthy", ["1", "true", "TRUE", "yes", "on"])
async def test_start_env_marker_accepts_truthy_spellings(
    monkeypatch: pytest.MonkeyPatch, truthy: str,
) -> None:
    inst = _make_inst(cpus=1)
    inst.task_env_config.env = {"XRLENV_CPU_PINNING": truthy}
    call_kwargs = await _start_and_capture_acquire(inst, monkeypatch)
    assert call_kwargs["runtime_limits"].cpu_pinning is True


@pytest.mark.asyncio
async def test_start_no_marker_leaves_pinning_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default (no marker, no kwarg): faithful quota-only — no cpuset."""
    inst = _make_inst(cpus=2)
    inst.task_env_config.env = {}
    call_kwargs = await _start_and_capture_acquire(inst, monkeypatch)
    assert call_kwargs["runtime_limits"] is None


@pytest.mark.asyncio
async def test_start_falsey_marker_leaves_pinning_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inst = _make_inst(cpus=2)
    inst.task_env_config.env = {"XRLENV_CPU_PINNING": "0"}
    call_kwargs = await _start_and_capture_acquire(inst, monkeypatch)
    assert call_kwargs["runtime_limits"] is None


@pytest.mark.asyncio
async def test_start_job_kwarg_enables_cpu_pinning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The blunt job-level ``environment.kwargs: {xrlenv_cpu_pinning: true}``
    channel still works (applies to every task in the job)."""
    inst = _make_inst(cpus=2, xrlenv_kwargs={"xrlenv_cpu_pinning": True})
    inst.task_env_config.env = {}
    call_kwargs = await _start_and_capture_acquire(inst, monkeypatch)
    assert call_kwargs["runtime_limits"].cpu_pinning is True


@pytest.mark.asyncio
async def test_start_resource_multipliers_scale_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``xrlenv_cpu_multiplier`` / ``xrlenv_mem_multiplier`` scale each task's
    *declared* cpu/mem (preserving relative sizing) rather than flattening to
    an absolute override."""
    inst = _make_inst(
        cpus=2, memory_mb=4096,
        xrlenv_kwargs={"xrlenv_cpu_multiplier": 2.0, "xrlenv_mem_multiplier": 1.5},
    )
    inst.task_env_config.env = {}
    call_kwargs = await _start_and_capture_acquire(inst, monkeypatch)
    assert call_kwargs["cpu_limit"] == 4.0            # 2 x 2.0
    assert call_kwargs["mem_limit_bytes"] == 6144 * 1024 * 1024  # 4096 x 1.5


@pytest.mark.asyncio
async def test_start_default_multiplier_is_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No multiplier → declared cpu/mem forwarded unchanged."""
    inst = _make_inst(cpus=2, memory_mb=4096)
    inst.task_env_config.env = {}
    call_kwargs = await _start_and_capture_acquire(inst, monkeypatch)
    assert call_kwargs["cpu_limit"] == 2.0
    assert call_kwargs["mem_limit_bytes"] == 4096 * 1024 * 1024


# ──────────────────────────────────────────────────────────────────────────────
# Sysbox routing + nested-dockerd bring-up (per-task task.toml markers).
# ──────────────────────────────────────────────────────────────────────────────


def _fresh_ok_stream(*_a: Any, **_k: Any) -> AsyncIterator[Any]:
    """A fresh success chunk generator per call — needed when ``start`` calls
    ``exec_stream`` more than once (log-dir setup + dockerd bring-up), since an
    async generator is single-use."""
    return _async_chunks([_chunk(stdout=b"ok\n", done=True, exit_code=0)])


@pytest.mark.asyncio
async def test_start_env_marker_routes_container_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-task ``[environment.env] XRLENV_CONTAINER_RUNTIME = "sysbox-runc"``
    marker threads ``container_runtime`` into ``acquire_container`` — the
    control plane's KwargsPolicy + scheduler then gate/pin it to a sysbox node.
    Routing is read ONLY from the task's own env (no job-level kwarg), so it's
    case-by-case and never a global default."""
    inst = _make_inst(cpus=1)
    inst.task_env_config.env = {"XRLENV_CONTAINER_RUNTIME": "sysbox-runc"}
    call_kwargs = await _start_and_capture_acquire(inst, monkeypatch)
    assert call_kwargs["container_runtime"] == "sysbox-runc"


@pytest.mark.asyncio
async def test_start_no_runtime_marker_defaults_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No marker → ``container_runtime=None`` → acquire on the node default
    runtime (runc). Unmarked tasks are unchanged."""
    inst = _make_inst(cpus=1)
    inst.task_env_config.env = {}
    call_kwargs = await _start_and_capture_acquire(inst, monkeypatch)
    assert call_kwargs["container_runtime"] is None


@pytest.mark.asyncio
async def test_start_inner_dockerd_marker_starts_nested_dockerd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``XRLENV_INNER_DOCKERD = "1"``, ``start`` runs a SECOND exec after
    the log-dir setup that brings up a nested dockerd (a real ``dockerd`` +
    ``docker info`` readiness poll) — the faithful DinD substrate."""
    inst = _make_inst(cpus=1)
    inst.task_env_config.env = {
        "XRLENV_CONTAINER_RUNTIME": "sysbox-runc",
        "XRLENV_INNER_DOCKERD": "1",
    }
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(side_effect=_fresh_ok_stream)
    fake_session.destroy = AsyncMock()
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)

    assert (
        fake_client.acquire_container.await_args.kwargs["container_runtime"]
        == "sysbox-runc"
    )
    # exec_stream called twice: [0] log-dir mkdir/chmod, [1] dockerd bring-up.
    assert fake_session.exec_stream.call_count == 2
    dockerd_cmd = fake_session.exec_stream.call_args_list[1].args[0]
    assert dockerd_cmd[0] == "bash"
    assert "dockerd" in dockerd_cmd[2]
    assert "docker info" in dockerd_cmd[2]
    # Socket is world-usable so a non-root image user can reach the daemon.
    assert "chmod 666 /var/run/docker.sock" in dockerd_cmd[2]


@pytest.mark.asyncio
async def test_start_inner_dockerd_install_installs_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``XRLENV_INSTALL_DOCKERD`` makes the bring-up apt-install the docker
    engine when the image ships none (CLI-only DooD)."""
    inst = _make_inst(cpus=1)
    inst.task_env_config.env = {
        "XRLENV_CONTAINER_RUNTIME": "sysbox-runc",
        "XRLENV_INNER_DOCKERD": "1",
        "XRLENV_INSTALL_DOCKERD": "1",
    }
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(side_effect=_fresh_ok_stream)
    fake_session.destroy = AsyncMock()
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)
    dockerd_cmd = fake_session.exec_stream.call_args_list[1].args[0]
    assert "apt-get install -y docker-ce" in dockerd_cmd[2]
    assert "docker.io" in dockerd_cmd[2]  # fallback


@pytest.mark.asyncio
async def test_start_inner_dockerd_no_install_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default DinD bring-up does NOT apt-install — a missing dockerd fails loud
    with a hint instead."""
    inst = _make_inst(cpus=1)
    inst.task_env_config.env = {
        "XRLENV_CONTAINER_RUNTIME": "sysbox-runc",
        "XRLENV_INNER_DOCKERD": "1",
    }
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(side_effect=_fresh_ok_stream)
    fake_session.destroy = AsyncMock()
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)
    dockerd_cmd = fake_session.exec_stream.call_args_list[1].args[0]
    assert "apt-get install" not in dockerd_cmd[2]
    assert "XRLENV_INSTALL_DOCKERD" in dockerd_cmd[2]  # the hint


@pytest.mark.asyncio
async def test_start_systemd_init_boots_with_sbin_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``XRLENV_SYSTEMD_INIT`` acquires the container with ``command=[/sbin/init]``
    (systemd PID 1) instead of ``sleep infinity`` and waits for the boot to
    settle before harbor runs solve.sh."""
    inst = _make_inst(cpus=1)
    inst.task_env_config.env = {
        "XRLENV_CONTAINER_RUNTIME": "sysbox-runc",
        "XRLENV_SYSTEMD_INIT": "1",
    }
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(side_effect=_fresh_ok_stream)
    fake_session.destroy = AsyncMock()
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)

    assert fake_client.acquire_container.await_args.kwargs["command"] == ["/sbin/init"]
    # exec_stream called twice: [0] log-dir setup, [1] systemd-ready wait.
    assert fake_session.exec_stream.call_count == 2
    wait_cmd = fake_session.exec_stream.call_args_list[1].args[0]
    assert "is-system-running" in wait_cmd[2]


@pytest.mark.asyncio
async def test_start_no_systemd_init_uses_sleep_infinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default acquire command stays ``sleep infinity`` (no systemd boot)."""
    inst = _make_inst(cpus=1)
    inst.task_env_config.env = {"XRLENV_CONTAINER_RUNTIME": "sysbox-runc"}
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(side_effect=_fresh_ok_stream)
    fake_session.destroy = AsyncMock()
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)
    assert (
        fake_client.acquire_container.await_args.kwargs["command"]
        == ["sleep", "infinity"]
    )


@pytest.mark.asyncio
async def test_start_inner_dockerd_legacy_store_writes_daemon_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``XRLENV_DOCKERD_LEGACY_STORE`` makes the dockerd bring-up write
    ``/etc/docker/daemon.json`` with the containerd image store disabled, so
    pushes produce schema2 (not OCI) manifests — for schema2-only tools."""
    inst = _make_inst(cpus=1)
    inst.task_env_config.env = {
        "XRLENV_CONTAINER_RUNTIME": "sysbox-runc",
        "XRLENV_INNER_DOCKERD": "1",
        "XRLENV_DOCKERD_LEGACY_STORE": "1",
    }
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(side_effect=_fresh_ok_stream)
    fake_session.destroy = AsyncMock()
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)

    dockerd_cmd = fake_session.exec_stream.call_args_list[1].args[0]
    assert "containerd-snapshotter" in dockerd_cmd[2]
    assert "/etc/docker/daemon.json" in dockerd_cmd[2]


@pytest.mark.asyncio
async def test_start_inner_dockerd_no_legacy_store_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default DinD bring-up does NOT touch daemon.json (keeps dockerd's modern
    default image store)."""
    inst = _make_inst(cpus=1)
    inst.task_env_config.env = {
        "XRLENV_CONTAINER_RUNTIME": "sysbox-runc",
        "XRLENV_INNER_DOCKERD": "1",
    }
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(side_effect=_fresh_ok_stream)
    fake_session.destroy = AsyncMock()
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)
    dockerd_cmd = fake_session.exec_stream.call_args_list[1].args[0]
    assert "daemon.json" not in dockerd_cmd[2]


@pytest.mark.asyncio
async def test_start_inner_dockerd_off_skips_bringup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sysbox task WITHOUT the inner-dockerd marker (e.g. a systemd task that
    boots its own daemon) only runs the log-dir setup exec — no dockerd
    bring-up."""
    inst = _make_inst(cpus=1)
    inst.task_env_config.env = {"XRLENV_CONTAINER_RUNTIME": "sysbox-runc"}
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(side_effect=_fresh_ok_stream)
    fake_session.destroy = AsyncMock()
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)
    assert fake_session.exec_stream.call_count == 1  # log-dir only


@pytest.mark.asyncio
async def test_start_inner_dockerd_failure_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the nested-dockerd exec exits non-zero (e.g. a CLI-only image with no
    ``dockerd``), ``start`` fails loud rather than letting solve.sh hit an opaque
    'Cannot connect to the Docker daemon'."""
    inst = _make_inst(cpus=1)
    inst.task_env_config.env = {
        "XRLENV_CONTAINER_RUNTIME": "sysbox-runc",
        "XRLENV_INNER_DOCKERD": "1",
    }
    fake_session = MagicMock()
    # First exec (log-dir) OK; second exec (dockerd) fails.
    fake_session.exec_stream = MagicMock(side_effect=[
        _async_chunks([_chunk(stdout=b"", done=True, exit_code=0)]),
        _async_chunks([_chunk(stderr=b"no dockerd", done=True, exit_code=3)]),
    ])
    fake_session.destroy = AsyncMock()
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    with pytest.raises(RuntimeError, match="nested dockerd bring-up failed"):
        await inst.start(force_build=False)


@pytest.mark.asyncio
async def test_ensure_dir_succeeds_first_try() -> None:
    inst = _make_inst()
    sess = MagicMock()
    sess.exec = AsyncMock(return_value=SimpleNamespace(exit_code=0, stderr=b""))
    inst._xrlenv_session = sess
    await inst._ensure_dir("/tests")
    sess.exec.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_dir_retries_transient_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A negative ``exit_code`` (aborted exec — the AddTestsDirError flake) is
    retried; a later success clears it without raising."""
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod.asyncio, "sleep", AsyncMock())
    inst = _make_inst()
    sess = MagicMock()
    sess.exec = AsyncMock(side_effect=[
        SimpleNamespace(exit_code=-1, stderr=b""),   # transient abort
        SimpleNamespace(exit_code=0, stderr=b""),    # heals on retry
    ])
    inst._xrlenv_session = sess
    await inst._ensure_dir("/tests")  # must NOT raise
    assert sess.exec.await_count == 2


@pytest.mark.asyncio
async def test_ensure_dir_raises_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod.asyncio, "sleep", AsyncMock())
    inst = _make_inst()
    sess = MagicMock()
    sess.exec = AsyncMock(return_value=SimpleNamespace(exit_code=-1, stderr=b""))
    inst._xrlenv_session = sess
    with pytest.raises(RuntimeError, match=r"failed after 3 attempts"):
        await inst._ensure_dir("/tests")
    assert sess.exec.await_count == 3


@pytest.mark.asyncio
async def test_start_rollout_metadata_overrides_default_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``xrlenv.rollout_metadata(...)`` set in scope before harbor's
    ``Job.run()`` should override the cluster Environment's default
    labels — gives operators a hook to label trials with their own
    job-level ids without forking the harbor adapter."""
    import xrlenv

    inst = _make_inst()
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(
        return_value=_async_chunks([_chunk(stdout=b"", done=True, exit_code=0)]),
    )
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    with xrlenv.rollout_metadata(
        artifact_path="/custom/path",
        displayed_name="my-job-step-7",
    ):
        await inst.start(force_build=False)

    labels = fake_client.acquire_container.await_args.kwargs["labels"]
    assert labels["xrlenv.rollout.artifact_path"] == "/custom/path"
    assert labels["xrlenv.rollout.displayed_name"] == "my-job-step-7"


@pytest.mark.asyncio
async def test_start_forwards_elevated_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``xrlenv_cap_add`` / ``xrlenv_devices`` / ``xrlenv_privileged`` set via
    ``environment.kwargs`` reach ``acquire_container`` verbatim, so infra tasks
    can request NET_ADMIN/SYS_ADMIN/SYS_PTRACE (and, where the operator opts in,
    privileged) without a forked adapter. The control plane's KwargsPolicy is the
    actual gate; the plugin only forwards intent."""
    inst = _make_inst(
        xrlenv_kwargs={
            "xrlenv_cap_add": ["NET_ADMIN", "SYS_ADMIN"],
            "xrlenv_devices": ["/dev/loop0:/dev/loop0"],
            "xrlenv_privileged": True,
        },
    )
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(
        return_value=_async_chunks([_chunk(stdout=b"", done=True, exit_code=0)]),
    )
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)

    call_kwargs = fake_client.acquire_container.await_args.kwargs
    assert call_kwargs["cap_add"] == ["NET_ADMIN", "SYS_ADMIN"]
    assert call_kwargs["devices"] == ["/dev/loop0:/dev/loop0"]
    assert call_kwargs["privileged"] is True


@pytest.mark.asyncio
async def test_start_defaults_to_no_elevated_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no elevated-cap kwargs, ``start`` forwards the safe defaults
    (cap_add/devices=None, privileged=False) — unchanged behavior for the
    overwhelming majority of tasks that need nothing special."""
    inst = _make_inst()  # _xrlenv_kwargs == {}
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(
        return_value=_async_chunks([_chunk(stdout=b"", done=True, exit_code=0)]),
    )
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)

    call_kwargs = fake_client.acquire_container.await_args.kwargs
    assert call_kwargs["cap_add"] is None
    assert call_kwargs["devices"] is None
    assert call_kwargs["privileged"] is False


@pytest.mark.asyncio
async def test_start_forwards_declared_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The task's declared cpus/memory_mb (via harbor's _effective_* accessors)
    reach acquire_container as cpu_limit/mem_limit_bytes — so the cluster honors
    task.toml instead of always applying the node's 2 CPU / 4 GiB default."""
    inst = _make_inst(cpus=4, memory_mb=8192)
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(
        return_value=_async_chunks([_chunk(stdout=b"", done=True, exit_code=0)]),
    )
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)

    call_kwargs = fake_client.acquire_container.await_args.kwargs
    assert call_kwargs["cpu_limit"] == 4.0
    assert call_kwargs["mem_limit_bytes"] == 8192 * 1024 * 1024


@pytest.mark.asyncio
async def test_start_omits_resources_when_undeclared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task that declares no cpus/memory → _effective_* are None → start omits
    the limits → the node keeps its safe default (unchanged behavior)."""
    inst = _make_inst()  # cpus/memory_mb default to None
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(
        return_value=_async_chunks([_chunk(stdout=b"", done=True, exit_code=0)]),
    )
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)

    call_kwargs = fake_client.acquire_container.await_args.kwargs
    assert call_kwargs["cpu_limit"] is None
    assert call_kwargs["mem_limit_bytes"] is None


@pytest.mark.asyncio
async def test_start_acquires_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """The container is always acquired OPEN (``network_mode=None``) — install
    needs network; an offline task is restricted post-install via apply_egress
    (open-setup→tighten), not with ``--network none`` at acquire."""
    inst = _make_inst()
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(
        return_value=_async_chunks([_chunk(stdout=b"", done=True, exit_code=0)]),
    )
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)
    assert fake_client.acquire_container.await_args.kwargs["network_mode"] is None


@pytest.mark.asyncio
async def test_apply_egress_builds_allowlist_and_calls_session() -> None:
    """The env's apply_egress passthrough builds an EgressAllowlist from the
    cidrs (+ ports) and forwards it to the cluster session."""
    inst = _make_inst()
    sess = MagicMock()
    sess.apply_egress = AsyncMock()
    inst._xrlenv_session = sess

    await inst.apply_egress(
        ["18.224.254.138/32", "18.225.128.251/32"], ports=(443,),
    )
    sess.apply_egress.assert_awaited_once()
    allowlist = sess.apply_egress.await_args.args[0]
    assert [(r.cidr, r.ports) for r in allowlist.rules] == [
        ("18.224.254.138/32", (443,)), ("18.225.128.251/32", (443,)),
    ]
    assert sess.apply_egress.await_args.kwargs["dns_resolver"] is None


@pytest.mark.asyncio
async def test_apply_egress_empty_is_block_all() -> None:
    """No cidrs → empty allowlist (block all external egress)."""
    inst = _make_inst()
    sess = MagicMock()
    sess.apply_egress = AsyncMock()
    inst._xrlenv_session = sess

    await inst.apply_egress([])
    allowlist = sess.apply_egress.await_args.args[0]
    assert allowlist.rules == ()


@pytest.mark.asyncio
async def test_apply_egress_before_start_raises() -> None:
    """apply_egress before start() (no session) fails loudly."""
    inst = _make_inst()  # _xrlenv_session is None
    with pytest.raises(RuntimeError, match="before"):
        await inst.apply_egress(["1.2.3.4/32"])


@pytest.mark.asyncio
async def test_start_force_build_logs_warning_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    inst = _make_inst()
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(
        return_value=_async_chunks([_chunk(stdout=b"", done=True, exit_code=0)]),
    )
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    with caplog.at_level(logging.INFO):
        await inst.start(force_build=True)

    assert any("force_build=True ignored" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_start_closes_client_on_acquire_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inst = _make_inst()
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(
        side_effect=RuntimeError("ImageNotFound"),
    )
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    with pytest.raises(RuntimeError, match="ImageNotFound"):
        await inst.start(force_build=False)

    fake_client.close.assert_awaited_once()
    assert inst._xrlenv_client is None
    assert inst._xrlenv_session is None


@pytest.mark.asyncio
async def test_stop_idempotent_when_not_started() -> None:
    inst = _make_inst()
    # Should not raise.
    await inst.stop(delete=True)


@pytest.mark.asyncio
async def test_stop_destroys_session_and_closes_client() -> None:
    inst = _make_inst()
    inst._xrlenv_session = MagicMock()
    inst._xrlenv_session.destroy = AsyncMock()
    inst._xrlenv_client = MagicMock()
    inst._xrlenv_client.close = AsyncMock()

    await inst.stop(delete=True)

    assert inst._xrlenv_session is None
    assert inst._xrlenv_client is None


@pytest.mark.asyncio
async def test_stop_keep_containers_logs_but_destroys() -> None:
    inst = _make_inst(keep_containers=True)
    session = MagicMock()
    session.destroy = AsyncMock()
    inst._xrlenv_session = session
    inst._xrlenv_client = MagicMock()
    inst._xrlenv_client.close = AsyncMock()

    await inst.stop(delete=False)
    session.destroy.assert_awaited_once()
    assert inst._xrlenv_session is None


# ──────────────────────────────────────────────────────────────────────────────
# exec: streamed-chunk aggregation + arg pass-through.
# ──────────────────────────────────────────────────────────────────────────────


def _chunk(
    *,
    stdout: bytes = b"", stderr: bytes = b"",
    done: bool = False, exit_code: int = 0,
    timed_out: bool = False,
) -> Any:
    from xrlenv.control.service import RawExecChunk
    return RawExecChunk(
        stdout=stdout, stderr=stderr, done=done,
        exit_code=exit_code, timed_out=timed_out,
    )


def _async_chunks(chunks: list[Any]) -> AsyncIterator[Any]:
    async def _gen() -> AsyncIterator[Any]:
        for c in chunks:
            yield c
    return _gen()


@pytest.mark.asyncio
async def test_exec_aggregates_stdout_stderr_and_exit_code() -> None:
    inst = _make_inst()
    session = MagicMock()
    session.exec_stream = MagicMock(
        return_value=_async_chunks([
            _chunk(stdout=b"line1\n"),
            _chunk(stdout=b"line2\n", stderr=b"warn\n"),
            _chunk(done=True, exit_code=7),
        ]),
    )
    inst._xrlenv_session = session

    result = await inst.exec("echo hi")
    assert result.return_code == 7
    assert result.stdout == "line1\nline2\n"
    assert result.stderr == "warn\n"


@pytest.mark.asyncio
async def test_exec_passes_cwd_env_user_and_timeout() -> None:
    inst = _make_inst()
    inst._persistent_env = {"PERSIST": "1"}
    session = MagicMock()
    session.exec_stream = MagicMock(
        return_value=_async_chunks([_chunk(done=True, exit_code=0)]),
    )
    inst._xrlenv_session = session

    await inst.exec(
        "echo hi", cwd="/work", env={"EXTRA": "v"},
        timeout_sec=120, user="agent",
    )
    call_kwargs = session.exec_stream.call_args.kwargs
    assert call_kwargs["cwd"] == "/work"
    assert call_kwargs["env"] == {"PERSIST": "1", "EXTRA": "v"}
    assert call_kwargs["user"] == "agent"
    assert call_kwargs["timeout_s"] == 120.0
    assert session.exec_stream.call_args.args[0] == ["bash", "-c", "echo hi"]


@pytest.mark.asyncio
async def test_exec_uses_default_timeout_when_none() -> None:
    from xrlenv_plugins.harbor.environment import _DEFAULT_EXEC_TIMEOUT_S

    inst = _make_inst()
    session = MagicMock()
    session.exec_stream = MagicMock(
        return_value=_async_chunks([_chunk(done=True, exit_code=0)]),
    )
    inst._xrlenv_session = session

    await inst.exec("noop", timeout_sec=None)
    assert session.exec_stream.call_args.kwargs["timeout_s"] == _DEFAULT_EXEC_TIMEOUT_S


@pytest.mark.asyncio
async def test_exec_falls_back_to_workdir_when_no_cwd() -> None:
    inst = _make_inst(workdir="/app")
    session = MagicMock()
    session.exec_stream = MagicMock(
        return_value=_async_chunks([_chunk(done=True, exit_code=0)]),
    )
    inst._xrlenv_session = session

    await inst.exec("ls")
    assert session.exec_stream.call_args.kwargs["cwd"] == "/app"


@pytest.mark.asyncio
async def test_exec_raises_when_session_missing() -> None:
    inst = _make_inst()
    with pytest.raises(RuntimeError, match="no active cluster session"):
        await inst.exec("echo hi")


# ──────────────────────────────────────────────────────────────────────────────
# upload_* / download_*.
# ──────────────────────────────────────────────────────────────────────────────


def _ok_exec_result(exit_code: int = 0) -> Any:
    """Build a successful RawExecResult — used for mocking the
    mkdir-before-put_archive exec calls."""
    from xrlenv.control.service import RawExecResult
    return RawExecResult(
        exit_code=exit_code, stdout=b"", stderr=b"", timed_out=False,
    )


@pytest.mark.asyncio
async def test_upload_file_packs_one_entry_under_parent_dir(
    tmp_path: Path,
) -> None:
    inst = _make_inst()
    session = MagicMock()
    session.put_archive = AsyncMock()
    session.exec = AsyncMock(return_value=_ok_exec_result())
    inst._xrlenv_session = session

    src = tmp_path / "data.txt"
    src.write_bytes(b"payload")

    await inst.upload_file(src, "/container/dir/data.txt")

    # First call: mkdir -p /container/dir as root, before put_archive.
    session.exec.assert_awaited_once()
    mkdir_args = session.exec.await_args.args[0]
    assert mkdir_args == ["mkdir", "-p", "/container/dir"]
    assert session.exec.await_args.kwargs.get("user") == "root"

    session.put_archive.assert_awaited_once()
    call_kwargs = session.put_archive.await_args.kwargs
    assert call_kwargs["target_dir"] == "/container/dir"
    with tarfile.open(fileobj=io.BytesIO(call_kwargs["tarball"]), mode="r") as tf:
        names = [m.name for m in tf.getmembers()]
        assert names == ["data.txt"]


@pytest.mark.asyncio
async def test_upload_dir_packs_children_only(tmp_path: Path) -> None:
    inst = _make_inst()
    session = MagicMock()
    session.put_archive = AsyncMock()
    session.exec = AsyncMock(return_value=_ok_exec_result())
    inst._xrlenv_session = session

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_bytes(b"a")
    (src / "sub").mkdir()
    (src / "sub" / "b.txt").write_bytes(b"b")

    await inst.upload_dir(src, "/container/dst")

    # First call: mkdir -p /container/dst before put_archive.
    mkdir_args = session.exec.await_args.args[0]
    assert mkdir_args == ["mkdir", "-p", "/container/dst"]

    call_kwargs = session.put_archive.await_args.kwargs
    assert call_kwargs["target_dir"] == "/container/dst"
    with tarfile.open(fileobj=io.BytesIO(call_kwargs["tarball"]), mode="r") as tf:
        names = sorted(m.name for m in tf.getmembers())
        # docker compose cp src/. main:dst → children land directly,
        # no leading "src/" component.
        assert "a.txt" in names
        assert "sub/b.txt" in names
        assert all(not n.startswith("src/") for n in names)


@pytest.mark.asyncio
async def test_upload_dir_rejects_non_directory(tmp_path: Path) -> None:
    inst = _make_inst()
    session = MagicMock()
    session.put_archive = AsyncMock()
    session.exec = AsyncMock(return_value=_ok_exec_result())
    inst._xrlenv_session = session

    bad = tmp_path / "not-a-dir.txt"
    bad.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        await inst.upload_dir(bad, "/container/dst")


@pytest.mark.asyncio
async def test_upload_dir_raises_when_mkdir_fails(tmp_path: Path) -> None:
    """A non-zero exit on the mkdir step is a hard error: silently
    swallowing means put_archive runs against a non-existent target
    and Docker returns 404 (the bug that motivated _ensure_dir in the
    first place)."""
    inst = _make_inst()
    session = MagicMock()
    session.put_archive = AsyncMock()
    session.exec = AsyncMock(return_value=_ok_exec_result(exit_code=1))
    inst._xrlenv_session = session

    src = tmp_path / "src"
    src.mkdir()

    with pytest.raises(RuntimeError, match="mkdir -p"):
        await inst.upload_dir(src, "/forbidden")
    session.put_archive.assert_not_awaited()


@pytest.mark.asyncio
async def test_download_file_extracts_single_entry(tmp_path: Path) -> None:
    inst = _make_inst()
    # Build a docker-style get_archive tarball: one entry named "data.txt".
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        ti = tarfile.TarInfo(name="data.txt")
        payload = b"contents from container"
        ti.size = len(payload)
        tf.addfile(ti, io.BytesIO(payload))

    session = MagicMock()
    session.get_archive = AsyncMock(return_value=buf.getvalue())
    inst._xrlenv_session = session

    dst = tmp_path / "out" / "landed.txt"
    await inst.download_file("/container/data.txt", dst)
    assert dst.read_bytes() == b"contents from container"
    session.get_archive.assert_awaited_once_with("/container/data.txt")


@pytest.mark.asyncio
async def test_download_dir_strips_root_component(tmp_path: Path) -> None:
    inst = _make_inst()
    # Build a docker-style get_archive tarball rooted at "verifier/...".
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, payload in (
            ("verifier/reward.txt", b"1.0"),
            ("verifier/test-stdout.txt", b"PASS"),
        ):
            ti = tarfile.TarInfo(name=name)
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))

    session = MagicMock()
    session.get_archive = AsyncMock(return_value=buf.getvalue())
    inst._xrlenv_session = session

    dst = tmp_path / "local-verifier"
    await inst.download_dir("/logs/verifier", dst)
    assert (dst / "reward.txt").read_bytes() == b"1.0"
    assert (dst / "test-stdout.txt").read_bytes() == b"PASS"


# ──────────────────────────────────────────────────────────────────────────────
# Sysbox exec-based file transfer (avoids the put_archive /etc/resolv.conf wall).
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upload_file_sysbox_uses_exec_not_put_archive(tmp_path: Path) -> None:
    """On the sysbox path, upload_file streams the tar via exec (base64 -d |
    tar -x) instead of docker put_archive — which 500s on the idmapped
    /etc/resolv.conf under sysbox."""
    inst = _make_inst()
    inst._sysbox_upload = True
    session = MagicMock()
    session.exec = AsyncMock(return_value=_ok_exec_result())
    session.put_archive = AsyncMock()
    inst._xrlenv_session = session

    src = tmp_path / "solve.sh"
    src.write_bytes(b"#!/bin/bash\necho hi\n")
    await inst.upload_file(src, "/solution/solve.sh")

    session.put_archive.assert_not_awaited()  # no docker archive API
    session.exec.assert_awaited()
    script = session.exec.await_args.args[0]
    assert script[0] == "bash"
    assert "base64 -d" in script[2]
    assert "tar -x -C" in script[2]
    assert "/solution" in script[2]  # target parent dir


@pytest.mark.asyncio
async def test_upload_dir_sysbox_uses_exec(tmp_path: Path) -> None:
    inst = _make_inst()
    inst._sysbox_upload = True
    session = MagicMock()
    session.exec = AsyncMock(return_value=_ok_exec_result())
    session.put_archive = AsyncMock()
    inst._xrlenv_session = session

    src = tmp_path / "tests"
    src.mkdir()
    (src / "test.sh").write_bytes(b"echo ok\n")
    await inst.upload_dir(src, "/tests")

    session.put_archive.assert_not_awaited()
    script = session.exec.await_args.args[0][2]
    assert "tar -x -C" in script and "/tests" in script


@pytest.mark.asyncio
async def test_upload_sysbox_raises_on_nonzero_exec(tmp_path: Path) -> None:
    """A failed in-container tar extract surfaces loudly, not silently."""
    inst = _make_inst()
    inst._sysbox_upload = True
    session = MagicMock()
    session.exec = AsyncMock(return_value=_ok_exec_result(exit_code=2))
    inst._xrlenv_session = session
    src = tmp_path / "f"
    src.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="sysbox upload exec failed"):
        await inst.upload_file(src, "/dst/f")


@pytest.mark.asyncio
async def test_download_dir_sysbox_uses_exec_and_extracts(tmp_path: Path) -> None:
    """On the sysbox path, download_dir runs ``tar -c -C <parent> <name> |
    base64`` via exec, decodes the base64 stdout, and extracts locally (the tar
    is rooted at <name>/… — the same shape docker get_archive emits — so the
    existing _untar_dir_contents consumes it)."""
    from xrlenv.control.service import RawExecResult

    # Build the tar the in-container `tar -c -C /logs verifier` would produce:
    # entries rooted at "verifier/…", then base64 it as the exec stdout.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, payload in (
            ("verifier/reward.txt", b"1.0"),
            ("verifier/test-stdout.txt", b"PASS"),
        ):
            ti = tarfile.TarInfo(name=name)
            ti.size = len(payload)
            tf.addfile(ti, io.BytesIO(payload))
    b64_stdout = base64.b64encode(buf.getvalue())

    inst = _make_inst()
    inst._sysbox_upload = True
    session = MagicMock()
    session.exec = AsyncMock(return_value=RawExecResult(
        exit_code=0, stdout=b64_stdout, stderr=b"", timed_out=False,
    ))
    session.get_archive = AsyncMock()
    inst._xrlenv_session = session

    dst = tmp_path / "local-verifier"
    await inst.download_dir("/logs/verifier", dst)

    session.get_archive.assert_not_awaited()
    script = session.exec.await_args.args[0][2]
    assert "tar -cf -" in script and "base64" in script
    assert (dst / "reward.txt").read_bytes() == b"1.0"
    assert (dst / "test-stdout.txt").read_bytes() == b"PASS"


@pytest.mark.asyncio
async def test_download_file_sysbox_uses_exec(tmp_path: Path) -> None:
    from xrlenv.control.service import RawExecResult

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        ti = tarfile.TarInfo(name="reward.txt")
        payload = b"0.5"
        ti.size = len(payload)
        tf.addfile(ti, io.BytesIO(payload))
    b64_stdout = base64.b64encode(buf.getvalue())

    inst = _make_inst()
    inst._sysbox_upload = True
    session = MagicMock()
    session.exec = AsyncMock(return_value=RawExecResult(
        exit_code=0, stdout=b64_stdout, stderr=b"", timed_out=False,
    ))
    inst._xrlenv_session = session

    dst = tmp_path / "out" / "reward.txt"
    await inst.download_file("/logs/verifier/reward.txt", dst)
    assert dst.read_bytes() == b"0.5"


@pytest.mark.asyncio
async def test_download_sysbox_raises_on_nonzero_exec() -> None:
    inst = _make_inst()
    inst._sysbox_upload = True
    session = MagicMock()
    session.exec = AsyncMock(return_value=_ok_exec_result(exit_code=1))
    inst._xrlenv_session = session
    with pytest.raises(RuntimeError, match="sysbox download"):
        await inst.download_dir("/logs/verifier", "/tmp/x")


@pytest.mark.asyncio
async def test_chown_to_host_user_is_noop() -> None:
    inst = _make_inst()
    # Should not raise even with no session set (it's an explicit no-op).
    await inst._chown_to_host_user("/some/path", recursive=True)


@pytest.mark.asyncio
async def test_upload_file_raises_when_session_missing(tmp_path: Path) -> None:
    inst = _make_inst()
    src = tmp_path / "f"
    src.write_bytes(b"")
    with pytest.raises(RuntimeError, match="no cluster session"):
        await inst.upload_file(src, "/dst/f")


@pytest.mark.asyncio
async def test_download_dir_raises_when_session_missing(tmp_path: Path) -> None:
    inst = _make_inst()
    with pytest.raises(RuntimeError, match="no cluster session"):
        await inst.download_dir("/src", tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# §4b-2 — multi-service compose routing
# ──────────────────────────────────────────────────────────────────────────────

_MULTI_SERVICE_YAML = (
    "services:\n"
    "  postgres:\n"
    "    image: postgres:14\n"
    "  app:\n"
    "    build: .\n"
)
_SINGLE_SERVICE_YAML = "services:\n  main: {}\n"
_SUBDIR_BUILD_YAML = (
    "services:\n"
    "  main: {}\n"
    "  solr:\n"
    "    build: ./solr-node\n"
)


def _make_compose_inst(
    tmp_path: Path,
    compose_yaml: str,
    *,
    task_id: str = "tw_522753",
    cpus: int | None = None,
    memory_mb: int | None = None,
    env: dict[str, str] | None = None,
    image_template: str | None = None,
) -> Any:
    """A cluster instance whose ``environment/docker-compose.yaml`` is on disk, so
    ``_multi_service_compose`` reads a real task compose. ``environment_dir`` is
    ``<tmp>/<task_id>/environment`` so ``parent.name`` == the task id."""
    env_dir = tmp_path / task_id / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "docker-compose.yaml").write_text(compose_yaml)
    inst = _make_inst(
        environment_dir=env_dir, environment_name=task_id,
        cpus=cpus, memory_mb=memory_mb,
        xrlenv_kwargs=(
            {"xrlenv_image_template": image_template} if image_template else None
        ),
    )
    if env is not None:
        inst.task_env_config.env = env
    return inst


async def _start_and_capture_compose(
    inst: Any, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Drive ``start()`` with a mocked client and return the kwargs passed to
    ``acquire_compose_project``; asserts the single ``acquire_container`` path was
    NOT taken."""
    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(
        return_value=_async_chunks([_chunk(stdout=b"", done=True, exit_code=0)]),
    )
    fake_session.destroy = AsyncMock()
    fake_client = MagicMock()
    fake_client.acquire_compose_project = AsyncMock(return_value=fake_session)
    fake_client.acquire_container = AsyncMock()
    fake_client.close = AsyncMock()

    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)
    fake_client.acquire_container.assert_not_awaited()  # NOT the single path
    fake_client.acquire_compose_project.assert_awaited_once()
    return fake_client.acquire_compose_project.await_args.kwargs


def test_capabilities_docker_compose_reflects_multi_service(tmp_path: Path) -> None:
    multi = _make_compose_inst(tmp_path / "m", _MULTI_SERVICE_YAML)
    single = _make_compose_inst(tmp_path / "s", _SINGLE_SERVICE_YAML)
    assert multi.capabilities.docker_compose is True
    assert single.capabilities.docker_compose is False


@pytest.mark.asyncio
async def test_start_routes_multi_service_to_compose_acquire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inst = _make_compose_inst(tmp_path, _MULTI_SERVICE_YAML)
    kwargs = await _start_and_capture_compose(inst, monkeypatch)

    assert kwargs["main_service"] == "main"
    assert kwargs["task_key"] == "tw_522753"
    # main synthesized + app (build .) repointed to the task image; postgres public.
    main_ref = inst._resolve_image_ref()
    assert set(kwargs["images"]) == {main_ref, "postgres:14"}
    doc = yaml.safe_load(kwargs["compose_yaml"])
    assert doc["services"]["main"]["image"] == main_ref
    assert doc["services"]["app"]["image"] == main_ref
    assert doc["services"]["postgres"]["image"] == "postgres:14"
    # labels carry the harbor trial identity (CP stamps the reserved xrlenv.* keys)
    assert kwargs["labels"]["harbor.environment_name"] == "tw_522753"


@pytest.mark.asyncio
async def test_start_single_service_still_uses_acquire_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a single-service compose (or none) must keep the raw acquire path untouched.
    inst = _make_compose_inst(tmp_path, _SINGLE_SERVICE_YAML)

    fake_session = MagicMock()
    fake_session.exec_stream = MagicMock(
        return_value=_async_chunks([_chunk(stdout=b"", done=True, exit_code=0)]),
    )
    fake_session.destroy = AsyncMock()
    fake_client = MagicMock()
    fake_client.acquire_container = AsyncMock(return_value=fake_session)
    fake_client.acquire_compose_project = AsyncMock()
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    await inst.start(force_build=False)
    fake_client.acquire_container.assert_awaited_once()
    fake_client.acquire_compose_project.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_compose_footprint_is_main_plus_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # main declares 6 cpu / 8 GiB; two undeclared sidecars (postgres, app) each add
    # the default 1 cpu / 1 GiB.
    inst = _make_compose_inst(tmp_path, _MULTI_SERVICE_YAML, cpus=6, memory_mb=8192)
    kwargs = await _start_and_capture_compose(inst, monkeypatch)

    assert kwargs["footprint_cpu"] == 6.0 + 2 * 1.0
    assert kwargs["footprint_mem_bytes"] == (8192 + 2 * 1024) * 1024 * 1024
    # main is capped in the shipped doc (cluster is harbor's resources-override).
    doc = yaml.safe_load(kwargs["compose_yaml"])
    assert doc["services"]["main"]["cpus"] == 6.0
    assert doc["services"]["main"]["mem_limit"] == 8192 * 1024 * 1024


@pytest.mark.asyncio
async def test_start_compose_footprint_defaults_when_undeclared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # main declares nothing → reserve the default main budget (2 cpu / 4 GiB) + the
    # two default sidecars; main is NOT capped in the doc (no declared size).
    inst = _make_compose_inst(tmp_path, _MULTI_SERVICE_YAML)
    kwargs = await _start_and_capture_compose(inst, monkeypatch)

    assert kwargs["footprint_cpu"] == 2.0 + 2 * 1.0
    assert kwargs["footprint_mem_bytes"] == (4 * 1024 + 2 * 1024) * 1024 * 1024
    doc = yaml.safe_load(kwargs["compose_yaml"])
    assert "cpus" not in doc["services"]["main"]


@pytest.mark.asyncio
async def test_start_compose_subdir_build_needs_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a sub-dir build service with no {task_id} image template → fail loud.
    inst = _make_compose_inst(tmp_path, _SUBDIR_BUILD_YAML)
    fake_client = MagicMock()
    fake_client.acquire_compose_project = AsyncMock()
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    from xrlenv.errors import XRLEnvError
    with pytest.raises(XRLEnvError, match="sub-dir build service"):
        await inst.start(force_build=False)
    fake_client.acquire_compose_project.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_compose_subdir_build_resolves_with_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # with a {task_id} template, sub-dir refs match what build_plan_gen pushed:
    # <registry>/<ns>/<task_id>-<service>:main.
    inst = _make_compose_inst(
        tmp_path, _SUBDIR_BUILD_YAML, task_id="tw_188260",
        image_template="reg:5011/terminalworld-verified/{task_id}:main",
    )
    kwargs = await _start_and_capture_compose(inst, monkeypatch)
    doc = yaml.safe_load(kwargs["compose_yaml"])
    assert doc["services"]["solr"]["image"] == (
        "reg:5011/terminalworld-verified/tw_188260-solr:main"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env",
    [
        {"XRLENV_CONTAINER_RUNTIME": "sysbox-runc"},
        {"XRLENV_SYSTEMD_INIT": "1"},
        {"XRLENV_INNER_DOCKERD": "true"},
        {"XRLENV_INSTALL_DOCKERD": "yes"},
    ],
)
async def test_start_compose_rejects_substrate_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str],
) -> None:
    # a multi-service compose + a single-container substrate marker is a
    # contradiction (compose runs under runc) — fail loud, never silently drop it.
    inst = _make_compose_inst(tmp_path, _MULTI_SERVICE_YAML, env=env)
    fake_client = MagicMock()
    fake_client.acquire_compose_project = AsyncMock()
    fake_client.close = AsyncMock()
    from xrlenv_plugins.harbor import environment as env_mod
    monkeypatch.setattr(env_mod, "_client_from_env", lambda: fake_client)

    from xrlenv.errors import XRLEnvError
    with pytest.raises(XRLEnvError, match="substrate marker"):
        await inst.start(force_build=False)
    fake_client.acquire_compose_project.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_compose_runc_runtime_marker_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # an explicit runc runtime marker is NOT a contradiction — compose runs runc.
    inst = _make_compose_inst(
        tmp_path, _MULTI_SERVICE_YAML, env={"XRLENV_CONTAINER_RUNTIME": "runc"},
    )
    kwargs = await _start_and_capture_compose(inst, monkeypatch)
    assert kwargs["main_service"] == "main"


# Silence the "asyncio is imported but unused if pytest-asyncio re-runs" lint;
# kept for the AsyncIterator import which is used.
_ = asyncio
_ = os

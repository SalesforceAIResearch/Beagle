"""Unit tests for the harbor-0.20 native per-phase network-policy seam on
``XrlenvHarborEnvironmentCluster`` (``migration/harbor-0.20.0``).

harbor 0.20 drives egress natively: its ``Trial`` calls
``environment.set_network_policy(policy)`` at each phase boundary, gated by
``capabilities.dynamic_network_policy`` and validated fail-closed at trial
creation. The cluster env implements the enforcement hook
(``_apply_network_policy``) on top of the spec-07 ``apply_egress`` primitive and
advertises exactly what that CIDR-only primitive can enforce. These tests pin:

- ``_can_enforce_egress`` (single-container + runc + unprivileged only);
- the capability advertisement (CIDR/IPv4 yes; hostname/wildcard/IPv6 no →
  harbor fail-closed-rejects such tasks);
- the mode→egress mapping (PUBLIC→allow-all, NO_NETWORK→block-all,
  ALLOWLIST→cidrs), including PUBLIC re-open on restore;
- the non-public startup-baseline application;
- that harbor's base ``set_network_policy`` actually drives our hook (and
  no-ops when the policy is unchanged).

Lives under ``tests/unit/`` for the same sys.path reason as the sibling
``test_harbor_cluster.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def _harbor_available() -> bool:
    try:
        import harbor  # noqa: F401
        from harbor.environments.docker.docker import DockerEnvironment  # noqa: F401
        from harbor.models.task.config import NetworkPolicy  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _harbor_available(),
    reason="harbor>=0.20 not installed (pip install 'xrlenv[terminal-bench-2]')",
)


def _mk(*, env: dict[str, str] | None = None,
        xrlenv_kwargs: dict[str, Any] | None = None) -> Any:
    """A cluster env stubbed via ``__new__`` with only the attributes the
    network-policy methods read: ``task_env_config.env`` and ``_xrlenv_kwargs``.
    ``environment_dir`` is left unset → ``_multi_service_compose()`` returns None
    (single-container), the common case."""
    from xrlenv_plugins.harbor import XrlenvHarborEnvironmentCluster

    inst = XrlenvHarborEnvironmentCluster.__new__(XrlenvHarborEnvironmentCluster)
    inst.task_env_config = SimpleNamespace(env=dict(env or {}))
    inst._xrlenv_kwargs = dict(xrlenv_kwargs or {})
    # harbor 0.20 DockerEnvironment.validate_network_policy_support (reached via
    # set_network_policy) reads this __init__-set flag; the __new__ stub supplies
    # it (Windows containers can't switch policy → we only test Linux).
    inst._is_windows_container = False
    return inst


def _policy(mode: str, hosts: list[str] | None = None) -> Any:
    from harbor.models.task.config import NetworkMode, NetworkPolicy

    return NetworkPolicy(network_mode=NetworkMode(mode), allowed_hosts=hosts or [])


# ── _can_enforce_egress ───────────────────────────────────────────────────────


def test_can_enforce_egress_single_container_runc() -> None:
    assert _mk()._can_enforce_egress() is True
    assert _mk(env={"XRLENV_CONTAINER_RUNTIME": "runc"})._can_enforce_egress() is True


def test_can_enforce_egress_false_for_sysbox() -> None:
    """sysbox's inner root owns its netns and can flush the nsenter-installed
    iptables — apply_egress is not a trusted boundary there, so we don't
    advertise it (harbor then rejects an offline sysbox task at validation)."""
    inst = _mk(env={"XRLENV_CONTAINER_RUNTIME": "sysbox-runc"})
    assert inst._can_enforce_egress() is False


def test_can_enforce_egress_false_for_privileged_or_net_admin() -> None:
    assert _mk(xrlenv_kwargs={"xrlenv_privileged": True})._can_enforce_egress() is False
    assert _mk(
        xrlenv_kwargs={"xrlenv_cap_add": ["NET_ADMIN"]},
    )._can_enforce_egress() is False
    # CAP_-prefixed spelling is also caught.
    assert _mk(
        xrlenv_kwargs={"xrlenv_cap_add": ["CAP_NET_ADMIN"]},
    )._can_enforce_egress() is False


def test_can_enforce_egress_false_for_compose(monkeypatch: pytest.MonkeyPatch) -> None:
    """apply_egress raises for a compose project (it would restrict only main);
    a multi-service task must not advertise dynamic egress."""
    inst = _mk()
    monkeypatch.setattr(
        inst, "_multi_service_compose",
        lambda: {"services": {"main": {}, "db": {}}},
    )
    assert inst._can_enforce_egress() is False


# ── capabilities advertisement ────────────────────────────────────────────────


def test_capabilities_advertise_native_seam_when_enforceable() -> None:
    caps = _mk().capabilities
    assert caps.dynamic_network_policy is True
    assert caps.network_allowlist is True
    assert caps.network_allowlist_ipv4_cidrs is True
    assert caps.network_allowlist_ipv4_addresses is True
    assert caps.disable_internet is True
    # Not yet supported by the CIDR-only apply_egress → False so harbor
    # fail-closed-REJECTS such a task rather than silently leaving it open.
    assert caps.network_allowlist_hostnames is False
    assert caps.network_allowlist_wildcard_hostnames is False
    assert caps.network_allowlist_ipv6_cidrs is False
    assert caps.network_allowlist_ipv6_addresses is False


def test_capabilities_disable_native_seam_when_not_enforceable() -> None:
    caps = _mk(env={"XRLENV_CONTAINER_RUNTIME": "sysbox-runc"}).capabilities
    assert caps.dynamic_network_policy is False
    assert caps.network_allowlist is False
    assert caps.network_allowlist_ipv4_cidrs is False
    # disable_internet is ALSO gated: we can't seal a sysbox container's egress
    # (its inner root owns the netns), so advertising False makes harbor
    # fail-closed-REJECT an offline (NO_NETWORK) sysbox task at validation rather
    # than run it open. An ONLINE sysbox task is PUBLIC → validation never checks
    # disable_internet → unaffected.
    assert caps.disable_internet is False


# ── _apply_network_policy: mode → egress program ──────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,hosts,expected",
    [
        ("public", [], ["0.0.0.0/0"]),          # allow-all (metadata still DROP'd)
        ("no-network", [], []),                 # block-all
        ("allowlist", ["internal-ip/32", "internal-ip/24"], ["internal-ip/32", "internal-ip/24"]),
    ],
)
async def test_apply_network_policy_maps_mode_to_egress(
    mode: str, hosts: list[str], expected: list[str],
) -> None:
    inst = _mk()
    seen: list[list[str]] = []

    async def _rec(cidrs: list[str], **_kw: Any) -> None:
        seen.append(list(cidrs))

    inst.apply_egress = _rec  # type: ignore[method-assign]
    await inst._apply_network_policy(_policy(mode, hosts))
    assert seen == [expected]


# ── _apply_baseline_network_policy (startup) ──────────────────────────────────


@pytest.mark.asyncio
async def test_apply_baseline_public_or_unset_is_noop() -> None:
    """Public (or an unset baseline in a __new__ stub) → acquired-open is already
    correct → no egress call."""
    seen: list[Any] = []

    async def _rec(cidrs: list[str], **_kw: Any) -> None:
        seen.append(list(cidrs))

    # Unset baseline (stub): no-op.
    inst = _mk()
    inst.apply_egress = _rec  # type: ignore[method-assign]
    await inst._apply_baseline_network_policy()
    # Explicit public baseline: still no-op.
    inst2 = _mk()
    inst2._network_policy = _policy("public")
    inst2.apply_egress = _rec  # type: ignore[method-assign]
    await inst2._apply_baseline_network_policy()
    assert seen == []


@pytest.mark.asyncio
async def test_apply_baseline_non_public_is_enforced_at_start() -> None:
    """A non-public ``[environment]`` baseline is applied right after acquire, so
    a task whose agent-phase policy equals a non-public baseline is still sealed
    (harbor's set_network_policy no-ops when phase == baseline)."""
    inst = _mk()
    inst._network_policy = _policy("no-network")
    seen: list[list[str]] = []

    async def _rec(cidrs: list[str], **_kw: Any) -> None:
        seen.append(list(cidrs))

    inst.apply_egress = _rec  # type: ignore[method-assign]
    await inst._apply_baseline_network_policy()
    assert seen == [[]]  # no-network → block all


# ── harbor's base set_network_policy drives our hook ──────────────────────────


@pytest.mark.asyncio
async def test_harbor_set_network_policy_drives_hook_and_restores() -> None:
    """The full phase round-trip harbor's Trial performs: baseline PUBLIC →
    agent-phase NO_NETWORK (sealed) → restore PUBLIC (re-opened). And a repeated
    set to the current policy is a no-op (harbor short-circuits on equality)."""
    inst = _mk()
    inst._network_policy = _policy("public")  # construction baseline
    seen: list[list[str]] = []

    async def _rec(cidrs: list[str], **_kw: Any) -> None:
        seen.append(list(cidrs))

    inst.apply_egress = _rec  # type: ignore[method-assign]

    await inst.set_network_policy(_policy("no-network"))   # agent phase → seal
    await inst.set_network_policy(_policy("public"))       # restore → re-open
    await inst.set_network_policy(_policy("public"))       # unchanged → no-op

    assert seen == [[], ["0.0.0.0/0"]]


@pytest.mark.asyncio
async def test_harbor_rejects_unsupported_hostname_allowlist() -> None:
    """A hostname allowlist is fail-closed-rejected upstream by harbor's
    ``validate_network_policy_support`` (we advertise
    ``network_allowlist_hostnames=False``) — never silently left open."""
    inst = _mk()
    inst._network_policy = _policy("public")

    async def _rec(cidrs: list[str], **_kw: Any) -> None:  # pragma: no cover
        raise AssertionError("apply_egress must not be reached for a rejected policy")

    inst.apply_egress = _rec  # type: ignore[method-assign]
    with pytest.raises(ValueError):
        await inst.set_network_policy(_policy("allowlist", ["pypi.org"]))


# ── migration assumption guard: harbor honors 0.8-era allow_internet ──────────


def test_harbor_honors_legacy_allow_internet_false_as_no_network() -> None:
    """The migration's load-bearing assumption: harbor 0.20's ``TaskConfig``
    deprecation shim maps a 0.8-era ``[environment] allow_internet = false`` to a
    ``NO_NETWORK`` baseline — so existing offline tasks are honored **without any
    task-config edits**, and our start()/_apply_baseline_network_policy enforces
    it. If a future harbor drops the shim (offline tasks would silently run open),
    this fails loud.
    """
    import warnings

    from harbor.models.task.config import NetworkMode, TaskConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the deprecation warning is expected
        offline = TaskConfig.model_validate(
            {
                "instruction": "x",
                "environment": {"docker_image": "alpine:3", "allow_internet": False},
            },
        )
    assert offline.environment.network_mode == NetworkMode.NO_NETWORK
    assert (
        offline.environment.resolve_baseline().network_mode == NetworkMode.NO_NETWORK
    )
    # An online task (no flag) stays PUBLIC — no accidental sealing.
    online = TaskConfig.model_validate(
        {"instruction": "x", "environment": {"docker_image": "alpine:3"}},
    )
    assert online.environment.network_mode == NetworkMode.PUBLIC

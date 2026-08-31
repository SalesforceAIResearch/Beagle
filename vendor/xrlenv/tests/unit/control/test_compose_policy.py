"""CP-side compose policy vet (``xrlenv.control.compose_policy``).

The vet is a thin adapter from compose services to ``validate_kwargs``, so these
tests confirm the *mapping* (each compose field reaches the right policy check,
across all services, tagged by service name) rather than re-testing the policy
semantics themselves. Fixtures mirror the corpus multi-service shapes.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import xrlenv_plugins.harbor.compose as hc
from xrlenv.control.compose_policy import vet_compose_project
from xrlenv.control.kwargs_policy import KwargsPolicy, KwargsPolicyViolation

ALLOW_PRIV = KwargsPolicy(allow_privileged=True)


def _violation(compose: dict, *, policy: KwargsPolicy | None = None) -> str:
    with pytest.raises(KwargsPolicyViolation) as ei:
        if policy is None:
            vet_compose_project(compose)
        else:
            vet_compose_project(compose, policy=policy)
    return str(ei.value)


# ── clean composes ────────────────────────────────────────────────────────────

def test_clean_service_dns_compose_passes() -> None:
    # tw_522753-shape: a public sidecar + an app, no elevated privileges.
    compose = {
        "services": {
            "postgres": {"image": "postgres:14"},
            "app": {"image": "ns/tw_522753:main", "cpus": 1.0, "mem_limit": "1024m"},
        },
    }
    vet_compose_project(compose)  # no raise


def test_empty_and_missing_services_pass() -> None:
    vet_compose_project({})
    vet_compose_project({"services": {}})
    # a null service value must not crash the vet.
    vet_compose_project({"services": {"main": None}})


def test_corpus_iptables_shape_passes_under_allow_privileged() -> None:
    # A privileged multi-peer shape with NET_ADMIN/NET_RAW passes under the
    # operator's allow_privileged opt-in. (tw_304270 once shipped this shape;
    # Option C now strips the redundant privileged:true so it runs under runc —
    # but this asserts the underlying policy mapping still holds.)
    compose = {
        "services": {
            "main": {"privileged": True, "cap_add": ["NET_ADMIN", "NET_RAW"]},
            "stapp02": {"privileged": True, "cap_add": ["NET_ADMIN", "NET_RAW"]},
        },
    }
    vet_compose_project(compose, policy=ALLOW_PRIV)


def test_default_caps_and_intra_project_network_are_clean() -> None:
    compose = {
        "services": {
            "a": {"cap_add": ["NET_RAW", "SYS_PTRACE"]},  # level-0/1, default-ok
            "b": {"network_mode": "service:a"},  # intra-project, safe
            "c": {"network_mode": "none"},
        },
    }
    vet_compose_project(compose)


# ── privileged ────────────────────────────────────────────────────────────────

def test_privileged_without_optin_rejected_and_names_service() -> None:
    msg = _violation({"services": {"stapp02": {"privileged": True}}})
    assert "services.stapp02.privileged" in msg


def test_privileged_with_optin_passes() -> None:
    vet_compose_project({"services": {"m": {"privileged": True}}}, policy=ALLOW_PRIV)


# ── caps / devices ────────────────────────────────────────────────────────────

def test_denied_cap_rejected() -> None:
    policy = KwargsPolicy(denied_caps=("SYS_ADMIN",))
    msg = _violation(
        {"services": {"m": {"cap_add": ["SYS_ADMIN"]}}}, policy=policy,
    )
    assert "services.m.cap_add" in msg


def test_device_not_in_allowlist_rejected() -> None:
    msg = _violation({"services": {"m": {"devices": ["/dev/sda:/dev/sda"]}}})
    assert "services.m.devices" in msg


def test_allowed_device_passes() -> None:
    vet_compose_project({"services": {"m": {"devices": ["/dev/kvm"]}}})


# ── network_mode ──────────────────────────────────────────────────────────────

def test_network_mode_host_gated() -> None:
    assert "services.m.network_mode" in _violation(
        {"services": {"m": {"network_mode": "host"}}},
    )
    # opt-in clears it
    vet_compose_project(
        {"services": {"m": {"network_mode": "host"}}},
        policy=KwargsPolicy(allow_host_network=True),
    )


def test_network_mode_container_always_rejected() -> None:
    # level-3, no policy override even with everything opted in.
    msg = _violation(
        {"services": {"m": {"network_mode": "container:other"}}},
        policy=KwargsPolicy(allow_privileged=True, allow_host_network=True),
    )
    assert "services.m.network_mode" in msg


# ── host binds ────────────────────────────────────────────────────────────────

def test_docker_sock_host_bind_rejected() -> None:
    # the DooD shortcut — a host bind of the docker socket must be refused.
    compose = {
        "services": {
            "m": {"volumes": ["/var/run/docker.sock:/var/run/docker.sock"]},
        },
    }
    msg = _violation(compose)
    assert "services.m.binds" in msg


def test_long_form_bind_rejected_named_volume_clean() -> None:
    compose = {
        "services": {
            "hostbind": {
                "volumes": [{"type": "bind", "source": "/etc", "target": "/etc"}],
            },
            "named": {
                "volumes": [{"type": "volume", "source": "data", "target": "/data"}],
            },
        },
    }
    msg = _violation(compose)
    assert "services.hostbind.binds" in msg
    assert "services.named" not in msg  # named volume is not a host bind


def test_allowed_host_path_bind_passes() -> None:
    policy = KwargsPolicy(allowed_host_paths=("/fsx/data",))
    vet_compose_project(
        {"services": {"m": {"volumes": ["/fsx/data/x:/x"]}}}, policy=policy,
    )


# ── multi-service aggregation ─────────────────────────────────────────────────

def test_all_service_violations_collected() -> None:
    compose = {
        "services": {
            "a": {"privileged": True},
            "b": {"network_mode": "host"},
            "ok": {"image": "busybox"},
        },
    }
    msg = _violation(compose)
    assert "services.a.privileged" in msg
    assert "services.b.network_mode" in msg


# ── real-corpus scan (gated on the shared harbor cache) ───────────────────────
# Proves the vet on the ACTUAL TerminalWorld compose documents matches the plan:
# all 6 unblocked tasks vet clean under the DEFAULT policy — including the 3 that
# once declared privileged (tw_304270/304271/305044), which now run under runc
# with just NET_ADMIN/NET_RAW after build_cache.py's COMPOSE_DROP_PRIVILEGED
# strips the redundant privileged:true (Option C) — while tw_488034 (separately
# blocked) is rejected for its /app host bind. This reads the PATCHED cache the
# operator builds, so it tracks the plan the cluster actually runs. Skips when
# the shared FSx cache isn't mounted.

_CACHE = Path(
    os.environ.get("XRLENV_BENCHMARK_CACHE", "/path/to/xrlenv_benchmark_cache"),
) / "terminalworld-verified"

# The 6 tasks this feature unblocks — all vet clean under the default policy
# (the formerly-privileged 3 have privileged:true stripped in the cache, so no
# allow_privileged is needed; the privileged-rejection mapping is covered by the
# synthetic tests above).
_UNBLOCKED = [
    "tw_522753",
    "tw_299387",
    "tw_188260",
    "tw_304270",
    "tw_304271",
    "tw_305044",
]


def _load_corpus_compose(task_id: str) -> dict:
    doc = hc.load_compose(
        (_CACHE / task_id / "environment" / "docker-compose.yaml").read_text(),
    )
    refs = hc.default_image_refs(task_id, doc, namespace="terminalworld-verified")
    return hc.rewrite_to_image_refs(doc, refs, main_service="main")


@pytest.mark.skipif(not _CACHE.is_dir(), reason="harbor cache shard not mounted")
@pytest.mark.parametrize("task_id", _UNBLOCKED)
def test_real_corpus_unblocked_tasks_vet_clean(task_id: str) -> None:
    compose = _load_corpus_compose(task_id)
    # Clean under the DEFAULT policy — no allow_privileged needed: Option C
    # dropped the redundant privileged:true from the 3 formerly-privileged
    # stacks, leaving only default-allowed NET_ADMIN/NET_RAW caps.
    vet_compose_project(compose)
    # And still clean under the operator's allow_privileged opt-in.
    vet_compose_project(compose, policy=ALLOW_PRIV)


@pytest.mark.skipif(not _CACHE.is_dir(), reason="harbor cache shard not mounted")
def test_real_corpus_tw_488034_host_bind_rejected() -> None:
    # The heavy harbor task mounts /app:/app — the gate must reject the host bind
    # even under allow_privileged (evidence the host-bind check works on real data).
    compose = _load_corpus_compose("tw_488034")
    with pytest.raises(KwargsPolicyViolation, match=r"/app"):
        vet_compose_project(compose, policy=ALLOW_PRIV)

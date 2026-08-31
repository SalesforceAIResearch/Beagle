"""Tests for the run-config layer (xrlenv/control/run_config.py).

The run-config is the user's per-experiment policy, loaded by
``Client(run_config=...)`` and merged into rollout requests at call
time. Manifests carry only the immutable benchmark contract; per-run
deadlines, idle TTL, and init_params live here so a single control
plane can serve multiple trainers with different policies.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from xrlenv.control.run_config import (
    DeadlinesPolicy,
    RunConfig,
    TemplatePolicy,
    load_run_config,
)
from xrlenv.errors import ManifestInvalid


def _write(tmp_path: Path, body: dict) -> Path:
    p = tmp_path / "run-config.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


def test_loads_minimal_run_config(tmp_path: Path) -> None:
    body = {
        "version": 1,
        "manifests": {
            "terminal-bench-2": {
                "deadlines": {"hard_s": 1800, "step_timeout_s": 90},
                "idle_ttl_s": 600,
                "init_params": {"verbose": True},
            },
            "hello-shell": {
                "deadlines": {"hard_s": 60},
            },
        },
    }
    rc = load_run_config(_write(tmp_path, body))
    assert rc.version == 1
    assert set(rc.manifests) == {"terminal-bench-2", "hello-shell"}

    tb = rc.policy_for("terminal-bench-2")
    assert tb is not None
    assert tb.deadlines is not None
    assert tb.deadlines.hard_s == 1800
    assert tb.deadlines.step_timeout_s == 90
    assert tb.idle_ttl_s == 600
    assert tb.init_params == {"verbose": True}

    hs = rc.policy_for("hello-shell")
    assert hs is not None
    assert hs.deadlines is not None
    assert hs.deadlines.hard_s == 60
    assert hs.idle_ttl_s is None  # not specified, stays None

    assert rc.policy_for("not-in-config") is None


def test_empty_run_config_is_valid(tmp_path: Path) -> None:
    """An empty manifests: map is allowed — equivalent to "no client-side
    policy", lets the control plane defaults flow through."""
    body = {"version": 1, "manifests": {}}
    rc = load_run_config(_write(tmp_path, body))
    assert rc.manifests == {}


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestInvalid, match="not found"):
        load_run_config(tmp_path / "missing.yaml")


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    p = tmp_path / "run-config.yaml"
    p.write_text("- just a list\n- of things\n")
    with pytest.raises(ManifestInvalid, match="must be a mapping"):
        load_run_config(p)


def test_invalid_network_rejected_at_load(tmp_path: Path) -> None:
    """Audit M1: ``network`` is typed as the ``NetworkPolicy`` literal,
    so a typo like ``nonee`` is a load-time error rather than a
    silent fall-through to bridge networking. The Docker backend
    treats unknown values as "open", which would be a fail-open
    security footgun for hermetic workloads."""
    body = {
        "version": 1,
        "manifests": {
            "x": {"network": "nonee"},  # typo: meant "none"
        },
    }
    with pytest.raises(ManifestInvalid):
        load_run_config(_write(tmp_path, body))


def test_valid_network_values_accepted_at_load(tmp_path: Path) -> None:
    for valid in ("none", "open", "egress-allowlist"):
        body = {
            "version": 1,
            "manifests": {"x": {"network": valid}},
        }
        rc = load_run_config(_write(tmp_path, body))
        assert rc.policy_for("x").network == valid  # type: ignore[union-attr]


def test_invalid_network_rejected_on_request_construction() -> None:
    """Same validation must hit at the service boundary so non-Python
    gRPC consumers can't slip a typo through the wire either."""
    from pydantic import ValidationError
    from xrlenv.control.service import StartRolloutRequest

    with pytest.raises(ValidationError):
        StartRolloutRequest(template="x", network="bogus")  # type: ignore[arg-type]


def test_extra_keys_rejected(tmp_path: Path) -> None:
    """``extra='forbid'`` on the pydantic model surfaces typos as errors,
    not silent omissions — important when "deadlines" → "deadline" would
    silently drop policy."""
    body = {
        "version": 1,
        "manifests": {
            "x": {
                "deadlinez": {"hard_s": 60},  # typo
            }
        },
    }
    with pytest.raises(ManifestInvalid):
        load_run_config(_write(tmp_path, body))


def test_template_policy_accepts_partial_deadlines() -> None:
    """Every deadline field is optional; the merge layer fills missing
    values from platform defaults."""
    pol = TemplatePolicy(deadlines=DeadlinesPolicy(hard_s=300))
    assert pol.deadlines is not None
    assert pol.deadlines.hard_s == 300
    assert pol.deadlines.step_timeout_s is None


def test_run_config_default_version_and_empty_manifests() -> None:
    rc = RunConfig()
    assert rc.version == 1
    assert rc.manifests == {}


# ──────────────────────────────────────────────────────────────────────────────
# Audit M1: Client.rollout must materialize every run-config policy field
# (idle_ttl_s, init_timeout_s, teardown_timeout_s) into the request, not just
# hard_s/step_timeout_s/setup_timeout_s. Asserted here at the merge level so a
# regression surfaces without needing a full sandbox.
# ──────────────────────────────────────────────────────────────────────────────


def test_client_run_config_merge_carries_every_policy_field(
    tmp_path: Path,
) -> None:
    """When the run-config sets idle_ttl_s + init_timeout_s +
    teardown_timeout_s, the StartRolloutRequest must carry them all
    through to the control plane. Audit M1: previously only hard_s,
    step_timeout_s, and setup_timeout_s were mapped."""
    import asyncio

    from xrlenv.client.client import Client
    from xrlenv.client.transport import ClientTransport
    from xrlenv.control.service import StartRolloutRequest, StartRolloutResponse

    captured: dict[str, StartRolloutRequest] = {}

    class _CapturingTransport(ClientTransport):
        async def start_rollout(
            self, req: StartRolloutRequest,
        ) -> StartRolloutResponse:
            captured["req"] = req
            return StartRolloutResponse(
                rollout_id="rid-1",
                init_obs={},
                reward_mode="env_step",
            )

        async def step(self, *_a, **_kw):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        async def cancel(self, *_a, **_kw):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        async def cancel_group(self, *_a, **_kw):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        async def replay(self, *_a, **_kw):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        async def heartbeat(self, *_a, **_kw):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        async def set_final_reward(self, *_a, **_kw):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        async def close(self) -> None:
            pass

    rc_path = _write(
        tmp_path,
        {
            "version": 1,
            "manifests": {
                "x": {
                    "deadlines": {
                        "hard_s": 1800,
                        "step_timeout_s": 90,
                        "setup_timeout_s": 120,
                        "teardown_timeout_s": 30,
                        "init_timeout_s": 60,
                    },
                    "idle_ttl_s": 600,
                    "init_params": {"foo": "bar"},
                },
            },
        },
    )

    client = Client(_CapturingTransport(), run_config=rc_path)

    async def _run() -> None:
        with pytest.raises(NotImplementedError):
            # Step raises in our fake transport; we only care that
            # start_rollout was called and captured.
            await (await client.rollout("x")).step({"cmd": "noop"})

    asyncio.run(_run())

    req = captured["req"]
    assert req.deadline is not None
    assert req.deadline.hard_s == 1800
    assert req.deadline.step_timeout_s == 90
    assert req.deadline.setup_timeout_s == 120
    assert req.deadline.teardown_timeout_s == 30, (
        "teardown_timeout_s from run-config must flow into the request "
        "(audit M1 regression)"
    )
    assert req.deadline.init_timeout_s == 60, (
        "init_timeout_s from run-config must flow into the request "
        "(audit M1 regression)"
    )
    assert req.deadline.idle_ttl_s == 600, (
        "idle_ttl_s from run-config must flow into the request — "
        "the central audit-M1 bug was silently dropping this field"
    )
    assert req.init.get("foo") == "bar"


def test_client_run_config_backend_and_network_flow_into_request(
    tmp_path: Path,
) -> None:
    """When the run-config sets backend / network, the
    StartRolloutRequest must carry them — these are the user-side
    policy fields the manifest no longer owns. Default (no
    run-config or unset fields) must leave them ``None`` so the
    coordinator falls back to ``DEFAULT_BACKEND`` / ``DEFAULT_NETWORK``
    (xrlenv.control.defaults)."""
    import asyncio

    from xrlenv.client.client import Client
    from xrlenv.client.transport import ClientTransport
    from xrlenv.control.service import StartRolloutRequest, StartRolloutResponse

    captured: dict[str, StartRolloutRequest] = {}

    class _CapturingTransport(ClientTransport):
        async def start_rollout(self, req: StartRolloutRequest) -> StartRolloutResponse:
            captured["req"] = req
            return StartRolloutResponse(
                rollout_id="rid-1", init_obs={}, reward_mode="env_step",
            )
        async def step(self, *_a, **_kw): raise NotImplementedError  # type: ignore[no-untyped-def]
        async def cancel(self, *_a, **_kw): raise NotImplementedError  # type: ignore[no-untyped-def]
        async def cancel_group(self, *_a, **_kw): raise NotImplementedError  # type: ignore[no-untyped-def]
        async def replay(self, *_a, **_kw): raise NotImplementedError  # type: ignore[no-untyped-def]
        async def heartbeat(self, *_a, **_kw): raise NotImplementedError  # type: ignore[no-untyped-def]
        async def set_final_reward(self, *_a, **_kw): raise NotImplementedError  # type: ignore[no-untyped-def]
        async def close(self) -> None: pass

    rc_path = _write(
        tmp_path,
        {
            "version": 1,
            "manifests": {
                "x": {
                    "deadlines": {"hard_s": 60},
                    "backend": "docker",
                    "network": "none",
                },
            },
        },
    )

    client = Client(_CapturingTransport(), run_config=rc_path)

    async def _run() -> None:
        with pytest.raises(NotImplementedError):
            await (await client.rollout("x")).step({"cmd": "noop"})

    asyncio.run(_run())

    req = captured["req"]
    assert req.backend == "docker"
    assert req.network == "none"


def test_start_rollout_request_proto_round_trip_with_backend_and_network() -> None:
    """The gRPC wire must carry backend / network end-to-end.
    Round-trip the pydantic StartRolloutRequest through the proto
    converters so a future regen drift surfaces immediately."""
    from xrlenv.api.converters import (
        start_rollout_request_from_proto,
        start_rollout_request_to_proto,
    )
    from xrlenv.control.service import StartRolloutRequest

    req = StartRolloutRequest(
        template="x",
        init={"k": 1},
        backend="docker",
        network="none",
    )
    p = start_rollout_request_to_proto(req)
    back = start_rollout_request_from_proto(p)
    assert back.backend == "docker"
    assert back.network == "none"

    # Unset both — wire should preserve None on the round trip.
    req2 = StartRolloutRequest(template="x", init={})
    p2 = start_rollout_request_to_proto(req2)
    back2 = start_rollout_request_from_proto(p2)
    assert back2.backend is None
    assert back2.network is None


def test_client_no_run_config_leaves_backend_and_network_unset() -> None:
    """Without a run-config, backend / network on the request are
    ``None`` — the coordinator handles the fallback to
    ``DEFAULT_BACKEND`` / ``DEFAULT_NETWORK``. Pin the contract so a
    future regression can't silently inject defaults at the SDK
    boundary."""
    import asyncio

    from xrlenv.client.client import Client
    from xrlenv.client.transport import ClientTransport
    from xrlenv.control.service import StartRolloutRequest, StartRolloutResponse

    captured: dict[str, StartRolloutRequest] = {}

    class _CapturingTransport(ClientTransport):
        async def start_rollout(self, req: StartRolloutRequest) -> StartRolloutResponse:
            captured["req"] = req
            return StartRolloutResponse(
                rollout_id="rid-1", init_obs={}, reward_mode="env_step",
            )
        async def step(self, *_a, **_kw): raise NotImplementedError  # type: ignore[no-untyped-def]
        async def cancel(self, *_a, **_kw): raise NotImplementedError  # type: ignore[no-untyped-def]
        async def cancel_group(self, *_a, **_kw): raise NotImplementedError  # type: ignore[no-untyped-def]
        async def replay(self, *_a, **_kw): raise NotImplementedError  # type: ignore[no-untyped-def]
        async def heartbeat(self, *_a, **_kw): raise NotImplementedError  # type: ignore[no-untyped-def]
        async def set_final_reward(self, *_a, **_kw): raise NotImplementedError  # type: ignore[no-untyped-def]
        async def close(self) -> None: pass

    client = Client(_CapturingTransport())  # no run_config

    async def _run() -> None:
        with pytest.raises(NotImplementedError):
            await (await client.rollout("x")).step({"cmd": "noop"})

    asyncio.run(_run())
    req = captured["req"]
    assert req.backend is None
    assert req.network is None


def test_client_run_config_idle_ttl_without_hard_s_raises(
    tmp_path: Path,
) -> None:
    """A run-config that sets idle_ttl_s but no deadlines.hard_s would
    silently drop idle_ttl_s under the pre-fix code (Deadline.hard_s
    is mandatory, so we'd never build a Deadline at all). Make this
    an explicit error instead — the user gets a clear message asking
    to add hard_s."""
    import asyncio

    from xrlenv.client.client import Client
    from xrlenv.client.transport import ClientTransport
    from xrlenv.control.service import StartRolloutRequest, StartRolloutResponse

    class _NoopTransport(ClientTransport):
        async def start_rollout(self, req: StartRolloutRequest) -> StartRolloutResponse:
            return StartRolloutResponse(rollout_id="rid", init_obs={}, reward_mode="env_step")
        async def step(self, *_a, **_kw): ...   # type: ignore[no-untyped-def]
        async def cancel(self, *_a, **_kw): ...   # type: ignore[no-untyped-def]
        async def cancel_group(self, *_a, **_kw): ...   # type: ignore[no-untyped-def]
        async def replay(self, *_a, **_kw): ...   # type: ignore[no-untyped-def]
        async def heartbeat(self, *_a, **_kw): ...   # type: ignore[no-untyped-def]
        async def set_final_reward(self, *_a, **_kw): ...   # type: ignore[no-untyped-def]
        async def close(self) -> None: ...

    rc_path = _write(
        tmp_path,
        {
            "version": 1,
            "manifests": {
                "x": {"idle_ttl_s": 600},  # idle TTL but no hard_s
            },
        },
    )

    client = Client(_NoopTransport(), run_config=rc_path)

    async def _run() -> None:
        with pytest.raises(ValueError, match=r"deadlines\.hard_s"):
            await client.rollout("x")

    asyncio.run(_run())

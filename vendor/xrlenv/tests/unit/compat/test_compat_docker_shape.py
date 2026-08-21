"""Daemon-free shape contract for the docker-py drop-in.

Every test here runs in CI / on dev machines without a Docker
daemon — they cover the static export + class-hierarchy + signature
contract the consumer's drop-in depends on. Real-engine round-trip
coverage lives in ``tests/smoke/test_compat_docker_smoke.py``
(excluded from default ``pytest -q`` per pyproject; opt in
explicitly when you want the live-daemon round-trip).
"""

from __future__ import annotations

import inspect

import docker
import xrlenv
from xrlenv.compat.docker_client import (
    LocalDockerContainerControl,
    XrlenvAPIClient,
    XrlenvDockerClient,
    from_env,
)

# ──────────────────────────────────────────────────────────────────────────────
# Top-level export — the one-line drop-in promise
# ──────────────────────────────────────────────────────────────────────────────


def test_xrlenv_from_env_is_exported_at_package_top_level() -> None:
    """The module docstring + commit messages promise
    ``import xrlenv; client = xrlenv.from_env()``. Verify the symbol
    actually exists at the top level (lazy-imported through
    xrlenv.__getattr__)."""
    assert hasattr(xrlenv, "from_env")
    # Resolves to the same callable as the compat-module entry point.
    assert xrlenv.from_env is from_env


def test_xrlenv_from_env_is_in_dunder_all() -> None:
    """``__all__`` advertises the public surface for star imports
    + linters; the drop-in entry point belongs there."""
    assert "from_env" in xrlenv.__all__


# ──────────────────────────────────────────────────────────────────────────────
# Class hierarchy — isinstance / issubclass contracts
# ──────────────────────────────────────────────────────────────────────────────


def test_xrlenv_docker_client_is_subclass_of_docker_DockerClient() -> None:
    """Consumer code that does
    ``isinstance(self.client, docker.DockerClient)`` keeps working
    whether or not a daemon is reachable — verifiable via
    issubclass without constructing."""
    assert issubclass(XrlenvDockerClient, docker.DockerClient)


def test_xrlenv_api_client_is_subclass_of_docker_APIClient() -> None:
    """Same contract one tier down — manager classes that read
    ``self.client.api`` and check its type get a real
    ``docker.APIClient`` (with our subclass on top)."""
    assert issubclass(XrlenvAPIClient, docker.APIClient)


# ──────────────────────────────────────────────────────────────────────────────
# Signature compatibility
# ──────────────────────────────────────────────────────────────────────────────


def test_from_env_signature_accepts_optional_control_kwarg() -> None:
    """``from_env()`` is callable with no args (LocalDocker default)
    and with an explicit ``control=`` kwarg (cluster mode opt-in).
    Pinning the signature here keeps the public entry point stable."""
    sig = inspect.signature(from_env)
    params = sig.parameters
    assert "control" in params
    # The ``control`` parameter has a default, so callers can omit it.
    assert params["control"].default is None
    # And it's keyword-only (per ``*,`` in the signature).
    assert params["control"].kind == inspect.Parameter.KEYWORD_ONLY


def test_local_docker_container_control_carries_local_mode() -> None:
    """The mode flag is what XrlenvAPIClient.__init__ branches on
    when deciding whether to call super().__init__ vs intercept.
    Pin the value so cluster-mode work doesn't accidentally rename
    the discriminator."""
    control = LocalDockerContainerControl()
    assert control.mode == "local"


def test_explicit_client_kwarg_beats_env_var_grpc_host(
    monkeypatch: object,
) -> None:
    """The docstring promises ``kwargs > environment variables``. With
    XRLENV_GRPC_HOST set in env AND an explicit ``client=`` passed, the
    caller's client must win — env-driven grpc_host must not silently
    spin up an owned Client and overwrite it. Regression test for a bug
    where the env-var path clobbered the explicit ``client=`` kwarg,
    causing fixture-mocked tests to hit a real cluster when the operator
    happened to have XRLENV_GRPC_* exported in their shell."""
    monkeypatch.setenv("XRLENV_GRPC_HOST", "127.0.0.1")  # type: ignore[attr-defined]
    monkeypatch.setenv("XRLENV_GRPC_PORT", "50051")  # type: ignore[attr-defined]
    monkeypatch.setenv("XRLENV_CONSUMER_TOKEN", "ignored")  # type: ignore[attr-defined]

    sentinel = object()
    client = from_env(client=sentinel)  # type: ignore[arg-type]

    # The cluster control wraps the caller's client verbatim — no
    # owned Client built from env-var grpc_host.
    assert client.api._control.mode == "cluster"  # type: ignore[attr-defined]
    assert client.api._control._client is sentinel  # type: ignore[attr-defined]


# ──────────────────────────────────────────────────────────────────────────────
# Module structure — import cheapness invariant
# ──────────────────────────────────────────────────────────────────────────────


def test_importing_xrlenv_does_not_eagerly_load_docker() -> None:
    """The package docstring promises imports stay cheap so the
    in-sandbox stub doesn't pay docker-py's import cost. ``from_env``
    is registered lazily; importing xrlenv must not pull docker-py
    in until the consumer touches the symbol.

    The test runs ``python -c 'import xrlenv'`` in a fresh subprocess
    so the assertion isn't polluted by other tests in this process
    that legitimately use docker-py.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c",
         "import xrlenv, sys; "
         "print('docker' in sys.modules)"],
        capture_output=True, text=True, check=True,
    )
    assert proc.stdout.strip() == "False", (
        "importing xrlenv eagerly loaded docker-py — the lazy export "
        "in xrlenv.__getattr__ may be wired wrong. stdout="
        f"{proc.stdout!r} stderr={proc.stderr!r}"
    )


def test_accessing_xrlenv_from_env_loads_docker_lazily() -> None:
    """Companion to the previous test: touching ``xrlenv.from_env``
    *does* trigger the docker-py import. Pinned so a future refactor
    that accidentally pre-imports is caught."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c",
         "import xrlenv, sys; "
         "_ = xrlenv.from_env; "
         "print('docker' in sys.modules)"],
        capture_output=True, text=True, check=True,
    )
    assert proc.stdout.strip() == "True"

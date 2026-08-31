"""Unit tests for the XRLENV_HARBOR_IMAGE_TEMPLATE removal guard (FIX M3).

``XrlenvHarborEnvironment.__init__`` raises ``RuntimeError`` when the
removed env var ``XRLENV_HARBOR_IMAGE_TEMPLATE`` is set AND
``xrlenv_image_template`` is NOT among the passed kwargs — silently
ignoring the legacy var would resolve the wrong image.

If the kwarg is present, or the env var is unset, the guard must NOT fire.

Guard-free tests (no live docker daemon required) — construction succeeds
or raises only the new RuntimeError, never a docker-connectivity error,
because the guard fires before ``super().__init__`` reaches the daemon.

Skipped cleanly when harbor is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _harbor_available() -> bool:
    try:
        import harbor  # noqa: F401
        from harbor.environments.docker.docker import DockerEnvironment  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _harbor_available(),
    reason="harbor not installed (pip install 'xrlenv[terminal-bench-2]')",
)

_LEGACY_ENV_VAR = "XRLENV_HARBOR_IMAGE_TEMPLATE"


@pytest.fixture()
def minimal_harbor_args(tmp_path: Path):  # type: ignore[return]
    """Yield a dict of the positional constructor args every test needs.

    Uses a minimal Dockerfile so harbor's environment-dir validation passes.
    The constructed object won't reach the docker daemon — all tests that
    call the constructor either raise before ``super().__init__`` (the guard
    path) or succeed only with ``EnvironmentConfig(docker_image=...)`` which
    lets harbor skip the image-build step entirely.
    """
    from harbor import EnvironmentConfig, TrialPaths

    env_dir = tmp_path / "env"
    env_dir.mkdir()
    # harbor validates the env dir by looking for docker_image, Dockerfile,
    # or docker-compose.yaml — provide the lightest option.
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()

    return {
        "environment_dir": env_dir,
        "environment_name": "test-env",
        "session_id": "sess-m3-test",
        "trial_paths": TrialPaths(trial_dir=trial_dir),
        "task_env_config": EnvironmentConfig(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# M3 guard: env var set + no kwarg → RuntimeError
# ──────────────────────────────────────────────────────────────────────────────


def test_legacy_env_var_set_without_kwarg_raises_runtime_error(
    minimal_harbor_args: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX M3: XRLENV_HARBOR_IMAGE_TEMPLATE set + no xrlenv_image_template kwarg
    → RuntimeError with a message pointing at the replacement kwarg.
    """
    from xrlenv_plugins.harbor.environment import XrlenvHarborEnvironment

    monkeypatch.setenv(_LEGACY_ENV_VAR, "ghcr.io/myorg/{task_id}:latest")

    with pytest.raises(RuntimeError, match="XRLENV_HARBOR_IMAGE_TEMPLATE"):
        XrlenvHarborEnvironment(**minimal_harbor_args)


def test_legacy_env_var_message_mentions_replacement_kwarg(
    minimal_harbor_args: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RuntimeError message must mention the replacement kwarg so the
    operator knows exactly how to fix their invocation.
    """
    from xrlenv_plugins.harbor.environment import XrlenvHarborEnvironment

    monkeypatch.setenv(_LEGACY_ENV_VAR, "some/image:{task_id}")

    with pytest.raises(RuntimeError, match="xrlenv_image_template"):
        XrlenvHarborEnvironment(**minimal_harbor_args)


# ──────────────────────────────────────────────────────────────────────────────
# M3 no-raise cases
# ──────────────────────────────────────────────────────────────────────────────


def test_legacy_env_var_set_but_kwarg_also_present_does_not_raise(
    minimal_harbor_args: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX M3: env var set + xrlenv_image_template kwarg present → no RuntimeError.
    The kwarg takes precedence; the env var is acknowledged and the guard is satisfied.
    """
    from xrlenv_plugins.harbor.environment import XrlenvHarborEnvironment

    monkeypatch.setenv(_LEGACY_ENV_VAR, "ghcr.io/myorg/{task_id}:latest")

    # Should NOT raise — the kwarg is present so the guard passes.
    env = XrlenvHarborEnvironment(
        **minimal_harbor_args,
        xrlenv_image_template="ghcr.io/myorg/{task_id}:latest",
    )
    # Verify the kwarg was actually recorded (not silently dropped).
    assert env.xrlenv_kwargs.get("xrlenv_image_template") == "ghcr.io/myorg/{task_id}:latest"


def test_legacy_env_var_unset_does_not_raise(
    minimal_harbor_args: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX M3: XRLENV_HARBOR_IMAGE_TEMPLATE unset → no RuntimeError regardless
    of whether xrlenv_image_template kwarg is passed.
    """
    from xrlenv_plugins.harbor.environment import XrlenvHarborEnvironment

    monkeypatch.delenv(_LEGACY_ENV_VAR, raising=False)

    # Neither env var nor kwarg — should construct fine.
    env = XrlenvHarborEnvironment(**minimal_harbor_args)
    assert env.xrlenv_kwargs.get("xrlenv_image_template") is None


def test_legacy_env_var_unset_with_kwarg_does_not_raise(
    minimal_harbor_args: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env var unset + kwarg present is the normal new-style invocation; must not raise."""
    from xrlenv_plugins.harbor.environment import XrlenvHarborEnvironment

    monkeypatch.delenv(_LEGACY_ENV_VAR, raising=False)

    env = XrlenvHarborEnvironment(
        **minimal_harbor_args,
        xrlenv_image_template="registry.local/{task_id}:main",
    )
    assert env.xrlenv_kwargs.get("xrlenv_image_template") == "registry.local/{task_id}:main"


def test_legacy_env_var_empty_string_does_not_raise(
    minimal_harbor_args: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty XRLENV_HARBOR_IMAGE_TEMPLATE is falsy — the guard condition is
    ``os.environ.get(...)``, which returns '' (falsy) for an empty var, so it
    must not raise.
    """
    from xrlenv_plugins.harbor.environment import XrlenvHarborEnvironment

    monkeypatch.setenv(_LEGACY_ENV_VAR, "")

    # Empty string → falsy → guard does not fire.
    env = XrlenvHarborEnvironment(**minimal_harbor_args)
    assert env.xrlenv_kwargs.get("xrlenv_image_template") is None

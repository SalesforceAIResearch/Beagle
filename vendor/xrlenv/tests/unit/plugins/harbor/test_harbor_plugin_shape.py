"""Daemon-free shape contract for ``xrlenv_plugins.harbor``.

Lives under ``tests/unit/`` rather than next to the plug-in to
avoid a sys.path collision: when pytest discovers tests inside
``xrlenv_plugins/harbor/tests/``, the package-discovery walk adds
``xrlenv_plugins/`` to sys.path, which then shadows the upstream
``harbor`` pip package (since our plug-in dir name matches).
``tests/unit/`` is on a separate path branch and stays out of the way.
"""

from __future__ import annotations

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


def test_xrlenv_harbor_environment_subclasses_docker_environment() -> None:
    """The plug-in adapts harbor by inheritance, not parallel impl —
    pin the inheritance chain so a future refactor that breaks it
    surfaces immediately."""
    import harbor
    from harbor.environments.docker.docker import DockerEnvironment
    from xrlenv_plugins.harbor import XrlenvHarborEnvironment

    assert issubclass(XrlenvHarborEnvironment, DockerEnvironment)
    # harbor's Job runner expects an environment_class that
    # satisfies the BaseEnvironment Protocol; DockerEnvironment
    # already does, pin that the chain preserves it.
    assert issubclass(XrlenvHarborEnvironment, harbor.BaseEnvironment)


def test_xrlenv_kwargs_constants_are_explicit() -> None:
    """The xrlenv-only kwargs that get popped from the constructor
    are listed explicitly on the class — pin so a typo or rename is
    caught + so other framework adapters can copy the canonical set."""
    from xrlenv_plugins.harbor import XrlenvHarborEnvironment

    expected = {
        "xrlenv_task_key", "xrlenv_group_id", "xrlenv_resources",
        "xrlenv_image_pin_mode", "xrlenv_owner_id",
        "xrlenv_project_id", "xrlenv_run_id",
        # Elevated-runtime kwargs (forwarded verbatim to acquire_container;
        # gated control-plane side by KwargsPolicy).
        "xrlenv_cap_add", "xrlenv_devices", "xrlenv_privileged",
        # Per-task opt-in for cpuset pinning (default off = harbor-faithful
        # quota-only); forwarded as RuntimeLimits(cpu_pinning=True).
        "xrlenv_cpu_pinning",
        # Per-task resource multipliers (scale declared cpu/mem; default 1.0).
        "xrlenv_cpu_multiplier", "xrlenv_mem_multiplier",
        # Per-run image-ref template (sweep-injected via EnvironmentConfig kwargs).
        # str.format with {task_id}.
        "xrlenv_image_template",
    }
    assert set(XrlenvHarborEnvironment._XRLENV_KWARGS) == expected


def test_route_command_default_is_pass_through() -> None:
    """LocalDocker mode: the routing seam is a pass-through.
    Cluster-mode override (next slice) intercepts; today every
    harbor docker invocation runs unchanged."""
    from xrlenv_plugins.harbor import XrlenvHarborEnvironment

    inst = XrlenvHarborEnvironment.__new__(XrlenvHarborEnvironment)
    cmd = ["docker", "compose", "up", "-d", "--build"]
    routed = inst._xrlenv_route_command(cmd, kind="docker-compose")
    assert routed == cmd


def test_xrlenv_kwargs_property_returns_a_copy() -> None:
    """``xrlenv_kwargs`` is a read-only snapshot — mutating the
    returned dict must not corrupt the instance's stored state."""
    from xrlenv_plugins.harbor import XrlenvHarborEnvironment

    inst = XrlenvHarborEnvironment.__new__(XrlenvHarborEnvironment)
    inst._xrlenv_kwargs = {"xrlenv_task_key": "bench/1"}
    snapshot = inst.xrlenv_kwargs
    snapshot["xrlenv_task_key"] = "MUTATED"
    assert inst._xrlenv_kwargs["xrlenv_task_key"] == "bench/1"

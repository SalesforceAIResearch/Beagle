"""Regression tests guarding the package against circular imports.

The phase-0 acceptance smoke surfaced a real cycle:

    xrlenv.node.cli
      -> xrlenv.node.__init__
        -> xrlenv.node.agent
          -> xrlenv.node.trajectory_reader
            -> xrlenv.control.trajectory_sink
              -> xrlenv.control.__init__   (eager re-exports)
                -> xrlenv.control.coordinator -> ... -> xrlenv.control.node_transport
                  -> xrlenv.node.trajectory_reader  (CYCLE — partially-initialised)

The unit suite did not catch this because every test imports
``xrlenv.control`` first (via fixtures), which warms the package cache
before anything reaches into ``xrlenv.node``. The cycle only fires
when ``xrlenv.node.cli`` is the *first* xrlenv import — i.e. the
``xrlenv-node serve`` entry point under systemd. These tests force
that ordering by spawning a fresh interpreter per import; if the cycle
ever returns, ``subprocess.run`` will exit non-zero with the
``ImportError`` traceback and pytest reports it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def _run_in_fresh_interpreter(snippet: str) -> subprocess.CompletedProcess[str]:
    """Run ``snippet`` in a fresh ``python -c`` so import caches start empty.

    ``cwd=/`` keeps the source tree off ``sys.path[0]`` so the test
    exercises the installed package, not whatever the test runner's CWD
    happens to be (matters when running tests against an installed wheel).
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        check=False,
        cwd="/",
    )


def test_xrlenv_node_cli_imports_without_cycle() -> None:
    """``xrlenv-node serve`` enters via this exact import path."""
    result = _run_in_fresh_interpreter(
        """
        from xrlenv.node.cli import main  # noqa: F401
        print("ok")
        """
    )
    assert result.returncode == 0, (
        "xrlenv.node.cli failed to import in a fresh interpreter — "
        "circular-import regression. Check xrlenv.node.trajectory_reader "
        "deferred imports.\n"
        f"stderr:\n{result.stderr}"
    )
    assert result.stdout.strip() == "ok"


def test_xrlenv_node_imports_without_cycle() -> None:
    """``import xrlenv.node`` alone (no submodule pre-warm) must succeed."""
    result = _run_in_fresh_interpreter(
        """
        import xrlenv.node  # triggers xrlenv.node.__init__ -> agent chain
        from xrlenv.node import NodeAgent  # noqa: F401
        print("ok")
        """
    )
    assert result.returncode == 0, (
        "xrlenv.node failed to import in a fresh interpreter.\n"
        f"stderr:\n{result.stderr}"
    )
    assert result.stdout.strip() == "ok"


def test_node_cli_build_node_agent_wires_image_cache(tmp_path: Path) -> None:
    """Operator-reported regression (2026-05-04): the node bootstrap
    constructed ``NodeAgent`` without an ``image_cache=`` kwarg, so
    every gRPC-attached node returned the empty-fallback ``NodeImageReport``
    and the admin /images page rendered "Cache is empty / free disk: 0.00
    GiB" for nodes that actually held a full image set. The build helper
    must wire an :class:`ImageCacheManager` so ``NodeAgent.report_images``
    has something live to report.
    """
    from xrlenv.node.cli import _build_node_agent
    from xrlenv.node.image_cache import ImageCacheManager

    agent = _build_node_agent(node_id="test-node", runs_root=tmp_path)
    assert isinstance(agent.image_cache, ImageCacheManager), (
        "node-cli bootstrap must wire an ImageCacheManager so the admin "
        "/images view can surface this node's cache state"
    )


def test_xrlenv_node_trajectory_reader_imports_without_cycle() -> None:
    """The historical cycle entry point — must stay safe to import alone."""
    result = _run_in_fresh_interpreter(
        """
        from xrlenv.node.trajectory_reader import (
            FetchRangeKind,
            JsonlTrajectoryReader,
            LocalTrajectoryReader,
        )  # noqa: F401
        # Instantiation forces the deferred PlatformJsonlSink import.
        JsonlTrajectoryReader('/tmp')
        print("ok")
        """
    )
    assert result.returncode == 0, (
        "xrlenv.node.trajectory_reader failed to import or instantiate "
        "JsonlTrajectoryReader in a fresh interpreter.\n"
        f"stderr:\n{result.stderr}"
    )
    assert result.stdout.strip() == "ok"


def test_stub_logging_import_does_not_pull_prometheus() -> None:
    """The in-sandbox stub bakes a slim image (aiohttp+pyyaml+pydantic only).
    ``from xrlenv.observability.logging import configure_logging`` must
    not transitively import ``prometheus_client`` via the package
    ``__init__`` — otherwise the stub crashes at startup with
    ``ModuleNotFoundError: No module named 'prometheus_client'``.

    Regression for the failure that surfaced during the Scenario-1
    acceptance smoke (sandbox stub couldn't bind its uds because the
    package import fanned out to the metrics module).
    """
    result = _run_in_fresh_interpreter(
        """
        import sys
        # Block prometheus_client at import time — mimics the slim image.
        sys.modules["prometheus_client"] = None  # type: ignore[assignment]

        from xrlenv.observability.logging import configure_logging  # noqa: F401
        # Resolving the parent package by NAME must also be safe (the
        # stub does this implicitly — Python loads __init__.py before
        # the submodule binding).
        import xrlenv.observability  # noqa: F401
        # Touching metrics-related names *should* try to load prometheus
        # and fail — proves the laziness wasn't accidentally bypassed.
        try:
            xrlenv.observability.MetricsRegistry  # noqa: B018
        except (ImportError, TypeError):
            pass
        else:
            print("FAIL: prometheus_client was reachable")
            sys.exit(1)
        print("ok")
        """
    )
    assert result.returncode == 0, (
        "stub-style import path failed in a fresh interpreter — the "
        "observability package is eagerly importing metrics again.\n"
        f"stderr:\n{result.stderr}"
    )
    assert result.stdout.strip() == "ok"


def test_plugin_adapter_imports_without_pulling_control_plane() -> None:
    """The in-sandbox stub loads a plug-in's ``adapter.py``; the
    adapter's typing imports (``from xrlenv.control.instance_resolver
    import VerifierUpload``) must not transitively pull
    ``prometheus_client``, ``docker``, or other host-only deps via
    the package ``__init__`` — those modules aren't installed in
    benchmark images.

    Regression for the failure caught during the tb2 ``--local``
    smoke (P1.4): ``xrlenv/control/__init__.py`` eagerly re-exported
    ``RolloutCoordinator``, which dragged in
    ``xrlenv.observability.metrics`` → ``prometheus_client`` whenever
    a sandbox-side import touched any ``xrlenv.control.<submodule>``
    path. The fix is the lazy ``__getattr__`` pattern in
    ``xrlenv/control/__init__.py`` — same shape as
    ``xrlenv/__init__.py``.

    Post-P1.7.D the in-tree tb2 adapter is gone; this test now
    exercises the same property via ``xrlenv.templates.hello_shell``
    (the in-repo case-1 worked template — same EnvAdapter Protocol
    shape, same risk of accidentally pulling in control-plane deps
    through typing-only imports).
    """
    result = _run_in_fresh_interpreter(
        """
        import sys
        # Block both host-only deps; the slim sandbox image has neither.
        sys.modules["prometheus_client"] = None  # type: ignore[assignment]
        sys.modules["docker"] = None  # type: ignore[assignment]

        # The exact import chain the stub triggers when loading a
        # plug-in adapter inside the sandbox. hello_shell is the
        # in-repo case-1 worked template under the slim pivot.
        from xrlenv.templates.hello_shell.adapter import (  # noqa: F401
            ShellEnvAdapter,
        )
        # Re-resolving the control-plane parent package by NAME must
        # also be safe (Python loads __init__.py before the submodule
        # binding).
        import xrlenv.control  # noqa: F401
        # Touching a name that LIVES in the control plane SHOULD trigger
        # the heavy chain — that proves laziness isn't accidentally
        # bypassed by the sandbox-side imports.
        try:
            xrlenv.control.RolloutCoordinator  # noqa: B018
        except (ImportError, TypeError):
            pass
        else:
            print("FAIL: control-plane chain was reachable from blocked deps")
            sys.exit(1)
        print("ok")
        """
    )
    assert result.returncode == 0, (
        "Plug-in adapter import path triggered host-only deps in a "
        "fresh interpreter — xrlenv.control.__init__.py is eagerly "
        "re-exporting again.\n"
        f"stderr:\n{result.stderr}"
    )
    assert result.stdout.strip() == "ok"


def test_distributed_runtime_model_resolves_in_fresh_interpreter() -> None:
    """Pydantic must be able to construct ``DistributedRuntime`` without the
    test-suite's fixture chain warming the ``AdminServer`` import.

    Regression for an issue where ``AdminServer`` was guarded under
    ``if TYPE_CHECKING:``; Pydantic v2 needs the class at runtime to
    resolve the ``admin_server: SkipValidation[AdminServer | None]``
    annotation, so a fresh ``python tests/smoke/cluster_bringup/cluster_smoke.py``
    invocation crashed with ``PydanticUserError: not fully defined``
    even though the unit suite passed.
    """
    result = _run_in_fresh_interpreter(
        """
        # Importing build_distributed_runtime forces Pydantic to resolve
        # the DistributedRuntime model's annotations at module-load time.
        from xrlenv.control.distributed_runtime import (
            DistributedRuntime,  # noqa: F401
            build_distributed_runtime,  # noqa: F401
        )
        # Probe the model_fields dict — accessing it triggers the same
        # forward-reference resolution path that ``DistributedRuntime(...)``
        # construction does, without needing real component instances.
        assert "admin_server" in DistributedRuntime.model_fields
        print("ok")
        """
    )
    assert result.returncode == 0, (
        "DistributedRuntime model failed to resolve in a fresh interpreter — "
        "an annotation is referencing a class that's not in scope at runtime "
        "(check TYPE_CHECKING-guarded imports of model field types).\n"
        f"stderr:\n{result.stderr}"
    )
    assert result.stdout.strip() == "ok"


def test_import_xrlenv_auto_loads_dotenv_from_cwd(tmp_path: Path) -> None:
    """Regression for the operator-UX promise that ``import xrlenv``
    populates ``os.environ`` from the nearest ``.env``. The other
    unit tests exercise ``load_dotenv()`` directly but run with
    ``XRLENV_DOTENV=off`` (set by tests/conftest.py for isolation),
    so the actual import-time hook isn't covered by them. This test
    spawns a fresh interpreter with ``.env`` in the run dir and
    asserts the auto-load fires end-to-end:

    - ``TEST_XRLENV_AUTOLOAD`` from ``.env`` lands in ``os.environ``.
    - A pre-set shell env var still wins (``.env`` is a fallback).
    - The auto-load doesn't transitively pull in ``docker`` /
      ``grpc`` / ``prometheus_client`` (slim-image contract from the
      sibling tests above).
    """
    import os as _os

    env_file = tmp_path / ".env"
    env_file.write_text(
        "TEST_XRLENV_AUTOLOAD=from-dotenv\n"
        "TEST_XRLENV_SHELL_WINS=this-should-lose\n",
        encoding="utf-8",
    )
    # Inherit current env but flip off the conftest's
    # ``XRLENV_DOTENV=off`` so the import-time hook actually runs.
    # Pre-set the shell-wins var so the .env value gets ignored.
    env = {
        k: v for k, v in _os.environ.items() if k != "XRLENV_DOTENV"
    }
    env["TEST_XRLENV_SHELL_WINS"] = "from-shell"

    result = subprocess.run(
        [
            sys.executable, "-c",
            textwrap.dedent(
                """
                import os
                import sys
                # Block heavy deps to also verify the auto-load doesn't
                # widen the import surface (slim-image contract).
                sys.modules["docker"] = None  # type: ignore[assignment]
                sys.modules["grpc"] = None    # type: ignore[assignment]
                sys.modules["prometheus_client"] = None  # type: ignore[assignment]

                import xrlenv  # triggers _maybe_auto_load_dotenv  # noqa: F401

                print("AUTO=" + repr(os.environ.get("TEST_XRLENV_AUTOLOAD")))
                print("WINS=" + repr(os.environ.get("TEST_XRLENV_SHELL_WINS")))
                print("ok")
                """
            ),
        ],
        capture_output=True, text=True, check=False,
        cwd=str(tmp_path),
        env=env,
    )
    assert result.returncode == 0, (
        f"import xrlenv failed in fresh interpreter.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    body = result.stdout
    assert "AUTO='from-dotenv'" in body, (
        f".env value didn't reach os.environ.\n{body}"
    )
    assert "WINS='from-shell'" in body, (
        f"shell-set value should have won over .env.\n{body}"
    )
    assert body.strip().endswith("ok")

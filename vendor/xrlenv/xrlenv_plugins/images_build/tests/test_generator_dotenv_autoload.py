"""Regression test for the 2026-05-11 audit M1 finding.

The audit confirmed that running ``python -m
xrlenv_plugins.images_build.<generator>.build_plan_gen`` did NOT
trigger the ``xrlenv/__init__.py`` import-time ``.env`` loader,
because the generator never imports ``xrlenv``. Operators following
the docs ("set ``DOCKERHUB_USER`` + ``DOCKERHUB_TOKEN`` in ``.env``")
got an "unauthenticated" auth banner and the rate-limit fallback the
docs were trying to prevent.

The fix calls ``_maybe_auto_load_dotenv()`` at the top of each
generator's ``main()``. This test spawns a fresh subprocess with
``DOCKERHUB_*`` scrubbed from the env, a ``.env`` containing those
vars in ``cwd``, runs the generator with ``--no-probe`` so no Hub
round-trip happens, and asserts the loaded values show up in the
subprocess's ``os.environ`` after ``main()`` returns.

A fresh subprocess (rather than ``monkeypatch`` + reset-the-flag
in-process) is the right call: the dotenv loader writes directly to
``os.environ`` and the module-level ``_AUTO_LOADED`` gate is shared
across the whole pytest process; isolating to a subprocess avoids
cross-test pollution and matches the existing pattern at
``tests/unit/test_import_cycles.py::test_import_xrlenv_auto_loads_dotenv_from_cwd``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_GENERATORS_TO_VERIFY = (
    "xrlenv_plugins.images_build.swebench_verified.build_plan_gen",
    "xrlenv_plugins.images_build.terminal_bench_2.build_plan_gen",
    # terminalworld's generator lives with the rest of its benchmark tooling
    # (self-contained under benchmarks/terminalworld), but shares this dotenv contract.
    "xrlenv_plugins.benchmarks.terminalworld.build_plan_gen",
)


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _spawn_generator_check(
    tmp_path: Path,
    *,
    generator_module: str,
    generator_args: list[str],
    extra_env: dict[str, str] | None = None,
) -> tuple[str, str, int]:
    """Run ``python -c "<probe>"`` in a fresh subprocess where the
    ``<probe>`` imports the generator's ``main`` after a ``.env`` has
    been placed in ``tmp_path``, invokes ``main()``, then prints the
    post-call values of ``DOCKERHUB_USER`` / ``DOCKERHUB_TOKEN`` on
    stdout. Returns (stdout, stderr, returncode).

    ``PYTHONPATH`` is pinned to the repo root so the subprocess
    imports the current source — not any stale ``xrlenv_plugins/``
    copy that might live in ``site-packages`` from an earlier non-
    editable install. Without this, the test would silently pass or
    fail depending on whether the developer's ``.venv`` is fresh.
    """
    (tmp_path / ".env").write_text(
        "DOCKERHUB_USER=fresh-test-user\n"
        "DOCKERHUB_TOKEN=dckr_pat_fresh\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "plan.yaml"
    args = [
        arg.replace("__OUT_PATH__", str(out_path))
        for arg in generator_args
    ]

    probe_script = textwrap.dedent(f"""
        import os, sys
        from {generator_module} import main
        rc = main({args!r})
        print(
            "POST_MAIN "
            f"USER={{os.environ.get('DOCKERHUB_USER', '<unset>')}} "
            f"TOKEN={{os.environ.get('DOCKERHUB_TOKEN', '<unset>')}}"
        )
        sys.exit(rc)
    """)

    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("DOCKERHUB_")
    }
    env["XRLENV_DOTENV"] = "on"
    env["PYTHONPATH"] = (
        f"{_REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    )
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        [sys.executable, "-c", probe_script],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    return result.stdout, result.stderr, result.returncode


def test_swebench_generator_main_loads_dotenv(tmp_path: Path) -> None:
    stdout, stderr, rc = _spawn_generator_check(
        tmp_path,
        generator_module=(
            "xrlenv_plugins.images_build.swebench_verified.build_plan_gen"
        ),
        generator_args=[
            "--no-probe",
            "--instances", "django__django-11099",
            "--output", "__OUT_PATH__",
        ],
    )
    assert rc == 0, f"generator exited {rc}; stderr=\n{stderr}"
    assert "POST_MAIN USER=fresh-test-user TOKEN=dckr_pat_fresh" in stdout, (
        f"DOCKERHUB_* env vars were not present after main(); the "
        f".env auto-load did not fire.\nstdout={stdout!r}\n"
        f"stderr={stderr!r}"
    )


def test_terminal_bench_2_generator_main_loads_dotenv(
    tmp_path: Path,
) -> None:
    stdout, stderr, rc = _spawn_generator_check(
        tmp_path,
        generator_module=(
            "xrlenv_plugins.images_build.terminal_bench_2.build_plan_gen"
        ),
        generator_args=[
            "--no-probe",
            "--tasks", "fix-git",
            "--output", "__OUT_PATH__",
        ],
    )
    assert rc == 0, f"generator exited {rc}; stderr=\n{stderr}"
    assert "POST_MAIN USER=fresh-test-user TOKEN=dckr_pat_fresh" in stdout, (
        f"DOCKERHUB_* env vars were not present after main(); the "
        f".env auto-load did not fire.\nstdout={stdout!r}\n"
        f"stderr={stderr!r}"
    )


def test_terminalworld_generator_main_loads_dotenv(
    tmp_path: Path,
) -> None:
    # terminalworld's generator emits type: local entries from the harbor cache,
    # so point XRLENV_BENCHMARK_CACHE at a self-contained fixture shard with one
    # buildable task (avoids depending on a populated real cache).
    shard = tmp_path / "cache" / "terminalworld-verified" / "tw_probe" / "environment"
    shard.mkdir(parents=True)
    (shard / "Dockerfile").write_text("FROM ubuntu:22.04\n", encoding="utf-8")

    stdout, stderr, rc = _spawn_generator_check(
        tmp_path,
        generator_module=(
            "xrlenv_plugins.benchmarks.terminalworld.build_plan_gen"
        ),
        generator_args=[
            "--tasks", "tw_probe",
            "--output", "__OUT_PATH__",
        ],
        extra_env={"XRLENV_BENCHMARK_CACHE": str(tmp_path / "cache")},
    )
    assert rc == 0, f"generator exited {rc}; stderr=\n{stderr}"
    assert "POST_MAIN USER=fresh-test-user TOKEN=dckr_pat_fresh" in stdout, (
        f"DOCKERHUB_* env vars were not present after main(); the "
        f".env auto-load did not fire.\nstdout={stdout!r}\nstderr={stderr!r}"
    )


def test_xrlenv_dotenv_off_disables_generator_autoload(
    tmp_path: Path,
) -> None:
    """The opt-out must work — if the operator set ``XRLENV_DOTENV=off``
    in their shell, the generator's main() must respect it and NOT
    overwrite the operator's shell-only env.
    """
    (tmp_path / ".env").write_text(
        "DOCKERHUB_USER=should-not-load\n"
        "DOCKERHUB_TOKEN=should-not-load\n",
        encoding="utf-8",
    )

    probe_script = textwrap.dedent("""
        import os, sys
        from xrlenv_plugins.images_build.swebench_verified.build_plan_gen \
            import main
        rc = main([
            "--no-probe", "--instances", "django__django-11099",
            "--output", "/dev/null",
        ])
        print(
            "POST_MAIN "
            f"USER={os.environ.get('DOCKERHUB_USER', '<unset>')} "
            f"TOKEN={os.environ.get('DOCKERHUB_TOKEN', '<unset>')}"
        )
        sys.exit(rc)
    """)

    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("DOCKERHUB_")
    }
    env["XRLENV_DOTENV"] = "off"
    env["PYTHONPATH"] = (
        f"{_REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe_script],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"generator exited {result.returncode}; stderr=\n{result.stderr}"
    )
    assert "POST_MAIN USER=<unset> TOKEN=<unset>" in result.stdout, (
        f"XRLENV_DOTENV=off was not honored — the generator loaded "
        f".env despite the opt-out.\nstdout={result.stdout!r}\n"
        f"stderr={result.stderr!r}"
    )

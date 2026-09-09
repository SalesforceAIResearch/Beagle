"""Tests for the repository installer."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_installer_uses_available_uv_from_repository_root(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "uv-call.txt"
    uv = bin_dir / "uv"
    uv.write_text(
        '#!/bin/sh\nprintf "%s\\n%s\\n" "$PWD" "$*" > "$UV_TEST_LOG"\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["UV_TEST_LOG"] = str(log)

    subprocess.run(
        ["bash", str(ROOT / "scripts/install.sh")],
        cwd=tmp_path,
        env=env,
        check=True,
    )

    assert log.read_text(encoding="utf-8").splitlines() == [
        str(ROOT),
        "sync --extra all",
    ]


def test_installer_bootstraps_uv_when_missing(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "uv-call.txt"
    curl = bin_dir / "curl"
    curl.write_text(
        """\
#!/bin/sh
cat <<'INSTALLER'
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/uv" <<'UV'
#!/bin/sh
printf "%s\\n%s\\n" "$PWD" "$*" > "$UV_TEST_LOG"
UV
chmod +x "$HOME/.local/bin/uv"
INSTALLER
""",
        encoding="utf-8",
    )
    curl.chmod(0o755)
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin"
    env["UV_TEST_LOG"] = str(log)

    subprocess.run(["bash", str(ROOT / "scripts/install.sh")], env=env, check=True)

    assert log.read_text(encoding="utf-8").splitlines() == [
        str(ROOT),
        "sync --extra all",
    ]

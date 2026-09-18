"""Tests for the batch onboarding wrapper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_batch_onboarding_uses_project_python_from_any_working_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "onboard_all_agents.sh"
    source = (ROOT / "scripts/onboard_all_agents.sh").read_text(encoding="utf-8")
    script.write_text(source, encoding="utf-8")
    (repo / ".env").write_text("YOUR_ORG=test-org\n", encoding="utf-8")
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        '#!/bin/sh\nprintf "%s|%s\\n" "$PWD" "$*" >> "$PYTHON_TEST_LOG"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    log = tmp_path / "python-calls.txt"
    env = os.environ.copy()
    env["PYTHON_TEST_LOG"] = str(log)

    subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, check=True)

    calls = log.read_text(encoding="utf-8").splitlines()
    # One call per invocation the script actually carries — counted from the script, not hardcoded.
    # A literal goes stale whenever an agent is added or commented out, and is simply WRONG in the
    # public mirror, where publication removes the marked block wrapping the unpublished agents.
    # Counting still guards what this test is for: every invocation ran, none silently skipped.
    expected = source.count("-m beagle.tools.onboard")
    assert expected >= 2, "the batch script lost its onboard invocations"
    assert len(calls) == expected
    assert all(call.startswith(f"{repo}|-m beagle.tools.onboard ") for call in calls)

"""Packaging metadata must reference files included in distributions."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_project_readme_exists() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^readme = "([^"]+)"$', pyproject, re.MULTILINE)

    assert match is not None
    assert (ROOT / match.group(1)).is_file()

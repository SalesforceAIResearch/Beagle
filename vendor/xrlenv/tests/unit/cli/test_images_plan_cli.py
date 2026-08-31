"""P1.7.B.2 W6 — `xrlenv images plan` operator CLI handler.

Pins the input-parsing + error-handling behaviour. The actual
gRPC dial path is exercised end-to-end via the smoke; here we
verify the CLI's argument validation, refs-file parsing, and
exit codes.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from xrlenv.cli.images_plan_cmd import _parse_refs_file, cmd_images_plan


def test_refs_file_parses_inline_size_hints(tmp_path: Path) -> None:
    refs = tmp_path / "refs.txt"
    refs.write_text(
        "# comment line\n"
        "first:1\n"
        "second:1\t1073741824\n"
        "\n"
        "third:1 5368709120\n",
    )
    rows = _parse_refs_file(refs, default_size=99)
    assert rows == [
        ("first:1", 99),
        ("second:1", 1073741824),
        ("third:1", 5368709120),
    ]


def test_refs_file_falls_back_to_default_for_bad_size(
    tmp_path: Path,
) -> None:
    refs = tmp_path / "refs.txt"
    refs.write_text("img:1 not-a-number\n")
    rows = _parse_refs_file(refs, default_size=42)
    assert rows == [("img:1", 42)]


def test_cli_rejects_no_refs_or_inline(tmp_path: Path) -> None:
    out = io.StringIO()
    code = cmd_images_plan(
        refs_file=None, refs_inline=[], default_size_bytes=1024,
        eager_prefetch=False, control_host="127.0.0.1",
        control_port=50051, operator_token="tok", out=out,
    )
    assert code == 2


def test_cli_rejects_both_refs_and_inline(tmp_path: Path) -> None:
    refs = tmp_path / "refs.txt"
    refs.write_text("a:1\n")
    out = io.StringIO()
    code = cmd_images_plan(
        refs_file=refs, refs_inline=["b:1"], default_size_bytes=1024,
        eager_prefetch=False, control_host="127.0.0.1",
        control_port=50051, operator_token="tok", out=out,
    )
    assert code == 2


def test_cli_requires_operator_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With operator_token=None and no XRLENV_OPERATOR_TOKEN env-var
    fallback, the CLI must refuse with code 2 — not silently dial the
    cluster. delenv is required because cmd_images_plan documents an
    env-var fallback (line ``operator_token or os.environ.get(...)``);
    without clearing the fallback, this test would pass spuriously
    when the operator's shell happens to have the token exported."""
    monkeypatch.delenv("XRLENV_OPERATOR_TOKEN", raising=False)

    refs = tmp_path / "refs.txt"
    refs.write_text("a:1\n")
    out = io.StringIO()
    code = cmd_images_plan(
        refs_file=refs, refs_inline=[], default_size_bytes=1024,
        eager_prefetch=False, control_host="127.0.0.1",
        control_port=50051, operator_token=None, out=out,
    )
    assert code == 2


def test_cli_rejects_empty_refs_file(tmp_path: Path) -> None:
    refs = tmp_path / "refs.txt"
    refs.write_text("# only comments\n")
    out = io.StringIO()
    code = cmd_images_plan(
        refs_file=refs, refs_inline=[], default_size_bytes=1024,
        eager_prefetch=False, control_host="127.0.0.1",
        control_port=50051, operator_token="tok", out=out,
    )
    assert code == 2

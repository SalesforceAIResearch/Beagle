"""Guard the xrlenv-AUTHORED notebook-compression solution (frontier-swe).

Not an upstream oracle: FrontierSWE withholds the reference, so this is an
xrlenv-authored `/app/run` submission (a lossless lzma compressor). Correctness
(byte-for-byte round-trip) is the task's hard gate, so this test pins exactly that —
plus the one-to-one path mapping and that decompress is self-contained (needs only
the artifact + compressed dirs). Offline; no cluster.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Load the authored /app/run submission (patches/notebook-compression/solution/run.py).
_RUN = (
    Path(__file__).resolve().parents[1]
    / "patches" / "notebook-compression" / "solution" / "run.py"
)


def _load_run():
    spec = importlib.util.spec_from_file_location("nbc_run", _RUN)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_authored_solution_present_and_labelled() -> None:
    assert _RUN.is_file(), "authored run.py overlay missing"
    solve = _RUN.parent / "solve.sh"
    assert solve.is_file()
    banner = solve.read_text().lower()
    # must be loudly labelled as authored, not an upstream oracle
    assert "authored" in banner and "not the upstream oracle" in banner
    # installs the task's required /app/run entry point
    assert "/app/run" in solve.read_text()


def _make_notebook(path: Path, n: int) -> None:
    import json
    cells = [
        {"cell_type": "code", "source": [f"x = {i}\nprint(x * {i})\n"],
         "metadata": {}, "outputs": [], "execution_count": i}
        for i in range(n)
    ]
    doc = {"cells": cells, "metadata": {"kernelspec": {"name": "python3"}},
           "nbformat": 4, "nbformat_minor": 5}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))


def test_round_trip_is_byte_for_byte_lossless(tmp_path: Path) -> None:
    run = _load_run()
    visible = tmp_path / "visible"
    inp = tmp_path / "input"
    art = tmp_path / "artifact"
    comp = tmp_path / "compressed"
    rec = tmp_path / "recovered"
    for d in (visible, inp, art, comp, rec):
        d.mkdir(parents=True, exist_ok=True)
    _make_notebook(visible / "v1.ipynb", 10)
    _make_notebook(inp / "a.ipynb", 20)
    _make_notebook(inp / "nested" / "b.ipynb", 40)  # nested path preserved

    assert run.main(["run", "fit", str(visible), str(art)]) == 0
    assert run.main(["run", "compress", str(art), str(inp), str(comp)]) == 0
    assert run.main(["run", "decompress", str(art), str(comp), str(rec)]) == 0

    # byte-for-byte identical + same relative paths (the task's hard gate)
    orig = {p.relative_to(inp): p.read_bytes() for p in inp.rglob("*") if p.is_file()}
    back = {p.relative_to(rec): p.read_bytes() for p in rec.rglob("*") if p.is_file()}
    assert back == orig

    # actually compresses (score < 1 → reward = 1 - score > 0)
    comp_bytes = sum(p.stat().st_size for p in comp.rglob("*") if p.is_file())
    orig_bytes = sum(len(v) for v in orig.values())
    assert comp_bytes < orig_bytes


def test_fit_arity_two_args(tmp_path: Path) -> None:
    # regression: fit takes 2 path args (<visible> <artifact>), not 3.
    run = _load_run()
    (tmp_path / "v").mkdir()
    assert run.main(["run", "fit", str(tmp_path / "v"), str(tmp_path / "a")]) == 0
    # unknown stage / missing args fail loud (non-zero), never silently pass
    assert run.main(["run", "bogus", "x", "y", "z"]) == 2
    assert run.main(["run", "compress", "only-one"]) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

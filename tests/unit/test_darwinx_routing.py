"""DarwinX integration routing — seam A (runner.run shim → our eval) + seam B (the vendored
meta_agent delegates to the injected beagle Editor). Hermetic: no cluster, no pipeline."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import beagle
from beagle.agents.core.base import EditResult
from beagle.algorithms.darwinx import meta_agent as shim

_DARWINX = Path(beagle.__file__).parent / "algorithms" / "darwinx"


def test_seam_a_runner_shim_routes_to_eval(monkeypatch, tmp_path) -> None:
    """`python -m runner.run <cfg> --results-root <dir> …` → our run_eval, extra flags tolerated."""
    seen: dict = {}

    def _fake_run_eval(config, *, results_root, run_id, include_task_name, campaign_id):  # noqa: ANN001
        seen.update(config=config, results_root=str(results_root), run_id=run_id,
                    include_task_name=include_task_name, campaign_id=campaign_id)
        return tmp_path / "out" / "runs" / "RID"

    monkeypatch.setattr("beagle.algorithms.darwinx.eval.run_eval", _fake_run_eval)

    shims = str(_DARWINX / "_shims")
    sys.path.insert(0, shims)
    for m in ("runner.run", "runner"):
        sys.modules.pop(m, None)
    try:
        import runner.run as runner_run

        rc = runner_run.main(["cfg.yaml", "--results-root", str(tmp_path), "--include-task-name",
                              "adaptive-rejection-sampler", "--campaign-id", "camp",
                              "--some-unknown-flag", "ignored"])
    finally:
        sys.path.remove(shims)
        for m in ("runner.run", "runner"):
            sys.modules.pop(m, None)

    assert rc == 0
    assert seen == {"config": "cfg.yaml", "results_root": str(tmp_path), "run_id": None,
                    "include_task_name": ["adaptive-rejection-sampler"], "campaign_id": "camp"}


def _load_vendored_meta_agent():
    path = _DARWINX / "vendor" / "evolve" / "meta_agent.py"
    spec = importlib.util.spec_from_file_location("_vendored_meta_agent", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_seam_b_vendored_meta_agent_delegates_to_injected_editor(tmp_path) -> None:
    """The (replaced) vendored `meta_agent.run` routes DarwinX's proposer call to the injected
    beagle Editor; `active_backend` reports its name. Extra kwargs are dropped."""
    meta_agent = _load_vendored_meta_agent()
    seen: dict = {}

    class _FakeEditor:
        name = "cursor"

        def edit(self, instruction, workspace, *, plan_mode, model, timeout_s, extra_args, log_path=None):  # noqa: ANN001
            seen.update(instruction=instruction, plan_mode=plan_mode, model=model)
            return EditResult(text="ok")

    shim.set_editor(_FakeEditor())
    try:
        # call sites pass (prompt, workspace) + keywords incl. ones the Editor doesn't take
        res = meta_agent.run("analyze this", tmp_path / "ws", plan_mode=True, model="m",
                             reasoning_effort="high", log_path="/tmp/x")
        assert isinstance(res, EditResult) and res.text == "ok"
        assert seen == {"instruction": "analyze this", "plan_mode": True, "model": "m"}
        assert meta_agent.active_backend() == "cursor"
    finally:
        shim.set_editor(None)


def test_seam_b_vendored_meta_agent_falls_back_to_standalone_without_an_editor() -> None:
    """No editor injected means "not hosted", and the vendored module says so.

    This assertion used to read ``== "beagle"``, which was right when beagle *replaced* this
    file with its own shim: the replacement had no standalone mode, so beagle was the only
    possible answer. The file is now DarwinX's own module, injectable rather than replaced, and
    the same copy runs standalone in the pipeline -- where reporting "beagle" would be a lie.

    The safety property that actually matters is not what this reports when uninjected; it is
    that beagle's own shim refuses to run at all without an editor, so a hosted run can never
    quietly fall through to a different proposer than the host configured. That is asserted
    directly below.
    """
    meta_agent = _load_vendored_meta_agent()
    shim.set_editor(None)
    assert meta_agent.active_backend() == "cursor"


def test_seam_b_shim_is_fail_closed_without_an_editor(tmp_path) -> None:
    """The invariant worth protecting: hosted + uninjected must raise, never guess a backend."""
    shim.set_editor(None)
    with pytest.raises(RuntimeError, match="no Editor injected"):
        shim.run("do something", tmp_path / "ws")


def test_seam_b_algorithm_injects_before_running() -> None:
    """DarwinX.evolve must wire the run's evolver into the shim, or the above raise fires."""
    source = (_DARWINX / "algorithm.py").read_text()
    assert "shim.set_editor(" in source

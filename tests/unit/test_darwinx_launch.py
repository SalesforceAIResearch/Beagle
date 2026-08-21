"""DarwinX launch routing (seam-level, hermetic) — the wiring in ``_launch`` + ``DarwinX.evolve``
that turns an beagle run into a vendored-driver launch, with NO paid work.

The pipeline's ``.run()`` is monkeypatched to a no-op, so these assert only the routing:
import path prepared, campaign config emitted (seam C), evolver Editor injected (seam B),
``PipelineConfig`` built correctly, and the winner read back from the genealogy DB.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from beagle.agents.core.spec import AgentSource, AgentSpec, ModelSpec
from beagle.algorithms.base import CandidateStatus
from beagle.algorithms.darwinx import _launch
from beagle.algorithms.darwinx import algorithm as dx
from beagle.algorithms.darwinx import meta_agent
from beagle.algorithms.darwinx._launch import EVOLVER_SPEC_KEY
from beagle.config import RunConfig


# --- fakes -------------------------------------------------------------------

class _FakeEvolvee:
    def __init__(self) -> None:
        self.spec = AgentSpec(
            name="target", model=ModelSpec(name="target-model"),
            config={"max_turns": 40, "timeout": 9000},
            source=AgentSource(repo="https://example.test/exp-copy", ref="baseline"),
        )

    @property
    def name(self) -> str:
        return self.spec.name

    def source(self) -> AgentSource:
        return self.spec.source


class _FakeEvolver:
    def __init__(self) -> None:
        self.spec = AgentSpec(name="cursor", model=ModelSpec(name="auto"), config={"timeout": 900})

    @property
    def name(self) -> str:
        return self.spec.name

    def can_be_evolver(self) -> bool:
        return True

    def edit(self, *a, **k):  # noqa: ANN002, ANN003
        raise AssertionError("hermetic launch smoke must not call the evolver")


def _run_config() -> RunConfig:
    return RunConfig.from_dict({
        "model": {"name": "target-model"},
        "agent": {"name": "target"},
        "benchmark": {"name": "terminal-bench-2-1", "task_ids": ["t1", "t2"], "dataset": "tb2"},
        "runtime": {"kind": "local"},
    })


# --- prepare_import_path -----------------------------------------------------

def test_prepare_import_path_makes_driver_and_shim_importable() -> None:
    import os
    import sys

    _launch.prepare_import_path()
    _launch.prepare_import_path()  # idempotent — no duplicate entries
    assert sum(p.endswith("/darwinx/_shims") for p in sys.path) == 1
    assert sum(p.endswith("/darwinx/vendor") for p in sys.path) == 1

    import runner.run as runner_run          # seam-A shim resolves (this process)
    import self_evolve.pipeline as pipe       # vendored driver resolves
    assert hasattr(pipe, "SelfEvolvePipeline") and hasattr(runner_run, "main")

    # PYTHONPATH carries both so the eval *subprocess* (`python -m runner.run`) resolves the
    # shim; _shims precedes vendor so our runner.run shadows any other, and no duplicates.
    pp = os.environ["PYTHONPATH"].split(os.pathsep)
    shims = next(i for i, p in enumerate(pp) if p.endswith("/darwinx/_shims"))
    vendor = next(i for i, p in enumerate(pp) if p.endswith("/darwinx/vendor"))
    assert shims < vendor
    assert pp.count(pp[shims]) == 1


# --- evolvee canonical clone + worktree paths --------------------------------

def test_ensure_canonical_clone_symlinks_local_checkout(tmp_path) -> None:
    checkout = tmp_path / "beagle-monet_code"   # the existing experiment copy (bare monet_code)
    (checkout / ".git").mkdir(parents=True)
    repo_root = tmp_path / "repo"

    canonical = _launch.ensure_canonical_clone(repo_root, checkout)
    assert canonical == repo_root / "monet_code"
    assert canonical.is_symlink() and (canonical / ".git").exists()   # driver's check passes
    assert canonical.resolve() == checkout.resolve()


def test_ensure_canonical_clone_accepts_existing_without_checkout(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "monet_code" / ".git").mkdir(parents=True)
    assert _launch.ensure_canonical_clone(repo_root) == repo_root / "monet_code"


def test_ensure_canonical_clone_requires_a_source(tmp_path) -> None:
    with pytest.raises(ValueError, match="standalone evolvee git clone"):
        _launch.ensure_canonical_clone(tmp_path / "repo")   # nothing there, no checkout given


def test_prepare_worktree_env_pins_vendored_overrides(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MONET_EVAL_REPO_ROOT", raising=False)
    monkeypatch.delenv("MONET_EVAL_WORKTREE_PARENT", raising=False)
    monkeypatch.delenv("MONET_EVAL_RESULTS_ROOT", raising=False)
    _launch.prepare_worktree_env(tmp_path / "repo", tmp_path / "wt")
    import os
    assert os.environ["MONET_EVAL_REPO_ROOT"] == str((tmp_path / "repo").resolve())
    assert os.environ["MONET_EVAL_WORKTREE_PARENT"] == str((tmp_path / "wt").resolve())
    # eval scratch lands UNDER the run dir (_evals/), not the vendored tree's DEFAULT_RESULTS_ROOT
    assert os.environ["MONET_EVAL_RESULTS_ROOT"] == str((tmp_path / "repo").resolve() / "_evals")
    assert (tmp_path / "wt").is_dir()   # created so the driver can drop worktrees under it


_RUNTIME_ENV_KEYS = ("SELF_EVOLVE_EVAL_RUNTIME", "SELF_EVOLVE_ROOT_COMMIT",
                     "SELF_EVOLVE_ROOT_FETCH_REF", "MONET_EVAL_HARBOR_N_CONCURRENT")


def test_prepare_runtime_env_translates_config_with_sha_seed(monkeypatch) -> None:
    import os

    for k in _RUNTIME_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    src = AgentSource(repo="r", ref="1261608f6530908e3a03218d8f4671b8c7b5b346")  # a bare sha
    _launch.prepare_runtime_env(_run_config(), src)   # _run_config: runtime local, parallelism 1
    assert os.environ["SELF_EVOLVE_EVAL_RUNTIME"] == "local"   # not the driver's cluster default
    assert os.environ["SELF_EVOLVE_ROOT_COMMIT"] == "1261608f6530908e3a03218d8f4671b8c7b5b346"
    assert "SELF_EVOLVE_ROOT_FETCH_REF" not in os.environ      # a bare sha isn't a fetch ref
    assert os.environ["MONET_EVAL_HARBOR_N_CONCURRENT"] == "1"


def test_prepare_runtime_env_branch_ref_is_fetchable(monkeypatch) -> None:
    import os

    for k in _RUNTIME_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    _launch.prepare_runtime_env(_run_config(), AgentSource(repo="r", ref="develop"))
    assert os.environ["SELF_EVOLVE_ROOT_COMMIT"] == "develop"
    assert os.environ["SELF_EVOLVE_ROOT_FETCH_REF"] == "develop"   # a branch → fetchable


_SUBSET_ENV_KEYS = ("MONET_EVAL_EXCLUDE_TASKS", "MONET_EVAL_PRIORITY_TASKS",
                    "MONET_EVAL_VARIANCE_TASKS", "MONET_EVAL_FULLSET_TASKS")


def test_prepare_task_subset_env_translates_selectors(monkeypatch) -> None:
    import os

    for k in _SUBSET_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    cfg = RunConfig.from_dict({
        "model": {"name": "m"}, "agent": {"name": "target"},
        "benchmark": {"name": "b", "task_ids": ["t1", "t2", "t3"],
                      "exclude_task_ids": ["flaky-a", "flaky-b"],
                      "options": {"priority_tasks": ["hard-1"], "variance_tasks": ["noisy"],
                                  "fullset_tasks": "t1,t2,t3"}},   # list OR bare csv both accepted
    })
    out = _launch.prepare_task_subset_env(cfg)
    assert out == {
        "MONET_EVAL_EXCLUDE_TASKS": "flaky-a,flaky-b",   # from benchmark.exclude_task_ids
        "MONET_EVAL_PRIORITY_TASKS": "hard-1",           # from benchmark.options
        "MONET_EVAL_VARIANCE_TASKS": "noisy",
        "MONET_EVAL_FULLSET_TASKS": "t1,t2,t3",
    }
    assert os.environ["MONET_EVAL_EXCLUDE_TASKS"] == "flaky-a,flaky-b"   # set on env for the driver


def test_prepare_task_subset_env_empty_when_unset(monkeypatch) -> None:
    for k in _SUBSET_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    # _run_config names task_ids (the primary set, rides subset_tasks) but no exclude/options
    # selectors → nothing emitted, the driver's own stratification defaults stand.
    assert _launch.prepare_task_subset_env(_run_config()) == {}


_GIT_ENV_KEYS = ("GIT_TERMINAL_PROMPT", "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0")


def test_prepare_git_auth_injects_credential_helper(monkeypatch) -> None:
    import os

    for k in _GIT_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MY_GH", "ghp_secret123")
    assert _launch.prepare_git_auth("MY_GH") == "MY_GH"
    assert os.environ["GIT_TERMINAL_PROMPT"] == "0"                 # fail loud, never prompt
    assert os.environ["GIT_CONFIG_COUNT"] == "1"
    assert os.environ["GIT_CONFIG_KEY_0"] == "credential.https://github.com.helper"
    helper = os.environ["GIT_CONFIG_VALUE_0"]
    assert "x-access-token" in helper and "ghp_secret123" in helper  # token inlined (env not forwarded)


def test_prepare_git_auth_noop_without_token(monkeypatch) -> None:
    import os

    for k in _GIT_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("NOPE_TOKEN", raising=False)
    assert _launch.prepare_git_auth("NOPE_TOKEN") is None
    assert os.environ["GIT_TERMINAL_PROMPT"] == "0"                 # still disables the prompt
    assert "GIT_CONFIG_COUNT" not in os.environ                    # but injects no helper


def test_align_git_origin_repoints_to_manifest_repo(tmp_path) -> None:
    import subprocess

    repo = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", "https://ex/stale"], check=True)

    changed = _launch.align_git_origin(repo, "https://ex/canonical")   # push target ← manifest repo
    assert changed == ("https://ex/stale", "https://ex/canonical")
    got = subprocess.run(["git", "-C", str(repo), "remote", "get-url", "origin"],
                         capture_output=True, text=True).stdout.strip()
    assert got == "https://ex/canonical"
    assert _launch.align_git_origin(repo, "https://ex/canonical") is None   # idempotent


def test_align_git_origin_noop_without_repo_or_url(tmp_path) -> None:
    assert _launch.align_git_origin(tmp_path / "not-a-repo", "https://ex/x") is None
    assert _launch.align_git_origin(tmp_path, "") is None


# --- emit_campaign_config (seam C) ------------------------------------------

def test_emit_campaign_config_shape_and_evolver_roundtrip(tmp_path) -> None:
    dest = _launch.emit_campaign_config(
        evolvee=_FakeEvolvee(), evolver=_FakeEvolver(), run_config=_run_config(),
        dest=tmp_path / "camp.yaml",
    )
    doc = yaml.safe_load(dest.read_text())

    assert doc["cursor_agent"]["model"] == "auto"                 # evolver proposer model
    # evolvee turn-budget + per-rollout timeout ride the monet block → build_codingbench_config
    assert doc["monet"] == {"model": "target-model", "max_turns": 40, "timeout": 9000}
    assert doc["benchmark"]["name"] == "terminal-bench-2-1"
    assert doc["benchmark"]["dataset"] == "tb2"
    assert doc["benchmark"]["repo_url"] == "https://example.test/exp-copy"  # evolvee θ
    assert doc["benchmark"]["agent_ref"] == "baseline"
    assert "task_ids" not in doc["benchmark"]                     # tasks ride subset_tasks, not here
    assert doc["runtime"]["kind"] == "local"

    # Seam C: the stashed evolver block rebuilds the Editor per-process (no env var).
    assert doc[EVOLVER_SPEC_KEY]["name"] == "cursor"
    meta_agent.set_editor_from_spec(doc[EVOLVER_SPEC_KEY])
    try:
        assert meta_agent.current_editor().name == "cursor"
    finally:
        meta_agent.set_editor(None)


def test_emit_requires_models(tmp_path) -> None:
    ev = _FakeEvolver()
    ev.spec = AgentSpec(name="cursor", model=None)   # no model
    with pytest.raises(ValueError, match="evolver.*has no model"):
        _launch.emit_campaign_config(evolvee=_FakeEvolvee(), evolver=ev,
                                     run_config=_run_config(), dest=tmp_path / "c.yaml")


# --- build_pipeline_config ---------------------------------------------------

def test_build_pipeline_config_is_drift_proof(tmp_path) -> None:
    _launch.prepare_import_path()
    from self_evolve.pipeline import PipelineConfig

    cfg = _launch.build_pipeline_config(
        PipelineConfig, campaign="camp", reports_root=tmp_path, repo_root=tmp_path,
        config_path=tmp_path / "camp.yaml", evolver_model="auto",
        subset_tasks=["t1", "t2"],
        hparams={"max_loop_iters": 3, "not_a_real_field": 999, "repo_root": "reserved-ignored"},
    )
    assert cfg.campaign == "camp"
    assert cfg.cursor_model == "auto"                # explicit → bypasses the stale default-factory
    assert list(cfg.subset_tasks) == ["t1", "t2"]
    assert cfg.subset_label == "subset"
    assert cfg.max_loop_iters == 3                   # real hparam applied
    assert not hasattr(cfg, "not_a_real_field")      # unknown hparam dropped, no TypeError


def test_build_pipeline_config_defaults_parent_strategy_to_a_registered_one(tmp_path) -> None:
    # the driver's PipelineConfig default is a retired strategy — we fall back to its own canonical
    # DEFAULT_STRATEGY_NAME so `parent_selection.get_strategy` accepts it (the stale default raises).
    _launch.prepare_import_path()
    from self_evolve import parent_selection
    from self_evolve.pipeline import PipelineConfig

    cfg = _launch.build_pipeline_config(
        PipelineConfig, campaign="c", reports_root=tmp_path, repo_root=tmp_path,
        config_path=tmp_path / "c.yaml", evolver_model="x", subset_tasks=None, hparams={})
    parent_selection.get_strategy(cfg.parent_strategy)          # must not raise
    assert cfg.parent_strategy == parent_selection.DEFAULT_STRATEGY_NAME

    override = _launch.build_pipeline_config(
        PipelineConfig, campaign="c", reports_root=tmp_path, repo_root=tmp_path,
        config_path=tmp_path / "c.yaml", evolver_model="x", subset_tasks=None,
        hparams={"parent_strategy": "llm_first"})
    assert override.parent_strategy == "llm_first"             # operator choice wins


# --- run_campaign: the two-phase bootstrap→score→evolve orchestration ---------

import dataclasses as _dc  # noqa: E402


@_dc.dataclass
class _FakeCfg:
    bootstrap_only: bool = False
    parent_id_override: str | None = None


def _record_pipeline(runs: list) -> type:
    class _P:
        def __init__(self, cfg) -> None:  # noqa: ANN001
            self.cfg = cfg

        def run(self) -> int:
            runs.append((self.cfg.bootstrap_only, self.cfg.parent_id_override))
            return 0
    return _P


def test_run_campaign_unscored_root_scores_then_evolves(monkeypatch) -> None:
    monkeypatch.setattr(_launch, "_unscored_root_id", lambda tree, cfg: "root7")  # already-bootstrapped
    runs: list = []
    rc = _launch.run_campaign(_record_pipeline(runs), _FakeCfg(), tree=object())
    assert runs == [(True, "root7"), (False, None)]   # precursor (score) then evolve
    assert rc == 0


def test_run_campaign_fresh_bootstraps_then_scores(monkeypatch) -> None:
    seq = iter([None, "root9"])   # no root yet → first run bootstraps → now an unscored root
    monkeypatch.setattr(_launch, "_unscored_root_id", lambda tree, cfg: next(seq))
    runs: list = []
    rc = _launch.run_campaign(_record_pipeline(runs), _FakeCfg(), tree=object())
    assert runs == [(False, None), (True, "root9"), (False, None)]   # bootstrap, score, evolve
    assert rc == 0


def test_run_campaign_already_scored_evolves_once(monkeypatch) -> None:
    monkeypatch.setattr(_launch, "_unscored_root_id", lambda tree, cfg: None)   # scored/absent
    runs: list = []
    _launch.run_campaign(_record_pipeline(runs), _FakeCfg(), tree=object())
    assert runs == [(False, None)]   # a single evolve run, no precursor


def test_unscored_root_id_none_without_db(tmp_path) -> None:
    from types import SimpleNamespace
    fake_tree = SimpleNamespace(db_path_for=lambda root, camp: tmp_path / "missing.db")
    cfg = SimpleNamespace(reports_root=tmp_path, campaign="c", subset_label="subset")
    assert _launch._unscored_root_id(fake_tree, cfg) is None


# --- read_best ---------------------------------------------------------------

def test_read_best_maps_node_to_candidate(tmp_path) -> None:
    db = tmp_path / "camp.db"
    db.write_text("")   # exists → read_best proceeds to the pool
    node = SimpleNamespace(
        id="n7", branch_name="evolve/n7", commit_sha="abc123", parent_id="n1", score=0.83,
        status="completed", improved_tasks_json='["t1"]', regressed_tasks_json="[]",
        resolved_tasks_json='["t1","t2"]',
    )
    fake_tree = SimpleNamespace(db_path_for=lambda root, camp: db,
                                connect=lambda p: SimpleNamespace(close=lambda: None))
    fake_pool = SimpleNamespace(best_node=lambda conn, *, campaign, subset: node)
    cfg = SimpleNamespace(reports_root=tmp_path, campaign="camp", subset_label="subset")

    cand = _launch.read_best(fake_tree, fake_pool, cfg,
                             evolvee_source=AgentSource(repo="r", ref="baseline"))
    assert cand is not None
    assert cand.id == "n7" and cand.status is CandidateStatus.KEPT and cand.score == 0.83
    assert cand.source.ref == "evolve/n7" and cand.source.repo == "r"  # θ at the winning branch
    assert cand.improved_tasks == ["t1"]
    assert cand.metadata["commit"] == "abc123"


def test_read_best_returns_none_without_db(tmp_path) -> None:
    fake_tree = SimpleNamespace(db_path_for=lambda root, camp: tmp_path / "missing.db")
    cfg = SimpleNamespace(reports_root=tmp_path, campaign="camp", subset_label="subset")
    assert _launch.read_best(fake_tree, None, cfg,
                             evolvee_source=AgentSource(repo="r")) is None


# --- DarwinX.evolve — the whole launch, hermetic -----------------------------

def _connect_seams(monkeypatch):
    """Monkeypatch the vendored pipeline to a no-op that captures its config."""
    _launch.prepare_import_path()
    import self_evolve.pipeline as pipe

    captured: dict = {}

    class _NoopPipeline:
        def __init__(self, cfg) -> None:  # noqa: ANN001
            captured["cfg"] = cfg

        def run(self) -> int:
            captured["ran"] = True
            return 0

    monkeypatch.setattr(pipe, "SelfEvolvePipeline", _NoopPipeline)
    return captured


def test_evolve_launch_smoke_wires_all_seams(monkeypatch, tmp_path) -> None:
    captured = _connect_seams(monkeypatch)
    repo_root = tmp_path / "repo"
    (repo_root / "monet_code" / ".git").mkdir(parents=True)   # canonical evolvee clone present
    for k in ("MONET_EVAL_REPO_ROOT", "MONET_EVAL_WORKTREE_PARENT", "ATELIER_CROSS_BENCH_GATE",
              *_RUNTIME_ENV_KEYS):
        monkeypatch.delenv(k, raising=False)
    algo = dx.DarwinX(repo_root=str(repo_root), reports_root=str(tmp_path / "reports"),
                      campaign="smoke", max_loop_iters=1, cross_bench_gate=True)
    evolvee, evolver = _FakeEvolvee(), _FakeEvolver()

    best = algo.evolve(evaluate=lambda c: None, evolvee=evolvee, evolver=evolver,
                       config=_run_config())

    # seam B — the evolver Editor is injected in-process
    assert meta_agent.current_editor() is evolver
    meta_agent.set_editor(None)
    # worktree paths pinned to our run via the driver's own overrides (set before import)
    import os
    assert os.environ["MONET_EVAL_REPO_ROOT"] == str(repo_root.resolve())
    # runtime/seed translated from config (evolvee ref "baseline" is branch-like → also a fetch ref)
    assert os.environ["SELF_EVOLVE_EVAL_RUNTIME"] == "local"
    assert os.environ["SELF_EVOLVE_ROOT_COMMIT"] == "baseline"
    assert os.environ["SELF_EVOLVE_ROOT_FETCH_REF"] == "baseline"
    # this algorithm's own gate knob (typed DarwinXConfig) → its ATELIER_* env
    assert os.environ["ATELIER_CROSS_BENCH_GATE"] == "1"
    # seam C — the campaign config was emitted under reports_root
    cfg = captured["cfg"]
    assert captured["ran"] is True
    assert cfg.campaign == "smoke"
    assert list(cfg.subset_tasks) == ["t1", "t2"]     # from benchmark.task_ids
    assert cfg.cursor_model == "auto"
    assert cfg.config_path.exists()
    doc = yaml.safe_load(cfg.config_path.read_text())
    assert doc["monet"]["model"] == "target-model" and doc[EVOLVER_SPEC_KEY]["name"] == "cursor"
    # no genealogy DB was written (noop pipeline) → best is None, gracefully
    assert best is None


def test_evolve_requires_benchmark(tmp_path) -> None:
    algo = dx.DarwinX(repo_root=str(tmp_path), reports_root=str(tmp_path))
    with pytest.raises(ValueError, match="needs a benchmark"):
        algo.evolve(evaluate=lambda c: None, evolvee=_FakeEvolvee(), evolver=_FakeEvolver(),
                    config=None)


def test_evolve_requires_launch_paths() -> None:
    algo = dx.DarwinX(campaign="x")   # no repo_root
    with pytest.raises(ValueError, match="repo_root.*unset|launch paths"):
        algo.evolve(evaluate=lambda c: None, evolvee=_FakeEvolvee(), evolver=_FakeEvolver(),
                    config=_run_config())


def test_launch_paths_defaults_reports_root_to_repo_root() -> None:
    # only repo_root is required; reports_root (DB + campaign config) defaults to it.
    algo = dx.DarwinX(repo_root="/tmp/rr", campaign="c")
    assert algo._launch_paths() == ("/tmp/rr", "/tmp/rr", "c")

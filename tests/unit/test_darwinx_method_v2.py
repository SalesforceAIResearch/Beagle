"""The mechanisms that make grow-then-consolidate actually fire.

Every test here pins a failure that has already happened on a live campaign, so
each one is a regression test rather than a specification:

* the depth lottery never fired, because depth needs a line to extend and the
  tree grew wide -- so compaction is now SCHEDULED (`consolidate_force_k`);
* a successful compaction has net gain 0 and lost parent selection to every
  no-op, so the line it created was abandoned -- hence the shrink bonus;
* the complexity measurement used a Python regex, so on a TypeScript evolvee it
  counted zero branches and a line-neutral rewrite could never show a win;
* an extension whose cue only matches the task it was written for was cheap to
  accept and was never removed.
"""
from __future__ import annotations

import types

import pytest

from beagle.algorithms.darwinx._launch import prepare_import_path
from beagle.algorithms.darwinx.config import DarwinXConfig
from beagle.algorithms.darwinx.vendor.evolve import harness_complexity as hc
from beagle.algorithms.darwinx.vendor.evolve import parent_selection as ps
from beagle.algorithms.darwinx.vendor.evolve import skill_fire as sf

# `pipeline` reaches gate_hook, which imports `evolve.worktree` absolutely -- the
# vendored package expects its own directory on sys.path as top-level `evolve`,
# which is what the launcher arranges in production. Use the same helper rather
# than a bespoke path hack, so this test exercises the real import shape.
prepare_import_path()
from evolve import pipeline as pl  # noqa: E402


# ── TypeScript complexity: the evolvee is a Bun/TS monorepo ────────────────

def test_typescript_branches_are_counted_at_all():
    """With the Python regex this returned 0 and every consolidation looked flat."""
    ts = "export function step(x: string) {\n  if (x === 'a') return 1;\n" \
         "  else if (x === 'b') return 2;\n  return x !== 'z' && x ? 3 : 4;\n}\n"
    lines = hc._code_lines(ts, js=True)
    assert sum(hc._count_branches(l, js=True) for l in lines) == 4


def test_a_commented_out_branch_is_not_a_branch():
    ts = "/* if (fake) return 1; */\nconst a = 1;  // if (also fake)\n"
    lines = hc._code_lines(ts, js=True)
    assert sum(hc._count_branches(l, js=True) for l in lines) == 0


def test_optional_chaining_is_not_a_ternary():
    """`?.` and `??` would otherwise inflate every modern TS file."""
    ts = "const y = obj?.field ?? fallback;\n"
    assert sum(hc._count_branches(l, js=True) for l in hc._code_lines(ts, js=True)) == 0


def test_python_counting_is_unchanged():
    py = "def f(x):\n    # if commented\n    if x and x > 1:\n        return 1\n"
    lines = hc._code_lines(py, js=False)
    assert len(lines) == 3
    assert sum(hc._count_branches(l, js=False) for l in lines) == 2


def test_typescript_extensions_are_recognised():
    for ext in (".ts", ".tsx", ".js", ".mjs"):
        assert ext in hc._JS_EXTS


# ── parent selection: keep a compaction's line alive ──────────────────────

def _node(nid, *, status="completed", improved=0, regressed=0, parent=None):
    return types.SimpleNamespace(
        id=nid, parent_id=parent, status=status, score=0.5,
        improved_tasks_json=f"[{','.join(chr(34) + f't{i}' + chr(34) for i in range(improved))}]",
        regressed_tasks_json=f"[{','.join(chr(34) + f'r{i}' + chr(34) for i in range(regressed))}]",
    )


def test_a_capability_held_compaction_earns_a_shrink_point():
    """improved=0 and regressed=0 is the CONSOLIDATE success shape, not a no-op."""
    assert ps._shrink_bonus(_node("a")) == 1


def test_a_node_that_changed_tasks_gets_no_shrink_point():
    assert ps._shrink_bonus(_node("a", improved=1)) == 0
    assert ps._shrink_bonus(_node("a", regressed=1)) == 0


def test_an_unfinished_node_gets_no_shrink_point():
    assert ps._shrink_bonus(_node("a", status="no_change")) == 0
    assert ps._shrink_bonus(_node("a", status="in_progress")) == 0


def test_deepest_parent_lottery_is_off_by_default(monkeypatch):
    monkeypatch.delenv("DARWINX_GATE_PARENT_DEEPEST_P", raising=False)
    assert ps._deepest_p() == 0.0
    assert ps._coin_deepest("any-pipeline", 0.0) is False


def test_deepest_parent_lottery_is_deterministic_per_pipeline(monkeypatch):
    monkeypatch.setenv("DARWINX_GATE_PARENT_DEEPEST_P", "0.5")
    assert ps._deepest_p() == 0.5
    first = ps._coin_deepest("pipe-abc", 0.5)
    assert all(ps._coin_deepest("pipe-abc", 0.5) is first for _ in range(5))
    assert ps._coin_deepest("pipe-abc", 1.0) is True


def test_lineage_depth_counts_hops_to_root():
    by_id = {"root": _node("root"), "a": _node("a", parent="root"),
             "b": _node("b", parent="a")}
    assert ps._lineage_depth(by_id["root"], by_id) == 0
    assert ps._lineage_depth(by_id["b"], by_id) == 2


# ── scheduled compaction ──────────────────────────────────────────────────

def test_force_k_is_off_by_default(monkeypatch):
    monkeypatch.delenv("DARWINX_GATE_CONSOLIDATE_FORCE_K", raising=False)
    assert pl._consolidate_force_k() == 0
    monkeypatch.delenv("DARWINX_GATE_CONSOLIDATE_ON_BLOAT", raising=False)
    assert pl._consolidate_on_bloat() is False


def test_force_k_reads_its_knob(monkeypatch):
    monkeypatch.setenv("DARWINX_GATE_CONSOLIDATE_FORCE_K", "3")
    assert pl._consolidate_force_k() == 3
    monkeypatch.setenv("DARWINX_GATE_CONSOLIDATE_FORCE_K", "nonsense")
    assert pl._consolidate_force_k() == 0


def _pipe(*, force=False, prune=False, depth=1):
    p = types.SimpleNamespace()
    p.log = types.SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None)
    p.pipeline_id = "pipe-1"
    p._consolidate_node = None
    p._prune_node = None
    p._lineage_depth = lambda: depth
    p._must_consolidate = lambda: force
    p._record_variant_kind = lambda kind: None
    p._is_prune_node = lambda: prune
    return p


def test_a_scheduled_compaction_fires_even_when_the_lottery_cannot(monkeypatch):
    """The whole point: depth 1 with the rate at 0 still compacts."""
    monkeypatch.setenv("DARWINX_GATE_CONSOLIDATE_ENABLED", "1")
    monkeypatch.setenv("DARWINX_GATE_CONSOLIDATE_RATE", "0")
    monkeypatch.setenv("DARWINX_GATE_CONSOLIDATE_MIN_LINEAGE", "9")
    assert pl.SelfEvolvePipeline._is_consolidate_node(_pipe(force=True)) is True


def test_a_scheduled_compaction_outranks_a_prune_draw(monkeypatch):
    monkeypatch.setenv("DARWINX_GATE_CONSOLIDATE_ENABLED", "1")
    assert pl.SelfEvolvePipeline._is_consolidate_node(
        _pipe(force=True, prune=True)) is True


def test_prune_yields_the_node_it_would_have_consumed(monkeypatch):
    monkeypatch.setenv("DARWINX_GATE_PRUNE_ENABLED", "1")
    monkeypatch.setenv("DARWINX_GATE_PRUNE_RATE", "1.0")
    monkeypatch.setenv("DARWINX_GATE_PRUNE_MIN_LINEAGE", "0")
    p = _pipe(force=True)
    p._is_prune_node = types.MethodType(pl.SelfEvolvePipeline._is_prune_node, p)
    assert p._is_prune_node() is False


def test_consolidate_disabled_means_no_scheduled_compaction(monkeypatch):
    monkeypatch.delenv("DARWINX_GATE_CONSOLIDATE_ENABLED", raising=False)
    p = _pipe()
    p._must_consolidate = types.MethodType(pl.SelfEvolvePipeline._must_consolidate, p)
    assert p._must_consolidate() is False


# ── proposer transport faults must not eat the iteration budget ───────────
# Iterations 5-12 of the opencode l15 campaign each burned an iteration on
# `Error: fetch failed` -- its proposer's private proxy had been killed -- and
# the run reported "12 iterations, 0 candidates", which reads as a search that
# found nothing rather than a loop shouting into a dead socket.

class _R:
    def __init__(self, error="", text=""):
        self.error, self.text = error, text


@pytest.mark.parametrize("err", [
    "Error: fetch failed",                      # the exact string from the dead campaign
    "connect ECONNREFUSED 127.0.0.1:18092",     # the exact port that died
    "502 Bad Gateway",
    "UpstreamUnreachable",
    "Temporary failure in name resolution",
])
def test_transport_faults_are_recognised(err):
    assert pl._is_proposer_transport_failure(_R(error=err)) is True


def test_a_timeout_is_not_a_transport_fault():
    """A stage that ran for its whole cap may have been doing real work on a
    starved host; aborting for that would discard a slow-but-working run."""
    assert pl._is_proposer_transport_failure(
        _R(error="monet meta-agent timed out after 2400s", text="partial plan")) is False


def test_a_model_that_answered_badly_is_not_a_transport_fault():
    assert pl._is_proposer_transport_failure(_R(error="", text="a weak plan")) is False


def test_abort_threshold_defaults_to_two(monkeypatch):
    """One blip is what the gateway's own retry window absorbs; two in a row is dead."""
    monkeypatch.delenv("DARWINX_GATE_TRANSPORT_ABORT_AFTER", raising=False)
    assert pl._transport_abort_after() == 2
    monkeypatch.setenv("DARWINX_GATE_TRANSPORT_ABORT_AFTER", "0")
    assert pl._transport_abort_after() == 0  # 0 restores the old spend-everything behaviour


# ── extension usefulness gate ─────────────────────────────────────────────

_SKILL_MD_DIFF = """\
--- /dev/null
+++ b/.opencode-evolve/skills/discriminating-checks/SKILL.md
@@ -0,0 +1,4 @@
+name: discriminating-checks
+description: Require a failing-then-passing check before claiming a risky fix
"""

_MONET_ARRAY_DIFF = """\
--- a/src/core/bundled-skills.js
+++ b/src/core/bundled-skills.js
@@ -1,0 +1,3 @@
+    name: 'discriminating-checks',
+    description: 'Require a failing-then-passing check before claiming a risky fix',
"""


def test_the_gate_is_off_by_default(monkeypatch):
    monkeypatch.delenv("DARWINX_GATE_SKILL_FIRE_GATE", raising=False)
    assert sf.enabled() is False


@pytest.mark.parametrize("diff", [_SKILL_MD_DIFF, _MONET_ARRAY_DIFF])
def test_cues_are_extracted_from_both_evolvees_skill_shapes(diff):
    """opencode writes SKILL.md frontmatter; monet writes a JS array."""
    cues = sf.cues_from_diff(diff)
    assert any("failing-then-passing" in c for c in cues), cues


def test_boilerplate_is_not_a_cue():
    assert sf.cues_from_diff(
        "+++ b/x/SKILL.md\n+description: Use this skill when you want the user to win\n") == []


def test_a_cue_that_only_matches_its_own_claim_is_unproven(tmp_path, monkeypatch):
    monkeypatch.setenv("DARWINX_GATE_SKILL_FIRE_MIN", "2")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "claimed-task.messages.jsonl").write_text(
        '{"content":"Require a failing-then-passing check before claiming a risky fix"}\n')
    (raw / "other-task.messages.jsonl").write_text('{"content":"prints hello world"}\n')
    ok, cues, hits = sf.gate(_SKILL_MD_DIFF, claimed=["claimed-task"], panel_dir=tmp_path)
    assert cues and ok is False
    assert "claimed-task" not in hits


def test_a_cue_that_recurs_off_the_claim_pool_is_proven(tmp_path, monkeypatch):
    monkeypatch.setenv("DARWINX_GATE_SKILL_FIRE_MIN", "2")
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("t1", "t2", "t3"):
        (raw / f"{name}.messages.jsonl").write_text(
            '{"content":"Require a failing-then-passing check before claiming a risky fix"}\n')
    ok, _cues, hits = sf.gate(_SKILL_MD_DIFF, claimed=["t1"], panel_dir=tmp_path)
    assert ok is True
    assert set(hits) == {"t2", "t3"}


def test_no_panel_is_unproven_not_a_free_pass(tmp_path):
    """"We could not check" must not be the cheapest way through a gate."""
    ok, _c, _h = sf.gate(_SKILL_MD_DIFF, claimed=[], panel_dir=None)
    assert ok is False


# ── campaign size: nodes, not iterations ──────────────────────────────────
# One pipeline run produces ONE node. Running it once gives a single lineage of
# length one, in which parent selection has nothing to choose between, depth
# never passes 1 (so PRUNE at depth>=3 and CONSOLIDATE at depth>=2 are
# unreachable), a scheduled compaction can never fire for want of a second
# accepted node, and recombination is impossible. The hosted path did exactly
# that while the campaign YAML said prune_enabled and consolidate_enabled.

def test_default_is_one_node_so_existing_runs_are_unchanged(monkeypatch):
    from beagle.algorithms.darwinx import _launch
    monkeypatch.delenv("DARWINX_EVOLVE_TOTAL_STEPS", raising=False)
    assert _launch._total_steps(object()) == 1
    monkeypatch.delenv("DARWINX_EVOLVE_MERGE_EVERY", raising=False)
    assert _launch._merge_every(object()) == 0


def test_total_steps_arrives_through_the_env(monkeypatch):
    """The vendored PipelineConfig has no field for it, so the env is the route."""
    from beagle.algorithms.darwinx import _launch
    monkeypatch.setenv("DARWINX_EVOLVE_TOTAL_STEPS", "24")
    monkeypatch.setenv("DARWINX_EVOLVE_MERGE_EVERY", "6")
    assert _launch._total_steps(object()) == 24
    assert _launch._merge_every(object()) == 6


def test_the_supervisor_runs_one_pipeline_per_node(monkeypatch):
    from beagle.algorithms.darwinx import _launch
    monkeypatch.setenv("DARWINX_EVOLVE_TOTAL_STEPS", "5")
    monkeypatch.delenv("DARWINX_EVOLVE_MERGE_EVERY", raising=False)
    runs = []

    class _P:
        def __init__(self, cfg): pass
        def run(self): runs.append(1); return 0

    _launch._evolve_steps(_P, object(), None, log=lambda *a: None)
    assert len(runs) == 5


def test_a_node_already_produced_is_charged_against_the_budget(monkeypatch):
    """A resumed campaign must not re-spend the whole budget, nor stop at one."""
    from beagle.algorithms.darwinx import _launch
    monkeypatch.setenv("DARWINX_EVOLVE_TOTAL_STEPS", "5")
    runs = []

    class _P:
        def __init__(self, cfg): pass
        def run(self): runs.append(1); return 0

    _launch._evolve_steps(_P, object(), None, log=lambda *a: None, already_done=1)
    assert len(runs) == 4


def test_total_steps_counts_nodes_and_is_validated():
    with pytest.raises(ValueError, match="NODES"):
        DarwinXConfig(total_steps=0)
    DarwinXConfig(total_steps=24)


def test_a_merge_cadence_that_could_never_fire_is_refused():
    with pytest.raises(ValueError, match="can ever run"):
        DarwinXConfig(total_steps=4, merge_every=6)
    DarwinXConfig(total_steps=24, merge_every=6)


def test_merge_pair_unpacking_accepts_the_selector_s_real_shape():
    """`eligible_pairs` returns list[tuple[Node, Node]]. _try_merge swallows
    exceptions, so a shape mismatch would look like "no pair eligible" forever
    rather than like a bug."""
    from beagle.algorithms.darwinx import _launch

    class _N:
        def __init__(self, i): self.id = i

    # the real shape: a bare 2-tuple of nodes
    assert _launch._pair_ids((_N("aaa"), _N("bbb"))) == ("aaa", "bbb")

    # the dataclass shape that also exists in that module
    class _Pair:
        primary, secondary = _N("ccc"), _N("ddd")
    assert _launch._pair_ids(_Pair()) == ("ccc", "ddd")


def test_campaign_size_reaches_the_driver_env():
    env = DarwinXConfig(total_steps=24, merge_every=6).to_driver_env()
    assert env["DARWINX_EVOLVE_TOTAL_STEPS"] == "24"
    assert env["DARWINX_EVOLVE_MERGE_EVERY"] == "6"


# ── the config surface refuses knobs that could never fire ────────────────

# CONSOLIDATE additionally requires complexity globs, or its accept rule would be
# judged against an empty measurement. Every consolidate-enabled config here
# therefore carries the evolvee's real source glob.
_TS_GLOBS = {"complexity_code_globs": "packages/opencode/src/**/*.ts"}


def test_a_scheduled_compaction_requires_consolidate_enabled():
    with pytest.raises(ValueError, match="consolidate_enabled"):
        DarwinXConfig(consolidate_force_k=2)
    DarwinXConfig(consolidate_force_k=2, consolidate_enabled=True, **_TS_GLOBS)


def test_bloat_trigger_requires_consolidate_enabled():
    with pytest.raises(ValueError, match="consolidate_enabled"):
        DarwinXConfig(consolidate_on_bloat=True)


def test_skill_fire_min_without_the_gate_is_refused():
    with pytest.raises(ValueError, match="skill_fire_gate"):
        DarwinXConfig(skill_fire_min=3)
    DarwinXConfig(skill_fire_min=3, skill_fire_gate=True)


def test_deepest_p_must_be_a_probability():
    with pytest.raises(ValueError, match="probability"):
        DarwinXConfig(parent_deepest_p=1.5)
    DarwinXConfig(parent_deepest_p=0.25)


def test_the_new_knobs_reach_the_driver_env():
    env = DarwinXConfig(
        consolidate_enabled=True, consolidate_force_k=2, consolidate_on_bloat=True,
        parent_deepest_p=0.25, skill_fire_gate=True, skill_fire_min=2, **_TS_GLOBS,
    ).to_driver_env()
    assert env["DARWINX_GATE_CONSOLIDATE_FORCE_K"] == "2"
    assert env["DARWINX_GATE_PARENT_DEEPEST_P"] == "0.25"
    assert env["DARWINX_GATE_SKILL_FIRE_GATE"] == "1"
    assert env["DARWINX_GATE_SKILL_FIRE_MIN"] == "2"
    assert env["DARWINX_GATE_CONSOLIDATE_ON_BLOAT"] == "1"


def test_an_untouched_config_sets_none_of_them():
    """The control arm must stay byte-identical."""
    env = DarwinXConfig().to_driver_env()
    for key in ("DARWINX_GATE_CONSOLIDATE_FORCE_K", "DARWINX_GATE_CONSOLIDATE_ON_BLOAT",
                "DARWINX_GATE_PARENT_DEEPEST_P", "DARWINX_GATE_SKILL_FIRE_GATE",
                "DARWINX_GATE_SKILL_FIRE_MIN"):
        assert key not in env


# --- the failure-theme signal, which read as enabled and delivered nothing -------------------
# Arm A1 launched with `failure_theme: true` and no BASELINE_LOGS, so `_failure_theme_digest`
# returned "" for every iteration: the flag was on, the proposer was told no theme. Two separate
# causes, one test each, because fixing either alone still yields an empty digest.


def _theme_run_json(tmp_path, rows):
    import json
    d = tmp_path / "themes"
    d.mkdir()
    (d / "run.json").write_text(json.dumps({"per_task_results": rows}))
    return d


def test_failure_theme_without_its_input_is_refused():
    """Cause 1: the hosted launcher never set BASELINE_LOGS."""
    with pytest.raises(ValueError, match="baseline_logs"):
        DarwinXConfig(failure_theme=True)


def test_failure_theme_rejects_an_aggregates_only_beagle_run(tmp_path):
    """Cause 2: a beagle run.json has no per_task_results, so the classifier tallies nothing.

    This is the shape that actually shipped -- pointing BASELINE_LOGS at the A0 run dir would
    have loaded fine and still produced an empty digest.
    """
    import json
    d = tmp_path / "beagle_run"
    d.mkdir()
    (d / "run.json").write_text(json.dumps(
        {"benchmarks": {"terminal_bench_2_1": {"score": 0.72, "num_resolved": 321}}}))
    with pytest.raises(ValueError, match="per_task_results"):
        DarwinXConfig(failure_theme=True, baseline_logs=str(d))


def test_failure_theme_accepts_a_converted_run_and_exports_it(tmp_path):
    d = _theme_run_json(tmp_path, [{"task_id": "t__s0", "reward": 0.0, "num_turns": 3}])
    env = DarwinXConfig(failure_theme=True, baseline_logs=str(d)).to_driver_env()
    assert env["BASELINE_LOGS"] == str(d)
    assert env["DARWINX_GATE_FAILURE_THEME"] == "1"


def test_the_digest_is_non_empty_on_a_converted_run(monkeypatch, tmp_path):
    """The end-to-end assertion: real per-trial rows in, targetable theme text out."""
    rows = []
    for task in ("alpha", "beta", "gamma"):
        for s in range(3):
            rows.append({"task_id": f"{task}__s{s}", "reward": 0.0, "error": None,
                         "num_turns": 40, "trajectory_path": None})
    d = _theme_run_json(tmp_path, rows)
    monkeypatch.setenv("BASELINE_LOGS", str(d))
    monkeypatch.setenv("DARWINX_GATE_FAILURE_THEME", "1")
    monkeypatch.setattr(pl, "_FAILURE_THEME_DIGEST_CACHE", None, raising=False)
    digest = pl._failure_theme_digest()
    assert "FAILURE THEME" in digest.upper()
    assert "wrong-output" in digest


def test_the_digest_stays_empty_when_the_flag_is_off(monkeypatch, tmp_path):
    d = _theme_run_json(tmp_path, [{"task_id": "t__s0", "reward": 0.0, "num_turns": 3}])
    monkeypatch.setenv("BASELINE_LOGS", str(d))
    monkeypatch.delenv("DARWINX_GATE_FAILURE_THEME", raising=False)
    monkeypatch.setattr(pl, "_FAILURE_THEME_DIGEST_CACHE", None, raising=False)
    assert pl._failure_theme_digest() == ""


def test_the_converter_groups_trials_into_tasks():
    """The classifier groups by stripping `__s<N>`; beagle names trials `<task>__<hash>`.

    Emitting the raw trial name made every trial its own task, so no task ever reached the >=2
    repeats the digest needs to NAME a task -- the tally looked right while the task list was
    empty.
    """
    import re
    ids = [f"{t}__s{i}" for t in ("alpha", "beta") for i in range(5)]
    grouped = {re.sub(r"__s\d+$", "", i) for i in ids}
    assert grouped == {"alpha", "beta"}


# --- the timeout sub-classification, which collapsed to one useless bucket -------------------
# `classify_trial` splits a timeout by WHERE the budget went, and that split is entirely
# downstream of reading the tool stream. The reader only knew monet's event shape, so on the
# opencode baseline it saw no tools and every one of the 35 timeout trials became
# `timeout-other` -- a theme the digest did not even render. So ~1/3 of all failures produced no
# steer whatsoever.


def _opencode_stream(tmp_path, commands, tool="bash"):
    import json
    p = tmp_path / "opencode.stream.jsonl"
    with open(p, "w") as fh:
        for cmd in commands:
            fh.write(json.dumps({
                "type": "tool_use",
                "part": {"type": "tool", "tool": tool,
                         "state": {"input": {"command": cmd}}},
            }) + "\n")
    return str(p)


def test_the_opencode_stream_shape_is_read_at_all(tmp_path):
    from dx_trace import failure_mode as fm
    tools, text = fm._read_trajectory_activity(
        _opencode_stream(tmp_path, ["apt-get install -y libfoo", "pip install bar"]))
    assert tools["bash"] == 2
    assert "apt-get install" in text


def test_the_monet_stream_shape_still_works(tmp_path):
    """The other arm's runs must classify identically after this change."""
    import json
    from dx_trace import failure_mode as fm
    p = tmp_path / "monet.stream.jsonl"
    p.write_text("\n".join(json.dumps(
        {"type": "tool_start", "name": "bash", "input": c})
        for c in ["apt-get update", "make -j4"]))
    tools, text = fm._read_trajectory_activity(str(p))
    assert tools["bash"] == 2
    assert "apt-get update" in text


def test_an_install_timeout_is_setup_not_other(tmp_path):
    from dx_trace import failure_mode as fm
    c = fm.classify_trial(
        reward=0.0, error="AgentTimeoutError",
        trajectory_path=_opencode_stream(tmp_path, ["apt-get install -y gcc"]), num_turns=30)
    assert c.failure_mode is fm.FailureMode.TIMEOUT_SETUP


def test_a_navigation_timeout_is_exploration(tmp_path):
    """The most actionable theme for a harness: the budget went on FINDING the code."""
    from dx_trace import failure_mode as fm
    c = fm.classify_trial(
        reward=0.0, error="AgentTimeoutError",
        trajectory_path=_opencode_stream(tmp_path, [""] * 10, tool="read"), num_turns=30)
    assert c.failure_mode is fm.FailureMode.TIMEOUT_EXPLORATION


def test_an_unreadable_stream_still_falls_back_safely(tmp_path):
    from dx_trace import failure_mode as fm
    c = fm.classify_trial(reward=0.0, error="AgentTimeoutError",
                          trajectory_path=str(tmp_path / "nope.jsonl"), num_turns=1)
    assert c.failure_mode is fm.FailureMode.TIMEOUT_OTHER


def test_every_classified_theme_is_rendered_by_the_digest(monkeypatch, tmp_path):
    """The digest used to render only 3 of the 8 modes, so classified failures were dropped.

    Pins that each failure theme the classifier can emit reaches the proposer.
    """
    import json
    rows = []
    # two tasks per theme, 2 trials each, so each clears the >=2-repeat bar
    specs = {
        "wrong": (0.0, None),                       # completed, failed grader
        # the real shape from the A0 baseline, which _TOOLERR_RE matches on "exited"
        "toolerr": (0.0, "opencode exited rc=137: bash: line 20: 3808 Done"),
    }
    for task, (reward, err) in specs.items():
        for s in range(2):
            rows.append({"task_id": f"{task}__s{s}", "reward": reward, "error": err,
                         "num_turns": 5, "trajectory_path": None})
    d = tmp_path / "themes"
    d.mkdir()
    (d / "run.json").write_text(json.dumps({"per_task_results": rows}))
    monkeypatch.setenv("BASELINE_LOGS", str(d))
    monkeypatch.setenv("DARWINX_GATE_FAILURE_THEME", "1")
    monkeypatch.setattr(pl, "_FAILURE_THEME_DIGEST_CACHE", None, raising=False)
    digest = pl._failure_theme_digest()
    assert "wrong-output" in digest
    assert "tool-error" in digest, "tool-error was classified but never rendered"


# --- the guard that rejected every candidate for four straight nodes ------------------------
# A1's first four nodes each produced a well-tested, purely additive commit and each was
# reverted as "broad shared-core change". Two independent causes, both pinned here.

from beagle.algorithms.darwinx.vendor.evolve import generalization as gen  # noqa: E402

_CORE = "packages/opencode/src/session/"


def _diff(*files: tuple[str, int, bool]) -> str:
    """Build a git-shaped diff. Each file is (path, added_lines, is_new)."""
    out = []
    for path, added, is_new in files:
        out.append(f"diff --git a/{path} b/{path}")
        if is_new:
            out.append("new file mode 100644")
            out.append("index 0000000..1111111")
            out.append("--- /dev/null")
        else:
            out.append("index 1111111..2222222 100644")
            out.append(f"--- a/{path}")
        out.append(f"+++ b/{path}")
        out.append(f"@@ -0,0 +1,{added} @@")
        out.extend(f"+line {i}" for i in range(added))
    return "\n".join(out)


def _core_env(monkeypatch, **over):
    monkeypatch.setenv("DARWINX_GATE_SHARED_CORE_PATHS", _CORE)
    for k, v in over.items():
        monkeypatch.setenv(k, str(v))


def test_a_new_file_is_not_charged_to_the_file_before_it(monkeypatch):
    """The parse bug: git writes `--- /dev/null` for a creation, which never matched
    `--- a/`, so `cur` kept the previous file and the new file's lines landed on it.

    Measured on candidate d99cfe39: prompt.ts was charged 228 (its real 34 plus the
    194 lines of the new test file after it) and tripped a budget of 40 it was inside.
    """
    _core_env(monkeypatch, DARWINX_GATE_SHARED_CORE_CHURN_BUDGET=40,
              DARWINX_GATE_NEW_CORE_FILE_CHURN_BUDGET=250)
    d = _diff((f"{_CORE}prompt.ts", 34, False),
              ("packages/opencode/test/session/big.test.ts", 194, True))
    vs = gen.scan_diff_locality(d)
    assert vs == [], f"prompt.ts (34 lines, under budget) was charged the next file's: {vs}"


def test_a_focused_new_capability_file_is_allowed(monkeypatch):
    """The shape all four rejected candidates had: new file + small hook + tests."""
    _core_env(monkeypatch, DARWINX_GATE_SHARED_CORE_CHURN_BUDGET=40,
              DARWINX_GATE_NEW_CORE_FILE_CHURN_BUDGET=250)
    d = _diff((f"{_CORE}completion-review.ts", 67, True),
              (f"{_CORE}prompt.ts", 34, False),
              ("packages/opencode/test/session/completion-review.test.ts", 194, True))
    assert gen.scan_diff_locality(d) == []


def test_a_sprawling_new_core_file_is_still_bounced(monkeypatch):
    """The larger budget is a budget, not an exemption -- a new file still runs on
    every task, so breadth still has to be bounded."""
    _core_env(monkeypatch, DARWINX_GATE_SHARED_CORE_CHURN_BUDGET=40,
              DARWINX_GATE_NEW_CORE_FILE_CHURN_BUDGET=250)
    d = _diff((f"{_CORE}completion-verification.ts", 347, True))
    vs = gen.scan_diff_locality(d)
    assert len(vs) == 1 and vs[0].kind == "broad_shared_core_change"


def test_rewriting_an_existing_core_file_keeps_the_tight_budget(monkeypatch):
    """The rule's actual purpose: a broad rewrite of logic every task depends on."""
    _core_env(monkeypatch, DARWINX_GATE_SHARED_CORE_CHURN_BUDGET=40,
              DARWINX_GATE_NEW_CORE_FILE_CHURN_BUDGET=250)
    vs = gen.scan_diff_locality(_diff((f"{_CORE}prompt.ts", 200, False)))
    assert len(vs) == 1 and vs[0].kind == "broad_shared_core_change"


def test_a_file_outside_the_core_surface_is_untouched_by_either_budget(monkeypatch):
    _core_env(monkeypatch, DARWINX_GATE_SHARED_CORE_CHURN_BUDGET=40,
              DARWINX_GATE_NEW_CORE_FILE_CHURN_BUDGET=250)
    assert gen.scan_diff_locality(_diff(("docs/notes.md", 900, True))) == []


# PATH-prepend helpers and ICLR arm YAML live on #28's campaign branch, not this DarwinX-only
# port. Dataset-override tests below cover the eval.py change that did come across.

# --- the dataset override, dropped on the way to the eval -----------------------------------
# `translate_config` discarded `benchmark.dataset` unconditionally. That is right for the
# vendored driver's own relative reference ("benchmarks/terminal_bench/vendor"), which would
# mis-glob, and fatal for an absolute path naming a real task pool: the eval silently fell
# back to $XRLENV_BENCHMARK_CACHE, so every synthesized task id raised
#   KeyError: task id 'canonical-debt-settler' not in benchmark 'terminal_bench_2_1'
# Arm A5 ran for hours looking healthy and scored nothing because of this.


def _translated(bench: dict):
    from beagle.algorithms.darwinx.eval import translate_config
    raw = {
        "model": {"name": "m"},
        "agent": {"name": "opencode"},
        "benchmark": dict(bench),
        "runtime": {"kind": "local"},
    }
    return translate_config(raw).benchmark_spec()


def test_an_absolute_pool_path_survives_translation(tmp_path):
    pool = tmp_path / "pool"
    (pool / "t1").mkdir(parents=True)
    (pool / "t1" / "task.toml").write_text('[environment]\ndocker_image = "img:1"\n')
    spec = _translated({"name": "terminal_bench_2_1", "dataset": str(pool)})
    assert spec.dataset == str(pool), "a deliberate task-source override must reach the loader"


def test_the_drivers_relative_reference_is_still_dropped():
    """The original behaviour, which must not regress: that path would mis-glob."""
    spec = _translated({"name": "terminal_bench_2_1",
                        "dataset": "benchmarks/terminal_bench/vendor"})
    assert spec.dataset is None


def test_an_absolute_path_that_does_not_exist_is_dropped():
    """Falling back to the cache beats globbing a directory that is not there."""
    spec = _translated({"name": "terminal_bench_2_1", "dataset": "/nope/definitely/missing"})
    assert spec.dataset is None


def test_no_dataset_is_unchanged():
    spec = _translated({"name": "terminal_bench_2_1"})
    assert spec.dataset is None

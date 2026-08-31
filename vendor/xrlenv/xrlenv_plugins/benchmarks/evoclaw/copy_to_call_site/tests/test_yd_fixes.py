"""Unit tests for yd_fixes — the opt-in ``--apply-yd-fixes`` monkey-patches.

The end-to-end proof (e662c19 flips to RESOLVED) lives in the eval; these lock in
the wiring: apply is idempotent, replaces the evaluator method, and the patched
checkout stages the test dir BEFORE ``git clean`` (so untracked GT tests survive).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The patch targets the real harness evaluator; skip cleanly if it isn't importable.
pytest.importorskip("harness.e2e.evaluator")
import yd_fixes


@pytest.fixture(autouse=True)
def _isolate_yd_globals():
    """apply_yd_fixes patches module globals (PatchEvaluator._checkout_to_tag and
    go_report_utils.parse_go_test_jsonl) with no built-in restore, so snapshot and
    restore them around every test to keep the suite order-independent."""
    import harness.e2e.container_setup as cs
    import harness.utils.go_report_utils as gu
    from harness.e2e.evaluator import PatchEvaluator

    saved = (PatchEvaluator._checkout_to_tag, gu.parse_go_test_jsonl,
             cs._poison_domain_list, PatchEvaluator.evaluate,
             PatchEvaluator.load_baseline_classification, yd_fixes._APPLIED)
    try:
        yield
    finally:
        (PatchEvaluator._checkout_to_tag, gu.parse_go_test_jsonl,
         cs._poison_domain_list, PatchEvaluator.evaluate,
         PatchEvaluator.load_baseline_classification, yd_fixes._APPLIED) = saved


@pytest.fixture
def restore_evaluator():
    """Save/restore PatchEvaluator._checkout_to_tag + reset the apply guard."""
    from harness.e2e.evaluator import PatchEvaluator

    orig = PatchEvaluator._checkout_to_tag
    yd_fixes._APPLIED = False
    try:
        yield PatchEvaluator
    finally:
        PatchEvaluator._checkout_to_tag = orig
        yd_fixes._APPLIED = False


def test_apply_replaces_checkout_method(restore_evaluator):
    PatchEvaluator = restore_evaluator
    before = PatchEvaluator._checkout_to_tag
    yd_fixes.apply_yd_fixes()
    assert PatchEvaluator._checkout_to_tag is not before


def test_apply_is_idempotent(restore_evaluator):
    PatchEvaluator = restore_evaluator
    yd_fixes.apply_yd_fixes()
    once = PatchEvaluator._checkout_to_tag
    yd_fixes.apply_yd_fixes()  # second call must NOT re-wrap
    assert PatchEvaluator._checkout_to_tag is once


def test_patched_clean_stages_test_dir_before_clean(restore_evaluator, monkeypatch):
    PatchEvaluator = restore_evaluator

    # Stub upstream's checkout (clean=False path) so it succeeds without docker.
    monkeypatch.setattr(
        PatchEvaluator, "_checkout_to_tag",
        lambda self, suffix, clean=True: (True, ""), raising=True,
    )
    yd_fixes._APPLIED = False
    yd_fixes.apply_yd_fixes()  # wraps the stub above

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["script"] = cmd[-1]  # docker exec ... bash -c <script>

        class _R:
            returncode, stdout, stderr = 0, "", ""

        return _R()

    monkeypatch.setattr(yd_fixes.subprocess, "run", fake_run)

    class _FakeEval:
        container_name = "c-eval"
        test_dir = "test/"

    ok, err = PatchEvaluator._checkout_to_tag(_FakeEval(), "end", clean=True)
    assert ok and err == ""
    script = captured["script"]
    assert "git add -A -- test/" in script
    assert "git clean -fd" in script
    # the whole point: stage the test dir BEFORE the clean, else it's deleted.
    assert script.index("git add") < script.index("git clean")


def test_non_clean_checkout_is_passthrough(restore_evaluator, monkeypatch):
    """clean=False must delegate straight to upstream, no extra clean issued."""
    PatchEvaluator = restore_evaluator
    calls = []
    monkeypatch.setattr(
        PatchEvaluator, "_checkout_to_tag",
        lambda self, suffix, clean=True: (calls.append(("orig", clean)) or (True, "")),
        raising=True,
    )
    yd_fixes._APPLIED = False
    yd_fixes.apply_yd_fixes()
    monkeypatch.setattr(
        yd_fixes.subprocess, "run",
        lambda *a, **k: pytest.fail("clean=False must not run a docker clean"),
    )

    class _FakeEval:
        container_name = "c-eval"
        test_dir = "test/"

    ok, _ = PatchEvaluator._checkout_to_tag(_FakeEval(), "end", clean=False)
    assert ok
    assert calls == [("orig", False)]


# --- go benchmark line-split reassembly ---
def test_go_benchmark_reassembly_recovers_split_line(tmp_path):
    """A benchmark whose `Benchmark-N ... ns/op` line is split across two Output
    events is dropped by the base parser but recovered by the reassembly patch.

    Calls the patch directly (not apply_yd_fixes) and restores the module global,
    so it's isolated from the other fixes' shared state.
    """
    import json

    import harness.utils.go_report_utils as gu
    orig = gu.parse_go_test_jsonl
    jl = tmp_path / "out.jsonl"
    # Real go split: the Benchmark line is emitted as package-level output (no Test
    # field) across two events, so the base parser (which only checks benchmark output
    # under a Test context) never sees it.
    jl.write_text("\n".join(json.dumps(e) for e in [
        {"Action": "run", "Package": "ex/p", "Test": "TestA"},
        {"Action": "pass", "Package": "ex/p", "Test": "TestA", "Elapsed": 0.1},
        {"Action": "output", "Package": "ex/p", "Output": "BenchmarkSplit-16      \t"},
        {"Action": "output", "Package": "ex/p", "Output": "       1\t     100 ns/op\n"},
        {"Action": "pass", "Package": "ex/p", "Elapsed": 0.1},
    ]))
    try:
        base = {r.test_name for r in orig(jl).test_results}
        assert "BenchmarkSplit" not in base  # base parser drops the split line
        yd_fixes._patch_go_benchmark_reassembly()  # patches gu.parse_go_test_jsonl
        assert gu.parse_go_test_jsonl is not orig
        fixed = {r.test_name for r in gu.parse_go_test_jsonl(jl).test_results}
        assert "BenchmarkSplit" in fixed  # reassembly recovers it
    finally:
        gu.parse_go_test_jsonl = orig


# --- quarantine: denied domains added to /etc/hosts poison (deterministic block) ---
def test_quarantine_poisons_deny_domains(monkeypatch):
    import harness.e2e.container_setup as cs
    monkeypatch.setenv("EVOCLAW_DENY_DOMAINS", "files.pythonhosted.org,pypi.org")
    assert "files.pythonhosted.org" not in cs._poison_domain_list(True)  # base: only IP-blocked
    yd_fixes._patch_quarantine_poison_deny_domains()
    poisoned = cs._poison_domain_list(True)
    assert "files.pythonhosted.org" in poisoned and "pypi.org" in poisoned  # now DNS-blocked
    assert "files.pythonhosted.org" not in cs._poison_domain_list(False)   # only under quarantine


# --- none_to_pass eval-retry (navidrome persistence-race flake) ---
class _FakeResult:
    """Minimal stand-in for EvaluationResult with the fields the retry inspects."""
    def __init__(self, *, resolved, n2p_fail=(), n2p_ok=(), f2p_fail=(),
                 p2p_fail=(), p2p_missing=0, f2p_req=3, f2p_ach=3):
        self.resolved = resolved
        self.none_to_pass_failure = list(n2p_fail)
        self.none_to_pass_success = list(n2p_ok)
        self.fail_to_pass_failure = list(f2p_fail)
        self.pass_to_pass_failure = list(p2p_fail)
        self.pass_to_pass_missing = p2p_missing
        self.fail_to_pass_required = f2p_req
        self.fail_to_pass_achieved = f2p_ach


def _install_fake_evaluate(monkeypatch, results):
    """Stub PatchEvaluator.evaluate to pop successive results, then apply the retry patch
    on top. Returns a callable that runs one evaluate() on a dummy self, counting calls."""
    from harness.e2e.evaluator import PatchEvaluator
    seq = list(results)
    calls = {"n": 0}

    def fake_evaluate(self):
        calls["n"] += 1
        return seq.pop(0)

    monkeypatch.setattr(PatchEvaluator, "evaluate", fake_evaluate, raising=True)
    yd_fixes._patch_n2p_eval_retry()  # wraps the stub above

    class _Dummy:
        milestone_id = "milestone_003_sub-01"

    return PatchEvaluator, _Dummy(), calls


def test_n2p_retry_recovers_flake(monkeypatch):
    """First eval is an n2p-only flake, a retry resolves -> return the resolved retry."""
    flake = _FakeResult(resolved=False, n2p_fail=["ArtistRepo:A"], n2p_ok=["B"])
    good = _FakeResult(resolved=True, n2p_ok=["A", "B"])
    PatchEvaluator, dummy, calls = _install_fake_evaluate(monkeypatch, [flake, good])
    out = PatchEvaluator.evaluate(dummy)
    assert out is good and out.resolved
    assert calls["n"] == 2  # one retry was needed


def test_n2p_retry_gives_up_after_attempts(monkeypatch):
    """Flake never clears -> keep the ORIGINAL result, bounded by _N2P_RETRY_ATTEMPTS."""
    monkeypatch.setattr(yd_fixes, "_N2P_RETRY_ATTEMPTS", 2)
    r0 = _FakeResult(resolved=False, n2p_fail=["A"])
    r1 = _FakeResult(resolved=False, n2p_fail=["A"])
    r2 = _FakeResult(resolved=False, n2p_fail=["A"])
    PatchEvaluator, dummy, calls = _install_fake_evaluate(monkeypatch, [r0, r1, r2])
    out = PatchEvaluator.evaluate(dummy)
    assert out is r0                # original kept, not a later attempt
    assert calls["n"] == 3          # 1 initial + 2 retries


def test_n2p_retry_skips_when_resolved(monkeypatch):
    """A resolved first eval must NOT trigger any retry."""
    good = _FakeResult(resolved=True, n2p_ok=["A"])
    PatchEvaluator, dummy, calls = _install_fake_evaluate(monkeypatch, [good])
    out = PatchEvaluator.evaluate(dummy)
    assert out is good and calls["n"] == 1


def test_n2p_retry_skips_real_regression(monkeypatch):
    """An f2p failure (real regression) is not a flake -> no retry, no masking."""
    bad = _FakeResult(resolved=False, f2p_fail=["real"], n2p_fail=["A"])
    PatchEvaluator, dummy, calls = _install_fake_evaluate(monkeypatch, [bad])
    out = PatchEvaluator.evaluate(dummy)
    assert out is bad and calls["n"] == 1  # ran once, kept the failure


def test_n2p_retry_does_not_abort_on_shifting_failure(monkeypatch):
    """A mid-loop retry whose failure shape shifts (e.g. an f2p failure appears) must NOT
    abort the remaining attempts — navidrome's contention varies run-to-run, so keep going
    and still recover on a later attempt."""
    monkeypatch.setattr(yd_fixes, "_N2P_RETRY_ATTEMPTS", 3)
    flake = _FakeResult(resolved=False, n2p_fail=["A"])          # original: clean n2p flake -> enter
    noisy = _FakeResult(resolved=False, f2p_fail=["x"], n2p_fail=["A", "B"])  # retry 1: not clean
    good = _FakeResult(resolved=True, n2p_ok=["A"])              # retry 2: recovers
    PatchEvaluator, dummy, calls = _install_fake_evaluate(monkeypatch, [flake, noisy, good])
    out = PatchEvaluator.evaluate(dummy)
    assert out is good and out.resolved   # recovered despite the noisy middle retry
    assert calls["n"] == 3                # did NOT break after the noisy attempt


# --- spawn/forkserver safety: a FRESH interpreter re-applies via the startup hook ---
_HOOK_ONBOARD = Path(__file__).resolve().parents[1]
_HOOK_BOOT = _HOOK_ONBOARD / "_yd_bootstrap"


def _fresh_interpreter_applied(env):
    """Run a fresh venv interpreter (like a spawn pool child) and report whether the
    eval-level patch got installed via the sitecustomize startup hook."""
    import subprocess
    script = "import yd_fixes; print('APPLIED' if yd_fixes.is_applied_here() else 'NOPE')"
    r = subprocess.run([sys.executable, "-c", script], env=env,
                       capture_output=True, text=True)
    return r.stdout, r.stderr


def test_startup_hook_applies_in_fresh_interpreter():
    """With EVOCLAW_APPLY_YD_FIXES=1 and _yd_bootstrap on PYTHONPATH, a brand-new interpreter
    (no fork inheritance) installs the patch at startup -- this is what a spawn/forkserver eval
    child does."""
    import os
    env = dict(os.environ)
    env["EVOCLAW_APPLY_YD_FIXES"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_HOOK_BOOT), str(_HOOK_ONBOARD), env.get("PYTHONPATH", "")])
    out, err = _fresh_interpreter_applied(env)
    assert "APPLIED" in out, f"stdout={out!r} stderr={err[-800:]!r}"


def test_startup_hook_inert_without_env_flag():
    """Same PYTHONPATH but no EVOCLAW_APPLY_YD_FIXES -> the hook is a no-op (a stray
    interpreter must not get silently patched)."""
    import os
    env = dict(os.environ)
    env.pop("EVOCLAW_APPLY_YD_FIXES", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_HOOK_BOOT), str(_HOOK_ONBOARD), env.get("PYTHONPATH", "")])
    out, err = _fresh_interpreter_applied(env)
    assert "NOPE" in out, f"stdout={out!r} stderr={err[-800:]!r}"


# --- drop known-flaky rate-limiter timing tests from graded pass_to_pass ---
_TTL = "github.com/zeromicro/go-zero/core/limit/TestTokenLimit_Take"


def test_drop_ids_removes_string_and_dict_entries():
    sec = {"pass_to_pass": [_TTL, {"test_id": "x/other"}, {"test_id": _TTL}, "x/keep"]}
    removed = yd_fixes._drop_ids_from_pass_to_pass(sec, {_TTL})
    assert removed == [_TTL, _TTL]                       # both the str and the dict form
    ids = [e if isinstance(e, str) else e["test_id"] for e in sec["pass_to_pass"]]
    assert ids == ["x/other", "x/keep"]                  # non-flaky kept, order preserved


def test_patch_drops_flaky_from_both_classifications(restore_evaluator):
    from harness.e2e.evaluator import PatchEvaluator
    doc = {
        "classification": {"pass_to_pass": [_TTL, "x/keep"], "none_to_pass": ["n1"]},
        "stable_classification": {"pass_to_pass": [_TTL, "x/keep"]},
    }
    PatchEvaluator.load_baseline_classification = lambda self: doc  # type: ignore[assignment]
    yd_fixes._patch_drop_flaky_timing_tests()

    class _Dummy:
        pass
    out = PatchEvaluator.load_baseline_classification(_Dummy())
    assert _TTL not in out["classification"]["pass_to_pass"]         # dropped from full
    assert _TTL not in out["stable_classification"]["pass_to_pass"]  # AND from stable (graded)
    assert "x/keep" in out["stable_classification"]["pass_to_pass"]  # non-flaky untouched
    assert out["classification"]["none_to_pass"] == ["n1"]           # other buckets untouched


def test_flaky_list_is_env_extensible(monkeypatch):
    monkeypatch.setenv("EVOCLAW_FLAKY_TIMING_TESTS", "pkg/TestExtra, pkg/TestTwo")
    s = yd_fixes._flaky_timing_tests()
    assert _TTL in s and "pkg/TestExtra" in s and "pkg/TestTwo" in s


_BREAKER = "github.com/zeromicro/go-zero/core/breaker/TestGoogleBreakerOpen"
_ROTATE = "github.com/zeromicro/go-zero/core/logx/TestRotateLogger_WithExistingFile"


def test_conc64_bystander_flakes_dropped_by_default():
    # The two go-zero bystander timing flakes that survived the core/limit drop-list and
    # failed M005 / M027 at conc-64 (resolved when pinned) must now be dropped by DEFAULT —
    # the faithful, contention-independent analog of pinning.
    s = yd_fixes._flaky_timing_tests()
    assert _BREAKER in s and _ROTATE in s
    # each drops from a graded p2p set (dict + str forms), non-flaky kept
    for flaky in (_BREAKER, _ROTATE):
        sec = {"pass_to_pass": [flaky, {"test_id": flaky}, "x/keep"]}
        yd_fixes._drop_ids_from_pass_to_pass(sec, s)
        ids = [e if isinstance(e, str) else e["test_id"] for e in sec["pass_to_pass"]]
        assert ids == ["x/keep"], flaky

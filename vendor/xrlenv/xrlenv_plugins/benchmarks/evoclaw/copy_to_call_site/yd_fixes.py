"""Opt-in local corrections for known UPSTREAM EvoClaw eval-protocol bugs.

Enabled ONLY via ``--apply-yd-fixes`` on ``run_all_xrlenv.py`` / ``run_e2e_xrlenv.py``
(default OFF -> faithful to upstream and leaderboard-comparable). Each correction is
applied as a runtime monkey-patch so the vendored harness files stay byte-for-byte
pristine: a run *without* the flag behaves exactly like the unpatched harness.

Fixes currently applied
-----------------------
* **Preserve untracked GT test files** (element-web ``e662c19`` / ``fba5938``).
  Some published milestone images ship required ``pass_to_pass`` test files that are
  present-on-disk but UNTRACKED at the milestone tags (an image-build defect). The
  evaluator's ``_checkout_to_tag`` runs ``git clean -fd`` to strip pollution before
  applying the agent's ``src`` -- but those GT test files are the benchmark's own
  evaluation, not pollution, so the clean deletes them and their tests can never be
  collected (reported as ``pass_to_pass missing``). This stages the milestone's test
  dir before the clean so the files survive, while genuine (non-test) pollution is
  still removed. Design principle: the agent container must never see the eval tests
  (upstream enforces that via ``SrcFileFilter`` on the agent's *output* snapshot), but
  the EVALUATION container must have them.

* **Recover split Go benchmark results** (go-zero M001/M003/M004/M005/M027). ``go test
  -json`` sometimes splits a benchmark's ``Benchmark<name>-N … ns/op`` line across two
  Output events at a buffer boundary, so ``go_report_utils``'s per-event regex drops it and
  it is reported as ``pass_to_pass missing``. Buffer-boundary FLAKY (varies run to run) and
  CPU-independent -- it is NOT resource contention (proven: cpuset pinning does not fix it).
  This wraps ``parse_go_test_output`` to reassemble per-package lines and recover the missed
  benchmarks. See ``_patch_go_benchmark_reassembly``.

* **Deterministic quarantine deny** (scikit / any quarantine repo). The anti-cheat quarantine
  denies hosts like ``files.pythonhosted.org`` but blocks them only at the IP layer
  (``EVOCLAW_DENY_CIDRS``), which DRIFTS for CDN-fronted hosts (PyPI rides Fastly) — the
  fail-closed ``verify_network_lockdown`` then randomly aborts the milestone with no verdict
  (observed scikit M04). This adds ``EVOCLAW_DENY_DOMAINS`` to the ``/etc/hosts`` → ``0.0.0.0``
  poison the harness already applies to code-hosting domains, so denied hosts are unreachable
  DETERMINISTICALLY (DNS-level, immune to IP drift). See ``_patch_quarantine_poison_deny_domains``.

* **Retry a none_to_pass-only flake** (navidrome sub-01 / sub-03). navidrome's ``persistence``
  Ginkgo specs race on shared DB state under CPU contention; cpuset pinning cuts it a lot but a
  residual ~1-in-N flake remains, showing up as a small ``none_to_pass`` failure set while every
  ``fail_to_pass`` / ``pass_to_pass`` bucket is clean. The upstream evaluator runs the tests
  exactly once (``evaluator.py``: *"Run tests once (no retry logic in E2E evaluator)"*). This
  wraps ``PatchEvaluator.evaluate`` to re-run the WHOLE eval (fresh container) up to K times when
  — and ONLY when — the sole reason a milestone is unresolved is a *small* ``none_to_pass``
  failure set, returning the first attempt that resolves. It never retries an f2p/p2p regression
  or a large n2p failure (a real bug, e.g. dubbo's image defect), so it de-flakes without masking
  genuine failures. See ``_patch_n2p_eval_retry``.

* **Drop known-flaky rate-limiter timing tests** (go-zero M014 & any whole-suite go-zero
  milestone). go-zero's ``core/limit`` rate-limiter tests (``TestTokenLimit_Take`` & siblings)
  assert an allowed-count against a wall-clock window → they flake under CPU/scheduler jitter, and
  are *bystander* ``pass_to_pass`` tests in every go-zero milestone (`go test ./...`). Using
  EvoClaw's OWN ``stable_classification`` mechanism, this removes them from the graded ``pass_to_pass``
  set so the evaluator no longer requires them. See ``_patch_drop_flaky_timing_tests``.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

_APPLIED = False

# How many EXTRA times to re-run a none_to_pass-only flake, and the largest n2p
# failure set we treat as a flake (above this it's a real regression, not a race).
_N2P_RETRY_ATTEMPTS = int(os.environ.get("EVOCLAW_N2P_RETRY_ATTEMPTS", "3"))
_N2P_RETRY_MAX_FAILS = int(os.environ.get("EVOCLAW_N2P_RETRY_MAX_FAILS", "12"))

# Known-flaky go-zero *timing/concurrency* tests. They are bystander pass_to_pass tests in
# every whole-suite go-zero milestone (`go test ./...`), so whichever milestone draws the
# flake in a run fails. Each asserts a timed/concurrent outcome (a wall-clock window, a
# scheduler-dependent race, or async-writer/fs timing) → flakes under CPU/scheduler jitter,
# NOT a real regression. Confirmed genuinely-flaky by CPU-pinning: with dedicated cores each
# passes cleanly (M005 22/0, M027 17/0), so they are contention/timing artifacts, not the
# milestone's own change. EvoClaw's own dataset construction didn't flag them (flaky_tests
# empty), so we drop them from the graded set via EvoClaw's stable_classification mechanism.
# Dropped from pass_to_pass ONLY (f2p untouched) — a milestone that genuinely TESTS one of
# these (fail_to_pass) is unaffected, so this never masks a real fix. Env-extensible
# (EVOCLAW_FLAKY_TIMING_TESTS). The faithful analog of pinning (Table A) — pinning treats the
# contention, this scoped exclusion removes the flaky bystander from grading regardless of
# concurrency (so the result is contention-independent).
_DEFAULT_FLAKY_TIMING_TESTS = (
    "github.com/zeromicro/go-zero/core/limit/TestTokenLimit_Take",       # observed flaky (M014)
    "github.com/zeromicro/go-zero/core/limit/TestTokenLimit_TakeBurst",  # same timing class
    "github.com/zeromicro/go-zero/core/limit/TestPeriodLimit_Take",
    "github.com/zeromicro/go-zero/core/limit/TestPeriodLimit_TakeWithAlign",
    # 2026-07-08 conc-64 full-98: the two bystander timing flakes that survived the core/limit
    # drop-list and failed M005 / M027 (each 1-of-2000+ p2p, resolved when pinned).
    "github.com/zeromicro/go-zero/core/breaker/TestGoogleBreakerOpen",   # SRE breaker time-window (M027)
    "github.com/zeromicro/go-zero/core/logx/TestRotateLogger_WithExistingFile",  # async-writer/fs timing (M005)
)


def _flaky_timing_tests() -> set[str]:
    """The test ids to drop from pass_to_pass grading (default + EVOCLAW_FLAKY_TIMING_TESTS)."""
    extra = [t.strip() for t in os.environ.get("EVOCLAW_FLAKY_TIMING_TESTS", "").split(",") if t.strip()]
    return set(_DEFAULT_FLAKY_TIMING_TESTS) | set(extra)


def apply_yd_fixes() -> None:
    """Install all YD fixes (idempotent). Call once, before the harness eval runs."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    _patch_preserve_gt_tests()
    _patch_go_benchmark_reassembly()
    _patch_quarantine_poison_deny_domains()
    _patch_n2p_eval_retry()
    _patch_drop_flaky_timing_tests()
    print(
        "⚠️  YD fixes ACTIVE (--apply-yd-fixes): local corrections to "
        "upstream eval-protocol bugs applied -- results are NOT leaderboard-comparable "
        "with an unpatched harness. See xrlenv_onboard/yd_fixes.py.",
        flush=True,
    )


# The known upstream eval-protocol bugs these fixes correct, and the milestones that
# spuriously FAIL without them. Shown in the loud launch-time warning when the flag is off.
_REQUIRED_BANNER = [
    "element-web  e662c19 / fba5938   evaluator's `git clean` deletes untracked GT test files",
    "go-zero      M001/M003/M004/M005/M027   `go test -json` splits benchmark lines -> dropped",
    "scikit-learn (any quarantine repo)   CDN-fronted deny host flakes the fail-closed net check",
    "navidrome    sub-01 / sub-03   persistence DB-race none_to_pass flake (n2p-only eval-retry)",
]


def warn_yd_fixes_off() -> None:
    """Loud launch-time warning when ``--apply-yd-fixes`` is NOT set.

    The corrections in this module fix KNOWN upstream eval-protocol bugs; without them
    the milestones below fail spuriously (not agent/solution errors). Call this once at
    launch when the flag is off so an operator can't silently under-count.
    """
    bar = "=" * 78
    lines = "\n".join(f"      * {b}" for b in _REQUIRED_BANNER)
    print(
        f"\n{bar}\n"
        "⚠️  --apply-yd-fixes is OFF. Known UPSTREAM eval-protocol bugs are NOT corrected.\n"
        "    Without it these milestones spuriously FAIL (harness bugs, not the agent/solution):\n"
        f"{lines}\n"
        "    Pass --apply-yd-fixes for the intent-correct result set (all are opt-in, harness\n"
        "    files stay pristine). Leave OFF *only* for a byte-faithful, leaderboard-comparable run.\n"
        f"{bar}\n",
        flush=True,
    )


def _patch_preserve_gt_tests() -> None:
    """Keep UNTRACKED GT test files across the evaluator's ``git clean -fd``.

    Wraps ``PatchEvaluator._checkout_to_tag``: it performs upstream's checkout with
    ``clean=False``, then runs a clean that first stages the milestone's test dir
    (``git add -A -- <test_dir>``) so the GT test files become tracked-in-index and
    ``git clean -fd`` leaves them, while still removing genuine non-test pollution.
    Non-clean checkouts (``clean=False``) are passed straight through unchanged.
    """
    from harness.e2e.evaluator import PatchEvaluator

    orig_checkout = PatchEvaluator._checkout_to_tag

    def _checkout_to_tag(self, tag_suffix, clean=True):  # type: ignore[no-untyped-def]
        if not clean:
            return orig_checkout(self, tag_suffix, clean=False)
        # Upstream's checkout, but WITHOUT its git clean...
        ok, err = orig_checkout(self, tag_suffix, clean=False)
        if not ok:
            return ok, err
        # ...then clean while preserving untracked GT test files: stage the test dir
        # first so they are tracked-in-index (git clean skips tracked paths).
        test_dir = (getattr(self, "test_dir", None) or "test/").strip()
        script = (
            f"cd /testbed && "
            f"git add -A -- {test_dir} 2>/dev/null; "
            f"git clean -fd"
        )
        subprocess.run(
            ["docker", "exec", self.container_name, "bash", "-c", script],
            capture_output=True,
            text=True,
        )
        print(
            f"\U0001f527 YD fix: staged {test_dir} before git clean "
            f"(preserve untracked GT tests)"
        )
        return True, ""

    PatchEvaluator._checkout_to_tag = _checkout_to_tag  # type: ignore[method-assign]


def _patch_go_benchmark_reassembly() -> None:
    """Recover Go benchmark results that ``go test -json`` splits across Output events.

    A benchmark's result line ``Benchmark<name>-<N> \\t <iters> \\t <ns> ns/op`` is
    sometimes emitted by ``go test -json`` as **two** Output events split at a buffer
    boundary (e.g. ``'BenchmarkNodeFind-16 \\t'`` then ``'1\\t 20594 ns/op ...'``).
    ``go_report_utils.parse_go_test_output`` applies its benchmark regex to each Output
    event independently, so a split line matches neither fragment and the benchmark is
    dropped — reported as ``pass_to_pass missing``. This is buffer-boundary FLAKY (which
    benchmark splits varies run to run) and CPU-independent (cpuset pinning does not help),
    which is why it looked like contention but is really a parser bug.

    Fix: wrap ``parse_go_test_output`` to reassemble per-package Output events into complete
    lines and add any benchmark the base parser missed. Proven on captured runs to recover
    exactly the milestone's missing benchmarks (e.g. M005: BenchmarkNodeFind + BenchmarkTopkHeap).
    """
    import json
    import re

    import harness.utils.go_report_utils as gu

    pat = re.compile(r"^(Benchmark\S+)-(\d+)\s+(\d+)\s+([\d.]+)\s+ns/op")
    orig_parse = gu.parse_go_test_jsonl  # the GoTestSummary parser (has the benchmark logic)

    def parse_go_test_jsonl(jsonl_path):  # type: ignore[no-untyped-def]
        summary = orig_parse(jsonl_path)
        try:
            if not getattr(jsonl_path, "exists", lambda: False)():
                return summary
            existing = {t.nodeid for t in summary.test_results}
            buf: dict[str, str] = {}
            for line in jsonl_path.read_text(errors="replace").splitlines():
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("Action") != "output":
                    continue
                pkg = d.get("Package", "")
                b = buf.get(pkg, "") + (d.get("Output", "") or "")
                while "\n" in b:
                    ln, b = b.split("\n", 1)
                    m = pat.match(ln.strip())
                    if m:
                        r = gu.GoTestResult(package=pkg, test_name=m.group(1),
                                            action="pass", elapsed=0.0, output_lines=[ln])
                        if r.nodeid not in existing:
                            existing.add(r.nodeid)
                            summary.test_results.append(r)
                buf[pkg] = b
        except Exception:
            return summary  # never break the eval on a parser-fix error
        return summary

    # ``parse_go_test_output`` (the dict wrapper the eval calls) invokes
    # ``parse_go_test_jsonl`` by module-global name, so patching the module attr routes.
    gu.parse_go_test_jsonl = parse_go_test_jsonl


def _patch_quarantine_poison_deny_domains() -> None:
    """Make the quarantine's denied-host block DETERMINISTIC (DNS-level).

    The eval quarantine denies hosts like ``files.pythonhosted.org`` and then
    ``verify_network_lockdown`` fails-closed if it can still reach them. Blocking is
    done at the IP layer (``EVOCLAW_DENY_CIDRS``): resolve the host, drop those CIDRs.
    But PyPI rides Fastly, whose IPs DRIFT — the deny-CIDR only covers what was
    resolved at setup, so a later DNS answer lands on an accepted range and the
    check flakes (aborting the milestone with no verdict; observed on scikit M04).

    ``lock_network`` step 4 already poisons ``/etc/hosts`` → ``0.0.0.0`` for the
    code-hosting / mirror domains (``_poison_domain_list``), a DNS-level block immune
    to IP drift and tamper-proof (``chmod 644`` + sudoers stripped). This extends that
    poison list to include ``EVOCLAW_DENY_DOMAINS`` under quarantine, so denied hosts
    resolve to ``0.0.0.0`` and are unreachable every run. The iptables deny-CIDR stays
    as belt-and-suspenders for direct-IP attempts.
    """
    import os

    import harness.e2e.container_setup as cs

    orig = cs._poison_domain_list

    def _poison_domain_list(quarantine_active):  # type: ignore[no-untyped-def]
        domains = list(orig(quarantine_active))
        if quarantine_active:
            deny = [d.strip() for d in os.environ.get("EVOCLAW_DENY_DOMAINS", "").split(",") if d.strip()]
            for d in deny:
                if d not in domains:
                    domains.append(d)
        return domains

    cs._poison_domain_list = _poison_domain_list


def _is_n2p_only_flake(result) -> bool:  # type: ignore[no-untyped-def]
    """True iff a milestone is unresolved *solely* because a small none_to_pass set failed.

    That is the navidrome contention-race signature: every fail_to_pass is achieved, no
    pass_to_pass regressed or went missing, and only a handful of new tests failed. A large
    n2p failure set (e.g. dubbo's image defect) or any f2p/p2p failure is NOT a flake, so we
    never retry those — de-flake without masking a real regression.
    """
    try:
        return (
            not result.resolved
            and result.fail_to_pass_achieved == result.fail_to_pass_required
            and len(result.fail_to_pass_failure) == 0
            and len(result.pass_to_pass_failure) == 0
            and result.pass_to_pass_missing == 0
            and 0 < len(result.none_to_pass_failure) <= _N2P_RETRY_MAX_FAILS
        )
    except AttributeError:
        return False


def _n2p_audit(self, msg: str) -> None:  # type: ignore[no-untyped-def]
    """Record retry activity where it's actually visible.

    The eval runs inside EvoClaw's ``ProcessPoolExecutor`` child, whose STDOUT is NOT
    captured in the run logs (harness ``print``s like "Comparing results..." never reach
    the worker log either). So a ``print`` here is effectively invisible. We append to a
    file in the results tree (``<workspace_root>/evaluation/<mid>/yd_n2p_retry.log``) so
    every retry decision is auditable; the print stays as best-effort.
    """
    mid = getattr(self, "milestone_id", "?")
    line = f"[yd-n2p-retry] {mid}: {msg}"
    print("\U0001f527 " + line, flush=True)  # best-effort (child stdout is usually swallowed)
    for base in (getattr(self, "workspace_root", None), getattr(self, "output_dir", None)):
        if base is None:
            continue
        try:
            p = Path(base) / "evaluation" / str(mid) / "yd_n2p_retry.log"
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as f:
                f.write(line + "\n")
            return
        except Exception:
            continue


def _patch_n2p_eval_retry() -> None:
    """Re-run the eval when the only failure is a small none_to_pass flake (navidrome).

    Wraps ``PatchEvaluator.evaluate``. The upstream evaluator runs the test suite exactly
    once ("no retry logic in E2E evaluator"), so a contention-race in navidrome's persistence
    Ginkgo specs randomly fails ``none_to_pass`` tests and the milestone flips unresolved. This
    re-runs the WHOLE eval (``evaluate`` is self-contained: fresh container, re-checkout,
    re-apply patch, re-run tests) up to ``_N2P_RETRY_ATTEMPTS`` extra times whenever the ORIGINAL
    attempt's only failure is a small n2p set (see ``_is_n2p_only_flake``). Returns the first
    attempt that resolves; otherwise keeps the original result unchanged (so a retry that happens
    to introduce a *different* regression can never be selected).

    We try ALL ``_N2P_RETRY_ATTEMPTS`` — we do NOT stop early if one retry's failure shape shifts:
    navidrome's contention race varies its failing-spec *count* run-to-run, so an intermediate
    retry that flakes harder must not abort the remaining attempts. Engagement note: the eval runs
    in a forked ``ProcessPoolExecutor`` child (Python fork inherits this monkey-patch); retry
    activity is written to ``yd_n2p_retry.log`` in the results tree (see ``_n2p_audit``) because
    the child's stdout isn't captured. No extra fleet footprint (the retry reuses the reserved
    eval slot).
    """
    from harness.e2e.evaluator import PatchEvaluator

    orig_evaluate = PatchEvaluator.evaluate

    def evaluate(self):  # type: ignore[no-untyped-def]
        result = orig_evaluate(self)
        n2p_fail = len(getattr(result, "none_to_pass_failure", []) or [])
        eligible = _is_n2p_only_flake(result)
        _n2p_audit(self, f"evaluate#0 resolved={result.resolved} n2p_fail={n2p_fail} "
                         f"flake_eligible={eligible}")
        if _N2P_RETRY_ATTEMPTS <= 0 or not eligible:
            return result
        for attempt in range(1, _N2P_RETRY_ATTEMPTS + 1):
            _n2p_audit(self, f"n2p-only flake (n2p_fail={n2p_fail}); "
                             f"eval-retry {attempt}/{_N2P_RETRY_ATTEMPTS}")
            retry = orig_evaluate(self)
            if retry.resolved:
                _n2p_audit(self, f"RESOLVED on eval-retry {attempt}")
                return retry
        _n2p_audit(self, f"still unresolved after {_N2P_RETRY_ATTEMPTS} retries; kept original")
        return result

    evaluate._yd_wrapped = True  # type: ignore[attr-defined]  # sentinel for subprocess checks
    PatchEvaluator.evaluate = evaluate  # type: ignore[method-assign]


def _drop_ids_from_pass_to_pass(section, flaky: set[str]) -> list[str]:
    """Remove ``flaky`` test ids from ``section['pass_to_pass']`` (entries may be plain
    strings or ``{'test_id': ...}`` dicts). Returns the ids actually removed."""
    p2p = section.get("pass_to_pass")
    if not isinstance(p2p, list):
        return []

    def _tid(e):
        return e if isinstance(e, str) else (e.get("test_id") if isinstance(e, dict) else None)

    removed = [t for e in p2p if (t := _tid(e)) in flaky]
    if removed:
        section["pass_to_pass"] = [e for e in p2p if _tid(e) not in flaky]
    return removed


def _patch_drop_flaky_timing_tests() -> None:
    """Drop known-flaky rate-limiter *timing* tests from the graded ``pass_to_pass`` set.

    go-zero's ``core/limit`` rate-limiter tests (``TestTokenLimit_Take`` & siblings) assert an
    allowed-count against a wall-clock window, so they flake under CPU/scheduler jitter. They are
    *bystander* ``pass_to_pass`` tests in every whole-suite go-zero milestone (`go test ./...`), so
    whichever milestone draws the flake in a run flips unresolved (observed: M014,
    ``TestTokenLimit_Take``). EvoClaw's own dataset construction never flagged them (``flaky_tests``
    empty; they sit in ``stable_classification.pass_to_pass``).

    This wraps ``PatchEvaluator.load_baseline_classification`` and, using EvoClaw's *own*
    stable-classification mechanism (the evaluator grades against ``stable_classification`` "excluding
    flaky tests"), removes ``_flaky_timing_tests()`` from ``pass_to_pass`` in both ``classification``
    and ``stable_classification`` so the evaluator no longer requires them. Faithful (uses the designed
    exclusion; the on-disk dataset is untouched), narrowly scoped to genuinely-flaky timing tests
    (unlike a p2p-retry, which would mask real regressions), and default OFF. Report upstream so
    EvoClaw marks these flaky.
    """
    from harness.e2e.evaluator import PatchEvaluator

    orig_load = PatchEvaluator.load_baseline_classification

    def load_baseline_classification(self):  # type: ignore[no-untyped-def]
        d = orig_load(self)
        flaky = _flaky_timing_tests()
        if not flaky or not isinstance(d, dict):
            return d
        removed: list[str] = []
        for key in ("classification", "stable_classification"):
            sec = d.get(key)
            if isinstance(sec, dict):
                removed += _drop_ids_from_pass_to_pass(sec, flaky)
        if removed:
            uniq = sorted(set(removed))
            print(
                f"\U0001f527 YD fix: dropped {len(uniq)} known-flaky timing test(s) from "
                f"pass_to_pass grading: {uniq}",
                flush=True,
            )
        return d

    PatchEvaluator.load_baseline_classification = load_baseline_classification  # type: ignore[method-assign]


def is_applied_here() -> bool:
    """True iff this *process* has the eval-level patch installed (checks the sentinel).

    Used to verify a spawn/forkserver eval child re-applied the fixes (a forked child
    inherits them; a spawned one must re-run ``apply_yd_fixes`` via the startup hook)."""
    try:
        from harness.e2e.evaluator import PatchEvaluator
        return bool(getattr(PatchEvaluator.evaluate, "_yd_wrapped", False))
    except Exception:
        return False

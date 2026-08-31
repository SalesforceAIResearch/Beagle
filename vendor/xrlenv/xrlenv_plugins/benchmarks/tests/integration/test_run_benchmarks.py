"""Unit tests for xrlenv_plugins/benchmarks/tests/integration/run_benchmarks.py — pure logic only (no cluster / network).

Exercises the config merge, the deterministic sampler, per-benchmark command
construction, config validation, the LIST_GREEN task-id filter, the parallel
scheduler + in-place banner, and the artifact-coverage gate (a sweep passes only
if every requested task produced a passing result — not on exit code alone). The
one impure helper (`_list_green`, which shells out to run_full_sweep.sh) is not
covered here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parent / "run_benchmarks.py"  # co-located harness


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_benchmarks", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rb = _load_module()


def _cfg() -> dict[str, Any]:
    return {
        "defaults": {"content_retries": 2, "seed": 0},
        "benchmarks": {"lhtb": {"workers": 8}, "seta": {"workers": 16}},
        "profiles": {
            "full": {"mode": "full"},
            "ci": {"mode": "sample", "k": 5, "overrides": {"lhtb": {"k": 3}}},
            "picked": {"mode": "full", "only": ["seta"]},
        },
    }


# ── _effective (defaults <- benchmark <- profile <- overrides) ────────────────


def test_effective_merge_precedence_override_wins() -> None:
    eff = rb._effective(_cfg(), "ci", "lhtb")
    assert eff["content_retries"] == 2   # defaults
    assert eff["workers"] == 8           # benchmark
    assert eff["mode"] == "sample"       # profile
    assert eff["k"] == 3                 # profile override wins over profile's k=5


def test_effective_no_override_uses_profile() -> None:
    eff = rb._effective(_cfg(), "ci", "seta")
    assert eff["workers"] == 16 and eff["mode"] == "sample" and eff["k"] == 5


def test_effective_full_profile() -> None:
    eff = rb._effective(_cfg(), "full", "lhtb")
    assert eff["mode"] == "full" and eff["workers"] == 8


def test_effective_unknown_profile_raises() -> None:
    with pytest.raises(SystemExit, match="unknown profile"):
        rb._effective(_cfg(), "nope", "lhtb")


def test_effective_drops_only_key() -> None:
    # `only:` is a profile-level selector, not a per-benchmark knob — it must not leak
    # into the merged config the sweep commands read.
    eff = rb._effective(_cfg(), "picked", "seta")
    assert "only" not in eff and eff["mode"] == "full"


# ── _select_names (--benchmark override > profile `only` > all) ────────────────


def test_select_names_default_all() -> None:
    assert rb._select_names(_cfg(), "full", None) == ["lhtb", "seta"]


def test_select_names_profile_only_restricts() -> None:
    assert rb._select_names(_cfg(), "picked", None) == ["seta"]


def test_select_names_only_preserves_config_order() -> None:
    cfg = _cfg()
    cfg["profiles"]["rev"] = {"mode": "full", "only": ["seta", "lhtb"]}  # reverse of config
    assert rb._select_names(cfg, "rev", None) == ["lhtb", "seta"]        # config order wins


def test_select_names_benchmark_arg_overrides_only() -> None:
    # an explicit --benchmark wins over the profile's `only` set
    assert rb._select_names(_cfg(), "picked", "lhtb") == ["lhtb"]


def test_select_names_unknown_benchmark_raises() -> None:
    with pytest.raises(SystemExit, match="unknown benchmark"):
        rb._select_names(_cfg(), "full", "nope")


def test_select_names_unknown_only_raises() -> None:
    cfg = _cfg()
    cfg["profiles"]["bad"] = {"mode": "full", "only": ["nope"]}
    with pytest.raises(SystemExit, match="unknown benchmark"):
        rb._select_names(cfg, "bad", None)


# ── config-validation guards against a false green (audit H4) ─────────────────


def test_select_names_empty_benchmark_arg_raises() -> None:
    # `--benchmark ,` parses to nothing but is truthy — it must FAIL, not run zero plans
    # + exit 0. (`--benchmark ""` is falsy -> treated as "no filter, run all", which is fine.)
    for arg in (",", " , ", ",,"):
        with pytest.raises(SystemExit, match="selected no benchmarks"):
            rb._select_names(_cfg(), "full", arg)


def test_select_names_empty_config_raises() -> None:
    cfg = {"defaults": {}, "benchmarks": {}, "profiles": {"full": {"mode": "full"}}}
    with pytest.raises(SystemExit, match="no benchmarks"):
        rb._select_names(cfg, "full", None)


def test_positive_k_rejects_zero_and_negative() -> None:
    assert rb._positive_k("b", {"k": 3}) == 3
    for bad in (0, -1, -50):
        with pytest.raises(SystemExit, match="needs k >= 1"):
            rb._positive_k("b", {"k": bad})


def test_plan_sample_rejects_nonpositive_k(monkeypatch: pytest.MonkeyPatch,
                                           tmp_path: Path) -> None:
    monkeypatch.setattr(rb, "_list_green", lambda name, **_k: (["t1", "t2", "t3"], 3))
    cfg = _cfg()
    cfg["profiles"]["ci"]["overrides"] = {"lhtb": {"k": 0}}   # k=0 -> --tasks "" footgun
    with pytest.raises(SystemExit, match="needs k >= 1"):
        rb._plan(cfg, "ci", "lhtb", tmp_path, None)


def test_list_green_nonzero_exit_is_not_trusted(monkeypatch: pytest.MonkeyPatch) -> None:
    # a wrapper that fails to compute its green set (rc!=0) must NOT have task-shaped
    # stdout accepted as the requested set — that would let a subset sweep false-green.
    class _R:
        returncode = 1
        stdout = "adaptive-rejection-sampler\ntw_522753\n"   # looks like valid ids
        stderr = "ERROR: cache not built"

    monkeypatch.setattr(rb.subprocess, "run", lambda *a, **k: _R())
    with pytest.raises(SystemExit, match="exited 1 — cannot trust"):
        rb._list_green("lhtb")


def test_list_green_passes_skip_build_cache_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    # audit M12/Low: _list_green must ALWAYS pass --skip-build-cache in the ACTUAL argv (the
    # read-only gate never builds during planning). Assert on the real command _list_green
    # builds — not by mocking _list_green itself.
    seen: dict[str, list[str]] = {}

    class _R:
        returncode = 0
        stdout = "task-a\ntask-b\n"
        stderr = ""

    def _capture(cmd: list[str], *a: object, **k: object) -> _R:
        seen["cmd"] = cmd
        return _R()

    monkeypatch.setattr(rb.subprocess, "run", _capture)
    ids, total = rb._list_green("lhtb")
    assert ids == ["task-a", "task-b"]
    assert total == 2  # stderr had no #TOTAL_PRESENT -> falls back to the green count
    assert seen["cmd"][:2] == ["bash", str(rb.BENCH_DIR / "lhtb" / "run_full_sweep.sh")]
    assert "--list-green" in seen["cmd"] and "--skip-build-cache" in seen["cmd"]


def test_list_green_parses_total_present_from_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    # The wrapper emits ``#TOTAL_PRESENT=<n>`` (full corpus, before EXCLUDE) on stderr; the
    # planning line shows green/total from it. Parse it; ignore other stderr noise.
    class _R:
        returncode = 0
        stdout = "task-a\ntask-b\n"
        stderr = "==> build log line\n#TOTAL_PRESENT=200\nanother line\n"

    monkeypatch.setattr(rb.subprocess, "run", lambda *a, **k: _R())
    ids, total = rb._list_green("terminalworld")
    assert ids == ["task-a", "task-b"]
    assert total == 200


def test_select_names_rejects_duplicate_benchmark() -> None:
    # audit M3: `--benchmark lhtb,lhtb` would give two plans one benchmark log + job-id.
    with pytest.raises(SystemExit, match="duplicate"):
        rb._select_names(_cfg(), "full", "lhtb,lhtb")


def test_plan_is_read_only_for_all_profiles(monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: Path) -> None:
    # audit M12: the runner is a READ-ONLY gate — planning (full OR sample) never builds the
    # cache; _list_green is always called and passes --skip-build-cache internally.
    called: list[str] = []

    def fake(name: str) -> tuple[list[str], int]:
        called.append(name)
        return ["t1", "t2", "t3"], 3

    monkeypatch.setattr(rb, "_list_green", fake)
    rb._plan(_cfg(), "full", "lhtb", tmp_path, None)   # full: read-only, no build
    rb._plan(_cfg(), "ci", "seta", tmp_path, None)     # sample: read-only
    assert called == ["lhtb", "seta"]


def test_resolve_run_dir_rejects_escapes(tmp_path: Path) -> None:
    # audit H5: --overwrite rmtree's run_dir, so run_name MUST be a bare component under
    # jobs_dir — an absolute / ../ / slashed name would escape and delete an arbitrary dir.
    jobs = str(tmp_path / "jobs")
    rd = rb._resolve_run_dir(jobs, "ci-abc")            # a bare name is fine
    assert rd.parent == Path(jobs).resolve() and rd.name == "ci-abc"
    for bad in ("/tmp/victim", "../victim", "a/b", "..", ".", ""):
        with pytest.raises(SystemExit, match=r"single path component|escapes"):
            rb._resolve_run_dir(jobs, bad)


def test_full_cmd_is_read_only_skip_build(tmp_path: Path) -> None:
    # audit M12: the full EXECUTION command must also carry --skip-build-cache — the gate
    # never builds a cache, in planning OR execution.
    cmd, _env = rb._full_cmd("deep_swe", {"workers": 8, "content_retries": 2}, "jid", "/run")
    assert "--skip-build-cache" in cmd


# ── _sample (deterministic) ───────────────────────────────────────────────────


def test_sample_is_deterministic_and_sorted() -> None:
    green = ["e", "a", "d", "b", "c"]
    a = rb._sample(green, 3, seed=0)
    b = rb._sample(green, 3, seed=0)
    assert a == b                 # deterministic
    assert a == sorted(a)         # stable --tasks order
    assert len(a) == 3
    assert set(a) <= set(green)


def test_sample_input_order_irrelevant() -> None:
    assert rb._sample(["c", "a", "b"], 2, 0) == rb._sample(["a", "b", "c"], 2, 0)


def test_sample_k_ge_len_returns_all_sorted() -> None:
    assert rb._sample(["b", "a"], 5, 0) == ["a", "b"]


def test_sample_different_seed_usually_differs() -> None:
    green = [f"t{i}" for i in range(50)]
    assert rb._sample(green, 5, 0) != rb._sample(green, 999, 0)


# ── command construction ──────────────────────────────────────────────────────


def test_full_cmd_uses_flags() -> None:
    # run_full_sweep.sh is a uniform FLAG interface — run knobs never travel via env
    # (a stale exported MAX_WORKERS/SKIP_BUILD must not silently change a sweep).
    cmd, _env = rb._full_cmd("deep_swe", {"workers": 32, "content_retries": 2}, "jid", "/run")
    assert cmd[0] == "bash" and cmd[1].endswith("deep_swe/run_full_sweep.sh")
    assert cmd[cmd.index("--max-workers") + 1] == "32"
    assert cmd[cmd.index("--content-retries") + 1] == "2"
    assert cmd[cmd.index("--job-id") + 1] == "jid"
    assert cmd[cmd.index("--jobs-dir") + 1] == "/run"   # full mode is grouped under the run dir too


def test_full_cmd_flags_lhtb() -> None:
    cmd, _env = rb._full_cmd("lhtb", {"workers": 8, "content_retries": 1}, "jid", "/run")
    assert cmd[cmd.index("--max-workers") + 1] == "8"
    assert cmd[cmd.index("--content-retries") + 1] == "1"
    assert cmd[cmd.index("--job-id") + 1] == "jid"


def test_full_cmd_does_not_inject_env_knobs() -> None:
    # the footgun fix: the runner must NOT add MAX_WORKERS/CONTENT_RETRIES/JOB_ID to
    # the child env (they're flags now). Guard against a regression that re-adds them.
    _cmd, env = rb._full_cmd("deep_swe", {"workers": 8}, "jid-xyz", "/run")
    assert env.get("JOB_ID") != "jid-xyz"        # not injected by the runner
    assert "jid-xyz" not in env.values()


def test_sample_cmd_jobs_dir_and_retries_from_config() -> None:
    # retries now comes from the merged config (benchmarks.yaml), not a Python table.
    cmd = rb._sample_cmd("deep_swe", ["a", "b"], {"workers": 32, "retries": 6}, "/jd", "jid")
    assert cmd[cmd.index("--tasks") + 1] == "a,b"
    assert cmd[cmd.index("--jobs-dir") + 1] == "/jd"
    assert "--retries" in cmd and cmd[cmd.index("--retries") + 1] == "6"


def test_sample_cmd_seta_unified_jobs_dir_null_retries() -> None:
    # seta unified onto --jobs-dir (alias added to its run_oracle_sweep.py); its
    # benchmarks.yaml `retries: null` -> no --retries flag emitted.
    cmd = rb._sample_cmd("seta", ["1", "2"], {"workers": 16, "retries": None}, "/jd", "jid")
    assert cmd[cmd.index("--jobs-dir") + 1] == "/jd"
    assert "--save-artifacts" not in cmd
    assert "--retries" not in cmd


def test_sample_cmd_no_retries_key_omits_flag() -> None:
    # a benchmark whose config omits `retries` entirely also emits no --retries.
    cmd = rb._sample_cmd("lhtb", ["x"], {"workers": 8}, "/jd", "jid")
    assert "--retries" not in cmd


def test_sample_cmd_passes_content_retries() -> None:
    # the ci gap fix: sample mode forwards content_retries so run_oracle_sweep re-runs
    # its own reward=0 flakes (the per-task content-retry now lives in the .py).
    cmd = rb._sample_cmd("terminalworld", ["x"], {"workers": 8, "content_retries": 2}, "/jd", "jid")
    assert "--content-retries" in cmd and cmd[cmd.index("--content-retries") + 1] == "2"


def test_sample_cmd_no_content_retries_key_omits_flag() -> None:
    cmd = rb._sample_cmd("terminalworld", ["x"], {"workers": 8}, "/jd", "jid")
    assert "--content-retries" not in cmd


# ── config validation + LIST_GREEN filter ─────────────────────────────────────


def test_load_config_rejects_missing_sections(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("benchmarks: {}\n", encoding="utf-8")   # no profiles:
    with pytest.raises(SystemExit, match="expected a mapping"):
        rb._load_config(p)


def test_task_id_filter_keeps_ids_drops_progress_lines() -> None:
    assert rb._TASK_ID.match("adaptive-rejection-sampler")
    assert rb._TASK_ID.match("tw_522753")
    assert rb._TASK_ID.match("0")
    # the "==> ..." progress lines run_full_sweep.sh prints have spaces -> dropped
    assert not rb._TASK_ID.match("==> --skip-build-cache — skipping build_cache")
    assert not rb._TASK_ID.match("present tasks: 89")


# ── _plan (n_tasks) + _schedule (parallel task-budget scheduler) ───────────────


def test_plan_sample_n_tasks_is_sampled_count(monkeypatch: pytest.MonkeyPatch,
                                              tmp_path: Path) -> None:
    monkeypatch.setattr(rb, "_list_green", lambda name, **_k: (["t1", "t2", "t3", "t4", "t5"], 5))
    plan = rb._plan(_cfg(), "ci", "seta", tmp_path, None)   # ci profile, k=5
    assert plan["mode"] == "sample"
    assert plan["n_tasks"] == len(plan["tasks"]) == 5       # n_tasks = sampled count


def test_plan_full_n_tasks_is_green_size(monkeypatch: pytest.MonkeyPatch,
                                         tmp_path: Path) -> None:
    monkeypatch.setattr(rb, "_list_green", lambda name, **_k: ([f"t{i}" for i in range(42)], 42))
    plan = rb._plan(_cfg(), "full", "lhtb", tmp_path, None)
    assert plan["mode"] == "full" and plan["n_tasks"] == 42  # n_tasks = green-set size


def _fake_plan(name: str, cmd: list[str], n_tasks: int,
               workers: int | None = None) -> dict[str, Any]:
    p: dict[str, Any] = {"benchmark": name, "mode": "full", "cmd": cmd,
                         "env": os.environ.copy(), "n_tasks": n_tasks, "green": n_tasks}
    if workers is not None:
        p["workers"] = workers
    return p


def test_schedule_runs_all_and_preserves_order(tmp_path: Path) -> None:
    plans = [_fake_plan("a", ["true"], 5),
             _fake_plan("b", ["false"], 4),
             _fake_plan("c", ["true"], 3)]
    results = rb._schedule(plans, budget=10, run_dir=tmp_path, profile="t")
    assert [r["benchmark"] for r in results] == ["a", "b", "c"]   # plan order preserved
    assert [r["passed"] for r in results] == [True, False, True]  # `false` -> rc 1 -> FAIL
    assert (tmp_path / "a-t.log").is_file()                       # per-benchmark log written


def test_schedule_oversized_plan_runs_alone(tmp_path: Path) -> None:
    # a benchmark with more tasks than the whole budget must still run (no deadlock).
    results = rb._schedule([_fake_plan("big", ["true"], 999)],
                           budget=10, run_dir=tmp_path, profile="t")
    assert results[0]["passed"] is True


def test_schedule_budget_charges_real_concurrency_not_task_count(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Two 100-task benchmarks with workers=32: their TASK counts (200) exceed the
    # budget (128), but their REAL concurrency cost (min(100,32)=32 each, 64 total)
    # fits — so BOTH must be admitted at once, not run one-at-a-time. Proven by both
    # "▶ start" lines appearing before either "✓ PASS".
    import io
    buf = io.StringIO()
    monkeypatch.setattr(rb.sys, "stdout", buf)
    plans = [_fake_plan("aaa", ["bash", "-c", "sleep 0.3"], 100, workers=32),
             _fake_plan("bbb", ["bash", "-c", "sleep 0.3"], 100, workers=32)]
    rb._schedule(plans, budget=128, run_dir=tmp_path, profile="t")
    out = buf.getvalue()
    first_pass = out.index("✓ PASS")
    assert out.index("▶ start   aaa") < first_pass
    assert out.index("▶ start   bbb") < first_pass   # admitted concurrently, not after aaa


def test_plan_cost_is_min_of_tasks_and_workers() -> None:
    # A benchmark's budget cost is its real peak concurrency = min(n_tasks, workers).
    assert rb._plan_cost({"n_tasks": 500, "workers": 8}) == 8      # workers < tasks
    assert rb._plan_cost({"n_tasks": 3, "workers": 32}) == 3       # capped by task count
    assert rb._plan_cost({"n_tasks": 50}) == 50                    # no workers -> task count


def test_schedule_tty_redraws_status_block_in_place(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # On a TTY the live-status block redraws in place (clear-to-end-of-screen) rather
    # than scrolling a new block each tick. Start/finish lines stay permanent above it.
    import io

    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    buf = _FakeTTY()
    monkeypatch.setattr(rb.sys, "stdout", buf)
    monkeypatch.setattr(rb, "_BANNER_TICK_S", 0.05)   # tick fast so the block redraws
    # a plan that outlives a tick so the block definitely renders at least once
    results = rb._schedule([_fake_plan("slow", ["bash", "-c", "sleep 0.4"], 3)],
                           budget=10, run_dir=tmp_path, profile="t")
    out = buf.getvalue()
    assert results[0]["passed"] is True
    assert "\x1b[J" in out             # sticky block cleared+redrawn in place (anti-scroll)
    assert "live status" in out and "elapsed" in out   # the status block rendered
    assert "▶ start" in out and "PASS" in out          # permanent event lines still printed


def test_schedule_non_tty_appends_heartbeat_lines(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # When output is captured (not a TTY) there must be NO in-place carriage-return
    # control codes — a pipe/file gets plain appended lines it can scroll/grep.
    import io

    buf = io.StringIO()   # StringIO.isatty() -> False
    monkeypatch.setattr(rb.sys, "stdout", buf)
    results = rb._schedule([_fake_plan("q", ["true"], 1)],
                           budget=10, run_dir=tmp_path, profile="t")
    out = buf.getvalue()
    assert results[0]["passed"] is True
    assert "\r" not in out and "\x1b[" not in out   # no terminal control codes in a log
    assert "Σ total" in out                          # final aggregate line appended


# ── live progress: finished / active / error from disk markers ─────────────────────────
# The banner must report REAL live tasks ("running" = active), not the scheduler's
# reserved concurrency budget, and a content-retry re-run must read as active/remaining —
# NOT as "finished" — even though its first-pass report.json still exists on disk.


def _swe_inst(job_dir: Path, run_id: str, instance: str, *,
              started: bool = True, graded: bool = False) -> None:
    """Fabricate a SWE-bench eval dir under logs/run_evaluation/<run_id>/<model>/<instance>/.
    ``run_instance.log`` is the START marker (written when the eval begins); ``report.json``
    is the GRADE marker (written when it finishes)."""
    d = job_dir / "logs" / "run_evaluation" / run_id / "xrlenv-oracle" / instance
    d.mkdir(parents=True, exist_ok=True)
    if started:
        (d / "run_instance.log").write_text("start\n", encoding="utf-8")
    if graded:
        (d / "report.json").write_text("{}", encoding="utf-8")


def test_progress_swe_first_pass_separates_finished_and_active(tmp_path: Path) -> None:
    # First pass: graded instances (report.json) are finished; started-but-ungraded
    # (run_instance.log only) are active/live. Disjoint — no instance is both.
    job = tmp_path / "swebench_verified-p"
    for i in range(3):
        _swe_inst(job, "xrlenv-oracle-sweep", f"done-{i}", graded=True)
    for i in range(2):
        _swe_inst(job, "xrlenv-oracle-sweep", f"live-{i}", graded=False)
    p = rb._benchmark_progress(tmp_path, "swebench_verified-p")
    assert p["swe"] is True
    assert p["finished"] == 3      # graded
    assert p["active"] == 2        # live: run_instance.log, no report.json
    assert p["error"] == 0         # not disk-measurable for swe


def test_progress_swe_content_retry_shows_rerun_as_active_not_finished(tmp_path: Path) -> None:
    # The exact bug this fixes: during content-retry the first-pass report.json for a
    # re-run instance still exists, so a naive report.json count reads "all finished"
    # while it is re-running. An instance graded in the base round but with a fresh,
    # report-less retry dir must read ACTIVE and drop OUT of finished until the re-grade.
    job = tmp_path / "swebench_verified-p"
    for i in range(4):
        _swe_inst(job, "xrlenv-oracle-sweep", f"inst-{i}", graded=True)   # first pass graded
    _swe_inst(job, "xrlenv-oracle-sweep-retry1", "inst-0", graded=False)  # inst-0 re-running
    p = rb._benchmark_progress(tmp_path, "swebench_verified-p")
    assert p["finished"] == 3      # inst-0 subtracted — it is re-running
    assert p["active"] == 1        # inst-0 is the sole live eval (NOT 32-reserved noise)
    # once the retry re-grades, inst-0 returns to finished and active drops to 0
    _swe_inst(job, "xrlenv-oracle-sweep-retry1", "inst-0", graded=True)
    p2 = rb._benchmark_progress(tmp_path, "swebench_verified-p")
    assert p2["finished"] == 4 and p2["active"] == 0


def test_progress_pier_harbor_active_is_markerless_trial(tmp_path: Path) -> None:
    # Pier/Harbor: result.json = finished; exception.txt with no result = error; a trial
    # dir with NEITHER marker = live/active. finished + error + remaining == total holds.
    job = tmp_path / "deep_swe-p"
    _write_trial(job, "done", {"reward": 1})                              # graded -> finished
    (job / "boom__x").mkdir(parents=True)
    (job / "boom__x" / "exception.txt").write_text("NodeLost", encoding="utf-8")   # error
    (job / "live__x").mkdir(parents=True)                                # no markers -> active
    p = rb._benchmark_progress(tmp_path, "deep_swe-p")
    assert p["swe"] is False
    assert p["finished"] == 1
    assert p["error"] == 1
    assert p["active"] == 1
    assert p["retried"] == 1       # one exception.txt occurrence (churn diagnostic)


def test_live_status_block_reports_active_running_and_reserved_budget(tmp_path: Path) -> None:
    # The headline fix: a swebench retry tail (6 tasks, first pass done, 1 re-running while
    # the scheduler still reserves 32 slots) must render "1 running / 32 reserved", not the
    # old misleading "32 running". finished drops to 5, remaining is the 1 re-run.
    job = tmp_path / "swebench_verified-g"
    for i in range(6):
        _swe_inst(job, "xrlenv-oracle-sweep", f"inst-{i}", graded=True)
    _swe_inst(job, "xrlenv-oracle-sweep-retry1", "inst-0", graded=False)   # re-running
    plan = {"benchmark": "swebench_verified", "n_tasks": 6,
            "job_id": "swebench_verified-g", "workers": 32}
    st = rb._LiveStatus([plan], tmp_path, budget=128, started=0.0, is_tty=True)
    st.mark(0, "running")
    text = "\n".join(st._block(in_flight=32))       # scheduler reserved 32 for this benchmark
    assert "1 running" in text                      # ONE live eval, not the 32 reservation
    assert "32/128 reserved" in text                # reservation labelled, not called "running"
    assert "finished 5" in text                     # inst-0 dropped out of finished
    assert "remaining 1" in text                    # the lone re-run is remaining, not done


# ── artifact-coverage gate (_trial_result_passes / _passing_tasks / _verify_coverage) ──
# A benchmark passes only if its sweep exited 0 AND every requested task produced a
# passing artifact — rc==0 alone is not trusted (catches a partial/failed sweep).


def _write_trial(job_dir: Path, task: str, rewards: dict[str, Any] | None,
                 suffix: str = "abc", dir_name: str | None = None) -> None:
    """Fabricate a <job_dir>/<dir>__<suffix>/result.json like the sweep contract writes.

    ``task`` is the canonical id, written into ``config.task.path`` (the untruncated
    identity harbor/pier record). ``dir_name`` (default = ``task``) is the on-disk dir
    stem — harbor TRUNCATES it for long ids, so pass a truncated stem to exercise that."""
    stem = dir_name if dir_name is not None else task
    td = job_dir / f"{stem}__{suffix}"
    td.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {"config": {"task": {"path": f"/cache/shard/{task}"}}}
    if rewards is not None:
        body["verifier_result"] = {"rewards": rewards}
    (td / "result.json").write_text(json.dumps(body), encoding="utf-8")


def test_passing_tasks_uses_canonical_id_not_truncated_dir(tmp_path: Path) -> None:
    # audit H3: harbor truncates the trial dir stem to ~32 chars, so the dir name is a
    # truncated ALIAS. The requested id lives in config.task.path — coverage must key on
    # THAT, else a long-named passing task reads as missing and forces a green run RED.
    job = tmp_path / "deep_swe-ci"
    _write_trial(job, "tengo-callable-instance-isolation", {"reward": 1},
                 dir_name="tengo-callable-instance-isolatio")  # 32-char cut (no final 'n')
    passing = rb._passing_tasks(tmp_path, "deep_swe-ci", "deep_swe")
    assert passing == {"tengo-callable-instance-isolation"}         # NOT the truncated stem
    assert rb._verify_coverage("deep_swe", ["tengo-callable-instance-isolation"], passing) is None


def test_canonical_task_id_falls_back_to_dir_name_without_path(tmp_path: Path) -> None:
    # a legacy/short artifact with no config.task.path -> dir-name split (legacy behavior)
    td = tmp_path / "short__xyz"
    td.mkdir()
    (td / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 1}}}))
    assert rb._canonical_task_id(td / "result.json", "short__xyz") == "short"


def test_trial_result_passes_keys_on_reward_field(tmp_path: Path) -> None:
    # deep_swe keys strictly on the reward field: reward>0 passes even when a side metric
    # (partial) is legitimately 0.
    _write_trial(tmp_path, "ok", {"reward": 1, "partial": 0})
    assert rb._trial_result_passes(tmp_path / "ok__abc" / "result.json", "deep_swe") is True
    _write_trial(tmp_path, "no", {"reward": 0, "f2p": 1})
    assert rb._trial_result_passes(tmp_path / "no__abc" / "result.json", "deep_swe") is False


def test_trial_result_passes_all_values_rule(tmp_path: Path) -> None:
    # seta/tb2.1/tw (and the default): gate on EVERY value > 0.
    _write_trial(tmp_path, "a", {"score": 0.5, "bonus": 0.2})
    assert rb._trial_result_passes(tmp_path / "a__abc" / "result.json", "seta") is True
    _write_trial(tmp_path, "b", {"score": 0.5, "bonus": 0})
    assert rb._trial_result_passes(tmp_path / "b__abc" / "result.json", "seta") is False


def test_trial_result_passes_rejects_recorded_exception(tmp_path: Path) -> None:
    # audit M18: a trial that ERRORED (non-null exception_info) must NOT read as passing even
    # with a stale positive reward — mirrors every benchmark's _trial_passes first check.
    td = tmp_path / "boom__abc"
    td.mkdir()
    (td / "result.json").write_text(json.dumps({
        "config": {"task": {"path": "/cache/shard/boom"}},
        "exception_info": {"exception_type": "NodeLost"},
        "verifier_result": {"rewards": {"reward": 1}},
    }), encoding="utf-8")
    assert rb._trial_result_passes(td / "result.json", "deep_swe") is False
    assert rb._trial_result_passes(td / "result.json", "lhtb") is False


def test_rewards_pass_is_benchmark_specific(tmp_path: Path) -> None:
    # audit M18: a multi-metric result with NO `reward` key must follow EACH benchmark's
    # own rule. {partial:0, diagnostic_score:1} PASSES lhtb (max>0, a partial-credit band)
    # but FAILS seta/tb2.1/tw (all>0) — the old universal all>0 rule false-red'd lhtb.
    mixed = {"partial": 0, "diagnostic_score": 1}
    assert rb._rewards_pass(mixed, "lhtb") is True
    assert rb._rewards_pass(mixed, "seta") is False
    assert rb._rewards_pass(mixed, "terminalworld") is False
    # deep_swe keys ONLY on the reward field: no key ⇒ fail even if other metrics are +ve.
    assert rb._rewards_pass({"score": 1}, "deep_swe") is False
    assert rb._rewards_pass({"reward": 1}, "deep_swe") is True
    # lhtb with a reward key uses it (not the max) — a 0 reward fails despite a +ve metric.
    assert rb._rewards_pass({"reward": 0, "diagnostic_score": 1}, "lhtb") is False


def test_trial_result_passes_missing_or_malformed_is_false(tmp_path: Path) -> None:
    assert rb._trial_result_passes(tmp_path / "gone" / "result.json", "seta") is False  # no file
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "result.json").write_text("{not json", encoding="utf-8")
    assert rb._trial_result_passes(tmp_path / "bad" / "result.json", "seta") is False
    _write_trial(tmp_path, "empty", {})                                          # no rewards
    assert rb._trial_result_passes(tmp_path / "empty__abc" / "result.json", "seta") is False


def test_passing_tasks_unions_across_retry_dirs(tmp_path: Path) -> None:
    # round 0: t1 passes, t2 fails; retry1: t2 passes -> a task passing in ANY round counts
    _write_trial(tmp_path / "bench-ci", "t1", {"reward": 1})
    _write_trial(tmp_path / "bench-ci", "t2", {"reward": 0})
    _write_trial(tmp_path / "bench-ci-retry1", "t2", {"reward": 1})
    assert rb._passing_tasks(tmp_path, "bench-ci", "deep_swe") == {"t1", "t2"}


def test_passing_tasks_glob_scoped_to_job_id(tmp_path: Path) -> None:
    # another job's dir under the same run root must NOT leak into this job's passing set
    _write_trial(tmp_path / "bench-ci", "t1", {"reward": 1})
    _write_trial(tmp_path / "other-ci", "x9", {"reward": 1})
    assert rb._passing_tasks(tmp_path, "bench-ci", "deep_swe") == {"t1"}


def test_verify_coverage_pass_missing_and_extra() -> None:
    assert rb._verify_coverage("b", ["a", "c"], {"a", "c"}) is None        # all covered
    assert rb._verify_coverage("b", ["a", "c"], {"a", "c", "z"}) is None   # extra TBD ignored
    msg = rb._verify_coverage("b", ["a", "c"], {"a"})                      # c missing
    assert msg is not None and "1/2" in msg and "c" in msg


def test_apply_coverage_flips_rc0_pass_to_fail_on_missing_artifact(tmp_path: Path) -> None:
    # the crux: a sweep that exited 0 but produced no passing artifact for a requested
    # task must be flipped to FAIL — exit code alone is not trusted.
    _write_trial(tmp_path / "bench-ci", "t1", {"reward": 1})   # t1 ok; t2 has NO artifact
    plans = [{"benchmark": "bench", "requested": ["t1", "t2"], "job_id": "bench-ci"}]
    results = [{"benchmark": "bench", "passed": True, "n_tasks": 2}]   # rc==0 said "pass"
    rb._apply_coverage(results, plans, tmp_path)
    assert results[0]["passed"] is False
    assert results[0]["passing"] == 1 and results[0]["requested"] == 2
    assert results[0]["coverage_error"] and "t2" in results[0]["coverage_error"]


def test_apply_coverage_keeps_a_fully_covered_pass(tmp_path: Path) -> None:
    _write_trial(tmp_path / "bench-ci", "t1", {"reward": 1})
    _write_trial(tmp_path / "bench-ci", "t2", {"reward": 0.5})
    plans = [{"benchmark": "bench", "requested": ["t1", "t2"], "job_id": "bench-ci"}]
    results = [{"benchmark": "bench", "passed": True, "n_tasks": 2}]
    rb._apply_coverage(results, plans, tmp_path)
    assert results[0]["passed"] is True and results[0]["coverage_error"] is None


# ── SWE-bench artifact shape (summary.json / resolved) — the H3 regression ─────────────
# SWE-bench writes NO per-task trial dirs, only <job>/summary.json with per-instance
# `resolved`. The coverage gate must understand it, else a resolved SWE run reads 0/N.


def _write_summary(job_dir: Path, resolved: list[str],
                   unresolved: tuple[str, ...] = ()) -> None:
    """Fabricate a SWE-bench <job_dir>/summary.json (per-instance resolved flags)."""
    job_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"instance_id": i, "resolved": True} for i in resolved]
    rows += [{"instance_id": i, "resolved": False} for i in unresolved]
    (job_dir / "summary.json").write_text(
        json.dumps({"expected": len(rows), "resolved": len(resolved),
                    "failed": list(unresolved), "instances": rows}), encoding="utf-8")


def test_summary_json_resolved_reads_swebench_shape(tmp_path: Path) -> None:
    _write_summary(tmp_path / "swe-ci", resolved=["a-1", "b-2"], unresolved=("c-3",))
    assert rb._summary_json_resolved(tmp_path / "swe-ci" / "summary.json") == {"a-1", "b-2"}
    # missing / malformed / non-SWE (Harbor writes no summary.json) -> empty, not a crash
    assert rb._summary_json_resolved(tmp_path / "nope.json") == set()


def test_passing_tasks_understands_swebench_summary(tmp_path: Path) -> None:
    # SWE writes only summary.json (no trial dirs). The Harbor-only gate returned an empty
    # set and forced every resolved SWE run red (H3); coverage must see the resolved ids.
    _write_summary(tmp_path / "swebench_verified-ci",
                   resolved=["django__django-11099", "astropy__astropy-14182"])
    assert rb._passing_tasks(tmp_path, "swebench_verified-ci", "swebench_verified") == {
        "django__django-11099", "astropy__astropy-14182"}


def test_apply_coverage_keeps_a_resolved_swebench_pass(tmp_path: Path) -> None:
    # H3 end-to-end: a fully-resolved SWE run (summary.json only) must STAY pass, not be
    # forced red by the coverage gate.
    reqs = ["django__django-11099", "astropy__astropy-14182"]
    _write_summary(tmp_path / "swebench_verified-ci", resolved=reqs)
    plans = [{"benchmark": "swebench_verified", "requested": reqs,
              "job_id": "swebench_verified-ci"}]
    results = [{"benchmark": "swebench_verified", "passed": True, "n_tasks": 2}]
    rb._apply_coverage(results, plans, tmp_path)
    assert results[0]["passed"] is True and results[0]["coverage_error"] is None
    assert results[0]["passing"] == 2


# ── _benchmark_progress (live per-benchmark finished/error/retried) ───────────


def _mk_trial(job_dir: Path, task_id: str, suffix: str, *, done: bool = True) -> None:
    """A trial dir <job_dir>/<task_id>__<suffix>; ``done`` writes the result.json
    (config.task.path basename = canonical task id) that marks it completed."""
    d = job_dir / f"{task_id}__{suffix}"
    d.mkdir(parents=True)
    if done:
        (d / "result.json").write_text(json.dumps(
            {"config": {"task": {"path": f"/cache/bench/{task_id}"}}}))


def test_completed_task_count_empty_when_no_dirs(tmp_path: Path) -> None:
    assert rb._benchmark_progress(tmp_path, "bench-full")["finished"] == 0


def test_completed_task_count_counts_only_finished_trials(tmp_path: Path) -> None:
    main = tmp_path / "bench-full-TS"
    _mk_trial(main, "taskA", "a1")
    _mk_trial(main, "taskB", "b1")
    _mk_trial(main, "taskC", "c1", done=False)   # in-flight — no result.json yet
    assert rb._benchmark_progress(tmp_path, "bench-full")["finished"] == 2


def test_completed_task_count_dedups_retry_rounds(tmp_path: Path) -> None:
    _mk_trial(tmp_path / "bench-full-TS", "taskB", "b1")           # attempt 0 (failed)
    _mk_trial(tmp_path / "bench-full-TS-retry1", "taskB", "b2")    # retry (passed)
    # same canonical task across main + retry dir → counted once, not twice
    assert rb._benchmark_progress(tmp_path, "bench-full")["finished"] == 1


def test_completed_task_count_fallback_dir_name(tmp_path: Path) -> None:
    d = tmp_path / "bench-full-TS" / "taskZ__zzz"
    d.mkdir(parents=True)
    (d / "result.json").write_text("not json")   # unreadable → fall back to dir name
    assert rb._benchmark_progress(tmp_path, "bench-full")["finished"] == 1


def _mk_swe_instance(job_dir: Path, run_id: str, inst: str, *,
                     completed: bool = True, model: str = "xrlenv-oracle") -> None:
    """A swebench per-instance eval dir. ``run_instance.log`` marks STARTED; a
    ``report.json`` (only when ``completed``) marks COMPLETION."""
    d = job_dir / "logs" / "run_evaluation" / run_id / model / inst
    d.mkdir(parents=True)
    (d / "run_instance.log").write_text("ran")   # always present once started
    if completed:
        (d / "report.json").write_text("{}")


def test_completed_task_count_swe_shape_counts_completed_instances(tmp_path: Path) -> None:
    # swebench writes no per-trial result.json; progress = distinct <instance> dirs
    # with a report.json (completion), deduped across the main / -retryN run_id folders.
    job = tmp_path / "swebench_verified-full-TS"
    _mk_swe_instance(job, "xrlenv-oracle-sweep", "django__django-1")
    _mk_swe_instance(job, "xrlenv-oracle-sweep", "astropy__astropy-2")
    _mk_swe_instance(job, "xrlenv-oracle-sweep-retry1", "django__django-1")  # re-run
    assert rb._benchmark_progress(tmp_path, "swebench_verified-full")["finished"] == 2


def test_completed_task_count_swe_ignores_in_flight_instances(tmp_path: Path) -> None:
    # An instance that has STARTED (run_instance.log) but not finished (no report.json)
    # must NOT count — else a 32-worker sweep reads "finished: 32" seconds in.
    job = tmp_path / "swebench_verified-full-TS"
    _mk_swe_instance(job, "xrlenv-oracle-sweep", "django__django-1")                 # done
    _mk_swe_instance(job, "xrlenv-oracle-sweep", "astropy__astropy-2", completed=False)  # in-flight
    assert rb._benchmark_progress(tmp_path, "swebench_verified-full")["finished"] == 1


def _mk_errored(job_dir: Path, task_id: str, suffix: str) -> None:
    """A trial dir that errored: exception.txt, no result.json."""
    d = job_dir / f"{task_id}__{suffix}"
    d.mkdir(parents=True)
    (d / "exception.txt").write_text("boom")


def test_benchmark_progress_error_and_retried_and_invariant(tmp_path: Path) -> None:
    main = tmp_path / "bench-full-TS"
    _mk_trial(main, "taskA", "a1")                       # finished
    _mk_errored(main, "taskB", "b1")                     # currently errored
    _mk_trial(main, "taskC", "c1", done=False)           # in-flight (neither)
    p = rb._benchmark_progress(tmp_path, "bench-full")
    assert p["finished"] == 1 and p["error"] == 1 and p["retried"] == 1
    total = 5                                            # say the benchmark has 5 tasks
    remaining = total - p["finished"] - p["error"]
    assert p["finished"] + p["error"] + remaining == total   # the invariant holds


def test_benchmark_progress_errored_then_graded_is_finished_not_errored(tmp_path: Path) -> None:
    # taskB errored on attempt 0, then a retry graded it → it's finished, NOT errored
    # (disjoint), but retried still counts the infra-error occurrence (churn).
    _mk_errored(tmp_path / "bench-full-TS", "taskB", "b0")           # attempt-0 error
    _mk_trial(tmp_path / "bench-full-TS-retry1", "taskB", "b1")      # retry graded it
    p = rb._benchmark_progress(tmp_path, "bench-full")
    assert p["finished"] == 1 and p["error"] == 0 and p["retried"] == 1


def test_benchmark_progress_swe_error_retried_unmeasurable(tmp_path: Path) -> None:
    job = tmp_path / "swebench_verified-full-TS"
    _mk_swe_instance(job, "xrlenv-oracle-sweep", "django__django-1")
    p = rb._benchmark_progress(tmp_path, "swebench_verified-full")
    assert p["swe"] is True and p["error"] == 0 and p["retried"] is None
    assert p["finished"] == 1


# ── _profile_budget (per-profile max_concurrent_tasks, level OR overrides) ─────


def test_profile_budget_profile_level() -> None:
    cfg = {"max_concurrent_tasks": 64}
    assert rb._profile_budget({"mode": "full", "max_concurrent_tasks": 192}, cfg) == 192


def test_profile_budget_from_overrides_block() -> None:
    # An operator who puts the budget under `overrides:` (a natural placement) must
    # have it RESPECTED, not silently ignored — the whole point of this run's fix.
    cfg = {"max_concurrent_tasks": 64}
    prof = {"mode": "full", "overrides": {"max_concurrent_tasks": 192}}
    assert rb._profile_budget(prof, cfg) == 192


def test_profile_budget_profile_level_beats_overrides() -> None:
    cfg = {"max_concurrent_tasks": 64}
    prof = {"max_concurrent_tasks": 100, "overrides": {"max_concurrent_tasks": 192}}
    assert rb._profile_budget(prof, cfg) == 100


def test_profile_budget_falls_back_to_top_level_then_default() -> None:
    assert rb._profile_budget({"mode": "full"}, {"max_concurrent_tasks": 64}) == 64
    assert rb._profile_budget({"mode": "full"}, {}) == 128


def test_profile_budget_overrides_scalar_does_not_break_per_benchmark_effective() -> None:
    # A scalar max_concurrent_tasks living alongside per-benchmark dicts in `overrides`
    # must not be mistaken for a benchmark named "max_concurrent_tasks".
    cfg = {
        "defaults": {}, "benchmarks": {"swebench_verified": {"workers": 32}},
        "profiles": {"p": {"mode": "full",
                           "overrides": {"max_concurrent_tasks": 192,
                                         "swebench_verified": {"workers": 8}}}},
    }
    eff = rb._effective(cfg, "p", "swebench_verified")
    assert eff["workers"] == 8            # per-benchmark override still applies
    assert "max_concurrent_tasks" not in eff   # scalar not leaked into the benchmark eff
    assert rb._profile_budget(cfg["profiles"]["p"], cfg) == 192


def test_rewards_pass_swebench_pro_keys_on_reward_only() -> None:
    # swebench_pro's reward.json carries diagnostic counts; p2p_total may legitimately be 0
    # (an instance with no PASS_TO_PASS tests) — only the ``reward`` key decides.
    resolved = {"reward": 1, "resolved": 1, "f2p_total": 3, "f2p_passed": 3, "p2p_total": 0, "p2p_passed": 0}
    assert rb._rewards_pass(resolved, "swebench_pro") is True
    assert rb._rewards_pass(dict(resolved, reward=0, resolved=0), "swebench_pro") is False
    assert rb._rewards_pass({"f2p": 1.0, "p2p": 1.0}, "swebench_pro") is False      # no reward key => fail

# Benchmark integration smokes

← back to [smoke runbook index](../README.md)

These drive a **real upstream harness** through xrlenv to catch
contract regressions a unit test can't see by construction.
Every change to `xrlenv/compat/docker_client.py` or any case-3
adapter should pass the smoke for that benchmark before
shipping.

| Smoke | What it validates |
|---|---|
| [`test_swebench_drop_in.py`](#test_swebench_drop_inpy) | SWE-bench Verified resolves an instance via the docker-py drop-in (case-2). Pins the upstream-harness contract: any change to `xrlenv/compat/docker_client.py` that breaks this is shipped-broken. |
| [`test_terminal_bench_2_drop_in.py`](#test_terminal_bench_2_drop_inpy) | One harbor task resolves via `XrlenvHarborEnvironment` + harbor's stock oracle agent (case-3). Pins the harbor adapter's contract with upstream task format + verifier. |

See [Conventions shared across smokes](../README.md#conventions-shared-across-smokes)
for invocation patterns, the three-mode structure, artifact
output, and cleanup recipes that apply across all groups.

---

## `test_swebench_drop_in.py`

**Group**: Benchmark integration (case-2). **Wall-clock**: ~5-10
min first run (image pull dominates), ~30-90 s per subsequent run.
**Modes**: pytest, script (single-instance default; `--instance-ids`
for multi).

**What it validates.** SWE-bench's stock
`swebench.harness.run_evaluation.main()` resolves a Verified
instance through `xrlenv.from_env()` end-to-end. Pulls a real
SWE-bench image (~1-3 GiB), runs the upstream harness unmodified,
asserts the per-instance `report.json` shows `resolved=True`. The
gold-patch path is wired so every instance should resolve; an
unresolved instance under the gold patch is a plumbing bug, not
an agent bug. **The architectural pin** — any change to
`xrlenv/compat/docker_client.py` that breaks the upstream-harness
contract is shipped-broken.

**Prerequisites.**
- Docker daemon reachable.
- `swebench` and `datasets` Python packages installed
  (`uv pip install swebench datasets`). Skip markers handle their
  absence.
- ~5 GB free disk for the default single-instance run; ~20 GB for
  the 8-instance soak.
- Internet access to pull from Docker Hub `swebench/sweb.eval.x86_64.*`
  on first run (cached after).

**Invocation.**

```bash
# pytest single-instance (default sphinx-doc__sphinx-10323):
.venv/bin/python -m pytest tests/smoke/test_swebench_drop_in.py -v

# script — single instance, no archiving:
.venv/bin/python tests/smoke/test_swebench_drop_in.py

# script — single instance, archive to <repo>/tmp/ (default; gitignored):
.venv/bin/python tests/smoke/test_swebench_drop_in.py --save-artifacts

# script — broader 8-instance soak, custom out-of-repo archive root:
.venv/bin/python tests/smoke/test_swebench_drop_in.py \
    --instance-ids \
    astropy__astropy-7166,django__django-11099,sympy__sympy-18189,astropy__astropy-12907,astropy__astropy-14182,sympy__sympy-13615,django__django-11138,sympy__sympy-12489 \
    --max-workers 2 \
    --save-artifacts "$XRLENV_SMOKE_ARCHIVE_ROOT" \
    --job-id claude-opus-4-7-50-v1.12.0
```

**Output.** Under `<save-artifacts>/<job-id>/`:

```
summary-<utc-ts>.json                                           # per-run snapshot
logs/run_evaluation/<run_id>/<model>/<instance>/                # swebench's per-instance tree
    report.json   run_instance.log   test_output.txt   eval.sh   patch.diff
```

**What "pass" means.** `summary["resolved_instances"] ==
summary["total_instances"]` — every requested Verified instance
resolved under the gold patch. Any non-resolution is a plumbing
regression (gold patches are by construction the canonical fix);
investigate `report.json` + `test_output.txt` for the failing
instance.

---

## `test_terminal_bench_2_drop_in.py`

**Group**: Benchmark integration (case-3). **Wall-clock**: ~3-5
min single task; ~15-30 min for the 8-task `SMOKE_8` set.
**Modes**: pytest, script.

**What it validates.** Drives one harbor task end-to-end through
harbor's stock `Trial` runner with `import_path` pointed at our
`XrlenvHarborEnvironment` plug-in. The agent is harbor's
**oracle agent**, which doesn't call an LLM — it reads
`<task_dir>/solution/`, uploads it into the container at
`EnvironmentPaths.solution_dir`, and execs `solve.sh`. After that,
harbor's verifier grades the post-fix state. **A pass confirms
the runtime path harbor → `XrlenvHarborEnvironment` → docker
container → solve.sh → verifier → rewards is wired end-to-end.**
This is the case-3 (per-framework adapter) analogue of
`test_swebench_drop_in.py`'s case-2 gate.

**Prerequisites.**
- Docker daemon reachable.
- Harbor task suite cached at `$HARBOR_TASKS_DIR` (default
  `~/.cache/harbor/tasks/`). The 8-task `SMOKE_8` reference set is
  the phase-0 acceptance set.
- ~2 GB free disk for the default single-task run.
- Per-task images pulled or pullable from Docker Hub
  (`alexgshaw/<task>:20251031`).

**Invocation.**

```bash
# pytest single-task (default fix-git):
.venv/bin/python -m pytest tests/smoke/test_terminal_bench_2_drop_in.py -v

# script — single task, no archiving:
.venv/bin/python tests/smoke/test_terminal_bench_2_drop_in.py

# script — single task, archive to <repo>/tmp/:
.venv/bin/python tests/smoke/test_terminal_bench_2_drop_in.py --save-artifacts

# script — 8-task SMOKE_8 soak:
.venv/bin/python tests/smoke/test_terminal_bench_2_drop_in.py \
    --task-ids \
    fix-git,build-pov-ray,overfull-hbox,cobol-modernization,prove-plus-comm,constraints-scheduling,nginx-request-logging,dna-insert \
    --save-artifacts "$XRLENV_SMOKE_ARCHIVE_ROOT" \
    --job-id claude-opus-4-7-50-v1.12.0
```

**Output.** Under `<save-artifacts>/<job-id>/`:

```
summary-<utc-ts>.json                              # per-run snapshot
trials/<task>__<short_id>/                         # harbor's per-trial tree
    config.json   result.json   trial.log
    agent/oracle.txt                               # solve.sh stdout/stderr
    verifier/{ctrf.json, reward.txt, test-stdout.txt}
```

**What "pass" means.** Every task's `result.json` reports
`reward > 0` and harbor's verifier emits a non-empty
`verifier/reward.txt`. A zero or missing reward under the oracle
solution is a plumbing regression in `XrlenvHarborEnvironment` or
the harbor adapter — not the task author's fix.

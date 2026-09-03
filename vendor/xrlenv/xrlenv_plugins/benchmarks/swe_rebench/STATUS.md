# SWE-rebench — status

**2026-09-02 · every task in the 856-task green set has been observed passing,
but not yet in one clean run.** The 860-task sweep measured 855/860; of its 5
misses, 4 are now excluded as ungateable upstream content and the 5th
(`guppylang-1259`) is fixed by a hermeticity marker and individually re-verified
green from the shared cache. A single end-to-end 856/856 sweep is still
outstanding — and worth running, since the recorded sweep predates both that
marker and the exec-deadline fix below.

## Corpus

| | |
|---|---|
| Source | Harbor Hub package [`swe-rebench/swe-rebench-leaderboard`](https://hub.harborframework.com/datasets/swe-rebench/swe-rebench-leaderboard/latest) |
| Version | `sha256:ebe7444e313a0d8db94fa541139826eaebe2b0abcd4900c6f73e750494910dca` (`@latest`) |
| Tasks | 860, all oracle-gateable |
| Harness | harbor 0.20.0, golden path (`XrlenvHarborEnvironmentCluster`) |
| Images | `swerebench/sweb.eval.x86_64.<slug>:latest`, all upstream prebuilts — **nothing is built** |
| Image plan | 860 × `type: registry`, all `registry-probe` sized; 1.19–12.70 GB, **1.65 TB** total |
| Monthly splits | 15 (`2025_01`…`2026_03`), union == the corpus; `--split` selects |
| Green set | **856** — 4 excluded (see below) |

Upstream adds a split monthly. On a refresh: re-pin `EXPECTED_PRESENT` in
`run_full_sweep.sh`, update the version above, re-run
`scripts/fetch_monthly_splits.py`.

## Gate configuration

| Knob | Value |
|---|---|
| Concurrency | `--max-workers 32` (any value is safe; xrlenv paces capacity) |
| Timeouts | native — no `--timeout-multiplier` (3000 s agent + verifier; 4 tasks 6000 s) |
| `--retries` | 6, infra-transient only |
| `--content-retries` | **0** — zero-tolerance; a task that only passes on a re-run is a finding, not a pass |
| Pass gate | every reward key `> 0` (upstream `parser.py` resolved the instance) |

## Last run

**855 / 860** · conc-32 · `tmp/sanity-checks/swe-rebench-full-sweep-2026-09-01_05-13-28`
(839/860 first pass; the 21 failures were re-run in place after the resource
routing landed, recovering all 16). Timing: median 361 s, p90 706 s, 96.5 h total
oracle work; slowest `UXARRAY__uxarray-1423` at 32.5 min (passed — pytest itself
reports 24 min).

Read the timings with care: that run predates the exec-deadline fix, so its long
verifiers were truncated at the plug-in's old flat 1800 s cap rather than their
declared 3000 s. The slow tail (33.4 h of the 96.5 h) is worth re-measuring.

**The exec-deadline fix.** The plug-in defaulted every `exec` to a flat 1800 s,
which became the binding constraint for any task declaring more — a truncated log
and reward 0, indistinguishable from broken content. `_default_exec_timeout_s`
now reads the task's own declared agent/verifier budget and the trial's timeout
multipliers, so the transport deadline is never tighter than what harbor itself
will enforce. Every swe-rebench task declares 3000 s; 4 declare 6000 s.

## Resource routing — 16 tasks

harbor sets a CFS cpu quota + memory cap but **no cpuset**, so `nproc` in a
`cpus = 1` container reports the host's 192 cores. Pools sized from
`os.cpu_count()` (joblib/loky, xdist `-n auto`, dask/ray, OMP/BLAS) fan out
~192-way inside an 8 GB cap and are SIGKILL'd — this cost 16 tasks.

`build_cache.py --stage patch` writes a per-task
`[environment.env] XRLENV_CPU_PINNING = "1"` for `CPU_PINNING_TASKS` (16), plus
`MEMORY_OVERRIDES` for 4 of them. Re-verified **16/16 solved** through the
markers alone (`tmp/triage/markers-e2e`).

| Override | Task |
|---|---|
| 16G | `ImperialCollegeLondon__virtual_ecosystem-1232`, `calliope-project__calliope-854`, `pybamm-team__PyBaMM-4871` |
| 32G | `owkin__PyDESeq2-356` |

A memory override is permitted **only where upstream declared none**;
`_assert_memory_override_is_fair` enforces it. 850 of 860 tasks have
`harbor_cpus`/`harbor_memory` `null` in `tests/config.json` (their `1 cpu / 8G`
is the converter's default); the 10 upstream sizes itself (2 cpu / 16 G) are
never overridden — the 3 sktime tasks among them needed pinning only.

## Persistence — all fixes survive a cache wipe

`--stage patch` runs inside `--stage all`, which is `run_full_sweep.sh` step 1,
so nothing here is a manual step anyone can forget. Re-verified 2026-09-02: a
from-scratch `--stage all` into an empty root (populate → repin → patch, 860
downloaded) produced **860/860 `task.toml`s byte-identical** to the live cache —
16 `XRLENV_CPU_PINNING` markers, 4 memory overrides, 1 `UV_NO_SYNC`, and all 860
`docker_image` repins included. `rm -rf` the cache and it all comes back.

## Hermeticity routing — 1 task

`CQCL__guppylang-1259`'s `test.sh` runs `uv run pytest`, and `uv run` re-resolves
the workspace on every invocation. PEP-517 **build** requirements are not covered
by the lockfile, so the resolve pulls whatever hatchling is current on PyPI —
since 2026-08 that is 1.32.0, which rejects the task's `readme = "../README.md"`.
The package never builds and every F2P plus 33 P2P report `NOT_FOUND`. The task
was authored 2025-09 against a hatchling that accepted it.

`build_cache.py --stage patch` writes `UV_NO_SYNC` = `"1"` into
`[environment.env]` (the `HERMETICITY_ENV` table), telling `uv run` to use the
environment the image already ships. Verified from the shared cache
(`tmp/triage/guppy-from-cache`, i.e. through `build_cache`, not a hand-edit):
reward 1, F2P `PASSED`, 33/33 P2P `PASSED`, `34 passed in 35.88s`, and **zero**
`Downloading` / `Building` / `Resolved N packages` lines — so this fixes the grade
*and* removes a live PyPI dependency from the verify phase.

This is deliberately a separate table from the resource routing: it changes how
the verifier resolves its dependencies, never its compute envelope, so the
fairness guard does not apply.

## Excluded (4) — ungateable upstream content

Non-hermetic — reach a live third-party API during verification, so no retry,
resource change or pinning makes them deterministic:

| Task | Evidence |
|---|---|
| `canonical__charmcraft-2084` | 4 P2P query Charmhub for real → `LibraryError: Library charms.mysql.v0.mysql not found in Charmhub` |
| `sigma67__ytmusicapi-909_interface` | 14 × `KeyError: 'auth'`; needs YouTube Music credentials, and pins nothing about a third-party schema |

Ungateable on any hardware:

| Task | Evidence |
|---|---|
| `bluesky__ophyd-async-1165` | its pinned commit `9b567b2` sets a repo-**global** `[tool.pytest.ini_options] timeout = 1`. The P2P `test_device_with_children_lazily_connects` drives a bluesky RunEngine round-trip through that 1 s budget and misses it intermittently. Both F2P pass every run. Upstream has since raised the setting to `timeout = 60`, and two sibling timing tests in the same file are already upstream-`xfail("Flaky test")` — they XPASS here |
| `modin-project__modin-7434` | zero progress: collects 5 items, prints `test_simple_import`, emits nothing further. 3 runs, pinned and not, at conc 1 and 32. Under its real 3000 s budget it burns the full 52.9 min and raises `VerifierTimeoutError`. No partial result to parse |

`ophyd-async-1165` was the one task that looked concurrency-sensitive. It is not
— it fails solo. Interleaved A/B, 8 pairs on the same box:

| `cpus` | oracle passes |
|---|---|
| 1 (stock) | 6 / 8 |
| 8 | 6 / 8 |

Identical, so **more CPU is not the remedy**; `cpus = 2` and `XRLENV_CPU_PINNING`
were also tried with no effect. On a loaded box the rate drops to ~1/4. A task
that flips a 25–75 % coin carries no gate signal, whatever it is given.

## Can more cpus/memory accelerate the slow tail?

No — checked, not assumed. Of the 157 tasks over 10 minutes, **none** invokes a
parallel test runner (no `-n auto`, no `xdist`, no parallel make), so extra cores
cannot shorten a serial pytest. The gains available from resources are exactly
the 16 already routed, and they come from *pinning* — stopping a 192-way fan-out,
not from adding cores. `UXARRAY__uxarray-1423` went 1837 s → 198 s (9.3×) at
identical concurrency on the marker alone.

## Reproduce

```bash
cd "$(git rev-parse --show-toplevel)"
set -a; source ./.env; set +a
S=xrlenv_plugins/benchmarks/swe_rebench
nohup bash $S/run_full_sweep.sh --max-workers 32 > /tmp/g1.log 2>&1 &   # full gate
```

Artifacts land in `./tmp/sanity-checks/<job-id>/`; a failure's
`verifier/report.json` names the exact tests.

## Checks

`pytest .../swe_rebench/tests -q` → **102 passed**; `pytest .../harbor/tests
tests/unit/plugins/harbor -q` → **200 passed** · ruff clean on every touched file
· `sphinx -W` clean · no baked registry host. The repo-wide suite's remaining
reds are pre-existing and environmental (this login box has no docker daemon);
repo-wide mypy/ruff findings are all in files this kit never touches.

## Retry only the failures (in place)

harbor resumes an existing job dir: it keeps every trial that has a
`result.json` and runs only the ones missing it. So delete the failed trial dirs
and re-run with the **identical** config — no `--tasks` (that changes the config
and harbor raises `FileExistsError`), `--content-retries 0` (a retry round
mutates `job_name`), and the same `--max-workers` / `--retries` / `--jobs-dir`.

```bash
J=tmp/sanity-checks/swe-rebench-full-sweep-2026-09-01_05-13-28

.venv/bin/python - "$J" <<'EOF'
import json, os, shutil, sys
J = sys.argv[1]
FAILED = {"CQCL__guppylang-1259", "ImperialCollegeLondon__virtual_ecosystem-1232",
"SciTools__iris-6754", "bluesky__ophyd-async-1165", "calliope-project__calliope-854",
"canonical__charmcraft-2084", "copier-org__copier-2646",
"joshuadavidthomas__django-bird-239", "modelcontextprotocol__python-sdk-1864",
"modin-project__modin-7434", "networkx__networkx-8369", "owkin__PyDESeq2-356",
"pybamm-team__PyBaMM-4871", "sigma67__ytmusicapi-909_interface", "sktime__skpro-574",
"sktime__sktime-8723", "sktime__sktime-8921", "sktime__sktime-8937",
"vyperlang__vyper-4462", "vyperlang__vyper-4677", "vyperlang__vyper-4801"}
for d in sorted(os.listdir(J)):                 # dir names truncate at 32 chars,
    p = f"{J}/{d}/config.json"                  # so map via each dir's own config
    if os.path.isfile(p) and os.path.basename(str(json.load(open(p))["task"]["path"])) in FAILED:
        shutil.rmtree(f"{J}/{d}")
EOF

nohup .venv/bin/python xrlenv_plugins/benchmarks/swe_rebench/run_oracle_sweep.py \
  --max-workers 32 --retries 6 --content-retries 0 \
  --jobs-dir tmp/sanity-checks --job-id swe-rebench-full-sweep-2026-09-01_05-13-28 \
  > /tmp/retry21.log 2>&1 &
```

The 16 resource-routing markers are already in the cache, so those tasks pick
them up on this run. `result.json` then covers all 860.

# swebench-pro — STATUS (2026-08-27)

The status of the **full gold-patch (oracle) sweep** — all 731 public instances of
`ScaleAI/SWE-bench_Pro`, the corpus as published. The partitions (filtered 478, subset-100, the
one-task smoke) and what has been verified on them are tracked in `scripts/README.md`.

## Inputs

- Dataset `ScaleAI/SWE-bench_Pro` (731 public rows) + the upstream kit `SWE-bench_Pro-os` (run
  scripts, parsers, Dockerfiles for all 731). Both are public and ungated, so `build_cache.py`
  fetches them itself — the only required input is `XRLENV_BENCHMARK_CACHE`. `SWEBENCH_PRO_PARQUET`
  / `SWEBENCH_PRO_HARNESS` remain as overrides to pin a local copy (see README).
- Images: prebuilt on Docker Hub (`jefzda/sweap-images:<tag>`), pulled on first use. Plan
  `build_plan_full.yaml`: 731 images, 1071 GB compressed (registry-probed). No warm-up is needed.

## Full sweep (731) — RUN 2026-08-27: 729 / 731 in the sweep; 731 / 731 after the element-web sizing fix + the tutanota re-run outside its failing time window

| | |
|---|---|
| job id | `swebench-pro-full-sweep-2026-08-27_21-12-56` (artifacts `tmp/sanity-checks/<job-id>/`, retry rounds consolidated in) |
| date | 2026-08-27 |
| command | `run_full_sweep.sh --max-workers 64 --content-retries 2` |
| resolved (first pass / after retry 1 / after retry 2) | 721 / 729 / **729** of 731 |
| resolved after the fixes below | **731** of 731 (element-web: sizing fix; tutanota: outside its failing time window) |
| oracle FAILs in the sweep (each is a corpus/plumbing defect, never a model signal) | 2 — see below |
| `EXCLUDE` entries in `run_full_sweep.sh` (with reason) | none — element-web fixed in `build_cache.py`; tutanota is a known wall-clock-dependent upstream test (see below), kept in the corpus as published |

The 8 first-pass FAILs that passed on retry were transient (jest suites failing as a unit, a Go
link step failing once, a `tool/tsh` 5-min test timeout) — the content retry exists for exactly these.

The two that failed all 3 attempts, gold patch applied cleanly both times:

- `instance_element-hq__element-web-53a9b6447bd7e6110ee4a63e2ec0322c250f08d1-vnan` — **FIXED
  (plumbing)**. In the sweep: f2p 15/15 but p2p 292–325/330 on all 3 attempts, jest workers
  **SIGKILLed** (`A jest worker process … was terminated by another process: signal=SIGKILL`) with a
  different set of suites dying each time — the 16 GiB memory limit's OOM killer (`npx jest` runs
  without `--maxWorkers`, so the worker count follows nproc). Fix: `HEAVY_REPOS` now gives
  element-web 32768 MB, and `refresh_kit_files` also refreshes `task.toml` so sizing changes reach an
  existing cache. Isolation re-run at 32 GiB (2026-08-27 22:42 UTC, job
  `swebench-pro-rerun-elementweb-53a9b644-32g`): **resolved** — f2p 15/15, p2p 330/330, 0 SIGKILL.
- `instance_tutao__tutanota-f373ac3808deefce8183dad8d16729839cc330c1-v2939aa9f4356f0dc9f523ee5ce19d09e08ab979b`
  — **upstream test defect, wall-clock-dependent; no kit-side fix.** f2p 1/2, p2p 2953/2953 on all
  3 attempts (21:22–22:30 UTC). The failing test,
  `test/tests/calendar/eventeditor/CalendarEventWhenModelTest.ts` "setting all-day to false will
  cause result to not be considered all-day and the times to be set to the default", builds an
  all-day event on the **UTC** date of `new Date()` (`DateTime.fromJSDate(now, { zone: "utc" })`)
  but resolves it through a model pinned to **Europe/Berlin** (`getModelBerlin`) and then applies a
  "next half hour from now" default time. Whenever the Berlin date is ahead of the UTC date —
  22:00–00:00 UTC in summer, 23:00–00:00 in winter — the result lands one day behind the
  expectation, which is exactly the assertion seen (`expected '2026-08-26T22:30:00.000Z' to be equal
  to '2026-08-27T22:30:00.000Z'`; the expected value tracks the wall clock). Upstream's own comment on
  the test admits it is time dependent; the image runs `Etc/UTC` with no `TZ`, and setting one would
  not help because both zones are hard-coded in the test. Patching upstream's test is out of scope
  for the plug-in. The task passes ~22 h/day. It is already dropped by our filter (5/5 votes,
  `low_coverage_tests`) and is in neither the filtered set nor subset-100 — only in **full**, the
  corpus as published. Decision: leave it in the full corpus and record it here as a known
  time-window failure; an `EXCLUDE` entry ("wall-clock-dependent upstream test; fails 22:00–00:00
  UTC") is the option if the full gate must be green at any hour. **Confirmed** 2026-08-28 00:07 UTC
  (job `swebench-pro-rerun-tutanota-f373ac3-after-midnight`): **resolved** — f2p 2/2, p2p 2953/2953,
  no assertion errors. So the corpus is 731/731 resolvable outside the 22:00–00:00 UTC window.

## Integration runner (`xrlenv_plugins/benchmarks/tests/integration`)

`swebench_pro` is wired into `benchmarks.yaml` (`workers: 32, retries: 6`; pass rule = `reward` key only).
Live `ci` profile, 2026-08-27: **PASS 2/2** (green 731/731; sample `flipt-e42da21a…`,
`navidrome-677d9947…`; coverage 2/2, exit 0). `--profile full-prod --benchmark swebench_pro` = the full gate
above through the runner (note `defaults.content_retries: 0` there vs. this kit's default of 1).

## Reproduce

```bash
# .env: XRLENV_BENCHMARK_CACHE, XRLENV_GRPC_HOST/_PORT/_TOKEN  (dataset + upstream kit are fetched)
.venv/bin/python xrlenv_plugins/benchmarks/swebench_pro/build_cache.py --all              # 731 task dirs (idempotent)
bash xrlenv_plugins/benchmarks/swebench_pro/run_full_sweep.sh --skip-build-cache --list-green | grep -c '^instance_'   # expect 731
bash xrlenv_plugins/benchmarks/swebench_pro/run_full_sweep.sh --max-workers 16 --content-retries 1
```

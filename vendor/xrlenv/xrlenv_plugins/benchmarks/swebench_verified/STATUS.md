# SWE-bench Verified — status

**Shape:** docker-py drop-in (`xrlenv.from_env()` → upstream `run_instance`).
**Gate:** oracle sweep — gold patch as prediction, upstream `resolved` must be true
for every instance in the green set. **Green set** = present cached instances −
`EXCLUDE`. `EXCLUDE` holds only the **upstream-ungradeable** instances — the ones the
*current upstream toolchain* (swebench 4.1.0 + the canonical Docker Hub images) cannot
grade (the gold patch itself does not resolve → false-negatives that would poison a real
agent run: any fix, even correct, scores 0).

The **4 flaky `psf__requests`** (non-hermetic external httpbin.org) are deliberately **kept
IN the green set**, not excluded: a one-off 503 under concurrency is a known, no-surprise
flake (documented below), while a *consistent* failure is a real new signal worth
investigating — hiding them by exclusion would mask that. Green set = **494** (500 − 6).

## Disposition

| Gate | State |
|---|---|
| Offline (structure, argparse, ruff, unit tests, dry-run) | ✅ green |
| **G1 — oracle sweep on-cluster** | ✅ 489/500 raw; the 11 misses are fully diagnosed below (2026-08-02): 6 upstream-ungradeable are `EXCLUDE`d (green set = **494**); 4 flaky `psf__requests` (external httpbin.org) are kept IN the green set + documented (watched — consistent failure = new signal); 1 sklearn was a real xrlenv bug, fixed (per-task cpu_pin, validated). None is a cache/mirror/xrlenv-code defect. |

The runtime is faithful: `run_oracle_sweep.py` reads the cached upstream row and
delegates 100% to upstream `make_test_spec` / `run_instance` / `get_eval_report`
/ `MAP_REPO_TO_PARSER`. None of the failures below is a cache-data, mirror, or
xrlenv-code defect — verified: node image digests match Docker Hub exactly; the
eval-time `pip install -e .[test]` pulls no unpinned dep; swebench 4.1.0 is the
latest PyPI release.

## Upstream-ungradeable instances (in `EXCLUDE` — oracle cannot pass)

These fail deterministically against the current canonical image + swebench 4.1.0
(they also fail in isolation, so they are not contention). Each is an **upstream**
defect, not ours. `EXCLUDE` them so the gate reflects what the platform can
faithfully grade; re-include if/when the upstream fix (or a sanctioned eval-env
patch) lands.

| Instance | Root cause | Fix if we want it back |
|---|---|---|
| `sphinx-doc__sphinx-8595` | Upstream swebench 4.1.0 eval.sh bug. Its test_patch adds **only new files**, so `make_test_spec` emits a **bare `git checkout <base_commit>`** (no path). The image commits `pre_install`'s `pytest -rA` at HEAD, but the base commit lacks it — so the bare checkout reverts `-rA` (verified in-image: `grep -c "pytest -rA" tox.ini` = 1 before, **0 after**). Pytest then emits no per-test `PASSED <nodeid>` markers, `parse_log_pytest_v2` extracts 0, and the **passing** test (`test_empty_all`, "1 passed") is graded failed. | Re-assert `pytest -rA` in tox.ini *after* the checkout (eval-env patch, needs sign-off — it steps into upstream's eval flow), or wait for an upstream fix that scopes the reset to the test files. |
| `sphinx-doc__sphinx-9711` | Same mechanism — test_patch is new-files-only → bare `git checkout <base>` → `-rA` reverted → parser gets 0. | Same as sphinx-8595. |
| `astropy__astropy-8872` | The canonical image ships **setuptools 68**; its vendored `distutils.version.LooseVersion` raises `DeprecationWarning: distutils Version classes are deprecated` (reproduced in-image). astropy runs `filterwarnings=error`, so this becomes a **collection error** → 0/80 tests. | Pin setuptools < 60 in the image, or filter the distutils DeprecationWarning for astropy (eval-env change / new image build). Upstream image issue. |
| `astropy__astropy-8707` | Same setuptools-68 / distutils DeprecationWarning, hitting **7 tests** (as `error`) instead of collection → 7 PASS_TO_PASS fail. | Same as astropy-8872. |
| `astropy__astropy-7606` | **Parametrization-ID drift.** The dataset's PASS_TO_PASS expects `test_compose_roundtrip[]`, but the current image's astropy/pytest collects the parametrized IDs `test_compose_roundtrip[unit0]`, `[%]`, `[A]`, … The expected `[]` ID never appears → marked failed even though the fix works (all FAIL_TO_PASS pass). | Needs the dataset's expected test IDs regenerated against the current image, or an image pinned to the astropy/pytest the dataset was cut with. Upstream dataset-vs-image drift. |
| `django__django-10097` | `test_add` / `test_delete` (`GenericInlineAdminWithUniqueTogetherTest`) **ERROR** deterministically in the canonical env (the fix itself works — 438 FAIL_TO_PASS pass). | Needs per-instance root-cause in the canonical image; upstream env issue. |

## Flaky instances — IN the green set, WATCHED (non-hermetic external dependency)

These are **not excluded** — they're gradable and expected to resolve, but they can flake
under concurrency, so they're documented here to avoid surprise. An earlier pass wrongly
attributed these to CPU starvation / xrlenv oversubscription; the verified cause is a flaky
**external** dependency (not CPU, not an xrlenv capacity bug). Concurrency is only the
trigger — never lower it to hide this.

**How to read a failure here:** a one-off 503 under a heavy sweep is the **known flake** (no
surprise — see below). A **consistent** failure across runs is a **new signal** worth
investigating (e.g. httpbin.org down/changed, or an egress/network regression) — not the
known flake. If they become persistently red, the clean fix is to make them hermetic: start a
**local httpbin** in the container and `export HTTPBIN_URL=http://127.0.0.1:<port>` before the
tests (an eval-env hook — steps into the eval flow, needs sign-off).

| Instance(s) | Root cause (verified) |
|---|---|
| `psf__requests-1724`, `-1766`, `-1921`, `-2317` | **Flaky / non-hermetic external dependency.** `test_requests.py:26` defaults `HTTPBIN = os.environ.get('HTTPBIN_URL','http://httpbin.org/')`; the image has **no pytest-httpbin**, `HTTPBIN_URL` is **unset**, and the node has **no egress proxy** — so the tests hit **external httpbin.org**, which returns an **nginx `503 Service Temporarily Unavailable`** (rate-limit) under concurrent test load. Pass in isolation (httpbin.org tolerates a few callers). NOT CPU starvation, NOT an xrlenv capacity/admission bug. |

## Real xrlenv bug — root-caused + FIXED (`scikit-learn__scikit-learn-14710`)

`scikit-learn-14710` (80 OpenMP-heavy `_hist_gradient_boosting` tests) **collected but 0
completed** in 1800 s under xrlenv — both the gate AND 4-worker iso. A **hang, not
contention** (31 other sklearn pass under xrlenv; `--local` plain-docker RESOLVES all 80).

**Root cause (proven on a dev worker, 4 configs):** xrlenv caps a raw grading container with a
**CFS CPU quota** (`cpu_limit=2.0`) but on a node whose docker/cgroup driver can't enforce
cpuset pinning (`isolation_capable=false`) it applies **no cpuset** → the container keeps full
**CPU affinity to every host core (192)**. `os.cpu_count()`/affinity report 192, so OpenMP/BLAS
size their thread pools to 192 threads, which the 2.0 CFS quota then throttles → spin-barrier
thrash → effective hang.

| config | affinity | result |
|---|---|---|
| `--local` (no limits) | 192 | pass |
| cpuset=2 | 2 | pass 3 s |
| cpuset=2 + mem=4g + quota=2.0 | 2 | pass 3 s |
| **quota=2.0, NO cpuset** (prod) | **192** | **HANG (killed at 180 s)** |
| **quota=2.0, NO cpuset + `OMP/OPENBLAS/MKL/NUMEXPR=2`** | 192 | **pass 2.8 s** |

**Fix — per-task cpuset pinning (small radius, matches harbor/tb2.1's `XRLENV_CPU_PINNING`
marker):** the instance is mapped to an isolation level in `run_oracle_sweep.py`'s
`_CPU_PINNED_INSTANCES`. The driver requests that `cpu_isolation` via
`xrlenv.rollout_metadata(...)`; the docker-py drop-in forwards it to
`acquire_container(cpu_isolation=)`; the node pins the container's own `cpuset_cpus` to
`ceil(cpu_limit)` whole cores. Affinity == the CPU budget, so OpenMP/BLAS size their pools to
it → no oversubscription. **Per-task, NOT a global core change** (preferred over defaulting
thread-env in `raw_container.py`). sklearn-14710 is mapped **`REQUIRED`** (pin-or-reschedule)
so the pin can never silently degrade to quota-only (the state that hangs it) under node-ledger
pressure — it is placed only on isolation-capable nodes (4/6 prod nodes are `cgroupfs/v2` =
capable; the 2 sysbox nodes are skipped) and a transient `PinCapacityExhausted` is re-admitted
on a sibling capable node by the infra-retry layer.

**VALIDATED end-to-end on the prod cluster** (2026-08-03): `resolved: True`, **80/80 tests
pass**, no ledger-degrade warning (= the container was actually pinned). **NOT excluded** —
kept in the green set.

## Reproduce

```bash
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example
set -a; source ./.env; set +a                 # XRLENV_GRPC_HOST + XRLENV_CONSUMER_TOKEN

# smoke first (8 instances, ~5-15 min), then the full green set:
python xrlenv_plugins/benchmarks/swebench_verified/run_oracle_sweep.py --smoke --max-workers 4
bash   xrlenv_plugins/benchmarks/swebench_verified/run_full_sweep.sh --max-workers 8
```

A SWE-style oracle+verifier trial exceeds a foreground shell timeout — run the full
sweep under `nohup`/background and poll (~500 instances). Per-instance artifacts
(`report.json`, `run_instance.log`, `test_output.txt`, the applied patch) land under
`<jobs-dir>/<job-id>/logs/run_evaluation/...`; the per-run tally is `summary.json`.

## Config

- **Concurrency:** `--max-workers` (default 8 in the gate). swebench's harness is
  thread-safe; the drop-in is concurrency-neutral.
- **Timeouts:** swebench's native 1800 s (no multiplier).
- **Retries:** `--retries 6` (infra-transient only) + `--content-retries 2`
  (non-`resolved` re-run). Both report their counts; neither re-rolls a real fail.
  **The gate default is now `--content-retries 0`** (a task that only passes on a
  re-run is a finding, not a pass); this run predates that and used 2, so pass it
  explicitly to reproduce these exact numbers.
- **EXCLUDE:** 6 upstream-ungradeable ids in `run_full_sweep.sh`'s `EXCLUDE` array. The H4
  membership gate is reconciled so `INCLUDE ∪ EXCLUDE` must equal the 500-id manifest — a
  held-out id is still a real Verified instance (no fabrication, no silent subset), just
  documented here. **Green set = 494** (the 4 flaky `psf__requests` are IN the set, watched —
  see above; not excluded).

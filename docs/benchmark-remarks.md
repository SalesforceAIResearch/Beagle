# Benchmark remarks — tasks we suggest you exclude

`beagle` runs **whatever the benchmark's corpus contains**. It does not filter tasks for you: a task's oracle
can give you zero reward if upstream dependencies are missing, changed or broken, e.g. a verifier that queries a 
live third-party API, a test whose one-second global timeout flakes, or a
suite that hangs. Their reward ceiling is 0, so including them **understates every agent equally**
and adds noise to comparisons between them. This page lists the ones we have measured, with the
evidence, so you can decide.

Exclusions are per run, in your config:

```yaml
data:
  - benchmark: swe-rebench
    exclude_task_ids: [canonical__charmcraft-2084, ...]
```

Each list below mirrors the exclusion set the corresponding xrlenv onboarding kit gates on
(`vendor/xrlenv/xrlenv_plugins/benchmarks/<kit>/run_full_sweep.sh`), which is where the measurements
were made. If you re-run the oracle sweep and a task recovers, drop it from your list — nothing here
is enforced in code.

---

## `swe-rebench` — 860 tasks, 856 green

The oracle sweep measured **855/860**. Four tasks are excluded on measured evidence.

```yaml
    exclude_task_ids:
      - canonical__charmcraft-2084
      - sigma67__ytmusicapi-909_interface
      - bluesky__ophyd-async-1165
      - modin-project__modin-7434
```

**Non-hermetic — the verifier reaches a live third-party API.** No retry, resource change or pinning
makes these deterministic.

| Task | What happens |
|---|---|
| `canonical__charmcraft-2084` | `test_store_commands.py` queries Charmhub for real → `LibraryError: Library charms.mysql.v0.mysql not found in Charmhub` (4 P2P tests) |
| `sigma67__ytmusicapi-909_interface` | hits YouTube Music unauthenticated → 14× `KeyError: 'auth'`, and the live response shape no longer matches what the task recorded. Needs credentials *and* pins nothing about a third-party schema |

**Ungateable upstream content — the oracle patch is correct, the task cannot be graded
deterministically on any hardware.**

| Task | What happens |
|---|---|
| `bluesky__ophyd-async-1165` | its pinned commit sets a repo-global `[tool.pytest.ini_options] timeout = 1` — one second for every test. A P2P test drives a bluesky RunEngine round-trip through that budget and misses it intermittently. Measured A/B, 8 interleaved pairs: `cpus=1` → 6/8, `cpus=8` → 6/8 — identical, so not a resource problem; on a loaded box it drops to ~1/4. Upstream has since raised the setting to 60 s |
| `modin-project__modin-7434` | hangs with zero progress: collects 5 items, prints `test_simple_import`, emits nothing further. Three independent runs, with and without CPU pinning, at concurrency 1 and 32. Under its real 3000 s budget it burns the full 52.9 min and raises `VerifierTimeoutError` — no partial result to parse, so nothing to grade |


---

## `terminal_bench_2_1` — 89 tasks, 88 green

```yaml
    exclude_task_ids: [caffe-cifar-10]
```

| Task | What happens |
|---|---|
| `caffe-cifar-10` | **operational, not a broken oracle**: its CIFAR-10 dataset host is very slow, so the oracle busts its wall clock on the *download*. Drop the exclusion once the dataset is pre-seeded into the image |

---

## `deep-swe` — 113 tasks, 113 green

Nothing to exclude: the full sweep passed **113/113** at native timeout budget, every task
`reward=1.0`. Needs the `beagle[deep-swe]` extra (pier) and a filtered-egress trial.

---

## `swe-bench-verified` — 500 tasks

No task exclusions: the corpus is human-validated upstream, so every instance is solvable in
principle.

Two behaviours to expect instead of exclusions:

- **Empty patch → `NoAttempt`.** An agent that produces no diff scores 0 but is flagged as a
  retryable error rather than a capability failure, so `--retry-errors` picks it up.
- **A tail of unscored instances.** The upstream evaluator occasionally leaves an instance without a
  `report.json` (transient image pulls, timeouts, a few deterministic harness failures). Those score
  0 with a logged reason — never a silent zero. Check the eval log before reading a low score as an
  agent result.

---

## `webarena-infinity`

**Not currently exercised — no measured advice.** It is registered (so `data[].benchmark:
webarena-infinity` resolves) and runs through WAI's own vendored orchestrator, which needs the
`vendor/benchmarks/webarena-infinity` submodule. It is deliberately absent from the supported table
in the README: nobody has run an oracle sweep or a smoke against it here, so this page has nothing
measured to tell you. Treat any exclusion list you build for it as your own, not ours.

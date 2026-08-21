# frontier-swe — curated task-content patches

Each `patches/<task_id>/<relative_path>` is a **full-file overlay** applied on top
of the faithfully-populated cache by `build_cache.py --stage patch` (idempotent,
survives re-populate). Overlays touch only benchmark content; xrlenv core is never
changed. Each is logged here + in `STATUS.md`.

There are **two kinds** of overlay, kept strictly distinct:

1. **Oracle fix** — repairs a *broken upstream oracle* by re-pathing / completing the
   **upstream reference**, the smallest change that lifts its reward ceiling to passing
   ("complete the partial, don't re-author the task"). The reference itself is
   unchanged. → `dependent-type-checker`.
2. **xrlenv-authored solution** — for a task where upstream **withholds** the reference
   entirely (ships no `solution/solve.sh`), an **xrlenv-authored** best-effort solution
   used to prove the task is solvable end-to-end (plumbing + a reachable positive-reward
   ceiling). It is **NOT** an upstream oracle and is loudly labelled as such wherever it
   lives. → `notebook-compression`.

STATUS.md's green breakdown never conflates the two (5 upstream oracles vs 1 authored).

## `dependent-type-checker/` — fix the oracle's reference-solution path (2026-08-06)

**Defect (surfaced by the G1 oracle sweep):** the task's own `solution/solve.sh`
read the reference type-checker from `/tests/reference_impl` **during the solve
phase**, but harbor mounts `/tests` only during the **verify** phase — a platform
contract upstream itself documents in its sibling oracles (`cranelift/solve.sh`:
"Cannot access /tests/ — bundle any needed resources under solution/";
`libexpat/solve.sh`: "/tests/ is only mounted during verification"). So
`cd /solution/../tests` failed, `solve.sh` aborted under `set -e`, no checker was
built, and the verifier's empty checker rejected all 174 valid programs
(`accept 0/174` → reward 0). It was a broken oracle, not a hard task.

**Fix (2 files, follows upstream's own documented pattern):**

| Overlay | What it does |
|---|---|
| `solution/reference_impl/{Cargo.toml,src/main.rs}` | the SAME reference implementation, **byte-identical** (sha256-verified) to the copy inside `tests/tests-bundle.tar.gz` that the verifier itself builds and compares against — bundled under `solution/` so it is present during the solve phase |
| `solution/solve.sh` | copy the reference from the bundled sibling `solution/reference_impl/` instead of `/tests/reference_impl` (the only change: WHERE it reads; the reference source is unchanged) |

**Why this is faithful:**
- The reference source is unchanged and byte-identical — the verifier builds its own
  copy from `tests/tests-bundle.tar.gz` and compares, so the oracle building the same
  source yields matching outputs (correctness 1.0).
- `solution/` is uploaded **only for the oracle**, never for a real agent run, so
  bundling the reference there leaks nothing to a policy.
- The verifier's anti-cheat checks are oracle-aware: the byte-identical-copy hash
  check is skipped when `HARBOR_ORACLE_MODE=1` (which the sweep injects), and the
  source scan runs on `/app/type-checker` (the built checker source is clean).

**Validation:** offline (overlay applies; `solve.sh` bash-valid; bundled `main.rs`
sha256 matches the bundle) **and confirmed on-cluster** (2026-08-06,
`--tasks dependent-type-checker --max-workers 1`): the oracle builds and scores
**reward 1.0005** (correctness full). It is in the green set.

## `notebook-compression/` — xrlenv-AUTHORED solution (2026-08-06)

**Not an upstream oracle.** FrontierSWE withholds the reference for this task (ships
no `solution/solve.sh`), so an oracle cannot be *derived* — `tests/` has only a hidden
holdout + scorer, no reference solution to bundle (see STATUS.md's derivability rule).
This overlay is an **xrlenv-authored best-effort solution**, added to prove the task is
solvable end-to-end (plumbing works + a reachable positive-reward ceiling).

| Overlay | What it does |
|---|---|
| `solution/run.py` | the task's required `/app/run {fit,compress,decompress}` submission — a lossless per-file compressor built on the Python stdlib `lzma` (xz preset 9\|EXTREME). No network, no third-party deps. |
| `solution/solve.sh` | installs `run.py` as `/app/run` (`install -m 0755`); loudly banners that it is xrlenv-authored, not the upstream oracle. |

**Why this is legitimate + labelled:** the task ships no reference, so *any* solution is
necessarily authored; this one earns reward on merit (a genuine lossless compressor),
and it is marked xrlenv-authored in the file banners, here, the wrapper, and STATUS.md
so it is never mistaken for an upstream oracle. Correctness (byte-for-byte round-trip)
is the hard gate, so it deliberately favours a simple provably-lossless codec.

**Validation:** offline (3-stage round-trip byte-identical on synthetic notebooks;
covered by `tests/test_authored_notebook_compression.py`) **and confirmed on-cluster**
(2026-08-06, `--tasks notebook-compression --max-workers 1`): **reward 0.3175**,
`round_trip OK (80 files)`, 122 MB → 66 MB, artifact 14 B. In the green set as the
1 authored solution (the other 5 green are upstream oracles).

**Follow-up (optional):** a dictionary-trained codec (zstd `--train` if present, else a
zlib `zdict` built in `fit`) would lower the ratio further; the current codec favours
robustness over ratio.

# frontier-swe — STATUS

_Last updated: 2026-08-07_

**Gate: GREEN 7/7.** 17 tasks total → 12 gateable (ship `solution/solve.sh` after the
patch stage) → **7 green, 5 excluded/not-gateable**. Pass = the downloaded
`reward.json`'s `reward` (fallback `score`) `> 0` (grade-from-artifact; see
[Key mechanics](#key-mechanics)). Rewards are graded *quality* scores, not binary —
performance tasks ≈ a speedup ratio (uncapped, can exceed 1), implementation tasks a
test-pass fraction, so sub-1 is normal.

Run: `bash xrlenv_plugins/benchmarks/frontier_swe/run_full_sweep.sh` (conc 8, native
timeouts). Last G1 sweep `frontier-swe-full-sweep-2026-08-05_22-34-34` + per-task
confirmations 2026-08-06/07.

## 1. Passed as-is — upstream oracle (5)

| task | reward | what the number is |
|---|---|---|
| `ffmpeg-swscale-rewrite` | 0.9939 | correctness 30/30; reward = geo-mean speedup |
| `git-to-zig` | 0.7307 | weighted test-pass fraction (the reference is partial) |
| `libexpat-to-x86asm` | 1.0002 | correctness full; uncapped speedup ratio (>1) |
| `revideo-perf-opt` | 0.9742 | correctness 8/8; reward = geo-mean speedup |
| `dart-style-haskell` | 1.0 | implementation; oracle wraps the bundled 340 MB Dart SDK. Needed the xrlenv **chunked `put_archive`** fix to upload (see [§ 4](#4-resolved)); confirmed on-cluster 2026-08-07 |

## 2. Passed with a fix (2)

| task | reward | fix kind (detail in [§ Fixes](#fixes)) |
|---|---|---|
| `dependent-type-checker` | 1.0005 | **oracle reference-path fix** — re-path the *upstream* reference |
| `notebook-compression` | 0.3175 | **xrlenv-authored solution** — upstream reference withheld |

## 3. Excluded / not gateable (10)

| task | class | reason |
|---|---|---|
| `granite-mamba2-inference-optimization` | GPU | `gpus=1`; dev cluster is CPU-only (not broken — revisit with GPU nodes) |
| `inference-system-optimization` | GPU | ″ |
| `optimizer-design` | GPU | ″ |
| `pcqm4mv2-autoresearch` | GPU | ″ |
| `cranelift-codegen-opt` | content defect | upstream oracle is a 7-line placeholder (`echo "…implement me"`); **no reference exists** to derive or complete |
| `lua-native-compiler` | withheld | no reference + not statically authorable (needs a real Lua→x86-64 native compiler) |
| `postgres-sqlite-wire-adapter` | withheld | ″ (needs full PostgreSQL-18 regression compatibility) |
| `pyright-type-checking-optimization` | withheld | ″ (needs a real perf win at output parity) |
| `modular-stack-wan21` | withheld + GPU | ″ + `gpus=1` |
| `frogsgame-rl` | withheld + infra | ″ + external Tinker API key + non-hermetic (stochastic API-graded) verifier |

*Withheld = upstream ships no `solution/solve.sh` (live-leaderboard anti-leakage) and
`tests/` holds only a spec/baseline, so an oracle can't be derived; those 5 are also
the multi-hour frontier problems a static `solve.sh` can't solve.*

## 4. Resolved

- **Chunked `container_put_archive`** (xrlenv core, commit 971602a) — `put_archive` was a
  unary RPC capped at the 128 MiB gRPC message limit on both hops, so an oracle bundling a
  340 MB SDK (`dart-style-haskell` → a 639 MB upload) failed deterministically; the
  `RESOURCE_EXHAUSTED` was also mislabelled `CapacityExhausted`. Now chunked (the mirror of
  the shipped `get_archive` streaming) so each frame stays ~4 MiB (heartbeat-safe), with a
  `NodeHello` capability + unary fallback for old peers. **Confirmed on-cluster
  2026-08-07**: `dart-style-haskell` reward 1.0 → it is now a normal upstream-oracle green
  (§ 1). General fix — unblocks any future >128 MiB upload.

## Still open (not blocking the gate)

| item | effect | status |
|---|---|---|
| Re-include the 4 GPU tasks | +4 gateable | when GPU nodes exist |
| `cranelift` real reference | +1 gateable | needs upstream to ship a solution |
| Wire into the CI integration runner | continuous gate | green set now stable at 7 |

## Fixes

### `dependent-type-checker` — oracle reference-path fix (confirmed reward 1.0005)

The oracle's `solve.sh` read the reference type-checker from `/tests/reference_impl`
**during the solve phase**, but harbor mounts `/tests` only during **verify** (a
contract upstream states in its own `cranelift`/`libexpat` oracles), so the build
no-op'd and the empty checker rejected all 174 valid programs. Overlay
(`patches/dependent-type-checker/`): bundle the **byte-identical** `reference_impl`
(sha256-matched to the copy the verifier itself builds) under `solution/` and point
`solve.sh` there — upstream's own documented pattern; the reference is unchanged.

### `notebook-compression` — xrlenv-authored solution (confirmed reward 0.3175)

Upstream withholds the reference (no `solution/solve.sh`) and `tests/` ships only a
hidden holdout + scorer, so an oracle can't be *derived*. Overlay
(`patches/notebook-compression/`) supplies an **xrlenv-authored** `/app/run` — a
lossless per-file `lzma` compressor (stdlib only, no network) — **clearly labelled as
authored, not an upstream oracle**. Confirmed on-cluster: round-trip lossless on 80
hidden notebooks (122 MB → 66 MB, 14 B artifact).

## Key mechanics

- **grade-from-artifact** — harbor 0.20's strict `VerifierResult` rejects FrontierSWE's
  rich `reward.json` (`subscores` list + `additional_data` dict), so `verifier_result`
  is `None` on every task; the verifier dir is downloaded *before* that parse, so the
  sweep grades from the on-disk `reward.json` (harbor's `ValidationError` is expected +
  ignored when a gradeable `reward.json` is present; only a **missing** one is a real
  infra failure).
- **oracle mode** — the sweep injects `HARBOR_ORACLE_MODE=1` (`environment.env` +
  `verifier.env`) to relax the verifiers' anti-cheat scan; never baked into `task.toml`.
- **retries** — `--retries` (infra-transient only) + `--content-retries` (per-task
  reward-0 re-run); both in `run_oracle_sweep.py`.

See `README.md` (how-to) and `patches/README.md` (the two overlay kinds, kept distinct).

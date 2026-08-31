# EvoClaw milestone status — single source of truth

What our oracle + xrlenv onboarding can achieve on the **98 graded milestones** (paper Table 3),
as of 2026-07-07. Evidence: a full 98-milestone sweep with our fixes on
(`tableA-FULL98-fixes`), 2 unfixed baseline sweeps (`upstream-match-full`,
`evoclaw-regression-full`), and repeated Table A OFF/ON runs. Detail lives in three linked docs:
[upstream-fixes](upstream-fixes.md) · [Table A / contention](table-a-resource-contention.md) ·
[Table B / unfixable](table-b-failure-verdict.md).

## Headline — the three expectations

Set your expectation per milestone by which tier it's in (with the two opt-in fixes on):

| Tier | Count | What to expect |
|---|---|---|
| ✅ **Always pass** | **92** | resolves every run (reliably), given `--apply-yd-fixes` + cpuset pinning for the contention set |
| 🎲 **Could pass, yet flaky** | **2** | resolves *most* runs but has a residual `none_to_pass` DB-race flake — **navidrome sub-01 / sub-03**, retried away to ~1-in-125 by the n2p eval-retry |
| ❌ **Fail under the 2 opt-in fixes** | **4** | 3 are content cannot-solve — dubbo M001.1/M001.2, nushell G01 (image ≠ classification, no faithful fix). The 4th, element **maintenance_ui_ux**, is **DinD-gated, not content**: it needs a 3rd flag `--sysbox-milestone` to route it to the cluster's `sysbox-runc` pool — recoverable, just not run in these sweeps (see below) |
| **Total graded** | **98** | **92 + 2 + 4 = 98, fully accounted** — exactly the paper's 98 (see [counting](#how-the-98-is-counted)) |

A **good full run** lands at **~94/98**: 92 always-pass + both navidrome flaky happening to pass. A
**bad run** for the flaky tier dips to 92. The 4 always-fail never move. (go-zero **M014** was the 3rd
flaky one until fix #5 — dropping its bystander rate-limiter timing tests — moved it into always-pass;
[validated](#validated-2026-07-07): M014 resolves unpinned, `pass_to_pass_required` 2132 → 2128.)

The opt-in fixes get us from **~76/98 unfixed** to **92 always-pass + up to 2 flaky = ~94/98 on a
good run**. Both default **OFF** (a run
without them is byte-faithful to upstream / leaderboard-comparable):
- **`--apply-yd-fixes`** — parser + eval-protocol corrections ([upstream-fixes](upstream-fixes.md)).
- **`--cpu-pinning-milestone [REPO/]MID`** — cpuset pinning for the contention set ([Table A](table-a-resource-contention.md)).

### Live full-98 oracle run (2026-07-07, `oracle-full`) — 94/98

A real full sweep landed at **89/98**, then **94/98** after the node-bug fix + a pinned backfill
(0 no-verdict). Findings this run surfaced that the earlier focused runs did not:
- **The Table A pin set was incomplete.** go-zero **M009**, **M014** and element **feature_enhancements**
  are contention-sensitive too and were *not* in the original pin list. Pinning resolves M009 and
  feature_enhancements → they sit in **always-pass** with pinning on.
- **go-zero M014 was a `pass_to_pass` timing flake** (`core/limit/TestTokenLimit_Take` rate-limiter),
  6/6 pinned-in-isolation but flaked once under the full-98 load. **Now fixed by #5** — dropping the
  bystander `core/limit` rate-limiter timing tests from grading via EvoClaw's own
  `stable_classification` ([upstream-fixes §6](upstream-fixes.md)); validated resolves unpinned. So
  M014 is now **always-pass**.
- **The infra fixes proved out end-to-end.** The 2 milestones lost to the node **`not a valid stream`**
  demux bug (ripgrep `5f5da48_sub-02`, nushell `G02`) both **resolved** on the redeployed cluster
  ([upstream-fixes](upstream-fixes.md) — xrlenv `raw_container.exec` resync-retry + the `docker_shim`
  watcher belt).

## ❌ Unresolved in these sweeps (4) — [Table B](table-b-failure-verdict.md)

3 are genuine content cannot-solves (the published **image disagrees with the published
classification** — no downstream fix without diverging from ground truth). The 4th
(**maintenance_ui_ux**) is **not** content — it's DinD-gated and recoverable with sysbox
(the cluster has a working pool; it just wasn't routed there in these runs).

| Milestone | Why |
|---|---|
| dubbo **M001.1** | content — `RestProtocolTest` fails in the image itself (`Tests run:103, Failures:2`) — image ≠ classification |
| dubbo **M001.2** | content — same, plus 32 `none_to_pass` failures |
| nushell **G01** (`milestone_G01_48bca0a`) | content — `loop_try_*` panic in the image (rustc 1.86.0) — image ≠ classification |
| element **maintenance_ui_ux** | **DinD, recoverable** — Playwright `testcontainers` E2E needs a Docker-in-Docker runtime. The cluster runs a working `sysbox-runc` pool (seta/TW DinD tasks use it — a patched Sysbox build), so `--sysbox-milestone <element-web-repo>/maintenance_ui_ux` should run it (unprivileged inner `dockerd`). Not exercised in the 2026-07-07 sweep (predates the cluster's sysbox), so **unvalidated for evoclaw** — a targeted run would confirm ~93–95/98. |

<a name="validated-2026-07-07"></a>
## 🎲 Could pass, yet flaky (2) — timing-sensitive tests — [Table A](table-a-resource-contention.md)

Genuine timing/contention flakes. Pinning takes them from *always-fails* to *mostly-passes*; the
n2p eval-retry then retries the residual (navidrome's own test/DB concurrency) away.

| Milestone | Flake | Unpinned | With `--cpu-pinning` | Mitigation |
|---|---|---|---|---|
| navidrome **sub-01** | `n2p` DB race | 0/4 (nf=5 every run) | ~4/6 (~1-in-3 residual) | n2p eval-retry → reliable (~1-in-125 miss) |
| navidrome **sub-03** | `n2p` DB race | 0/4 (nf=10) | ~5/6 (~1-in-6 residual) | n2p eval-retry → reliable (~1-in-125 miss) |

**go-zero M014 was here, now in *always-pass*.** Its flake was a **bystander** rate-limiter timing
test (`core/limit/TestTokenLimit_Take` & siblings — a wall-clock-window assertion, graded as p2p in
every whole-suite go-zero milestone). **Fix #5** ([upstream-fixes §6](upstream-fixes.md)) drops those
`core/limit` timing tests from the graded `pass_to_pass` via EvoClaw's own `stable_classification`
mechanism. **Validated 2026-07-07**: M014 resolves **unpinned** with `--apply-yd-fixes`, and
`pass_to_pass_required` drops exactly 2132 → 2128 (the 4 timing tests). It's not a p2p-retry (which
would mask real regressions) — it's a faithful, scoped exclusion of genuinely-flaky timing tests.

**Decision taken (was open):** the *serialize-the-persistence-specs* option is a dead end — the
ginkgo command has no `-p` so it is already process-serial, and the only remaining lever
(`GOMAXPROCS=1`) would change timing for every navidrome milestone and risk regressing green ones.
We took the **`none_to_pass` eval-retry** instead: under `--apply-yd-fixes`, when the *sole* reason a
milestone is unresolved is a small n2p failure set, the evaluator re-runs (fresh container) up to 2
extra times and takes the first resolve — mechanism-agnostic, and it never retries an f2p/p2p
regression or a large n2p set, so it de-flakes without masking real failures
([upstream-fixes §5](upstream-fixes.md)). **Validated end-to-end** (2026-07-06): 3 fresh
sub-01+sub-03 iters all resolved **2/2 (6/6)**, and the per-milestone `yd_n2p_retry.log` audit
files caught two real flake-recoveries (`n2p_fail=5 → eval-retry → RESOLVED on attempt 1/attempt
2`) — the attempt-2 case is exactly what the old give-up-early code missed (it failed 1/2 that
same run). See [Table A](table-a-resource-contention.md).

## ✅ Always pass (92)

- **~82** resolve out of the box (no fix needed) — stable across all runs.
- **fixes made reliable** (proven, incl. under full-98 load):
  - go-zero **M001, M003, M004, M005, M027** — `go test -json` benchmark line-split → `--apply-yd-fixes` ([upstream-fixes §3](upstream-fixes.md))
  - go-zero **M019** (rate-limiter) · dubbo **M003.1** (thread-pool) — timing tests starved under oversubscription → `--cpu-pinning` ([Table A](table-a-resource-contention.md))
  - go-zero **M009** · element **feature_enhancements** — contention `p2p` flakes surfaced by the 2026-07-07 full run → `--cpu-pinning` (the *expanded* pin set; resolve reliably pinned)
  - element **e662c19**, **fba5938** — untracked GT test files deleted by the evaluator's `git clean` → `--apply-yd-fixes` ([upstream-fixes §2](upstream-fixes.md))
  - go-zero **M014** — dropped its bystander `core/limit` rate-limiter timing tests from grading → `--apply-yd-fixes` (fix #5, [upstream-fixes §6](upstream-fixes.md)); validated resolves unpinned
- Also enabling many go-zero milestones at all: the **base-image `.git` fix** ([upstream-fixes §1](upstream-fixes.md)).

## Proof it holds under load

The doubt "focused-10 runs are just easier" is answered by the **full-98 run with fixes**: all 10
Table A milestones resolved under the *exact* full-98 load that originally flipped M019/M003.1/
sub-01/sub-03 (whole run 93/97 distinct resolved). See [Table A](table-a-resource-contention.md)
§ "Full-98 proof".

<a name="how-the-98-is-counted"></a>
## How the 98 is counted

The **98 is exact and matches the paper** — no "parent vs sub-milestone" expansion. Straight from
the dataset (`selected_milestone_ids.txt` − `non-graded_milestone_ids.txt`, summed over the 7 repos):

| | count | |
|---|---|---|
| selected (graded + context) | 101 | |
| − non-graded **context** milestones | 3 | ripgrep 2, dubbo 1 (excluded from scoring) |
| **= graded milestones** | **98** | go-zero 23 · element-web 18 · nushell 13 · dubbo 12 · scikit 12 · ripgrep 11 · navidrome 9 |

This lines up with the paper's other figures, all dataset-confirmed: **7 repositories**, **5
languages** (Go · Rust · Java · TypeScript · Python), **109 inter-milestone dependency edges**
(`dependencies.csv` rows: 28+25+14+12+12+9+9). The **109 is dependencies (DAG edges), not
milestones** — don't conflate the two.

**Sub-milestones are already part of the 98, not an expansion of it.** 14 graded IDs carry a
`_sub-NN` suffix, spread over 6 parents — e.g. navidrome `milestone_003` is **four** graded
milestones (`sub-01`…`sub-04`), each counted once. So a milestone-level sweep launches **98 tasks**.
(A run may emit *more* than 98 result files — retries and multi-state evals write extra — and
occasionally *fewer* distinct verdicts, e.g. 97, when one milestone isn't given what it needs — e.g.
the DinD `maintenance_ui_ux` when it isn't routed to the sysbox pool via `--sysbox-milestone`. The
graded denominator is always 98.)

- The per-bucket *membership* is **exact** (the 4 unfixable, the 2 flaky, the 11 we fixed); "83 out
  of the box" is approximate because the unfixed baselines predate the scikit repo.
- Numbers are **with the opt-in fixes on**; a default (faithful) run resolves ~76/98. Reproduce both
  runs with the commands in [README.md §3 "Run the oracle sweep"](README.md).
- **scikit M04** is solvable (isolated re-run RESOLVED, f2p 1/1); its full-98 abort was a
  **quarantine flake** — the fail-closed check couldn't confirm PyPI (`files.pythonhosted.org`)
  was blocked, because PyPI rides Fastly and the deny-CIDR only covers setup-time IPs.
  **Fixed** under `--apply-yd-fixes` (`_patch_quarantine_poison_deny_domains`, [upstream-fixes §4](upstream-fixes.md)):
  denied hosts are added to the `/etc/hosts` → `0.0.0.0` poison, a DNS-level block immune to CDN
  IP drift → no more random aborts. Report upstream to make it default.

## Reproduce

The batch driver is `run_all_xrlenv.py` (README.md §3 "Run the oracle sweep"). A full 98-milestone
oracle sweep **with** the two opt-in fixes lands at ~94/98; both default **OFF**, so a run
without them is byte-faithful to upstream / leaderboard-comparable (~76/98):

- `--apply-yd-fixes` — parser + eval-protocol corrections ([upstream-fixes](upstream-fixes.md)).
- `--cpu-pinning-milestone [REPO/]MID` — cpuset pinning for the contention set
  ([Table A](table-a-resource-contention.md)).

Per-milestone artifacts land under `<run>/`; read `<run>/xrlenv_summary.json` for the
unresolved mids. Exact faithful vs. fixed invocations: README.md §3 "Run the oracle sweep".

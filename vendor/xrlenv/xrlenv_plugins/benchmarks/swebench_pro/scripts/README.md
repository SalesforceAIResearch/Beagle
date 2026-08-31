# swebench-pro/scripts — the partitions of the corpus

The kit root runs the **full** corpus (731). This directory holds the two derived selections and
the one-task smoke test, each a manifest + an image plan + a pinned sweep wrapper, plus the
generators that produce those files. Everything here goes through the same pipeline as the full
sweep (`../build_cache.py` → `../run_full_sweep.sh` → `../run_oracle_sweep.py`); the wrappers only
pin the selection and forward every flag.

| | **full** (root) | **filtered** | **subset-100** |
|---|---|---|---|
| what | every public instance — the corpus as published | the quality-filtered set: full minus the tasks our filter calls broken | a 100-task sample of the filtered set, balanced over the 11 repos |
| tasks | 731 | 478 | 100 |
| `build_cache.py` flag | `--all` | `--filtered` | `--subset-100` |
| manifest | (the whole dataset) | `filtered_instance_ids.txt` (+ `filter_report.json`) | `subset_100_instance_ids.txt` (+ `subset_100.json`) |
| image plan (compressed) | `../build_plan_full.yaml` — 731 images, ~1071 GB | `build_plan_filtered.yaml` — 478 images, ~693 GB | `build_plan_subset_100.yaml` — 100 images, ~144 GB |
| gold-patch sweep | `../run_full_sweep.sh` | `run_filtered_sweep.sh` | `run_100_subset_sweep.sh` |
| default job id (+ timestamp) | `swebench-pro-full-sweep-…` | `swebench-pro-filtered-sweep-…` | `swebench-pro-subset100-sweep-…` |
| use it for | numbers comparable with published full-set results; the reference corpus | the evaluation corpus — no tasks whose reward ceiling is 0 or whose tests over-constrain the fix | smoke gates, agent/critic iteration, warm-up validation — every repo's image family, language stack, run script and parser at 1/5 of the filtered cost |

The three are nested (subset-100 ⊂ filtered ⊂ full), so warming a bigger plan covers the smaller
ones, and a task that passes the full oracle passes it in every partition.

## How they differ

- **filtered** drops 253 of the 731. Following OpenAI's July 2026 audit of SWE-bench Pro
  ("Separating signal from noise") we ran our own three-stage filter (theirs is not
  reproducible — they published counts, not ids): (1) a language-aware static screen for the
  four defect classes — overly strict tests, underspecified prompt, low-coverage tests,
  misleading prompt — calibrated to flag ~40 % (301/731; OpenAI: 39.1 %); (2) gpt-5.6 judges
  every flagged task 5×; (3) a task is dropped when ≥3 of 5 judges call it broken. Kept
  478 / dropped 253 (OpenAI's human campaign kept 482); kept by language: Python 184, Go 184,
  JS 103, TS 7; dropped: Go 96, Python 82, JS 62, TS 13; primary drop reasons: low-coverage tests
  169, overly strict tests 51, misleading prompt 25, underspecified prompt 8. The pipeline itself
  is not part of this kit — its output is the committed manifest plus `filter_report.json`
  (categories + votes per instance). Caveat: three of the four defect classes cause false
  negatives, so filtering **raises** measured pass rates — compare against your own baseline on
  the same configuration, never against full-set numbers.
- **subset-100** is 100 of the 478, drawn by `sample_subset.py --total 100 --policy random
  --seed 0`: each repo gets a share proportional to its kept count (every repo ≥ 1) — NodeBB 6,
  ansible 13, element-web 7, flipt 9, vuls 10, teleport 12, openlibrary 12, navidrome 8,
  webclients 8, qutebrowser 13, tutanota 2 (144.0 GB of images). Same task shape and grading as
  the other two, so a result on it transfers to the filtered set with sampling noise only.
- **smoke** (`run_smoke_one.sh`) is not a partition: it runs the oracle on ONE task — by default
  the first id of subset-100, a member of all three selections — as the quickest plumbing check
  (one image pull, one container, one verifier run; a couple of minutes). `../run_full_sweep.sh
  --smoke` is the other quick check: the first 8 rows of the dataset.

## How to run

Inputs and `.env` are the same as the root README. Start with the smoke test, then a partition:

```bash
# one task (first id of subset-100; --instance <id> / --index N for another)
bash xrlenv_plugins/benchmarks/swebench_pro/scripts/run_smoke_one.sh
bash xrlenv_plugins/benchmarks/swebench_pro/scripts/run_smoke_one.sh --instance <id>

# subset-100
.venv/bin/python xrlenv_plugins/benchmarks/swebench_pro/build_cache.py --subset-100
bash xrlenv_plugins/benchmarks/swebench_pro/scripts/run_100_subset_sweep.sh --max-workers 16 --content-retries 1

# filtered (478)
.venv/bin/python xrlenv_plugins/benchmarks/swebench_pro/build_cache.py --filtered
bash xrlenv_plugins/benchmarks/swebench_pro/scripts/run_filtered_sweep.sh --max-workers 16 --content-retries 1

# OPTIONAL pre-warm of a partition's plan (operator; see the root README for why --connect-host)
xrlenv build apply --plan xrlenv_plugins/benchmarks/swebench_pro/scripts/build_plan_subset_100.yaml \
    --connect-host <admin-host> --connect-port 8080
```

The wrappers rebuild the cache for their selection themselves (`--skip-build-cache` to skip),
accept everything `../run_full_sweep.sh` accepts (`--list-green`, `--max-workers`,
`--content-retries`, `--job-id`, `--jobs-dir`, and any `run_oracle_sweep.py` knob), and share its
pass gate: exit 0 iff every task rewards > 0; an oracle FAIL is a corpus/plumbing defect, never a
model signal.

## What's here

| File | Role |
|---|---|
| `run_filtered_sweep.sh` / `run_100_subset_sweep.sh` | `../run_full_sweep.sh --filtered` / `--subset-100` (all flags forwarded) |
| `run_smoke_one.sh` | the oracle on ONE task (`--instance ID` / `--index N`; default: first id of subset-100) |
| `filtered_instance_ids.txt` / `filter_report.json` | the filtered configuration: kept ids (478) + drop categories/votes per instance |
| `subset_100_instance_ids.txt` / `subset_100.json` | the subset-100 configuration: ids (100) + per-pick repo/language/image/size |
| `build_plan_filtered.yaml` / `build_plan_subset_100.yaml` | the derived `type: registry` warm-up plans (478 / 100 images), sizes carried from the full plan |
| `sample_subset.py` | the repo-balanced sampler (defaults reproduce subset-100; `--total K`, `--per-repo N`, `--policy random/first/smallest-image`, `--dry-run`) |
| `build_plan_gen.py` | plan generator from the dataset (no cache needed): `--all` / `--filtered` / `--subset-100` / `--ids-file`; sizes probed from Docker Hub, or carried over from a prior plan with `--resume <plan> --no-probe` |

## Regenerating the manifests and plans

```bash
# subset-100 (defaults = the committed sample; change --total/--seed/--policy for another one)
.venv/bin/python -m xrlenv_plugins.benchmarks.swebench_pro.scripts.sample_subset
# the derived plans, sizes carried from the full plan (no Docker Hub probes)
.venv/bin/python -m xrlenv_plugins.benchmarks.swebench_pro.scripts.build_plan_gen --subset-100 \
    --resume xrlenv_plugins/benchmarks/swebench_pro/build_plan_full.yaml --no-probe \
    --output xrlenv_plugins/benchmarks/swebench_pro/scripts/build_plan_subset_100.yaml
.venv/bin/python -m xrlenv_plugins.benchmarks.swebench_pro.scripts.build_plan_gen --filtered \
    --resume xrlenv_plugins/benchmarks/swebench_pro/build_plan_full.yaml --no-probe \
    --output xrlenv_plugins/benchmarks/swebench_pro/scripts/build_plan_filtered.yaml
# the full plan itself re-probes Docker Hub (set DOCKERHUB_USER/TOKEN; --resume finishes a rate-limited run)
.venv/bin/python -m xrlenv_plugins.benchmarks.swebench_pro.scripts.build_plan_gen --all --max-workers 8 \
    --resume xrlenv_plugins/benchmarks/swebench_pro/build_plan_full.yaml \
    --output xrlenv_plugins/benchmarks/swebench_pro/build_plan_full.yaml
```

`../tests/test_configurations.py` pins the three configurations together (manifests ⊆ each
other, plan order = manifest order, sizes consistent, wrappers + READMEs naming them, the kit
layout, no absolute data paths anywhere in the kit).

## Results (what the partitions have verified so far)

- One task (2026-08-26, `instance_NodeBB__NodeBB-00c70ce7…-vnan`): **resolved** (f2p 681/681). The first attempt failed 677/681 because 4 dataset
  `FAIL_TO_PASS` names are mangled — the lenient-name rule in `grade.py` (root README, "The
  verifier") came out of this.
- 8-instance smoke (2026-08-26, `../run_full_sweep.sh --smoke --max-workers 4`): **8/8 resolved**
  (reward 1 on every trial; f2p and p2p fully passing). One trial was mis-reported FAIL by the
  first pass gate (it required every numeric field > 0 and that instance has `p2p_total 0`) — the
  gate now keys on `reward` only.
- One instance per repository (2026-08-26, 11 tasks = `sample_subset.py --per-repo 1 --policy
  smallest-image`, `--max-workers 4`): **11/11 resolved** — every repo's image family, run script
  and parser verified end to end (12.8 GB of images: NodeBB 0.76 GB, ansible 0.49, element-web
  1.02, flipt 0.55, vuls 0.82, teleport 2.07, openlibrary 0.69, navidrome 0.76,
  protonmail/webclients 3.94, qutebrowser 0.54, tutanota 1.18).
- **subset-100** (2026-08-27, `run_100_subset_sweep.sh --max-workers 100 --content-retries 1`):
  **99/100 resolved on the first pass**. The one FAIL
  (`instance_future-architect__vuls-bff6b755…`) was not the oracle: all 7 tests PASSED but the
  cached `tests/grade.py` predated the lenient-name rule (3 `FAIL_TO_PASS` names lack their closing
  `"`), and `build_cache.py` never rewrote a complete task dir — 481/731 cached dirs were stale.
  Fixed: `build_cache.py` now refreshes kit-rendered files in place when the renderer changed
  (`refresh_kit_files`); regrading that trial's verifier output with the current rule gives
  reward 1. The full sweep (root `STATUS.md`) re-runs it with the refreshed cache.
- **filtered (478)**: NOT YET RUN as its own sweep; it is a subset of the full sweep.

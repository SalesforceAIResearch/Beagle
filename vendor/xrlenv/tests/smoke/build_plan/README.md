# Build-plan dispatch smokes

← back to [smoke runbook index](../README.md)

These pin `xrlenv build apply --plan` end-to-end: a plan
distributes correctly across the cluster, re-applies behave as
documented (including the `--skip-if-present` fast path), size
hints stay calibrated, and source-build dispatch covers both
git and tarball entries. Cancel orchestration + build-on-acquire
recovery are pinned by adjacent files in this group.

| Smoke | What it pins |
|---|---|
| [`test_dispatch_tb2.py`](#test_dispatch_tb2py) | Five tests against the canonical phase-0 tb2 plan: fresh apply, idempotent re-apply, `--force` re-apply, calibration with side-artifact, operator-defined fresh plan. |
| [`test_dispatch_seta_env.py`](#test_dispatch_seta_envpy) | Three tests against the seta-env git-source plan: schema round-trip, dry-run reaches placement, programmatic callers that skip `resolve_tarball_sources` still get a clear apply-time rejection. |
| [`test_tarball_dispatch.py`](#test_tarball_dispatchpy) | Sub-slice 1.b tarball dispatch: operator-side cap rejects oversized contexts before the wire; happy-path build produces a labelled image; the per-image_ref source-spec registry persists on disk for build-on-acquire. |
| [`test_pin_budget_and_calibrate.py`](#test_pin_budget_and_calibratepy) | Phase B: pin-budget over-commit rejects at apply time (both modes); `xrlenv build calibrate` writes a YAML with cluster-reported sizes (remote-only). |
| [`test_cancel_regression.py`](#test_cancel_regressionpy) | `xrlenv build cancel` operator surface: local-only mode flips state.db and warns about cluster builds; cluster-mode round-trips through the admin endpoint and ends with the plan `cancelled`. |
| [`test_build_on_acquire_after_eviction.py`](#test_build_on_acquire_after_evictionpy--plan-4) | Sub-slice 2 build-on-acquire: after the operator manually `docker rmi`-s a source-built image on its preferred-home node, the next SDK `acquire_container` rebuilds it via the persistent source-spec registry. Operator-driven (eviction step needs SSH); the smoke drives the acquire side + reports timing. |

See [Conventions shared across smokes](../README.md#conventions-shared-across-smokes)
for invocation patterns, the three-mode structure, artifact
output, and cleanup recipes that apply across all groups.

---

## `test_dispatch_tb2.py`

**Group**: Build-plan dispatch. **Wall-clock**: ~30-90s when
images aren't cached on nodes; <5s when they are. **Modes**:
pytest, script. Each test parametrized on `(local, remote)`.

**What it validates.** End-to-end build-plan dispatch onto your
cluster: `xrlenv build apply --plan <yaml>` distributes the plan's
entries across connected nodes, each node pulls (or rebuilds) the
images it's been assigned, the cluster's persistent state — admin
`/builds` panel + `state.db` — reflects the result, and
re-applying the same plan behaves correctly.

The plan used is the committed phase-0 set at
`xrlenv_plugins/images_build/terminal_bench_2/build_plan.yaml`:
8 terminal-bench-2 task images from `alexgshaw/<task>:20251031` on
Docker Hub.

The five tests cover different operator semantics:

| Test | What it pins |
|---|---|
| `test_apply_canonical_plan` | A fresh apply distributes 8 entries across the cluster, every image reaches `done`. |
| `test_idempotent_reapply` | Applying the same plan twice short-circuits the second apply — no re-pulls, no extra state writes. |
| `test_force_reapply` | `--force` re-dispatches every entry even if the plan is already `completed`. Cache hits make it fast. |
| `test_calibration_hint_vs_actual` | The size hints in the YAML match what Docker reports on disk after a real pull (local mode only — needs `docker image inspect`). Writes a `build_plan.calibrated.yaml` side-artifact you can promote into the canonical YAML. |
| `test_fresh_eight_dispatch` | Generates a fresh plan from 8 task ids that aren't in the canonical set, applies it, verifies the cluster pulled the new images. Exercises the "operator builds a custom plan" path. |

### Prerequisites — local mode

- Docker daemon reachable from this host.
- ~3 GB free disk for the canonical 8 phase-0 images, plus another
  ~3 GB if the fresh-8 test runs.
- The 89-task harbor cache populated under
  `~/.cache/harbor/tasks/` (or `$XRLENV_BENCHMARK_CACHE`) for the
  fresh-8 test. Other tests don't need it.

### Prerequisites — remote mode

- A running `xrlenv up` reachable from this host. **Wait until at
  least one node shows `connected` in `xrlenv nodes`** before
  running — a freshly-started `xrlenv up` typically takes 5-15s
  before the first remote node reattaches via its systemd retry
  loop. Applying a plan against zero connected nodes fails fast
  with a clear message but wastes a run.
- The standard `$XRLENV_GRPC_HOST` env var set (the same one the
  SDK and docker-py drop-in use to find the control plane). The
  smoke reuses it to locate the admin endpoint at port 8080.
  Override with `$XRLENV_ADMIN_HOST` / `$XRLENV_ADMIN_PORT` only
  if the admin endpoint lives on a different host or port.
- Operator token via `$XRLENV_OPERATOR_TOKEN` or
  `~/.xrlenv/secrets/operator.token` (mode 0600). Issue with
  `xrlenv tokens issue operator`.

The remote tests only run when `$XRLENV_GRPC_HOST` (or
`$XRLENV_ADMIN_HOST`) is set; otherwise pytest emits one
`local`-only run per test.

### Invocation

```bash
# Local only (default):
.venv/bin/python tests/smoke/build_plan/test_dispatch_tb2.py

# Local + remote (reuses your existing $XRLENV_GRPC_HOST):
XRLENV_GRPC_HOST=internal-ip \
XRLENV_OPERATOR_TOKEN=$(cat ~/.xrlenv/secrets/operator.token) \
.venv/bin/python tests/smoke/build_plan/test_dispatch_tb2.py --mode all

# Remote only:
XRLENV_GRPC_HOST=internal-ip \
XRLENV_OPERATOR_TOKEN=... \
.venv/bin/python tests/smoke/build_plan/test_dispatch_tb2.py --mode remote

# Admin endpoint on a different host or port from gRPC:
XRLENV_ADMIN_HOST=admin.internal XRLENV_ADMIN_PORT=8443 \
XRLENV_OPERATOR_TOKEN=... \
.venv/bin/python tests/smoke/build_plan/test_dispatch_tb2.py --mode remote
```

After every run, artifacts land under
`<repo>/tmp/smoke-build-plan-tb2-<mode>-<utc-ts>/` (one dir per
mode per pytest invocation):

```
test_apply_canonical_plan.json     # outcome from test 1
test_idempotent_reapply.json       # both apply outcomes from test 2
test_force_reapply.json            # forced apply outcome from test 3
test_fresh_eight_dispatch.json     # task_ids picked + outcome from test 5
calibration.txt                    # local-only — pretty table
build_plan.calibrated.yaml         # local-only — calibrated side-artifact
```

---

### Test 1 — `test_apply_canonical_plan`

**What you should see end-to-end**

1. The smoke loads the committed canonical plan and submits it to
   the control plane (`xrlenv build apply --plan ...` from the
   CLI takes the same path).
2. The control plane records the plan, picks per-node placements
   via FFD bin-packing against current free-disk on each connected
   node, and dispatches an `ensure_present(image_ref)` call to
   each chosen node.
3. Each node pulls the image from Docker Hub if it's not already
   cached (~5-90s per image depending on size + bandwidth) or
   completes the call immediately if it is.
4. The plan transitions to `completed` once every assignment
   reaches `done`.

**Verifying in the admin panel**

Open `/builds`. You should see the canonical plan's row:

- `done / total`: **8 / 8**.
- `applied at`: the timestamp of this run (refreshes on every
  re-apply, so re-running always shows "just now").
- `by`: `smoke-remote` for remote-mode runs, `smoke-local` for
  local-mode runs.
- status: `completed`.

Click the row for the per-assignment breakdown. **Exactly 8 rows**,
all `done`, distributed across the cluster's connected nodes.
Each row's `started` and `completed` are seconds apart on
warm-cache runs and longer (proportional to image size) on
cold-cache runs.

**Verifying on each cluster node**

```bash
docker images --filter 'reference=alexgshaw/*' --format '{{.Repository}}:{{.Tag}}'
```

Across all nodes combined, you should see exactly the 8 unique
image refs in the canonical plan. By default each ref appears on
exactly one node (`preferred_home_count: 1`).

**Verifying in the smoke artifact**

`<repo>/tmp/smoke-build-plan-tb2-<mode>-<utc-ts>/test_apply_canonical_plan.json`:

- `status: completed`, `successes: 8`, `failures: 0`.
- `assignment_count: 8`, `per_status: {done: 8}`.
- 8 rows in `assignments`, each with one of the 8 image refs from
  the YAML.

**Why placement varies between runs (and why that's correct)**

The control plane's bin-packer places each entry on the first
node that has room, using each node's free disk **measured at
apply time**. Free disk drifts run-to-run as images, containers,
logs, and OS state come and go. So the same plan can land:

- All 8 entries on a single node when one node has plenty of
  headroom.
- Mixed across two nodes (e.g. 6 + 2 or 5 + 3) when one node has
  tighter headroom from prior activity.

This isn't randomness — given identical free-disk snapshots the
bin-packer is deterministic. It's the cluster reflecting current
capacity, gracefully spilling under disk pressure, and
rebalancing on the next apply. Whichever node receives an entry
pulls it; image-affinity scoring at acquire time later steers
rollouts toward whichever node holds each image, so a non-uniform
distribution doesn't disadvantage the rollout side.

If you need **deterministic placement** for reproducible benchmark
distribution across runs, three options:

- Set every entry's `placement.preferred_home_count` to your
  cluster's node count. Each image lands on every node; no
  spillover possible. Costs ~N× disk per image.
- Set `budget.cap_per_node_gb` in the plan. The bin-packer treats
  every node as having that exact capacity, removing free-disk
  drift as a variable.
- Set `pinned: true` per entry. Pinned entries skip eviction, so
  once placed they stay put across reboots and re-applies.
  (Pinning controls eviction, not initial placement — `cap_per_node_gb`
  is what actually pins where the bin-packer picks.)

For most workflows, the default "use what's actually available
now" behavior is the right shape.

**Re-running this test, and what `--force` actually costs**

The smoke always runs this test with `force=True`. That keeps the
test state-independent — you can re-run it back-to-back without
needing to know whether the cluster has applied this plan before.
On a re-run, here's what each side actually does:

| Run | What the cluster does | Wall-clock |
|---|---|---|
| First-ever apply (cold cache) | Bin-pack → dispatch `ensure_present` per (node, image) → real registry pulls for everything missing. | Dominated by pull bandwidth (~30-90s for 8 entries on the canonical plan). |
| Re-apply with `force=True` (warm cache) | Coordinator re-bin-packs, replaces assignment rows, re-issues `ensure_present` per (node, image). Each `ensure_present` is a **fast cache hit** because the image is already on the node — no registry pull. State rows churn; no Docker pull. | A few seconds for 8 entries; coordinator bookkeeping dominates. |
| Re-apply without `force` (already `completed`) | Coordinator short-circuits at the idempotency check; returns `no_op_already_completed` immediately. **No dispatch, no state writes, no node-side work**. | Sub-second. |

So "running the same test twice" doesn't pull images twice — it
just exercises the dispatch path twice. The only way a re-apply
triggers real registry traffic is if the cluster's image cache
actually evicted something between runs (e.g. disk pressure
reclaimed cold images).

**What "idempotent" means here**

Three ways to think about idempotency, in increasing strictness:

1. **Operator-level**: applying the same plan twice ends in the
   same observable state — `8 / 8 done`, every image present
   somewhere on the cluster, plan status `completed`. ✓ always.
2. **No-wasted-work**: re-applying without `--force` performs zero
   work (instant short-circuit). ✓ always — this is what test 2
   pins.
3. **Bit-level placement**: the same `(plan_id, node_id, image_ref)`
   triples appear after every apply. ✗ — the bin-packer's
   free-disk-driven placement can shift entries between nodes
   across runs (see "Why placement varies"). What the smoke
   asserts is the operator-level shape, not bit-level placement.

**A note on very large plans**

The coordinator doesn't currently optimize `force=True` re-applies
by detecting "desired placement already matches current state, so
skip dispatch." For an 8-entry plan that's negligible; for a
5,000-entry plan it'd add a few seconds of cache-hit RPCs per
re-apply. Worth knowing if you pin a much larger plan; not a
concern at the canonical phase-0 size.

---

### Test 2 — `test_idempotent_reapply`

**What you should see end-to-end**

This test pins the contract: **applying the same plan twice in a
row, where the first finishes successfully, makes the second a
true no-op**. The cluster does no node-side work, the state.db
isn't touched, and the admin panel's `applied_at` doesn't move
during the second apply.

The test runs two applies back-to-back:

1. **Baseline apply with `force=True`**. Behaves exactly like
   Test 1: re-bin-packs, re-records 8 assignment rows,
   dispatches `ensure_present` per (node, image), all 8 reach
   `done`. On a warm cache (which you almost always have when
   running tests in sequence), each `ensure_present` is a fast
   cache hit. `applied_at` advances to the start of this apply.
2. **Second apply with no `force`**. The control plane sees the
   plan is already `completed` and short-circuits the entire
   path — no `record_build_plan` call, no
   `delete_assignments`, no node-side dispatch. The response is
   immediately `no_op_already_completed`.

The force baseline is there for state-independence: the test
runs the same way whether the cluster has applied this plan
before or not. The actual contract under test is the second
apply's short-circuit.

**Verifying in the admin panel**

Open `/builds`. The canonical plan's row should:

- Still read **8 / 8 done**, status `completed`.
- Show `applied at` from the FIRST apply within this test (the
  force baseline), NOT the second apply. The two are seconds
  apart so you can't visually distinguish them; what you can
  observe is that the panel doesn't continue to bump after
  pytest reports PASSED.

Click the row. The 8 assignment rows' `started` / `completed`
timestamps reflect the first apply only — the second apply
didn't write any rows.

**Verifying in the smoke artifact**

```bash
LATEST=$(ls -td tmp/smoke-build-plan-tb2-remote-*/ | head -1)
cat "$LATEST/test_idempotent_reapply.json" | python3 -m json.tool | head -50
```

(`tmp/smoke-build-plan-tb2-remote-*` matches every prior run's
dir; the `ls -td | head -1` picks the latest. Cat'ing the bare
glob will fail to parse because multiple JSON documents get
concatenated.)

The artifact has two top-level keys:

- `first` — the force baseline outcome. `status: "completed"`,
  `successes: 8`, full assignment list.
- `second` — the short-circuit outcome.
  `status: "no_op_already_completed"`, `successes: 0`, no
  `assignments` payload (the path didn't even query state.db
  for the rows). The mere presence of the bare-shape `second`
  alongside the populated `first` is the visible signal that
  the no-op fast path fired.

**Verifying on each cluster node**

```bash
docker images --filter 'reference=alexgshaw/*' --format '{{.Repository}}:{{.Tag}}' | sort
```

Identical to before the test — no new images pulled, no images
removed. The 8 image refs from the canonical plan stay present.

**Wall-clock expectation**

~3-7 seconds total per run on a typical GCP topology. The first
apply (force baseline, warm cache) takes a few seconds for
coordinator bookkeeping + cache-hit RPCs to each node; the
second apply (short-circuit) is sub-second. Variance run-to-run
is dominated by network round-trip jitter to the cluster, not
coordinator behavior — at this scale a few seconds either way
is normal.

**Running this test multiple times**

`applied_at` advances each time you re-run the test (because each
test invocation does its own force baseline) but the panel still
shows `8 / 8 done` with no inflation. The contract being pinned
is robust across re-runs: every test invocation produces exactly
one assignment-list refresh from the force baseline + zero
writes from the no-force second apply.

---

### Test 3 — `test_force_reapply`

**What you should see end-to-end**

This test pins the contract: **`--force` re-dispatches the plan
even when it's already `completed`**. Compared to Test 2's no-op
short-circuit, force tells the coordinator "do the work anyway."
On a warm cache (which is the common case after Test 1 has run)
the work is mostly bookkeeping + cache-hit RPCs, not real pulls.

The test runs two applies back-to-back, **both with `force=True`**:

1. First apply: re-bin-pack, purge prior assignments, record 8
   fresh `pending` rows, dispatch `ensure_present` per node, all
   reach `done`. `applied_at` advances.
2. Second apply: same exact shape. Coordinator doesn't detect
   "the cluster's already in the desired state" — it just runs
   the dispatch path again. `applied_at` advances again.

`record_build_plan`'s upsert means each apply's `applied_at`
lands as the latest UPSERT time. The panel always shows the
most recent force apply.

**Verifying in the admin panel**

Open `/builds` for the canonical plan:

- **8 / 8 done**, `completed`, `by smoke-remote`.
- `applied at` reads **just now** — the second force apply
  bumped it. Contrast with Test 2 where the second apply was a
  no-op and didn't bump.

Click the row. The 8 assignment rows' `started` /
`completed` timestamps reflect the **second** apply only — the
first apply's rows got deleted by the purge before the second
apply re-recorded them. The placement (which node holds which
image) matches whatever the bin-packer chose for this most
recent apply, which can differ from Test 1's or Test 2's
placement (FFD non-determinism, see Test 1's section).

**Verifying in the smoke artifact**

```bash
LATEST=$(ls -td tmp/smoke-build-plan-tb2-remote-*/ | head -1)
cat "$LATEST/test_force_reapply.json" | python3 -m json.tool | head -30
```

Unlike Test 2's artifact (which captured both `first` and
`second`), this one writes **only the second apply's outcome**
as the top-level dict. Reason: both applies are forced, but the
test's contract is about the second one (the "force re-apply"
being demonstrated); the first is just a normalizer to keep the
test state-independent.

You should see:

- `plan_id: 9d546b30eaf6...`
- `status: completed`, `successes: 8`, `failures: 0`.
- `assignment_count: 8`, `assignments: [...]` — 8 rows, all
  `status: done`, all with `started_at` / `completed_at` from
  this run.

**Verifying on each cluster node**

```bash
docker images --filter 'reference=alexgshaw/*' --format '{{.Repository}}:{{.Tag}}' | sort
```

Identical before and after — two force applies on a warm cache
trigger zero registry traffic. The same 8 image refs stay
distributed across the cluster (placement may shift, total set
is unchanged).

**Wall-clock expectation**

~10-15 seconds total. That's roughly 2× the cost of Test 2,
which is the expected signature: Test 2 = one real dispatch +
one short-circuit; Test 3 = two real dispatches. At 8 entries
on a 2-node GCP topology, each force apply's a few seconds of
coordinator bookkeeping + cache-hit RPCs + network round-trips.

**When you'd actually use `--force` as an operator**

- After uninstalling and reinstalling the cluster: state.db is
  fresh but the cluster's docker daemon may still have cached
  images from before. `xrlenv build apply --plan ... --force`
  re-records assignments without depending on prior state.
- After a Dockerfile bump for a `git`/`tarball`-source plan
  (when those ship): the image_ref didn't change but you want
  every node to re-pull / re-build.
- For "make the cluster match this plan exactly" semantics in a
  CI pipeline, where you want a known-good baseline regardless
  of what the cluster looked like before.

For day-to-day operation, plain `xrlenv build apply --plan ...`
(no force) is what you usually want — it's idempotent and cheap.

---

### Test 4 — `test_calibration_hint_vs_actual`

**What you should see end-to-end**

This test answers a practical operator question: **are the
`size_hint_bytes` values in the YAML accurate enough to trust
for capacity planning?** Each entry in `build_plan.yaml` carries
a `size_hint_bytes` (set when the plan was generated, sourced
from the registry's manifest API at generation time). The
bin-packer trusts those numbers when deciding which node has
room. If they're off by 10x, the bin-packer will mis-place
entries and the cluster will run out of disk surprisingly.

The test:

1. Applies the canonical plan locally (all 8 images get pulled
   into the host's Docker daemon).
2. Runs `docker image inspect` against each pulled image to read
   the actual on-disk uncompressed bytes Docker reports.
3. Compares each entry's `size_hint_bytes` (from the YAML)
   against the measured `actual_bytes`. Asserts the ratio is in
   `[1.0, 5.0]` — a generous band that catches order-of-magnitude
   regressions in the hint generator without false-flagging
   normal compression-vs-uncompressed differences.
4. Writes a calibrated side-artifact YAML next to the canonical
   one, with `size_hint_source: cluster-reported` set on every
   entry. Operators can promote this calibrated file into the
   canonical plan when they want to lock in measured sizes.

**This test is local-mode only.** It needs `docker image inspect`
access on the same host the test runs on. On remote mode it
skips with a clear message — the cluster's nodes have the images
but the test runner doesn't.

**Running the local variant**

```bash
.venv/bin/python -m pytest \
    'tests/smoke/build_plan/test_dispatch_tb2.py::test_calibration_hint_vs_actual[local]' -v -s
```

The `-s` flag matters here — the test prints the calibration
table to stdout and pytest swallows it without `-s`.

**What the printed table looks like**

```
image_ref                                                 hint MB   actual MB   ratio
-------------------------------------------------------------------------------------
alexgshaw/fix-git:20251031                                  150.7       150.7   1.00x
alexgshaw/build-pov-ray:20251031                            155.5       155.5   1.00x
alexgshaw/overfull-hbox:20251031                            125.7       125.7   1.00x
alexgshaw/cobol-modernization:20251031                      154.5       154.5   1.00x
alexgshaw/prove-plus-comm:20251031                          468.7       468.7   1.00x
alexgshaw/constraints-scheduling:20251031                    47.5        47.5   1.00x
alexgshaw/nginx-request-logging:20251031                     64.2        64.2   1.00x
alexgshaw/dna-insert:20251031                                28.3        28.4   1.00x
```

`hint MB` is what's recorded in the YAML; `actual MB` is what
Docker reports on disk. A `1.00x` ratio means the YAML's hint is
accurate — the bin-packer is operating on truth. Any entry
showing `>2x` or `<0.5x` is worth investigating: either the
generator's source-of-truth (Docker Hub manifest) drifted, or
the image's structure changed (e.g. a base layer got rebuilt
much larger).

**Verifying the side-artifact**

```bash
LATEST=$(ls -td tmp/smoke-build-plan-tb2-local-*/ | head -1)
cat "$LATEST/build_plan.calibrated.yaml" | head -25
```

You should see a YAML with the same shape as the canonical
`build_plan.yaml`, except:

- Every entry's `placement.size_hint_bytes` reflects the
  measured on-disk size (instead of the registry-probe estimate).
- Every entry's `placement.size_hint_source` is set to
  `cluster-reported` (instead of `registry-probe` or
  `heuristic`).

This file is a side-artifact, not a replacement. The canonical
YAML stays as the operator-curated source of truth; if you want
to lock in the calibrated values, copy them in by hand. The
tracked canonical YAML preserves intent; the calibrated artifact
is the empirical truth from a specific cluster.

**Why the test asserts a 1x-5x band, not exact equality**

Docker's manifest API and `docker image inspect` should report
identical byte counts (both sum the uncompressed layer sizes),
which is why you typically see `1.00x`. But a few legitimate
sources of drift can push the ratio higher:

- The image was rebuilt between when the YAML was generated and
  when the test runs (different layer sizes).
- A different OCI image format reports sizes differently.
- The hint was set heuristically (e.g. for a tarball-source
  entry that hasn't been calibrated yet) and the actual build
  produced a different-sized image.

The 5x ceiling catches order-of-magnitude regressions
(e.g. someone returns bytes-per-layer instead of
bytes-total-image) without flagging normal drift.

**On remote mode (skip)**

Running this test against a remote cluster:

```bash
.venv/bin/python -m pytest \
    'tests/smoke/build_plan/test_dispatch_tb2.py::test_calibration_hint_vs_actual[remote]' -v -s
```

reports `SKIPPED (calibration is local-only — runs against the
same host's Docker daemon, not a remote cluster)`. That's the
intended behavior: calibration measures Docker's view of an
image, which lives on whichever host actually pulled it. The
remote cluster's nodes have the images but the test runner
doesn't, and there's no admin endpoint that exposes per-node
on-disk sizes today.

---

### Test 5 — `test_fresh_eight_dispatch`

**What you should see end-to-end**

This test pins the contract: **an operator can build their own
plan from scratch and the cluster materializes it correctly.**
Tests 1-4 all reuse the committed canonical phase-0 plan;
Test 5 generates a different plan at runtime so the dispatch
path runs against a `plan_id` the cluster has never seen before.

The test:

1. Walks the local harbor cache (`~/.cache/harbor/tasks/` or
   `$XRLENV_BENCHMARK_CACHE`) and picks 8 task ids that aren't in
   the canonical phase-0 set. The selection is alphabetical, so
   you get a deterministic 8-task slice off the top of whatever
   tasks aren't reserved for the smoke set.
2. Generates a new build plan in memory using the same generator
   the canonical YAML was emitted from. Computes its `plan_id`
   (different from `9d546b30...` because the entries are
   different).
3. Applies the plan with `force=True`.
4. Asserts: status `completed`, all 8 reach `done`, every
   chosen image_ref appears in the cluster's assignments list.

The 8 task ids picked at runtime show up in the artifact
JSON's `task_ids` field — useful when reproducing a specific
run later.

**Verifying in the admin panel**

Open `/builds`. You should see a **brand new row** alongside the
canonical plan's row:

- A different `plan_id` than `9d546b30eaf6...`. The exact value
  depends on the 8 tasks the generator picked.
- `done / total`: **8 / 8**, status `completed`.
- `applied at`: just now.
- `by`: `smoke-remote`.

Click into it — 8 assignments, all `done`, distributed across
your cluster's nodes per the bin-packer's choice. With clean VMs
and roughly comparable free disk, you'll typically see an even
split (e.g. 4/4 across two nodes); under disk pressure on one
node it'll lopside (6/2 or 5/3). Either is correct — see Test 1's
"Why placement varies" section for the why.

**Verifying in the smoke artifact**

```bash
LATEST=$(ls -td tmp/smoke-build-plan-tb2-remote-*/ | head -1)
cat "$LATEST/test_fresh_eight_dispatch.json" | python3 -m json.tool | head -30
```

Two top-level keys:

- `task_ids`: the 8 tasks the generator picked at runtime.
  Useful if you ever need to reproduce a specific run's plan
  (e.g. `astropy_1776_astropy-7166`-style ids for swebench, or
  `alexgshaw/<task>` task ids for tb2).
- `result`: the apply outcome. `status: completed`,
  `successes: 8`, `failures: 0`, full assignments list with
  `started_at`/`completed_at` reflecting this run.

**Verifying on each cluster node**

```bash
docker images --filter 'reference=alexgshaw/*' --format '{{.Repository}}' | sort -u
```

After Tests 1-4 you had 8 unique image refs across the cluster
(the canonical phase-0 set). After Test 5 you have **16** unique
image refs — the canonical 8 plus the 8 the fresh-eight test
just pulled. Run the command on each VM and union the results;
each ref appears on exactly one node by default.

**Wall-clock expectation**

Two regimes:

- **Cold cache** (the 8 chosen task images aren't on the cluster
  yet — typical first run after `docker image prune` or against
  a freshly-bootstrapped cluster): ~10-30 seconds. Each node
  pulls its share of the 8 images in parallel; total time is
  ~max-per-node-pull-time.
- **Warm cache** (you've run Test 5 before with the same harbor
  cache state, so the generator picks the same 8 tasks again):
  ~3-5 seconds. Same coordinator bookkeeping cost as Tests 2-3
  warm-cache; cache-hit `ensure_present` per entry.

15-20 seconds with two nodes pulling in parallel is the typical
cold-cache observation.

**Why this test exists**

The first four tests all dispatch the same canonical plan. If a
bug only surfaced when an operator authored their own plan
(e.g. a generator emits an entry shape the dispatch path
mishandles, or the `plan_id` collision logic depends on a
specific YAML key order), Tests 1-4 would not catch it. Test 5
exercises the full operator path: generate → compute plan_id →
apply → verify, with a plan the cluster has never seen.

A parallel `test_fresh_eight_dispatch` for git-source plans
(that builds a fresh slice of seta-env tasks each run) is a
natural extension once tarball-source dispatch lands and the
build-on-acquire eviction-recovery hook is in place. Today the
seta-env smoke covers the boundary contract; the actual
clone+build path is exercised by operator-driven
`xrlenv build apply --plan ...` runs against the canonical
seta-env starter.

**Skip behavior**

If your harbor cache doesn't have at least 8 task ids outside
the canonical smoke set, the test skips with a clear message.
Populate the cache (`harbor cache populate <subset>` or copy
from a previous machine's `~/.cache/harbor/tasks/`) to enable
the test.

---

### All five tests covered

That's the full surface of `test_dispatch_tb2.py`.
Quick recap of what each one pins:

| Test | Pins |
|---|---|
| `test_apply_canonical_plan` | First-ever apply distributes 8 entries across the cluster, all reach `done`. Plus the FFD non-determinism rule. |
| `test_idempotent_reapply` | No-force re-apply on a `completed` plan is a true no-op. |
| `test_force_reapply` | `--force` re-dispatches even when the plan is already `completed` (cache-hit on warm cluster). |
| `test_calibration_hint_vs_actual` | YAML's `size_hint_bytes` matches Docker's on-disk reality. Local-only; writes a calibrated side-artifact. |
| `test_fresh_eight_dispatch` | Operator-authored plans (different `plan_id`) dispatch correctly. |

Running all five in one shot:

```bash
# Local + remote (if remote is configured):
.venv/bin/python tests/smoke/build_plan/test_dispatch_tb2.py --mode all

# Local only:
.venv/bin/python tests/smoke/build_plan/test_dispatch_tb2.py
```

Total wall-clock for all five against a warm cache: ~30-60s
local, ~30-90s remote depending on cluster topology and the
fresh-eight test's pull cost.

---

## `test_dispatch_seta_env.py`

**Group**: Build-plan dispatch. **Wall-clock**: <2s total.
**Modes**: pytest, script. Three tests; the latter two parametrized
on `(local, remote)`.

**What it validates.** seta-env's Harbor-Dataset publishes
Dockerfiles only — no prebuilt images on a public registry. So
the canonical seta-env plan declares `context_source: type: git`
entries instead of `type: registry`. **Both git and tarball
source-build dispatch are live**: the coordinator routes
source-build entries to a per-node `GitSourceBuilder` that
clones / untars + `docker build`s + tags. These three tests
pin the schema + the live boundary for git + the remaining
"programmatic caller skipped `resolve_tarball_sources`"
rejection contract. The tarball happy-path apply lives in
[`test_tarball_dispatch.py`](#test_tarball_dispatchpy).

| Test | Pins |
|---|---|
| `test_canonical_plan_loads` | The committed seta-env starter plan parses cleanly; every entry is a `GitSource` pointing at `https://github.com/camel-ai/seta-env` with the `Harbor-Dataset/<id>/environment` subdir and the `xrlenv.benchmark: seta-env` label. Pure schema test, no cluster work. |
| `test_git_source_dry_run_reaches_placement` | `dry_run=True` apply against the seta-env plan reaches the placement layer without rejection. Confirms the coordinator's source-type gate no longer blocks git. Doesn't actually build anything. |
| `test_apply_rejects_unresolved_tarball_with_operator_friendly_error` | A synthetic tarball plan that skips the CLI's `resolve_tarball_sources` helper is rejected at apply time with a `ManifestInvalid` pointing at the helper. (The tarball happy path itself ships in [`test_tarball_dispatch.py`](#test_tarball_dispatchpy).) |

A real cluster apply of the seta-env starter (clone all 16 repos,
run 16 ``docker build``s, tag each, ~30+ min wall-clock + real
network) is **operator-driven**, not automated here:

```bash
xrlenv build apply \
    --plan xrlenv_plugins/benchmarks/seta/build_plan.yaml \
    --connect-host 127.0.0.1
```

The smoke covers the schema + boundary contracts; the cost of an
actual end-to-end build run is high enough that re-running it
casually wastes time and bandwidth. Operators run it deliberately
when they want the seta-env tasks materialized on the cluster.

> Can I `Ctrl-C` the running command?
Yes. The cluster's background apply task keeps running — you can see the per-assignment table updating as nodes finish builds, regardless of whether anything's polling.

To re-run cleanly: a fresh `xrlenv build apply --plan ...` (even with `--force`) would hit `rejected_in_flight` because the current apply is still `in_flight` in state.db. Three ways out:

- **Cancel the in-flight plan** (recommended once you've decided you don't want this build): `xrlenv build cancel --plan <id> --connect-host <admin-host>`. The admin marks the plan `cancelled`, dispatches `CancelBuildImageCommand` to each building node, and each node interrupts its in-flight `docker build` (kills the running build container, cancels the asyncio task). The admin /builds page shows the new terminal status; re-applying the same plan_id then proceeds cleanly. This is the right move when you don't want the build to finish.
- **Wait it out**: kill the CLI now → cluster keeps building → wait for completion or failure → re-run the new CLI to verify the fix on the next apply. Right move when you DO want the build to finish (you're just iterating on a separate change).
- **Watch progress locally**: `xrlenv build status --plan <id>` reads state.db directly (bypasses the admin HTTP path) so you can re-attach without disturbing the apply.

### Prerequisites

- Local mode: Docker daemon reachable. `dry_run=True` doesn't
  trigger any builds, so the daemon just needs to be answering
  pings.
- Remote mode: same `XRLENV_GRPC_HOST` / `XRLENV_OPERATOR_TOKEN`
  setup as the tb2 smoke. The cluster's nodes don't need to be
  connected to run these tests — neither dry-run nor the
  tarball-rejection synthetic plan reach a node.

### Invocation

```bash
# Local only:
.venv/bin/python tests/smoke/build_plan/test_dispatch_seta_env.py

# Local + remote:
XRLENV_GRPC_HOST=internal-ip \
XRLENV_OPERATOR_TOKEN=$(cat ~/.xrlenv/secrets/operator.token) \
.venv/bin/python tests/smoke/build_plan/test_dispatch_seta_env.py --mode all
```

---

### Test 1 — `test_canonical_plan_loads`

**What you should see end-to-end**

This is a pure schema test. The smoke loads
`xrlenv_plugins/benchmarks/seta/build_plan.yaml` through
`load_build_plan`, then walks the entries and asserts each one's
shape. **Nothing happens on the cluster.** Wall-clock: <0.1s.

If this test fails, either the canonical YAML drifted from the
schema or the schema regressed.

**Why it doesn't parametrize on mode**

Loading is a pure-Python operation against the YAML file in your
local checkout. Mode (`local` vs `remote`) only matters when the
test actually drives the cluster — which Test 1 never does.

---

### Test 2 — `test_git_source_dry_run_reaches_placement`

**What you should see end-to-end**

The smoke applies the seta-env starter plan with `dry_run=True`
and asserts the coordinator's source-type gate doesn't reject —
the apply reaches the placement layer and returns a
`status="dry_run"` outcome with all 16 entries placed.

This is a fast sanity check: it confirms git-source entries are
accepted, the bin-packer can find homes for them, and the plan is
structurally valid against the current coordinator. **It does
not** actually clone any repos or run `docker build` — that's
what happens on a non-dry-run apply.

**Verifying in the admin panel**

`dry_run` does not write any state.db rows. No new `/builds` row
should appear after this test.

**Verifying in the smoke artifact**

```bash
LATEST=$(ls -td tmp/smoke-build-plan-seta-env-*/ | head -1)
cat "$LATEST/test_git_source_dry_run.json" | python3 -m json.tool | head -40
```

You should see `status: dry_run` plus a `placement` block with
16 assignment rows, one per seta-env entry, distributed across
the cluster's connected nodes.

**On the cluster nodes**

Nothing changes. Dry-run dispatches no node-side work.

---

### Test 3 — `test_apply_rejects_unresolved_tarball_with_operator_friendly_error`

**What you should see end-to-end**

Tarball-source dispatch is **live** (sub-slice 1.b). The remaining
boundary contract: a programmatic caller that builds a `BuildPlan`
in-memory and skips the CLI's `resolve_tarball_sources` helper
ships a wire payload with no bytes. The coordinator catches that
upfront and rejects with a `ManifestInvalid` pointing at the
helper.

This smoke synthesizes a one-entry plan with
`context_source: type: tarball` and **no** `content_b64`, then
applies it with `dry_run=True`. The coordinator's tarball-gate
rejects with a message naming `resolve_tarball_sources` and the
offending `image_ref`.

`dry_run=True` is required for the same reason as before: a
non-dry-run remote apply spawns a background asyncio task that
catches the exception and persists a `partial_failure` plan
record (correct admin behavior, but extra noise we don't want
here). `dry_run` surfaces the validation rejection synchronously
through the admin server's 400 response, which the smoke helper
translates to a `ManifestInvalid` so test bodies stay
mode-agnostic.

The tarball **happy-path apply** lives in
[`test_tarball_dispatch.py`](#test_tarball_dispatchpy) — that's
the file to read if you want to see what a real tarball build
looks like end-to-end.

**Verifying in the smoke artifact**

```bash
LATEST=$(ls -td tmp/smoke-build-plan-seta-env-*/ | head -1)
cat "$LATEST/test_apply_rejects_unresolved_tarball.json" \
    | python3 -m json.tool
```

Single key: `message` — the rejection message verbatim.

---

## `test_tarball_dispatch.py`

**Group**: Build-plan dispatch. **Wall-clock**: cap test <1s;
happy-path build ~30-60s on first run (busybox base layer pull
+ trivial `docker build`). **Modes**: pytest, script. The three
tests have mixed mode constraints documented per-test.

**What it validates.** The complete sub-slice 1.b surface for
tarball-source build dispatch: the operator-side cap rejection
that fires before any wire traffic, the happy-path build that
produces a labelled image, and the on-disk source-spec registry
that's load-bearing for build-on-acquire (sub-slice 2).

All three tests are **local-only** (no `[local]`/`[remote]`
parametrization — they don't run twice). The file-level docstring
explains why: every assertion in this file inspects state that
isn't reachable over the admin API today (docker labels,
on-disk source-spec registry, local cap check).

| Test | Pins |
|---|---|
| `test_tarball_cap_rejects_oversized` | `resolve_tarball_sources(plan, max_bytes=<small>)` raises `ManifestInvalid` naming the image_ref + `--build-tarball-max-bytes` when a tarball exceeds the cap. Pure CLI-side check; no cluster reachability needed. |
| `test_tarball_happy_path` | A real `BuildImageCommand` round-trip materializes a tagged image; the image carries the two reserved labels `xrlenv.image.rebuild-cost=local-build-cheap` and `xrlenv.cancel-key=<image_ref>`. |
| `test_tarball_source_registry_persists` | After a successful tarball build, the per-image_ref source-spec registry has `spec.json` + `content.bin` on disk under the builder's cache root. Uses a per-test `XRLENV_BUILD_CONTEXT_CACHE` so it doesn't poison the operator's persistent registry. |

### Prerequisites — local mode

- Docker daemon reachable. The happy-path test pulls
  `busybox:1.36` (a few MB) the first time it runs.
- ~10 MB free disk under `$TMPDIR` for the per-test cache root.

### Invocation

```bash
# All three tests (default, local-only):
.venv/bin/python tests/smoke/build_plan/test_tarball_dispatch.py

# Just the cap-reject (fast — useful as a regression smoke):
.venv/bin/python -m pytest \
    tests/smoke/build_plan/test_tarball_dispatch.py::test_tarball_cap_rejects_oversized \
    -v -s
```

Artifacts land under `<repo>/tmp/smoke-build-plan-tarball-local-<utc-ts>/`:

```
test_tarball_cap_rejects.json            # rejection message verbatim
test_tarball_happy_path.json             # apply outcome + image_ref
test_tarball_source_registry_persists.json   # spec.json keys + content size
```

### Why this contract matters

Tarball dispatch is the path operators use for **private build
contexts** — Dockerfiles that aren't checked into a public git
repo, or proprietary base layers. The operator-side cap stops a
runaway 5 GB context from being shipped over gRPC before they
notice; the cancel-key label is the operator's only way to
interrupt a tarball build mid-flight; the source-spec registry
is what lets build-on-acquire rebuild the image after eviction
without re-shipping bytes.

If any of these three regress, the operator-visible workflow
degrades silently (tarballs ship oversized, cancel can't find
the build container, evicted images stay evicted). The smoke
catches all three.

---

## `test_pin_budget_and_calibrate.py`

**Group**: Build-plan dispatch. **Wall-clock**: pin-budget test
~1s (no build); calibrate ~5-30s depending on cluster size
(one `report_images` round-trip per connected node). **Modes**:
pytest, script.

**What it validates.** The two operator-facing safety / tooling
features shipped in Phase B: hard-reject pin-budget over-commit
at apply time, and the `xrlenv build calibrate` flow that
replaces heuristic size hints with cluster-measured values.

| Test | Pins | Parametrize |
|---|---|---|
| `test_pin_budget_rejects_at_dry_run` | A plan whose pinned entries' `size_hint_bytes` collectively exceed each node's `cap_per_node_gb` budget rejects with `ManifestInvalid` containing `pin-budget over-commit` + the per-node over-by amount. The check runs inside `BuildCoordinator._apply_per_image_ref` so both local and remote modes exercise it. | `[local]` always; `[remote]` when `XRLENV_GRPC_HOST` / `XRLENV_ADMIN_HOST` is set |
| `test_calibrate_writes_cluster_reported_sizes` | `xrlenv build calibrate` walks connected nodes' `report_images()`, picks per-image_ref max sizes, writes a calibrated YAML with `size_hint_source: cluster-reported` on measured entries. | none — remote-only; the fixture skips upfront when no admin endpoint is configured (no empty `[local]` parametrization) |

### Prerequisites — local mode

- Just a working Python env. The pin-budget test runs entirely
  through the in-process `LocalRuntime`'s `BuildCoordinator`;
  no Docker daemon needed for the rejection path.

### Prerequisites — remote mode

- `xrlenv up` reachable; at least one node `connected` in
  `xrlenv nodes`.
- For the calibrate test specifically: at least one image
  materialized on the cluster — the test uses the canonical
  tb2 plan as input, so running `test_dispatch_tb2.py` first
  (or `xrlenv build apply --plan
  xrlenv_plugins/images_build/terminal_bench_2/build_plan.yaml
  --connect-host ...`) populates the cluster with images
  the calibrate flow can measure. With zero overlap the
  calibrate still passes — `unmeasured` lists all entries
  and the YAML keeps operator hints — but the test then
  doesn't assert any actual measurement.
- Standard env vars: `XRLENV_GRPC_HOST` (or `XRLENV_ADMIN_HOST`),
  `XRLENV_OPERATOR_TOKEN`.

### Invocation

```bash
# Pin-budget reject only (fast, mode-agnostic regression):
.venv/bin/python -m pytest \
    tests/smoke/build_plan/test_pin_budget_and_calibrate.py::test_pin_budget_rejects_at_dry_run \
    -v -s

# All tests against a live cluster:
XRLENV_GRPC_HOST=internal-ip \
XRLENV_OPERATOR_TOKEN=$(cat ~/.xrlenv/secrets/operator.token) \
.venv/bin/python tests/smoke/build_plan/test_pin_budget_and_calibrate.py --mode all
```

Artifacts under `<repo>/tmp/smoke-build-plan-pin-cal-<mode>-<utc-ts>/`:

```
test_pin_budget_rejects_<mode>.json   # rejection message verbatim
test_calibrate.json                    # cmd stdout + output_path
```

### Why this contract matters

Pin-budget over-commit is a silent operational landmine: a plan
that fits today might over-commit a node tomorrow when an
unrelated workload claims disk space. Hard-rejecting at apply
time forces the operator to make an explicit choice (unpin,
raise the budget, or recalibrate sizes) instead of debugging
why rollouts on node X are slow weeks later.

Calibrate is the **only way** to get accurate size hints into a
build-plan.yaml without hand-measuring every image. Without
calibrate working, operators ship heuristic hints (which the
bin-packer pads conservatively) and FFD makes worse placement
decisions than it could. If calibrate regresses to "writes
empty sizes" or "fails to flip size_hint_source", the
operator-facing workflow silently drifts back toward bad
placement.

---

## `test_cancel_regression.py`

**Group**: Build-plan dispatch. **Wall-clock**: local test <1s;
cluster test ~30-60s (a fast-failing apply + a cancel
round-trip). **Modes**: pytest, script.

**What it validates.** The `xrlenv build cancel` operator
surface end-to-end. Unit tests in `tests/unit/cli/` and
`tests/unit/admin/` cover the branching logic exhaustively;
this smoke pins the **wire round-trip** from CLI →
`/api/build/cancel` → admin orchestrator → spec-21
`CancelBuildImageCommand` → state.db status flip.

Both tests are single-mode by design — no parametrization, no
SKIPPED entries. Their mode is encoded in the function name.

| Test | Pins | Mode |
|---|---|---|
| `test_local_cancel_flips_plan_status` | Local synthetic regression: a plan in state.db with `in_flight` status flips to `cancelled` via `cmd_build_cancel` (no `--connect-host`), and the warning message names `--connect-host` as the way to interrupt running cluster builds. | local only |
| `test_cluster_cancel_interrupts_pending_assignments` | A real cluster apply (registry-source plan referencing a non-existent image, so dispatch is fast) is cancelled via the CLI; the cluster ends up with the plan + assignments marked `cancelled` (not `failed`). The fixture skips upfront when no admin endpoint is configured. | remote only |

### Prerequisites — local mode

- Just a writable `tmp_path` (handled by pytest). The test
  drives the state.db directly through `SqliteStateStore`.

### Prerequisites — remote mode

- `xrlenv up` reachable; at least one node `connected`.
- Standard env vars: `XRLENV_GRPC_HOST` (or `XRLENV_ADMIN_HOST`),
  `XRLENV_OPERATOR_TOKEN`.

### Invocation

```bash
# Local-mode synthetic regression (always cheap):
.venv/bin/python -m pytest \
    tests/smoke/build_plan/test_cancel_regression.py::test_local_cancel_flips_plan_status \
    -v -s

# Cluster-mode round-trip:
XRLENV_GRPC_HOST=internal-ip \
XRLENV_OPERATOR_TOKEN=$(cat ~/.xrlenv/secrets/operator.token) \
.venv/bin/python tests/smoke/build_plan/test_cancel_regression.py --mode remote
```

Artifacts under `<repo>/tmp/smoke-build-plan-cancel-<mode>-<utc-ts>/`:

```
test_local_cancel.json       # rc + stdout from cmd_build_cancel
status_pre_cancel.json       # build status output before cancel
cancel_output.json           # build cancel rc + stdout
apply_output.json            # the original apply's eventual exit
status_post_cancel.json      # build status post-cancel (must show "cancelled")
```

### Why this contract matters

`xrlenv build cancel --connect-host` is the **only** operator
recovery path for a stuck or unwanted cluster apply, short of
killing `xrlenv up` and rebuilding state.db. The cancel
orchestrator touches three persistence layers (the plan record,
per-assignment rows, the per-node spec-21 fanout) — a regression
in any layer leaves the operator with no clean way out.

The cluster-mode smoke uses a fast-failing apply (registry ref
that doesn't exist) so the cancel round-trip doesn't need to
race against a real long-running build. The kill-mid-build
timing path is covered by the unit tests in
`tests/unit/node/test_source_builder.py::test_cancel_signals_in_flight_task`.

---

## `test_build_on_acquire_after_eviction.py` — Plan 4

**Group**: Build-plan dispatch. **Wall-clock**: 10-60s on rebuild
(typical Harbor task image build); <1s if the eviction precondition
was missed and the scheduler picked a cache-warm node. **Mode**:
remote only.

**What it validates.** Sub-slice 2's load-bearing property: an
evicted source-built image rebuilds automatically on the next
`acquire_container` — the persistent source-spec registry
(`<cache_root>/source-registry/<sha>/spec.json` + `content.bin`)
fires the `lookup_producer` hook inside `ImageCacheManager.
ensure_present`, no operator re-apply needed.

| Test | Pins | Mode |
|---|---|---|
| `test_acquire_after_eviction_rebuilds_image` | A `Client.acquire_container` for an operator-evicted source-built ref completes successfully + within 5min. Timing distinguishes "real rebuild fired" (≥2s) from "cache hit, precondition missed" (<2s). | remote only |

The pytest test itself **cannot** delete a remote node's docker
image — that requires SSH access we don't want to couple to.
The operator performs the eviction by hand (steps below), then
runs the smoke, which drives the acquire side + reports the
timing.

### Step-by-step operator runbook

#### 1. Pick a target image

Use a source-built image (`context_source: type: git` or
`type: tarball`) that's already on your cluster. Registry-pulled
refs won't exercise build-on-acquire — they'd just re-pull.

```bash
# Easiest pick: any of your seta-env builds.
export TARGET=xrlenv-seta-env/0:main

# Or a tarball image left by Plan 2's smoke (if you ran it
# against a single-host setup, this lives only on localhost):
#   export TARGET=xrlenv-smoke/tarball-hello:1
```

#### 2. Find the node that holds it

```bash
# Iterate over all rostered nodes:
for NODE in gcp-osworld-agent-junnan-li-3 gcp-osworld-exp-1; do
  echo "=== $NODE ==="
  ssh "$NODE" docker images $TARGET --format '{{.Repository}}:{{.Tag}} ({{.CreatedAt}})'
done
# Note the node + the original CreatedAt — you'll compare against
# this after the test runs.
export NODE=<the one with the image>
```

#### 3. Confirm the source-spec registry has it on that node

```bash
ssh $NODE sudo find /var/cache/xrlenv/source-registry/ -name spec.json -exec grep -l "$TARGET" {} \;
# Should print at least one spec.json path. If empty, this image
# wasn't source-built or its registry entry is missing — pick a
# different target.
```

#### 4. Evict the image on that node

```bash
ssh $NODE docker rmi $TARGET
ssh $NODE docker images $TARGET   # must print no rows
```

The source-spec registry **persists** through `docker rmi` (it
lives in `/var/cache/xrlenv/source-registry/`, not in Docker's
own storage). That's the whole point — eviction loses the
image but keeps the recipe.

#### 5. Run the smoke

```bash
export SMOKE_TARGET_IMAGE=$TARGET
# (XRLENV_GRPC_HOST + XRLENV_CONSUMER_TOKEN already exported
# from your earlier smoke runs.)

.venv/bin/python -m pytest -v -s \
    tests/smoke/build_plan/test_build_on_acquire_after_eviction.py
```

What the smoke does internally:
1. Dials the cluster's consumer-facing control plane.
2. Issues `Client.acquire_container(image=$TARGET, command=["sleep", "5"])`.
3. Times the acquire (start → sandbox returned).
4. Destroys the sandbox immediately.
5. Writes timing + interpretation to
   `tmp/smoke-build-plan-build-on-acquire-remote-<ts>/test_acquire_after_eviction.json`.

#### 6. What "pass" looks like

Test passes as long as the acquire **succeeds** (sandbox handle
returned). The timing in the JSON artifact tells you whether the
rebuild actually fired:

| `acquire_duration_s` | `rebuild_likely_fired` | Interpretation |
|---|---|---|
| ≥ 10s | `true` | Real source-spec rebuild fired ✅ |
| 2-10s | `true` | Probably rebuild, possibly a registry pull of a base layer |
| < 2s | `false` | Almost certainly a docker cache hit. Did the eviction precondition fail? Check that you `docker rmi`-ed on the SAME node the scheduler picked. |

The "Did this actually exercise the property?" check is the
`acquire_duration_s` value, not the pass/fail of the test.

#### 7. Verify post-acquire (optional but recommended)

```bash
# Image should be back on the SAME node, with a fresh CreatedAt.
ssh $NODE docker images $TARGET --format '{{.Repository}}:{{.Tag}} ({{.CreatedAt}})'
# The CreatedAt should be seconds ago (during the smoke run),
# not the original build timestamp from step 2.

# Confirm the rebuild fired (look for the producer log line):
ssh $NODE sudo journalctl -u xrlenv-node -n 200 | grep -i 'producing'
# Expect: image_cache: producing xrlenv-seta-env/0:main via builder ...
```

If the journal line says `producing ... via builder` AND the
CreatedAt is fresh, ✅ build-on-acquire works.

#### 8. (Optional) Negative case — confirm rebuild only fires for source-built refs

`docker rmi` an image that was **never source-built** — e.g.
the `busybox:1.36` base layer left around from Plan 2 — then
acquire it. The journal should show:

```
image_cache: pulling busybox:1.36 (timeout=...)
```

NOT `producing ... via builder`. This proves build-on-acquire
is gated on a registered source spec (it doesn't accidentally
try to rebuild registry-pull images).

### Why this isn't fully automated

The eviction step (`docker rmi` on a remote node) is the only
piece that couples to your SSH setup. Three options to fully
automate would each add surface we'd rather not ship:

1. **Take an `--ssh-cmd` flag**. Operator passes their SSH
   wrapper; smoke runs `<ssh_cmd> docker rmi <ref>`. Works but
   ties the test to your env.
2. **Add a `ForceRemoveImageCommand` over spec-21**. Cluster
   gains an admin-driven `docker rmi` primitive. Useful for
   fault-injection smokes generally; would be its own design
   discussion.
3. **Skip the live smoke entirely**. The unit tests
   `test_lookup_producer_rebuilds_after_eviction` +
   `test_source_registry_survives_builder_recreation` already
   cover the load-bearing logic.

For now, this operator-driven recipe is the live-cluster
sanity check. Revisit the automation question if eviction
recovery shows real bugs in the field.

---


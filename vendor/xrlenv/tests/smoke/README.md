# XRLEnv smoke tests — navigation map

Manually-run smokes that exercise xrlenv end-to-end against a real
Docker daemon and (where applicable) a real running control plane.
**Excluded from the default `pytest -q` suite** via `addopts =
"--ignore=tests/smoke"` in `pyproject.toml`. Run them deliberately,
not on every push — most pull large images and / or take minutes
to hours.

These smokes are **the** load-bearing validation layer for xrlenv's
real-world correctness. Unit tests catch logic regressions; these
catch wiring regressions across the SDK ↔ control plane ↔ node ↔
Docker boundary that unit tests can't see by construction. Treat
them as **more important** than the unit suite for any change that
touches the dispatch path, the docker-py drop-in, the build
coordinator, or the spec-21 wire format.

## Layout

Each smoke group lives in its own subdirectory under `tests/smoke/`,
with its tests and runbook sitting side-by-side:

```
tests/smoke/
├── README.md                            ← you are here (nav map)
├── _build_plan_dispatch_helpers.py      ← shared helpers (do not edit casually)
├── _artifacts.py                        ← shared artifact-dir helper
├── cluster_bringup/                     ← first thing after a fresh deploy
│   ├── README.md
│   ├── single_rollout.py
│   └── cluster_smoke.py
├── api_surface/                         ← case-2 consumer wire contract
│   ├── README.md
│   ├── raw_container_smoke.py
│   └── dropin_cluster_smoke.py
├── benchmark_integration/               ← real upstream harness round-trip
│   ├── README.md
│   ├── test_swebench_drop_in.py
│   ├── test_terminal_bench_2_drop_in.py
│   └── test_compat_docker_smoke.py
├── multi_tenancy/                       ← two users, real owner-scoped traffic
│   ├── README.md
│   └── two_user_cluster_smoke.py
└── build_plan/                          ← xrlenv build apply / cancel / calibrate
    ├── README.md
    ├── test_dispatch_tb2.py
    ├── test_dispatch_seta_env.py
    ├── test_tarball_dispatch.py
    ├── test_pin_budget_and_calibrate.py
    └── test_cancel_regression.py
```

Each group's `README.md` is the operator-facing runbook for the
smokes in that directory — depth (per-test prerequisites,
expected output, what "pass" means, recovery recipes) lives
there. This top-level page covers the conventions all groups
share plus the **Phased manual-run plan** map below.

## Smoke groups

| Group | Purpose | Page |
|---|---|---|
| **Cluster bring-up** | First thing to run after a fresh deploy or any control-plane change | [cluster_bringup/README.md](cluster_bringup/README.md) |
| **API surface (case-2 primitives)** | Pin the consumer-facing wire contract before any harness layer is involved | [api_surface/README.md](api_surface/README.md) |
| **Benchmark integration** | Drive a real upstream harness through xrlenv to catch contract regressions a unit test can't see | [benchmark_integration/README.md](benchmark_integration/README.md) |
| **Multi-tenancy** | Two users submit real raw-container + rollout traffic; verify owner-scoped admin views + cross-owner 404 + live fair-share by eye | [multi_tenancy/README.md](multi_tenancy/README.md) |
| **Build-plan dispatch** | Pin `xrlenv build apply --plan` + `cancel` + `calibrate` end-to-end: distribution, re-apply semantics, tarball dispatch, pin-budget, source-spec registry | [build_plan/README.md](build_plan/README.md) |

## Phased manual-run plan → smoke command map

After a Phase A/B/C release, run these in order against your
cluster. Each row maps a numbered plan from the operator-facing
checklist to the exact pytest invocation that exercises it —
so you can run them **one at a time**, eyeball the output, and
move on. All commands assume `cwd = <repo root>`. Set the
environment once at the top of the session:

```bash
export ADMIN=127.0.0.1                                   # admin host
export XRLENV_GRPC_HOST=$ADMIN                           # smokes read this for remote mode
export XRLENV_OPERATOR_TOKEN=$(cat ~/.xrlenv/secrets/operator.token)
```

| Plan | What it verifies | Smoke command |
|---|---|---|
| **1a — pin-budget reject** | Apply-time guard rejects plans whose pinned entries over-commit a node's budget. Both modes exercise the same coordinator code path. | `.venv/bin/python -m pytest -v -s tests/smoke/build_plan/test_pin_budget_and_calibrate.py::test_pin_budget_rejects_at_dry_run` |
| **1b — tarball cap reject** | Operator-side cap rejects oversized tarballs *before any wire traffic*. Local-only; never touches the cluster. | `.venv/bin/python -m pytest -v -s tests/smoke/build_plan/test_tarball_dispatch.py::test_tarball_cap_rejects_oversized` |
| **2 — tarball happy path** | Real `BuildImageCommand` builds a tiny FROM-busybox image; the resulting image carries `xrlenv.image.rebuild-cost=local-build-cheap` + `xrlenv.cancel-key=<image_ref>`. Includes the source-spec registry persistence check (Plan 4 prerequisite). | `.venv/bin/python -m pytest -v -s tests/smoke/build_plan/test_tarball_dispatch.py::test_tarball_happy_path tests/smoke/build_plan/test_tarball_dispatch.py::test_tarball_source_registry_persists` |
| **3 — calibrate** | After at least one image is materialized on the cluster, `xrlenv build calibrate` walks each node's `report_images`, writes a calibrated YAML with `size_hint_source: cluster-reported`. Remote-only. | `.venv/bin/python -m pytest -v -s tests/smoke/build_plan/test_pin_budget_and_calibrate.py::test_calibrate_writes_cluster_reported_sizes` |
| **4 — build-on-acquire after eviction** | Operator-driven (the `docker rmi` step needs SSH; can't fully automate). After the operator evicts a source-built image on its preferred-home node, the smoke drives the acquire side + asserts the rebuild fired. See [build_plan/README.md § `test_build_on_acquire_after_eviction.py`](build_plan/README.md#test_build_on_acquire_after_evictionpy--plan-4) for the step-by-step runbook (pick image, find node, `docker rmi`, run test). | `SMOKE_TARGET_IMAGE=<ref> .venv/bin/python -m pytest -v -s tests/smoke/build_plan/test_build_on_acquire_after_eviction.py` |
| **5a — local cancel** | `xrlenv build cancel` without `--connect-host` flips state.db status and emits a clear "use --connect-host to interrupt running cluster builds" warning. | `.venv/bin/python -m pytest -v -s tests/smoke/build_plan/test_cancel_regression.py::test_local_cancel_flips_plan_status` |
| **5b — cluster cancel round-trip** | `xrlenv build cancel --connect-host` round-trips through admin's `/api/build/cancel`, the plan ends `cancelled` cluster-side. Remote-only. | `.venv/bin/python -m pytest -v -s tests/smoke/build_plan/test_cancel_regression.py::test_cluster_cancel_interrupts_pending_assignments` |

**Notes**

- `-s` matters: every smoke writes per-test JSON summaries to
  stdout AND to `<repo>/tmp/smoke-build-plan-*-<mode>-<ts>/` for
  later inspection. `pytest -s` keeps the stdout from being
  swallowed.
- Plan 1b doesn't need a running `xrlenv up` — it's a pure
  client-side function call. Run it before bringing the cluster
  up to catch the case where the venv hasn't been refreshed.
- Plan 1a + 3 + 5b need the **admin process to be running the
  current code**. If a smoke fails with an error message that
  doesn't match what `pytest -q` would produce on the same code,
  refresh the admin's venv (`bash deploy/refresh.sh` on the
  admin VM) — the running admin is almost certainly on stale
  code.

## Conventions shared across smokes

**Dual-mode invocation.** Every smoke runs as a pytest module *and*
as a standalone script. Pick whichever is more convenient:

```bash
# pytest mode (recommended for CI-shaped runs):
.venv/bin/python -m pytest tests/smoke/<group>/<smoke>.py -v -s

# script mode (recommended for ad-hoc operator iteration):
.venv/bin/python tests/smoke/<group>/<smoke>.py [...flags...]
```

`-s` matters for any smoke whose output goes to stdout via
`print()` (the calibration tables, in particular) — pytest swallows
that without it.

**Three-mode structure** (cluster_smoke, raw_container_smoke,
dropin_cluster_smoke share this pattern):

| Mode | When to use |
|---|---|
| `--in-process` | Single Python process, `LocalRuntime`, no gRPC. Fastest sanity check; verifies the wire contract itself works. |
| Embedded (default, no `--connect-host`) | Script boots the control plane in-process on `--grpc-port` (default 50051). Cloud nodes already attached via systemd reconnect to it. Useful when no `xrlenv up` is already running. |
| `--connect-host`, `--consumer-token` | Script leaves the operator's `xrlenv up` running and dials it. The realistic SDK shape: consumer + control plane on different machines. |

**Build-plan smokes use a two-mode `local | remote` shape** instead
(local = in-process `LocalRuntime` + host Docker daemon; remote =
admin API + spec-21 fanout). See the build-plan group page for
details.

**Artifact output.** Smokes route durable output to
`<repo>/tmp/smoke-<label>-<utc-ts>/` (gitignored). Per-test JSON
summaries land there alongside calibration tables, side-artifact
YAMLs, harbor trial trees, etc.

**Cleanup.** Pulled images stay around so subsequent runs hit the
warm cache. To reclaim disk:

```bash
docker image prune                                   # untagged layers only
docker image rm $(docker images -q 'alexgshaw/*')    # all tb2 task images
docker image rm $(docker images -q 'swebench/*')     # all swebench instance images
docker image rm $(docker images -q 'xrlenv-smoke/*') # smoke-created images
```

The cluster's node-side `ImageCacheManager` only evicts under disk
pressure, so the remote cache also persists across runs (intentional).

## Adding a new smoke

1. Pick the group whose subdirectory under `tests/smoke/` fits
   the smoke's purpose. Drop the new `test_*.py` (or
   `*_smoke.py`) file there.
2. **Make it dual-mode**: pytest test functions plus an
   `if __name__ == "__main__":` script entry point that delegates
   to `pytest.main([__file__, ...])`. Smokes are manual-run; they
   should be invokable both ways.
3. **Document with the same depth as every other smoke in that
   group's README.** Add a row to the group's overview table, then
   a full section: group, wall-clock, modes, what it validates,
   prerequisites, invocation examples, output layout, and what
   "pass" means.
4. **If the smoke maps to a phased manual-run plan, add a row to
   the "Phased manual-run plan → smoke command map" above** so
   future operators can run it one-at-a-time without spelunking
   the per-group READMEs.
5. Route durable artifacts to `<repo>/tmp/smoke-<label>-<utc-ts>/`
   (use `_build_plan_dispatch_helpers.smoke_artifact_dir(...)` or
   the same shape).
6. If the smoke needs a remote cluster, gate the remote path on
   the standard `XRLENV_GRPC_HOST` (with optional `XRLENV_ADMIN_HOST`
   / `XRLENV_ADMIN_PORT` overrides) so an operator who's already
   pointed their environment at a cluster gets remote mode for free
   — don't invent smoke-specific env vars.
7. State what the smoke pins in operator-facing terms (e.g. "every
   plan entry reaches `done` on at least one node") rather than
   internal phase / slice labels. The smokes outlive any single
   release cycle; the runbook should still read clean a year from
   now.

If the new smoke fits one of the existing groups, add it there. If
it pins a meaningfully new shape (e.g. multi-tenant isolation,
preemption recovery), create a new subdirectory under
`tests/smoke/`, add its `README.md`, and reference it in the
table above.

# Sample build plans

Cluster image distribution plans for `xrlenv build apply`. Each
file is a complete, runnable plan. See
`docs/deployment/build_plans.md` for the full schema + recovery
flows.

## Canonical plug-in plans

Each integrated benchmark ships per-selection plans next to its
manifest. Apply these directly, or copy + edit for a different
selection / budget / replication.

- swebench-verified
  - `xrlenv_plugins/benchmarks/swebench_verified/build-plans/smoke.yaml` — 8 acceptance instances.
  - `xrlenv_plugins/benchmarks/swebench_verified/build-plans/instances.yaml` — explicit instance list (edit before apply).
  - `xrlenv_plugins/benchmarks/swebench_verified/build-plans/all.yaml` — full ~500-instance catalog. Nominal ~3 GiB/instance; effective on-disk after layer dedup ≈ ~100 GiB cluster-wide at R=1, divided across whatever nodes the bin-packer places to. Run `--dry-run` first.
- terminal-bench-2
  - `xrlenv_plugins/benchmarks/terminal_bench_2/build-plans/smoke.yaml` — 8 acceptance tasks.
  - `xrlenv_plugins/benchmarks/terminal_bench_2/build-plans/instances.yaml` — explicit task list (edit before apply).
  - `xrlenv_plugins/benchmarks/terminal_bench_2/build-plans/all.yaml` — every task in the harbor cache.

## Cross-benchmark example

- `multi_benchmark_smoke.yaml` — tb2 + swebench-verified smoke
  sets in one apply. Bin-packer treats them as one cluster-wide
  image set.

## Testing recipe

End-to-end smoke for the build-plan flow on your laptop:

```bash
# 1. Validate the YAML schema + see the placement (no Docker calls):
xrlenv build apply --plan examples/build-plans/multi_benchmark_smoke.yaml --dry-run

# 2. Apply against an in-process LocalRuntime (uses local Docker):
xrlenv build apply --plan examples/build-plans/multi_benchmark_smoke.yaml

# 3. Inspect the persisted snapshot:
xrlenv build status                       # most recent plan
xrlenv build status --plan <plan_id>      # specific plan

# 4. Open the admin /builds page (after `xrlenv up --admin-port 8080`):
open http://127.0.0.1:8080/builds
```

For the cluster flow against a running control plane, append
`--connect-host <admin-host> --connect-port 8080` to step 2 + 3.
The CLI POSTs the plan to `/api/build/apply`, polls
`/api/build/plans/<id>` every 3 s, and prints a per-status update
each time the rollup advances.

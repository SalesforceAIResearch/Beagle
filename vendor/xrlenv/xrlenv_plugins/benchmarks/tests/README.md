# Benchmark tests

## `integration/` — unified benchmark sweep runner
A config-driven harness (`run_benchmarks.py` + `benchmarks.yaml`) that runs each
benchmark's `run_full_sweep.sh` / `run_oracle_sweep.py` over a profile — the full green
set, or a deterministic CI sample — and exits 0 iff every selected benchmark passed. Its
pure logic is unit-tested offline (`test_run_benchmarks.py`, `test_wrapper_env_ordering.py`).
Details + example invocations: [`integration/README.md`](integration/README.md). Meant to
be **run by hand** (real cluster sweeps).

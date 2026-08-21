# echo_bench — own-benchmark worked example

A trivially-simple pip package implementing a brand-new benchmark.

## What it does

Three instances. Each instance gives the agent a target string in
the initial observation; the agent's only legal action is
`{"output": "<string>"}`. Reward `1.0` if the agent's output
matches the target byte-for-byte, `0.0` otherwise. The grader is
a 30-line bash script that does `cmp` between two files.

| Instance | Target string |
|---|---|
| `echo-hello` | `Hello, world!` |
| `echo-multiline` | `line one\nline two\nline three` |
| `echo-symbols` | `<>&|$"' \t!@#%^*()` |

This is the simplest possible **scenario 1** plug-in: own
adapter, own manifest, own grader. If you're shipping a more
realistic benchmark, the same package shape applies — just with a
larger adapter and richer scoring.

## Status

Requires the **D22** runtime fix (sandbox import-path extension)
shipped on `dev/phase1`. Without D22, rollouts seal
`setup_failed: ModuleNotFoundError`. See
`notes/deferred_audit_todos.md`.

## Install + smoke

```bash
# 1. Install the plug-in. Editable install lets you tweak the
#    adapter and re-run without re-installing.
uv pip install -e examples/pip_new_datasets_or_benchmark/echo_bench

# 2. Build the per-instance images. Three tiny images on a single
#    python:3.12-slim base; ~150 MB total disk after the first
#    pull, seconds per re-build.
bash examples/pip_new_datasets_or_benchmark/echo_bench/scripts/build-task-images.sh

# 3. Run the oracle smoke locally (no cloud, no SSH tunnels).
.venv/bin/python examples/pip_new_datasets_or_benchmark/echo_bench/examples/echo_smoke.py --local
```

Expected: 3 / 3 rollouts seal `finished` with `final_reward = 1.0`.

## Where everything lives

```
echo_bench/
├── pyproject.toml                       # name=xrlenv-echo-bench; entry-point at xrlenv.benchmarks
├── README.md                            # this file
├── xrlenv_plugins/benchmarks/echo_bench/
│   ├── __init__.py
│   ├── plugin.py                        # entry-point callable; returns manifest.yaml path
│   ├── manifest.yaml                    # spec-06 template manifest
│   └── adapter.py                       # EchoBenchInstanceResolver + EchoBenchEnvAdapter
├── scripts/
│   ├── build-task-images.sh             # builds 3 images (python:3.12-slim base)
│   ├── Dockerfile                       # one Dockerfile shared across all 3 instances
│   └── run-echo-tests.sh                # in-sandbox grader, uploaded at reward time
├── examples/
│   └── echo_smoke.py                    # 3-instance oracle driver
└── tests/
    └── test_smoke.py                    # cheap pre-flight: entry-point registers; adapter imports
```

## Adapting this to your own benchmark

1. Rename the package and the entry-point in `pyproject.toml`.
2. Replace `EchoBenchInstanceResolver._INSTANCES` with your own
   `instance_id → init_params` map (or load it from a vendored
   dataset like `byo_dataset_harbor/` does).
3. Replace `EchoBenchEnvAdapter.step` with whatever interaction
   loop your benchmark needs.
4. Replace `scripts/run-echo-tests.sh` with your scoring command.
5. Adjust `scripts/Dockerfile` for any extra deps your benchmark
   needs at runtime (the stub-runtime layer always adds python3
   + xrlenv core; you only ship the benchmark-specific deps).

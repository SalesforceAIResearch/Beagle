# `templates/external_benchmark_package/` — copy-paste skeleton for an external xrlenv plug-in

This directory is the canonical skeleton for shipping a new benchmark
as an **external pip package**. Copy this whole tree into a new repo,
search-and-replace the placeholders, and `uv pip install -e .` against
your local xrlenv checkout — the plug-in registers via the
`xrlenv.benchmarks` Python entry-points group with no source-tree
edits to xrlenv itself.

For fully-runnable end-to-end examples (not stubs), see
`examples/pip_new_datasets_or_benchmark/`:

- `echo_bench/` — own benchmark with new adapter + scoring; ~30 s smoke.
- `byo_dataset_harbor/` — own dataset reusing tb2's adapter; ~1 min smoke.

This skeleton is intentionally minimal — adapter methods raise
`NotImplementedError`. The worked examples above are the real
copy-paste sources for production plug-ins.

## What's in here

```
templates/external_benchmark_package/
├── pyproject.toml            # declares the entry-point + deps + name
├── xrlenv_plugins/           # PEP-420 namespace package contribution
│   └── benchmarks/
│       └── example_bench/    # your benchmark's package directory
│           ├── __init__.py
│           ├── adapter.py    # EnvAdapter + InstanceResolver
│           ├── manifest.yaml # spec-06 template manifest
│           └── plugin.py     # entry-point callable: returns manifest paths
├── scripts/
│   └── build-task-images.sh  # Pattern-A per-task image build (optional)
├── tests/
│   └── test_smoke.py         # plug-in's own pytest suite
└── README.md                 # this file
```

## Customising

1. Replace `example_bench` with your benchmark's package name everywhere
   (directory name, `pyproject.toml` `name`, `manifest.yaml` `name`,
   entry-point key).
2. Replace `xrlenv-example-bench` (the distribution name in
   `pyproject.toml`) with your published pip name.
3. Implement `adapter.py`'s `setup` / `step` / `teardown` against
   your benchmark's harness. The shipped stub raises
   `NotImplementedError` — replace it.
4. Update `manifest.yaml` with your real image, resources, and
   reward contract per **spec 06** (`specs/06-templates-and-environments.md`).
5. If your benchmark needs per-task overrides (Pattern A), point
   `manifest.yaml`'s `instances:` block at your `InstanceResolver`
   subclass and ship the per-task overrides under a `tasks/`
   subdirectory.

See `docs/integration/tutorials/own_benchmark.md` (Bring your own
benchmark) for the full publishing recipe walked end-to-end.
`docs/integration/reference/distribution_paths.md` covers the path
choice (in-tree vs. external pip vs. image-bundled).

For richer references beyond a smoke skeleton:

- The fully-runnable worked example at
  `examples/pip_new_datasets_or_benchmark/echo_bench/` — adapter +
  resolver + grader script with the in-package script-resolution
  pattern (see `pyproject.toml` force-include comments above for
  when you need this).
- The in-tree `xrlenv_plugins/benchmarks/terminal_bench_2/` plug-in
  for Pattern-A resolver, reward wrapper, image build script at
  realistic scale.

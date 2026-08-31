# `xrlenv_plugins/` — Benchmark plug-ins

This directory is a **PEP-420 namespace package**, not a regular Python
package — there is no `__init__.py` at this level (or under
`xrlenv_plugins/benchmarks/`). That lets external pip packages
contribute new plug-in subdirectories under the same import name without
forking this repo.

## Layout convention

```
xrlenv_plugins/
└── benchmarks/                          # namespace
    └── <benchmark-name>/                # regular package (has __init__.py)
        ├── manifest.yaml                # spec-06 template manifest
        ├── adapter.py                   # EnvAdapter implementation
        ├── tasks/                       # optional Pattern-A overrides
        ├── scripts/                     # build / setup scripts
        ├── tests/                       # plug-in's own pytest suite
        └── examples/                    # run-config samples
```

The platform discovers in-tree plug-ins by walking
`xrlenv_plugins/*/*/manifest.yaml` from the repo root. External plug-in
packages register themselves via the `xrlenv.benchmarks` Python
entry-points group — see `docs/integration/index-as-a-package.md` for
the full publishing recipe.

The in-tree `xrlenv_plugins/benchmarks/terminal_bench_2/` plug-in is the
canonical reference + the platform's own CI signal. Third parties should
ship as separate pip packages that drop their plug-in subdirectory under
`xrlenv_plugins/benchmarks/` at install time (the namespace package
makes this a one-line `pyproject.toml` declaration).

## Co-located smoke template

The platform's spine smoke template (`hello-shell` + `ShellEnvAdapter`)
lives at `xrlenv/templates/hello_shell/`, NOT here — it's part of the
core platform's own test surface, not a plug-in.

# Worked examples — extending xrlenv via a pip package

> **⚠️ DEPRECATED.** This demonstrates the spec-14 **case-1** path (a custom
> `EnvAdapter` + `manifest.yaml`, registered via a pip package) — which **no
> onboarded benchmark uses today**. Real onboardings reuse the shared harbor/pier
> cluster environment (or an outlier pattern) per
> `xrlenv_plugins/benchmarks/GUIDELINE_onboard_benchmarks.md`; see the plug-ins
> under `xrlenv_plugins/benchmarks/` for worked examples. Kept for historical
> reference only.

You're here because you want xrlenv to run **your** task suite.
Two scenarios cover ~all of what end users do; pick the one that
matches your data + start from the corresponding worked example
in this directory.

## Decision tree

| Scenario | "I want to ship..." | Worked example | Smoke time |
|---|---|---|---|
| **1. Own benchmark** | new evaluation logic — your own adapter (env_setup / step / teardown) and your own scoring. | [`echo_bench/`](./echo_bench/) — agent receives a target string in the initial observation, must echo it; reward `1.0` on exact match, `0.0` otherwise. New adapter, new manifest, single pip package. | ~30 s |
| **2. Own dataset, reuse benchmark** | tasks that conform to an existing benchmark's format (e.g. terminal-bench / harbor format), no new adapter logic. | [`byo_dataset_harbor/`](./byo_dataset_harbor/) — subclasses the in-tree `terminal-bench-2` resolver, ships 2 small Harbor-Dataset tasks bundled in the package, optional script downloads more upstream tasks side-by-side. | ~1 min |

If neither fits — your tasks aren't terminal-bench-format and you
don't want a custom adapter — you probably want scenario 1 with
a slightly fuller adapter than `echo_bench`'s. The
[`templates/external_benchmark_package/`](../../templates/external_benchmark_package/)
skeleton has a more abstract starting point with adapter stubs.

## Both examples are cheap by default

Both packages run on a laptop in ~1 min with tiny base images
(`python:3.12-slim` for `echo_bench`; alpine-derived per-task
images for `byo_dataset_harbor`'s bundled tasks). No external
network is required for the default smoke; no GB-sized image
downloads. The Harbor-Dataset *download* script is optional —
you only run it when you want to scale up to the full
~1000-task upstream set.

## Both examples exercise D22

Each package installs via `pip install -e .` and registers via
the `xrlenv.benchmarks` Python entry-point group. That's exactly
the path D22 (the runtime's external-plug-in import-path
extension) was built to support. If you run the smoke and see
`setup_failed: ModuleNotFoundError`, the runtime is on a build
that predates D22 — pull and rebuild.

## See also

- [`docs/integration/index.md`](../../docs/integration/index.md) —
  the abstract bring-your-own-benchmark guide. Read this if you
  want to understand *why* the package layout looks the way it
  does before copying from a worked example.
- [`templates/external_benchmark_package/`](../../templates/external_benchmark_package/)
  — the abstract pip-package skeleton with adapter stubs.
- [`docs/integration/benchmarks/terminal_bench_2/developer.md`](../../docs/integration/benchmarks/terminal_bench_2/developer.md)
  — the in-tree terminal-bench-2 plug-in's design notes; the
  scenario-2 example here subclasses its resolver.

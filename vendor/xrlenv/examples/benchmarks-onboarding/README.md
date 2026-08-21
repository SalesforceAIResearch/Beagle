# Benchmark onboarding examples

> **⚠️ DEPRECATED.** Onboarded benchmarks now live as GUIDELINE plug-ins under
> `xrlenv_plugins/benchmarks/` (`swebench_verified/` — the docker-py drop-in;
> `terminal_bench_2_1/`; etc.). This `examples/benchmarks-onboarding/` tree is kept
> only for historical reference. The *how-to-onboard* tutorials moved to the Sphinx
> docs: the docker-py drop-in →
> `docs/build_with_xrlenv/work_with_xrlenv_managed_containers/docker_py_dropin.md`;
> the plug-in path → `xrlenv_plugins/benchmarks/GUIDELINE_onboard_benchmarks.md`.

Each subdirectory ships a **runnable end-to-end smoke** for one
upstream benchmark, plus a short README that walks through how the
integration was wired so you can adapt it to your own benchmark.

The examples default to a small smoke set (8 instances) so a fresh
checkout on a clean cluster yields a green/red signal in a few
minutes. Each smoke also has an `--all` flag to run the full task
set when you're ready for a real evaluation.

## Pick your path

xrlenv has two complementary onboarding paths. **Which one fits
your benchmark depends on how the benchmark already drives its
sandbox.**

| If your benchmark... | Use this path | Worked example |
|---|---|---|
| ...has an existing harness that uses **docker-py** to spawn / exec / archive containers (SWE-bench-style sync harnesses, OSWorld's docker provider, etc.) | **docker-py drop-in** — swap `docker.from_env()` for `xrlenv.from_env(grpc_host=...)`. The harness keeps its existing flow; xrlenv intercepts every docker-py call and routes it to a cluster-picked node. | [`swebench-verified/`](swebench-verified/) |
| ...is **step-driven** (state-machine that takes an action, returns an observation; trainer drives the loop), or wraps `docker compose` / a custom non-docker-py runtime | **Plug-in mechanism** — write a small `EnvAdapter` + `manifest.yaml`, register via the `xrlenv.benchmarks` entry-point. xrlenv drives the rollout loop; your adapter translates between trainer actions and the sandbox. | [`terminal-bench-2/`](terminal-bench-2/) |

The two paths target different benchmark shapes — they are not
alternatives for the same benchmark. SWE-bench's harness drives
docker-py directly, so the drop-in fits naturally; terminal-bench-2
runs through harbor's `docker compose` subprocess, which the
docker-py drop-in doesn't intercept, so it goes through the plug-in
mechanism instead.

## Operator pre-requisites

Both paths require the per-task images to be present **on each
cluster node** before consumers acquire — xrlenv does not pull
images implicitly. Each subdirectory's `scripts/` carries the
build / pre-pull recipe:

- `swebench-verified/scripts/pre-pull-images.sh` — pulls the
  `swebench/sweb.eval.*` images from Docker Hub onto the local
  Docker daemon. Run once on each cluster node before driving the
  smoke.
- `terminal-bench-2/scripts/build-task-images.sh` — builds the
  per-task images from the upstream Harbor task definitions. Same
  per-node prerequisite.

If the chosen node doesn't have the requested image, the
smoke fails fast at `acquire_container` with a clear
`ImageNotFound` rather than hanging.

## See also

- [`docs/integration/byob.md`](../../docs/integration/byob.md) — the
  authoritative byob (bring-your-own-benchmark) guide.
- [`examples/pip_new_datasets_or_benchmark/`](../pip_new_datasets_or_benchmark/) —
  parallel directory showing how to package the plug-in path as a
  pip-installable wheel (entry-point registration, package layout,
  vendored test data). The benchmarks here ship in-tree; that
  directory shows the external-package path.
